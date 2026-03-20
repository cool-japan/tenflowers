//! Advanced bioinformatics ML algorithms: single-cell RNA-seq VAE (ZINB),
//! genomic sequence CNNs, survival analysis, and multi-omics integration.

use super::{matvec, normal_samples_f64, random_matrix, random_vec, relu, sigmoid, softmax, vecadd};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 6. Single-Cell RNA-seq Analysis (scVAE / ZINB)
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse count matrix for scRNA-seq data (cells × genes).
///
/// Stores data as dense `Vec<Vec<f64>>` but provides normalization utilities.
/// Log1p CPM (counts per million) normalization is applied on request.
#[derive(Debug, Clone)]
pub struct ScRnaMatrix {
    /// Raw count data: cells × genes.
    pub counts: Vec<Vec<f64>>,
    /// Number of cells.
    pub n_cells: usize,
    /// Number of genes.
    pub n_genes: usize,
}

impl ScRnaMatrix {
    /// Create from raw count matrix (cells × genes).
    pub fn new(counts: Vec<Vec<f64>>) -> Result<Self> {
        if counts.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ScRnaMatrix::new",
                "counts must not be empty",
            ));
        }
        let n_genes = counts[0].len();
        if counts.iter().any(|row| row.len() != n_genes) {
            return Err(TensorError::invalid_argument_op(
                "ScRnaMatrix::new",
                "all rows must have equal length",
            ));
        }
        let n_cells = counts.len();
        Ok(Self { counts, n_cells, n_genes })
    }

    /// Compute log1p(CPM) normalized matrix.
    /// CPM = counts / library_size * 1_000_000, then log1p.
    pub fn log1p_cpm(&self) -> Vec<Vec<f64>> {
        self.counts.iter().map(|cell| {
            let lib: f64 = cell.iter().sum();
            let scale = if lib > 0.0 { 1_000_000.0 / lib } else { 1.0 };
            cell.iter().map(|&c| (c * scale + 1.0).ln()).collect()
        }).collect()
    }

    /// Compute per-gene mean and variance from log1p-CPM.
    pub fn gene_stats(&self) -> Vec<(f64, f64)> {
        let norm = self.log1p_cpm();
        (0..self.n_genes).map(|g| {
            let vals: Vec<f64> = norm.iter().map(|cell| cell[g]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            (mean, var)
        }).collect()
    }
}

/// Encoder network for scVAE (Lopez 2018): input → hidden → (μ, log σ²).
#[derive(Debug, Clone)]
pub struct ScvaeEncoder {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w_mu: Vec<Vec<f64>>,
    b_mu: Vec<f64>,
    w_lv: Vec<Vec<f64>>,
    b_lv: Vec<f64>,
    // Layer norm scale/shift
    gamma: Vec<f64>,
    beta: Vec<f64>,
}

impl ScvaeEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            w1: random_matrix(hidden_dim, input_dim, &mut rng),
            b1: random_vec(hidden_dim, &mut rng),
            w_mu: random_matrix(latent_dim, hidden_dim, &mut rng),
            b_mu: random_vec(latent_dim, &mut rng),
            w_lv: random_matrix(latent_dim, hidden_dim, &mut rng),
            b_lv: random_vec(latent_dim, &mut rng),
            gamma: vec![1.0_f64; hidden_dim],
            beta: vec![0.0_f64; hidden_dim],
        }
    }

    /// Encode: x → (μ, log_var).
    pub fn encode(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "ScvaeEncoder::encode",
                "input dimension mismatch",
            ));
        }
        let pre = vecadd(&matvec(&self.w1, x), &self.b1);
        // Layer norm
        let n = pre.len() as f64;
        let mean = pre.iter().sum::<f64>() / n;
        let var = pre.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = (var + 1e-5).sqrt();
        let h: Vec<f64> = pre.iter().enumerate()
            .map(|(i, &v)| relu(self.gamma[i] * (v - mean) / std + self.beta[i]))
            .collect();
        let mu = vecadd(&matvec(&self.w_mu, &h), &self.b_mu);
        // Clamp log_var for numerical stability
        let lv: Vec<f64> = vecadd(&matvec(&self.w_lv, &h), &self.b_lv)
            .into_iter()
            .map(|v| v.clamp(-10.0, 10.0))
            .collect();
        Ok((mu, lv))
    }

    /// Reparameterization trick: z = μ + ε·exp(0.5·log_var).
    pub fn reparameterize(&self, mu: &[f64], log_var: &[f64], seed: u64) -> Vec<f64> {
        let eps = normal_samples_f64(mu.len(), seed);
        mu.iter().zip(log_var.iter()).zip(eps.iter())
            .map(|((&m, &lv), &e)| m + e * (0.5 * lv).exp())
            .collect()
    }
}

/// Decoder for scVAE with Zero-Inflated Negative Binomial (ZINB) output.
///
/// ZINB outputs: mean (μ), dispersion (θ), dropout (π).
/// The reconstruction loss is the negative log-ZINB likelihood.
#[derive(Debug, Clone)]
pub struct ScvaeDecoder {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    // Mean output (softplus activation for positivity)
    w_mu: Vec<Vec<f64>>,
    b_mu: Vec<f64>,
    // Dispersion (log-scale, per gene)
    log_theta: Vec<f64>,
    // Dropout probability (sigmoid output)
    w_pi: Vec<Vec<f64>>,
    b_pi: Vec<f64>,
}

impl ScvaeDecoder {
    pub fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            latent_dim,
            hidden_dim,
            output_dim,
            w1: random_matrix(hidden_dim, latent_dim, &mut rng),
            b1: random_vec(hidden_dim, &mut rng),
            w_mu: random_matrix(output_dim, hidden_dim, &mut rng),
            b_mu: random_vec(output_dim, &mut rng),
            log_theta: random_vec(output_dim, &mut rng),
            w_pi: random_matrix(output_dim, hidden_dim, &mut rng),
            b_pi: random_vec(output_dim, &mut rng),
        }
    }

    /// Decode z → (μ, θ, π) — mean, dispersion, zero-inflation probability.
    pub fn decode(&self, z: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if z.len() != self.latent_dim {
            return Err(TensorError::invalid_argument_op(
                "ScvaeDecoder::decode",
                "latent dimension mismatch",
            ));
        }
        let h: Vec<f64> = vecadd(&matvec(&self.w1, z), &self.b1)
            .into_iter().map(relu).collect();
        // μ: softplus for positivity
        let mu: Vec<f64> = vecadd(&matvec(&self.w_mu, &h), &self.b_mu)
            .into_iter().map(|v| (1.0 + v.exp()).ln().max(1e-8)).collect();
        // θ: exp(log_theta) for dispersion
        let theta: Vec<f64> = self.log_theta.iter()
            .map(|&lt| lt.exp().clamp(1e-4, 1e4)).collect();
        // π: dropout probability via sigmoid
        let pi: Vec<f64> = vecadd(&matvec(&self.w_pi, &h), &self.b_pi)
            .into_iter().map(sigmoid).collect();
        Ok((mu, theta, pi))
    }

    /// Negative log ZINB likelihood for a single cell.
    ///
    /// ZINB(x; μ, θ, π) = π·δ(x=0) + (1-π)·NB(x; μ, θ)
    pub fn zinb_loss(&self, x: &[f64], mu: &[f64], theta: &[f64], pi: &[f64]) -> f64 {
        let n = x.len();
        let mut loss = 0.0_f64;
        for i in 0..n {
            let xi = x[i];
            let mui = mu[i].max(1e-8);
            let ti = theta[i].max(1e-8);
            let pii = pi[i].clamp(1e-8, 1.0 - 1e-8);
            // log NB(x; μ, θ): log Γ(x+θ) - log Γ(θ) - log x! + θ log(θ/(θ+μ)) + x log(μ/(θ+μ))
            let log_nb = {
                // Stirling/lgamma approximation via lgamma
                fn lgamma_approx(n: f64) -> f64 {
                    if n <= 0.0 { return 0.0; }
                    if n < 0.5 { return -(n.ln()); }
                    // Lanczos approximation g=5
                    let g = 5.0_f64;
                    let c = [1.000000000190015_f64, 76.18009172947146, -86.50532032941677,
                              24.01409824083091, -1.231739572450155, 1.208650973866179e-3, -5.395239384953e-6];
                    let x = n - 1.0;
                    let mut s = c[0];
                    for j in 1..7 { s += c[j] / (x + j as f64); }
                    let t = x + g + 0.5;
                    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
                }
                let t_mu = ti + mui;
                lgamma_approx(xi + ti) - lgamma_approx(ti) - lgamma_approx(xi + 1.0)
                    + ti * (ti / t_mu).ln() + xi * (mui / t_mu).ln()
            };
            // log ZINB
            let log_zinb = if xi < 0.5 {
                // x = 0: log(π + (1-π)·NB(0))
                let log_nb0 = {
                    let t_mu = ti + mui;
                    ti * (ti / t_mu).ln()
                };
                let val = (pii + (1.0 - pii) * log_nb0.exp()).max(1e-300);
                val.ln()
            } else {
                // x > 0: log((1-π)·NB(x))
                ((1.0 - pii).max(1e-300)).ln() + log_nb
            };
            loss -= log_zinb;
        }
        loss / n as f64
    }
}

/// scVAE: variational autoencoder for scRNA-seq with ZINB decoder.
///
/// Based on Lopez et al. 2018 (scVI).
#[derive(Debug, Clone)]
pub struct ScvaeModel {
    pub encoder: ScvaeEncoder,
    pub decoder: ScvaeDecoder,
    /// KL annealing weight (0→1 over training).
    pub kl_weight: f64,
}

impl ScvaeModel {
    pub fn new(
        n_genes: usize,
        hidden_dim: usize,
        latent_dim: usize,
        seed: u64,
    ) -> Self {
        Self {
            encoder: ScvaeEncoder::new(n_genes, hidden_dim, latent_dim, seed),
            decoder: ScvaeDecoder::new(latent_dim, hidden_dim, n_genes, seed.wrapping_add(1)),
            kl_weight: 1.0,
        }
    }

    /// Forward pass: encode + reparameterize + decode.
    /// Returns (z, mu, log_var, dec_mu, dec_theta, dec_pi).
    pub fn forward(
        &self,
        x: &[f64],
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (mu, lv) = self.encoder.encode(x)?;
        let z = self.encoder.reparameterize(&mu, &lv, seed);
        let (dec_mu, dec_theta, dec_pi) = self.decoder.decode(&z)?;
        Ok((z, mu, lv, dec_mu, dec_theta, dec_pi))
    }

    /// ELBO: ZINB reconstruction loss + KL divergence (annealed).
    pub fn elbo(&self, x: &[f64], mu: &[f64], lv: &[f64],
                 dec_mu: &[f64], dec_theta: &[f64], dec_pi: &[f64]) -> f64 {
        let recon = self.decoder.zinb_loss(x, dec_mu, dec_theta, dec_pi);
        let kl: f64 = mu.iter().zip(lv.iter())
            .map(|(&m, &v)| -0.5 * (1.0 + v - m.powi(2) - v.exp()))
            .sum::<f64>() / mu.len() as f64;
        recon + self.kl_weight * kl
    }
}

/// Approximate Leiden/Louvain-style clustering on latent space via greedy modularity.
///
/// Builds a k-NN graph on the provided latent embeddings, then iteratively
/// merges cells by maximising modularity gain ΔQ (Newman-Girvan 2004).
#[derive(Debug, Clone)]
pub struct LeidenClustering {
    /// k for k-NN graph construction.
    pub k: usize,
    /// Maximum iterations of modularity optimisation.
    pub max_iter: usize,
}

impl LeidenClustering {
    pub fn new(k: usize, max_iter: usize) -> Self {
        Self { k, max_iter }
    }

    fn euclidean(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    /// Build weighted k-NN adjacency (weight = 1/(1+dist)).
    fn build_knn_graph(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<(usize, f64)>> {
        let n = embeddings.len();
        let k = self.k.min(n.saturating_sub(1));
        (0..n).map(|i| {
            let mut dists: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
                .map(|j| (j, 1.0 / (1.0 + Self::euclidean(&embeddings[i], &embeddings[j]))))
                .collect();
            dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.into_iter().take(k).collect()
        }).collect()
    }

    /// Compute cluster assignments via greedy modularity maximisation.
    /// Returns a cluster label for each cell.
    pub fn fit(&self, embeddings: &[Vec<f64>]) -> Result<Vec<usize>> {
        let n = embeddings.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "LeidenClustering::fit",
                "empty embeddings",
            ));
        }
        let adj = self.build_knn_graph(embeddings);
        // Total edge weight
        let m: f64 = adj.iter().flat_map(|edges| edges.iter().map(|&(_, w)| w)).sum::<f64>() / 2.0;
        let m = m.max(1e-10);

        // Degree (sum of weights)
        let degree: Vec<f64> = adj.iter().map(|edges| edges.iter().map(|&(_, w)| w).sum()).collect();

        // Initialise each cell as its own community
        let mut labels: Vec<usize> = (0..n).collect();

        for _ in 0..self.max_iter {
            let mut changed = false;
            for i in 0..n {
                // Find best community among neighbours
                let mut community_gain: std::collections::HashMap<usize, f64> =
                    std::collections::HashMap::new();
                for &(j, w) in &adj[i] {
                    let cj = labels[j];
                    *community_gain.entry(cj).or_insert(0.0) += w;
                }
                // ΔQ for moving i to community c:
                // ΔQ ∝ gain[c] - (degree[i] * Σ_{k in c} degree[k]) / (2m)
                let ci_old = labels[i];
                let mut best_c = ci_old;
                let mut best_dq = 0.0_f64;

                for (&c, &gain) in &community_gain {
                    if c == ci_old { continue; }
                    // Sum of degrees in community c
                    let sum_degree_c: f64 = labels.iter().zip(degree.iter())
                        .filter(|(&lbl, _)| lbl == c)
                        .map(|(_, &d)| d)
                        .sum();
                    let dq = gain - degree[i] * sum_degree_c / (2.0 * m);
                    if dq > best_dq {
                        best_dq = dq;
                        best_c = c;
                    }
                }
                if best_c != ci_old {
                    labels[i] = best_c;
                    changed = true;
                }
            }
            if !changed { break; }
        }

        // Relabel communities to contiguous 0..K
        let mut label_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut next_id = 0usize;
        for l in labels.iter_mut() {
            let entry = label_map.entry(*l).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *l = *entry;
        }
        Ok(labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Genomic Sequence Models
// ─────────────────────────────────────────────────────────────────────────────

/// k-mer tokenizer for DNA sequences.
///
/// Builds a vocabulary of all 4^k k-mers. Overlapping k-mers are extracted
/// from the sequence with stride 1.
#[derive(Debug, Clone)]
pub struct DnaTokenizer {
    /// k-mer size (typically 3, 4, or 6).
    pub k: usize,
    /// Vocabulary size = 4^k.
    pub vocab_size: usize,
}

impl DnaTokenizer {
    /// Create a new k-mer tokenizer.
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 || k > 8 {
            return Err(TensorError::invalid_argument_op(
                "DnaTokenizer::new",
                "k must be between 1 and 8",
            ));
        }
        Ok(Self { k, vocab_size: 4_usize.pow(k as u32) })
    }

    fn base_to_idx(c: char) -> usize {
        match c.to_ascii_uppercase() {
            'A' => 0,
            'C' => 1,
            'G' => 2,
            'T' | 'U' => 3,
            _ => 0, // unknown treated as A
        }
    }

    /// Tokenize a DNA string into k-mer indices (overlapping, stride=1).
    pub fn tokenize(&self, seq: &str) -> Vec<usize> {
        let chars: Vec<char> = seq.chars().collect();
        let l = chars.len();
        if l < self.k { return Vec::new(); }
        (0..=l - self.k).map(|i| {
            let mut idx = 0usize;
            for j in 0..self.k {
                idx = idx * 4 + Self::base_to_idx(chars[i + j]);
            }
            idx
        }).collect()
    }

    /// One-hot encode a k-mer token index to a vector of size `vocab_size`.
    pub fn one_hot(&self, token: usize) -> Vec<f64> {
        let mut v = vec![0.0_f64; self.vocab_size];
        if token < self.vocab_size { v[token] = 1.0; }
        v
    }
}

/// Dilated residual convolutional network for genomic sequence classification.
///
/// Inspired by DeepSEA (Zhou & Troyanskaya 2015): 1-D dilated convolutions
/// over k-mer embeddings with residual connections and global max-pooling.
#[derive(Debug, Clone)]
pub struct DnaConvNet {
    pub n_classes: usize,
    pub embed_dim: usize,
    /// Embedding table for k-mers.
    embed_table: Vec<Vec<f64>>,
    /// Convolutional filters per layer: [n_filters][kernel_size * in_channels].
    conv_layers: Vec<(Vec<Vec<f64>>, Vec<f64>, usize)>, // (weights, bias, dilation)
    /// Final linear layer: n_filters → n_classes.
    linear_w: Vec<Vec<f64>>,
    linear_b: Vec<f64>,
}

impl DnaConvNet {
    /// Build a dilated CNN with `n_layers` dilated conv layers.
    ///
    /// - `vocab_size`: k-mer vocabulary size (4^k).
    /// - `embed_dim`: k-mer embedding dimensionality.
    /// - `n_filters`: number of output channels per conv layer.
    /// - `kernel_size`: convolutional kernel size.
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        n_filters: usize,
        kernel_size: usize,
        n_layers: usize,
        n_classes: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Embedding table
        let embed_table = (0..vocab_size).map(|_| {
            (0..embed_dim).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 0.02).collect()
        }).collect();

        // Dilated conv layers: dilation = 2^i
        let mut conv_layers = Vec::new();
        let mut in_ch = embed_dim;
        for i in 0..n_layers {
            let dilation = 1 << i; // 1, 2, 4, 8, ...
            let kernel_w = kernel_size * in_ch;
            let scale = (2.0 / kernel_w as f64).sqrt();
            let filters: Vec<Vec<f64>> = (0..n_filters).map(|_| {
                (0..kernel_w).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale).collect()
            }).collect();
            let bias: Vec<f64> = (0..n_filters).map(|_| 0.0).collect();
            conv_layers.push((filters, bias, dilation));
            in_ch = n_filters;
        }

        let linear_w = random_matrix(n_classes, n_filters, &mut rng);
        let linear_b = random_vec(n_classes, &mut rng);

        Self {
            n_classes,
            embed_dim,
            embed_table,
            conv_layers,
            linear_w,
            linear_b,
        }
    }

    fn embed_tokens(&self, tokens: &[usize]) -> Vec<Vec<f64>> {
        tokens.iter().map(|&t| {
            let idx = t.min(self.embed_table.len() - 1);
            self.embed_table[idx].clone()
        }).collect()
    }

    /// 1D dilated convolution with valid padding.
    /// Input: seq_len × in_ch. Returns (seq_len - dilation*(kernel_size-1)) × n_filters.
    fn dilated_conv(
        input: &[Vec<f64>],
        filters: &[Vec<f64>],
        bias: &[f64],
        kernel_size: usize,
        dilation: usize,
    ) -> Vec<Vec<f64>> {
        let l = input.len();
        let in_ch = if l > 0 { input[0].len() } else { 0 };
        let n_filters = filters.len();
        let receptive = dilation * (kernel_size - 1) + 1;
        if l < receptive { return Vec::new(); }
        let out_len = l - receptive + 1;

        (0..out_len).map(|pos| {
            let mut out = bias.to_vec();
            for f in 0..n_filters {
                let mut acc = 0.0_f64;
                for k in 0..kernel_size {
                    let src_pos = pos + k * dilation;
                    let k_offset = k * in_ch;
                    for c in 0..in_ch {
                        acc += input[src_pos][c] * filters[f][k_offset + c];
                    }
                }
                out[f] += acc;
            }
            out.into_iter().map(relu).collect()
        }).collect()
    }

    /// Forward pass: token indices → class logits.
    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        if tokens.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DnaConvNet::forward",
                "empty token sequence",
            ));
        }
        let mut hidden = self.embed_tokens(tokens);
        for (filters, bias, dilation) in &self.conv_layers {
            let kernel_size = filters[0].len() / (if hidden.is_empty() { 1 } else { hidden[0].len() });
            let kernel_size = kernel_size.max(1);
            let next = Self::dilated_conv(&hidden, filters, bias, kernel_size, *dilation);
            if next.is_empty() { break; }
            hidden = next;
        }
        if hidden.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DnaConvNet::forward",
                "sequence too short for network depth",
            ));
        }
        // Global max pooling
        let n_filters = hidden[0].len();
        let pooled: Vec<f64> = (0..n_filters).map(|f| {
            hidden.iter().map(|h| h[f]).fold(f64::NEG_INFINITY, f64::max)
        }).collect();
        let logits = vecadd(&matvec(&self.linear_w, &pooled), &self.linear_b);
        Ok(logits)
    }
}

/// Predict open chromatin regions from DNA sequence (binary classification).
///
/// Thin wrapper around `DnaConvNet` with sigmoid output for binary prediction.
#[derive(Debug, Clone)]
pub struct ChromatinAccessibility {
    pub net: DnaConvNet,
}

impl ChromatinAccessibility {
    pub fn new(vocab_size: usize, embed_dim: usize, n_filters: usize,
               kernel_size: usize, n_layers: usize, seed: u64) -> Self {
        Self {
            net: DnaConvNet::new(vocab_size, embed_dim, n_filters, kernel_size, n_layers, 1, seed),
        }
    }

    /// Returns probability of chromatin accessibility for the given sequence.
    pub fn predict(&self, tokens: &[usize]) -> Result<f64> {
        let logits = self.net.forward(tokens)?;
        Ok(sigmoid(logits[0]))
    }
}

/// In-silico mutagenesis: compute effect of single-nucleotide variants.
///
/// For each position, each alternative allele is scored, and the
/// log-fold-change (LFC) vs. reference is returned.
#[derive(Debug, Clone)]
pub struct VariantEffectPredictor {
    pub net: DnaConvNet,
    pub tokenizer: DnaTokenizer,
}

impl VariantEffectPredictor {
    pub fn new(k: usize, embed_dim: usize, n_filters: usize,
               kernel_size: usize, n_layers: usize, n_classes: usize, seed: u64) -> Result<Self> {
        let tokenizer = DnaTokenizer::new(k)?;
        let net = DnaConvNet::new(
            tokenizer.vocab_size, embed_dim, n_filters, kernel_size, n_layers, n_classes, seed,
        );
        Ok(Self { net, tokenizer })
    }

    /// Compute per-position log-fold-change relative to reference sequence.
    ///
    /// Returns a vec of `(position, alt_base, lfc_per_class)` for all variants.
    pub fn mutagenesis(
        &self,
        sequence: &str,
    ) -> Result<Vec<(usize, char, Vec<f64>)>> {
        let bases = ['A', 'C', 'G', 'T'];
        let chars: Vec<char> = sequence.chars().collect();
        let ref_tokens = self.tokenizer.tokenize(sequence);
        let ref_logits = self.net.forward(&ref_tokens)?;

        let mut results = Vec::new();
        for pos in 0..chars.len() {
            let ref_base = chars[pos];
            for &alt in &bases {
                if alt == ref_base { continue; }
                let mut mut_seq = chars.clone();
                mut_seq[pos] = alt;
                let mut_str: String = mut_seq.iter().collect();
                let mut_tokens = self.tokenizer.tokenize(&mut_str);
                if mut_tokens.is_empty() { continue; }
                let alt_logits = self.net.forward(&mut_tokens)?;
                let lfc: Vec<f64> = alt_logits.iter().zip(ref_logits.iter())
                    .map(|(&a, &r)| a - r)
                    .collect();
                results.push((pos, alt, lfc));
            }
        }
        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Survival Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Kaplan-Meier non-parametric survival curve estimator.
///
/// Computes S(t) = Π_{t_i ≤ t} (1 - d_i/n_i) using Breslow tie handling.
#[derive(Debug, Clone)]
pub struct KaplanMeier {
    /// Sorted event times with (time, n_at_risk, n_events).
    pub timeline: Vec<(f64, usize, usize)>,
    /// Survival probabilities at each event time.
    pub survival: Vec<f64>,
}

impl KaplanMeier {
    /// Fit Kaplan-Meier estimator.
    ///
    /// - `times`: observation times.
    /// - `events`: 1 = event occurred, 0 = censored.
    pub fn fit(times: &[f64], events: &[u8]) -> Result<Self> {
        if times.len() != events.len() || times.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "KaplanMeier::fit",
                "times and events must have equal non-zero length",
            ));
        }
        let n = times.len();
        // Collect event times (not censored)
        let mut event_times: Vec<f64> = times.iter().zip(events.iter())
            .filter(|(_, &e)| e == 1).map(|(&t, _)| t).collect();
        event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        event_times.dedup();

        let mut timeline = Vec::new();
        let mut survival = Vec::new();
        let mut s = 1.0_f64;
        let mut sorted_times: Vec<f64> = times.to_vec();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for &t in &event_times {
            let n_at_risk = sorted_times.iter().filter(|&&ti| ti >= t).count();
            let n_events = times.iter().zip(events.iter())
                .filter(|(&ti, &e)| (ti - t).abs() < 1e-10 && e == 1).count();
            s *= if n_at_risk > 0 { 1.0 - n_events as f64 / n_at_risk as f64 } else { 1.0 };
            let _ = n;
            timeline.push((t, n_at_risk, n_events));
            survival.push(s);
        }

        Ok(Self { timeline, survival })
    }

    /// Estimate survival probability at time `t` (step function).
    pub fn predict(&self, t: f64) -> f64 {
        let mut s = 1.0_f64;
        for (i, &(ti, _, _)) in self.timeline.iter().enumerate() {
            if ti <= t { s = self.survival[i]; } else { break; }
        }
        s
    }
}

/// Cox Proportional Hazards model with partial likelihood (Breslow tie handling).
///
/// Models hazard h(t|x) = h₀(t)·exp(β·x).
#[derive(Debug, Clone)]
pub struct CoxPh {
    pub n_features: usize,
    /// Regression coefficients β.
    pub beta: Vec<f64>,
}

impl CoxPh {
    pub fn new(n_features: usize) -> Self {
        Self { n_features, beta: vec![0.0_f64; n_features] }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Compute negative partial log-likelihood (for gradient-based optimisation).
    ///
    /// Uses Breslow approximation for ties.
    pub fn neg_partial_likelihood(
        &self,
        x: &[Vec<f64>],
        times: &[f64],
        events: &[u8],
    ) -> Result<f64> {
        let n = x.len();
        if n != times.len() || n != events.len() {
            return Err(TensorError::invalid_argument_op(
                "CoxPh::neg_partial_likelihood",
                "x, times, events must have equal length",
            ));
        }
        // Sort by time descending
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| times[b].partial_cmp(&times[a]).unwrap_or(std::cmp::Ordering::Equal));

        let log_hazards: Vec<f64> = (0..n).map(|i| Self::dot(&self.beta, &x[i])).collect();

        let mut loss = 0.0_f64;
        for &i in &idx {
            if events[i] == 0 { continue; }
            // Risk set: all j with times[j] >= times[i]
            let log_sum_risk: f64 = (0..n)
                .filter(|&j| times[j] >= times[i])
                .map(|j| log_hazards[j])
                .fold(f64::NEG_INFINITY, |acc, v| {
                    let mx = acc.max(v);
                    mx + ((acc - mx).exp() + (v - mx).exp()).ln()
                });
            loss += log_sum_risk - log_hazards[i];
        }
        Ok(loss)
    }

    /// Fit via gradient descent on partial likelihood.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        times: &[f64],
        events: &[u8],
        lr: f64,
        n_iter: usize,
    ) -> Result<()> {
        let n = x.len();
        for _ in 0..n_iter {
            let log_hazards: Vec<f64> = (0..n).map(|i| Self::dot(&self.beta, &x[i])).collect();
            let hazards: Vec<f64> = log_hazards.iter().map(|&lh| lh.exp()).collect();
            let mut grad = vec![0.0_f64; self.n_features];

            for i in 0..n {
                if events[i] == 0 { continue; }
                // Σ_{j in risk} hazard_j * x_j / Σ hazard_j
                let risk_sum: f64 = (0..n).filter(|&j| times[j] >= times[i]).map(|j| hazards[j]).sum();
                let risk_sum = risk_sum.max(1e-300);
                for f in 0..self.n_features {
                    let weighted_x: f64 = (0..n)
                        .filter(|&j| times[j] >= times[i])
                        .map(|j| hazards[j] * x[j][f])
                        .sum();
                    grad[f] += weighted_x / risk_sum - x[i][f];
                }
            }
            for f in 0..self.n_features {
                self.beta[f] -= lr * grad[f];
            }
        }
        Ok(())
    }

    /// Predict log-hazard for new samples.
    pub fn predict_log_hazard(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| Self::dot(&self.beta, xi)).collect()
    }
}

/// DeepSurv: neural network extension of Cox model (Katzman et al. 2018).
///
/// A two-hidden-layer MLP → scalar log-risk output, trained with partial likelihood.
#[derive(Debug, Clone)]
pub struct DeepSurv {
    pub n_features: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w3: Vec<Vec<f64>>,
    b3: Vec<f64>,
}

impl DeepSurv {
    pub fn new(n_features: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            n_features,
            w1: random_matrix(hidden_dim, n_features, &mut rng),
            b1: random_vec(hidden_dim, &mut rng),
            w2: random_matrix(hidden_dim, hidden_dim, &mut rng),
            b2: random_vec(hidden_dim, &mut rng),
            w3: random_matrix(1, hidden_dim, &mut rng),
            b3: vec![0.0_f64],
        }
    }

    /// Compute log-risk score for a single sample.
    pub fn log_risk(&self, x: &[f64]) -> f64 {
        let h1: Vec<f64> = vecadd(&matvec(&self.w1, x), &self.b1)
            .into_iter().map(relu).collect();
        let h2: Vec<f64> = vecadd(&matvec(&self.w2, &h1), &self.b2)
            .into_iter().map(relu).collect();
        matvec(&self.w3, &h2)[0] + self.b3[0]
    }

    /// Compute log-risk for all samples.
    pub fn predict_log_risk(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.log_risk(xi)).collect()
    }
}

/// Survival analysis metrics.
#[derive(Debug, Clone)]
pub struct SurvivalMetrics;

impl SurvivalMetrics {
    /// Harrell's C-index (concordance index).
    ///
    /// Fraction of admissible pairs where predicted risk correctly orders outcomes.
    pub fn c_index(log_risks: &[f64], times: &[f64], events: &[u8]) -> Result<f64> {
        let n = log_risks.len();
        if n != times.len() || n != events.len() {
            return Err(TensorError::invalid_argument_op(
                "SurvivalMetrics::c_index",
                "inputs must have equal length",
            ));
        }
        let mut concordant = 0.0_f64;
        let mut admissible = 0.0_f64;
        for i in 0..n {
            if events[i] == 0 { continue; }
            for j in 0..n {
                if times[j] <= times[i] || i == j { continue; }
                admissible += 1.0;
                if log_risks[i] > log_risks[j] { concordant += 1.0; }
                else if (log_risks[i] - log_risks[j]).abs() < 1e-10 { concordant += 0.5; }
            }
        }
        if admissible < 1.0 { return Ok(0.5); }
        Ok(concordant / admissible)
    }

    /// Brier score at time `t` (probability calibration for survival).
    ///
    /// BS(t) = (1/n) Σ [(S(t|xi) - I(Ti > t))²]
    pub fn brier_score(
        survival_probs: &[f64],
        times: &[f64],
        events: &[u8],
        t: f64,
    ) -> Result<f64> {
        let n = survival_probs.len();
        if n != times.len() || n != events.len() {
            return Err(TensorError::invalid_argument_op(
                "SurvivalMetrics::brier_score",
                "inputs must have equal length",
            ));
        }
        let score: f64 = (0..n).map(|i| {
            let indicator = if times[i] > t { 1.0 } else { 0.0 };
            (survival_probs[i] - indicator).powi(2)
        }).sum::<f64>() / n as f64;
        Ok(score)
    }

    /// Integrated Brier Score over a set of time points.
    pub fn integrated_brier_score(
        survival_probs_at_times: &[Vec<f64>], // [n_time_points][n_samples]
        times: &[f64],
        events: &[u8],
        eval_times: &[f64],
    ) -> Result<f64> {
        if survival_probs_at_times.len() != eval_times.len() {
            return Err(TensorError::invalid_argument_op(
                "SurvivalMetrics::integrated_brier_score",
                "survival_probs_at_times must match eval_times",
            ));
        }
        if eval_times.len() < 2 {
            return Ok(0.0);
        }
        let mut ibs = 0.0_f64;
        for t in 0..eval_times.len() {
            let bs = Self::brier_score(&survival_probs_at_times[t], times, events, eval_times[t])?;
            if t > 0 {
                let dt = eval_times[t] - eval_times[t - 1];
                ibs += bs * dt;
            }
        }
        let total_time = eval_times.last().unwrap_or(&1.0) - eval_times.first().unwrap_or(&0.0);
        Ok(ibs / total_time.max(1e-10))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Multi-Omics Integration
// ─────────────────────────────────────────────────────────────────────────────

/// Paired multi-omics dataset: genomics, transcriptomics, proteomics.
#[derive(Debug, Clone)]
pub struct OmicsDataset {
    /// Genomic features per sample (e.g., copy-number, mutations).
    pub genomics: Vec<Vec<f64>>,
    /// Transcriptomic features per sample (e.g., RNA-seq expression).
    pub transcriptomics: Vec<Vec<f64>>,
    /// Proteomic features per sample (e.g., mass-spec abundances).
    pub proteomics: Vec<Vec<f64>>,
    /// Number of samples.
    pub n_samples: usize,
}

impl OmicsDataset {
    /// Create a new paired multi-omics dataset.
    pub fn new(
        genomics: Vec<Vec<f64>>,
        transcriptomics: Vec<Vec<f64>>,
        proteomics: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let n = genomics.len();
        if transcriptomics.len() != n || proteomics.len() != n {
            return Err(TensorError::invalid_argument_op(
                "OmicsDataset::new",
                "all modalities must have equal number of samples",
            ));
        }
        Ok(Self { genomics, transcriptomics, proteomics, n_samples: n })
    }

    /// Concatenate all modalities for a given sample.
    pub fn concat_sample(&self, i: usize) -> Vec<f64> {
        let mut v = self.genomics[i].clone();
        v.extend_from_slice(&self.transcriptomics[i]);
        v.extend_from_slice(&self.proteomics[i]);
        v
    }
}

/// Multi-Omics Factor Analysis (MOFA) — Argelaguet et al. 2018.
///
/// Joint factor model: each modality Y_m ≈ W_m · Z + ε_m
/// Fitted via EM: alternating updates of latent Z and weights W_m.
#[derive(Debug, Clone)]
pub struct MoFa {
    /// Number of latent factors.
    pub n_factors: usize,
    /// Weights per modality: W_m ∈ ℝ^{d_m × n_factors}.
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Latent factor matrix Z ∈ ℝ^{n_samples × n_factors}.
    pub factors: Vec<Vec<f64>>,
    /// Per-modality feature dimensions.
    pub dims: Vec<usize>,
}

impl MoFa {
    /// Initialise MOFA with random weights.
    pub fn new(dims: Vec<usize>, n_factors: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = dims.iter().map(|&d| {
            (0..d).map(|_| {
                (0..n_factors).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 0.1).collect()
            }).collect()
        }).collect();
        Self { n_factors, weights, factors: Vec::new(), dims }
    }

    /// Fit MOFA via alternating EM updates.
    ///
    /// `data`: one `Vec<Vec<f64>>` per modality, each of shape n_samples × d_m.
    pub fn fit(&mut self, data: &[Vec<Vec<f64>>], n_iter: usize, seed: u64) -> Result<()> {
        let n_modalities = data.len();
        if n_modalities == 0 {
            return Err(TensorError::invalid_argument_op("MoFa::fit", "no data provided"));
        }
        let n_samples = data[0].len();
        if n_samples == 0 {
            return Err(TensorError::invalid_argument_op("MoFa::fit", "empty data"));
        }

        // Initialise Z
        let mut rng = StdRng::seed_from_u64(seed);
        self.factors = (0..n_samples).map(|_| {
            (0..self.n_factors).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 0.1).collect()
        }).collect();

        for _ in 0..n_iter {
            // E-step: update Z given W (least squares: Z = (Σ W_m^T W_m)^{-1} Σ W_m^T Y_m)
            // For simplicity: Z_n = Σ_m W_m^T y_{nm} / (n_modalities * n_factors)
            for s in 0..n_samples {
                let mut z_new = vec![0.0_f64; self.n_factors];
                for (m, modality_data) in data.iter().enumerate() {
                    if m >= n_modalities { break; }
                    let y = &modality_data[s];
                    let w = &self.weights[m];
                    // W_m^T y: [n_factors]
                    for f in 0..self.n_factors {
                        let wty: f64 = y.iter().enumerate()
                            .map(|(d, &yd)| if d < w.len() { w[d][f] * yd } else { 0.0 })
                            .sum();
                        z_new[f] += wty;
                    }
                }
                let scale = (n_modalities * self.n_factors).max(1) as f64;
                for f in 0..self.n_factors {
                    self.factors[s][f] = z_new[f] / scale;
                }
            }

            // M-step: update W_m given Z (W_m = Y_m Z^T (Z Z^T)^{-1})
            for (m, modality_data) in data.iter().enumerate() {
                if m >= self.weights.len() { break; }
                let d_m = self.dims[m].min(if modality_data.is_empty() { 0 } else { modality_data[0].len() });
                for d in 0..d_m {
                    for f in 0..self.n_factors {
                        // W[d][f] = Σ_n Y_n[d] * Z_n[f] / Σ_n Z_n[f]²
                        let num: f64 = (0..n_samples)
                            .map(|s| modality_data[s][d] * self.factors[s][f])
                            .sum();
                        let denom: f64 = (0..n_samples)
                            .map(|s| self.factors[s][f].powi(2))
                            .sum::<f64>().max(1e-10);
                        if d < self.weights[m].len() && f < self.weights[m][d].len() {
                            self.weights[m][d][f] = num / denom;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Project a new sample through the factor model (using pre-fitted weights).
    /// Returns latent factors Z for the sample.
    pub fn transform_sample(&self, sample_data: &[Vec<f64>]) -> Result<Vec<f64>> {
        if sample_data.len() != self.weights.len() {
            return Err(TensorError::invalid_argument_op(
                "MoFa::transform_sample",
                "sample_data must match number of modalities",
            ));
        }
        let mut z = vec![0.0_f64; self.n_factors];
        let mut count = 0usize;
        for (m, y) in sample_data.iter().enumerate() {
            let w = &self.weights[m];
            for f in 0..self.n_factors {
                let wty: f64 = y.iter().enumerate()
                    .map(|(d, &yd)| if d < w.len() { w[d][f] * yd } else { 0.0 })
                    .sum();
                z[f] += wty;
                count += 1;
            }
        }
        let scale = count.max(1) as f64;
        for zf in z.iter_mut() { *zf /= scale; }
        Ok(z)
    }
}

/// Cross-modality attention fusion for multi-omics data.
///
/// Uses multi-head cross-attention between modality embeddings to produce
/// a fused representation.
#[derive(Debug, Clone)]
pub struct OmicsAttentionFusion {
    pub n_modalities: usize,
    pub embed_dim: usize,
    pub n_heads: usize,
    /// Per-modality linear projections: d_m → embed_dim.
    proj_in: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Cross-attention query/key/value projections.
    w_q: Vec<Vec<f64>>,
    w_k: Vec<Vec<f64>>,
    w_v: Vec<Vec<f64>>,
    w_o: Vec<Vec<f64>>,
    /// Output MLP.
    w_mlp1: Vec<Vec<f64>>,
    b_mlp1: Vec<f64>,
    w_mlp2: Vec<Vec<f64>>,
    b_mlp2: Vec<f64>,
}

impl OmicsAttentionFusion {
    pub fn new(
        modality_dims: &[usize],
        embed_dim: usize,
        n_heads: usize,
        seed: u64,
    ) -> Result<Self> {
        if embed_dim % n_heads != 0 {
            return Err(TensorError::invalid_argument_op(
                "OmicsAttentionFusion::new",
                "embed_dim must be divisible by n_heads",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let proj_in = modality_dims.iter().map(|&d| {
            (random_matrix(embed_dim, d, &mut rng), random_vec(embed_dim, &mut rng))
        }).collect();
        Ok(Self {
            n_modalities: modality_dims.len(),
            embed_dim,
            n_heads,
            proj_in,
            w_q: random_matrix(embed_dim, embed_dim, &mut rng),
            w_k: random_matrix(embed_dim, embed_dim, &mut rng),
            w_v: random_matrix(embed_dim, embed_dim, &mut rng),
            w_o: random_matrix(embed_dim, embed_dim, &mut rng),
            w_mlp1: random_matrix(embed_dim * 4, embed_dim, &mut rng),
            b_mlp1: random_vec(embed_dim * 4, &mut rng),
            w_mlp2: random_matrix(embed_dim, embed_dim * 4, &mut rng),
            b_mlp2: random_vec(embed_dim, &mut rng),
        })
    }

    /// Fuse a list of per-modality feature vectors into a single embedding.
    pub fn fuse(&self, modality_features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if modality_features.len() != self.n_modalities {
            return Err(TensorError::invalid_argument_op(
                "OmicsAttentionFusion::fuse",
                "modality_features length must match n_modalities",
            ));
        }
        // Project each modality to embed_dim
        let embeddings: Vec<Vec<f64>> = modality_features.iter().zip(self.proj_in.iter())
            .map(|(feat, (w, b))| {
                vecadd(&matvec(w, feat), b).into_iter().map(relu).collect()
            }).collect();

        // Multi-head self-attention over modality tokens
        let m = embeddings.len();
        let head_dim = self.embed_dim / self.n_heads;
        let scale = (head_dim as f64).sqrt();

        let q: Vec<Vec<f64>> = embeddings.iter().map(|e| matvec(&self.w_q, e)).collect();
        let k: Vec<Vec<f64>> = embeddings.iter().map(|e| matvec(&self.w_k, e)).collect();
        let v: Vec<Vec<f64>> = embeddings.iter().map(|e| matvec(&self.w_v, e)).collect();

        let mut attn_out = vec![vec![0.0_f64; self.embed_dim]; m];
        for h in 0..self.n_heads {
            let start = h * head_dim;
            let mut scores = vec![vec![0.0_f64; m]; m];
            for i in 0..m {
                for j in 0..m {
                    let dot: f64 = q[i][start..start + head_dim].iter()
                        .zip(k[j][start..start + head_dim].iter())
                        .map(|(&a, &b)| a * b).sum();
                    scores[i][j] = dot / scale;
                }
            }
            let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();
            for i in 0..m {
                for j in 0..m {
                    let a = attn[i][j];
                    for d in 0..head_dim {
                        attn_out[i][start + d] += a * v[j][start + d];
                    }
                }
            }
        }

        // Output projection, then mean pool across modalities
        let projected: Vec<Vec<f64>> = attn_out.iter().zip(embeddings.iter())
            .map(|(ao, emb)| vecadd(&matvec(&self.w_o, ao), emb))
            .collect();
        let mut pooled = vec![0.0_f64; self.embed_dim];
        for pe in &projected {
            for (i, &v) in pe.iter().enumerate() {
                pooled[i] += v;
            }
        }
        let m_f = m as f64;
        for p in pooled.iter_mut() { *p /= m_f; }

        // FFN
        let h1: Vec<f64> = vecadd(&matvec(&self.w_mlp1, &pooled), &self.b_mlp1)
            .into_iter().map(relu).collect();
        let out = vecadd(&matvec(&self.w_mlp2, &h1), &self.b_mlp2);
        Ok(out)
    }
}

/// GSEA-like pathway enrichment analysis via Kolmogorov-Smirnov running sum.
///
/// Computes enrichment scores for gene sets given a ranked gene list.
#[derive(Debug, Clone)]
pub struct PathwayEnrichment {
    /// Total number of genes in the ranked list.
    pub n_genes: usize,
}

impl PathwayEnrichment {
    pub fn new(n_genes: usize) -> Self {
        Self { n_genes }
    }

    /// Compute GSEA enrichment score (KS running sum) for a gene set.
    ///
    /// - `ranked_genes`: gene indices sorted by correlation/metric (highest first).
    /// - `gene_set`: set of gene indices belonging to the pathway.
    /// - `p`: exponent for weighting (default p=1 in GSEA).
    ///
    /// Returns (enrichment_score, leading_edge_size).
    pub fn enrichment_score(
        &self,
        ranked_genes: &[usize],
        gene_set: &std::collections::HashSet<usize>,
        p: f64,
    ) -> Result<(f64, usize)> {
        if ranked_genes.is_empty() || gene_set.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "PathwayEnrichment::enrichment_score",
                "ranked_genes and gene_set must not be empty",
            ));
        }
        let n_r: f64 = ranked_genes.iter()
            .filter(|g| gene_set.contains(g))
            .enumerate()
            .map(|(i, _)| (i + 1) as f64)
            .fold(0.0, |acc, rank| acc + rank.powf(p));
        let n_r = n_r.max(1e-10);
        let n_miss = (ranked_genes.len() - gene_set.len().min(ranked_genes.len())) as f64;
        let n_miss = n_miss.max(1.0);

        let mut running_sum = 0.0_f64;
        let mut es = 0.0_f64;
        let mut leading_edge = 0usize;
        let mut hit_count = 0usize;

        for (rank, &gene) in ranked_genes.iter().enumerate() {
            if gene_set.contains(&gene) {
                running_sum += (rank + 1) as f64 / n_r;
                hit_count += 1;
            } else {
                running_sum -= 1.0 / n_miss;
            }
            if running_sum.abs() > es.abs() {
                es = running_sum;
                leading_edge = hit_count;
            }
        }
        Ok((es, leading_edge))
    }

    /// Compute enrichment scores for multiple pathways.
    pub fn compute_all(
        &self,
        ranked_genes: &[usize],
        pathways: &[std::collections::HashSet<usize>],
        p: f64,
    ) -> Result<Vec<(f64, usize)>> {
        pathways.iter()
            .map(|gene_set| self.enrichment_score(ranked_genes, gene_set, p))
            .collect()
    }
}
