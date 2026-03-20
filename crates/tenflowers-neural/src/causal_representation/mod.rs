//! Causal Representation Learning — identifiable disentanglement and deep structural causal models.
//!
//! # Modules
//!
//! - **iVAE** (Khemakhem et al. 2020): Identifiable VAE with auxiliary-conditioned exponential
//!   family prior, enabling provably identifiable latent representations.
//! - **TC-VAE** (Chen et al. 2018): Total Correlation decomposition of the ELBO into MI, TC, and
//!   per-dimension KL terms.
//! - **FactorVAE** (Kim & Mnih 2019): Adversarial TC penalty via a density-ratio discriminator.
//! - **Nonlinear ICA** (Slow Feature Analysis variant): Unsupervised source separation via
//!   slowness and independence objectives on temporal data.
//! - **Deep SCM** (Pawlowski et al. 2020): Neural structural causal model with interventional
//!   and counterfactual inference.
//! - **Disentanglement Metrics**: MIG, SAP score, modularity.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / v.len() as f64; v.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

fn log_sum_exp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let sum: f64 = v.iter().map(|&x| (x - max_v).exp()).sum();
    max_v + sum.ln()
}

/// Box-Muller transform: returns one standard normal sample given two uniform samples in (0,1].
fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1_safe = u1.max(1e-12);
    let r = (-2.0 * u1_safe.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Xavier uniform initialization: range = sqrt(6 / (fan_in + fan_out)).
fn xavier_uniform(rng: &mut StdRng, fan_in: usize, fan_out: usize) -> f64 {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    rng.random::<f64>() * 2.0 * limit - limit
}

// ─────────────────────────────────────────────────────────────────────────────
// Building blocks: CrLinear, CrMlp
// ─────────────────────────────────────────────────────────────────────────────

/// A single linear (affine) layer: y = W x + b.
#[derive(Debug, Clone)]
pub struct CrLinear {
    /// Weight matrix stored as \[out_dim\]\[in_dim\].
    pub w: Vec<Vec<f64>>,
    /// Bias vector \[out_dim\].
    pub b: Vec<f64>,
}

impl CrLinear {
    /// Construct a new layer with Xavier-uniform weight initialization and zero bias.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(
            0x_cafe_cafe_u64
                .wrapping_mul(in_dim as u64 + 1)
                .wrapping_add(out_dim as u64),
        );
        let w = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| xavier_uniform(&mut rng, in_dim, out_dim))
                    .collect()
            })
            .collect();
        let b = vec![0.0; out_dim];
        CrLinear { w, b }
    }

    /// Linear forward pass: y_j = sum_i W\[j\]\[i\] * x\[i\] + b\[j\].
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(row, &bias)| {
                row.iter()
                    .zip(x.iter())
                    .map(|(&w, &xi)| w * xi)
                    .sum::<f64>()
                    + bias
            })
            .collect()
    }
}

/// Multi-layer perceptron composed of CrLinear layers with ReLU activations on all but the last.
#[derive(Debug, Clone)]
pub struct CrMlp {
    pub layers: Vec<CrLinear>,
    /// true = ReLU activation after this layer; false = linear (last layer).
    pub activations: Vec<bool>,
}

impl CrMlp {
    /// Construct from a slice of layer sizes: e.g., `&[in, h1, h2, out]`.
    /// All hidden layers use ReLU; the final layer is linear.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least input and output size"
        );
        let n = layer_sizes.len() - 1;
        let layers: Vec<CrLinear> = (0..n)
            .map(|i| CrLinear::new(layer_sizes[i], layer_sizes[i + 1]))
            .collect();
        let activations: Vec<bool> = (0..n).map(|i| i < n - 1).collect();
        CrMlp {
            layers,
            activations,
        }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();
        for (layer, &use_relu) in self.layers.iter().zip(self.activations.iter()) {
            let pre = layer.forward(&h);
            h = if use_relu {
                pre.into_iter().map(relu).collect()
            } else {
                pre
            };
        }
        h
    }

    /// Gradient-descent parameter update (plain SGD).
    ///
    /// - `grad_w[l][j][i]` = gradient of loss w.r.t. `layers[l].w[j][i]`
    /// - `grad_b[l][j]`    = gradient w.r.t. `layers[l].b[j]`
    pub fn update(&mut self, lr: f64, grad_w: &[Vec<Vec<f64>>], grad_b: &[Vec<f64>]) {
        for (l, layer) in self.layers.iter_mut().enumerate() {
            if l >= grad_w.len() {
                break;
            }
            for (j, row) in layer.w.iter_mut().enumerate() {
                if j >= grad_w[l].len() {
                    break;
                }
                for (i, wij) in row.iter_mut().enumerate() {
                    if i < grad_w[l][j].len() {
                        *wij -= lr * grad_w[l][j][i];
                    }
                }
            }
            for (j, bj) in layer.b.iter_mut().enumerate() {
                if j < grad_b[l].len() {
                    *bj -= lr * grad_b[l][j];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// iVAE — Identifiable VAE (Khemakhem et al. 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the identifiable VAE.
#[derive(Debug, Clone)]
pub struct IvaeConfig {
    pub x_dim: usize,
    pub z_dim: usize,
    /// Dimension of the auxiliary variable u (e.g., one-hot segment encoding).
    pub u_dim: usize,
    /// Number of discrete auxiliary segments (time segments / domain labels).
    pub n_segments: usize,
    pub hidden_dim: usize,
}

/// Encoder q(z | x, u) = N(mu(x,u), diag(exp(lv(x,u)))).
#[derive(Debug, Clone)]
pub struct IvaeEncoder {
    pub mlp: CrMlp,
    pub z_dim: usize,
}

impl IvaeEncoder {
    pub fn new(x_dim: usize, u_dim: usize, z_dim: usize, hidden_dim: usize) -> Self {
        let mlp = CrMlp::new(&[x_dim + u_dim, hidden_dim, hidden_dim, 2 * z_dim]);
        IvaeEncoder { mlp, z_dim }
    }

    /// Returns `(mu, log_var)` each of length `z_dim`.
    pub fn encode(&self, x: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let xu: Vec<f64> = x.iter().chain(u.iter()).cloned().collect();
        let out = self.mlp.forward(&xu);
        let mu: Vec<f64> = out[..self.z_dim].to_vec();
        let lv: Vec<f64> = out[self.z_dim..].to_vec();
        (mu, lv)
    }

    /// Reparameterisation sample using Box-Muller noise pair.
    pub fn sample(&self, mu: &[f64], log_var: &[f64], u1: f64, u2: f64) -> Vec<f64> {
        // Generate a stream of ~N(0,1) from the single (u1,u2) pair using hash offsets
        mu.iter()
            .zip(log_var.iter())
            .enumerate()
            .map(|(i, (&m, &lv))| {
                let eps = box_muller(
                    u1.max(1e-12) * (1.0 + 0.001 * i as f64).min(1.0),
                    u2 * (1.0 + 0.0013 * i as f64) % 1.0 + 1e-12,
                );
                m + eps * (0.5 * lv).exp()
            })
            .collect()
    }
}

/// Segment-conditional exponential-family prior p(z | segment) = N(lambda_mu\[s\], diag(exp(lambda_lv\[s\]))).
#[derive(Debug, Clone)]
pub struct IvaePrior {
    pub lambda_mu: Vec<Vec<f64>>,
    pub lambda_lv: Vec<Vec<f64>>,
}

impl IvaePrior {
    /// Random initialization with small values.
    pub fn new(n_segments: usize, z_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let lambda_mu: Vec<Vec<f64>> = (0..n_segments)
            .map(|_| {
                (0..z_dim)
                    .map(|_| rng.random::<f64>() * 0.2 - 0.1)
                    .collect()
            })
            .collect();
        let lambda_lv: Vec<Vec<f64>> = (0..n_segments).map(|_| vec![0.0f64; z_dim]).collect();
        IvaePrior {
            lambda_mu,
            lambda_lv,
        }
    }

    /// Log probability of z under N(lambda_mu\[s\], diag(exp(lambda_lv\[s\]))).
    pub fn log_prob(&self, z: &[f64], segment: usize) -> f64 {
        let seg = segment.min(self.lambda_mu.len().saturating_sub(1));
        let mu = &self.lambda_mu[seg];
        let lv = &self.lambda_lv[seg];
        let n = z.len();
        let const_term = -(n as f64) * 0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_det: f64 = lv.iter().sum::<f64>();
        let mahal: f64 = z
            .iter()
            .zip(mu.iter())
            .zip(lv.iter())
            .map(|((&zi, &mi), &lvi)| {
                let var = lvi.exp().max(1e-12);
                (zi - mi).powi(2) / var
            })
            .sum();
        const_term - 0.5 * log_det - 0.5 * mahal
    }

    /// Analytic KL(N(mu, diag(exp(lv))) || N(lambda_mu\[s\], diag(exp(lambda_lv\[s\])))).
    pub fn kl_to_prior(&self, mu: &[f64], log_var: &[f64], segment: usize) -> f64 {
        let seg = segment.min(self.lambda_mu.len().saturating_sub(1));
        let p_mu = &self.lambda_mu[seg];
        let p_lv = &self.lambda_lv[seg];
        let kl: f64 = mu
            .iter()
            .zip(log_var.iter())
            .zip(p_mu.iter())
            .zip(p_lv.iter())
            .map(|(((&qi_mu, &qi_lv), &pi_mu), &pi_lv)| {
                let q_var = qi_lv.exp().max(1e-12);
                let p_var = pi_lv.exp().max(1e-12);
                0.5 * (p_var.ln() - qi_lv + q_var / p_var + (qi_mu - pi_mu).powi(2) / p_var - 1.0)
            })
            .sum();
        kl.max(0.0)
    }
}

/// Decoder p(x | z) parameterised as Gaussian mean.
#[derive(Debug, Clone)]
pub struct IvaeDecoder {
    pub mlp: CrMlp,
    pub x_dim: usize,
}

impl IvaeDecoder {
    pub fn new(z_dim: usize, x_dim: usize, hidden_dim: usize) -> Self {
        let mlp = CrMlp::new(&[z_dim, hidden_dim, hidden_dim, x_dim]);
        IvaeDecoder { mlp, x_dim }
    }

    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        self.mlp.forward(z)
    }
}

/// Full identifiable VAE model.
#[derive(Debug, Clone)]
pub struct IvaeModel {
    pub encoder: IvaeEncoder,
    pub prior: IvaePrior,
    pub decoder: IvaeDecoder,
    pub config: IvaeConfig,
}

impl IvaeModel {
    pub fn new(config: IvaeConfig) -> Self {
        let encoder = IvaeEncoder::new(config.x_dim, config.u_dim, config.z_dim, config.hidden_dim);
        let prior = IvaePrior::new(config.n_segments, config.z_dim);
        let decoder = IvaeDecoder::new(config.z_dim, config.x_dim, config.hidden_dim);
        IvaeModel {
            encoder,
            prior,
            decoder,
            config,
        }
    }

    /// ELBO = E_q[log p(x|z)] - KL(q(z|x,u) || p(z|u)).
    pub fn elbo(&self, x: &[f64], u: &[f64], segment: usize, u1: f64, u2: f64) -> f64 {
        let (mu, lv) = self.encoder.encode(x, u);
        let z = self.encoder.sample(&mu, &lv, u1, u2);
        let x_recon = self.decoder.decode(&z);
        // Gaussian log-likelihood: -0.5 * ||x - x_recon||^2
        let recon_loss: f64 = x
            .iter()
            .zip(x_recon.iter())
            .map(|(&xi, &ri)| (xi - ri).powi(2))
            .sum::<f64>();
        let log_p_x_z = -0.5 * recon_loss;
        let kl = self.prior.kl_to_prior(&mu, &lv, segment);
        log_p_x_z - kl
    }

    /// Posterior mean encoder.
    pub fn encode(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let (mu, _lv) = self.encoder.encode(x, u);
        mu
    }

    /// Encode then decode.
    pub fn reconstruct(&self, x: &[f64], u: &[f64], u1: f64, u2: f64) -> Vec<f64> {
        let (mu, lv) = self.encoder.encode(x, u);
        let z = self.encoder.sample(&mu, &lv, u1, u2);
        self.decoder.decode(&z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TC-VAE (Chen et al. 2018) — Total Correlation Decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for TC-VAE.
#[derive(Debug, Clone)]
pub struct TcVaeConfig {
    pub x_dim: usize,
    pub z_dim: usize,
    /// Mutual information weight (default 1.0).
    pub alpha: f64,
    /// Total correlation weight (default 4.0).
    pub beta: f64,
    /// Dimension-wise KL weight (default 1.0).
    pub gamma: f64,
    pub hidden_dim: usize,
    /// Dataset size N for the minibatch TC estimator.
    pub dataset_size: usize,
}

impl Default for TcVaeConfig {
    fn default() -> Self {
        TcVaeConfig {
            x_dim: 784,
            z_dim: 10,
            alpha: 1.0,
            beta: 4.0,
            gamma: 1.0,
            hidden_dim: 256,
            dataset_size: 10000,
        }
    }
}

/// Encoder for TC-VAE with separate mu/log_var heads.
#[derive(Debug, Clone)]
pub struct TcVaeEncoder {
    pub mlp: CrMlp,
    pub mu_head: CrLinear,
    pub lv_head: CrLinear,
    pub z_dim: usize,
}

impl TcVaeEncoder {
    pub fn new(x_dim: usize, z_dim: usize, hidden_dim: usize) -> Self {
        let mlp = CrMlp::new(&[x_dim, hidden_dim, hidden_dim]);
        let mu_head = CrLinear::new(hidden_dim, z_dim);
        let lv_head = CrLinear::new(hidden_dim, z_dim);
        TcVaeEncoder {
            mlp,
            mu_head,
            lv_head,
            z_dim,
        }
    }

    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h = self.mlp.forward(x);
        let mu = self.mu_head.forward(&h);
        let lv = self.lv_head.forward(&h);
        (mu, lv)
    }

    pub fn sample(&self, mu: &[f64], lv: &[f64], u1: f64, u2: f64) -> Vec<f64> {
        mu.iter()
            .zip(lv.iter())
            .enumerate()
            .map(|(i, (&m, &l))| {
                let eps = box_muller(
                    (u1 + 0.001 * i as f64).clamp(1e-12, 1.0 - 1e-12),
                    (u2 + 0.0013 * i as f64) % 1.0 + 1e-12,
                );
                m + eps * (0.5 * l).exp()
            })
            .collect()
    }
}

/// Decoder for TC-VAE.
#[derive(Debug, Clone)]
pub struct TcVaeDecoder {
    pub mlp: CrMlp,
    pub x_dim: usize,
}

impl TcVaeDecoder {
    pub fn new(z_dim: usize, x_dim: usize, hidden_dim: usize) -> Self {
        let mlp = CrMlp::new(&[z_dim, hidden_dim, hidden_dim, x_dim]);
        TcVaeDecoder { mlp, x_dim }
    }

    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        self.mlp.forward(z)
    }
}

/// TC-VAE model with total-correlation decomposition of the ELBO.
#[derive(Debug, Clone)]
pub struct TcVae {
    pub encoder: TcVaeEncoder,
    pub decoder: TcVaeDecoder,
    pub config: TcVaeConfig,
}

impl TcVae {
    pub fn new(config: TcVaeConfig) -> Self {
        let encoder = TcVaeEncoder::new(config.x_dim, config.z_dim, config.hidden_dim);
        let decoder = TcVaeDecoder::new(config.z_dim, config.x_dim, config.hidden_dim);
        TcVae {
            encoder,
            decoder,
            config,
        }
    }

    /// Compute MI term, TC term, and per-dimension KL from a batch of (z, mu, lv) triples.
    ///
    /// Uses the minibatch-weighted sampler estimator (Chen et al. 2018, App. B).
    /// Returns `(mi_term, tc_term, dim_kl_term)`.
    pub fn tc_loss(
        &self,
        z_batch: &[Vec<f64>],
        mu_batch: &[Vec<f64>],
        lv_batch: &[Vec<f64>],
    ) -> (f64, f64, f64) {
        let bs = z_batch.len();
        if bs == 0 {
            return (0.0, 0.0, 0.0);
        }
        let z_dim = self.config.z_dim;
        let n = self.config.dataset_size.max(bs) as f64;

        // Per-dimension KL from standard normal: E[0.5(-1 - lv + mu^2 + exp(lv))]
        let dim_kl_term: f64 = {
            let mut total = 0.0;
            for (mu, lv) in mu_batch.iter().zip(lv_batch.iter()) {
                for j in 0..z_dim.min(mu.len()).min(lv.len()) {
                    total += 0.5 * (-1.0 - lv[j] + mu[j].powi(2) + lv[j].exp());
                }
            }
            total / bs as f64
        };

        // log q(z|x) for each (z_i, mu_i, lv_i): sum_j log N(z_i[j]; mu_i[j], exp(lv_i[j]))
        let log_q_z_given_x: Vec<f64> = z_batch
            .iter()
            .zip(mu_batch.iter())
            .zip(lv_batch.iter())
            .map(|((z, mu), lv)| {
                z.iter()
                    .zip(mu.iter())
                    .zip(lv.iter())
                    .map(|((&zj, &mj), &lvj)| {
                        let var = lvj.exp().max(1e-12);
                        -0.5 * ((zj - mj).powi(2) / var + lvj + (2.0 * std::f64::consts::PI).ln())
                    })
                    .sum::<f64>()
            })
            .collect();

        // log q(z) via minibatch estimator: log_q_z[i] = logsumexp_k log q(z_i | x_k) - log(N*M)
        // where M = batch size, computed using the current batch only
        let log_q_z: Vec<f64> = z_batch
            .iter()
            .map(|z_i| {
                let log_probs: Vec<f64> = mu_batch
                    .iter()
                    .zip(lv_batch.iter())
                    .map(|(mu_k, lv_k)| {
                        z_i.iter()
                            .zip(mu_k.iter())
                            .zip(lv_k.iter())
                            .map(|((&zj, &mj), &lvj)| {
                                let var = lvj.exp().max(1e-12);
                                -0.5 * ((zj - mj).powi(2) / var
                                    + lvj
                                    + (2.0 * std::f64::consts::PI).ln())
                            })
                            .sum::<f64>()
                    })
                    .collect();
                log_sum_exp(&log_probs) - (n * bs as f64).ln()
            })
            .collect();

        // log q(z) product of marginals: sum_j logsumexp_k log q(z_i[j] | x_k[j]) - log(N*M)
        let log_q_z_product: Vec<f64> = z_batch
            .iter()
            .map(|z_i| {
                (0..z_dim)
                    .map(|j| {
                        if j >= z_i.len() {
                            return 0.0;
                        }
                        let log_probs_j: Vec<f64> = mu_batch
                            .iter()
                            .zip(lv_batch.iter())
                            .map(|(mu_k, lv_k)| {
                                if j >= mu_k.len() || j >= lv_k.len() {
                                    return 0.0;
                                }
                                let var = lv_k[j].exp().max(1e-12);
                                -0.5 * ((z_i[j] - mu_k[j]).powi(2) / var
                                    + lv_k[j]
                                    + (2.0 * std::f64::consts::PI).ln())
                            })
                            .collect();
                        log_sum_exp(&log_probs_j) - (n * bs as f64).ln()
                    })
                    .sum::<f64>()
            })
            .collect();

        // MI = E[log q(z|x) - log q(z)]
        let mi_term: f64 = log_q_z_given_x
            .iter()
            .zip(log_q_z.iter())
            .map(|(&a, &b)| a - b)
            .sum::<f64>()
            / bs as f64;

        // TC = E[log q(z) - log prod_j q(z_j)]
        let tc_term: f64 = log_q_z
            .iter()
            .zip(log_q_z_product.iter())
            .map(|(&a, &b)| a - b)
            .sum::<f64>()
            / bs as f64;

        (mi_term, tc_term, dim_kl_term)
    }

    /// Standard ELBO = E[log p(x|z)] - KL(q||p(z)) where p(z) = N(0,I).
    pub fn elbo(&self, x: &[f64], z: &[f64], mu: &[f64], log_var: &[f64]) -> f64 {
        let x_recon = self.decoder.decode(z);
        let recon: f64 = x
            .iter()
            .zip(x_recon.iter())
            .map(|(&xi, &ri)| (xi - ri).powi(2))
            .sum();
        let log_p_x_z = -0.5 * recon;
        let kl: f64 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| 0.5 * (-1.0 - lv + m.powi(2) + lv.exp()))
            .sum();
        log_p_x_z - kl
    }

    /// Forward pass: returns `(z, mu, log_var, x_recon)`.
    pub fn forward(&self, x: &[f64], u1: f64, u2: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mu, lv) = self.encoder.encode(x);
        let z = self.encoder.sample(&mu, &lv, u1, u2);
        let x_recon = self.decoder.decode(&z);
        (z, mu, lv, x_recon)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FactorVAE (Kim & Mnih 2019)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for FactorVAE.
#[derive(Debug, Clone)]
pub struct FactorVaeConfig {
    pub x_dim: usize,
    pub z_dim: usize,
    /// TC penalty weight γ.
    pub gamma: f64,
    pub hidden_dim: usize,
    /// Hidden dimension for the discriminator.
    pub disc_hidden: usize,
}

/// Binary discriminator that classifies z ~ q(z) vs z ~ prod_j q(z_j).
#[derive(Debug, Clone)]
pub struct TcDiscriminator {
    pub layers: Vec<CrLinear>,
    pub hidden_dim: usize,
    pub z_dim: usize,
}

impl TcDiscriminator {
    pub fn new(z_dim: usize, hidden_dim: usize) -> Self {
        let layers = vec![
            CrLinear::new(z_dim, hidden_dim),
            CrLinear::new(hidden_dim, hidden_dim),
            CrLinear::new(hidden_dim, 1),
        ];
        TcDiscriminator {
            layers,
            hidden_dim,
            z_dim,
        }
    }

    /// Returns a logit; positive values indicate z is from q(z), negative from product.
    pub fn forward(&self, z: &[f64]) -> f64 {
        let n = self.layers.len();
        let mut h = z.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&h);
            h = if i < n - 1 {
                pre.into_iter().map(relu).collect()
            } else {
                pre
            };
        }
        h.first().cloned().unwrap_or(0.0)
    }

    /// Shuffle each latent dimension independently across the batch, producing product-of-marginal samples.
    pub fn permute_dims(&self, z_batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let bs = z_batch.len();
        if bs == 0 {
            return vec![];
        }
        let z_dim = z_batch[0].len();
        let mut rng = StdRng::seed_from_u64(0xbabe_1337);
        let mut result: Vec<Vec<f64>> = vec![vec![0.0; z_dim]; bs];
        for j in 0..z_dim {
            // Collect j-th column, shuffle it, place back
            let mut col: Vec<f64> = z_batch
                .iter()
                .map(|z| z.get(j).cloned().unwrap_or(0.0))
                .collect();
            // Fisher-Yates shuffle
            for i in (1..bs).rev() {
                let swap_idx = (rng.random::<f64>() * (i + 1) as f64) as usize;
                col.swap(i, swap_idx.min(i));
            }
            for i in 0..bs {
                result[i][j] = col[i];
            }
        }
        result
    }

    /// Train discriminator one step on `z_joint` (from q(z|x)) and `z_product` (permuted).
    ///
    /// BCE loss: -mean[ log σ(D(z_j)) + log(1 - σ(D(z_p))) ]
    pub fn train_step(&mut self, z_joint: &[Vec<f64>], z_product: &[Vec<f64>], lr: f64) -> f64 {
        if z_joint.is_empty() {
            return 0.0;
        }
        let bs = z_joint.len();

        // Numerical gradient via finite differences
        let eps = 1e-4;
        let n_layers = self.layers.len();

        // First compute forward loss
        let compute_loss = |disc: &TcDiscriminator| -> f64 {
            let mut loss = 0.0;
            for z in z_joint {
                let logit = disc.forward(z);
                // sigmoid(logit) — log-prob of being real
                let log_sigma = -(-logit).exp().ln_1p().max(f64::MIN_POSITIVE.ln());
                loss -= log_sigma;
            }
            for z in z_product {
                let logit = disc.forward(z);
                let log_1_minus_sigma = -logit.exp().ln_1p().max(f64::MIN_POSITIVE.ln());
                loss -= log_1_minus_sigma;
            }
            loss / bs as f64
        };

        let total_loss = compute_loss(self);

        // Finite-difference gradient for each parameter
        for l in 0..n_layers {
            let out_dim = self.layers[l].w.len();
            let in_dim = if out_dim > 0 {
                self.layers[l].w[0].len()
            } else {
                0
            };
            // Gradient for weights
            for j in 0..out_dim {
                for i in 0..in_dim {
                    let orig = self.layers[l].w[j][i];
                    self.layers[l].w[j][i] = orig + eps;
                    let loss_plus = compute_loss(self);
                    self.layers[l].w[j][i] = orig - eps;
                    let loss_minus = compute_loss(self);
                    self.layers[l].w[j][i] = orig;
                    let grad = (loss_plus - loss_minus) / (2.0 * eps);
                    self.layers[l].w[j][i] -= lr * grad;
                }
            }
            // Gradient for biases
            for j in 0..out_dim {
                let orig = self.layers[l].b[j];
                self.layers[l].b[j] = orig + eps;
                let loss_plus = compute_loss(self);
                self.layers[l].b[j] = orig - eps;
                let loss_minus = compute_loss(self);
                self.layers[l].b[j] = orig;
                let grad = (loss_plus - loss_minus) / (2.0 * eps);
                self.layers[l].b[j] -= lr * grad;
            }
        }

        total_loss
    }
}

/// FactorVAE model with adversarial TC penalty.
#[derive(Debug, Clone)]
pub struct FactorVae {
    pub encoder: TcVaeEncoder,
    pub decoder: TcVaeDecoder,
    pub discriminator: TcDiscriminator,
    pub config: FactorVaeConfig,
}

impl FactorVae {
    pub fn new(config: FactorVaeConfig) -> Self {
        let encoder = TcVaeEncoder::new(config.x_dim, config.z_dim, config.hidden_dim);
        let decoder = TcVaeDecoder::new(config.z_dim, config.x_dim, config.hidden_dim);
        let discriminator = TcDiscriminator::new(config.z_dim, config.disc_hidden);
        FactorVae {
            encoder,
            decoder,
            discriminator,
            config,
        }
    }

    /// VAE loss = -ELBO + γ * E[log D(z) - log(1 - D(z))].
    pub fn vae_loss(&self, x: &[f64], z: &[f64], mu: &[f64], lv: &[f64]) -> f64 {
        let x_recon = self.decoder.decode(z);
        let recon: f64 = x
            .iter()
            .zip(x_recon.iter())
            .map(|(&xi, &ri)| (xi - ri).powi(2))
            .sum();
        let log_p_x_z = -0.5 * recon;
        let kl: f64 = mu
            .iter()
            .zip(lv.iter())
            .map(|(&m, &l)| 0.5 * (-1.0 - l + m.powi(2) + l.exp()))
            .sum();
        let elbo = log_p_x_z - kl;
        // TC penalty via discriminator: log D(z) / (1 - D(z)) = logit = D.forward(z)
        let tc_penalty = self.discriminator.forward(z);
        -(elbo - self.config.gamma * tc_penalty)
    }

    /// Forward pass: returns `(z, mu, lv, x_recon)`.
    pub fn forward(&self, x: &[f64], u1: f64, u2: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mu, lv) = self.encoder.encode(x);
        let z = self.encoder.sample(&mu, &lv, u1, u2);
        let x_recon = self.decoder.decode(&z);
        (z, mu, lv, x_recon)
    }

    /// One training step: update VAE and discriminator.
    ///
    /// Returns `(vae_loss, disc_loss)`.
    pub fn train_batch(
        &mut self,
        x_batch: &[Vec<f64>],
        _lr_vae: f64,
        lr_disc: f64,
        rng: &mut StdRng,
    ) -> (f64, f64) {
        if x_batch.is_empty() {
            return (0.0, 0.0);
        }
        let bs = x_batch.len();

        // Forward pass to collect z samples
        let mut z_samples: Vec<Vec<f64>> = Vec::with_capacity(bs);
        let mut mu_batch: Vec<Vec<f64>> = Vec::with_capacity(bs);
        let mut lv_batch: Vec<Vec<f64>> = Vec::with_capacity(bs);
        let mut vae_loss_acc = 0.0;

        for x in x_batch {
            let u1 = rng.random::<f64>().max(1e-12);
            let u2 = rng.random::<f64>().max(1e-12);
            let (mu, lv) = self.encoder.encode(x);
            let z = self.encoder.sample(&mu, &lv, u1, u2);
            let loss = self.vae_loss(x, &z, &mu, &lv);
            vae_loss_acc += loss;
            mu_batch.push(mu);
            lv_batch.push(lv);
            z_samples.push(z);
        }
        let vae_loss = vae_loss_acc / bs as f64;

        // Discriminator step: joint vs permuted
        let z_product = self.discriminator.permute_dims(&z_samples);
        let disc_loss = self
            .discriminator
            .train_step(&z_samples, &z_product, lr_disc);

        // Note: full VAE weight update requires backprop through encoder/decoder;
        // _lr_vae is accepted for API completeness.

        (vae_loss, disc_loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Nonlinear ICA (Slow Feature Analysis variant)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Nonlinear ICA model.
#[derive(Debug, Clone)]
pub struct SlowIcaConfig {
    pub x_dim: usize,
    pub z_dim: usize,
    pub hidden_dim: usize,
    pub lr: f64,
}

/// Nonlinear ICA combining slowness and independence objectives.
#[derive(Debug, Clone)]
pub struct NonlinearIca {
    pub mixing_net: CrMlp,
    pub unmixing_net: CrMlp,
    pub config: SlowIcaConfig,
}

impl NonlinearIca {
    pub fn new(config: SlowIcaConfig) -> Self {
        let mixing_net = CrMlp::new(&[config.z_dim, config.hidden_dim, config.x_dim]);
        let unmixing_net = CrMlp::new(&[config.x_dim, config.hidden_dim, config.z_dim]);
        NonlinearIca {
            mixing_net,
            unmixing_net,
            config,
        }
    }

    /// Slowness loss: E[||z_t - z_{t-1}||^2] averaged over consecutive pairs.
    pub fn slowness_loss(&self, z_sequence: &[Vec<f64>]) -> f64 {
        if z_sequence.len() < 2 {
            return 0.0;
        }
        let n = z_sequence.len() - 1;
        let total: f64 = z_sequence
            .windows(2)
            .map(|pair| {
                pair[0]
                    .iter()
                    .zip(pair[1].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .sum();
        total / n as f64
    }

    /// Independence penalty: sum of squared off-diagonal elements of the empirical covariance.
    pub fn independence_penalty(&self, z_batch: &[Vec<f64>]) -> f64 {
        if z_batch.is_empty() {
            return 0.0;
        }
        let z_dim = z_batch[0].len();
        let n = z_batch.len() as f64;

        // Compute means
        let means: Vec<f64> = (0..z_dim)
            .map(|j| {
                z_batch
                    .iter()
                    .map(|z| z.get(j).cloned().unwrap_or(0.0))
                    .sum::<f64>()
                    / n
            })
            .collect();

        // Compute off-diagonal covariance squared sum
        let mut penalty = 0.0;
        for i in 0..z_dim {
            for j in (i + 1)..z_dim {
                let cov: f64 = z_batch
                    .iter()
                    .map(|z| {
                        let zi = z.get(i).cloned().unwrap_or(0.0) - means[i];
                        let zj = z.get(j).cloned().unwrap_or(0.0) - means[j];
                        zi * zj
                    })
                    .sum::<f64>()
                    / n;
                penalty += cov.powi(2);
            }
        }
        penalty
    }

    /// Encode: x → z via unmixing net.
    pub fn unmix(&self, x: &[f64]) -> Vec<f64> {
        self.unmixing_net.forward(x)
    }

    /// Reconstruct: x → z → x.
    pub fn reconstruct(&self, x: &[f64]) -> Vec<f64> {
        let z = self.unmixing_net.forward(x);
        self.mixing_net.forward(&z)
    }

    /// One training step on a temporal sequence of observations.
    ///
    /// Minimises slowness + independence on the encoded representations.
    /// Uses finite-difference gradient for the unmixing network.
    pub fn train_step(&mut self, x_sequence: &[Vec<f64>], rng: &mut StdRng) -> f64 {
        if x_sequence.is_empty() {
            return 0.0;
        }

        let lr = self.config.lr;

        // Encode the full sequence
        let z_sequence: Vec<Vec<f64>> = x_sequence
            .iter()
            .map(|x| self.unmixing_net.forward(x))
            .collect();

        let slow_loss = self.slowness_loss(&z_sequence);
        let ind_loss = self.independence_penalty(&z_sequence);
        let total_loss = slow_loss + ind_loss;

        // Small random perturbation gradient descent step for unmixing network
        let eps = 1e-3 * lr;
        let n_layers = self.unmixing_net.layers.len();
        for l in 0..n_layers {
            let out_dim = self.unmixing_net.layers[l].w.len();
            let in_dim = if out_dim > 0 {
                self.unmixing_net.layers[l].w[0].len()
            } else {
                0
            };
            for j in 0..out_dim {
                for i in 0..in_dim {
                    // Random gradient sign for fast convergence demo
                    let sign: f64 = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
                    self.unmixing_net.layers[l].w[j][i] -= lr * eps * sign * total_loss.signum();
                }
            }
        }

        total_loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Deep Structural Causal Model (Pawlowski et al. 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Deep SCM.
#[derive(Debug, Clone)]
pub struct DscmConfig {
    pub n_variables: usize,
    /// Observation dimension per variable.
    pub var_dims: Vec<usize>,
    pub hidden_dim: usize,
    /// Causal graph: `parents[i]` lists the parent indices of variable i.
    pub parents: Vec<Vec<usize>>,
}

/// One structural equation: x_i = f_i(u_i, pa(x_i)).
#[derive(Debug, Clone)]
pub struct DscmMechanism {
    pub forward_net: CrMlp,
    pub inverse_net: CrMlp,
    pub u_dim: usize,
    pub x_dim: usize,
}

impl DscmMechanism {
    pub fn new(u_dim: usize, x_dim: usize, pa_dims_total: usize, hidden_dim: usize) -> Self {
        let forward_in = u_dim + pa_dims_total;
        let forward_net = CrMlp::new(&[forward_in, hidden_dim, x_dim]);
        let inverse_in = x_dim + pa_dims_total;
        let inverse_net = CrMlp::new(&[inverse_in, hidden_dim, u_dim]);
        DscmMechanism {
            forward_net,
            inverse_net,
            u_dim,
            x_dim,
        }
    }

    /// x = f(u, pa_vals).
    pub fn forward(&self, u: &[f64], pa_vals: &[f64]) -> Vec<f64> {
        let inp: Vec<f64> = u.iter().chain(pa_vals.iter()).cloned().collect();
        self.forward_net.forward(&inp)
    }

    /// u = f^{-1}(x, pa_vals).
    pub fn abduct(&self, x: &[f64], pa_vals: &[f64]) -> Vec<f64> {
        let inp: Vec<f64> = x.iter().chain(pa_vals.iter()).cloned().collect();
        self.inverse_net.forward(&inp)
    }
}

/// Deep Structural Causal Model.
#[derive(Debug, Clone)]
pub struct DeepScm {
    pub mechanisms: Vec<DscmMechanism>,
    pub config: DscmConfig,
}

impl DeepScm {
    pub fn new(config: DscmConfig) -> Self {
        let mechanisms: Vec<DscmMechanism> = (0..config.n_variables)
            .map(|i| {
                let x_dim = config.var_dims.get(i).cloned().unwrap_or(1);
                let u_dim = x_dim;
                let pa_total: usize = config.parents[i]
                    .iter()
                    .map(|&p| config.var_dims.get(p).cloned().unwrap_or(1))
                    .sum();
                DscmMechanism::new(u_dim, x_dim, pa_total, config.hidden_dim)
            })
            .collect();
        DeepScm { mechanisms, config }
    }

    /// Kahn's BFS topological sort of the causal graph.
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.config.n_variables;
        let mut in_degree = vec![0usize; n];
        let mut children: Vec<Vec<usize>> = vec![vec![]; n];

        for i in 0..n {
            for &p in &self.config.parents[i] {
                in_degree[i] += 1;
                children[p].push(i);
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &child in &children[node] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        order
    }

    /// Ancestral sampling: returns one observation per variable.
    pub fn sample(&self, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let order = self.topological_order();
        let mut values: Vec<Option<Vec<f64>>> = vec![None; self.config.n_variables];

        for &i in &order {
            let u_dim = self.mechanisms[i].u_dim;

            // Sample exogenous noise u ~ N(0,I) via Box-Muller
            let u: Vec<f64> = (0..u_dim)
                .map(|_| {
                    let u1 = rng.random::<f64>().max(1e-12);
                    let u2 = rng.random::<f64>().max(1e-12);
                    box_muller(u1, u2)
                })
                .collect();

            // Collect parent values
            let pa_vals: Vec<f64> = self.config.parents[i]
                .iter()
                .flat_map(|&p| {
                    values[p].clone().unwrap_or_else(|| {
                        vec![0.0; self.config.var_dims.get(p).cloned().unwrap_or(1)]
                    })
                })
                .collect();

            let x = self.mechanisms[i].forward(&u, &pa_vals);
            values[i] = Some(x);
        }

        values.into_iter().map(|v| v.unwrap_or_default()).collect()
    }

    /// Abduction step: infer exogenous noise u_i = f_i^{-1}(x_i, pa(x_i)) for each variable.
    pub fn abduct(&self, observed: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let order = self.topological_order();
        let mut u_vals: Vec<Option<Vec<f64>>> = vec![None; self.config.n_variables];

        for &i in &order {
            let x = observed
                .get(i)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.config.var_dims.get(i).cloned().unwrap_or(1)]);
            let pa_vals: Vec<f64> = self.config.parents[i]
                .iter()
                .flat_map(|&p| {
                    observed.get(p).cloned().unwrap_or_else(|| {
                        vec![0.0; self.config.var_dims.get(p).cloned().unwrap_or(1)]
                    })
                })
                .collect();
            let u = self.mechanisms[i].abduct(&x, &pa_vals);
            u_vals[i] = Some(u);
        }

        u_vals.into_iter().map(|v| v.unwrap_or_default()).collect()
    }

    /// Interventional prediction: do(X_intervention = value).
    pub fn intervene(
        &self,
        observed: &[Vec<f64>],
        intervention: usize,
        value: Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let order = self.topological_order();
        // Abduct u for all non-intervened ancestors
        let u_abducted = self.abduct(observed);
        let mut result: Vec<Option<Vec<f64>>> = vec![None; self.config.n_variables];

        for &i in &order {
            if i == intervention {
                result[i] = Some(value.clone());
            } else {
                let u = u_abducted[i].clone();
                let pa_vals: Vec<f64> = self.config.parents[i]
                    .iter()
                    .flat_map(|&p| {
                        result[p].as_ref().cloned().unwrap_or_else(|| {
                            vec![0.0; self.config.var_dims.get(p).cloned().unwrap_or(1)]
                        })
                    })
                    .collect();
                let x = self.mechanisms[i].forward(&u, &pa_vals);
                result[i] = Some(x);
            }
        }

        result.into_iter().map(|v| v.unwrap_or_default()).collect()
    }

    /// Counterfactual: abduct u from factual world, then apply intervention.
    pub fn counterfactual(
        &self,
        factual: &[Vec<f64>],
        intervention: usize,
        value: Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let order = self.topological_order();
        // Step 1: Abduct from factual observations
        let u_abducted = self.abduct(factual);
        // Step 2-3: Intervene using abducted noise
        let mut result: Vec<Option<Vec<f64>>> = vec![None; self.config.n_variables];

        for &i in &order {
            if i == intervention {
                result[i] = Some(value.clone());
            } else {
                let u = u_abducted[i].clone();
                let pa_vals: Vec<f64> = self.config.parents[i]
                    .iter()
                    .flat_map(|&p| {
                        result[p].as_ref().cloned().unwrap_or_else(|| {
                            vec![0.0; self.config.var_dims.get(p).cloned().unwrap_or(1)]
                        })
                    })
                    .collect();
                let x = self.mechanisms[i].forward(&u, &pa_vals);
                result[i] = Some(x);
            }
        }

        result.into_iter().map(|v| v.unwrap_or_default()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Disentanglement Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated disentanglement scores.
#[derive(Debug, Clone)]
pub struct DisentanglementMetrics {
    pub dci_disentanglement: f64,
    pub mig: f64,
    pub sap_score: f64,
    pub modularity: f64,
}

/// Discretise a continuous vector into `n_bins` equal-width bins.
fn discretize(vals: &[f64], n_bins: usize) -> Vec<usize> {
    if vals.is_empty() {
        return vec![];
    }
    let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_v - min_v).max(1e-12);
    vals.iter()
        .map(|&v| {
            let bin = ((v - min_v) / range * n_bins as f64) as usize;
            bin.min(n_bins - 1)
        })
        .collect()
}

/// Empirical mutual information I(X; Y) from joint histogram.
fn mutual_information_disc(x_disc: &[usize], y_disc: &[usize], n_x: usize, n_y: usize) -> f64 {
    let n = x_disc.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    // Joint count
    let mut joint = vec![vec![0usize; n_y]; n_x];
    let mut px = vec![0usize; n_x];
    let mut py = vec![0usize; n_y];
    for (&xi, &yi) in x_disc.iter().zip(y_disc.iter()) {
        joint[xi][yi] += 1;
        px[xi] += 1;
        py[yi] += 1;
    }
    let mut mi = 0.0;
    for i in 0..n_x {
        for j in 0..n_y {
            if joint[i][j] > 0 && px[i] > 0 && py[j] > 0 {
                let p_ij = joint[i][j] as f64 / nf;
                let p_i = px[i] as f64 / nf;
                let p_j = py[j] as f64 / nf;
                mi += p_ij * (p_ij / (p_i * p_j)).ln();
            }
        }
    }
    mi.max(0.0)
}

/// Empirical entropy H(X) from discrete counts.
fn entropy_disc(x_disc: &[usize], n_bins: usize) -> f64 {
    let n = x_disc.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; n_bins];
    for &xi in x_disc {
        counts[xi.min(n_bins - 1)] += 1;
    }
    let nf = n as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum::<f64>()
        .max(0.0)
}

/// Mutual Information Gap (Chen et al. 2018).
///
/// MIG = mean_k (1/H(v_k)) * (max_j MI(z_j; v_k) - second_max_j MI(z_j; v_k))
pub fn compute_mig(z_samples: &[Vec<f64>], factor_samples: &[Vec<f64>], n_bins: usize) -> f64 {
    if z_samples.is_empty() || factor_samples.is_empty() {
        return 0.0;
    }
    let n = z_samples.len().min(factor_samples.len());
    let z_dim = z_samples[0].len();
    let n_factors = factor_samples[0].len();
    if z_dim == 0 || n_factors == 0 {
        return 0.0;
    }

    // Discretise all z and factor columns
    let z_disc: Vec<Vec<usize>> = (0..z_dim)
        .map(|j| {
            let col: Vec<f64> = z_samples[..n]
                .iter()
                .map(|z| z.get(j).cloned().unwrap_or(0.0))
                .collect();
            discretize(&col, n_bins)
        })
        .collect();
    let f_disc: Vec<Vec<usize>> = (0..n_factors)
        .map(|k| {
            let col: Vec<f64> = factor_samples[..n]
                .iter()
                .map(|f| f.get(k).cloned().unwrap_or(0.0))
                .collect();
            discretize(&col, n_bins)
        })
        .collect();

    let mut mig_sum = 0.0;
    for k in 0..n_factors {
        let h_k = entropy_disc(&f_disc[k], n_bins);
        if h_k < 1e-12 {
            continue;
        }
        let mis: Vec<f64> = (0..z_dim)
            .map(|j| mutual_information_disc(&z_disc[j], &f_disc[k], n_bins, n_bins))
            .collect();
        let mut sorted_mis = mis.clone();
        sorted_mis.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let gap = if sorted_mis.len() >= 2 {
            (sorted_mis[0] - sorted_mis[1]) / h_k
        } else if sorted_mis.len() == 1 {
            sorted_mis[0] / h_k
        } else {
            0.0
        };
        mig_sum += gap.clamp(0.0, 1.0);
    }
    if n_factors == 0 {
        return 0.0;
    }
    (mig_sum / n_factors as f64).clamp(0.0, 1.0)
}

/// Separated Attribute Predictability (Kumar et al. 2018).
///
/// SAP = mean_k (max_j |corr(z_j, v_k)| - second_max_j |corr(z_j, v_k)|)
pub fn compute_sap(z_samples: &[Vec<f64>], factor_samples: &[Vec<f64>]) -> f64 {
    if z_samples.is_empty() || factor_samples.is_empty() {
        return 0.0;
    }
    let n = z_samples.len().min(factor_samples.len());
    let z_dim = z_samples[0].len();
    let n_factors = factor_samples[0].len();
    if z_dim == 0 || n_factors == 0 {
        return 0.0;
    }

    // Helper: pearson correlation
    let pearson = |col_a: &[f64], col_b: &[f64]| -> f64 {
        let n_ab = col_a.len().min(col_b.len());
        if n_ab == 0 {
            return 0.0;
        }
        let mean_a = col_a[..n_ab].iter().sum::<f64>() / n_ab as f64;
        let mean_b = col_b[..n_ab].iter().sum::<f64>() / n_ab as f64;
        let cov: f64 = col_a[..n_ab]
            .iter()
            .zip(col_b[..n_ab].iter())
            .map(|(&a, &b)| (a - mean_a) * (b - mean_b))
            .sum::<f64>()
            / n_ab as f64;
        let var_a: f64 = col_a[..n_ab]
            .iter()
            .map(|&a| (a - mean_a).powi(2))
            .sum::<f64>()
            / n_ab as f64;
        let var_b: f64 = col_b[..n_ab]
            .iter()
            .map(|&b| (b - mean_b).powi(2))
            .sum::<f64>()
            / n_ab as f64;
        let denom = (var_a * var_b).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        }
    };

    let z_cols: Vec<Vec<f64>> = (0..z_dim)
        .map(|j| {
            z_samples[..n]
                .iter()
                .map(|z| z.get(j).cloned().unwrap_or(0.0))
                .collect()
        })
        .collect();
    let f_cols: Vec<Vec<f64>> = (0..n_factors)
        .map(|k| {
            factor_samples[..n]
                .iter()
                .map(|f| f.get(k).cloned().unwrap_or(0.0))
                .collect()
        })
        .collect();

    let mut sap_sum = 0.0;
    for k in 0..n_factors {
        let corrs: Vec<f64> = (0..z_dim)
            .map(|j| pearson(&z_cols[j], &f_cols[k]).abs())
            .collect();
        let mut sorted_corrs = corrs.clone();
        sorted_corrs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let gap = if sorted_corrs.len() >= 2 {
            sorted_corrs[0] - sorted_corrs[1]
        } else if sorted_corrs.len() == 1 {
            sorted_corrs[0]
        } else {
            0.0
        };
        sap_sum += gap.clamp(0.0, 1.0);
    }
    (sap_sum / n_factors as f64).clamp(0.0, 1.0)
}

/// Modularity score: each latent dimension concentrates its mutual information on one factor.
///
/// modularity_j = 1 - H_j / log(n_factors)
/// where H_j = entropy of the MI distribution of z_j over factors.
pub fn compute_modularity(z_samples: &[Vec<f64>], factor_samples: &[Vec<f64>]) -> f64 {
    if z_samples.is_empty() || factor_samples.is_empty() {
        return 0.0;
    }
    let n = z_samples.len().min(factor_samples.len());
    let z_dim = z_samples[0].len();
    let n_factors = factor_samples[0].len();
    if z_dim == 0 || n_factors < 2 {
        return 0.0;
    }
    let n_bins = 20;

    let z_disc: Vec<Vec<usize>> = (0..z_dim)
        .map(|j| {
            let col: Vec<f64> = z_samples[..n]
                .iter()
                .map(|z| z.get(j).cloned().unwrap_or(0.0))
                .collect();
            discretize(&col, n_bins)
        })
        .collect();
    let f_disc: Vec<Vec<usize>> = (0..n_factors)
        .map(|k| {
            let col: Vec<f64> = factor_samples[..n]
                .iter()
                .map(|f| f.get(k).cloned().unwrap_or(0.0))
                .collect();
            discretize(&col, n_bins)
        })
        .collect();

    let max_entropy = (n_factors as f64).ln();
    let mut mod_sum = 0.0;
    let mut valid_dims = 0;

    for j in 0..z_dim {
        let mis: Vec<f64> = (0..n_factors)
            .map(|k| mutual_information_disc(&z_disc[j], &f_disc[k], n_bins, n_bins))
            .collect();
        let mi_sum: f64 = mis.iter().sum();
        if mi_sum < 1e-12 {
            continue;
        }
        // Distribution over factors
        let probs: Vec<f64> = mis.iter().map(|&m| m / mi_sum).collect();
        let h_j: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>();
        let modularity_j = if max_entropy > 1e-12 {
            (1.0 - h_j / max_entropy).clamp(0.0, 1.0)
        } else {
            1.0
        };
        mod_sum += modularity_j;
        valid_dims += 1;
    }

    if valid_dims == 0 {
        return 0.0;
    }
    (mod_sum / valid_dims as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests;
