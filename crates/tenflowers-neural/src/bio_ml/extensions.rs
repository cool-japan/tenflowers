//! Bio ML extensions — sections 4 (Single-Cell Genomics) and 5 (Drug Discovery).

use super::{matvec, normal_samples_f64, random_matrix, random_vec, relu, sigmoid, softmax, vecadd};
use tenflowers_core::{Result, TensorError};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// 4. Single-Cell Genomics
// ─────────────────────────────────────────────────────────────────────────────

/// scRNA-seq normalization: library-size normalize then log1p transform.
#[derive(Debug, Clone)]
pub struct ScRnaSeqNormalizer {
    /// Target library size (counts per cell, default 10 000).
    pub target_sum: f64,
}

impl ScRnaSeqNormalizer {
    pub fn new(target_sum: f64) -> Self {
        Self { target_sum }
    }

    /// Normalize a matrix of raw counts (cells × genes).
    /// Returns log1p-transformed normalized counts.
    pub fn normalize(&self, counts: &[Vec<f64>]) -> Vec<Vec<f64>> {
        counts
            .iter()
            .map(|cell| {
                let lib_size: f64 = cell.iter().sum();
                let scale = if lib_size > 0.0 {
                    self.target_sum / lib_size
                } else {
                    1.0
                };
                cell.iter().map(|&c| (c * scale + 1.0).ln()).collect()
            })
            .collect()
    }
}

/// PCA via power iteration (randomized SVD, simplified).
///
/// Fits top-K principal components on a data matrix (samples × features).
#[derive(Debug, Clone)]
pub struct PcaReducer {
    pub n_components: usize,
    /// Mean of training data (features).
    pub mean: Vec<f64>,
    /// Principal components: `[n_components][n_features]`.
    pub components: Vec<Vec<f64>>,
}

impl PcaReducer {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            mean: Vec::new(),
            components: Vec::new(),
        }
    }

    /// Fit PCA via power iteration.
    pub fn fit(&mut self, x: &[Vec<f64>], seed: u64) -> Result<()> {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
        let n = x.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "PcaReducer::fit",
                "empty data",
            ));
        }
        let d = x[0].len();
        // Compute mean
        self.mean = vec![0.0_f64; d];
        for row in x {
            for (j, &v) in row.iter().enumerate() {
                self.mean[j] += v;
            }
        }
        for m in self.mean.iter_mut() {
            *m /= n as f64;
        }

        // Center data
        let xc: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .zip(self.mean.iter())
                    .map(|(&v, &m)| v - m)
                    .collect()
            })
            .collect();

        // Power iteration for each component (deflation)
        let mut residual = xc.clone();
        self.components = Vec::with_capacity(self.n_components);

        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..self.n_components.min(d) {
            // Random init
            let mut v: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
            // Normalize
            let vnorm: f64 = v.iter().map(|x| x.powi(2)).sum::<f64>().sqrt().max(1e-10);
            for vi in v.iter_mut() {
                *vi /= vnorm;
            }

            // Power iterations
            for _ in 0..20 {
                // Av = X^T (X v)
                let xv: Vec<f64> = residual
                    .iter()
                    .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f64>())
                    .collect();
                let mut av = vec![0.0_f64; d];
                for (i, row) in residual.iter().enumerate() {
                    for (j, &r) in row.iter().enumerate() {
                        av[j] += r * xv[i];
                    }
                }
                let avnorm: f64 = av.iter().map(|x| x.powi(2)).sum::<f64>().sqrt().max(1e-10);
                for vi in av.iter_mut() {
                    *vi /= avnorm;
                }
                v = av;
            }

            // Deflate: residual -= (residual · v) v^T
            let projections: Vec<f64> = residual
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f64>())
                .collect();
            for (i, row) in residual.iter_mut().enumerate() {
                let p = projections[i];
                for (j, r) in row.iter_mut().enumerate() {
                    *r -= p * v[j];
                }
            }

            self.components.push(v);
        }
        Ok(())
    }

    /// Project data onto principal components.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.components.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "PcaReducer::transform",
                "call fit() first",
            ));
        }
        x.iter()
            .map(|row| {
                let centered: Vec<f64> = row
                    .iter()
                    .zip(self.mean.iter())
                    .map(|(&v, &m)| v - m)
                    .collect();
                Ok(self
                    .components
                    .iter()
                    .map(|pc| {
                        pc.iter()
                            .zip(centered.iter())
                            .map(|(&a, &b)| a * b)
                            .sum::<f64>()
                    })
                    .collect())
            })
            .collect()
    }
}

/// scVI-style VAE for scRNA-seq (Negative Binomial decoder approximation).
#[derive(Debug, Clone)]
pub struct VariationalAutoencoder {
    pub input_dim: usize,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    // Encoder
    enc_w1: Vec<Vec<f64>>,
    enc_b1: Vec<f64>,
    enc_mu_w: Vec<Vec<f64>>,
    enc_mu_b: Vec<f64>,
    enc_lv_w: Vec<Vec<f64>>,
    enc_lv_b: Vec<f64>,
    // Decoder
    dec_w1: Vec<Vec<f64>>,
    dec_b1: Vec<f64>,
    dec_w2: Vec<Vec<f64>>,
    dec_b2: Vec<f64>,
}

impl VariationalAutoencoder {
    pub fn new(input_dim: usize, latent_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            input_dim,
            latent_dim,
            hidden_dim,
            enc_w1: random_matrix(hidden_dim, input_dim, &mut rng),
            enc_b1: random_vec(hidden_dim, &mut rng),
            enc_mu_w: random_matrix(latent_dim, hidden_dim, &mut rng),
            enc_mu_b: random_vec(latent_dim, &mut rng),
            enc_lv_w: random_matrix(latent_dim, hidden_dim, &mut rng),
            enc_lv_b: random_vec(latent_dim, &mut rng),
            dec_w1: random_matrix(hidden_dim, latent_dim, &mut rng),
            dec_b1: random_vec(hidden_dim, &mut rng),
            dec_w2: random_matrix(input_dim, hidden_dim, &mut rng),
            dec_b2: random_vec(input_dim, &mut rng),
        }
    }

    /// Encode input → (mu, log_var).
    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h: Vec<f64> = vecadd(&matvec(&self.enc_w1, x), &self.enc_b1)
            .into_iter()
            .map(relu)
            .collect();
        let mu = vecadd(&matvec(&self.enc_mu_w, &h), &self.enc_mu_b);
        let lv = vecadd(&matvec(&self.enc_lv_w, &h), &self.enc_lv_b);
        (mu, lv)
    }

    /// Reparameterize: z = mu + eps * exp(0.5 * log_var).
    pub fn reparameterize(&self, mu: &[f64], log_var: &[f64], seed: u64) -> Vec<f64> {
        let eps = normal_samples_f64(mu.len(), seed);
        mu.iter()
            .zip(log_var.iter())
            .zip(eps.iter())
            .map(|((&m, &lv), &e)| m + e * (0.5 * lv).exp())
            .collect()
    }

    /// Decode latent z → reconstruction.
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        let h: Vec<f64> = vecadd(&matvec(&self.dec_w1, z), &self.dec_b1)
            .into_iter()
            .map(relu)
            .collect();
        vecadd(&matvec(&self.dec_w2, &h), &self.dec_b2)
            .into_iter()
            .map(relu) // non-negative gene expression
            .collect()
    }

    /// ELBO loss: reconstruction MSE + KL divergence.
    pub fn elbo_loss(&self, x: &[f64], recon: &[f64], mu: &[f64], log_var: &[f64]) -> f64 {
        let recon_loss: f64 = x
            .iter()
            .zip(recon.iter())
            .map(|(&xi, &ri)| (xi - ri).powi(2))
            .sum::<f64>()
            / x.len() as f64;
        let kl: f64 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| -0.5 * (1.0 + lv - m.powi(2) - lv.exp()))
            .sum();
        recon_loss + kl / mu.len() as f64
    }
}

/// Classify cell types from gene expression: MLP → softmax.
#[derive(Debug, Clone)]
pub struct CellTypeClassifier {
    pub n_classes: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl CellTypeClassifier {
    pub fn new(input_dim: usize, hidden_dim: usize, n_classes: usize, seed: u64) -> Self {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            n_classes,
            w1: random_matrix(hidden_dim, input_dim, &mut rng),
            b1: random_vec(hidden_dim, &mut rng),
            w2: random_matrix(n_classes, hidden_dim, &mut rng),
            b2: random_vec(n_classes, &mut rng),
        }
    }

    /// Returns class probabilities (softmax).
    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        let h: Vec<f64> = vecadd(&matvec(&self.w1, x), &self.b1)
            .into_iter()
            .map(relu)
            .collect();
        let logits = vecadd(&matvec(&self.w2, &h), &self.b2);
        softmax(&logits)
    }

    /// Returns argmax class.
    pub fn predict(&self, x: &[f64]) -> usize {
        let probs = self.predict_proba(x);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Pseudotime trajectory inference via k-NN graph + BFS from root cell.
#[derive(Debug, Clone)]
pub struct TrajectoryInference {
    pub k: usize,
}

impl TrajectoryInference {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute Euclidean distance between two cell expression vectors.
    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Build k-NN adjacency list from cell embeddings.
    fn build_knn(&self, cells: &[Vec<f64>]) -> Vec<Vec<usize>> {
        let n = cells.len();
        let k = self.k.min(n.saturating_sub(1));
        (0..n)
            .map(|i| {
                let mut dists: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, Self::dist(&cells[i], &cells[j])))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(k).map(|&(j, _)| j).collect()
            })
            .collect()
    }

    /// Compute pseudotime via BFS from root cell.
    /// Returns a pseudotime value for each cell (0.0 = root).
    pub fn compute_pseudotime(&self, cells: &[Vec<f64>], root: usize) -> Result<Vec<f64>> {
        let n = cells.len();
        if root >= n {
            return Err(TensorError::invalid_argument_op(
                "TrajectoryInference::compute_pseudotime",
                "root index out of bounds",
            ));
        }
        let adj = self.build_knn(cells);
        let mut pseudotime = vec![f64::INFINITY; n];
        pseudotime[root] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);

        while let Some(cell) = queue.pop_front() {
            let current_pt = pseudotime[cell];
            for &neighbor in &adj[cell] {
                if pseudotime[neighbor].is_infinite() {
                    pseudotime[neighbor] = current_pt + 1.0;
                    queue.push_back(neighbor);
                }
            }
        }
        // Replace unreachable cells with max pseudotime + 1
        let max_pt = pseudotime
            .iter()
            .filter(|&&v| v.is_finite())
            .cloned()
            .fold(0.0_f64, f64::max);
        for v in pseudotime.iter_mut() {
            if v.is_infinite() {
                *v = max_pt + 1.0;
            }
        }
        Ok(pseudotime)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Drug Discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Tanimoto coefficient between binary Morgan fingerprint vectors.
#[derive(Debug, Clone)]
pub struct FingerprintSimilarity;

impl FingerprintSimilarity {
    /// Tanimoto (Jaccard) similarity for bit fingerprints stored as `f64` (0 or 1).
    pub fn tanimoto(fp1: &[f64], fp2: &[f64]) -> f64 {
        let intersection: f64 = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| a.min(b)).sum();
        let union: f64 = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| a.max(b)).sum();
        if union < 1e-10 {
            1.0
        } else {
            intersection / union
        }
    }
}

/// Drug-target interaction predictor.
///
/// Concatenate drug + protein embeddings → two-layer MLP → sigmoid DTI score.
#[derive(Debug, Clone)]
pub struct DrugTargetInteraction {
    pub drug_dim: usize,
    pub protein_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl DrugTargetInteraction {
    pub fn new(drug_dim: usize, protein_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        let in_dim = drug_dim + protein_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            drug_dim,
            protein_dim,
            w1: random_matrix(hidden_dim, in_dim, &mut rng),
            b1: random_vec(hidden_dim, &mut rng),
            w2: random_matrix(1, hidden_dim, &mut rng),
            b2: random_vec(1, &mut rng),
        }
    }

    /// Predict interaction score in [0, 1].
    pub fn score(&self, drug_emb: &[f64], protein_emb: &[f64]) -> f64 {
        let mut feat = drug_emb.to_vec();
        feat.extend_from_slice(protein_emb);
        let h1: Vec<f64> = vecadd(&matvec(&self.w1, &feat), &self.b1)
            .into_iter()
            .map(relu)
            .collect();
        let logit = matvec(&self.w2, &h1)[0] + self.b2[0];
        sigmoid(logit)
    }
}

/// Rank a compound library by DTI score against a target.
#[derive(Debug, Clone)]
pub struct VirtualScreening {
    pub dti: DrugTargetInteraction,
}

impl VirtualScreening {
    pub fn new(dti: DrugTargetInteraction) -> Self {
        Self { dti }
    }

    /// Returns compound indices sorted descending by DTI score.
    pub fn screen(&self, compounds: &[Vec<f64>], target: &[f64]) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = compounds
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.dti.score(c, target)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().map(|&(i, _)| i).collect()
    }
}

/// Multi-task ADMET property predictor.
///
/// Predicts: absorption (0), distribution (1), metabolism (2),
///           excretion (3), toxicity (4).
#[derive(Debug, Clone)]
pub struct AdmetPredictor {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl AdmetPredictor {
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            w1: random_matrix(hidden_dim, input_dim, &mut rng),
            b1: random_vec(hidden_dim, &mut rng),
            w2: random_matrix(5, hidden_dim, &mut rng),
            b2: random_vec(5, &mut rng),
        }
    }

    /// Returns 5 ADMET scores, each in [0, 1].
    pub fn predict(&self, mol_features: &[f64]) -> Vec<f64> {
        let h1: Vec<f64> = vecadd(&matvec(&self.w1, mol_features), &self.b1)
            .into_iter()
            .map(relu)
            .collect();
        let logits = vecadd(&matvec(&self.w2, &h1), &self.b2);
        logits.into_iter().map(sigmoid).collect()
    }
}

/// Simplified rigid-body molecular docking via grid energy scoring.
///
/// Energy = Σ_{pairs} [ε_LJ ((r_min/r)^12 - 2(r_min/r)^6) + k_e q_i q_j / r]
/// where `r` is the distance, `r_min=3.5 Å`, `ε_LJ=0.1`, `k_e=332`.
#[derive(Debug, Clone)]
pub struct MolecularDocking {
    pub r_min: f64,
    pub eps_lj: f64,
    pub k_elec: f64,
}

impl MolecularDocking {
    pub fn new() -> Self {
        Self {
            r_min: 3.5,
            eps_lj: 0.1,
            k_elec: 332.0,
        }
    }

    /// Score a ligand pose against a binding pocket.
    ///
    /// `ligand_atoms`: list of (xyz, charge) for ligand atoms.
    /// `pocket_atoms`: list of (xyz, charge) for pocket atoms.
    pub fn score(&self, ligand_atoms: &[([f64; 3], f64)], pocket_atoms: &[([f64; 3], f64)]) -> f64 {
        use super::atom_dist;
        let mut energy = 0.0_f64;
        for &(la, lq) in ligand_atoms {
            for &(pa, pq) in pocket_atoms {
                let r = atom_dist(&la, &pa).max(0.5);
                let ratio = self.r_min / r;
                let lj = self.eps_lj * (ratio.powi(12) - 2.0 * ratio.powi(6));
                let elec = self.k_elec * lq * pq / r;
                energy += lj + elec;
            }
        }
        energy
    }
}

impl Default for MolecularDocking {
    fn default() -> Self {
        Self::new()
    }
}
