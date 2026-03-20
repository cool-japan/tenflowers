//! §5 — Evaluation Metrics.
//!
//! - [`FrechetInceptionDistance`]   — FID from feature statistics
//! - [`InceptionScore`]             — IS = exp(E_x[KL(p(y|x) ‖ p(y))])
//! - [`RecallPrecision`]            — manifold precision/recall
//! - [`DiffusionLoss`]              — MSE + VLB combined loss
//! - [`NoisePredictionEvaluator`]   — SNR-weighted per-timestep tracking

use tenflowers_core::TensorError;

use super::helpers::{make_err, softmax, sq_dist};

// ─────────────────────────────────────────────────────────────────────────────

/// Fréchet Inception Distance (FID) approximation.
///
/// Uses the diagonal covariance approximation:
/// `FID = ‖μ_r − μ_g‖² + Σ_i (σ_r[i] + σ_g[i] − 2·√(σ_r[i]·σ_g[i]))`
#[derive(Debug, Clone)]
pub struct FrechetInceptionDistance {
    pub feature_dim: usize,
    pub mu_real: Vec<f64>,
    pub sigma_real: Vec<f64>,
}

impl FrechetInceptionDistance {
    /// Create a new FID computer with pre-computed real statistics.
    pub fn new(mu_real: Vec<f64>, sigma_real: Vec<f64>) -> Result<Self, TensorError> {
        if mu_real.len() != sigma_real.len() {
            return Err(make_err(
                "FrechetInceptionDistance: mu and sigma length mismatch",
            ));
        }
        let feature_dim = mu_real.len();
        Ok(Self {
            feature_dim,
            mu_real,
            sigma_real,
        })
    }

    /// Compute FID given generated feature statistics.
    pub fn compute(&self, mu_gen: &[f64], sigma_gen: &[f64]) -> Result<f64, TensorError> {
        if mu_gen.len() != self.feature_dim || sigma_gen.len() != self.feature_dim {
            return Err(make_err(
                "FrechetInceptionDistance: generated statistics dimension mismatch",
            ));
        }
        let mu_dist_sq = sq_dist(&self.mu_real, mu_gen);
        let sigma_term: f64 = self
            .sigma_real
            .iter()
            .zip(sigma_gen.iter())
            .map(|(&sr, &sg)| {
                let sqrt_prod = (sr.abs() * sg.abs()).sqrt();
                sr.abs() + sg.abs() - 2.0 * sqrt_prod
            })
            .sum();
        Ok(mu_dist_sq + sigma_term)
    }

    /// Compute from raw feature samples (estimates μ and σ internally).
    pub fn compute_from_samples(
        &self,
        real_samples: &[Vec<f64>],
        gen_samples: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        if real_samples.is_empty() || gen_samples.is_empty() {
            return Err(make_err("FrechetInceptionDistance: empty sample sets"));
        }
        let (mu_r, sigma_r) = feature_stats(real_samples, self.feature_dim)?;
        let (mu_g, sigma_g) = feature_stats(gen_samples, self.feature_dim)?;
        let fid_obj = FrechetInceptionDistance::new(mu_r, sigma_r)?;
        fid_obj.compute(&mu_g, &sigma_g)
    }
}

/// Compute per-feature mean and std from a list of feature vectors.
pub(crate) fn feature_stats(
    samples: &[Vec<f64>],
    feature_dim: usize,
) -> Result<(Vec<f64>, Vec<f64>), TensorError> {
    let n = samples.len() as f64;
    let mut mu = vec![0.0_f64; feature_dim];
    for s in samples {
        if s.len() != feature_dim {
            return Err(make_err("feature_stats: dimension mismatch"));
        }
        for (m, &v) in mu.iter_mut().zip(s.iter()) {
            *m += v / n;
        }
    }
    let mut var = vec![0.0_f64; feature_dim];
    for s in samples {
        for (v2, (&vi, &mi)) in var.iter_mut().zip(s.iter().zip(mu.iter())) {
            *v2 += (vi - mi).powi(2) / n;
        }
    }
    let sigma: Vec<f64> = var.iter().map(|&v| v.sqrt()).collect();
    Ok((mu, sigma))
}

// ─────────────────────────────────────────────────────────────────────────────

/// Inception Score (IS).
///
/// `IS = exp(E_x [KL(p(y|x) ‖ p(y))])`
#[derive(Debug, Clone)]
pub struct InceptionScore;

impl InceptionScore {
    /// Compute IS from a list of conditional class probability vectors.
    pub fn compute(conditional_probs: &[Vec<f64>]) -> Result<f64, TensorError> {
        if conditional_probs.is_empty() {
            return Err(make_err("InceptionScore: empty input"));
        }
        let num_classes = conditional_probs[0].len();
        if num_classes == 0 {
            return Err(make_err("InceptionScore: zero classes"));
        }

        let probs: Vec<Vec<f64>> = conditional_probs.iter().map(|v| softmax(v)).collect();

        let n = probs.len() as f64;
        let mut marginal = vec![0.0_f64; num_classes];
        for p in &probs {
            if p.len() != num_classes {
                return Err(make_err("InceptionScore: inconsistent number of classes"));
            }
            for (m, &pi) in marginal.iter_mut().zip(p.iter()) {
                *m += pi / n;
            }
        }

        let kl_sum: f64 = probs
            .iter()
            .map(|p| {
                p.iter()
                    .zip(marginal.iter())
                    .map(|(&pyx, &py)| {
                        if pyx > 1e-15 && py > 1e-15 {
                            pyx * (pyx / py).ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();

        Ok((kl_sum / n).exp())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Manifold Precision and Recall for generative models (Kynkäänniemi et al. 2019).
#[derive(Debug, Clone)]
pub struct RecallPrecision {
    pub real_features: Vec<Vec<f64>>,
    pub gen_features: Vec<Vec<f64>>,
    pub k: usize,
}

impl RecallPrecision {
    /// Create a new precision/recall evaluator.
    pub fn new(
        real_features: Vec<Vec<f64>>,
        gen_features: Vec<Vec<f64>>,
        k: usize,
    ) -> Result<Self, TensorError> {
        if real_features.is_empty() || gen_features.is_empty() {
            return Err(make_err("RecallPrecision: empty feature sets"));
        }
        if k == 0 {
            return Err(make_err("RecallPrecision: k must be >= 1"));
        }
        Ok(Self {
            real_features,
            gen_features,
            k,
        })
    }

    fn knn_radius(point: &[f64], set: &[Vec<f64>], k: usize) -> f64 {
        let mut dists: Vec<f64> = set.iter().map(|s| sq_dist(point, s)).collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = k.min(dists.len()).saturating_sub(1);
        dists[idx].sqrt()
    }

    /// Compute precision: fraction of generated samples inside the real manifold.
    pub fn precision(&self) -> f64 {
        let inside = self
            .gen_features
            .iter()
            .filter(|g| {
                self.real_features.iter().any(|r| {
                    let d = sq_dist(g, r).sqrt();
                    let radius = Self::knn_radius(r, &self.real_features, self.k);
                    d <= radius
                })
            })
            .count();
        inside as f64 / self.gen_features.len() as f64
    }

    /// Compute recall: fraction of real samples inside the generated manifold.
    pub fn recall(&self) -> f64 {
        let inside = self
            .real_features
            .iter()
            .filter(|r| {
                self.gen_features.iter().any(|g| {
                    let d = sq_dist(r, g).sqrt();
                    let radius = Self::knn_radius(g, &self.gen_features, self.k);
                    d <= radius
                })
            })
            .count();
        inside as f64 / self.real_features.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Combined diffusion loss: simple MSE + variational lower bound.
///
/// `L = λ · L_simple + (1 − λ) · L_vlb`
#[derive(Debug, Clone)]
pub struct DiffusionLoss {
    pub lambda: f64,
}

impl DiffusionLoss {
    /// Create a new diffusion loss.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Simple MSE loss on noise prediction.
    pub fn simple_loss(&self, eps_pred: &[f64], eps_target: &[f64]) -> Result<f64, TensorError> {
        if eps_pred.len() != eps_target.len() {
            return Err(make_err("DiffusionLoss: dimension mismatch in simple_loss"));
        }
        let mse = eps_pred
            .iter()
            .zip(eps_target.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / eps_pred.len() as f64;
        Ok(mse)
    }

    /// Variational lower bound loss (KL divergence approximation).
    pub fn vlb_loss(
        &self,
        mu_pred: &[f64],
        mu_target: &[f64],
        sigma_target_sq: f64,
    ) -> Result<f64, TensorError> {
        if mu_pred.len() != mu_target.len() {
            return Err(make_err("DiffusionLoss: dimension mismatch in vlb_loss"));
        }
        let kl = mu_pred
            .iter()
            .zip(mu_target.iter())
            .map(|(&a, &b)| 0.5 * (a - b).powi(2) / sigma_target_sq.max(1e-8))
            .sum::<f64>()
            / mu_pred.len() as f64;
        Ok(kl)
    }

    /// Combined loss: `λ · L_simple + (1 − λ) · L_vlb`.
    pub fn combined(
        &self,
        eps_pred: &[f64],
        eps_target: &[f64],
        mu_pred: &[f64],
        mu_target: &[f64],
        sigma_target_sq: f64,
    ) -> Result<f64, TensorError> {
        let l_simple = self.simple_loss(eps_pred, eps_target)?;
        let l_vlb = self.vlb_loss(mu_pred, mu_target, sigma_target_sq)?;
        Ok(self.lambda * l_simple + (1.0 - self.lambda) * l_vlb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Per-timestep noise prediction evaluator with SNR-weighted loss tracking.
#[derive(Debug, Clone)]
pub struct NoisePredictionEvaluator {
    pub num_timesteps: usize,
    pub timestep_losses: Vec<f64>,
    pub timestep_counts: Vec<u64>,
    pub snr_weights: Vec<f64>,
}

impl NoisePredictionEvaluator {
    /// Create a new evaluator from a noise schedule.
    ///
    /// `alpha_bars`: pre-computed ᾱ_t.
    /// `gamma`: Min-SNR clipping parameter (typically 5.0).
    pub fn new(alpha_bars: &[f64], gamma: f64) -> Result<Self, TensorError> {
        if alpha_bars.is_empty() {
            return Err(make_err("NoisePredictionEvaluator: empty alpha_bars"));
        }
        let num_timesteps = alpha_bars.len();
        let snr_weights: Vec<f64> = alpha_bars
            .iter()
            .map(|&ab| {
                let snr = ab / (1.0 - ab).max(1e-8);
                snr.min(gamma) / snr.max(1e-8)
            })
            .collect();

        Ok(Self {
            num_timesteps,
            timestep_losses: vec![0.0; num_timesteps],
            timestep_counts: vec![0; num_timesteps],
            snr_weights,
        })
    }

    /// Record a noise prediction loss at a given timestep.
    pub fn record(&mut self, t: usize, loss: f64) -> Result<(), TensorError> {
        if t >= self.num_timesteps {
            return Err(make_err("NoisePredictionEvaluator: t out of range"));
        }
        self.timestep_losses[t] += loss;
        self.timestep_counts[t] += 1;
        Ok(())
    }

    /// Compute SNR-weighted loss over all recorded timesteps.
    pub fn snr_weighted_loss(&self) -> f64 {
        let mut total = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for t in 0..self.num_timesteps {
            if self.timestep_counts[t] > 0 {
                let avg = self.timestep_losses[t] / self.timestep_counts[t] as f64;
                let w = self.snr_weights[t];
                total += w * avg;
                weight_sum += w;
            }
        }
        if weight_sum < 1e-15 {
            0.0
        } else {
            total / weight_sum
        }
    }

    /// Get per-timestep average losses.
    pub fn per_timestep_avg(&self) -> Vec<f64> {
        self.timestep_losses
            .iter()
            .zip(self.timestep_counts.iter())
            .map(|(&l, &c)| if c > 0 { l / c as f64 } else { 0.0 })
            .collect()
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.timestep_losses.fill(0.0);
        self.timestep_counts.fill(0);
    }
}
