//! CASTLE + CausalBoosting — causal structure learning via autoencoders and gradient boosting.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::continuous_opt::h_constraint;
use super::shared::mean_f32;

// ─────────────────────────────────────────────────────────────────────────────
// 6. CASTLE — Causal Structure Learning via neural autoencoder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CASTLE.
#[derive(Debug, Clone)]
pub struct CastleConfig {
    /// Number of observed variables.
    pub n_vars: usize,
    /// Hidden dimension for per-variable encoder/decoder.
    pub hidden_dim: usize,
    /// Regularisation weight for the acyclicity constraint.
    pub lambda_reg: f32,
}

impl Default for CastleConfig {
    fn default() -> Self {
        Self {
            n_vars: 4,
            hidden_dim: 8,
            lambda_reg: 1.0,
        }
    }
}

/// Per-variable causal autoencoder: encoder maps X → h_j, decoder maps h → x̂_j.
#[derive(Debug, Clone)]
pub struct CausalAutoEncoder {
    /// Per-variable encoder weight matrices (n_vars × [n_vars × hidden_dim]).
    pub encoder_weights: Vec<Vec<f32>>,
    /// Per-variable decoder weight matrices (n_vars × [hidden_dim × 1]).
    pub decoder_weights: Vec<Vec<f32>>,
    /// Number of variables.
    pub n_vars: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl CausalAutoEncoder {
    /// Create a randomly initialised CASTLE autoencoder.
    pub fn new(config: &CastleConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = config.n_vars;
        let h = config.hidden_dim;
        let encoder_weights: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..n * h)
                    .map(|_| (rng.random::<f32>() - 0.5) * 0.1)
                    .collect()
            })
            .collect();
        let decoder_weights: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..h).map(|_| (rng.random::<f32>() - 0.5) * 0.1).collect())
            .collect();
        Self {
            encoder_weights,
            decoder_weights,
            n_vars: n,
            hidden_dim: h,
        }
    }

    /// Forward pass for variable j: encode all inputs → hidden → decode to x̂_j.
    pub fn forward_j(&self, x: &[f32], j: usize) -> f32 {
        let n = self.n_vars.min(x.len());
        let h = self.hidden_dim;
        if j >= self.encoder_weights.len() || j >= self.decoder_weights.len() {
            return 0.0;
        }
        let enc_w = &self.encoder_weights[j];
        // Encode: hidden[h] = sum_i x[i] * enc_w[i*h + hh]
        let mut hidden = vec![0.0f32; h];
        for i in 0..n {
            for hh in 0..h {
                let idx = i * h + hh;
                if idx < enc_w.len() {
                    hidden[hh] += x[i] * enc_w[idx];
                }
            }
        }
        // ReLU activation
        for hh in 0..h {
            hidden[hh] = hidden[hh].max(0.0);
        }
        // Decode: x̂_j = sum_h hidden[h] * dec_w[h]
        let dec_w = &self.decoder_weights[j];
        let out: f32 = hidden
            .iter()
            .zip(dec_w.iter())
            .map(|(&hv, &dw)| hv * dw)
            .sum();
        out
    }

    /// Extract the induced adjacency matrix W\[i\]\[j\] = norm of encoder weights from i to j.
    pub fn adjacency_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.n_vars;
        let h = self.hidden_dim;
        let mut w = vec![vec![0.0f32; n]; n];
        for j in 0..n {
            if j >= self.encoder_weights.len() {
                continue;
            }
            let enc_w = &self.encoder_weights[j];
            for i in 0..n {
                if i == j {
                    continue;
                }
                // L2 norm of weights from input i to all hidden units
                let norm: f32 = (0..h)
                    .map(|hh| {
                        let idx = i * h + hh;
                        if idx < enc_w.len() {
                            enc_w[idx] * enc_w[idx]
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>()
                    .sqrt();
                w[i][j] = norm;
            }
        }
        w
    }
}

/// Reconstruction loss for CASTLE: mean squared error over all variables and samples.
pub fn castle_reconstruction_loss(outputs: &[Vec<f32>], x: &[Vec<f32>]) -> f32 {
    let n = outputs.len().min(x.len());
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        let n_vars = outputs[i].len().min(x[i].len());
        for j in 0..n_vars {
            let diff = outputs[i][j] - x[i][j];
            total += diff * diff;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Acyclicity loss for CASTLE: h(W) = trace(exp(W∘W)) − n using the adjacency matrix.
pub fn castle_acyclicity_loss(w_flat: &[f32]) -> f32 {
    let n_sq = w_flat.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        return 0.0;
    }
    h_constraint(w_flat, n)
}

/// One step of CASTLE optimisation: update encoder/decoder weights via finite-difference gradients.
pub fn castle_step(model: &mut CausalAutoEncoder, x: &[Vec<f32>], lr: f32) -> f32 {
    let n_samples = x.len();
    if n_samples == 0 {
        return 0.0;
    }
    let n = model.n_vars;
    let h = model.hidden_dim;

    // Compute outputs and current loss
    let compute_loss = |m: &CausalAutoEncoder| -> f32 {
        let mut mse = 0.0f32;
        let mut cnt = 0usize;
        for row in x {
            for j in 0..n {
                let xj = if j < row.len() { row[j] } else { 0.0 };
                let pred = m.forward_j(row, j);
                mse += (xj - pred) * (xj - pred);
                cnt += 1;
            }
        }
        if cnt == 0 {
            0.0
        } else {
            mse / cnt as f32
        }
    };

    let base_loss = compute_loss(model);
    let eps = 1e-4f32;

    // Update encoder weights
    for j in 0..n {
        let enc_len = model.encoder_weights[j].len();
        for idx in 0..enc_len.min(n * h) {
            let orig = model.encoder_weights[j][idx];
            model.encoder_weights[j][idx] = orig + eps;
            let l_p = compute_loss(model);
            model.encoder_weights[j][idx] = orig;
            let grad = (l_p - base_loss) / eps;
            model.encoder_weights[j][idx] -= lr * grad;
        }
    }

    // Update decoder weights
    for j in 0..n {
        let dec_len = model.decoder_weights[j].len();
        for idx in 0..dec_len.min(h) {
            let orig = model.decoder_weights[j][idx];
            model.decoder_weights[j][idx] = orig + eps;
            let l_p = compute_loss(model);
            model.decoder_weights[j][idx] = orig;
            let grad = (l_p - base_loss) / eps;
            model.decoder_weights[j][idx] -= lr * grad;
        }
    }

    compute_loss(model)
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CausalBoosting
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CausalBoosting.
#[derive(Debug, Clone)]
pub struct CausalBoostConfig {
    /// Number of observed variables.
    pub n_vars: usize,
    /// Maximum stump depth (currently only depth-1 stumps are implemented).
    pub max_depth: usize,
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Shrinkage (learning rate).
    pub learning_rate: f32,
}

impl Default for CausalBoostConfig {
    fn default() -> Self {
        Self {
            n_vars: 4,
            max_depth: 1,
            n_estimators: 50,
            learning_rate: 0.1,
        }
    }
}

/// A depth-1 decision stump for causal boosting.
///
/// Predicts the effect of `cause` variable by splitting on `feature` at `threshold`.
#[derive(Debug, Clone)]
pub struct CausalDecisionStump {
    /// Feature index used for splitting.
    pub feature: usize,
    /// Split threshold.
    pub threshold: f32,
    /// Prediction for samples where `feature ≤ threshold`.
    pub left_val: f32,
    /// Prediction for samples where `feature > threshold`.
    pub right_val: f32,
    /// The cause variable this stump models.
    pub cause: usize,
}

impl CausalDecisionStump {
    /// Predict outcome for a single sample.
    pub fn predict_one(&self, x: &[f32]) -> f32 {
        let feat_val = if self.feature < x.len() {
            x[self.feature]
        } else {
            0.0
        };
        if feat_val <= self.threshold {
            self.left_val
        } else {
            self.right_val
        }
    }

    /// Predict outcomes for multiple samples.
    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        x.iter().map(|row| self.predict_one(row)).collect()
    }
}

/// Fit a single depth-1 decision stump minimising MSE on residuals.
pub fn fit_stump(x: &[Vec<f32>], residuals: &[f32], cause: usize) -> CausalDecisionStump {
    let n = x.len().min(residuals.len());
    if n == 0 {
        return CausalDecisionStump {
            feature: 0,
            threshold: 0.0,
            left_val: 0.0,
            right_val: 0.0,
            cause,
        };
    }
    let n_vars = x[0].len();
    let mut best_loss = f32::MAX;
    let mut best_stump = CausalDecisionStump {
        feature: 0,
        threshold: 0.0,
        left_val: mean_f32(&residuals[..n]),
        right_val: mean_f32(&residuals[..n]),
        cause,
    };

    for feat in 0..n_vars {
        // Collect feature values
        let mut vals: Vec<(f32, f32)> = (0..n)
            .map(|i| {
                let fv = if feat < x[i].len() { x[i][feat] } else { 0.0 };
                (fv, residuals[i])
            })
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Try each midpoint as threshold
        for split in 0..(n - 1) {
            let thresh = (vals[split].0 + vals[split + 1].0) / 2.0;
            let left: Vec<f32> = vals[..=split].iter().map(|(_, r)| *r).collect();
            let right: Vec<f32> = vals[(split + 1)..].iter().map(|(_, r)| *r).collect();
            let lv = mean_f32(&left);
            let rv = mean_f32(&right);
            let loss: f32 = left.iter().map(|&r| (r - lv) * (r - lv)).sum::<f32>()
                + right.iter().map(|&r| (r - rv) * (r - rv)).sum::<f32>();
            if loss < best_loss {
                best_loss = loss;
                best_stump = CausalDecisionStump {
                    feature: feat,
                    threshold: thresh,
                    left_val: lv,
                    right_val: rv,
                    cause,
                };
            }
        }
    }

    best_stump
}

/// Gradient boost toward target `y` with causal variable `cause` as context.
pub fn causal_boost(
    x: &[Vec<f32>],
    y: &[f32],
    cause: usize,
    config: &CausalBoostConfig,
) -> Vec<CausalDecisionStump> {
    let n = x.len().min(y.len());
    if n == 0 {
        return vec![];
    }

    let mut ensemble = Vec::with_capacity(config.n_estimators);
    let mut predictions = vec![mean_f32(&y[..n]); n];

    for _ in 0..config.n_estimators {
        // Compute pseudo-residuals
        let residuals: Vec<f32> = (0..n).map(|i| y[i] - predictions[i]).collect();
        let stump = fit_stump(x, &residuals, cause);

        // Update predictions
        for i in 0..n {
            predictions[i] += config.learning_rate * stump.predict_one(&x[i]);
        }

        ensemble.push(stump);
    }

    ensemble
}
