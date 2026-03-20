//! Deep SVDD anomaly detection (f64 variant).

use super::building_blocks::{percentile, AdMlp};
use scirs2_core::random::rngs::StdRng;

/// Configuration for [`AdDeepSvdd`].
#[derive(Clone)]
pub struct AdDeepSvddConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Output (hypersphere embedding) dimension.
    pub output_dim: usize,
    /// Hidden layer dimensions.
    pub hidden_dims: Vec<usize>,
    /// Fraction of expected outliers (0 < nu < 1). Default: 0.1.
    pub nu: f64,
    /// Training epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Epochs of AE pretraining before SVDD loss.
    pub warm_up_epochs: usize,
}

impl Default for AdDeepSvddConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            output_dim: 4,
            hidden_dims: vec![16],
            nu: 0.1,
            n_epochs: 5,
            lr: 1e-3,
            warm_up_epochs: 2,
        }
    }
}

/// Deep SVDD anomaly detector (f64 variant, prefixed `Ad` to avoid conflict
/// with the existing f32-based `DeepSVDD`).
pub struct AdDeepSvdd {
    /// Encoder network mapping inputs to hypersphere space.
    pub network: AdMlp,
    /// Hypersphere center (fixed after initialisation).
    pub center: Vec<f64>,
    /// Hypersphere radius (soft-boundary).
    pub radius: f64,
    /// Configuration.
    pub config: AdDeepSvddConfig,
}

impl AdDeepSvdd {
    /// Construct a new Deep SVDD detector.
    pub fn new(config: AdDeepSvddConfig) -> Self {
        let sizes: Vec<usize> = std::iter::once(config.input_dim)
            .chain(config.hidden_dims.iter().copied())
            .chain(std::iter::once(config.output_dim))
            .collect();
        let network = AdMlp::new(&sizes);
        let center = vec![0.0_f64; config.output_dim];
        Self {
            network,
            center,
            radius: 1.0,
            config,
        }
    }

    /// Initialise the center as the mean of network outputs over `x_train`.
    pub fn initialize_center(&mut self, x_train: &[Vec<f64>]) {
        if x_train.is_empty() {
            return;
        }
        let out_dim = self.config.output_dim;
        let mut sum = vec![0.0_f64; out_dim];
        let mut count = 0_usize;

        for x in x_train.iter() {
            let phi = self.network.forward(x);
            for (s, p) in sum.iter_mut().zip(phi.iter()) {
                *s += p;
            }
            count += 1;
        }

        let n = count as f64;
        self.center = sum.iter().map(|s| s / n).collect();

        let norm: f64 = self.center.iter().map(|c| c.powi(2)).sum::<f64>().sqrt();
        if norm < 1e-8 {
            for (i, c) in self.center.iter_mut().enumerate() {
                *c = 0.01 * ((i + 1) as f64);
            }
        }
    }

    /// Compute soft-boundary SVDD loss over a batch.
    ///
    /// Loss = R² + (1/(ν·n)) · Σ max(0, ||φ(xᵢ) - c||² - R²).
    pub fn svdd_loss(&self, x_batch: &[Vec<f64>]) -> f64 {
        if x_batch.is_empty() {
            return 0.0;
        }
        let r2 = self.radius.powi(2);
        let n = x_batch.len() as f64;
        let penalty: f64 = x_batch
            .iter()
            .map(|x| {
                let phi = self.network.forward(x);
                let dist2: f64 = phi
                    .iter()
                    .zip(self.center.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum();
                (dist2 - r2).max(0.0)
            })
            .sum();
        r2 + penalty / (self.config.nu.max(1e-12) * n)
    }

    /// Train the network. Pretrain for `warm_up_epochs`, then train with SVDD loss.
    pub fn fit(&mut self, x_train: &[Vec<f64>], rng: &mut StdRng) -> Vec<f64> {
        let _ = rng;
        let mut history = Vec::new();
        let eps = 1e-5;
        let lr = self.config.lr;

        // Phase 1: pretraining
        for _ in 0..self.config.warm_up_epochs {
            let mut epoch_loss = 0.0_f64;
            for sample in x_train.iter() {
                let phi = self.network.forward(sample);
                let mean: f64 = phi.iter().sum::<f64>() / phi.len().max(1) as f64;
                let target: Vec<f64> = phi.iter().map(|_| 0.0_f64).collect();
                let loss: f64 = phi
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / phi.len().max(1) as f64;
                epoch_loss += loss;

                for layer_idx in 0..self.network.layers.len() {
                    let out_dim = self.network.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.network.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.network.layers[layer_idx].w[j][i];
                            self.network.layers[layer_idx].w[j][i] = orig + eps;
                            let phi_p = self.network.forward(sample);
                            let loss_p = phi_p
                                .iter()
                                .zip(target.iter())
                                .map(|(p, t)| (p - t).powi(2))
                                .sum::<f64>()
                                / phi_p.len().max(1) as f64;
                            let g = (loss_p - loss) / eps;
                            self.network.layers[layer_idx].w[j][i] = orig - lr * g;
                        }
                        let orig_b = self.network.layers[layer_idx].b[j];
                        self.network.layers[layer_idx].b[j] = orig_b + eps;
                        let phi_p = self.network.forward(sample);
                        let loss_p = phi_p
                            .iter()
                            .zip(target.iter())
                            .map(|(p, t)| (p - t).powi(2))
                            .sum::<f64>()
                            / phi_p.len().max(1) as f64;
                        let g = (loss_p - loss) / eps;
                        self.network.layers[layer_idx].b[j] = orig_b - lr * g;
                    }
                }
                let _ = mean;
            }
            history.push(epoch_loss / x_train.len().max(1) as f64);
        }

        self.initialize_center(x_train);

        // Phase 2: SVDD loss
        let svdd_epochs = self
            .config
            .n_epochs
            .saturating_sub(self.config.warm_up_epochs);
        for _ in 0..svdd_epochs {
            let base_loss = self.svdd_loss(x_train);
            history.push(base_loss);

            for sample in x_train.iter() {
                let phi = self.network.forward(sample);
                let dist2: f64 = phi
                    .iter()
                    .zip(self.center.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum();
                let r2 = self.radius.powi(2);
                let violation = (dist2 - r2).max(0.0);
                let coeff = 1.0 / (self.config.nu.max(1e-12) * x_train.len().max(1) as f64);

                for layer_idx in 0..self.network.layers.len() {
                    let out_dim = self.network.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.network.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.network.layers[layer_idx].w[j][i];
                            self.network.layers[layer_idx].w[j][i] = orig + eps;
                            let phi_p = self.network.forward(sample);
                            let dist2_p: f64 = phi_p
                                .iter()
                                .zip(self.center.iter())
                                .map(|(p, c)| (p - c).powi(2))
                                .sum();
                            let v_p = (dist2_p - r2).max(0.0);
                            let g = coeff * (v_p - violation) / eps;
                            self.network.layers[layer_idx].w[j][i] = orig - lr * g;
                        }
                        let orig_b = self.network.layers[layer_idx].b[j];
                        self.network.layers[layer_idx].b[j] = orig_b + eps;
                        let phi_p = self.network.forward(sample);
                        let dist2_p: f64 = phi_p
                            .iter()
                            .zip(self.center.iter())
                            .map(|(p, c)| (p - c).powi(2))
                            .sum();
                        let v_p = (dist2_p - r2).max(0.0);
                        let g = coeff * (v_p - violation) / eps;
                        self.network.layers[layer_idx].b[j] = orig_b - lr * g;
                    }
                }
            }

            self.set_radius_by_percentile(x_train, (1.0 - self.config.nu) * 100.0);
        }

        history
    }

    /// Anomaly score = ||φ(x) - c||² (squared distance from center).
    pub fn score(&self, x: &[f64]) -> f64 {
        let phi = self.network.forward(x);
        phi.iter()
            .zip(self.center.iter())
            .map(|(p, c)| (p - c).powi(2))
            .sum()
    }

    /// Predict whether `x` is an anomaly (score > R²).
    pub fn predict(&self, x: &[f64]) -> bool {
        self.score(x) > self.radius.powi(2)
    }

    /// Set radius at the given percentile of distances to center over normal data.
    pub fn set_radius_by_percentile(&mut self, x_normal: &[Vec<f64>], percentile_val: f64) {
        let dists: Vec<f64> = x_normal.iter().map(|x| self.score(x)).collect();
        if dists.is_empty() {
            return;
        }
        let r2 = percentile(&dists, percentile_val);
        self.radius = r2.sqrt().max(1e-8);
    }
}
