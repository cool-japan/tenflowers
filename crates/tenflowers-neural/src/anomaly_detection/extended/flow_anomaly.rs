//! Normalizing flow (RealNVP coupling) anomaly detection.

use super::building_blocks::{percentile, AdMlp};
use std::f64::consts::PI;

/// Configuration for [`FlowAnomaly`].
#[derive(Clone)]
pub struct FlowAnomalyConfig {
    /// Input dimension (must be ≥ 2).
    pub input_dim: usize,
    /// Number of RealNVP coupling layers.
    pub n_coupling_layers: usize,
    /// Hidden dimension in scale/translate networks.
    pub hidden_dim: usize,
    /// Training epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Percentile for threshold.
    pub threshold_percentile: f64,
}

impl Default for FlowAnomalyConfig {
    fn default() -> Self {
        Self {
            input_dim: 4,
            n_coupling_layers: 2,
            hidden_dim: 8,
            n_epochs: 5,
            lr: 1e-3,
            threshold_percentile: 95.0,
        }
    }
}

/// A RealNVP-style affine coupling layer (f64).
pub struct CouplingLayer {
    /// Scale network: `x_A → s [active_half_size]`.
    pub scale_net: AdMlp,
    /// Translate network: `x_A → t [active_half_size]`.
    pub translate_net: AdMlp,
    /// Mask: `true` = identity (pass through unchanged).
    pub mask: Vec<bool>,
}

impl CouplingLayer {
    /// Construct a new coupling layer.
    ///
    /// When `invert_mask` is false, the first half is identity; when true, the second half is identity.
    pub fn new(input_dim: usize, hidden_dim: usize, invert_mask: bool) -> Self {
        let half = input_dim / 2;
        let other_half = input_dim - half;

        let (cond_dim, out_dim) = if invert_mask {
            (other_half, half)
        } else {
            (half, other_half)
        };

        let sizes = vec![cond_dim, hidden_dim, out_dim];
        let scale_net = AdMlp::new(&sizes);
        let translate_net = AdMlp::new(&sizes);

        let mask: Vec<bool> = (0..input_dim)
            .map(|i| {
                let in_first_half = i < half;
                if invert_mask {
                    !in_first_half
                } else {
                    in_first_half
                }
            })
            .collect();

        Self {
            scale_net,
            translate_net,
            mask,
        }
    }

    /// Forward pass: `(y, log_det_jacobian)`.
    ///
    /// - Masked (identity) dimensions: `y_A = x_A`
    /// - Active dimensions: `y_B = x_B * exp(s(x_A)) + t(x_A)`
    /// - `log_det = Σ s(x_A)_i`  (over active dimensions)
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, f64) {
        let d = x.len();
        let cond: Vec<f64> = x
            .iter()
            .zip(self.mask.iter())
            .filter_map(|(xi, &m)| if m { Some(*xi) } else { None })
            .collect();

        let s_vals = self.scale_net.forward(&cond);
        let t_vals = self.translate_net.forward(&cond);

        let mut y = vec![0.0_f64; d];
        let mut log_det = 0.0_f64;
        let mut active_idx = 0_usize;

        for i in 0..d {
            if self.mask[i] {
                y[i] = x[i];
            } else {
                let s = s_vals[active_idx].clamp(-5.0, 5.0);
                let t = t_vals[active_idx];
                y[i] = x[i] * s.exp() + t;
                log_det += s;
                active_idx += 1;
            }
        }

        (y, log_det)
    }

    /// Inverse pass: `x = inverse(y)`.
    ///
    /// - Masked: `x_A = y_A`
    /// - Active: `x_B = (y_B - t(y_A)) * exp(-s(y_A))`
    pub fn inverse(&self, y: &[f64]) -> Vec<f64> {
        let d = y.len();
        let cond: Vec<f64> = y
            .iter()
            .zip(self.mask.iter())
            .filter_map(|(yi, &m)| if m { Some(*yi) } else { None })
            .collect();

        let s_vals = self.scale_net.forward(&cond);
        let t_vals = self.translate_net.forward(&cond);

        let mut x = vec![0.0_f64; d];
        let mut active_idx = 0_usize;

        for i in 0..d {
            if self.mask[i] {
                x[i] = y[i];
            } else {
                let s = s_vals[active_idx].clamp(-5.0, 5.0);
                let t = t_vals[active_idx];
                x[i] = (y[i] - t) * (-s).exp();
                active_idx += 1;
            }
        }

        x
    }
}

/// Normalizing flow anomaly detector.
///
/// Anomaly score = -log_prob(x) (negative log-likelihood under the flow).
pub struct FlowAnomaly {
    /// Coupling layers.
    pub layers: Vec<CouplingLayer>,
    /// Anomaly threshold.
    pub threshold: f64,
    /// Configuration.
    pub config: FlowAnomalyConfig,
}

impl FlowAnomaly {
    /// Construct a new flow anomaly detector.
    pub fn new(config: FlowAnomalyConfig) -> Self {
        let layers: Vec<CouplingLayer> = (0..config.n_coupling_layers)
            .map(|i| CouplingLayer::new(config.input_dim, config.hidden_dim, i % 2 == 1))
            .collect();
        Self {
            layers,
            threshold: f64::MAX,
            config,
        }
    }

    /// Compute log-probability under the flow model.
    ///
    /// `log p(x) = log p₀(z) + Σ log_det_j`
    /// where `p₀` is N(0,I).
    pub fn log_prob(&self, x: &[f64]) -> f64 {
        let d = x.len() as f64;
        let mut z = x.to_vec();
        let mut log_det_sum = 0.0_f64;

        for layer in self.layers.iter() {
            let (z_new, ld) = layer.forward(&z);
            z = z_new;
            log_det_sum += ld;
        }

        let log_p0: f64 = -0.5 * (d * (2.0 * PI).ln() + z.iter().map(|zi| zi.powi(2)).sum::<f64>());

        log_p0 + log_det_sum
    }

    /// Train the flow by maximising log-likelihood via FD gradients.
    pub fn fit(&mut self, x_train: &[Vec<f64>], n_epochs: usize, lr: f64) -> Vec<f64> {
        let mut history = Vec::with_capacity(n_epochs);
        let eps = 1e-4;

        for _ in 0..n_epochs {
            let mut epoch_loss = 0.0_f64;
            for sample in x_train.iter() {
                let base_ll = self.log_prob(sample);
                epoch_loss -= base_ll;

                for layer_idx in 0..self.layers.len() {
                    // Scale net
                    for l_idx in 0..self.layers[layer_idx].scale_net.layers.len() {
                        let out_dim = self.layers[layer_idx].scale_net.layers[l_idx].w.len();
                        let in_dim = if out_dim > 0 {
                            self.layers[layer_idx].scale_net.layers[l_idx].w[0].len()
                        } else {
                            0
                        };
                        for j in 0..out_dim {
                            for i in 0..in_dim {
                                let orig = self.layers[layer_idx].scale_net.layers[l_idx].w[j][i];
                                self.layers[layer_idx].scale_net.layers[l_idx].w[j][i] = orig + eps;
                                let ll_p = self.log_prob(sample);
                                let g = (ll_p - base_ll) / eps;
                                self.layers[layer_idx].scale_net.layers[l_idx].w[j][i] =
                                    orig + lr * g;
                            }
                            let orig_b = self.layers[layer_idx].scale_net.layers[l_idx].b[j];
                            self.layers[layer_idx].scale_net.layers[l_idx].b[j] = orig_b + eps;
                            let ll_p = self.log_prob(sample);
                            let g = (ll_p - base_ll) / eps;
                            self.layers[layer_idx].scale_net.layers[l_idx].b[j] = orig_b + lr * g;
                        }
                    }

                    // Translate net
                    for l_idx in 0..self.layers[layer_idx].translate_net.layers.len() {
                        let out_dim = self.layers[layer_idx].translate_net.layers[l_idx].w.len();
                        let in_dim = if out_dim > 0 {
                            self.layers[layer_idx].translate_net.layers[l_idx].w[0].len()
                        } else {
                            0
                        };
                        for j in 0..out_dim {
                            for i in 0..in_dim {
                                let orig =
                                    self.layers[layer_idx].translate_net.layers[l_idx].w[j][i];
                                self.layers[layer_idx].translate_net.layers[l_idx].w[j][i] =
                                    orig + eps;
                                let ll_p = self.log_prob(sample);
                                let g = (ll_p - base_ll) / eps;
                                self.layers[layer_idx].translate_net.layers[l_idx].w[j][i] =
                                    orig + lr * g;
                            }
                            let orig_b = self.layers[layer_idx].translate_net.layers[l_idx].b[j];
                            self.layers[layer_idx].translate_net.layers[l_idx].b[j] = orig_b + eps;
                            let ll_p = self.log_prob(sample);
                            let g = (ll_p - base_ll) / eps;
                            self.layers[layer_idx].translate_net.layers[l_idx].b[j] =
                                orig_b + lr * g;
                        }
                    }
                }
            }
            history.push(epoch_loss / x_train.len().max(1) as f64);
        }

        history
    }

    /// Anomaly score = -log_prob(x).
    pub fn anomaly_score(&self, x: &[f64]) -> f64 {
        -self.log_prob(x)
    }

    /// Set threshold at configured percentile of anomaly scores on normal data.
    pub fn set_threshold(&mut self, x_normal: &[Vec<f64>]) {
        let scores: Vec<f64> = x_normal.iter().map(|x| self.anomaly_score(x)).collect();
        if scores.is_empty() {
            self.threshold = 0.0;
        } else {
            self.threshold = percentile(&scores, self.config.threshold_percentile);
        }
    }

    /// Predict whether `x` is an anomaly (anomaly_score > threshold).
    pub fn predict(&self, x: &[f64]) -> bool {
        self.anomaly_score(x) > self.threshold
    }
}
