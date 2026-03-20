//! Autoencoder-based anomaly detection.

use super::building_blocks::{percentile, AdMlp};

/// Configuration for [`AeAnomaly`].
#[derive(Clone)]
pub struct AeAnomalyConfig {
    /// Input (and reconstruction) dimension.
    pub input_dim: usize,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Hidden layer sizes for the encoder (between input and latent).
    pub encoder_hidden: Vec<usize>,
    /// Hidden layer sizes for the decoder (between latent and output).
    pub decoder_hidden: Vec<usize>,
    /// Percentile of reconstruction errors on normal data used to set the threshold.
    /// Default: 95.0.
    pub threshold_percentile: f64,
}

impl Default for AeAnomalyConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            latent_dim: 4,
            encoder_hidden: vec![8],
            decoder_hidden: vec![8],
            threshold_percentile: 95.0,
        }
    }
}

/// Autoencoder-based anomaly detector.
///
/// Anomaly score = MSE reconstruction error. Points whose score exceeds the
/// threshold (set via `set_threshold`) are flagged as anomalies.
pub struct AeAnomaly {
    /// Encoder network: `input_dim → ... → latent_dim`.
    pub encoder: AdMlp,
    /// Decoder network: `latent_dim → ... → input_dim`.
    pub decoder: AdMlp,
    /// Anomaly threshold (set after calling `set_threshold` or `fit`).
    pub threshold: f64,
    /// Configuration.
    pub config: AeAnomalyConfig,
}

impl AeAnomaly {
    /// Construct a new autoencoder with the given configuration.
    pub fn new(config: AeAnomalyConfig) -> Self {
        let enc_sizes: Vec<usize> = std::iter::once(config.input_dim)
            .chain(config.encoder_hidden.iter().copied())
            .chain(std::iter::once(config.latent_dim))
            .collect();
        let dec_sizes: Vec<usize> = std::iter::once(config.latent_dim)
            .chain(config.decoder_hidden.iter().copied())
            .chain(std::iter::once(config.input_dim))
            .collect();
        Self {
            encoder: AdMlp::new(&enc_sizes),
            decoder: AdMlp::new(&dec_sizes),
            threshold: f64::MAX,
            config,
        }
    }

    /// Encode input to latent vector.
    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.encoder.forward(x)
    }

    /// Decode latent vector to reconstruction.
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        self.decoder.forward(z)
    }

    /// Encode then decode: full reconstruction pass.
    pub fn reconstruct(&self, x: &[f64]) -> Vec<f64> {
        let z = self.encode(x);
        self.decode(&z)
    }

    /// MSE reconstruction error.
    pub fn reconstruction_error(&self, x: &[f64]) -> f64 {
        let xhat = self.reconstruct(x);
        let n = xhat.len().max(1);
        xhat.iter()
            .zip(x.iter())
            .map(|(r, o)| (r - o).powi(2))
            .sum::<f64>()
            / n as f64
    }

    /// Train the autoencoder for `n_epochs` epochs with learning rate `lr`.
    ///
    /// Returns the per-epoch average loss history.
    ///
    /// Gradients are clipped to [-1, 1] to avoid numerical instability.
    pub fn fit(&mut self, x_train: &[Vec<f64>], n_epochs: usize, lr: f64) -> Vec<f64> {
        let mut history = Vec::with_capacity(n_epochs);
        let eps = 1e-4;
        let clip = 1.0_f64;

        for _ in 0..n_epochs {
            let mut epoch_loss = 0.0_f64;
            for sample in x_train.iter() {
                let target = sample.as_slice();

                let base_loss = {
                    let xhat = self.reconstruct(sample);
                    let n = xhat.len().max(1) as f64;
                    xhat.iter()
                        .zip(target.iter())
                        .map(|(r, o)| (r - o).powi(2))
                        .sum::<f64>()
                        / n
                };
                epoch_loss += if base_loss.is_finite() {
                    base_loss
                } else {
                    0.0
                };

                let recon_loss = |ae: &AeAnomaly| -> f64 {
                    let xhat = ae.reconstruct(sample);
                    let n = xhat.len().max(1) as f64;
                    xhat.iter()
                        .zip(target.iter())
                        .map(|(r, o)| (r - o).powi(2))
                        .sum::<f64>()
                        / n
                };

                // Update encoder weights via FD gradient descent.
                for layer_idx in 0..self.encoder.layers.len() {
                    let out_dim = self.encoder.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.encoder.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.encoder.layers[layer_idx].w[j][i];
                            self.encoder.layers[layer_idx].w[j][i] = orig + eps;
                            let loss_p = recon_loss(self);
                            let g = ((loss_p - base_loss) / eps).clamp(-clip, clip);
                            self.encoder.layers[layer_idx].w[j][i] =
                                if g.is_finite() { orig - lr * g } else { orig };
                        }
                        let orig_b = self.encoder.layers[layer_idx].b[j];
                        self.encoder.layers[layer_idx].b[j] = orig_b + eps;
                        let loss_p = recon_loss(self);
                        let g = ((loss_p - base_loss) / eps).clamp(-clip, clip);
                        self.encoder.layers[layer_idx].b[j] = if g.is_finite() {
                            orig_b - lr * g
                        } else {
                            orig_b
                        };
                    }
                }

                // Update decoder weights via FD gradient descent.
                let z_current = self.encode(sample);
                let dec_loss_base = {
                    let xhat = self.decode(&z_current);
                    let n = xhat.len().max(1) as f64;
                    xhat.iter()
                        .zip(target.iter())
                        .map(|(r, o)| (r - o).powi(2))
                        .sum::<f64>()
                        / n
                };

                for layer_idx in 0..self.decoder.layers.len() {
                    let out_dim = self.decoder.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.decoder.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.decoder.layers[layer_idx].w[j][i];
                            self.decoder.layers[layer_idx].w[j][i] = orig + eps;
                            let xhat_p = self.decode(&z_current);
                            let loss_p = xhat_p
                                .iter()
                                .zip(target.iter())
                                .map(|(r, o)| (r - o).powi(2))
                                .sum::<f64>()
                                / xhat_p.len().max(1) as f64;
                            let g = ((loss_p - dec_loss_base) / eps).clamp(-clip, clip);
                            self.decoder.layers[layer_idx].w[j][i] =
                                if g.is_finite() { orig - lr * g } else { orig };
                        }
                        let orig_b = self.decoder.layers[layer_idx].b[j];
                        self.decoder.layers[layer_idx].b[j] = orig_b + eps;
                        let xhat_p = self.decode(&z_current);
                        let loss_p = xhat_p
                            .iter()
                            .zip(target.iter())
                            .map(|(r, o)| (r - o).powi(2))
                            .sum::<f64>()
                            / xhat_p.len().max(1) as f64;
                        let g = ((loss_p - dec_loss_base) / eps).clamp(-clip, clip);
                        self.decoder.layers[layer_idx].b[j] = if g.is_finite() {
                            orig_b - lr * g
                        } else {
                            orig_b
                        };
                    }
                }
            }
            let n = x_train.len().max(1) as f64;
            history.push(epoch_loss / n);
        }

        history
    }

    /// Compute reconstruction errors on `x_normal` and set the threshold at
    /// `config.threshold_percentile`.
    pub fn set_threshold(&mut self, x_normal: &[Vec<f64>]) {
        let errors: Vec<f64> = x_normal
            .iter()
            .map(|x| self.reconstruction_error(x))
            .collect();
        if errors.is_empty() {
            self.threshold = 0.0;
        } else {
            self.threshold = percentile(&errors, self.config.threshold_percentile);
        }
    }

    /// Anomaly score = reconstruction error.
    pub fn score(&self, x: &[f64]) -> f64 {
        self.reconstruction_error(x)
    }

    /// Predict whether `x` is an anomaly (`score > threshold`).
    pub fn predict(&self, x: &[f64]) -> bool {
        self.score(x) > self.threshold
    }

    /// Predict for a batch of samples.
    pub fn batch_predict(&self, x: &[Vec<f64>]) -> Vec<bool> {
        x.iter().map(|xi| self.predict(xi)).collect()
    }
}
