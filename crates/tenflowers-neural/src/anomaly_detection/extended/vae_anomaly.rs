//! Variational autoencoder anomaly detection.

use super::building_blocks::{box_muller, percentile, AdMlp};
use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

/// Configuration for [`VaeAnomaly`].
#[derive(Clone)]
pub struct VaeAnomalyConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Hidden sizes for encoder.
    pub encoder_hidden: Vec<usize>,
    /// Hidden sizes for decoder.
    pub decoder_hidden: Vec<usize>,
    /// KL divergence weight (β-VAE). Default: 1.0.
    pub beta: f64,
    /// Number of Monte Carlo samples used for anomaly scoring.
    pub n_samples: usize,
    /// Percentile for threshold computation.
    pub threshold_percentile: f64,
}

impl Default for VaeAnomalyConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            latent_dim: 4,
            encoder_hidden: vec![8],
            decoder_hidden: vec![8],
            beta: 1.0,
            n_samples: 10,
            threshold_percentile: 95.0,
        }
    }
}

/// Variational autoencoder anomaly detector.
///
/// Anomaly score = -ELBO (higher = more anomalous).
pub struct VaeAnomaly {
    /// Encoder μ network: `x → z_mu [latent_dim]`.
    pub encoder_mu: AdMlp,
    /// Encoder log-variance network: `x → z_log_var [latent_dim]`.
    pub encoder_lv: AdMlp,
    /// Decoder: `z → x_recon [input_dim]`.
    pub decoder: AdMlp,
    /// Anomaly threshold.
    pub threshold: f64,
    /// Configuration.
    pub config: VaeAnomalyConfig,
}

impl VaeAnomaly {
    /// Construct a new VAE anomaly detector.
    pub fn new(config: VaeAnomalyConfig) -> Self {
        let enc_sizes: Vec<usize> = std::iter::once(config.input_dim)
            .chain(config.encoder_hidden.iter().copied())
            .chain(std::iter::once(config.latent_dim))
            .collect();
        let dec_sizes: Vec<usize> = std::iter::once(config.latent_dim)
            .chain(config.decoder_hidden.iter().copied())
            .chain(std::iter::once(config.input_dim))
            .collect();
        Self {
            encoder_mu: AdMlp::new(&enc_sizes),
            encoder_lv: AdMlp::new(&enc_sizes),
            decoder: AdMlp::new(&dec_sizes),
            threshold: f64::MAX,
            config,
        }
    }

    /// Encode: returns (μ, log_var).
    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mu = self.encoder_mu.forward(x);
        let lv = self.encoder_lv.forward(x);
        (mu, lv)
    }

    /// Sample z ~ N(μ, σ²) using Box-Muller with u1 and u2 in (0,1).
    pub fn sample_z(&self, mu: &[f64], lv: &[f64], u1: f64, u2: f64) -> Vec<f64> {
        mu.iter()
            .zip(lv.iter())
            .enumerate()
            .map(|(i, (m, lv_i))| {
                let u2_shifted = ((u2 + i as f64 * 0.61803398874989) % 1.0).max(1e-30);
                let u1_safe = u1.max(1e-30);
                let eps = box_muller(u1_safe, u2_shifted);
                let sigma = (0.5 * lv_i).exp();
                m + sigma * eps
            })
            .collect()
    }

    /// Compute ELBO = E[log p(x|z)] - β * KL(q||p) using one MC sample.
    ///
    /// Reconstruction term: -MSE (Gaussian log-likelihood up to constant).
    /// KL term: 0.5 * Σ (μ² + σ² - log σ² - 1).
    pub fn elbo(&self, x: &[f64], u1: f64, u2: f64) -> f64 {
        let (mu, lv) = self.encode(x);
        let z = self.sample_z(&mu, &lv, u1, u2);
        let xhat = self.decoder.forward(&z);

        let n = x.len().max(1) as f64;
        let recon = -xhat
            .iter()
            .zip(x.iter())
            .map(|(r, o)| (r - o).powi(2))
            .sum::<f64>()
            / n;

        let kl: f64 = mu
            .iter()
            .zip(lv.iter())
            .map(|(m, lv_i)| 0.5 * (m.powi(2) + lv_i.exp() - lv_i - 1.0))
            .sum();

        recon - self.config.beta * kl
    }

    /// Train the VAE using finite-difference gradient estimates on the ELBO.
    pub fn fit(
        &mut self,
        x_train: &[Vec<f64>],
        n_epochs: usize,
        lr: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut history = Vec::with_capacity(n_epochs);
        let eps = 1e-4;

        for _ in 0..n_epochs {
            let mut epoch_loss = 0.0_f64;
            for sample in x_train.iter() {
                let u1: f64 = rng.random::<f64>().max(1e-30);
                let u2: f64 = rng.random::<f64>();
                let base_elbo = self.elbo(sample, u1, u2);
                epoch_loss += -base_elbo;

                // Gradient ascent on ELBO via FD for encoder_mu
                for layer_idx in 0..self.encoder_mu.layers.len() {
                    let out_dim = self.encoder_mu.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.encoder_mu.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.encoder_mu.layers[layer_idx].w[j][i];
                            self.encoder_mu.layers[layer_idx].w[j][i] = orig + eps;
                            let elbo_p = self.elbo(sample, u1, u2);
                            let g = (elbo_p - base_elbo) / eps;
                            self.encoder_mu.layers[layer_idx].w[j][i] = orig + lr * g;
                        }
                        let orig_b = self.encoder_mu.layers[layer_idx].b[j];
                        self.encoder_mu.layers[layer_idx].b[j] = orig_b + eps;
                        let elbo_p = self.elbo(sample, u1, u2);
                        let g = (elbo_p - base_elbo) / eps;
                        self.encoder_mu.layers[layer_idx].b[j] = orig_b + lr * g;
                    }
                }

                // Gradient ascent for encoder_lv
                for layer_idx in 0..self.encoder_lv.layers.len() {
                    let out_dim = self.encoder_lv.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.encoder_lv.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.encoder_lv.layers[layer_idx].w[j][i];
                            self.encoder_lv.layers[layer_idx].w[j][i] = orig + eps;
                            let elbo_p = self.elbo(sample, u1, u2);
                            let g = (elbo_p - base_elbo) / eps;
                            self.encoder_lv.layers[layer_idx].w[j][i] = orig + lr * g;
                        }
                        let orig_b = self.encoder_lv.layers[layer_idx].b[j];
                        self.encoder_lv.layers[layer_idx].b[j] = orig_b + eps;
                        let elbo_p = self.elbo(sample, u1, u2);
                        let g = (elbo_p - base_elbo) / eps;
                        self.encoder_lv.layers[layer_idx].b[j] = orig_b + lr * g;
                    }
                }

                // Gradient ascent for decoder
                let (mu, lv) = self.encode(sample);
                let z = self.sample_z(&mu, &lv, u1, u2);
                let xhat = self.decoder.forward(&z);
                let dec_loss: f64 = xhat
                    .iter()
                    .zip(sample.iter())
                    .map(|(r, o)| (r - o).powi(2))
                    .sum::<f64>()
                    / xhat.len().max(1) as f64;

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
                            let xhat_p = self.decoder.forward(&z);
                            let loss_p = xhat_p
                                .iter()
                                .zip(sample.iter())
                                .map(|(r, o)| (r - o).powi(2))
                                .sum::<f64>()
                                / xhat_p.len().max(1) as f64;
                            let g = (loss_p - dec_loss) / eps;
                            self.decoder.layers[layer_idx].w[j][i] = orig - lr * g;
                        }
                        let orig_b = self.decoder.layers[layer_idx].b[j];
                        self.decoder.layers[layer_idx].b[j] = orig_b + eps;
                        let xhat_p = self.decoder.forward(&z);
                        let loss_p = xhat_p
                            .iter()
                            .zip(sample.iter())
                            .map(|(r, o)| (r - o).powi(2))
                            .sum::<f64>()
                            / xhat_p.len().max(1) as f64;
                        let g = (loss_p - dec_loss) / eps;
                        self.decoder.layers[layer_idx].b[j] = orig_b - lr * g;
                    }
                }
            }
            history.push(epoch_loss / x_train.len().max(1) as f64);
        }

        history
    }

    /// Anomaly score = -mean ELBO over `n_samples` MC samples.
    pub fn anomaly_score(&self, x: &[f64], rng: &mut StdRng) -> f64 {
        let n = self.config.n_samples.max(1);
        let sum: f64 = (0..n)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-30);
                let u2: f64 = rng.random::<f64>();
                -self.elbo(x, u1, u2)
            })
            .sum();
        sum / n as f64
    }

    /// Compute anomaly scores on normal data and set threshold at configured percentile.
    pub fn set_threshold(&mut self, x_normal: &[Vec<f64>], rng: &mut StdRng) {
        let scores: Vec<f64> = x_normal
            .iter()
            .map(|x| self.anomaly_score(x, rng))
            .collect();
        if scores.is_empty() {
            self.threshold = 0.0;
        } else {
            self.threshold = percentile(&scores, self.config.threshold_percentile);
        }
    }

    /// Predict whether `x` is an anomaly.
    pub fn predict(&self, x: &[f64], rng: &mut StdRng) -> bool {
        self.anomaly_score(x, rng) > self.threshold
    }
}
