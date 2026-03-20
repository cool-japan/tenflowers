//! Sparse Autoencoder (SAE) for mechanistic interpretability.
//!
//! Implements the architecture from Anthropic (2023):
//! ```text
//! features = ReLU(W_enc * (x − b_pre) + b_enc)
//! x̂       = W_dec * features + b_pre
//! loss     = ||x − x̂||² + λ · ||features||₁
//! ```

use super::helpers::{matvec, random_matrix, relu, zeros_vec};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

/// Configuration for the mechanistic-interpretability sparse autoencoder.
#[derive(Debug, Clone)]
pub struct MiSaeConfig {
    /// Dimensionality of the activations to decompose (e.g. `d_model`).
    pub d_input: usize,
    /// Number of dictionary features (`> d_input` for over-completeness).
    pub d_hidden: usize,
    /// L1 penalty coefficient for feature sparsity.
    pub l1_coefficient: f64,
    /// Learning rate for the gradient-descent update.
    pub learning_rate: f64,
}

/// Sparse autoencoder following the Anthropic (2023) mechanistic-interpretability design.
pub struct MiSparseAutoencoder {
    /// SAE configuration.
    pub config: MiSaeConfig,
    /// Encoder weight matrix `[d_hidden][d_input]`.
    pub w_enc: Vec<Vec<f64>>,
    /// Encoder bias `[d_hidden]`.
    pub b_enc: Vec<f64>,
    /// Decoder weight matrix `[d_input][d_hidden]`.
    pub w_dec: Vec<Vec<f64>>,
    /// Pre-encoder bias `[d_input]`.
    pub b_pre: Vec<f64>,
}

impl MiSparseAutoencoder {
    /// Create a new SAE with Xavier-scaled random weights and zero biases.
    pub fn new(config: MiSaeConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);
        let s_enc = (1.0 / config.d_input as f64).sqrt();
        let s_dec = (1.0 / config.d_hidden as f64).sqrt();
        let w_enc = random_matrix(config.d_hidden, config.d_input, &mut rng, s_enc);
        let b_enc = zeros_vec(config.d_hidden);
        let w_dec = random_matrix(config.d_input, config.d_hidden, &mut rng, s_dec);
        let b_pre = zeros_vec(config.d_input);
        let mut sae = Self {
            config,
            w_enc,
            b_enc,
            w_dec,
            b_pre,
        };
        sae.normalize_decoder();
        sae
    }

    /// Encode `x`: `features = ReLU(W_enc * (x − b_pre) + b_enc)`.
    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        let x_centered: Vec<f64> = x
            .iter()
            .zip(self.b_pre.iter())
            .map(|(xi, bi)| xi - bi)
            .collect();
        matvec(&self.w_enc, &x_centered)
            .into_iter()
            .zip(self.b_enc.iter())
            .map(|(h, &be)| relu(h + be))
            .collect()
    }

    /// Decode `features`: `x̂ = W_dec * features + b_pre`.
    pub fn decode(&self, features: &[f64]) -> Vec<f64> {
        let out = matvec(&self.w_dec, features);
        out.iter()
            .zip(self.b_pre.iter())
            .map(|(o, b)| o + b)
            .collect()
    }

    /// Full forward pass.  Returns `(features, reconstruction)`.
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let features = self.encode(x);
        let reconstruction = self.decode(&features);
        (features, reconstruction)
    }

    /// `MSE(x, x̂) + l1_coefficient * ||features||₁`.
    pub fn loss(&self, x: &[f64]) -> f64 {
        let (features, reconstruction) = self.forward(x);
        let mse = x
            .iter()
            .zip(reconstruction.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / x.len() as f64;
        let l1: f64 = features.iter().map(|&f| f.abs()).sum();
        mse + self.config.l1_coefficient * l1
    }

    /// One gradient-descent step on a mini-batch of activations.
    pub fn train_step(&mut self, batch: &[Vec<f64>]) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        let batch_size = batch.len();
        let d_in = self.config.d_input;
        let d_hid = self.config.d_hidden;
        let lr = self.config.learning_rate;
        let lambda = self.config.l1_coefficient;

        let mut dw_enc = vec![vec![0.0_f64; d_in]; d_hid];
        let mut db_enc = vec![0.0_f64; d_hid];
        let mut dw_dec = vec![vec![0.0_f64; d_hid]; d_in];
        let mut db_pre = vec![0.0_f64; d_in];
        let mut total_loss = 0.0_f64;

        for x in batch {
            let x_c: Vec<f64> = x
                .iter()
                .zip(self.b_pre.iter())
                .map(|(xi, bi)| xi - bi)
                .collect();
            let pre_relu: Vec<f64> = matvec(&self.w_enc, &x_c)
                .into_iter()
                .zip(self.b_enc.iter())
                .map(|(h, &be)| h + be)
                .collect();
            let features: Vec<f64> = pre_relu.iter().map(|&h| relu(h)).collect();
            let recon: Vec<f64> = {
                let r = matvec(&self.w_dec, &features);
                r.iter()
                    .zip(self.b_pre.iter())
                    .map(|(o, b)| o + b)
                    .collect()
            };

            let mse = x
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / d_in as f64;
            let l1: f64 = features.iter().map(|&f| f.abs()).sum();
            total_loss += mse + lambda * l1;

            let d_recon: Vec<f64> = recon
                .iter()
                .zip(x.iter())
                .map(|(r, xi)| 2.0 * (r - xi) / d_in as f64)
                .collect();

            for (i, &dr) in d_recon.iter().enumerate() {
                db_pre[i] += dr;
            }

            let wdt_dr = {
                let mut out_vec = vec![0.0_f64; d_hid];
                for i in 0..d_in {
                    for j in 0..d_hid {
                        out_vec[j] += self.w_dec[i][j] * d_recon[i];
                    }
                }
                out_vec
            };
            let d_features: Vec<f64> = wdt_dr
                .iter()
                .zip(features.iter())
                .map(|(g, &f)| g + lambda * f.signum())
                .collect();

            for i in 0..d_in {
                for j in 0..d_hid {
                    dw_dec[i][j] += d_recon[i] * features[j];
                }
            }

            let d_pre: Vec<f64> = d_features
                .iter()
                .zip(pre_relu.iter())
                .map(|(g, &p)| if p > 0.0 { *g } else { 0.0 })
                .collect();

            for (i, &dp) in d_pre.iter().enumerate() {
                db_enc[i] += dp;
            }

            for i in 0..d_hid {
                for j in 0..d_in {
                    dw_enc[i][j] += d_pre[i] * x_c[j];
                }
            }

            for j in 0..d_in {
                for i in 0..d_hid {
                    db_pre[j] -= self.w_enc[i][j] * d_pre[i];
                }
            }
        }

        let n = batch_size as f64;
        for i in 0..d_hid {
            for j in 0..d_in {
                self.w_enc[i][j] -= lr * dw_enc[i][j] / n;
            }
            self.b_enc[i] -= lr * db_enc[i] / n;
        }
        for i in 0..d_in {
            for j in 0..d_hid {
                self.w_dec[i][j] -= lr * dw_dec[i][j] / n;
            }
            self.b_pre[i] -= lr * db_pre[i] / n;
        }

        self.normalize_decoder();

        total_loss / n
    }

    /// Train the SAE for `n_epochs` over `data`.  Returns loss history.
    pub fn fit(&mut self, data: &[Vec<f64>], n_epochs: usize) -> Vec<f64> {
        (0..n_epochs).map(|_| self.train_step(data)).collect()
    }

    /// Renormalize decoder columns to unit L2 norm.
    pub fn normalize_decoder(&mut self) {
        let d_hid = self.config.d_hidden;
        let d_in = self.config.d_input;
        for j in 0..d_hid {
            let col_norm = (0..d_in)
                .map(|i| self.w_dec[i][j].powi(2))
                .sum::<f64>()
                .sqrt();
            if col_norm > 1e-10 {
                for i in 0..d_in {
                    self.w_dec[i][j] /= col_norm;
                }
            }
        }
    }

    /// Indices of features whose activation is strictly positive for input `x`.
    pub fn active_features(&self, x: &[f64]) -> Vec<usize> {
        self.encode(x)
            .iter()
            .enumerate()
            .filter(|(_, &f)| f > 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Fraction of data points on which each feature fires (activation > 0).
    pub fn feature_frequency(&self, data: &[Vec<f64>]) -> Vec<f64> {
        if data.is_empty() {
            return zeros_vec(self.config.d_hidden);
        }
        let mut counts = zeros_vec(self.config.d_hidden);
        for x in data {
            for idx in self.active_features(x) {
                counts[idx] += 1.0;
            }
        }
        let n = data.len() as f64;
        counts.iter().map(|&c| c / n).collect()
    }
}

/// Public type alias so that `SparseAutoencoder` can be re-exported from MI.
pub type SparseAutoencoder = MiSparseAutoencoder;
/// Public type alias so that `SparseAutoencoderConfig` can be re-exported from MI.
pub type SparseAutoencoderConfig = MiSaeConfig;
