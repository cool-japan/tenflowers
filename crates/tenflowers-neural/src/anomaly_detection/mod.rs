//! Deep Anomaly Detection Methods
//!
//! This module provides a comprehensive suite of deep learning and statistical
//! anomaly detection algorithms, including:
//!
//! - [`AnomalyVAE`]: VAE-based anomaly detection via reconstruction error + KL divergence
//! - [`DeepSVDD`]: Deep Support Vector Data Description using a hypersphere in latent space
//! - [`PatchTSAD`]: Patch-based time series anomaly detection
//! - [`AnomalyTransformerModel`]: Association discrepancy-based anomaly detection
//! - [`MemoryAugmentedAE`]: Memory-augmented autoencoder for anomaly detection
//! - [`RobustRCF`]: Robust Random Cut Forest streaming anomaly detector
//! - [`SpectralResidual`]: FFT-based saliency map anomaly detection (SRAD)
//! - [`GaussianMixtureAnomaly`]: GMM-based density estimation for anomaly scoring
//! - [`AnomalyThresholder`]: Threshold selection strategies (POT, EVT, adaptive)
//! - [`AnomalyEvaluationMetrics`]: ROC-AUC, PR-AUC, F1, point-adjusted F1

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from anomaly detection operations.
#[derive(Debug, Clone)]
pub enum AnomalyError {
    /// Input dimensions do not match configuration.
    DimensionMismatch { expected: usize, found: usize },
    /// Numerical failure (non-finite, singular matrix, etc.).
    NumericalFailure(String),
    /// Empty input provided.
    EmptyInput,
    /// Configuration error (invalid parameter).
    InvalidConfig(String),
}

impl std::fmt::Display for AnomalyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            AnomalyError::NumericalFailure(msg) => write!(f, "numerical failure: {msg}"),
            AnomalyError::EmptyInput => write!(f, "empty input"),
            AnomalyError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for AnomalyError {}

type AnomalyResult<T> = Result<T, AnomalyError>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller normal sample.
#[inline]
fn sample_normal_f32(rng: &mut StdRng) -> f32 {
    let u1: f32 = (rng.random::<f32>()).max(1e-30);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
#[inline]
fn l2_norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Softmax over a slice, returns a new Vec.
fn softmax_vec(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let max_v = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max_v).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s == 0.0 {
        return vec![1.0 / v.len() as f32; v.len()];
    }
    exps.iter().map(|e| e / s).collect()
}

/// Numerically stable log-sum-exp.
#[inline]
fn log_sum_exp_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NEG_INFINITY;
    }
    let max_v = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let s: f32 = v.iter().map(|x| (x - max_v).exp()).sum();
    max_v + s.ln()
}

/// Naive DFT: returns (real, imag) complex pairs.
fn dft(x: &[f32]) -> Vec<(f32, f32)> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f32;
    (0..n)
        .map(|k| {
            let kf = k as f32;
            let (mut re, mut im) = (0.0_f32, 0.0_f32);
            for (i, &xi) in x.iter().enumerate() {
                let angle = -2.0 * PI * kf * i as f32 / nf;
                re += xi * angle.cos();
                im += xi * angle.sin();
            }
            (re, im)
        })
        .collect()
}

/// Inverse DFT from (real, imag) pairs.
fn idft(spectrum: &[(f32, f32)]) -> Vec<f32> {
    let n = spectrum.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f32;
    (0..n)
        .map(|i| {
            let sum: f32 = spectrum
                .iter()
                .enumerate()
                .map(|(k, &(re, im))| {
                    let angle = 2.0 * PI * k as f32 * i as f32 / nf;
                    re * angle.cos() - im * angle.sin()
                })
                .sum();
            sum / nf
        })
        .collect()
}

/// Simple matrix-vector multiply: A (rows x cols), v (cols) -> result (rows).
fn matvec(a: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    a.iter().map(|row| dot(row, v)).collect()
}

/// Cosine similarity between two vectors (clamped to [-1, 1]).
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. AnomalyVAE
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`AnomalyVAE`].
#[derive(Debug, Clone)]
pub struct AnomalyVaeConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Latent dimensionality.
    pub latent_dim: usize,
    /// Hidden layer size for encoder/decoder.
    pub hidden_dim: usize,
    /// KL divergence weight (beta-VAE).
    pub beta: f32,
    /// Random seed for weight initialisation.
    pub seed: u64,
}

impl Default for AnomalyVaeConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            latent_dim: 8,
            hidden_dim: 64,
            beta: 1.0,
            seed: 42,
        }
    }
}

/// VAE-based anomaly detector.
///
/// Anomaly score = reconstruction MSE + β * KL(q(z|x) ‖ p(z)).
///
/// Architecture:
/// - Encoder: input → hidden (ReLU) → mu, log_var (linear)
/// - Decoder: latent → hidden (ReLU) → reconstruction (linear)
pub struct AnomalyVAE {
    config: AnomalyVaeConfig,
    /// Encoder: weight matrix [hidden_dim × input_dim].
    enc_w1: Vec<Vec<f32>>,
    enc_b1: Vec<f32>,
    /// Encoder mu head [latent_dim × hidden_dim].
    enc_mu_w: Vec<Vec<f32>>,
    enc_mu_b: Vec<f32>,
    /// Encoder log_var head [latent_dim × hidden_dim].
    enc_lv_w: Vec<Vec<f32>>,
    enc_lv_b: Vec<f32>,
    /// Decoder [hidden_dim × latent_dim].
    dec_w1: Vec<Vec<f32>>,
    dec_b1: Vec<f32>,
    /// Decoder output [input_dim × hidden_dim].
    dec_w2: Vec<Vec<f32>>,
    dec_b2: Vec<f32>,
}

impl AnomalyVAE {
    /// Constructs a new `AnomalyVAE` with Xavier-initialised weights.
    pub fn new(config: AnomalyVaeConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let xavier = |rows: usize, cols: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
            let scale = (2.0_f32 / (rows + cols) as f32).sqrt();
            (0..rows)
                .map(|_| (0..cols).map(|_| sample_normal_f32(rng) * scale).collect())
                .collect()
        };
        let zeros = |n: usize| vec![0.0_f32; n];

        let enc_w1 = xavier(config.hidden_dim, config.input_dim, &mut rng);
        let enc_b1 = zeros(config.hidden_dim);
        let enc_mu_w = xavier(config.latent_dim, config.hidden_dim, &mut rng);
        let enc_mu_b = zeros(config.latent_dim);
        let enc_lv_w = xavier(config.latent_dim, config.hidden_dim, &mut rng);
        let enc_lv_b = zeros(config.latent_dim);
        let dec_w1 = xavier(config.hidden_dim, config.latent_dim, &mut rng);
        let dec_b1 = zeros(config.hidden_dim);
        let dec_w2 = xavier(config.input_dim, config.hidden_dim, &mut rng);
        let dec_b2 = zeros(config.input_dim);

        Self {
            config,
            enc_w1,
            enc_b1,
            enc_mu_w,
            enc_mu_b,
            enc_lv_w,
            enc_lv_b,
            dec_w1,
            dec_b1,
            dec_w2,
            dec_b2,
        }
    }

    /// Encodes input to `(mu, log_var)` vectors.
    pub fn encode(&self, x: &[f32]) -> AnomalyResult<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.config.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.config.input_dim,
                found: x.len(),
            });
        }
        // hidden = ReLU(enc_w1 @ x + enc_b1)
        let hidden: Vec<f32> = matvec(&self.enc_w1, x)
            .iter()
            .zip(self.enc_b1.iter())
            .map(|(h, b)| (h + b).max(0.0))
            .collect();
        let mu: Vec<f32> = matvec(&self.enc_mu_w, &hidden)
            .iter()
            .zip(self.enc_mu_b.iter())
            .map(|(v, b)| v + b)
            .collect();
        let log_var: Vec<f32> = matvec(&self.enc_lv_w, &hidden)
            .iter()
            .zip(self.enc_lv_b.iter())
            .map(|(v, b)| v + b)
            .collect();
        Ok((mu, log_var))
    }

    /// Decodes latent vector `z` to reconstruction.
    pub fn decode(&self, z: &[f32]) -> AnomalyResult<Vec<f32>> {
        if z.len() != self.config.latent_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.config.latent_dim,
                found: z.len(),
            });
        }
        let hidden: Vec<f32> = matvec(&self.dec_w1, z)
            .iter()
            .zip(self.dec_b1.iter())
            .map(|(h, b)| (h + b).max(0.0))
            .collect();
        let recon: Vec<f32> = matvec(&self.dec_w2, &hidden)
            .iter()
            .zip(self.dec_b2.iter())
            .map(|(v, b)| v + b)
            .collect();
        Ok(recon)
    }

    /// Anomaly score = reconstruction MSE + β * KL.
    ///
    /// Uses the mean of the approximate posterior (z = mu) for a deterministic
    /// reconstruction during inference.
    pub fn anomaly_score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let (mu, log_var) = self.encode(x)?;
        let recon = self.decode(&mu)?;

        // Reconstruction loss: mean squared error.
        let recon_loss: f32 = x
            .iter()
            .zip(recon.iter())
            .map(|(xi, ri)| (xi - ri).powi(2))
            .sum::<f32>()
            / x.len() as f32;

        // KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var)).
        let kl: f32 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(m, lv)| {
                let lv_c = lv.clamp(-10.0, 10.0);
                -0.5 * (1.0 + lv_c - m * m - lv_c.exp())
            })
            .sum::<f32>()
            / mu.len() as f32;

        Ok(recon_loss + self.config.beta * kl)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. DeepSVDD
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`DeepSVDD`].
#[derive(Debug, Clone)]
pub struct DeepSvddConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Output (latent / embedding) dimensionality.
    pub output_dim: usize,
    /// Hidden layer size.
    pub hidden_dim: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for DeepSvddConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            output_dim: 16,
            hidden_dim: 64,
            seed: 0,
        }
    }
}

/// Deep Support Vector Data Description.
///
/// Maps inputs to an embedding space and minimises the volume of a
/// hypersphere centred at `c`. Anomaly score = ‖φ(x) − c‖².
pub struct DeepSVDD {
    config: DeepSvddConfig,
    /// First layer weights [hidden_dim × input_dim].
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    /// Second (output) layer weights [output_dim × hidden_dim].
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    /// Hypersphere centre in embedding space.
    pub center: Vec<f32>,
}

impl DeepSVDD {
    /// Creates a new `DeepSVDD` with randomly initialised weights and zero centre.
    pub fn new(config: DeepSvddConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let scale1 = (2.0_f32 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let scale2 = (2.0_f32 / (config.hidden_dim + config.output_dim) as f32).sqrt();

        let w1: Vec<Vec<f32>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| sample_normal_f32(&mut rng) * scale1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0_f32; config.hidden_dim];
        let w2: Vec<Vec<f32>> = (0..config.output_dim)
            .map(|_| {
                (0..config.hidden_dim)
                    .map(|_| sample_normal_f32(&mut rng) * scale2)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0_f32; config.output_dim];
        let center = vec![0.0_f32; config.output_dim];

        Self {
            config,
            w1,
            b1,
            w2,
            b2,
            center,
        }
    }

    /// Forward pass: input → embedding.
    pub fn forward(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.config.input_dim,
                found: x.len(),
            });
        }
        let h: Vec<f32> = matvec(&self.w1, x)
            .iter()
            .zip(self.b1.iter())
            .map(|(v, b)| (v + b).max(0.0))
            .collect();
        let out: Vec<f32> = matvec(&self.w2, &h)
            .iter()
            .zip(self.b2.iter())
            .map(|(v, b)| v + b)
            .collect();
        Ok(out)
    }

    /// Anomaly score = ‖φ(x) − c‖².
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let phi = self.forward(x)?;
        let dist_sq: f32 = phi
            .iter()
            .zip(self.center.iter())
            .map(|(p, c)| (p - c).powi(2))
            .sum();
        Ok(dist_sq)
    }

    /// Sets the hypersphere centre to the mean of provided embeddings.
    ///
    /// Embeddings must all have dimension `output_dim`.
    pub fn update_center(&mut self, embeddings: &[Vec<f32>]) -> AnomalyResult<()> {
        if embeddings.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        let d = self.config.output_dim;
        let mut mean = vec![0.0_f32; d];
        for emb in embeddings {
            if emb.len() != d {
                return Err(AnomalyError::DimensionMismatch {
                    expected: d,
                    found: emb.len(),
                });
            }
            for (m, e) in mean.iter_mut().zip(emb.iter()) {
                *m += e;
            }
        }
        let n = embeddings.len() as f32;
        for m in mean.iter_mut() {
            *m /= n;
        }
        self.center = mean;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. PatchTSAD
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`PatchTSAD`].
#[derive(Debug, Clone)]
pub struct PatchTsadConfig {
    /// Latent embedding dimension for each patch.
    pub latent_dim: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for PatchTsadConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            seed: 7,
        }
    }
}

/// Patch-based time-series anomaly detector.
///
/// Divides a series into overlapping patches, encodes each with a linear
/// projection followed by a decoder, and returns per-patch reconstruction
/// error as the anomaly score.
pub struct PatchTSAD {
    config: PatchTsadConfig,
    /// Encoder weight: latent_dim × patch_size (set lazily on first use).
    enc_w: Option<Vec<Vec<f32>>>,
    /// Decoder weight: patch_size × latent_dim.
    dec_w: Option<Vec<Vec<f32>>>,
    last_patch_size: usize,
}

impl PatchTSAD {
    /// Creates a new `PatchTSAD`.
    pub fn new(config: PatchTsadConfig) -> Self {
        Self {
            config,
            enc_w: None,
            dec_w: None,
            last_patch_size: 0,
        }
    }

    /// Ensures encoder/decoder weights are initialised for the given patch size.
    fn ensure_weights(&mut self, patch_size: usize) {
        if self.enc_w.is_none() || self.last_patch_size != patch_size {
            let mut rng = StdRng::seed_from_u64(self.config.seed);
            let d = self.config.latent_dim;
            let scale = (2.0_f32 / (patch_size + d) as f32).sqrt();
            let enc_w: Vec<Vec<f32>> = (0..d)
                .map(|_| {
                    (0..patch_size)
                        .map(|_| sample_normal_f32(&mut rng) * scale)
                        .collect()
                })
                .collect();
            let dec_w: Vec<Vec<f32>> = (0..patch_size)
                .map(|_| {
                    (0..d)
                        .map(|_| sample_normal_f32(&mut rng) * scale)
                        .collect()
                })
                .collect();
            self.enc_w = Some(enc_w);
            self.dec_w = Some(dec_w);
            self.last_patch_size = patch_size;
        }
    }

    /// Per-patch reconstruction error.
    fn patch_score(&self, patch: &[f32]) -> f32 {
        let enc_w = match &self.enc_w {
            Some(w) => w,
            None => return 0.0,
        };
        let dec_w = match &self.dec_w {
            Some(w) => w,
            None => return 0.0,
        };
        // encode
        let latent: Vec<f32> = matvec(enc_w, patch).iter().map(|v| v.max(0.0)).collect();
        // decode
        let recon = matvec(dec_w, &latent);
        // MSE
        patch
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / patch.len() as f32
    }

    /// Returns per-patch anomaly scores for the provided time series.
    ///
    /// `patch_size`: length of each patch.
    /// `stride`: step between consecutive patches.
    pub fn score_series(
        &mut self,
        series: &[f32],
        patch_size: usize,
        stride: usize,
    ) -> AnomalyResult<Vec<f32>> {
        if series.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        if patch_size == 0 {
            return Err(AnomalyError::InvalidConfig("patch_size must be > 0".into()));
        }
        if stride == 0 {
            return Err(AnomalyError::InvalidConfig("stride must be > 0".into()));
        }
        self.ensure_weights(patch_size);

        let mut scores = Vec::new();
        let n = series.len();
        let mut start = 0_usize;
        while start + patch_size <= n {
            let patch = &series[start..start + patch_size];
            scores.push(self.patch_score(patch));
            start += stride;
        }
        Ok(scores)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. AnomalyTransformerModel
// ─────────────────────────────────────────────────────────────────────────────

/// Association discrepancy-based anomaly transformer.
///
/// Based on Xu et al. (2022) "Anomaly Transformer: Time Series Anomaly Detection
/// with Association Discrepancy". Computes the KL divergence between a learnable
/// prior association (Gaussian kernel) and the data-driven series association
/// (self-attention weights).
pub struct AnomalyTransformerModel {
    /// Standard deviation for the Gaussian kernel prior.
    pub prior_sigma: f32,
    /// Embedding / query-key dimension.
    pub d_model: usize,
}

impl AnomalyTransformerModel {
    /// Creates a new `AnomalyTransformerModel`.
    pub fn new(d_model: usize, prior_sigma: f32) -> AnomalyResult<Self> {
        if d_model == 0 {
            return Err(AnomalyError::InvalidConfig("d_model must be > 0".into()));
        }
        let sigma = if prior_sigma > 0.0 { prior_sigma } else { 1.0 };
        Ok(Self {
            prior_sigma: sigma,
            d_model,
        })
    }

    /// Prior association matrix via Gaussian kernel over row-distance.
    ///
    /// `P[i][j] = exp(-|i-j|^2 / (2σ²))`, row-normalised via softmax.
    pub fn prior_association(&self, seq_len: usize) -> Vec<Vec<f32>> {
        let two_sig2 = 2.0 * self.prior_sigma * self.prior_sigma;
        (0..seq_len)
            .map(|i| {
                let logits: Vec<f32> = (0..seq_len)
                    .map(|j| {
                        let d = (i as f32 - j as f32).powi(2);
                        -d / two_sig2
                    })
                    .collect();
                softmax_vec(&logits)
            })
            .collect()
    }

    /// Series association from the input sequence via scaled dot-product attention.
    ///
    /// `x` is a flat row-major matrix of shape `[seq_len × d]`.
    /// Returns a `seq_len × seq_len` row-stochastic attention matrix.
    pub fn series_association(
        &self,
        x: &[f32],
        seq_len: usize,
        d: usize,
    ) -> AnomalyResult<Vec<Vec<f32>>> {
        if x.len() != seq_len * d {
            return Err(AnomalyError::DimensionMismatch {
                expected: seq_len * d,
                found: x.len(),
            });
        }
        let scale = 1.0 / (d as f32).sqrt();
        let rows: Vec<&[f32]> = (0..seq_len).map(|i| &x[i * d..(i + 1) * d]).collect();

        let attn: Vec<Vec<f32>> = rows
            .iter()
            .map(|q| {
                let logits: Vec<f32> = rows.iter().map(|k| dot(q, k) * scale).collect();
                softmax_vec(&logits)
            })
            .collect();
        Ok(attn)
    }

    /// KL divergence between prior and series association (averaged over rows).
    ///
    /// KL(P ‖ Q) = Σ P\[i\] * log(P\[i\] / Q\[i\]), averaged across all rows.
    pub fn association_discrepancy(
        &self,
        prior: &[Vec<f32>],
        series: &[Vec<f32>],
    ) -> AnomalyResult<f32> {
        if prior.len() != series.len() {
            return Err(AnomalyError::DimensionMismatch {
                expected: prior.len(),
                found: series.len(),
            });
        }
        if prior.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        let mut total = 0.0_f32;
        let eps = 1e-12_f32;
        for (p_row, s_row) in prior.iter().zip(series.iter()) {
            let kl: f32 = p_row
                .iter()
                .zip(s_row.iter())
                .map(|(&p, &q)| {
                    let p_c = p.max(eps);
                    let q_c = q.max(eps);
                    p_c * (p_c / q_c).ln()
                })
                .sum();
            total += kl;
        }
        Ok(total / prior.len() as f32)
    }

    /// Compute anomaly score for a single input sequence.
    ///
    /// Returns the symmetric association discrepancy KL(P‖Q) + KL(Q‖P).
    pub fn anomaly_score_sequence(
        &self,
        x: &[f32],
        seq_len: usize,
        d: usize,
    ) -> AnomalyResult<f32> {
        let prior = self.prior_association(seq_len);
        let series = self.series_association(x, seq_len, d)?;
        let kl_pq = self.association_discrepancy(&prior, &series)?;
        let kl_qp = self.association_discrepancy(&series, &prior)?;
        Ok(kl_pq + kl_qp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. MemoryAugmentedAE
// ─────────────────────────────────────────────────────────────────────────────

/// A memory bank used by [`MemoryAugmentedAE`].
#[derive(Debug, Clone)]
pub struct AnomalyMemory {
    /// Memory slot matrix: `n_slots × slot_dim`.
    pub slots: Vec<Vec<f32>>,
    /// Number of memory slots.
    pub n_slots: usize,
    /// Dimensionality of each slot.
    pub slot_dim: usize,
}

impl AnomalyMemory {
    /// Initialises memory with small random values.
    pub fn new(n_slots: usize, slot_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let slots: Vec<Vec<f32>> = (0..n_slots)
            .map(|_| {
                (0..slot_dim)
                    .map(|_| sample_normal_f32(&mut rng) * 0.01)
                    .collect()
            })
            .collect();
        Self {
            slots,
            n_slots,
            slot_dim,
        }
    }

    /// Addressing weights via softmax of cosine similarities.
    pub fn address(&self, z: &[f32]) -> AnomalyResult<Vec<f32>> {
        if z.len() != self.slot_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.slot_dim,
                found: z.len(),
            });
        }
        let sims: Vec<f32> = self.slots.iter().map(|s| cosine_sim(s, z)).collect();
        Ok(softmax_vec(&sims))
    }

    /// Weighted sum of memory slots (read operation).
    pub fn query(&self, z: &[f32]) -> AnomalyResult<Vec<f32>> {
        let weights = self.address(z)?;
        let mut result = vec![0.0_f32; self.slot_dim];
        for (w, slot) in weights.iter().zip(self.slots.iter()) {
            for (r, s) in result.iter_mut().zip(slot.iter()) {
                *r += w * s;
            }
        }
        Ok(result)
    }

    /// Soft write: updates each slot proportionally to its addressing weight.
    pub fn update(&mut self, z: &[f32]) -> AnomalyResult<()> {
        let weights = self.address(z)?;
        for (w, slot) in weights.iter().zip(self.slots.iter_mut()) {
            for (s, zi) in slot.iter_mut().zip(z.iter()) {
                *s = (1.0 - w) * (*s) + w * zi;
            }
        }
        Ok(())
    }
}

/// Configuration for [`MemoryAugmentedAE`].
#[derive(Debug, Clone)]
pub struct MemAeConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Latent dimensionality (also the memory slot dimension).
    pub latent_dim: usize,
    /// Number of memory slots.
    pub n_slots: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for MemAeConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            latent_dim: 16,
            n_slots: 10,
            seed: 99,
        }
    }
}

/// Memory-augmented autoencoder for anomaly detection.
///
/// The encoder maps x → z; the memory module retrieves a reconstruction
/// ẑ = query(z); the anomaly score is ‖z − ẑ‖².  Normal data has compact
/// memory representations; anomalies produce high retrieval error.
pub struct MemoryAugmentedAE {
    config: MemAeConfig,
    /// Encoder [latent_dim × input_dim].
    enc_w: Vec<Vec<f32>>,
    enc_b: Vec<f32>,
    /// Decoder [input_dim × latent_dim].
    dec_w: Vec<Vec<f32>>,
    dec_b: Vec<f32>,
    /// Memory bank.
    pub memory: AnomalyMemory,
}

impl MemoryAugmentedAE {
    /// Creates a new `MemoryAugmentedAE`.
    pub fn new(config: MemAeConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let scale_enc = (2.0_f32 / (config.input_dim + config.latent_dim) as f32).sqrt();
        let scale_dec = (2.0_f32 / (config.latent_dim + config.input_dim) as f32).sqrt();

        let enc_w: Vec<Vec<f32>> = (0..config.latent_dim)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| sample_normal_f32(&mut rng) * scale_enc)
                    .collect()
            })
            .collect();
        let enc_b = vec![0.0_f32; config.latent_dim];
        let dec_w: Vec<Vec<f32>> = (0..config.input_dim)
            .map(|_| {
                (0..config.latent_dim)
                    .map(|_| sample_normal_f32(&mut rng) * scale_dec)
                    .collect()
            })
            .collect();
        let dec_b = vec![0.0_f32; config.input_dim];
        let memory = AnomalyMemory::new(config.n_slots, config.latent_dim, config.seed + 1);

        Self {
            config,
            enc_w,
            enc_b,
            dec_w,
            dec_b,
            memory,
        }
    }

    /// Encodes input to latent vector.
    pub fn encode(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.config.input_dim,
                found: x.len(),
            });
        }
        let z: Vec<f32> = matvec(&self.enc_w, x)
            .iter()
            .zip(self.enc_b.iter())
            .map(|(v, b)| (v + b).max(0.0))
            .collect();
        Ok(z)
    }

    /// Decodes latent vector to reconstruction.
    pub fn decode(&self, z: &[f32]) -> AnomalyResult<Vec<f32>> {
        if z.len() != self.config.latent_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.config.latent_dim,
                found: z.len(),
            });
        }
        let out: Vec<f32> = matvec(&self.dec_w, z)
            .iter()
            .zip(self.dec_b.iter())
            .map(|(v, b)| v + b)
            .collect();
        Ok(out)
    }

    /// Anomaly score = ‖z − memory_readout(z)‖².
    pub fn anomaly_score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let z = self.encode(x)?;
        let z_hat = self.memory.query(&z)?;
        let score: f32 = z
            .iter()
            .zip(z_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        Ok(score)
    }

    /// Performs a memory update for a given input.
    pub fn update_memory(&mut self, x: &[f32]) -> AnomalyResult<()> {
        let z = self.encode(x)?;
        self.memory.update(&z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. RobustRCF (Robust Random Cut Forest)
// ─────────────────────────────────────────────────────────────────────────────

/// A single axis-aligned cut stored in the RCF tree.
#[derive(Debug, Clone)]
struct RcfCut {
    dim: usize,
    val: f32,
}

/// A node in an RCF tree.
#[derive(Debug, Clone)]
enum RcfNode {
    Leaf {
        /// Stored data point.
        point: Vec<f32>,
        /// Depth of this leaf.
        depth: usize,
    },
    Internal {
        cut: RcfCut,
        left: Box<RcfNode>,
        right: Box<RcfNode>,
        /// Total number of data points in this subtree.
        count: usize,
        /// Depth of this node.
        depth: usize,
    },
}

impl RcfNode {
    fn count(&self) -> usize {
        match self {
            RcfNode::Leaf { .. } => 1,
            RcfNode::Internal { count, .. } => *count,
        }
    }

    fn depth(&self) -> usize {
        match self {
            RcfNode::Leaf { depth, .. } => *depth,
            RcfNode::Internal { depth, .. } => *depth,
        }
    }
}

/// Build an RCF tree from a set of points using random axis-aligned cuts.
fn build_rcf_tree(points: &[Vec<f32>], depth: usize, rng: &mut StdRng) -> RcfNode {
    if points.len() == 1 {
        return RcfNode::Leaf {
            point: points[0].clone(),
            depth,
        };
    }

    let dim = points[0].len();
    if dim == 0 {
        return RcfNode::Leaf {
            point: Vec::new(),
            depth,
        };
    }

    // Find bounding box and compute ranges.
    let mut lo = vec![f32::INFINITY; dim];
    let mut hi = vec![f32::NEG_INFINITY; dim];
    for p in points {
        for (d, &v) in p.iter().enumerate() {
            if v < lo[d] {
                lo[d] = v;
            }
            if v > hi[d] {
                hi[d] = v;
            }
        }
    }
    let ranges: Vec<f32> = lo.iter().zip(hi.iter()).map(|(l, h)| h - l).collect();
    let total_range: f32 = ranges.iter().sum();

    // Choose cut dimension proportional to range.
    let cut_dim = if total_range < 1e-12 {
        rng.random_range(0..dim)
    } else {
        let mut target = rng.random::<f32>() * total_range;
        let mut chosen = dim - 1;
        for (d, &r) in ranges.iter().enumerate() {
            target -= r;
            if target <= 0.0 {
                chosen = d;
                break;
            }
        }
        chosen
    };

    // Cut value uniformly in [lo[cut_dim], hi[cut_dim]].
    let lo_d = lo[cut_dim];
    let hi_d = hi[cut_dim];
    let cut_val = if (hi_d - lo_d).abs() < 1e-12 {
        lo_d
    } else {
        lo_d + rng.random::<f32>() * (hi_d - lo_d)
    };

    let (left_pts, right_pts): (Vec<Vec<f32>>, Vec<Vec<f32>>) =
        points.iter().cloned().partition(|p| p[cut_dim] <= cut_val);

    // Avoid degenerate splits.
    if left_pts.is_empty() || right_pts.is_empty() {
        let mid = points.len() / 2;
        let (l, r) = points.split_at(mid.max(1));
        if r.is_empty() {
            return RcfNode::Leaf {
                point: l[0].clone(),
                depth,
            };
        }
        let left = build_rcf_tree(l, depth + 1, rng);
        let right = build_rcf_tree(r, depth + 1, rng);
        let count = left.count() + right.count();
        return RcfNode::Internal {
            cut: RcfCut {
                dim: cut_dim,
                val: cut_val,
            },
            left: Box::new(left),
            right: Box::new(right),
            count,
            depth,
        };
    }

    let left = build_rcf_tree(&left_pts, depth + 1, rng);
    let right = build_rcf_tree(&right_pts, depth + 1, rng);
    let count = left.count() + right.count();

    RcfNode::Internal {
        cut: RcfCut {
            dim: cut_dim,
            val: cut_val,
        },
        left: Box::new(left),
        right: Box::new(right),
        count,
        depth,
    }
}

/// Compute CODISP (Conditional Displacement) for a query point in a tree.
///
/// CODISP = max displacement if the point were deleted = count_of_sibling / depth.
fn codisp(tree: &RcfNode, point: &[f32]) -> f32 {
    let mut node = tree;
    let mut codisp_val = 0.0_f32;

    loop {
        match node {
            RcfNode::Leaf { .. } => break,
            RcfNode::Internal {
                cut, left, right, ..
            } => {
                let (same_side, other_side) = if point[cut.dim] <= cut.val {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };
                let d = same_side.depth() as f32;
                let displacement = other_side.count() as f32;
                if d > 0.0 {
                    let c = displacement / d;
                    if c > codisp_val {
                        codisp_val = c;
                    }
                }
                node = same_side;
            }
        }
    }
    codisp_val
}

/// Configuration for [`RobustRCF`].
#[derive(Debug, Clone)]
pub struct RobustRcfConfig {
    /// Number of trees in the forest.
    pub n_trees: usize,
    /// Sliding window size — maximum number of points stored.
    pub window_size: usize,
    /// Dimensionality of each data point.
    pub dim: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for RobustRcfConfig {
    fn default() -> Self {
        Self {
            n_trees: 40,
            window_size: 256,
            dim: 1,
            seed: 123,
        }
    }
}

/// Robust Random Cut Forest streaming anomaly detector.
///
/// Maintains a sliding window of recent data points and a forest of RCF trees.
/// Anomaly score = average CODISP across all trees.
pub struct RobustRCF {
    config: RobustRcfConfig,
    /// Circular buffer of recent points.
    window: Vec<Vec<f32>>,
    /// Forest of RCF trees (rebuilt on demand).
    trees: Vec<RcfNode>,
    /// Whether the forest needs rebuilding.
    dirty: bool,
    rng: StdRng,
}

impl RobustRCF {
    /// Creates a new empty `RobustRCF`.
    pub fn new(config: RobustRcfConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            window: Vec::new(),
            trees: Vec::new(),
            dirty: false,
            rng,
        }
    }

    /// Inserts a new data point into the sliding window.
    pub fn insert(&mut self, point: Vec<f32>) {
        if self.window.len() >= self.config.window_size {
            self.window.remove(0);
        }
        self.window.push(point);
        self.dirty = true;
    }

    /// Rebuilds the forest of trees from the current window.
    fn rebuild_forest(&mut self) {
        if self.window.is_empty() {
            self.trees.clear();
            self.dirty = false;
            return;
        }
        self.trees.clear();
        for _ in 0..self.config.n_trees {
            // Sample a bootstrap subset of the window.
            let n = self.window.len();
            let sample_size = n.min(self.config.window_size);
            let sample: Vec<Vec<f32>> = (0..sample_size)
                .map(|_| {
                    let idx = self.rng.random_range(0..n);
                    self.window[idx].clone()
                })
                .collect();
            let tree = build_rcf_tree(&sample, 0, &mut self.rng);
            self.trees.push(tree);
        }
        self.dirty = false;
    }

    /// Returns the anomaly score for a query point (average CODISP).
    pub fn score(&mut self, point: &[f32]) -> f32 {
        if self.dirty {
            self.rebuild_forest();
        }
        if self.trees.is_empty() {
            return 0.0;
        }
        let total: f32 = self.trees.iter().map(|t| codisp(t, point)).sum();
        total / self.trees.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SpectralResidual
// ─────────────────────────────────────────────────────────────────────────────

/// FFT-based Spectral Residual anomaly detector (Microsoft SRAD / SR-CNN).
///
/// Algorithm:
/// 1. Compute DFT of input series.
/// 2. Take log of the amplitude spectrum.
/// 3. Subtract a moving average (smoothed log-amplitude).
/// 4. Reconstruct via IDFT to get the saliency map.
/// 5. Optionally normalise the saliency map.
pub struct SpectralResidual {
    /// Size of the moving-average window for log-amplitude smoothing.
    pub avg_window: usize,
}

impl SpectralResidual {
    /// Creates a new `SpectralResidual` with the specified average window.
    pub fn new(avg_window: usize) -> Self {
        Self {
            avg_window: avg_window.max(1),
        }
    }

    /// Computes saliency map from the input time series.
    ///
    /// Returns per-point anomaly scores (absolute values of the saliency map).
    pub fn compute(&self, series: &[f32]) -> AnomalyResult<Vec<f32>> {
        if series.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        let n = series.len();

        // Step 1: DFT.
        let spectrum = dft(series);

        // Step 2: log amplitude.
        let log_amp: Vec<f32> = spectrum
            .iter()
            .map(|(re, im)| {
                let amp = (re * re + im * im).sqrt().max(1e-12);
                amp.ln()
            })
            .collect();

        // Step 3: moving average of log amplitude.
        let w = self.avg_window.min(n);
        let smoothed: Vec<f32> = (0..n)
            .map(|i| {
                let start = (i + 1).saturating_sub(w);
                let end = i + 1;
                let sum: f32 = log_amp[start..end].iter().sum();
                sum / (end - start) as f32
            })
            .collect();

        // Step 4: spectral residual = exp(log_amp - smoothed).
        let residual_amp: Vec<f32> = log_amp
            .iter()
            .zip(smoothed.iter())
            .map(|(la, sm)| (la - sm).exp())
            .collect();

        // Reconstruct spectrum with residual amplitude, preserving phase.
        let residual_spectrum: Vec<(f32, f32)> = spectrum
            .iter()
            .zip(residual_amp.iter())
            .map(|((re, im), &ra)| {
                let amp = (re * re + im * im).sqrt().max(1e-30);
                let scale = ra / amp;
                (re * scale, im * scale)
            })
            .collect();

        // Step 5: IDFT to saliency map.
        let saliency = idft(&residual_spectrum);

        // Return absolute value as anomaly score.
        Ok(saliency.iter().map(|v| v.abs()).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. GaussianMixtureAnomaly
// ─────────────────────────────────────────────────────────────────────────────

/// GMM-based density estimation for anomaly detection.
///
/// Fits a Gaussian Mixture Model via the Expectation-Maximisation algorithm.
/// Anomaly score = -log p(x) under the fitted mixture.
pub struct GaussianMixtureAnomaly {
    /// Number of Gaussian components.
    pub n_components: usize,
    /// Component means — shape `[n_components × d]`.
    pub means: Vec<Vec<f32>>,
    /// Diagonal covariance (variance) per component — shape `[n_components × d]`.
    pub covs: Vec<Vec<f32>>,
    /// Mixture weights — shape `[n_components]`.
    pub weights: Vec<f32>,
    /// Minimum variance to prevent numerical collapse.
    min_var: f32,
}

impl GaussianMixtureAnomaly {
    /// Creates a new uninitialised `GaussianMixtureAnomaly`.
    pub fn new(n_components: usize) -> AnomalyResult<Self> {
        if n_components == 0 {
            return Err(AnomalyError::InvalidConfig(
                "n_components must be > 0".into(),
            ));
        }
        Ok(Self {
            n_components,
            means: Vec::new(),
            covs: Vec::new(),
            weights: Vec::new(),
            min_var: 1e-4,
        })
    }

    /// Log-probability of x under a diagonal Gaussian N(mu, diag(var)).
    fn log_gaussian(x: &[f32], mu: &[f32], var: &[f32]) -> f32 {
        let d = x.len() as f32;
        let log_det: f32 = var.iter().map(|v| v.max(1e-12).ln()).sum();
        let maha: f32 = x
            .iter()
            .zip(mu.iter())
            .zip(var.iter())
            .map(|((xi, mi), vi)| (xi - mi).powi(2) / vi.max(1e-12))
            .sum();
        -0.5 * (d * (2.0 * PI).ln() + log_det + maha)
    }

    /// Fits the GMM to data using EM (10 iterations).
    pub fn fit(&mut self, data: &[Vec<f32>]) -> AnomalyResult<()> {
        if data.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        let n = data.len();
        let d = data[0].len();
        if d == 0 {
            return Err(AnomalyError::InvalidConfig(
                "data dimension must be > 0".into(),
            ));
        }
        let k = self.n_components;

        // K-means++ initialisation.
        let mut rng = StdRng::seed_from_u64(42);
        let mut means: Vec<Vec<f32>> = Vec::with_capacity(k);

        // Pick first centroid uniformly.
        let first = rng.random_range(0..n);
        means.push(data[first].clone());

        // Pick remaining centroids proportional to D² distance.
        for _ in 1..k {
            let dists: Vec<f32> = data
                .iter()
                .map(|pt| {
                    means
                        .iter()
                        .map(|m| {
                            pt.iter()
                                .zip(m.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>()
                        })
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();
            let total: f32 = dists.iter().sum();
            if total < 1e-12 {
                // All points coincide; pick random.
                means.push(data[rng.random_range(0..n)].clone());
            } else {
                let mut target = rng.random::<f32>() * total;
                let mut chosen = n - 1;
                for (i, &d_sq) in dists.iter().enumerate() {
                    target -= d_sq;
                    if target <= 0.0 {
                        chosen = i;
                        break;
                    }
                }
                means.push(data[chosen].clone());
            }
        }

        // Initialise equal weights and unit variances.
        let mut weights = vec![1.0_f32 / k as f32; k];
        let mut covs: Vec<Vec<f32>> = vec![vec![1.0_f32; d]; k];

        // EM iterations.
        let max_iter = 10_usize;
        for _ in 0..max_iter {
            // --- E-step: compute responsibilities ---
            let mut resps: Vec<Vec<f32>> = Vec::with_capacity(n);
            for pt in data.iter() {
                let log_probs: Vec<f32> = (0..k)
                    .map(|c| {
                        weights[c].max(1e-12).ln() + Self::log_gaussian(pt, &means[c], &covs[c])
                    })
                    .collect();
                let log_norm = log_sum_exp_f32(&log_probs);
                let r: Vec<f32> = log_probs.iter().map(|lp| (lp - log_norm).exp()).collect();
                resps.push(r);
            }

            // --- M-step: update parameters ---
            for c in 0..k {
                let n_c: f32 = resps.iter().map(|r| r[c]).sum();
                let n_c = n_c.max(1e-6);

                // Update mean.
                let mut new_mean = vec![0.0_f32; d];
                for (pt, r) in data.iter().zip(resps.iter()) {
                    for (m, x) in new_mean.iter_mut().zip(pt.iter()) {
                        *m += r[c] * x;
                    }
                }
                for m in new_mean.iter_mut() {
                    *m /= n_c;
                }

                // Update covariance (diagonal).
                let mut new_cov = vec![0.0_f32; d];
                for (pt, r) in data.iter().zip(resps.iter()) {
                    for (cv, (x, m)) in new_cov.iter_mut().zip(pt.iter().zip(new_mean.iter())) {
                        *cv += r[c] * (x - m).powi(2);
                    }
                }
                for cv in new_cov.iter_mut() {
                    *cv = (*cv / n_c).max(self.min_var);
                }

                means[c] = new_mean;
                covs[c] = new_cov;
                weights[c] = n_c / n as f32;
            }
        }

        self.means = means;
        self.covs = covs;
        self.weights = weights;
        Ok(())
    }

    /// Log-likelihood of a single data point under the fitted GMM.
    pub fn log_likelihood(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.means.is_empty() {
            return Err(AnomalyError::InvalidConfig("model not yet fitted".into()));
        }
        let log_probs: Vec<f32> = (0..self.n_components)
            .map(|c| {
                self.weights[c].max(1e-12).ln()
                    + Self::log_gaussian(x, &self.means[c], &self.covs[c])
            })
            .collect();
        Ok(log_sum_exp_f32(&log_probs))
    }

    /// Anomaly score = −log p(x).
    pub fn anomaly_score(&self, x: &[f32]) -> AnomalyResult<f32> {
        Ok(-self.log_likelihood(x)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. AnomalyThresholder
// ─────────────────────────────────────────────────────────────────────────────

/// Threshold selection strategies for anomaly detection.
pub struct AnomalyThresholder;

impl AnomalyThresholder {
    /// Peak Over Threshold (POT): returns the empirical `quantile`-th percentile.
    ///
    /// `quantile` should be in [0, 1].
    pub fn peak_over_threshold(scores: &[f32], quantile: f32) -> AnomalyResult<f32> {
        if scores.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        let q = quantile.clamp(0.0, 1.0);
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((q * (sorted.len() as f32 - 1.0)).round() as usize).min(sorted.len() - 1);
        Ok(sorted[idx])
    }

    /// Extreme Value Theory (EVT): fits a Generalised Pareto Distribution to the
    /// upper tail and returns the 99th percentile.
    ///
    /// Uses the method of moments estimator for GPD parameters.
    pub fn extreme_value_theory(scores: &[f32]) -> AnomalyResult<f32> {
        if scores.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        // Use upper 10% as tail.
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let tail_start_idx = ((sorted.len() as f32 * 0.9).ceil() as usize).min(sorted.len() - 1);
        let u = sorted[tail_start_idx];
        let tail: Vec<f32> = sorted[tail_start_idx..].iter().map(|&x| x - u).collect();

        if tail.len() < 2 {
            return Ok(sorted[sorted.len() - 1]);
        }

        // Method of moments for GPD (Hosking & Wallis, 1987).
        let nt = tail.len() as f32;
        let mean: f32 = tail.iter().sum::<f32>() / nt;
        let var: f32 = tail.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / nt;

        // GPD: scale sigma, shape xi.
        let (sigma, xi) = if var < 1e-12 {
            (mean, 0.0_f32)
        } else {
            let xi = 0.5 * (1.0 - mean * mean / var);
            let sigma = mean * (1.0 - xi);
            (sigma.max(1e-6), xi)
        };

        // Quantile at probability p = 0.99.
        let n_total = scores.len() as f32;
        let n_tail = tail.len() as f32;
        let p = 0.99_f32;
        // Exceedance probability of u in the full dataset.
        let p_exceed = n_tail / n_total;
        // Conditional p so that overall P(X > q) = (1-p).
        let p_cond = ((1.0 - p) / p_exceed).clamp(0.0, 1.0);

        let quantile_tail = if xi.abs() < 1e-6 {
            // Exponential limit.
            -sigma * (p_cond).max(1e-30).ln()
        } else {
            sigma / xi * (p_cond.powf(-xi) - 1.0)
        };
        Ok(u + quantile_tail.max(0.0))
    }

    /// Adaptive threshold: rolling mean + `k` standard deviations.
    ///
    /// Returns a per-point threshold vector of the same length as `scores`.
    pub fn adaptive_threshold(scores: &[f32], window: usize, k: f32) -> AnomalyResult<Vec<f32>> {
        if scores.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        let w = window.max(1).min(scores.len());
        let thresholds: Vec<f32> = scores
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = (i + 1).saturating_sub(w);
                let end = i + 1;
                let slice = &scores[start..end];
                let n = slice.len() as f32;
                let mean: f32 = slice.iter().sum::<f32>() / n;
                let std: f32 = (slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
                mean + k * std
            })
            .collect();
        Ok(thresholds)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. AnomalyEvaluationMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for anomaly detection.
///
/// Note: exported as `AnomalyEvaluationMetrics` to avoid collision with any
/// existing `EvaluationMetrics` in the crate.
pub struct AnomalyEvaluationMetrics;

impl AnomalyEvaluationMetrics {
    /// Area under the ROC curve (trapezoidal rule).
    pub fn roc_auc(scores: &[f32], labels: &[bool]) -> AnomalyResult<f32> {
        Self::validate(scores, labels)?;
        let n = scores.len();
        let mut pairs: Vec<(f32, bool)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        // Sort descending by score.
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_pos = labels.iter().filter(|&&l| l).count() as f32;
        let n_neg = n as f32 - n_pos;
        if n_pos == 0.0 || n_neg == 0.0 {
            return Err(AnomalyError::InvalidConfig(
                "both positive and negative labels are required".into(),
            ));
        }

        let mut tp = 0.0_f32;
        let mut fp = 0.0_f32;
        let mut auc = 0.0_f32;
        let mut prev_fp = 0.0_f32;
        let mut prev_tp = 0.0_f32;

        for (_, label) in &pairs {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            // Trapezoidal rule.
            auc += (fp / n_neg - prev_fp / n_neg) * (tp / n_pos + prev_tp / n_pos) * 0.5;
            prev_fp = fp;
            prev_tp = tp;
        }
        Ok(auc.clamp(0.0, 1.0))
    }

    /// Area under the Precision-Recall curve (trapezoidal rule).
    pub fn pr_auc(scores: &[f32], labels: &[bool]) -> AnomalyResult<f32> {
        Self::validate(scores, labels)?;
        let n_pos = labels.iter().filter(|&&l| l).count() as f32;
        if n_pos == 0.0 {
            return Err(AnomalyError::InvalidConfig("no positive labels".into()));
        }

        let mut pairs: Vec<(f32, bool)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tp = 0.0_f32;
        let mut fp = 0.0_f32;
        let mut auc = 0.0_f32;
        let mut prev_recall = 0.0_f32;
        let mut prev_prec = 1.0_f32;

        for (_, label) in &pairs {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let prec = tp / (tp + fp);
            let recall = tp / n_pos;
            auc += (recall - prev_recall) * (prec + prev_prec) * 0.5;
            prev_recall = recall;
            prev_prec = prec;
        }
        Ok(auc.clamp(0.0, 1.0))
    }

    /// F1 score at a fixed threshold.
    pub fn f1_at_threshold(scores: &[f32], labels: &[bool], threshold: f32) -> AnomalyResult<f32> {
        Self::validate(scores, labels)?;
        let preds: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();
        let (tp, fp, fn_) = Self::tp_fp_fn(&preds, labels);
        let denom = 2.0 * tp as f32 + fp as f32 + fn_ as f32;
        if denom < 1e-12 {
            return Ok(0.0);
        }
        Ok(2.0 * tp as f32 / denom)
    }

    /// Point-adjusted F1: standard anomaly detection convention where an entire
    /// contiguous anomalous segment is counted as detected if at least one
    /// point in that segment exceeds the threshold.
    pub fn point_adjust_f1(scores: &[f32], labels: &[bool], threshold: f32) -> AnomalyResult<f32> {
        Self::validate(scores, labels)?;
        let n = scores.len();
        let raw_pred: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();

        // Find contiguous anomalous segments.
        let mut adjusted = raw_pred.clone();
        let mut i = 0_usize;
        while i < n {
            if labels[i] {
                // Start of anomalous segment.
                let mut j = i;
                while j < n && labels[j] {
                    j += 1;
                }
                // Check if any raw pred fires in [i, j).
                let segment_detected = raw_pred[i..j].iter().any(|&p| p);
                if segment_detected {
                    for k in i..j {
                        adjusted[k] = true;
                    }
                }
                i = j;
            } else {
                i += 1;
            }
        }

        let (tp, fp, fn_) = Self::tp_fp_fn(&adjusted, labels);
        let denom = 2.0 * tp as f32 + fp as f32 + fn_ as f32;
        if denom < 1e-12 {
            return Ok(0.0);
        }
        Ok(2.0 * tp as f32 / denom)
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn validate(scores: &[f32], labels: &[bool]) -> AnomalyResult<()> {
        if scores.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        if scores.len() != labels.len() {
            return Err(AnomalyError::DimensionMismatch {
                expected: scores.len(),
                found: labels.len(),
            });
        }
        Ok(())
    }

    fn tp_fp_fn(preds: &[bool], labels: &[bool]) -> (usize, usize, usize) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_ = 0;
        for (&p, &l) in preds.iter().zip(labels.iter()) {
            match (p, l) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        (tp, fp, fn_)
    }
}

// Extended f64-based anomaly detection methods
pub mod extended;
pub use extended::{
    compute_anomaly_metrics, compute_auc_roc, compute_average_precision, AdDeepSvdd,
    AdDeepSvddConfig, AdLinear, AdMlp, AeAnomaly, AeAnomalyConfig, AnomalyMetrics, CouplingLayer,
    FlowAnomaly, FlowAnomalyConfig, IsolationForestConfig, IsolationNode, IsolationTree,
    NeuralIsolationForest, VaeAnomaly, VaeAnomalyConfig,
};

#[cfg(test)]
mod tests;
