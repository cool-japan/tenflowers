//! Neural Data Compression — TenfloweRS.
//!
//! Implements learned data compression following Ballé et al. 2016/2018,
//! VQ-VAE (van den Oord 2017), and related information-theoretic approaches.
//! Distinct from `compression.rs` (model compression/pruning/quantization):
//! this module targets *content* compression — encoding signals/images into
//! compact latent representations using neural transforms + entropy coding.
//!
//! ## Key Papers
//! - Ballé et al. (2016) "End-to-end Optimized Image Compression"
//! - Ballé et al. (2018) "Variational Image Compression with a Scale Hyperprior"
//! - van den Oord et al. (2017) "Neural Discrete Representation Learning" (VQ-VAE)
//! - Zeghidour et al. (2021) "SoundStream: An End-to-End Neural Audio Codec"
//!
//! ## Architecture
//!
//! ```text
//! Input x ──► Encoder ──► Latent y ──► Entropy Model ──► Bitstream
//!                              │                              │
//!                              ▼                              ▼
//!                         Quantize ŷ ◄──── Arithmetic Decoder
//!                              │
//!                              ▼
//!                          Decoder ──► Reconstruction x̂
//! ```
//!
//! All components are prefix-`Nc` to avoid collisions with `compression.rs`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]

use std::fmt;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to the neural compression module.
#[derive(Debug, Clone, PartialEq)]
pub enum NcError {
    /// Dimension mismatch or out-of-bounds index.
    InvalidDimension(String),
    /// Numerical failure: NaN, divergence, singular matrix, etc.
    NumericalError(String),
    /// Invalid configuration parameter.
    ConfigError(String),
}

impl fmt::Display for NcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NcError::InvalidDimension(m) => write!(f, "NcError::InvalidDimension: {}", m),
            NcError::NumericalError(m) => write!(f, "NcError::NumericalError: {}", m),
            NcError::ConfigError(m) => write!(f, "NcError::ConfigError: {}", m),
        }
    }
}

impl std::error::Error for NcError {}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers (no ndarray, no rand, pure f64 arithmetic)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard normal CDF via the Abramowitz & Stegun erf approximation.
/// Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun series 7.1.26 approximation to erf(x).
/// Maximum error < 1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

/// log(Φ(x)) in a numerically stable way.
#[inline]
#[allow(dead_code)]
fn log_normal_cdf(x: f64) -> f64 {
    let p = standard_normal_cdf(x);
    if p <= 0.0 {
        -1e30
    } else {
        p.ln()
    }
}

/// Softplus: log(1 + exp(x)), clamped for stability.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// LeakyReLU with slope 0.01 for negative inputs.
#[inline]
fn leaky_relu(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        0.01 * x
    }
}

/// Sigmoid σ(x) = 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Matrix-vector product: y = W @ x + b where W is [out × in].
fn matvec(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    let out_dim = w.len();
    let mut y = vec![0.0_f64; out_dim];
    for (i, (row, bias)) in w.iter().zip(b.iter()).enumerate() {
        let mut acc = *bias;
        for (wij, xj) in row.iter().zip(x.iter()) {
            acc += wij * xj;
        }
        y[i] = acc;
    }
    y
}

/// Xavier uniform initialisation: U(-limit, +limit) where limit = sqrt(6/(fan_in+fan_out)).
/// Deterministic LCG seeded with (fan_in, fan_out, row, col) for reproducibility.
fn xavier_weight(fan_in: usize, fan_out: usize, row: usize, col: usize) -> f64 {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    // Deterministic pseudo-random via LCG
    let seed: u64 = (fan_in as u64)
        .wrapping_mul(2654435761)
        .wrapping_add((fan_out as u64).wrapping_mul(2246822519))
        .wrapping_add((row as u64).wrapping_mul(2246822530))
        .wrapping_add((col as u64).wrapping_mul(1234567891));
    let lcg = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to [-1, 1]
    let normalized = (lcg as f64 / u64::MAX as f64) * 2.0 - 1.0;
    normalized * limit
}

/// Construct weight matrix [out_dim × in_dim] with Xavier init.
fn make_weight_matrix(in_dim: usize, out_dim: usize) -> Vec<Vec<f64>> {
    (0..out_dim)
        .map(|r| {
            (0..in_dim)
                .map(|c| xavier_weight(in_dim, out_dim, r, c))
                .collect()
        })
        .collect()
}

/// Compute mean and standard deviation of a 1-D slice.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 1.0);
    }
    let n = xs.len() as f64;
    let mu = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n;
    (mu, var.sqrt().max(1e-8))
}

/// Round towards nearest integer (ties go to nearest even — but for simplicity: round half up).
#[inline]
fn round_f64(x: f64) -> f64 {
    x.round()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  NcEntropyModel — Factorized entropy model (Ballé 2016)
// ─────────────────────────────────────────────────────────────────────────────

/// Factorized entropy model for learned compression (Ballé et al. 2016).
///
/// Models the marginal distribution of each latent channel independently:
/// `p(y) = ∏_i p(y_i)` where each marginal is approximated as a Gaussian CDF
/// difference (box likelihood centred on the quantisation bin).
///
/// During inference: hard quantise to integers and look up the CDF.
/// During training: use a continuous relaxation (soft quantisation).
#[derive(Debug, Clone)]
pub struct NcEntropyModel {
    /// Learned mean (location) for each latent channel.
    pub means: Vec<f64>,
    /// Learned log-scale for each latent channel (sigma = exp(log_scale)).
    pub log_scales: Vec<f64>,
    /// Number of latent channels.
    pub n_channels: usize,
}

impl NcEntropyModel {
    /// Create a new entropy model with zero means and log-scale = 0 (sigma = 1).
    pub fn new(n_channels: usize) -> Self {
        Self {
            means: vec![0.0; n_channels],
            log_scales: vec![0.0; n_channels],
            n_channels,
        }
    }

    /// Log-probability of latent vector `y` under the factorised entropy model.
    ///
    /// Uses the box likelihood: `log p(y_i) = log[Φ((y_i + 0.5 - μ_i)/σ_i) − Φ((y_i − 0.5 − μ_i)/σ_i)]`
    /// where Φ is the standard-normal CDF. Falls back to a continuous approximation
    /// for very large sigma (numerical safety).
    pub fn log_prob(&self, y: &[f64]) -> f64 {
        let len = y.len().min(self.n_channels);
        let mut log_p = 0.0_f64;
        for i in 0..len {
            let mu = self.means[i];
            let sigma = self.log_scales[i].exp().max(1e-8);
            let upper = (y[i] + 0.5 - mu) / sigma;
            let lower = (y[i] - 0.5 - mu) / sigma;
            let p_upper = standard_normal_cdf(upper);
            let p_lower = standard_normal_cdf(lower);
            let bin_prob = (p_upper - p_lower).max(1e-30);
            log_p += bin_prob.ln();
        }
        log_p
    }

    /// Information content of `y` in bits: `-log₂ p(y) = -log p(y) / log(2)`.
    /// Returned value is always ≥ 0.
    pub fn bits_per_sample(&self, y: &[f64]) -> f64 {
        (-self.log_prob(y) / std::f64::consts::LN_2).max(0.0)
    }

    /// Hard quantisation: round each element to the nearest integer.
    pub fn quantize(&self, y: &[f64]) -> Vec<i32> {
        y.iter().map(|&v| round_f64(v) as i32).collect()
    }

    /// Soft quantisation for training (differentiable approximation).
    ///
    /// Uses the identity: `ŷ = y + τ · sin(2π y) / (2π)`
    /// which smoothly interpolates between `y` (τ=0, identity) and the
    /// "straight-through" regime (τ → 1 gives a periodic correction).
    ///
    /// This is a numerically convenient surrogate for the noise-based
    /// `y + U(−0.5, 0.5)` used in Ballé 2016 training.
    pub fn soft_quantize(&self, y: &[f64], tau: f64) -> Vec<f64> {
        use std::f64::consts::TAU; // 2π
        y.iter().map(|&v| v + tau * (TAU * v).sin() / TAU).collect()
    }

    /// M-step update: set μ_i = E\[y_i\] and log σ_i = log std(y_i) over the batch.
    pub fn update_from_data(&mut self, y_batch: &[Vec<f64>], _lr: f64) {
        for ch in 0..self.n_channels {
            let col: Vec<f64> = y_batch
                .iter()
                .filter_map(|row| row.get(ch).copied())
                .collect();
            if !col.is_empty() {
                let (mu, sigma) = mean_std(&col);
                self.means[ch] = mu;
                self.log_scales[ch] = sigma.ln();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  NcHyperprior — Scale hyperprior (Ballé 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Scale hyperprior for conditional entropy coding (Ballé et al. 2018).
///
/// Introduces side-information `z` that captures the standard deviation of
/// each latent component:
/// - `z = |enc_hyper(y)|`  — hyper-encoder extracts scale information
/// - `σ(z) = softplus(dec_hyper(z))`  — hyper-decoder predicts per-element scales
/// - `p(y | z) = ∏_i N(y_i ; 0, σ_i²)`  — conditional likelihood
/// - `p(z)` modelled by `NcEntropyModel`  — factorised marginal of side-info
#[derive(Debug, Clone)]
pub struct NcHyperprior {
    /// Hyper-encoder weight matrix [hyper_dim × latent_dim].
    pub hyper_encoder_w: Vec<Vec<f64>>,
    /// Hyper-encoder bias \[hyper_dim\].
    pub hyper_encoder_b: Vec<f64>,
    /// Hyper-decoder weight matrix [latent_dim × hyper_dim].
    pub hyper_decoder_w: Vec<Vec<f64>>,
    /// Hyper-decoder bias \[latent_dim\].
    pub hyper_decoder_b: Vec<f64>,
    /// Factorised entropy model for the side information `z`.
    pub hyper_entropy: NcEntropyModel,
    /// Dimension of the main latent `y`.
    pub latent_dim: usize,
    /// Dimension of the hyper-latent `z`.
    pub hyper_dim: usize,
}

impl NcHyperprior {
    /// Create a new hyperprior with Xavier-initialised MLP weights.
    pub fn new(latent_dim: usize, hyper_dim: usize) -> Self {
        let hyper_encoder_w = make_weight_matrix(latent_dim, hyper_dim);
        let hyper_encoder_b = vec![0.0; hyper_dim];
        let hyper_decoder_w = make_weight_matrix(hyper_dim, latent_dim);
        let hyper_decoder_b = vec![0.0; latent_dim];
        Self {
            hyper_encoder_w,
            hyper_encoder_b,
            hyper_decoder_w,
            hyper_decoder_b,
            hyper_entropy: NcEntropyModel::new(hyper_dim),
            latent_dim,
            hyper_dim,
        }
    }

    /// Encode `y` to hyper-latent `z`: `z = ReLU(W_he @ |y| + b_he)`.
    ///
    /// Uses absolute value of `y` to capture scale information (as in Ballé 2018).
    pub fn encode_hyper(&self, y: &[f64]) -> Vec<f64> {
        let abs_y: Vec<f64> = y.iter().map(|v| v.abs()).collect();
        let pre = matvec(&self.hyper_encoder_w, &self.hyper_encoder_b, &abs_y);
        pre.iter().map(|&v| v.max(0.0)).collect() // ReLU
    }

    /// Decode hyper-latent `z` to per-element log-scales:
    /// `log_sigma = softplus(W_hd @ z + b_hd)`.
    ///
    /// Returns log-scale values (natural log). Caller can take exp() to get sigma.
    pub fn decode_hyper(&self, z: &[f64]) -> Vec<f64> {
        let pre = matvec(&self.hyper_decoder_w, &self.hyper_decoder_b, z);
        pre.iter().map(|&v| softplus(v)).collect()
    }

    /// Conditional log-probability `log p(y | z)` using predicted scales.
    ///
    /// `sigma` is the vector returned by `decode_hyper`. Treats each y_i as
    /// conditionally Gaussian with mean 0 and standard deviation sigma_i.
    pub fn conditional_log_prob(&self, y: &[f64], sigma: &[f64]) -> f64 {
        let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
        y.iter()
            .zip(sigma.iter())
            .map(|(&yi, &si)| {
                let s = si.max(1e-8);
                -0.5 * (yi / s).powi(2) - s.ln() - ln_sqrt_2pi
            })
            .sum()
    }

    /// Rate of the hyper-latent in bits: `-log₂ p(z)`.
    pub fn hyper_rate(&self, z: &[f64]) -> f64 {
        (-self.hyper_entropy.log_prob(z) / std::f64::consts::LN_2).max(0.0)
    }

    /// Estimate total rate `H(y) + H(z)` using the scale hyperprior.
    ///
    /// The conditional rate `H(y|z)` is computed using the scales predicted from `z`.
    pub fn total_rate(&self, y: &[f64]) -> f64 {
        let z = self.encode_hyper(y);
        let sigma = self.decode_hyper(&z);
        let h_y_given_z = (-self.conditional_log_prob(y, &sigma) / std::f64::consts::LN_2).max(0.0);
        let h_z = self.hyper_rate(&z);
        h_y_given_z + h_z
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  NcRateDistortionLoss — RD trade-off (λ D + R)
// ─────────────────────────────────────────────────────────────────────────────

/// Rate-distortion loss: `L = D + λ · R` as used in Ballé et al.
///
/// The distortion D is MSE scaled by 255² (standard in image compression
/// when pixel values are in [0, 1]). The rate R is measured in bits.
#[derive(Debug, Clone)]
pub struct NcRateDistortionLoss {
    /// Lagrange multiplier balancing distortion and rate.
    pub lambda: f64,
}

impl NcRateDistortionLoss {
    /// Create a new RD loss with the given lambda.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Distortion: MSE × 255² (standardised for \[0,1\]-normalised image data).
    pub fn distortion(&self, original: &[f64], reconstructed: &[f64]) -> f64 {
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return 0.0;
        }
        let mse: f64 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        mse * 255.0_f64.powi(2)
    }

    /// Rate of the latent code in bits per element.
    pub fn rate(&self, entropy_model: &NcEntropyModel, latent: &[f64]) -> f64 {
        entropy_model.bits_per_sample(latent) / latent.len().max(1) as f64
    }

    /// Combined rate-distortion loss: `L = MSE·255² + λ · rate_bits`.
    pub fn loss(&self, original: &[f64], reconstructed: &[f64], rate_bits: f64) -> f64 {
        self.distortion(original, reconstructed) + self.lambda * rate_bits
    }

    /// Peak Signal-to-Noise Ratio: `PSNR = 10 · log₁₀(255² / MSE)`.
    pub fn psnr(&self, original: &[f64], reconstructed: &[f64]) -> f64 {
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return 0.0;
        }
        let mse: f64 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        if mse <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0_f64.powi(2) / mse).log10()
    }

    /// Bits per pixel: `rate_bits / n_pixels`.
    pub fn bpp(rate_bits: f64, n_pixels: usize) -> f64 {
        if n_pixels == 0 {
            return 0.0;
        }
        rate_bits / n_pixels as f64
    }

    /// Build a rate-distortion curve: returns (bpp, psnr) pairs.
    ///
    /// `reconstructed_at_lambdas` is a slice of `(lambda, reconstruction, rate_bits)`.
    pub fn rate_distortion_curve(
        &self,
        original: &[f64],
        reconstructed_at_lambdas: &[(f64, Vec<f64>, f64)],
    ) -> Vec<(f64, f64)> {
        let n_pixels = original.len();
        reconstructed_at_lambdas
            .iter()
            .map(|(_lam, recon, rate)| {
                let bpp = Self::bpp(*rate, n_pixels);
                let psnr = self.psnr(original, recon);
                (bpp, psnr)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  NcAutoencoder — Learned transform coding autoencoder
// ─────────────────────────────────────────────────────────────────────────────

/// Learned autoencoder for transform coding.
///
/// Architecture (Ballé 2016 style MLP variant):
/// - Encoder: `y = W₂ · LeakyReLU(W₁ · x + b₁) + b₂`
/// - Decoder: `x̂ = sigmoid(W₄ · LeakyReLU(W₃ · y + b₃) + b₄)`
///
/// The sigmoid output maps to [0, 1], appropriate for normalised signal data.
#[derive(Debug, Clone)]
pub struct NcAutoencoder {
    enc_w1: Vec<Vec<f64>>,
    enc_b1: Vec<f64>,
    enc_w2: Vec<Vec<f64>>,
    enc_b2: Vec<f64>,
    dec_w1: Vec<Vec<f64>>,
    dec_b1: Vec<f64>,
    dec_w2: Vec<Vec<f64>>,
    dec_b2: Vec<f64>,
    /// Input dimensionality.
    pub input_dim: usize,
    /// Latent space dimensionality.
    pub latent_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Per-channel entropy model for the latent.
    pub entropy_model: NcEntropyModel,
}

impl NcAutoencoder {
    /// Create a new autoencoder with Xavier-initialised weights.
    pub fn new(input_dim: usize, latent_dim: usize, hidden_dim: usize) -> Self {
        Self {
            enc_w1: make_weight_matrix(input_dim, hidden_dim),
            enc_b1: vec![0.0; hidden_dim],
            enc_w2: make_weight_matrix(hidden_dim, latent_dim),
            enc_b2: vec![0.0; latent_dim],
            dec_w1: make_weight_matrix(latent_dim, hidden_dim),
            dec_b1: vec![0.0; hidden_dim],
            dec_w2: make_weight_matrix(hidden_dim, input_dim),
            dec_b2: vec![0.0; input_dim],
            input_dim,
            latent_dim,
            hidden_dim,
            entropy_model: NcEntropyModel::new(latent_dim),
        }
    }

    /// Encode input `x` to latent `y`.
    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        let h1 = matvec(&self.enc_w1, &self.enc_b1, x);
        let h1_act: Vec<f64> = h1.iter().map(|&v| leaky_relu(v)).collect();
        matvec(&self.enc_w2, &self.enc_b2, &h1_act)
    }

    /// Decode a batch of latents `y_hat` to reconstructions.
    ///
    /// Each element of `y_hat` is a latent vector; the method returns a
    /// matching vector of reconstructed inputs in [0, 1].
    pub fn decode(&self, y_hat: &[Vec<f64>]) -> Vec<Vec<f64>> {
        y_hat
            .iter()
            .map(|y| {
                let h1 = matvec(&self.dec_w1, &self.dec_b1, y);
                let h1_act: Vec<f64> = h1.iter().map(|&v| leaky_relu(v)).collect();
                let out = matvec(&self.dec_w2, &self.dec_b2, &h1_act);
                out.iter().map(|&v| sigmoid(v)).collect()
            })
            .collect()
    }

    /// Compress `x` to an integer code: returns `(quantized_latent, rate_in_bits)`.
    pub fn compress(&self, x: &[f64]) -> (Vec<i32>, f64) {
        let y = self.encode(x);
        let rate = self.entropy_model.bits_per_sample(&y);
        let quantized = self.entropy_model.quantize(&y);
        (quantized, rate)
    }

    /// Decode from quantized integer latent.
    pub fn decompress(&self, quantized: &[i32]) -> Vec<f64> {
        let y_f: Vec<f64> = quantized.iter().map(|&q| q as f64).collect();
        let batch = vec![y_f];
        let recons = self.decode(&batch);
        recons.into_iter().next().unwrap_or_default()
    }

    /// Rate-distortion loss over a batch.
    ///
    /// `L = mean_i [MSE(x_i, x̂_i) + λ · bits(y_i)]`
    pub fn rd_loss(&self, x_batch: &[Vec<f64>], lambda: f64) -> f64 {
        if x_batch.is_empty() {
            return 0.0;
        }
        let rd_loss_obj = NcRateDistortionLoss::new(lambda);
        let total: f64 = x_batch
            .iter()
            .map(|x| {
                let y = self.encode(x);
                let rate = self.entropy_model.bits_per_sample(&y);
                let y_hat_batch = vec![y.iter().map(|&v| round_f64(v)).collect::<Vec<_>>()];
                let x_hat = self.decode(&y_hat_batch);
                let x_hat_flat = x_hat.into_iter().next().unwrap_or_default();
                let distortion = rd_loss_obj.distortion(x, &x_hat_flat);
                distortion + lambda * rate
            })
            .sum();
        total / x_batch.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  NcVectorQuantizer — VQ-VAE codebook (van den Oord 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Vector quantizer implementing the VQ-VAE discrete codebook.
///
/// Uses the straight-through estimator during training:
/// - Forward: `ẑ = e_{k*}` where `k* = argmin_k ||z - e_k||₂²`
/// - VQ loss: `||sg[z] - e_k*||² + β·||z - sg[e_k*]||²`
/// - EMA update: `e_k ← γ·e_k + (1-γ)·z`
#[derive(Debug, Clone)]
pub struct NcVectorQuantizer {
    /// Codebook: `K` entries each of dimension `D`.
    pub codebook: Vec<Vec<f64>>,
    /// Number of codebook entries.
    pub k_size: usize,
    /// Codebook entry dimensionality.
    pub d_size: usize,
    /// Commitment loss coefficient β (van den Oord 2017 used 0.25).
    pub commitment_beta: f64,
    /// Per-codeword usage counter (resets with dead-code replacement).
    pub usage_counts: Vec<usize>,
}

impl NcVectorQuantizer {
    /// Create a new vector quantizer with K-means++ inspired spread initialisation.
    ///
    /// Codewords are placed at deterministic pseudo-random positions scaled
    /// to a unit hypersphere for diverse initialisation.
    #[allow(non_snake_case)]
    pub fn new(K: usize, D: usize, beta: f64) -> Self {
        // Deterministic spread initialisation
        let mut codebook = Vec::with_capacity(K);
        for k in 0..K {
            let mut entry = Vec::with_capacity(D);
            let mut norm_sq = 0.0_f64;
            for d in 0..D {
                let seed: u64 = ((k as u64).wrapping_mul(2654435761))
                    .wrapping_add((d as u64).wrapping_mul(2246822519))
                    .wrapping_add(1234567891);
                let lcg = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (lcg as f64 / u64::MAX as f64) * 2.0 - 1.0;
                entry.push(v);
                norm_sq += v * v;
            }
            // Normalise to unit sphere, then scale by sqrt(D/K) for spread
            let scale = (D as f64 / K as f64).sqrt() / norm_sq.sqrt().max(1e-10);
            for v in entry.iter_mut() {
                *v *= scale;
            }
            codebook.push(entry);
        }
        Self {
            codebook,
            k_size: K,
            d_size: D,
            commitment_beta: beta,
            usage_counts: vec![0; K],
        }
    }

    /// Quantise a single vector `z` to the nearest codeword.
    ///
    /// Returns `(quantized_z, codebook_index, commitment_loss)` where
    /// `commitment_loss = ||z - e_{k*}||²`.
    pub fn quantize(&self, z: &[f64]) -> (Vec<f64>, usize, f64) {
        let mut best_k = 0usize;
        let mut best_dist = f64::INFINITY;
        for (k, ek) in self.codebook.iter().enumerate() {
            let dist: f64 = z
                .iter()
                .zip(ek.iter())
                .map(|(zi, ei)| (zi - ei).powi(2))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }
        let e_k = self.codebook[best_k].clone();
        let commitment_loss: f64 = z
            .iter()
            .zip(e_k.iter())
            .map(|(zi, ei)| (zi - ei).powi(2))
            .sum();
        (e_k, best_k, commitment_loss)
    }

    /// Quantise a batch of vectors.
    pub fn quantize_batch(&self, z_batch: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
        let mut all_q = Vec::with_capacity(z_batch.len());
        let mut all_idx = Vec::with_capacity(z_batch.len());
        let mut total_loss = 0.0_f64;
        for z in z_batch {
            let (q, idx, loss) = self.quantize(z);
            all_q.push(q);
            all_idx.push(idx);
            total_loss += loss;
        }
        let avg_loss = if z_batch.is_empty() {
            0.0
        } else {
            total_loss / z_batch.len() as f64
        };
        (all_q, all_idx, avg_loss)
    }

    /// Full VQ loss:
    /// `L_vq = ||sg[z] - e_k*||² + β · ||z - sg[e_k*]||²`
    ///
    /// Here `sg[·]` denotes stop-gradient (constant). In forward-only mode:
    /// both terms evaluate to the same value `||z - e_k*||²`, so we weight
    /// the commitment term by β.
    pub fn vq_loss(&self, z: &[f64], e_k: &[f64]) -> f64 {
        let dist_sq: f64 = z
            .iter()
            .zip(e_k.iter())
            .map(|(zi, ei)| (zi - ei).powi(2))
            .sum();
        // codebook_loss = ||sg[z] - e_k||^2 = dist_sq (treating z as constant)
        // commitment_loss = beta * ||z - sg[e_k]||^2 = beta * dist_sq
        dist_sq + self.commitment_beta * dist_sq
    }

    /// Exponential moving average update: `e_k ← γ · e_k + (1−γ) · z`.
    pub fn update_codebook_ema(&mut self, z: &[f64], k: usize, decay: f64) {
        if k >= self.k_size {
            return;
        }
        let one_minus_decay = 1.0 - decay;
        for (ei, zi) in self.codebook[k].iter_mut().zip(z.iter()) {
            *ei = decay * *ei + one_minus_decay * zi;
        }
        self.usage_counts[k] = self.usage_counts[k].saturating_add(1);
    }

    /// Codebook perplexity: `exp(H) = exp(-∑_k p_k log p_k)`.
    ///
    /// A perplexity of K means all codewords are used equally; values << K
    /// indicate codebook collapse.
    pub fn codebook_perplexity(&self) -> f64 {
        let total: usize = self.usage_counts.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let total_f = total as f64;
        let entropy: f64 = self
            .usage_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_f;
                -p * p.ln()
            })
            .sum();
        entropy.exp().max(1.0)
    }

    /// Reset codebook entries with usage count below `threshold` to
    /// perturbed versions of the most-used entry (codebook revival).
    pub fn reset_dead_codes(&mut self, threshold: usize) {
        // Find the most-used entry
        let best_k = self
            .usage_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .map(|(k, _)| k)
            .unwrap_or(0);
        let best_entry = self.codebook[best_k].clone();
        for k in 0..self.k_size {
            if self.usage_counts[k] < threshold {
                // Reinitialise with small perturbation of the best entry
                let perturb: Vec<f64> = best_entry
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| {
                        let seed: u64 = ((k as u64).wrapping_mul(2654435769))
                            .wrapping_add((d as u64).wrapping_mul(2246822577));
                        let lcg = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let noise = (lcg as f64 / u64::MAX as f64) * 0.02 - 0.01;
                        v + noise
                    })
                    .collect();
                self.codebook[k] = perturb;
                self.usage_counts[k] = 0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  NcResidualQuantizer — Cascaded VQ stages (SoundStream / EnCodec style)
// ─────────────────────────────────────────────────────────────────────────────

/// Residual vector quantizer: cascade of `n_stages` VQ layers.
///
/// Each stage quantises the residual error from previous stages:
/// ```text
/// r_0 = z
/// ẑ_1, k_1 = VQ_1(r_0)
/// r_1 = r_0 - ẑ_1
/// ẑ_2, k_2 = VQ_2(r_1)
/// r_2 = r_1 - ẑ_2  ...
/// ẑ_final = ẑ_1 + ẑ_2 + ... + ẑ_n
/// ```
///
/// Used in neural audio codecs (SoundStream, EnCodec) and image compression.
#[derive(Debug, Clone)]
pub struct NcResidualQuantizer {
    stages: Vec<NcVectorQuantizer>,
    /// Number of quantization stages.
    pub n_stages: usize,
    /// Codebook size per stage.
    pub k_size: usize,
    /// Dimension of each entry.
    pub d_size: usize,
}

impl NcResidualQuantizer {
    /// Create a cascaded residual quantizer.
    #[allow(non_snake_case)]
    pub fn new(n_stages: usize, K: usize, D: usize, beta: f64) -> Self {
        let stages = (0..n_stages)
            .map(|_| NcVectorQuantizer::new(K, D, beta))
            .collect();
        Self {
            stages,
            n_stages,
            k_size: K,
            d_size: D,
        }
    }

    /// Perform residual quantization.
    ///
    /// Returns `(quantized_total, indices_per_stage, total_vq_loss)`.
    pub fn quantize(&self, z: &[f64]) -> (Vec<f64>, Vec<usize>, f64) {
        let mut residual: Vec<f64> = z.to_vec();
        let mut quantized_total: Vec<f64> = vec![0.0; z.len()];
        let mut indices = Vec::with_capacity(self.n_stages);
        let mut total_loss = 0.0_f64;

        for stage in &self.stages {
            let (q, k, loss) = stage.quantize(&residual);
            // Update quantized total and residual
            for (qt, qi) in quantized_total.iter_mut().zip(q.iter()) {
                *qt += qi;
            }
            residual = residual.iter().zip(q.iter()).map(|(r, q)| r - q).collect();
            indices.push(k);
            total_loss += loss;
        }
        (quantized_total, indices, total_loss)
    }

    /// Bits per step (log₂ K per stage).
    pub fn bits_per_step(&self) -> f64 {
        if self.k_size <= 1 {
            return 0.0;
        }
        (self.k_size as f64).log2()
    }

    /// Total bits: `n_stages × log₂(K)`.
    pub fn total_bits(&self) -> f64 {
        self.n_stages as f64 * self.bits_per_step()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  NcArithmeticCoder — Conceptual ANS-inspired arithmetic coding
// ─────────────────────────────────────────────────────────────────────────────

/// Conceptual arithmetic coder (approximation of entropy coding).
///
/// In a production system this would implement ANS (rANS/tANS) or classical
/// arithmetic coding. Here we implement the key theoretical quantities:
/// - Shannon entropy bound
/// - Per-symbol encode/decode approximation
///
/// The encode/decode functions use a simple probability-interval approximation
/// that captures the correct expected code length but is not a lossless coder.
#[derive(Debug, Clone)]
pub struct NcArithmeticCoder {
    /// Number of bits of precision for probability intervals.
    precision_bits: usize,
}

impl NcArithmeticCoder {
    /// Create a new arithmetic coder with given precision.
    pub fn new(precision_bits: usize) -> Self {
        Self {
            precision_bits: precision_bits.max(1),
        }
    }

    /// Encode a symbol given its probability `p`.
    ///
    /// Conceptual approximation: map symbol to an interval code.
    /// `code ≈ floor(symbol_magnitude / p)` scaled to `2^precision_bits`.
    pub fn encode_symbol(&self, symbol: i32, prob: f64) -> u64 {
        let p = prob.clamp(1e-15, 1.0);
        let scale = (1_u64 << self.precision_bits) as f64;
        // Approximation of arithmetic coding interval: symbol / p mapped to code space
        
        (symbol.unsigned_abs() as f64 / p * scale) as u64
    }

    /// Decode a symbol from its encoded form given probability `p`.
    ///
    /// `symbol = floor(code * p / 2^precision_bits)`.
    pub fn decode_symbol(&self, code: u64, prob: f64) -> i32 {
        let p = prob.clamp(1e-15, 1.0);
        let scale = (1_u64 << self.precision_bits) as f64;
        (code as f64 * p / scale) as i32
    }

    /// Theoretical minimum average code length (Shannon entropy): `-∑_i p_i log₂(p_i)`.
    pub fn expected_code_length(&self, probs: &[f64]) -> f64 {
        probs
            .iter()
            .map(|&p| {
                let p = p.clamp(1e-30, 1.0);
                -p * p.log2()
            })
            .sum::<f64>()
            .max(0.0)
    }

    /// Encode a sequence of symbols.
    pub fn encode_sequence(&self, symbols: &[i32], probs: &[f64]) -> Vec<u64> {
        symbols
            .iter()
            .zip(probs.iter())
            .map(|(&s, &p)| self.encode_symbol(s, p))
            .collect()
    }

    /// Decode a sequence of codes.
    pub fn decode_sequence(&self, codes: &[u64], probs: &[f64]) -> Vec<i32> {
        codes
            .iter()
            .zip(probs.iter())
            .map(|(&c, &p)| self.decode_symbol(c, p))
            .collect()
    }

    /// Overhead ratio: how many more bits were used vs. Shannon entropy.
    ///
    /// Values close to 1.0 indicate near-optimal coding.
    pub fn overhead_ratio(coded_bits: usize, entropy_bits: f64) -> f64 {
        if entropy_bits <= 0.0 {
            return 1.0;
        }
        coded_bits as f64 / entropy_bits
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  NcPatchCompressor — Patch-based compression for 1D signals
// ─────────────────────────────────────────────────────────────────────────────

/// Patch-based compressor: divides a signal into overlapping patches,
/// compresses each patch independently, then overlap-adds for reconstruction.
///
/// Suitable for audio waveforms, 1D sensor data, or flattened image rows.
#[derive(Debug, Clone)]
pub struct NcPatchCompressor {
    autoencoder: NcAutoencoder,
    /// Patch width in samples.
    pub patch_size: usize,
    /// Hop length between consecutive patches.
    pub stride: usize,
}

impl NcPatchCompressor {
    /// Create a new patch compressor.
    pub fn new(patch_size: usize, stride: usize, latent_dim: usize, hidden_dim: usize) -> Self {
        let effective_stride = stride.max(1);
        let effective_patch = patch_size.max(1);
        Self {
            autoencoder: NcAutoencoder::new(effective_patch, latent_dim, hidden_dim),
            patch_size: effective_patch,
            stride: effective_stride,
        }
    }

    /// Extract overlapping 1D patches from `signal` with the configured stride.
    pub fn extract_patches(&self, signal: &[f64], _signal_len: usize) -> Vec<Vec<f64>> {
        let n = signal.len();
        if n < self.patch_size {
            return vec![signal.to_vec()];
        }
        let mut patches = Vec::new();
        let mut start = 0;
        while start + self.patch_size <= n {
            patches.push(signal[start..start + self.patch_size].to_vec());
            start += self.stride;
        }
        // Include a final partial patch if necessary (zero-padded)
        if start < n {
            let mut last_patch = vec![0.0_f64; self.patch_size];
            last_patch[..n - start].copy_from_slice(&signal[start..]);
            patches.push(last_patch);
        }
        patches
    }

    /// Compress a 1D signal: returns `(quantized_patches, total_rate_bits)`.
    pub fn compress_signal(&self, signal: &[f64]) -> (Vec<Vec<i32>>, f64) {
        let patches = self.extract_patches(signal, signal.len());
        let mut total_rate = 0.0_f64;
        let quantized: Vec<Vec<i32>> = patches
            .iter()
            .map(|p| {
                let (q, rate) = self.autoencoder.compress(p);
                total_rate += rate;
                q
            })
            .collect();
        (quantized, total_rate)
    }

    /// Reconstruct a signal from quantized patches using overlap-add.
    pub fn decompress_signal(&self, patches: &[Vec<i32>], original_len: usize) -> Vec<f64> {
        if patches.is_empty() || original_len == 0 {
            return vec![0.0; original_len];
        }
        let mut output = vec![0.0_f64; original_len];
        let mut weights = vec![0.0_f64; original_len];

        let mut start = 0;
        for patch in patches {
            let recon = self.autoencoder.decompress(patch);
            let end = (start + self.patch_size).min(original_len);
            let len = end - start;
            for (i, &v) in recon.iter().take(len).enumerate() {
                output[start + i] += v;
                weights[start + i] += 1.0;
            }
            start += self.stride;
            if start >= original_len {
                break;
            }
        }
        // Normalise by overlap count
        for (o, w) in output.iter_mut().zip(weights.iter()) {
            if *w > 0.0 {
                *o /= w;
            }
        }
        output
    }

    /// Compression ratio: `original_bits / compressed_bits`.
    pub fn compression_ratio(&self, original_bits: usize, compressed_bits: usize) -> f64 {
        if compressed_bits == 0 {
            return 0.0;
        }
        original_bits as f64 / compressed_bits as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  NcProgressiveCoder — Multi-level scalable coding
// ─────────────────────────────────────────────────────────────────────────────

/// Progressive / scalable coder supporting multiple quality levels.
///
/// Layer 0 provides base quality; each subsequent layer refines the
/// representation. This models SNR/spatial scalability in HEVC/VVC but
/// realised through cascaded learned autoencoders.
///
/// The lambda values decrease across levels (smaller lambda = lower distortion
/// tolerance = higher quality).
#[derive(Debug, Clone)]
pub struct NcProgressiveCoder {
    coders: Vec<NcAutoencoder>,
    /// Number of quality levels.
    pub n_levels: usize,
    #[allow(dead_code)]
    lambdas: Vec<f64>,
}

impl NcProgressiveCoder {
    /// Create a new progressive coder with `n_levels` quality layers.
    ///
    /// Lambda values are spaced logarithmically from 0.1 (high quality) to 10.0 (low quality).
    pub fn new(n_levels: usize, input_dim: usize, latent_dim: usize) -> Self {
        let n = n_levels.max(1);
        let hidden_dim = (input_dim * 2).max(8);
        let coders: Vec<NcAutoencoder> = (0..n)
            .map(|_| NcAutoencoder::new(input_dim, latent_dim, hidden_dim))
            .collect();
        // Log-spaced lambdas: level 0 = high quality (small lambda), level n-1 = low quality
        let lambdas: Vec<f64> = (0..n)
            .map(|i| {
                let t = if n == 1 {
                    0.5
                } else {
                    i as f64 / (n - 1) as f64
                };
                // Interpolate log10(0.01) to log10(10.0) → [0.01, 10.0]
                10.0_f64.powf(-2.0 + 3.0 * t)
            })
            .collect();
        Self {
            coders,
            n_levels: n,
            lambdas,
        }
    }

    /// Encode input at all quality levels.
    ///
    /// Returns a `Vec` of `(quantized_latent, rate_bits)` per level.
    pub fn encode_all_levels(&self, x: &[f64]) -> Vec<(Vec<i32>, f64)> {
        self.coders.iter().map(|coder| coder.compress(x)).collect()
    }

    /// Decode at a specific quality level from quantized patches.
    pub fn decode_at_level(&self, level: usize, quantized_patches: &[Vec<i32>]) -> Vec<Vec<f64>> {
        let level = level.min(self.n_levels - 1);
        quantized_patches
            .iter()
            .map(|q| self.coders[level].decompress(q))
            .collect()
    }

    /// Cumulative rate up to and including `level`.
    pub fn rate_at_level(&self, level: usize, quantized_patches: &[Vec<i32>]) -> f64 {
        let level = level.min(self.n_levels - 1);
        let mut total = 0.0_f64;
        for l in 0..=level {
            for q in quantized_patches {
                let y: Vec<f64> = q.iter().map(|&v| v as f64).collect();
                total += self.coders[l].entropy_model.bits_per_sample(&y);
            }
        }
        total
    }

    /// Build the quality ladder: returns `(bpp, psnr)` per level.
    ///
    /// `bpp` uses the number of elements in `x` as a proxy for "pixels".
    pub fn quality_ladder(&self, x: &[f64]) -> Vec<(f64, f64)> {
        let n_elements = x.len().max(1);
        let rd = NcRateDistortionLoss::new(1.0);
        self.coders
            .iter()
            .map(|coder| {
                let (q, rate) = coder.compress(x);
                let x_hat = coder.decompress(&q);
                let bpp = NcRateDistortionLoss::bpp(rate, n_elements);
                let psnr = rd.psnr(x, &x_hat);
                (bpp, psnr)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  NcMetrics — Compression quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Quality and compression efficiency metrics for learned compression systems.
///
/// Covers distortion (MSE, PSNR, SSIM), rate (BPP), and comparative
/// benchmarks (BD-Rate, Bjøntegaard delta).
pub struct NcMetrics;

impl NcMetrics {
    /// Mean squared error between `original` and `reconstructed`.
    pub fn mse(original: &[f64], reconstructed: &[f64]) -> f64 {
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return 0.0;
        }
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64
    }

    /// Peak Signal-to-Noise Ratio: `10 · log₁₀(max_val² / MSE)`.
    pub fn psnr(original: &[f64], reconstructed: &[f64], max_val: f64) -> f64 {
        let mse = Self::mse(original, reconstructed);
        if mse <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (max_val.powi(2) / mse).log10()
    }

    /// Simplified structural similarity index (SSIM).
    ///
    /// Computes SSIM over the full signal rather than local windows:
    /// `SSIM = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))`
    /// with `C₁ = (0.01·255)²` and `C₂ = (0.03·255)²` (standard constants).
    pub fn ssim(x: &[f64], y: &[f64], _window_size: usize) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 {
            return 0.0;
        }
        let n_f = n as f64;
        let mu_x: f64 = x.iter().take(n).sum::<f64>() / n_f;
        let mu_y: f64 = y.iter().take(n).sum::<f64>() / n_f;
        let var_x: f64 = x.iter().take(n).map(|&v| (v - mu_x).powi(2)).sum::<f64>() / n_f;
        let var_y: f64 = y.iter().take(n).map(|&v| (v - mu_y).powi(2)).sum::<f64>() / n_f;
        let cov: f64 = x
            .iter()
            .take(n)
            .zip(y.iter().take(n))
            .map(|(&xi, &yi)| (xi - mu_x) * (yi - mu_y))
            .sum::<f64>()
            / n_f;

        let c1 = (0.01 * 255.0_f64).powi(2);
        let c2 = (0.03 * 255.0_f64).powi(2);
        let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2);
        let denominator = (mu_x.powi(2) + mu_y.powi(2) + c1) * (var_x + var_y + c2);
        if denominator.abs() < 1e-30 {
            return 1.0;
        }
        (numerator / denominator).clamp(-1.0, 1.0)
    }

    /// Bits per element.
    pub fn bits_per_element(n_bits: f64, n_elements: usize) -> f64 {
        if n_elements == 0 {
            return 0.0;
        }
        n_bits / n_elements as f64
    }

    /// Bjøntegaard delta-rate (BD-Rate) between two RD curves.
    ///
    /// Approximation using log-domain cubic spline interpolation over the
    /// overlapping PSNR range. Positive BD-Rate means `rd_curve2` costs more
    /// bits for the same quality.
    ///
    /// Each curve is a slice of `(bpp, psnr)` pairs — must have ≥ 2 points.
    pub fn bjontegaard_delta_rate(rd_curve1: &[(f64, f64)], rd_curve2: &[(f64, f64)]) -> f64 {
        if rd_curve1.len() < 2 || rd_curve2.len() < 2 {
            return 0.0;
        }

        // Find overlapping PSNR range
        let psnr1_min = rd_curve1
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::INFINITY, f64::min);
        let psnr1_max = rd_curve1
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::NEG_INFINITY, f64::max);
        let psnr2_min = rd_curve2
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::INFINITY, f64::min);
        let psnr2_max = rd_curve2
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::NEG_INFINITY, f64::max);

        let psnr_lo = psnr1_min.max(psnr2_min);
        let psnr_hi = psnr1_max.min(psnr2_max);

        if psnr_hi <= psnr_lo {
            return 0.0;
        }

        // Sample PSNR points in the overlap region
        let n_samples = 100;
        let mut delta_rate_sum = 0.0_f64;
        let mut count = 0;

        for i in 0..n_samples {
            let t = i as f64 / (n_samples - 1) as f64;
            let psnr = psnr_lo + t * (psnr_hi - psnr_lo);

            let r1 = Self::interpolate_rate_at_psnr(rd_curve1, psnr);
            let r2 = Self::interpolate_rate_at_psnr(rd_curve2, psnr);

            if r1 > 0.0 && r2 > 0.0 {
                delta_rate_sum += r2.ln() - r1.ln();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }
        // Convert log-domain mean back to percentage
        (delta_rate_sum / count as f64).exp() * 100.0 - 100.0
    }

    /// Linear interpolation of rate at a given PSNR target.
    fn interpolate_rate_at_psnr(rd_curve: &[(f64, f64)], target_psnr: f64) -> f64 {
        // Sort by PSNR
        let mut sorted: Vec<(f64, f64)> = rd_curve.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if target_psnr <= sorted[0].1 {
            return sorted[0].0;
        }
        if target_psnr >= sorted[sorted.len() - 1].1 {
            return sorted[sorted.len() - 1].0;
        }
        // Find bracketing pair and linearly interpolate
        for i in 1..sorted.len() {
            let (bpp_lo, psnr_lo) = sorted[i - 1];
            let (bpp_hi, psnr_hi) = sorted[i];
            if psnr_lo <= target_psnr && target_psnr <= psnr_hi {
                let t = if (psnr_hi - psnr_lo).abs() < 1e-12 {
                    0.0
                } else {
                    (target_psnr - psnr_lo) / (psnr_hi - psnr_lo)
                };
                return bpp_lo + t * (bpp_hi - bpp_lo);
            }
        }
        sorted[sorted.len() - 1].0
    }

    /// Codebook utilisation: fraction of entries used at least once.
    pub fn codebook_utilization(usage_counts: &[usize]) -> f64 {
        if usage_counts.is_empty() {
            return 0.0;
        }
        let used = usage_counts.iter().filter(|&&c| c > 0).count();
        used as f64 / usage_counts.len() as f64
    }
}
