//! Variational Autoencoder (VAE) primitives — Track U.
//!
//! Implements the core mathematical building blocks and a lightweight `Vae`
//! model backed by plain `Vec<f32>` weight buffers (no `Tensor` dependency),
//! making the module self-contained and easy to integrate with any downstream
//! training loop.
//!
//! # Mathematical background
//!
//! A VAE learns an approximate posterior `q(z|x) ≈ p(z|x)` parameterised as
//! `N(μ, diag(σ²))`.  The encoder maps input `x` to `(μ, log σ²)`.
//! Samples are drawn via the reparameterisation trick
//! `z = μ + ε · σ`,  `ε ∼ N(0, I)`, and then decoded to reconstruct `x`.
//!
//! The training objective (ELBO) is:
//! ```text
//! L = -E[log p(x|z)] + β · KL(q(z|x) ∥ p(z))
//! ```
//! where `β = 1` gives the standard VAE and `β > 1` gives β-VAE.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::vae::{VaeConfig, Vae};
//!
//! let config = VaeConfig::new(784, 32)
//!     .with_hidden_dims(vec![256, 128], vec![128, 256])
//!     .with_beta(1.0);
//!
//! let vae = Vae::new(config)?;
//! let (recon, mu, log_var) = vae.forward(&input, 42)?;
//! let loss = vae.loss(&input, &recon, &mu, &log_var)?;
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Elementwise ReLU: `max(0, x)`.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Elementwise sigmoid: `1 / (1 + exp(-x))`, clamped for numerical safety.
#[inline]
fn sigmoid(x: f32) -> f32 {
    // Clamp input to avoid overflow in `exp`.
    let x_clamped = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-x_clamped).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal sampling — Box-Muller, seeded
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` i.i.d. N(0,1) samples using the Box-Muller transform with a
/// deterministic seed.
///
/// Samples are generated in pairs; if `n` is odd, the last surplus sample is
/// discarded.
fn normal_samples(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);

    let mut i = 0_usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = (r * theta.cos()) as f32;
        let z1 = (r * theta.sin()) as f32;

        out.push(z0);
        i += 1;
        if i < n {
            out.push(z1);
            i += 1;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense-layer forward pass (Vec<f32> weights)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a single linear transformation followed by ReLU:
/// `output[j] = relu(Σ_i input[i] * weight[i * out_dim + j] + bias[j])`
///
/// `weight` is stored in row-major order: row = input index, column = output
/// index, so `weight[i * out_dim + j]` is the weight from input `i` to
/// output `j`.
fn linear_relu(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let in_dim = input.len();
    let expected_weight_len = in_dim * out_dim;
    if weight.len() != expected_weight_len {
        return Err(TensorError::invalid_argument_op(
            "linear_relu",
            &format!(
                "weight length mismatch: expected {expected_weight_len}, got {}",
                weight.len()
            ),
        ));
    }
    if bias.len() != out_dim {
        return Err(TensorError::invalid_argument_op(
            "linear_relu",
            &format!(
                "bias length mismatch: expected {out_dim}, got {}",
                bias.len()
            ),
        ));
    }

    let mut out = bias.to_vec();
    for i in 0..in_dim {
        let xi = input[i];
        for j in 0..out_dim {
            out[j] += xi * weight[i * out_dim + j];
        }
    }
    for v in out.iter_mut() {
        *v = relu(*v);
    }
    Ok(out)
}

/// Apply a single linear transformation (no activation):
/// `output[j] = Σ_i input[i] * weight[i * out_dim + j] + bias[j]`
fn linear(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let in_dim = input.len();
    let expected_weight_len = in_dim * out_dim;
    if weight.len() != expected_weight_len {
        return Err(TensorError::invalid_argument_op(
            "linear",
            &format!(
                "weight length mismatch: expected {expected_weight_len}, got {}",
                weight.len()
            ),
        ));
    }
    if bias.len() != out_dim {
        return Err(TensorError::invalid_argument_op(
            "linear",
            &format!(
                "bias length mismatch: expected {out_dim}, got {}",
                bias.len()
            ),
        ));
    }

    let mut out = bias.to_vec();
    for i in 0..in_dim {
        let xi = input[i];
        for j in 0..out_dim {
            out[j] += xi * weight[i * out_dim + j];
        }
    }
    Ok(out)
}

/// Apply a linear transformation followed by elementwise sigmoid:
/// `output[j] = sigmoid(Σ_i input[i] * weight[i * out_dim + j] + bias[j])`
fn linear_sigmoid(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let mut out = linear(input, weight, bias, out_dim)?;
    for v in out.iter_mut() {
        *v = sigmoid(*v);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight initialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xavier/Glorot uniform initialisation for a weight matrix of shape
/// `[fan_in, fan_out]`.  The scale is `sqrt(6 / (fan_in + fan_out))`.
fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let n = fan_in * fan_out;
    (0..n)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core VAE functions
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian reparameterisation: `z = μ + ε · exp(log_var / 2)`, `ε ∼ N(0, I)`.
///
/// Returns `(z, epsilon)` where `epsilon` contains the raw standard-normal
/// samples used in the computation.  The seed makes the operation fully
/// deterministic and reproducible.
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] if `mu` and `log_var` have
/// different lengths.
pub fn reparameterize(mu: &[f32], log_var: &[f32], seed: u64) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = mu.len();
    if log_var.len() != n {
        return Err(TensorError::invalid_argument_op(
            "reparameterize",
            &format!(
                "mu and log_var must have the same length, got {} vs {}",
                n,
                log_var.len()
            ),
        ));
    }

    let epsilon = normal_samples(n, seed);
    let z: Vec<f32> = mu
        .iter()
        .zip(log_var.iter())
        .zip(epsilon.iter())
        .map(|((&m, &lv), &eps)| {
            let sigma = (lv * 0.5).exp();
            m + eps * sigma
        })
        .collect();

    Ok((z, epsilon))
}

/// Analytical KL divergence from `N(μ, σ²)` to `N(0, 1)`.
///
/// `KL = -0.5 · Σ (1 + log σ² - μ² - σ²)`
///
/// A well-converged VAE will drive this towards zero.  The function returns
/// the *sum* (not mean) over the latent dimensions.
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] if `mu` and `log_var` have
/// different lengths, or if `log_var` contains values that would produce
/// non-finite results.
pub fn kl_divergence_gaussian(mu: &[f32], log_var: &[f32]) -> Result<f32> {
    let n = mu.len();
    if log_var.len() != n {
        return Err(TensorError::invalid_argument_op(
            "kl_divergence_gaussian",
            &format!(
                "mu and log_var must have the same length, got {} vs {}",
                n,
                log_var.len()
            ),
        ));
    }
    if n == 0 {
        return Ok(0.0);
    }

    let kl: f32 = mu
        .iter()
        .zip(log_var.iter())
        .map(|(&m, &lv)| {
            // Clamp log_var to avoid exp overflow.
            let lv_safe = lv.clamp(-20.0, 20.0);
            1.0 + lv_safe - m * m - lv_safe.exp()
        })
        .sum::<f32>()
        * -0.5;

    Ok(kl)
}

/// Binary cross-entropy reconstruction loss for sigmoid-output decoders.
///
/// `BCE = -Σ [ x · log(p) + (1 - x) · log(1 - p) ]`
///
/// where `x` are the original values and `p = recon` are the predicted
/// probabilities (must be in `(0, 1)`).
///
/// A small `ε = 1e-7` is added inside the logarithm for numerical stability.
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] if `input` and `recon` have
/// different lengths.
pub fn bce_reconstruction_loss(input: &[f32], recon: &[f32]) -> Result<f32> {
    let n = input.len();
    if recon.len() != n {
        return Err(TensorError::invalid_argument_op(
            "bce_reconstruction_loss",
            &format!(
                "input and recon must have the same length, got {} vs {}",
                n,
                recon.len()
            ),
        ));
    }
    if n == 0 {
        return Ok(0.0);
    }

    const EPS: f32 = 1e-7;
    let loss: f32 = input
        .iter()
        .zip(recon.iter())
        .map(|(&x, &p)| {
            let p_safe = p.clamp(EPS, 1.0 - EPS);
            -(x * p_safe.ln() + (1.0 - x) * (1.0 - p_safe).ln())
        })
        .sum();

    Ok(loss)
}

/// Mean-squared error reconstruction loss for continuous-output decoders.
///
/// `MSE = Σ (x - recon)²`
///
/// Returns the *sum* (not mean) over all elements, consistent with the
/// convention used by the ELBO formulation in this module.
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] if `input` and `recon` have
/// different lengths.
pub fn mse_reconstruction_loss(input: &[f32], recon: &[f32]) -> Result<f32> {
    let n = input.len();
    if recon.len() != n {
        return Err(TensorError::invalid_argument_op(
            "mse_reconstruction_loss",
            &format!(
                "input and recon must have the same length, got {} vs {}",
                n,
                recon.len()
            ),
        ));
    }
    if n == 0 {
        return Ok(0.0);
    }

    let loss: f32 = input
        .iter()
        .zip(recon.iter())
        .map(|(&x, &r)| {
            let diff = x - r;
            diff * diff
        })
        .sum();

    Ok(loss)
}

/// Evidence Lower BOund (ELBO) loss.
///
/// `total = reconstruction_loss + β · kl`
///
/// With `β = 1` this is the standard VAE objective.  Values of `β > 1`
/// (β-VAE) encourage a more disentangled latent space at the cost of
/// reconstruction quality.
///
/// # Arguments
///
/// * `reconstruction_loss` — already-computed reconstruction term (scalar).
/// * `kl`                  — already-computed KL term (scalar).
/// * `beta`                — KL weight coefficient (`≥ 0`).
pub fn elbo_loss(reconstruction_loss: f32, kl: f32, beta: f32) -> f32 {
    reconstruction_loss + beta * kl
}

// ─────────────────────────────────────────────────────────────────────────────
// VAE configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a Variational Autoencoder.
///
/// Builds the encoder as a stack of fully-connected ReLU layers from
/// `input_dim` → `encoder_hidden_dims[0]` → … → `encoder_hidden_dims[-1]`,
/// and then two parallel heads `→ latent_dim` for `μ` and `log σ²`.
///
/// The decoder mirrors this structure: `latent_dim` → `decoder_hidden_dims[0]`
/// → … → `input_dim`.  The final decoder activation is **sigmoid** so that
/// outputs are in `(0, 1)` (suitable for the BCE loss).
#[derive(Debug, Clone)]
pub struct VaeConfig {
    /// Dimensionality of the input `x`.
    pub input_dim: usize,
    /// Dimensionality of the latent space `z`.
    pub latent_dim: usize,
    /// Hidden layer widths for the encoder (may be empty for a linear VAE).
    pub encoder_hidden_dims: Vec<usize>,
    /// Hidden layer widths for the decoder (may be empty for a linear VAE).
    pub decoder_hidden_dims: Vec<usize>,
    /// β-VAE coefficient.  Use `1.0` for a standard VAE.
    pub beta: f32,
}

impl VaeConfig {
    /// Create a minimal VAE configuration with no hidden layers and `β = 1`.
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            latent_dim,
            encoder_hidden_dims: Vec::new(),
            decoder_hidden_dims: Vec::new(),
            beta: 1.0,
        }
    }

    /// Override both encoder and decoder hidden-layer widths.
    pub fn with_hidden_dims(mut self, enc: Vec<usize>, dec: Vec<usize>) -> Self {
        self.encoder_hidden_dims = enc;
        self.decoder_hidden_dims = dec;
        self
    }

    /// Override the β coefficient.
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss output
// ─────────────────────────────────────────────────────────────────────────────

/// Decomposed loss returned by [`Vae::loss`].
#[derive(Debug, Clone)]
pub struct VaeLoss {
    /// Reconstruction term (`BCE` or `MSE`, caller's choice).
    pub reconstruction_loss: f32,
    /// KL divergence term.
    pub kl_loss: f32,
    /// Combined ELBO: `reconstruction_loss + β · kl_loss`.
    pub total_loss: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// VAE model
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight VAE backed by `Vec<f32>` weight buffers.
///
/// Weight layout for each layer `k`:
/// ```text
/// encoder_weights[k]: [in_dim_k * out_dim_k]  (row-major: row = input index)
/// encoder_biases[k]:  [out_dim_k]
/// ```
/// The same layout applies to the `μ`/`log σ²` heads and the decoder layers.
#[derive(Debug, Clone)]
pub struct Vae {
    /// VAE hyperparameters.
    pub config: VaeConfig,

    // Encoder trunk: one (weight, bias) pair per hidden layer.
    encoder_weights: Vec<Vec<f32>>,
    encoder_biases: Vec<Vec<f32>>,

    // μ head: maps last-encoder-hidden → latent_dim.
    mu_weight: Vec<f32>,
    mu_bias: Vec<f32>,

    // log σ² head: same shape as the μ head.
    log_var_weight: Vec<f32>,
    log_var_bias: Vec<f32>,

    // Decoder trunk: one (weight, bias) pair per hidden layer.
    decoder_weights: Vec<Vec<f32>>,
    decoder_biases: Vec<Vec<f32>>,

    // Output projection: maps last-decoder-hidden → input_dim (sigmoid applied).
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
}

impl Vae {
    /// Construct a new `Vae` with Xavier-uniform initialised weights.
    ///
    /// Seeds are derived deterministically from layer indices so that the
    /// model is reproducible given the same [`VaeConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidArgument`] if any of the dimensions in
    /// `config` are zero.
    pub fn new(config: VaeConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "Vae::new",
                "input_dim must be > 0",
            ));
        }
        if config.latent_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "Vae::new",
                "latent_dim must be > 0",
            ));
        }
        for (idx, &dim) in config.encoder_hidden_dims.iter().enumerate() {
            if dim == 0 {
                return Err(TensorError::invalid_argument_op(
                    "Vae::new",
                    &format!("encoder_hidden_dims[{idx}] must be > 0"),
                ));
            }
        }
        for (idx, &dim) in config.decoder_hidden_dims.iter().enumerate() {
            if dim == 0 {
                return Err(TensorError::invalid_argument_op(
                    "Vae::new",
                    &format!("decoder_hidden_dims[{idx}] must be > 0"),
                ));
            }
        }

        // Build encoder layer dimensions: input_dim → h0 → h1 → … → h_last
        let enc_dims = encoder_dims(&config);
        let dec_dims = decoder_dims(&config);

        let mut seed_counter: u64 = 1;
        let mut next_seed = || {
            let s = seed_counter;
            seed_counter = seed_counter.wrapping_add(1);
            s
        };

        // ── encoder trunk ──────────────────────────────────────────────────
        let mut encoder_weights = Vec::with_capacity(enc_dims.len().saturating_sub(1));
        let mut encoder_biases = Vec::with_capacity(enc_dims.len().saturating_sub(1));
        for i in 0..enc_dims.len().saturating_sub(1) {
            let fan_in = enc_dims[i];
            let fan_out = enc_dims[i + 1];
            encoder_weights.push(xavier_uniform(fan_in, fan_out, next_seed()));
            encoder_biases.push(vec![0.0_f32; fan_out]);
        }

        // ── μ and log σ² heads ─────────────────────────────────────────────
        let last_enc_dim = *enc_dims.last().ok_or_else(|| {
            TensorError::invalid_argument_op("Vae::new", "encoder dimension list is empty")
        })?;
        let mu_weight = xavier_uniform(last_enc_dim, config.latent_dim, next_seed());
        let mu_bias = vec![0.0_f32; config.latent_dim];
        let log_var_weight = xavier_uniform(last_enc_dim, config.latent_dim, next_seed());
        let log_var_bias = vec![0.0_f32; config.latent_dim];

        // ── decoder trunk ──────────────────────────────────────────────────
        let mut decoder_weights = Vec::with_capacity(dec_dims.len().saturating_sub(1));
        let mut decoder_biases = Vec::with_capacity(dec_dims.len().saturating_sub(1));
        for i in 0..dec_dims.len().saturating_sub(1) {
            let fan_in = dec_dims[i];
            let fan_out = dec_dims[i + 1];
            decoder_weights.push(xavier_uniform(fan_in, fan_out, next_seed()));
            decoder_biases.push(vec![0.0_f32; fan_out]);
        }

        // ── output projection ──────────────────────────────────────────────
        let last_dec_dim = *dec_dims.last().ok_or_else(|| {
            TensorError::invalid_argument_op("Vae::new", "decoder dimension list is empty")
        })?;
        let output_weight = xavier_uniform(last_dec_dim, config.input_dim, next_seed());
        let output_bias = vec![0.0_f32; config.input_dim];

        Ok(Self {
            config,
            encoder_weights,
            encoder_biases,
            mu_weight,
            mu_bias,
            log_var_weight,
            log_var_bias,
            decoder_weights,
            decoder_biases,
            output_weight,
            output_bias,
        })
    }

    /// Encode input `x` to the latent distribution parameters `(μ, log σ²)`.
    ///
    /// `x` must have length `config.input_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidArgument`] if `x.len() ≠ input_dim`.
    pub fn encode(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.config.input_dim {
            return Err(TensorError::invalid_argument_op(
                "Vae::encode",
                &format!(
                    "x must have length {}, got {}",
                    self.config.input_dim,
                    x.len()
                ),
            ));
        }

        let enc_dims = encoder_dims(&self.config);
        let mut h: Vec<f32> = x.to_vec();

        for i in 0..self.encoder_weights.len() {
            let out_dim = enc_dims[i + 1];
            h = linear_relu(
                &h,
                &self.encoder_weights[i],
                &self.encoder_biases[i],
                out_dim,
            )?;
        }

        let mu = linear(&h, &self.mu_weight, &self.mu_bias, self.config.latent_dim)?;
        let log_var = linear(
            &h,
            &self.log_var_weight,
            &self.log_var_bias,
            self.config.latent_dim,
        )?;

        Ok((mu, log_var))
    }

    /// Decode latent vector `z` back to the input space.
    ///
    /// `z` must have length `config.latent_dim`.  The output is passed through
    /// sigmoid so that all values are in `(0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidArgument`] if `z.len() ≠ latent_dim`.
    pub fn decode(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.config.latent_dim {
            return Err(TensorError::invalid_argument_op(
                "Vae::decode",
                &format!(
                    "z must have length {}, got {}",
                    self.config.latent_dim,
                    z.len()
                ),
            ));
        }

        let dec_dims = decoder_dims(&self.config);
        let mut h: Vec<f32> = z.to_vec();

        for i in 0..self.decoder_weights.len() {
            let out_dim = dec_dims[i + 1];
            h = linear_relu(
                &h,
                &self.decoder_weights[i],
                &self.decoder_biases[i],
                out_dim,
            )?;
        }

        // Final output projection with sigmoid activation.
        let recon = linear_sigmoid(
            &h,
            &self.output_weight,
            &self.output_bias,
            self.config.input_dim,
        )?;

        Ok(recon)
    }

    /// Full forward pass: encode → reparameterise → decode.
    ///
    /// Returns `(reconstruction, μ, log σ²)`.
    ///
    /// # Arguments
    ///
    /// * `x`    — input vector of length `input_dim`.
    /// * `seed` — seed for the reparameterisation noise.
    ///
    /// # Errors
    ///
    /// Propagates errors from \[`encode`\], [`reparameterize`] and \[`decode`\].
    pub fn forward(&self, x: &[f32], seed: u64) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (mu, log_var) = self.encode(x)?;
        let (z, _epsilon) = reparameterize(&mu, &log_var, seed)?;
        let recon = self.decode(&z)?;
        Ok((recon, mu, log_var))
    }

    /// Compute the ELBO loss components for one sample.
    ///
    /// Uses BCE reconstruction loss (appropriate for sigmoid decoder output).
    ///
    /// # Arguments
    ///
    /// * `x`       — original input.
    /// * `recon`   — reconstruction produced by \[`decode`\].
    /// * `mu`      — posterior mean from \[`encode`\].
    /// * `log_var` — posterior log-variance from \[`encode`\].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`bce_reconstruction_loss`] and
    /// [`kl_divergence_gaussian`].
    pub fn loss(&self, x: &[f32], recon: &[f32], mu: &[f32], log_var: &[f32]) -> Result<VaeLoss> {
        let reconstruction_loss = bce_reconstruction_loss(x, recon)?;
        let kl_loss = kl_divergence_gaussian(mu, log_var)?;
        let total_loss = elbo_loss(reconstruction_loss, kl_loss, self.config.beta);
        Ok(VaeLoss {
            reconstruction_loss,
            kl_loss,
            total_loss,
        })
    }

    /// Draw a single sample from the prior `N(0, I)` and decode it.
    ///
    /// # Arguments
    ///
    /// * `seed` — seed for the prior sample.
    ///
    /// # Errors
    ///
    /// Propagates errors from \[`decode`\].
    pub fn sample(&self, seed: u64) -> Result<Vec<f32>> {
        let z = normal_samples(self.config.latent_dim, seed);
        self.decode(&z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal dimension helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the full sequence of dimensions for the encoder trunk:
/// `[input_dim, h0, h1, …, h_last]`.
fn encoder_dims(config: &VaeConfig) -> Vec<usize> {
    let mut dims = Vec::with_capacity(1 + config.encoder_hidden_dims.len());
    dims.push(config.input_dim);
    dims.extend_from_slice(&config.encoder_hidden_dims);
    dims
}

/// Build the full sequence of dimensions for the decoder trunk:
/// `[latent_dim, h0, h1, …, h_last]`.
fn decoder_dims(config: &VaeConfig) -> Vec<usize> {
    let mut dims = Vec::with_capacity(1 + config.decoder_hidden_dims.len());
    dims.push(config.latent_dim);
    dims.extend_from_slice(&config.decoder_hidden_dims);
    dims
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Acceptable floating-point tolerance for almost-zero comparisons.
    const EPS: f32 = 1e-5;

    // ── reparameterize ────────────────────────────────────────────────────────

    #[test]
    fn test_reparameterize_output_shape() -> Result<()> {
        let mu = vec![0.0_f32; 8];
        let log_var = vec![0.0_f32; 8];
        let (z, eps) = reparameterize(&mu, &log_var, 42)?;
        assert_eq!(z.len(), 8, "z must have the same length as mu");
        assert_eq!(eps.len(), 8, "epsilon must have the same length as mu");
        Ok(())
    }

    #[test]
    fn test_reparameterize_variance_positive() -> Result<()> {
        // With log_var = 0  →  sigma = 1, so z should not be all equal to mu.
        let mu = vec![0.0_f32; 16];
        let log_var = vec![0.0_f32; 16];
        let (z, _) = reparameterize(&mu, &log_var, 7)?;
        let all_zero = z.iter().all(|&v| v.abs() < EPS);
        assert!(!all_zero, "z must have non-zero variance when sigma=1");
        Ok(())
    }

    #[test]
    fn test_reparameterize_deterministic_with_same_seed() -> Result<()> {
        let mu = vec![1.0_f32; 4];
        let log_var = vec![-1.0_f32; 4];
        let (z1, _) = reparameterize(&mu, &log_var, 99)?;
        let (z2, _) = reparameterize(&mu, &log_var, 99)?;
        assert_eq!(z1, z2, "same seed must produce identical samples");
        Ok(())
    }

    #[test]
    fn test_reparameterize_length_mismatch_error() {
        let mu = vec![0.0_f32; 4];
        let log_var = vec![0.0_f32; 5];
        assert!(
            reparameterize(&mu, &log_var, 0).is_err(),
            "mismatched lengths must return an error"
        );
    }

    #[test]
    fn test_reparameterize_mean_preserved_at_zero_var() -> Result<()> {
        // When log_var → -∞ the noise vanishes, but we use a finite approximation:
        // log_var = -40.0 → sigma ≈ exp(-20) ≈ 2e-9, negligible.
        let mu: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let log_var = vec![-40.0_f32; 8];
        let (z, _) = reparameterize(&mu, &log_var, 123)?;
        for (m, zv) in mu.iter().zip(z.iter()) {
            assert!(
                (m - zv).abs() < 1e-3,
                "z ≈ mu when sigma is negligibly small: mu={m} z={zv}"
            );
        }
        Ok(())
    }

    // ── kl_divergence_gaussian ────────────────────────────────────────────────

    #[test]
    fn test_kl_zero_for_standard_normal() -> Result<()> {
        // KL(N(0,1) ∥ N(0,1)) = 0.
        let mu = vec![0.0_f32; 8];
        let log_var = vec![0.0_f32; 8];
        let kl = kl_divergence_gaussian(&mu, &log_var)?;
        assert!(
            kl.abs() < EPS,
            "KL should be ~0 for standard normal, got {kl}"
        );
        Ok(())
    }

    #[test]
    fn test_kl_positive_for_nonzero_mu() -> Result<()> {
        let mu = vec![2.0_f32; 4];
        let log_var = vec![0.0_f32; 4];
        let kl = kl_divergence_gaussian(&mu, &log_var)?;
        assert!(kl > 0.0, "KL must be > 0 when mu != 0, got {kl}");
        Ok(())
    }

    #[test]
    fn test_kl_length_mismatch_error() {
        let mu = vec![0.0_f32; 3];
        let log_var = vec![0.0_f32; 5];
        assert!(kl_divergence_gaussian(&mu, &log_var).is_err());
    }

    #[test]
    fn test_kl_empty_slices() -> Result<()> {
        let kl = kl_divergence_gaussian(&[], &[])?;
        assert_eq!(kl, 0.0);
        Ok(())
    }

    #[test]
    fn test_kl_known_value() -> Result<()> {
        // For a single dimension with mu=0, log_var=ln(4) (sigma²=4):
        // KL = -0.5 * (1 + ln(4) - 0 - 4) = -0.5 * (1 + ln(4) - 4)
        //    = -0.5 * (ln(4) - 3)
        //    ≈ -0.5 * (1.3863 - 3) = -0.5 * (-1.6137) ≈ 0.8069
        let mu = vec![0.0_f32];
        let log_var = vec![(4.0_f32).ln()];
        let kl = kl_divergence_gaussian(&mu, &log_var)?;
        let expected = 0.5 * (4.0_f32 - 1.0 - 4.0_f32.ln());
        assert!(
            (kl - expected).abs() < 1e-4,
            "KL value mismatch: got {kl}, expected {expected}"
        );
        Ok(())
    }

    // ── bce_reconstruction_loss ───────────────────────────────────────────────

    #[test]
    fn test_bce_perfect_reconstruction_near_zero() -> Result<()> {
        // BCE with x == recon is the binary entropy H(x), which is non-negative.
        // For n elements with x=recon=0.9:
        //   BCE = -n*(0.9*ln(0.9) + 0.1*ln(0.1)) ≈ n * 0.325 per element.
        // With n=8 this is ≈ 2.6, which is the minimum BCE achievable at p=0.9.
        // The key property is: BCE(x, x) ≤ BCE(x, y) for any y ≠ x.
        let x = vec![0.9_f32; 8];
        let recon = x.clone();

        // Loss when reconstruction is perfect.
        let loss_perfect = bce_reconstruction_loss(&x, &recon)?;

        // Loss when reconstruction is wrong (recon = 0.1 for x = 0.9).
        let recon_wrong = vec![0.1_f32; 8];
        let loss_wrong = bce_reconstruction_loss(&x, &recon_wrong)?;

        assert!(
            loss_perfect >= 0.0,
            "BCE must be non-negative, got {loss_perfect}"
        );
        assert!(
            loss_perfect < loss_wrong,
            "BCE(x, x) must be less than BCE(x, bad_recon): {loss_perfect} vs {loss_wrong}"
        );
        Ok(())
    }

    #[test]
    fn test_bce_zero_for_perfect_reconstruction() -> Result<()> {
        // When x=recon=0.5, BCE is at its minimum for this value pair.
        let x = vec![0.5_f32; 4];
        let recon = x.clone();
        let loss = bce_reconstruction_loss(&x, &recon)?;
        assert!(loss >= 0.0, "BCE must be non-negative");
        Ok(())
    }

    #[test]
    fn test_bce_length_mismatch_error() {
        assert!(bce_reconstruction_loss(&[0.5], &[0.5, 0.5]).is_err());
    }

    #[test]
    fn test_bce_non_negative() -> Result<()> {
        let x: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
        let recon: Vec<f32> = (0..10).map(|i| 0.5 + i as f32 / 20.0 - 0.25).collect();
        let loss = bce_reconstruction_loss(&x, &recon)?;
        assert!(loss >= 0.0, "BCE must always be >= 0, got {loss}");
        Ok(())
    }

    // ── mse_reconstruction_loss ───────────────────────────────────────────────

    #[test]
    fn test_mse_zero_for_perfect_reconstruction() -> Result<()> {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let recon = x.clone();
        let loss = mse_reconstruction_loss(&x, &recon)?;
        assert!(
            loss.abs() < EPS,
            "MSE must be 0 for identical inputs, got {loss}"
        );
        Ok(())
    }

    #[test]
    fn test_mse_known_value() -> Result<()> {
        // MSE([1, 2, 3], [4, 5, 6]) = (3²+3²+3²) = 27
        let x = vec![1.0_f32, 2.0, 3.0];
        let recon = vec![4.0_f32, 5.0, 6.0];
        let loss = mse_reconstruction_loss(&x, &recon)?;
        assert!(
            (loss - 27.0).abs() < EPS,
            "MSE mismatch: expected 27.0, got {loss}"
        );
        Ok(())
    }

    #[test]
    fn test_mse_length_mismatch_error() {
        assert!(mse_reconstruction_loss(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_mse_non_negative() -> Result<()> {
        let x = vec![-1.0_f32, 0.5, -0.3];
        let recon = vec![0.8_f32, -0.2, 1.1];
        let loss = mse_reconstruction_loss(&x, &recon)?;
        assert!(loss >= 0.0, "MSE must always be >= 0, got {loss}");
        Ok(())
    }

    // ── elbo_loss ─────────────────────────────────────────────────────────────

    #[test]
    fn test_elbo_equals_recon_plus_beta_times_kl() {
        let recon = 3.7_f32;
        let kl = 1.2_f32;
        let beta = 2.5_f32;
        let total = elbo_loss(recon, kl, beta);
        let expected = recon + beta * kl;
        assert!(
            (total - expected).abs() < EPS,
            "elbo mismatch: got {total}, expected {expected}"
        );
    }

    #[test]
    fn test_elbo_beta_one_standard_vae() {
        let recon = 5.0_f32;
        let kl = 2.0_f32;
        let total = elbo_loss(recon, kl, 1.0);
        assert!((total - 7.0).abs() < EPS);
    }

    #[test]
    fn test_elbo_beta_four_quadruples_kl() {
        let recon = 0.0_f32;
        let kl = 1.0_f32;
        let beta = 4.0_f32;
        let total = elbo_loss(recon, kl, beta);
        assert!(
            (total - 4.0).abs() < EPS,
            "With recon=0 and kl=1, beta=4 should give total=4, got {total}"
        );
    }

    // ── VaeConfig ─────────────────────────────────────────────────────────────

    #[test]
    fn test_vae_config_defaults() {
        let cfg = VaeConfig::new(784, 32);
        assert_eq!(cfg.input_dim, 784);
        assert_eq!(cfg.latent_dim, 32);
        assert!(cfg.encoder_hidden_dims.is_empty());
        assert!(cfg.decoder_hidden_dims.is_empty());
        assert!((cfg.beta - 1.0).abs() < EPS);
    }

    #[test]
    fn test_vae_config_with_hidden_dims() {
        let cfg = VaeConfig::new(100, 8).with_hidden_dims(vec![64, 32], vec![32, 64]);
        assert_eq!(cfg.encoder_hidden_dims, vec![64, 32]);
        assert_eq!(cfg.decoder_hidden_dims, vec![32, 64]);
    }

    #[test]
    fn test_vae_config_with_beta() {
        let cfg = VaeConfig::new(10, 4).with_beta(4.0);
        assert!((cfg.beta - 4.0).abs() < EPS);
    }

    // ── Vae::new ──────────────────────────────────────────────────────────────

    #[test]
    fn test_vae_new_valid_config() -> Result<()> {
        let cfg = VaeConfig::new(16, 4).with_hidden_dims(vec![8], vec![8]);
        let _vae = Vae::new(cfg)?;
        Ok(())
    }

    #[test]
    fn test_vae_new_minimal_no_hidden() -> Result<()> {
        let cfg = VaeConfig::new(10, 3);
        let _vae = Vae::new(cfg)?;
        Ok(())
    }

    #[test]
    fn test_vae_new_zero_input_dim_error() {
        let cfg = VaeConfig::new(0, 4);
        assert!(Vae::new(cfg).is_err());
    }

    #[test]
    fn test_vae_new_zero_latent_dim_error() {
        let cfg = VaeConfig::new(8, 0);
        assert!(Vae::new(cfg).is_err());
    }

    // ── Vae::encode ───────────────────────────────────────────────────────────

    #[test]
    fn test_vae_encode_output_shape() -> Result<()> {
        let cfg = VaeConfig::new(12, 4).with_hidden_dims(vec![8], vec![8]);
        let vae = Vae::new(cfg)?;
        let x = vec![0.5_f32; 12];
        let (mu, log_var) = vae.encode(&x)?;
        assert_eq!(mu.len(), 4, "mu must have latent_dim elements");
        assert_eq!(log_var.len(), 4, "log_var must have latent_dim elements");
        Ok(())
    }

    #[test]
    fn test_vae_encode_wrong_input_length_error() -> Result<()> {
        let cfg = VaeConfig::new(12, 4);
        let vae = Vae::new(cfg)?;
        assert!(vae.encode(&[0.0_f32; 10]).is_err());
        Ok(())
    }

    // ── Vae::decode ───────────────────────────────────────────────────────────

    #[test]
    fn test_vae_decode_output_shape() -> Result<()> {
        let cfg = VaeConfig::new(20, 5).with_hidden_dims(vec![10], vec![10]);
        let vae = Vae::new(cfg)?;
        let z = vec![0.0_f32; 5];
        let recon = vae.decode(&z)?;
        assert_eq!(
            recon.len(),
            20,
            "reconstruction must have input_dim elements"
        );
        Ok(())
    }

    #[test]
    fn test_vae_decode_output_in_unit_interval() -> Result<()> {
        let cfg = VaeConfig::new(16, 4);
        let vae = Vae::new(cfg)?;
        let z = vec![1.0_f32; 4];
        let recon = vae.decode(&z)?;
        for &v in &recon {
            assert!(
                (0.0..=1.0).contains(&v),
                "sigmoid output must be in [0, 1], got {v}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_vae_decode_wrong_latent_length_error() -> Result<()> {
        let cfg = VaeConfig::new(16, 4);
        let vae = Vae::new(cfg)?;
        assert!(vae.decode(&[0.0_f32; 3]).is_err());
        Ok(())
    }

    // ── Vae::forward ─────────────────────────────────────────────────────────

    #[test]
    fn test_vae_forward_output_shapes() -> Result<()> {
        let cfg = VaeConfig::new(16, 4).with_hidden_dims(vec![8], vec![8]);
        let vae = Vae::new(cfg)?;
        let x = vec![0.5_f32; 16];
        let (recon, mu, log_var) = vae.forward(&x, 42)?;
        assert_eq!(recon.len(), 16, "recon must have input_dim elements");
        assert_eq!(mu.len(), 4, "mu must have latent_dim elements");
        assert_eq!(log_var.len(), 4, "log_var must have latent_dim elements");
        Ok(())
    }

    #[test]
    fn test_vae_forward_recon_in_unit_interval() -> Result<()> {
        let cfg = VaeConfig::new(8, 2);
        let vae = Vae::new(cfg)?;
        let x = vec![0.3_f32; 8];
        let (recon, _, _) = vae.forward(&x, 1)?;
        for &v in &recon {
            assert!((0.0..=1.0).contains(&v), "recon must be in [0, 1], got {v}");
        }
        Ok(())
    }

    // ── Vae::loss ─────────────────────────────────────────────────────────────

    #[test]
    fn test_vae_loss_components_non_negative() -> Result<()> {
        let cfg = VaeConfig::new(12, 3).with_hidden_dims(vec![6], vec![6]);
        let vae = Vae::new(cfg)?;
        let x = vec![0.5_f32; 12];
        let (recon, mu, log_var) = vae.forward(&x, 7)?;
        let loss = vae.loss(&x, &recon, &mu, &log_var)?;
        assert!(
            loss.reconstruction_loss >= 0.0,
            "reconstruction_loss must be >= 0, got {}",
            loss.reconstruction_loss
        );
        assert!(
            loss.kl_loss >= 0.0,
            "kl_loss must be >= 0, got {}",
            loss.kl_loss
        );
        assert!(
            loss.total_loss >= 0.0,
            "total_loss must be >= 0, got {}",
            loss.total_loss
        );
        Ok(())
    }

    #[test]
    fn test_vae_loss_total_equals_elbo() -> Result<()> {
        let beta = 2.0_f32;
        let cfg = VaeConfig::new(8, 2).with_beta(beta);
        let vae = Vae::new(cfg)?;
        let x = vec![0.4_f32; 8];
        let (recon, mu, log_var) = vae.forward(&x, 5)?;
        let loss = vae.loss(&x, &recon, &mu, &log_var)?;
        let expected_total = loss.reconstruction_loss + beta * loss.kl_loss;
        assert!(
            (loss.total_loss - expected_total).abs() < EPS,
            "total_loss mismatch: got {}, expected {}",
            loss.total_loss,
            expected_total
        );
        Ok(())
    }

    // ── Vae::sample ──────────────────────────────────────────────────────────

    #[test]
    fn test_vae_sample_output_shape() -> Result<()> {
        let cfg = VaeConfig::new(16, 4).with_hidden_dims(vec![8], vec![8]);
        let vae = Vae::new(cfg)?;
        let sample = vae.sample(42)?;
        assert_eq!(sample.len(), 16, "sample must have input_dim elements");
        Ok(())
    }

    #[test]
    fn test_vae_sample_output_in_unit_interval() -> Result<()> {
        let cfg = VaeConfig::new(10, 3);
        let vae = Vae::new(cfg)?;
        let sample = vae.sample(0)?;
        for &v in &sample {
            assert!(
                (0.0..=1.0).contains(&v),
                "sample must be in [0, 1], got {v}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_vae_sample_deterministic() -> Result<()> {
        let cfg = VaeConfig::new(8, 2);
        let vae = Vae::new(cfg)?;
        let s1 = vae.sample(77)?;
        let s2 = vae.sample(77)?;
        assert_eq!(s1, s2, "same seed must produce identical samples");
        Ok(())
    }

    // ── beta-VAE: 4x KL in total loss ────────────────────────────────────────

    #[test]
    fn test_beta_vae_beta4_quadruples_kl_contribution() -> Result<()> {
        // Build two VAEs with the same architecture but beta=1 vs beta=4.
        // The KL term in the total loss should be 4× larger for beta=4.
        let cfg1 = VaeConfig::new(8, 2).with_beta(1.0);
        let cfg4 = VaeConfig::new(8, 2).with_beta(4.0);
        let vae1 = Vae::new(cfg1)?;
        let vae4 = Vae::new(cfg4)?;

        // Use a fixed mu and log_var for a controlled KL value.
        let mu = vec![1.0_f32; 2];
        let log_var = vec![0.0_f32; 2];
        let recon = vec![0.5_f32; 8];
        let x = vec![0.5_f32; 8];

        let loss1 = vae1.loss(&x, &recon, &mu, &log_var)?;
        let loss4 = vae4.loss(&x, &recon, &mu, &log_var)?;

        // total = recon + beta * kl  ⟹  total4 - total1 = 3 * kl
        let delta = loss4.total_loss - loss1.total_loss;
        let expected_delta = 3.0 * loss1.kl_loss;
        assert!(
            (delta - expected_delta).abs() < 1e-4,
            "beta-VAE(4) - beta-VAE(1) should equal 3*kl: delta={delta} expected={expected_delta}"
        );

        // Also verify total4 = recon + 4*kl.
        let expected_total4 = loss4.reconstruction_loss + 4.0 * loss4.kl_loss;
        assert!(
            (loss4.total_loss - expected_total4).abs() < EPS,
            "beta-VAE total_loss mismatch"
        );
        Ok(())
    }
}
