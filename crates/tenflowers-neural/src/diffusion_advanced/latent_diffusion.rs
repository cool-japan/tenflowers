//! §2 — Latent Diffusion implementations.
//!
//! - [`VariationalEncoder`]    — x → (μ, log σ²)
//! - [`VariationalDecoder`]    — z → x̂
//! - [`LatentDiffusionModel`]  — VAE + DDPM in latent space
//! - [`ConditioningEncoder`]   — text/class embedding
//! - [`ClassifierFreeGuidance`]— ε_guided = ε_uncond + w*(ε_cond − ε_uncond)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

use super::helpers::{linear_fwd, make_err, normal_samples_f64, relu};

// ─────────────────────────────────────────────────────────────────────────────

/// Variational Encoder: x → (μ, log σ²).
///
/// Two-layer MLP encoder with ReLU hidden activation.
#[derive(Debug, Clone)]
pub struct VariationalEncoder {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w_mu: Vec<f64>,
    b_mu: Vec<f64>,
    w_lv: Vec<f64>,
    b_lv: Vec<f64>,
}

impl VariationalEncoder {
    /// Create a new variational encoder with Xavier-like initialisation.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale1 = (2.0_f64 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0_f64 / (hidden_dim + latent_dim) as f64).sqrt();

        let mut init = |n: usize, scale: f64| -> Vec<f64> {
            (0..n)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        };

        let w1 = init(input_dim * hidden_dim, scale1);
        let b1 = vec![0.0; hidden_dim];
        let w_mu = init(hidden_dim * latent_dim, scale2);
        let b_mu = vec![0.0; latent_dim];
        let w_lv = init(hidden_dim * latent_dim, scale2);
        let b_lv = vec![0.0; latent_dim];

        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            w1,
            b1,
            w_mu,
            b_mu,
            w_lv,
            b_lv,
        }
    }

    /// Encode: returns `(mu, log_var)`, each of length `latent_dim`.
    pub fn encode(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>), TensorError> {
        if x.len() != self.input_dim {
            return Err(make_err("VariationalEncoder: input dimension mismatch"));
        }
        let h: Vec<f64> = linear_fwd(x, &self.w1, &self.b1, self.hidden_dim)
            .into_iter()
            .map(relu)
            .collect();
        let mu = linear_fwd(&h, &self.w_mu, &self.b_mu, self.latent_dim);
        let log_var = linear_fwd(&h, &self.w_lv, &self.b_lv, self.latent_dim);
        Ok((mu, log_var))
    }

    /// Reparameterisation trick: z = μ + ε·σ, ε ~ N(0, I).
    pub fn reparameterize(
        &self,
        mu: &[f64],
        log_var: &[f64],
        seed: u64,
    ) -> Result<Vec<f64>, TensorError> {
        if mu.len() != self.latent_dim || log_var.len() != self.latent_dim {
            return Err(make_err(
                "VariationalEncoder: reparameterize dimension mismatch",
            ));
        }
        let eps = normal_samples_f64(self.latent_dim, seed);
        let z: Vec<f64> = mu
            .iter()
            .zip(log_var.iter())
            .zip(eps.iter())
            .map(|((&m, &lv), &e)| m + (0.5 * lv).exp() * e)
            .collect();
        Ok(z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Variational Decoder: z → x̂.
///
/// Two-layer MLP decoder with ReLU hidden activation.
#[derive(Debug, Clone)]
pub struct VariationalDecoder {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

impl VariationalDecoder {
    /// Create a new variational decoder with Xavier-like initialisation.
    pub fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale1 = (2.0_f64 / (latent_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0_f64 / (hidden_dim + output_dim) as f64).sqrt();

        let mut init = |n: usize, scale: f64| -> Vec<f64> {
            (0..n)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        };

        let w1 = init(latent_dim * hidden_dim, scale1);
        let b1 = vec![0.0; hidden_dim];
        let w2 = init(hidden_dim * output_dim, scale2);
        let b2 = vec![0.0; output_dim];

        Self {
            latent_dim,
            hidden_dim,
            output_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Decode z → x̂.
    pub fn decode(&self, z: &[f64]) -> Result<Vec<f64>, TensorError> {
        if z.len() != self.latent_dim {
            return Err(make_err("VariationalDecoder: latent dimension mismatch"));
        }
        let h: Vec<f64> = linear_fwd(z, &self.w1, &self.b1, self.hidden_dim)
            .into_iter()
            .map(relu)
            .collect();
        Ok(linear_fwd(&h, &self.w2, &self.b2, self.output_dim))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Latent Diffusion Model — VAE + DDPM in latent space.
///
/// Pipeline: encode(x) → diffuse in latent space → decode(z)
#[derive(Debug, Clone)]
pub struct LatentDiffusionModel {
    pub encoder: VariationalEncoder,
    pub decoder: VariationalDecoder,
    pub num_timesteps: usize,
    pub alpha_bars: Vec<f64>,
}

impl LatentDiffusionModel {
    /// Create a new latent diffusion model with cosine noise schedule.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        num_timesteps: usize,
        seed: u64,
    ) -> Self {
        let encoder = VariationalEncoder::new(input_dim, hidden_dim, latent_dim, seed);
        let decoder = VariationalDecoder::new(latent_dim, hidden_dim, input_dim, seed + 1);
        let s = 0.008_f64;
        let alpha_bars: Vec<f64> = (0..=num_timesteps)
            .map(|t| {
                let inner =
                    (t as f64 / num_timesteps as f64 + s) / (1.0 + s) * std::f64::consts::PI / 2.0;
                inner.cos().powi(2)
            })
            .collect();
        Self {
            encoder,
            decoder,
            num_timesteps,
            alpha_bars,
        }
    }

    /// Encode x → z (latent), using reparameterisation.
    pub fn encode(&self, x: &[f64], seed: u64) -> Result<Vec<f64>, TensorError> {
        let (mu, log_var) = self.encoder.encode(x)?;
        self.encoder.reparameterize(&mu, &log_var, seed)
    }

    /// Forward diffuse z to timestep t: `z_t = √ᾱ_t·z + √(1-ᾱ_t)·ε`.
    pub fn diffuse(&self, z: &[f64], t: usize, noise: &[f64]) -> Result<Vec<f64>, TensorError> {
        if t >= self.num_timesteps {
            return Err(make_err("LatentDiffusionModel: t out of range"));
        }
        let ab = self.alpha_bars[t];
        let sqrt_ab = ab.sqrt();
        let sqrt_one_minus_ab = (1.0 - ab).sqrt();
        let z_t = z
            .iter()
            .zip(noise.iter())
            .map(|(&zi, &ei)| sqrt_ab * zi + sqrt_one_minus_ab * ei)
            .collect();
        Ok(z_t)
    }

    /// Decode z_t → x̂.
    pub fn decode(&self, z: &[f64]) -> Result<Vec<f64>, TensorError> {
        self.decoder.decode(z)
    }

    /// KL divergence: `0.5 * Σ(1 + log_var − mu² − exp(log_var))`.
    pub fn kl_loss(&self, mu: &[f64], log_var: &[f64]) -> f64 {
        mu.iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| 0.5 * (1.0 + lv - m * m - lv.exp()))
            .sum::<f64>()
            .abs()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Conditioning Encoder for text/class conditioning.
///
/// Maps a condition vector to a cross-attention conditioning signal.
#[derive(Debug, Clone)]
pub struct ConditioningEncoder {
    pub cond_dim: usize,
    pub embed_dim: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

impl ConditioningEncoder {
    /// Create a new conditioning encoder.
    pub fn new(cond_dim: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f64 / (cond_dim + embed_dim) as f64).sqrt();
        let mut init = |n: usize| -> Vec<f64> {
            (0..n)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        };
        let hidden = (cond_dim + embed_dim) / 2 + 1;
        let w1 = init(cond_dim * hidden);
        let b1 = vec![0.0; hidden];
        let w2 = init(hidden * embed_dim);
        let b2 = vec![0.0; embed_dim];
        Self {
            cond_dim,
            embed_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Encode condition vector to embedding.
    pub fn encode(&self, cond: &[f64]) -> Result<Vec<f64>, TensorError> {
        if cond.len() != self.cond_dim {
            return Err(make_err(
                "ConditioningEncoder: condition dimension mismatch",
            ));
        }
        let hidden_dim = self.b1.len();
        let h: Vec<f64> = linear_fwd(cond, &self.w1, &self.b1, hidden_dim)
            .into_iter()
            .map(relu)
            .collect();
        Ok(linear_fwd(&h, &self.w2, &self.b2, self.embed_dim))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Classifier-Free Guidance (Ho et al. 2022).
///
/// `ε_guided = ε_uncond + w · (ε_cond − ε_uncond)`
#[derive(Debug, Clone)]
pub struct ClassifierFreeGuidance {
    pub dim: usize,
}

impl ClassifierFreeGuidance {
    /// Create a new CFG helper.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Apply classifier-free guidance.
    pub fn apply(
        &self,
        eps_cond: &[f64],
        eps_uncond: &[f64],
        guidance_scale: f64,
    ) -> Result<Vec<f64>, TensorError> {
        if eps_cond.len() != self.dim || eps_uncond.len() != self.dim {
            return Err(make_err("ClassifierFreeGuidance: dimension mismatch"));
        }
        let out: Vec<f64> = eps_uncond
            .iter()
            .zip(eps_cond.iter())
            .map(|(&unc, &cond)| unc + guidance_scale * (cond - unc))
            .collect();
        Ok(out)
    }
}
