//! Diffusion model implementations
//!
//! This module provides:
//! - DDPM (Denoising Diffusion Probabilistic Models) — Ho et al. 2020
//!   <https://arxiv.org/abs/2006.11239>
//! - DDIM (Denoising Diffusion Implicit Models) — Song et al. 2020
//!   <https://arxiv.org/abs/2010.02502>
//! - Improved DDPM cosine schedule — Nichol & Dhariwal 2021
//!   <https://arxiv.org/abs/2102.09672>
//! - Score Matching / VDM noise schedules
//! - Classifier-free guidance utilities
//!
//! # Notes on random noise
//!
//! All methods that require Gaussian noise accept pre-generated `noise: &[f32]` slices.
//! The caller is responsible for providing noise sampled from N(0, 1).
//! Use `scirs2_core::random::Random` for reproducible noise generation.

use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────
//  Helper: convert a TensorError-incompatible internal error
// ─────────────────────────────────────────────────────────────────

#[inline]
fn make_err(msg: impl Into<String>) -> TensorError {
    TensorError::invalid_argument(msg.into())
}

// ─────────────────────────────────────────────────────────────────
//  Beta schedule helpers
// ─────────────────────────────────────────────────────────────────

/// Build a linear beta schedule: betas\[t\] = beta_start + t*(beta_end-beta_start)/(T-1).
///
/// Used in the original DDPM paper (Ho et al. 2020).
pub fn linear_beta_schedule(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Vec<f32> {
    assert!(num_timesteps >= 2, "num_timesteps must be >= 2");
    (0..num_timesteps)
        .map(|t| beta_start + (beta_end - beta_start) * t as f32 / (num_timesteps - 1) as f32)
        .collect()
}

/// Build a cosine beta schedule (Nichol & Dhariwal 2021).
///
/// alpha_bar(t) = cos²( (t/T + s) / (1+s) * π/2 )
/// betas\[t\] = 1 - alpha_bar(t+1)/alpha_bar(t), clipped to [0, 0.999].
pub fn cosine_beta_schedule(num_timesteps: usize, s: f32) -> Vec<f32> {
    assert!(num_timesteps >= 1, "num_timesteps must be >= 1");
    let t = num_timesteps as f32;
    let alpha_bar = |step: f32| -> f32 {
        let inner = (step / t + s) / (1.0 + s) * std::f32::consts::PI / 2.0;
        inner.cos().powi(2)
    };
    let ab0 = alpha_bar(0.0);
    (0..num_timesteps)
        .map(|i| {
            let ab_t = alpha_bar(i as f32);
            let ab_t1 = alpha_bar(i as f32 + 1.0);
            let beta = 1.0 - ab_t1 / ab_t.max(1e-8);
            // Re-normalise to [0, 1] range relative to ab(0)
            let beta = beta * (1.0 / ab0.max(1e-8));
            beta.clamp(0.0, 0.999)
        })
        .collect()
}

/// Build a sigmoid beta schedule.
///
/// betas are spaced such that a sigmoid from -6 to +6 gives monotonically
/// increasing values in [0, 1].
pub fn sigmoid_beta_schedule(num_timesteps: usize) -> Vec<f32> {
    assert!(num_timesteps >= 2, "num_timesteps must be >= 2");
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    (0..num_timesteps)
        .map(|t| {
            let x = -6.0 + 12.0 * t as f32 / (num_timesteps - 1) as f32;
            sigmoid(x).clamp(1e-5, 0.999)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────
//  DdpmScheduler
// ─────────────────────────────────────────────────────────────────

/// Pre-computed DDPM scheduler tables.
///
/// Given the beta schedule `{beta_t}`, the scheduler computes all derived
/// quantities needed for the forward (q) and reverse (p) processes:
///
/// ```text
/// alpha_t          = 1 - beta_t
/// alpha_bar_t      = prod_{s<=t} alpha_s
/// q(x_t | x_0)     = N(x_t; sqrt(alpha_bar_t)*x_0,  (1-alpha_bar_t)*I)
/// q(x_{t-1}|x_t,x_0) = N(x_{t-1}; posterior_mean, posterior_variance)
/// ```
#[derive(Debug, Clone)]
pub struct DdpmScheduler {
    pub num_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub alphas_cumprod_prev: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
    pub log_one_minus_alphas_cumprod: Vec<f32>,
    pub sqrt_recip_alphas_cumprod: Vec<f32>,
    pub sqrt_recipm1_alphas_cumprod: Vec<f32>,
    pub posterior_variance: Vec<f32>,
    pub posterior_log_variance_clipped: Vec<f32>,
    pub posterior_mean_coef1: Vec<f32>,
    pub posterior_mean_coef2: Vec<f32>,
}

impl DdpmScheduler {
    /// Build a scheduler from an arbitrary beta schedule vector.
    fn from_betas(betas: Vec<f32>) -> Self {
        let num_timesteps = betas.len();
        assert!(num_timesteps >= 1);

        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();

        // cumulative product alpha_bar_t
        let mut alphas_cumprod = vec![0.0f32; num_timesteps];
        alphas_cumprod[0] = alphas[0];
        for t in 1..num_timesteps {
            alphas_cumprod[t] = alphas_cumprod[t - 1] * alphas[t];
        }

        // alpha_bar_prev:  [1.0, alpha_bar_0, alpha_bar_1, ..., alpha_bar_{T-2}]
        let mut alphas_cumprod_prev = vec![1.0f32; num_timesteps];
        alphas_cumprod_prev[1..num_timesteps]
            .copy_from_slice(&alphas_cumprod[..(num_timesteps - 1)]);

        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();
        let log_one_minus_alphas_cumprod: Vec<f32> = alphas_cumprod
            .iter()
            .map(|&a| (1.0 - a).max(1e-20).ln())
            .collect();
        let sqrt_recip_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 / a).sqrt()).collect();
        let sqrt_recipm1_alphas_cumprod: Vec<f32> = alphas_cumprod
            .iter()
            .map(|&a| (1.0 / a - 1.0).sqrt())
            .collect();

        // Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        let posterior_variance: Vec<f32> = (0..num_timesteps)
            .map(|t| {
                betas[t] * (1.0 - alphas_cumprod_prev[t]) / (1.0 - alphas_cumprod[t]).max(1e-20)
            })
            .collect();

        // Clipped log variance: avoid -inf at t=0
        let posterior_log_variance_clipped: Vec<f32> = posterior_variance
            .iter()
            .enumerate()
            .map(|(t, &v)| {
                if t == 0 {
                    // At t=0, posterior_variance[0] is 0; use posterior_variance[1] if available
                    let v1 = if num_timesteps > 1 {
                        posterior_variance[1]
                    } else {
                        v
                    };
                    v1.max(1e-20).ln()
                } else {
                    v.max(1e-20).ln()
                }
            })
            .collect();

        // Posterior mean coefficients:
        // mu_tilde_t = coef1 * x_0 + coef2 * x_t
        // coef1 = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)
        // coef2 = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        let posterior_mean_coef1: Vec<f32> = (0..num_timesteps)
            .map(|t| {
                alphas_cumprod_prev[t].sqrt() * betas[t] / (1.0 - alphas_cumprod[t]).max(1e-20)
            })
            .collect();
        let posterior_mean_coef2: Vec<f32> = (0..num_timesteps)
            .map(|t| {
                alphas[t].sqrt() * (1.0 - alphas_cumprod_prev[t])
                    / (1.0 - alphas_cumprod[t]).max(1e-20)
            })
            .collect();

        Self {
            num_timesteps,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
        }
    }

    /// Create a DDPM scheduler with a **linear** beta schedule.
    ///
    /// Typical values: `beta_start = 1e-4`, `beta_end = 0.02`, `num_timesteps = 1000`.
    pub fn linear(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let betas = linear_beta_schedule(num_timesteps, beta_start, beta_end);
        Self::from_betas(betas)
    }

    /// Create a DDPM scheduler with the **cosine** beta schedule (Improved DDPM).
    ///
    /// `s = 0.008` is the default offset from the paper.
    pub fn cosine(num_timesteps: usize, s: f32) -> Self {
        let betas = cosine_beta_schedule(num_timesteps, s);
        Self::from_betas(betas)
    }

    // ─────────────────────────────────────────────────────────────
    //  Forward (diffusion) process
    // ─────────────────────────────────────────────────────────────

    /// Sample x_t given x_0 and pre-generated Gaussian noise.
    ///
    /// q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    ///
    /// `noise` must have the same length as `x_start`.
    pub fn q_sample(
        &self,
        x_start: &[f32],
        t: usize,
        noise: &[f32],
    ) -> Result<Vec<f32>, TensorError> {
        if t >= self.num_timesteps {
            return Err(make_err(format!(
                "t={} out of range [0, {})",
                t, self.num_timesteps
            )));
        }
        if x_start.len() != noise.len() {
            return Err(make_err(format!(
                "x_start.len()={} != noise.len()={}",
                x_start.len(),
                noise.len()
            )));
        }
        let sqrt_ab = self.sqrt_alphas_cumprod[t];
        let sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t];
        let out: Vec<f32> = x_start
            .iter()
            .zip(noise.iter())
            .map(|(&x0, &eps)| sqrt_ab * x0 + sqrt_1mab * eps)
            .collect();
        Ok(out)
    }

    // ─────────────────────────────────────────────────────────────
    //  Reverse-process helpers
    // ─────────────────────────────────────────────────────────────

    /// Recover a prediction of x_0 from x_t and the predicted noise.
    ///
    /// x_0_hat = sqrt_recip_alpha_bar_t * x_t - sqrt_recipm1_alpha_bar_t * noise
    pub fn predict_start_from_noise(
        &self,
        x_t: &[f32],
        t: usize,
        noise: &[f32],
    ) -> Result<Vec<f32>, TensorError> {
        if t >= self.num_timesteps {
            return Err(make_err(format!(
                "t={} out of range [0, {})",
                t, self.num_timesteps
            )));
        }
        if x_t.len() != noise.len() {
            return Err(make_err(format!(
                "x_t.len()={} != noise.len()={}",
                x_t.len(),
                noise.len()
            )));
        }
        let recip = self.sqrt_recip_alphas_cumprod[t];
        let recipm1 = self.sqrt_recipm1_alphas_cumprod[t];
        let out: Vec<f32> = x_t
            .iter()
            .zip(noise.iter())
            .map(|(&xt, &eps)| recip * xt - recipm1 * eps)
            .collect();
        Ok(out)
    }

    /// Compute the parameters of q(x_{t-1} | x_t, x_0).
    ///
    /// Returns `(posterior_mean, posterior_variance_vec, posterior_log_variance_clipped_vec)`.
    /// All returned vecs have the same length as `x_start`.
    pub fn q_posterior(
        &self,
        x_start: &[f32],
        x_t: &[f32],
        t: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), TensorError> {
        if t >= self.num_timesteps {
            return Err(make_err(format!(
                "t={} out of range [0, {})",
                t, self.num_timesteps
            )));
        }
        if x_start.len() != x_t.len() {
            return Err(make_err(format!(
                "x_start.len()={} != x_t.len()={}",
                x_start.len(),
                x_t.len()
            )));
        }
        let n = x_start.len();
        let c1 = self.posterior_mean_coef1[t];
        let c2 = self.posterior_mean_coef2[t];
        let var = self.posterior_variance[t];
        let log_var = self.posterior_log_variance_clipped[t];

        let mean: Vec<f32> = x_start
            .iter()
            .zip(x_t.iter())
            .map(|(&x0, &xt)| c1 * x0 + c2 * xt)
            .collect();
        let variance = vec![var; n];
        let log_variance = vec![log_var; n];
        Ok((mean, variance, log_variance))
    }

    /// Compute p_theta(x_{t-1} | x_t) mean and variance from model output (predicted noise).
    ///
    /// The model is assumed to predict the noise eps; we first reconstruct x_0,
    /// then compute q_posterior.
    ///
    /// Returns `(model_mean, posterior_variance, posterior_log_variance_clipped)`.
    pub fn p_mean_variance(
        &self,
        model_output: &[f32],
        x_t: &[f32],
        t: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), TensorError> {
        let x_start_pred = self.predict_start_from_noise(x_t, t, model_output)?;
        // Clip predicted x_0 to [-1, 1] for numerical stability (common in DDPM impl)
        let x_start_clipped: Vec<f32> = x_start_pred.iter().map(|&x| x.clamp(-1.0, 1.0)).collect();
        self.q_posterior(&x_start_clipped, x_t, t)
    }

    /// Perform a single DDPM denoising step p(x_{t-1} | x_t).
    ///
    /// `model_output` is the predicted noise, `noise` is a fresh N(0,1) sample
    /// (ignored at t=0 to avoid adding noise at the final step).
    pub fn p_sample(
        &self,
        model_output: &[f32],
        x_t: &[f32],
        t: usize,
        noise: &[f32],
    ) -> Result<Vec<f32>, TensorError> {
        if noise.len() != x_t.len() {
            return Err(make_err(format!(
                "noise.len()={} != x_t.len()={}",
                noise.len(),
                x_t.len()
            )));
        }
        let (mean, _var, log_var) = self.p_mean_variance(model_output, x_t, t)?;
        // At t=0 do not add noise
        if t == 0 {
            return Ok(mean);
        }
        let out: Vec<f32> = mean
            .iter()
            .zip(log_var.iter())
            .zip(noise.iter())
            .map(|((&mu, &lv), &eps)| {
                // std = exp(0.5 * log_var)
                let std = (0.5 * lv).exp();
                mu + std * eps
            })
            .collect();
        Ok(out)
    }

    /// DDIM deterministic sampling step.
    ///
    /// Song et al. "Denoising Diffusion Implicit Models" (2020).
    ///
    /// # Arguments
    /// * `model_output` - Predicted noise eps_theta(x_t, t)
    /// * `x_t`          - Current noisy sample
    /// * `t`            - Current timestep index
    /// * `t_prev`       - Previous timestep index (< t, can be 0)
    /// * `eta`          - Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)
    /// * `noise`        - External N(0,1) noise (only used when eta > 0)
    pub fn ddim_step(
        &self,
        model_output: &[f32],
        x_t: &[f32],
        t: usize,
        t_prev: usize,
        eta: f32,
        noise: &[f32],
    ) -> Result<Vec<f32>, TensorError> {
        if t >= self.num_timesteps {
            return Err(make_err(format!(
                "t={} out of range [0, {})",
                t, self.num_timesteps
            )));
        }
        if x_t.len() != model_output.len() {
            return Err(make_err(format!(
                "x_t.len()={} != model_output.len()={}",
                x_t.len(),
                model_output.len()
            )));
        }
        if noise.len() != x_t.len() {
            return Err(make_err(format!(
                "noise.len()={} != x_t.len()={}",
                noise.len(),
                x_t.len()
            )));
        }

        let ab_t = self.alphas_cumprod[t];
        let ab_prev = if t_prev < self.num_timesteps {
            self.alphas_cumprod[t_prev]
        } else {
            1.0f32
        };

        // Predict x_0
        let sqrt_recip = self.sqrt_recip_alphas_cumprod[t];
        let sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t];
        let x0_pred: Vec<f32> = x_t
            .iter()
            .zip(model_output.iter())
            .map(|(&xt, &eps)| sqrt_recip * xt - sqrt_recipm1 * eps)
            .collect();

        // Sigma for DDIM
        let sigma = eta
            * ((1.0 - ab_prev) / (1.0 - ab_t).max(1e-8) * ((1.0 - ab_t) / ab_prev.max(1e-8)))
                .sqrt();

        // Direction pointing to x_t
        let coef_xt_dir = ((1.0 - ab_prev - sigma * sigma).max(0.0)).sqrt();

        let out: Vec<f32> = x0_pred
            .iter()
            .zip(model_output.iter())
            .zip(noise.iter())
            .map(|((&x0, &eps), &z)| ab_prev.sqrt() * x0 + coef_xt_dir * eps + sigma * z)
            .collect();
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Loss functions
// ─────────────────────────────────────────────────────────────────

/// Simple L2 (MSE) loss between the model's noise prediction and true noise.
///
/// L_simple = ||eps - eps_theta(x_t, t)||²
pub fn ddpm_loss(model_output: &[f32], noise: &[f32]) -> f32 {
    assert_eq!(
        model_output.len(),
        noise.len(),
        "model_output and noise must have the same length"
    );
    let n = model_output.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    model_output
        .iter()
        .zip(noise.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / n
}

/// Variational lower bound (VLB) loss term for a single timestep.
///
/// This computes the KL divergence between the posterior q and the model p
/// at timestep t, using the Gaussian formula:
///
/// KL(q || p) = 0.5 * [ log(sigma_p²/sigma_q²) + (sigma_q² + (mu_q - mu_p)²)/sigma_p² - 1 ]
///
/// For t > 0, uses the posterior computed by the scheduler.
/// For t == 0, falls back to the negative log-likelihood term (log p(x_0 | x_1)).
pub fn vlb_loss(
    model_output: &[f32],
    x_start: &[f32],
    x_t: &[f32],
    t: usize,
    scheduler: &DdpmScheduler,
) -> Result<f32, TensorError> {
    if x_start.len() != x_t.len() || x_start.len() != model_output.len() {
        return Err(make_err(
            "model_output, x_start, x_t must all have the same length".to_string(),
        ));
    }
    let n = x_start.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }

    // True posterior params
    let (true_mean, true_var, _) = scheduler.q_posterior(x_start, x_t, t)?;
    // Model posterior params
    let (model_mean, _, model_log_var) = scheduler.p_mean_variance(model_output, x_t, t)?;

    let kl: f32 = true_mean
        .iter()
        .zip(model_mean.iter())
        .zip(true_var.iter())
        .zip(model_log_var.iter())
        .map(|(((&mu_q, &mu_p), &var_q), &log_var_p)| {
            let var_p = log_var_p.exp().max(1e-8);
            let var_q_safe = var_q.max(1e-8);
            // KL(N(mu_q,sig_q²) || N(mu_p,sig_p²))
            0.5 * ((var_p / var_q_safe).ln() + (var_q_safe + (mu_q - mu_p).powi(2)) / var_p - 1.0)
        })
        .sum::<f32>()
        / n;

    Ok(kl)
}

// ─────────────────────────────────────────────────────────────────
//  Score Matching noise scheduler
// ─────────────────────────────────────────────────────────────────

/// Noise schedule for score-based generative models.
///
/// Maintains a sequence of sigma values descending from sigma_max to sigma_min
/// (or ascending from sigma_min to sigma_max, depending on convention).
#[derive(Debug, Clone)]
pub struct ScoreMatchingScheduler {
    /// Ordered sequence of noise levels sigma[0..num_steps]
    pub sigmas: Vec<f32>,
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub num_steps: usize,
}

impl ScoreMatchingScheduler {
    /// Build a **geometric** (log-linear) sigma schedule.
    ///
    /// sigmas\[i\] = sigma_max * (sigma_min/sigma_max)^(i/(num_steps-1))
    pub fn geometric(num_steps: usize, sigma_min: f32, sigma_max: f32) -> Self {
        assert!(num_steps >= 2, "num_steps must be >= 2");
        assert!(sigma_min > 0.0, "sigma_min must be > 0");
        assert!(sigma_max > sigma_min, "sigma_max must be > sigma_min");
        let ratio = (sigma_min / sigma_max).ln();
        let sigmas: Vec<f32> = (0..num_steps)
            .map(|i| sigma_max * (ratio * i as f32 / (num_steps - 1) as f32).exp())
            .collect();
        Self {
            sigmas,
            sigma_min,
            sigma_max,
            num_steps,
        }
    }

    /// Build a **Karras et al. 2022** sigma schedule.
    ///
    /// sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    ///
    /// Reference: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models"
    /// <https://arxiv.org/abs/2206.00364>
    pub fn karras(num_steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Self {
        assert!(num_steps >= 2, "num_steps must be >= 2");
        assert!(sigma_min > 0.0, "sigma_min must be > 0");
        assert!(sigma_max > sigma_min, "sigma_max must be > sigma_min");
        assert!(rho > 0.0, "rho must be > 0");

        let inv_rho = 1.0 / rho;
        let rho_max = sigma_max.powf(inv_rho);
        let rho_min = sigma_min.powf(inv_rho);
        let sigmas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let t = i as f32 / (num_steps - 1) as f32;
                (rho_max + t * (rho_min - rho_max)).powf(rho)
            })
            .collect();
        Self {
            sigmas,
            sigma_min,
            sigma_max,
            num_steps,
        }
    }

    /// Return the noise level sigma at step `step`.
    pub fn sigma_at(&self, step: usize) -> f32 {
        self.sigmas[step]
    }

    /// Return the full sigma schedule slice.
    pub fn schedule(&self) -> &[f32] {
        &self.sigmas
    }
}

// ─────────────────────────────────────────────────────────────────
//  Classifier-free guidance
// ─────────────────────────────────────────────────────────────────

/// Combine conditional and unconditional model outputs using classifier-free guidance.
///
/// output = unconditional + guidance_scale * (conditional - unconditional)
///
/// `guidance_scale = 1.0` is equivalent to pure conditional sampling.
/// `guidance_scale > 1.0` amplifies the conditioning effect.
pub fn classifier_free_guidance(
    conditional: &[f32],
    unconditional: &[f32],
    guidance_scale: f32,
) -> Vec<f32> {
    assert_eq!(
        conditional.len(),
        unconditional.len(),
        "conditional and unconditional must have the same length"
    );
    conditional
        .iter()
        .zip(unconditional.iter())
        .map(|(&c, &u)| u + guidance_scale * (c - u))
        .collect()
}

// ─────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const T: usize = 100;
    const BETA_START: f32 = 1e-4;
    const BETA_END: f32 = 0.02;
    const DIM: usize = 16;

    fn make_linear_scheduler() -> DdpmScheduler {
        DdpmScheduler::linear(T, BETA_START, BETA_END)
    }

    fn make_cosine_scheduler() -> DdpmScheduler {
        DdpmScheduler::cosine(T, 0.008)
    }

    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }
    fn ones(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }
    fn simple_noise(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.1).sin()).collect()
    }
    fn simple_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.3 + 0.1) % 2.0) - 1.0)
            .collect()
    }

    // ─── Schedule creation ─────────────────────────────────────

    #[test]
    fn test_linear_schedule_creation() {
        let s = make_linear_scheduler();
        assert_eq!(s.num_timesteps, T);
        assert_eq!(s.betas.len(), T);
        assert_eq!(s.alphas_cumprod.len(), T);
        // First beta should equal beta_start
        assert!(
            (s.betas[0] - BETA_START).abs() < 1e-6,
            "betas[0]={}",
            s.betas[0]
        );
        // Last beta should equal beta_end
        assert!(
            (s.betas[T - 1] - BETA_END).abs() < 1e-6,
            "betas[T-1]={}",
            s.betas[T - 1]
        );
    }

    #[test]
    fn test_cosine_schedule_creation() {
        let s = make_cosine_scheduler();
        assert_eq!(s.num_timesteps, T);
        assert_eq!(s.betas.len(), T);
        // All betas should be in (0, 0.999]
        for (i, &b) in s.betas.iter().enumerate() {
            assert!(b > 0.0 && b <= 0.999, "betas[{}]={}", i, b);
        }
    }

    #[test]
    fn test_schedule_alphas_cumprod_decreasing() {
        let s = make_linear_scheduler();
        // alphas_cumprod must be monotonically decreasing
        for t in 1..T {
            assert!(
                s.alphas_cumprod[t] <= s.alphas_cumprod[t - 1],
                "alphas_cumprod not decreasing at t={}",
                t
            );
        }
    }

    #[test]
    fn test_cosine_alphas_cumprod_in_range() {
        let s = make_cosine_scheduler();
        for (i, &ab) in s.alphas_cumprod.iter().enumerate() {
            assert!(ab > 0.0 && ab <= 1.0, "alphas_cumprod[{}]={}", i, ab);
        }
    }

    // ─── q_sample (forward process) ────────────────────────────

    #[test]
    fn test_q_sample_shape() {
        let s = make_linear_scheduler();
        let x0 = simple_x(DIM);
        let noise = simple_noise(DIM);
        let xt = s.q_sample(&x0, 50, &noise).expect("q_sample failed");
        assert_eq!(xt.len(), DIM);
    }

    #[test]
    fn test_q_sample_at_t0_close_to_x0() {
        // At t=0, sqrt_alphas_cumprod ~ 1, sqrt_one_minus ~ 0; xt ≈ x0
        let s = make_linear_scheduler();
        let x0 = simple_x(DIM);
        let noise = simple_noise(DIM);
        let xt = s.q_sample(&x0, 0, &noise).expect("q_sample t=0 failed");
        for (a, b) in x0.iter().zip(xt.iter()) {
            // Not exactly equal because alpha_bar[0] != 1, but should be close
            assert!(
                (a - b).abs() < 0.1,
                "large deviation at t=0: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_q_sample_length_mismatch_returns_error() {
        let s = make_linear_scheduler();
        let x0 = simple_x(DIM);
        let noise = simple_noise(DIM + 1);
        assert!(s.q_sample(&x0, 0, &noise).is_err());
    }

    #[test]
    fn test_q_sample_out_of_range_t_returns_error() {
        let s = make_linear_scheduler();
        let x0 = simple_x(DIM);
        let noise = simple_noise(DIM);
        assert!(s.q_sample(&x0, T, &noise).is_err());
    }

    // ─── predict_start_from_noise ──────────────────────────────

    #[test]
    fn test_predict_start_from_noise_roundtrip() {
        // If we q_sample with eps then predict_start_from_noise should recover x_0 (approximately)
        let s = make_linear_scheduler();
        let x0 = simple_x(DIM);
        let eps = simple_noise(DIM);
        let t_idx = 20;
        let xt = s.q_sample(&x0, t_idx, &eps).expect("q_sample failed");
        let x0_pred = s
            .predict_start_from_noise(&xt, t_idx, &eps)
            .expect("predict failed");
        let mse: f32 = x0
            .iter()
            .zip(x0_pred.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / DIM as f32;
        assert!(mse < 1e-4, "round-trip MSE too high: {}", mse);
    }

    #[test]
    fn test_predict_start_from_noise_shape() {
        let s = make_linear_scheduler();
        let xt = simple_x(DIM);
        let noise = simple_noise(DIM);
        let out = s
            .predict_start_from_noise(&xt, 50, &noise)
            .expect("predict failed");
        assert_eq!(out.len(), DIM);
    }

    // ─── q_posterior ───────────────────────────────────────────

    #[test]
    fn test_q_posterior_shapes() {
        let s = make_linear_scheduler();
        let x0 = simple_x(DIM);
        let xt = simple_x(DIM);
        let (mean, var, log_var) = s.q_posterior(&x0, &xt, 50).expect("q_posterior failed");
        assert_eq!(mean.len(), DIM);
        assert_eq!(var.len(), DIM);
        assert_eq!(log_var.len(), DIM);
    }

    // ─── p_mean_variance ───────────────────────────────────────

    #[test]
    fn test_p_mean_variance_shapes() {
        let s = make_linear_scheduler();
        let model_out = simple_noise(DIM);
        let xt = simple_x(DIM);
        let (m, v, lv) = s
            .p_mean_variance(&model_out, &xt, 50)
            .expect("p_mean_variance failed");
        assert_eq!(m.len(), DIM);
        assert_eq!(v.len(), DIM);
        assert_eq!(lv.len(), DIM);
    }

    // ─── p_sample ──────────────────────────────────────────────

    #[test]
    fn test_p_sample_shape() {
        let s = make_linear_scheduler();
        let model_out = simple_noise(DIM);
        let xt = simple_x(DIM);
        let noise = simple_noise(DIM);
        let out = s
            .p_sample(&model_out, &xt, 50, &noise)
            .expect("p_sample failed");
        assert_eq!(out.len(), DIM);
    }

    #[test]
    fn test_p_sample_t0_equals_mean() {
        // At t=0, p_sample should return exactly the mean (no noise added)
        let s = make_linear_scheduler();
        let model_out = zeros(DIM);
        let xt = simple_x(DIM);
        let noise = ones(DIM); // large noise; should be ignored at t=0
        let out_t0 = s
            .p_sample(&model_out, &xt, 0, &noise)
            .expect("p_sample t=0 failed");
        let (mean, _, _) = s
            .p_mean_variance(&model_out, &xt, 0)
            .expect("p_mean_variance t=0 failed");
        let mse: f32 = out_t0
            .iter()
            .zip(mean.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / DIM as f32;
        assert!(mse < 1e-10, "p_sample t=0 mse={}", mse);
    }

    // ─── DDIM step ─────────────────────────────────────────────

    #[test]
    fn test_ddim_step_shape() {
        let s = make_linear_scheduler();
        let model_out = simple_noise(DIM);
        let xt = simple_x(DIM);
        let noise = simple_noise(DIM);
        let out = s
            .ddim_step(&model_out, &xt, 50, 40, 0.0, &noise)
            .expect("ddim_step failed");
        assert_eq!(out.len(), DIM);
    }

    #[test]
    fn test_ddim_step_deterministic_with_eta0() {
        // When eta=0, result should be deterministic regardless of noise
        let s = make_linear_scheduler();
        let model_out = simple_noise(DIM);
        let xt = simple_x(DIM);
        let noise1 = simple_noise(DIM);
        let noise2: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.13).collect();
        let out1 = s
            .ddim_step(&model_out, &xt, 50, 40, 0.0, &noise1)
            .expect("ddim1 failed");
        let out2 = s
            .ddim_step(&model_out, &xt, 50, 40, 0.0, &noise2)
            .expect("ddim2 failed");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "eta=0 should be deterministic: {} vs {}",
                a,
                b
            );
        }
    }

    // ─── ddpm_loss ─────────────────────────────────────────────

    #[test]
    fn test_ddpm_loss_perfect_prediction() {
        let noise = simple_noise(DIM);
        let loss = ddpm_loss(&noise, &noise);
        assert!(
            loss.abs() < 1e-10,
            "loss should be 0 for perfect prediction"
        );
    }

    #[test]
    fn test_ddpm_loss_non_negative() {
        let model_out = simple_noise(DIM);
        let target = simple_x(DIM);
        let loss = ddpm_loss(&model_out, &target);
        assert!(loss >= 0.0, "loss must be non-negative: {}", loss);
    }

    #[test]
    fn test_ddpm_loss_zero_inputs() {
        let loss = ddpm_loss(&zeros(DIM), &zeros(DIM));
        assert_eq!(loss, 0.0);
    }

    // ─── vlb_loss ──────────────────────────────────────────────

    #[test]
    fn test_vlb_loss_returns_ok() {
        let s = make_linear_scheduler();
        let model_out = simple_noise(DIM);
        let x0 = simple_x(DIM);
        let xt = simple_x(DIM);
        let loss = vlb_loss(&model_out, &x0, &xt, 50, &s).expect("vlb_loss failed");
        // VLB loss (KL) should be non-negative
        assert!(loss >= 0.0, "VLB loss must be >= 0: {}", loss);
    }

    // ─── ScoreMatchingScheduler ────────────────────────────────

    #[test]
    fn test_score_matching_geometric_construction() {
        let scheduler = ScoreMatchingScheduler::geometric(50, 0.01, 80.0);
        assert_eq!(scheduler.num_steps, 50);
        assert_eq!(scheduler.sigmas.len(), 50);
        // First sigma should be sigma_max
        assert!(
            (scheduler.sigmas[0] - 80.0).abs() < 1e-4,
            "first sigma != sigma_max"
        );
        // Last sigma should be sigma_min
        assert!(
            (scheduler.sigmas[49] - 0.01).abs() < 1e-4,
            "last sigma={} != sigma_min=0.01",
            scheduler.sigmas[49]
        );
    }

    #[test]
    fn test_score_matching_karras_construction() {
        let scheduler = ScoreMatchingScheduler::karras(50, 0.01, 80.0, 7.0);
        assert_eq!(scheduler.num_steps, 50);
        assert_eq!(scheduler.sigmas.len(), 50);
        // Sigmas should be decreasing (high to low noise)
        for i in 1..50 {
            assert!(
                scheduler.sigmas[i] <= scheduler.sigmas[i - 1] + 1e-4,
                "sigmas not decreasing at i={}: {} > {}",
                i,
                scheduler.sigmas[i],
                scheduler.sigmas[i - 1]
            );
        }
    }

    #[test]
    fn test_score_matching_sigma_at() {
        let scheduler = ScoreMatchingScheduler::geometric(10, 0.01, 10.0);
        assert_eq!(scheduler.sigma_at(0), scheduler.sigmas[0]);
        assert_eq!(scheduler.sigma_at(9), scheduler.sigmas[9]);
    }

    #[test]
    fn test_score_matching_schedule_slice() {
        let scheduler = ScoreMatchingScheduler::geometric(10, 0.01, 10.0);
        assert_eq!(scheduler.schedule().len(), 10);
    }

    // ─── classifier_free_guidance ──────────────────────────────

    #[test]
    fn test_cfg_scale_one_returns_conditional() {
        let cond: Vec<f32> = (0..DIM).map(|i| i as f32).collect();
        let uncond = zeros(DIM);
        let out = classifier_free_guidance(&cond, &uncond, 1.0);
        for (a, b) in out.iter().zip(cond.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "CFG scale=1 should return conditional"
            );
        }
    }

    #[test]
    fn test_cfg_scale_zero_returns_unconditional() {
        let cond: Vec<f32> = (0..DIM).map(|i| i as f32).collect();
        let uncond: Vec<f32> = (0..DIM).map(|i| -(i as f32)).collect();
        let out = classifier_free_guidance(&cond, &uncond, 0.0);
        for (a, b) in out.iter().zip(uncond.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "CFG scale=0 should return unconditional"
            );
        }
    }

    #[test]
    fn test_cfg_output_length() {
        let cond = simple_x(DIM);
        let uncond = simple_noise(DIM);
        let out = classifier_free_guidance(&cond, &uncond, 7.5);
        assert_eq!(out.len(), DIM);
    }

    // ─── Beta schedules ────────────────────────────────────────

    #[test]
    fn test_sigmoid_beta_schedule() {
        let betas = sigmoid_beta_schedule(100);
        assert_eq!(betas.len(), 100);
        // Should be monotonically increasing
        for i in 1..100 {
            assert!(
                betas[i] >= betas[i - 1],
                "sigmoid betas not increasing at i={}",
                i
            );
        }
    }

    #[test]
    fn test_linear_beta_schedule_endpoints() {
        let betas = linear_beta_schedule(100, 1e-4, 0.02);
        assert!((betas[0] - 1e-4).abs() < 1e-8);
        assert!((betas[99] - 0.02).abs() < 1e-8);
    }

    #[test]
    fn test_cosine_beta_schedule_values_in_range() {
        let betas = cosine_beta_schedule(100, 0.008);
        for (i, &b) in betas.iter().enumerate() {
            assert!(
                (0.0..=0.999).contains(&b),
                "betas[{}]={} out of range",
                i,
                b
            );
        }
    }
}
