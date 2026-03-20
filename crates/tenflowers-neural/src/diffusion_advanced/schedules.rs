//! §3 — Noise Schedules & Samplers.
//!
//! - [`CosineNoiseSchedule`]  — cosine β schedule (Nichol & Dhariwal)
//! - [`FlowSchedule`]         — exponential interpolant schedule
//! - [`DpmSolverSampler`]     — DPM-Solver++ 2nd-order multistep
//! - [`PndmSampler`]          — PLMS/PNDM 4th-order linear multistep
//! - [`SdeBasedSampler`]      — SDE Euler-Maruyama step

use tenflowers_core::TensorError;

use super::helpers::{axpy, make_err};

// ─────────────────────────────────────────────────────────────────────────────

/// Cosine noise schedule (Nichol & Dhariwal 2021).
///
/// `ᾱ_t = cos²(π/2 · (t/T + s) / (1 + s))`
#[derive(Debug, Clone)]
pub struct CosineNoiseSchedule {
    pub num_timesteps: usize,
    pub s: f64,
    pub alpha_bars: Vec<f64>,
    pub betas: Vec<f64>,
}

impl CosineNoiseSchedule {
    /// Build a cosine schedule with `num_timesteps` steps and offset `s`.
    pub fn new(num_timesteps: usize, s: f64) -> Self {
        let alpha_bar_fn = |t: f64| -> f64 {
            let inner = (t / num_timesteps as f64 + s) / (1.0 + s) * std::f64::consts::PI / 2.0;
            inner.cos().powi(2)
        };

        let alpha_bars: Vec<f64> = (0..=num_timesteps)
            .map(|t| alpha_bar_fn(t as f64))
            .collect();

        let betas: Vec<f64> = (0..num_timesteps)
            .map(|t| {
                let beta = 1.0 - alpha_bars[t + 1] / alpha_bars[t].max(1e-8);
                beta.clamp(0.0, 0.999)
            })
            .collect();

        Self {
            num_timesteps,
            s,
            alpha_bars,
            betas,
        }
    }

    /// Get ᾱ_t at discrete timestep t.
    pub fn alpha_bar(&self, t: usize) -> f64 {
        self.alpha_bars[t.min(self.num_timesteps)]
    }

    /// Get β_t at timestep t (0-indexed).
    pub fn beta(&self, t: usize) -> f64 {
        self.betas[t.min(self.num_timesteps - 1)]
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Exponential interpolant schedule for flow matching.
///
/// `σ(t) = σ_min · (σ_max / σ_min)^t`
#[derive(Debug, Clone)]
pub struct FlowSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub num_steps: usize,
    pub sigmas: Vec<f64>,
}

impl FlowSchedule {
    /// Build an exponential flow schedule.
    pub fn new(sigma_min: f64, sigma_max: f64, num_steps: usize) -> Self {
        let sigmas: Vec<f64> = (0..=num_steps)
            .map(|i| {
                let t = i as f64 / num_steps as f64;
                sigma_min * (sigma_max / sigma_min).powf(t)
            })
            .collect();
        Self {
            sigma_min,
            sigma_max,
            num_steps,
            sigmas,
        }
    }

    /// Get σ at fractional time t ∈ [0, 1].
    pub fn sigma_at(&self, t: f64) -> f64 {
        self.sigma_min * (self.sigma_max / self.sigma_min).powf(t.clamp(0.0, 1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// DPM-Solver++ 2nd-order multistep sampler (Lu et al. 2022).
///
/// Maintains a history buffer of at most 2 previous model outputs.
#[derive(Debug, Clone)]
pub struct DpmSolverSampler {
    pub num_steps: usize,
    pub lambdas: Vec<f64>,
    pub history: Vec<Vec<f64>>,
}

impl DpmSolverSampler {
    /// Create a new DPM-Solver++ sampler with a linear lambda schedule.
    pub fn new(num_steps: usize, lambda_min: f64, lambda_max: f64) -> Self {
        let lambdas: Vec<f64> = (0..=num_steps)
            .map(|i| lambda_max - (lambda_max - lambda_min) * i as f64 / num_steps as f64)
            .collect();
        Self {
            num_steps,
            lambdas,
            history: Vec::new(),
        }
    }

    /// Single DPM-Solver++ 2nd-order step.
    ///
    /// Falls back to 1st-order (DDIM-like) when no history is available.
    pub fn sample_step(
        &mut self,
        x: &[f64],
        _t_cur: f64,
        _t_next: f64,
        model_out: Vec<f64>,
    ) -> Result<Vec<f64>, TensorError> {
        let step_idx = self.history.len();
        let (lambda_cur, lambda_next) = if step_idx < self.lambdas.len().saturating_sub(1) {
            (self.lambdas[step_idx], self.lambdas[step_idx + 1])
        } else {
            (
                self.lambdas[self.lambdas.len().saturating_sub(2)],
                self.lambdas[self.lambdas.len().saturating_sub(1)],
            )
        };
        let h = lambda_next - lambda_cur;
        let phi1 = (-h).exp() - 1.0;

        let x_next = if self.history.is_empty() {
            axpy(x, -phi1, &model_out)
        } else {
            let prev = &self.history[self.history.len() - 1];
            let r = (lambda_cur - self.lambdas[step_idx.saturating_sub(1)])
                / (lambda_next - lambda_cur + 1e-8);
            let d: Vec<f64> = model_out
                .iter()
                .zip(prev.iter())
                .map(|(&cur, &prv)| cur + r * (cur - prv))
                .collect();
            axpy(x, -phi1, &d)
        };

        self.history.push(model_out);
        if self.history.len() > 2 {
            self.history.remove(0);
        }

        Ok(x_next)
    }

    /// Reset the history buffer.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// PLMS/PNDM 4th-order linear multistep sampler (Song et al. 2021).
///
/// Maintains a circular buffer of the last 4 model outputs.
#[derive(Debug, Clone)]
pub struct PndmSampler {
    pub num_steps: usize,
    pub buffer: Vec<Vec<f64>>,
    pub coefficients: [[f64; 4]; 4],
}

impl PndmSampler {
    /// 4th-order Adams-Bashforth-like PLMS coefficients.
    const PLMS_COEFFS: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [3.0 / 2.0, -1.0 / 2.0, 0.0, 0.0],
        [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0, 0.0],
        [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
    ];

    /// Create a new PNDM sampler.
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps,
            buffer: Vec::with_capacity(4),
            coefficients: Self::PLMS_COEFFS,
        }
    }

    /// Add model output to buffer; return PLMS-extrapolated step.
    pub fn step(
        &mut self,
        x: &[f64],
        model_out: Vec<f64>,
        dt: f64,
    ) -> Result<Vec<f64>, TensorError> {
        self.buffer.push(model_out);
        if self.buffer.len() > 4 {
            self.buffer.remove(0);
        }
        let order = self.buffer.len() - 1;
        let coeffs = &self.coefficients[order];

        let dim = self.buffer[0].len();
        let mut d = vec![0.0_f64; dim];
        let buf_len = self.buffer.len();
        for (k, &c) in coeffs[..buf_len].iter().rev().enumerate() {
            let buf_idx = buf_len - 1 - k;
            for j in 0..dim {
                d[j] += c * self.buffer[buf_idx][j];
            }
        }

        Ok(axpy(x, dt, &d))
    }

    /// Reset the buffer for a new sampling run.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Current buffer length.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// SDE-based sampler with Euler-Maruyama discretisation.
///
/// `dx = f(x, t) dt + g(t) dW`
#[derive(Debug, Clone)]
pub struct SdeBasedSampler {
    pub sigmas: Vec<f64>,
    pub num_steps: usize,
}

impl SdeBasedSampler {
    /// Create a new SDE sampler with a geometric noise schedule.
    pub fn new(num_steps: usize, sigma_min: f64, sigma_max: f64) -> Self {
        let sigmas: Vec<f64> = (0..=num_steps)
            .map(|i| sigma_max * (sigma_min / sigma_max).powf(i as f64 / num_steps as f64))
            .collect();
        Self { sigmas, num_steps }
    }

    /// Euler-Maruyama step.
    ///
    /// `x_{t+1} = x_t − σ_t² · score(x_t) · |dt| + σ_t · √|dt| · noise`
    pub fn em_step(
        &self,
        x: &[f64],
        score: &[f64],
        step_idx: usize,
        noise: &[f64],
    ) -> Result<Vec<f64>, TensorError> {
        if step_idx >= self.num_steps {
            return Err(make_err("SdeBasedSampler: step_idx out of range"));
        }
        if x.len() != score.len() || x.len() != noise.len() {
            return Err(make_err("SdeBasedSampler: dimension mismatch"));
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let dt = sigma_next - sigma;
        let g = sigma;

        let x_next: Vec<f64> = x
            .iter()
            .zip(score.iter())
            .zip(noise.iter())
            .map(|((&xi, &si), &ni)| xi - sigma * sigma * si * dt.abs() + g * dt.abs().sqrt() * ni)
            .collect();

        Ok(x_next)
    }
}
