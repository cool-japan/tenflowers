//! §1 — Flow Matching implementations.
//!
//! - [`OtFlowMatching`]        — Optimal Transport flow matching (Lipman et al. 2022)
//! - [`CfmModel`]              — Continuous Flow Matching with σ_min interpolation
//! - [`RectifiedFlow`]         — Straight-trajectory reflow
//! - [`ConsistencyModel`]      — One-step generation via consistency distillation
//! - [`FlowMatchingIntegrator`]— ODE solver for inference (Euler / Heun / DPM-Solver)

use scirs2_core::random::rngs::StdRng;
use tenflowers_core::TensorError;

use super::helpers::{axpy, linear_fwd, make_err, sub_vecs};

// ─────────────────────────────────────────────────────────────────────────────

/// Optimal Transport Flow Matching (Lipman et al. 2022).
///
/// Conditional vector field:
/// `u(x, t | x1) = (x1 − x0) / (1 − t)`
///
/// Training loss:
/// `L = ‖v_θ(x_t, t) − u(x_t, t)‖²`
#[derive(Debug, Clone)]
pub struct OtFlowMatching {
    /// Dimension of the data.
    pub dim: usize,
    /// Small ε to avoid division by zero near t=1.
    pub eps: f64,
}

impl OtFlowMatching {
    /// Create a new OT flow matching instance.
    pub fn new(dim: usize) -> Self {
        Self { dim, eps: 1e-5 }
    }

    /// Interpolate: `x_t = (1 − t) · x0 + t · x1`.
    pub fn interpolate(&self, x0: &[f64], x1: &[f64], t: f64) -> Result<Vec<f64>, TensorError> {
        if x0.len() != self.dim || x1.len() != self.dim {
            return Err(make_err("OtFlowMatching: dimension mismatch"));
        }
        let xt = x0
            .iter()
            .zip(x1.iter())
            .map(|(&a, &b)| (1.0 - t) * a + t * b)
            .collect();
        Ok(xt)
    }

    /// Conditional vector field: `u = (x1 − x0) / (1 − t)`.
    pub fn conditional_vf(&self, x0: &[f64], x1: &[f64], t: f64) -> Result<Vec<f64>, TensorError> {
        if x0.len() != self.dim || x1.len() != self.dim {
            return Err(make_err("OtFlowMatching: dimension mismatch"));
        }
        let denom = (1.0 - t).max(self.eps);
        let u = x0
            .iter()
            .zip(x1.iter())
            .map(|(&a, &b)| (b - a) / denom)
            .collect();
        Ok(u)
    }

    /// Compute training loss `‖v_pred − u‖²` (MSE).
    pub fn loss(&self, v_pred: &[f64], v_target: &[f64]) -> Result<f64, TensorError> {
        if v_pred.len() != self.dim || v_target.len() != self.dim {
            return Err(make_err("OtFlowMatching: loss dimension mismatch"));
        }
        let mse = v_pred
            .iter()
            .zip(v_target.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / self.dim as f64;
        Ok(mse)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Continuous Flow Matching (Lipman et al. 2022, σ_min variant).
///
/// Interpolant: `x_t = (1 − (1 − σ_min) · t) · x0 + t · x1`
/// Target vector field: dx_t/dt = x1 − (1 − σ_min) · x0
#[derive(Debug, Clone)]
pub struct CfmModel {
    /// Data dimensionality.
    pub dim: usize,
    /// Minimum noise standard deviation (σ_min), typically 1e-4.
    pub sigma_min: f64,
}

impl CfmModel {
    /// Create a new CFM model.
    pub fn new(dim: usize, sigma_min: f64) -> Self {
        Self { dim, sigma_min }
    }

    /// Interpolate x0 and x1 at time t and compute the target vector field.
    ///
    /// Returns `(x_t, target_vf)`.
    pub fn forward(
        &self,
        x0: &[f64],
        x1: &[f64],
        t: f64,
        _rng: &mut StdRng,
    ) -> Result<(Vec<f64>, Vec<f64>), TensorError> {
        if x0.len() != self.dim || x1.len() != self.dim {
            return Err(make_err("CfmModel: dimension mismatch"));
        }
        // x_t = (1 − (1 − σ_min)·t) · x0 + t · x1
        let coeff0 = 1.0 - (1.0 - self.sigma_min) * t;
        let xt: Vec<f64> = x0
            .iter()
            .zip(x1.iter())
            .map(|(&a, &b)| coeff0 * a + t * b)
            .collect();
        // target_vf = x1 − (1 − σ_min) · x0
        let target_vf: Vec<f64> = x1
            .iter()
            .zip(x0.iter())
            .map(|(&b, &a)| b - (1.0 - self.sigma_min) * a)
            .collect();
        Ok((xt, target_vf))
    }

    /// MSE loss between predicted and target vector fields.
    pub fn loss(&self, v_pred: &[f64], v_target: &[f64]) -> Result<f64, TensorError> {
        if v_pred.len() != self.dim || v_target.len() != self.dim {
            return Err(make_err("CfmModel: loss dimension mismatch"));
        }
        let mse = v_pred
            .iter()
            .zip(v_target.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / self.dim as f64;
        Ok(mse)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Rectified Flow — straight-line trajectories from z_0 to z_1.
///
/// `z_t = (1 − t) · z_0 + t · z_1`
/// Train `v_θ(z_t, t) = z_1 − z_0`.
#[derive(Debug, Clone)]
pub struct RectifiedFlow {
    /// Data dimensionality.
    pub dim: usize,
}

impl RectifiedFlow {
    /// Create a new rectified flow instance.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Interpolate: `z_t = (1 − t) · z0 + t · z1`.
    pub fn interpolate(&self, z0: &[f64], z1: &[f64], t: f64) -> Result<Vec<f64>, TensorError> {
        if z0.len() != self.dim || z1.len() != self.dim {
            return Err(make_err("RectifiedFlow: dimension mismatch"));
        }
        let zt = z0
            .iter()
            .zip(z1.iter())
            .map(|(&a, &b)| (1.0 - t) * a + t * b)
            .collect();
        Ok(zt)
    }

    /// Target velocity: `v = z1 − z0` (constant along the trajectory).
    pub fn target_velocity(&self, z0: &[f64], z1: &[f64]) -> Result<Vec<f64>, TensorError> {
        if z0.len() != self.dim || z1.len() != self.dim {
            return Err(make_err(
                "RectifiedFlow: dimension mismatch in target_velocity",
            ));
        }
        Ok(sub_vecs(z1, z0))
    }

    /// MSE loss between predicted velocity and target `z1 − z0`.
    pub fn loss(&self, v_pred: &[f64], z0: &[f64], z1: &[f64]) -> Result<f64, TensorError> {
        let target = self.target_velocity(z0, z1)?;
        let mse = v_pred
            .iter()
            .zip(target.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / self.dim as f64;
        Ok(mse)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Consistency Model — one-step generation via consistency distillation.
///
/// Consistency function: `f_θ(x_t, t) → x_0`
/// Consistency loss: `‖f_θ(x_t, t) − f_θ⁻(x_{t-Δ}, t-Δ)‖²`
#[derive(Debug, Clone)]
pub struct ConsistencyModel {
    /// Data dimensionality.
    pub dim: usize,
    /// Number of discrete timesteps.
    pub num_timesteps: usize,
    /// Trained weights for a single linear consistency head (dim × dim).
    pub weight: Vec<f64>,
    /// Bias for the consistency head.
    pub bias: Vec<f64>,
}

impl ConsistencyModel {
    /// Create a new consistency model with identity initialisation.
    pub fn new(dim: usize, num_timesteps: usize) -> Self {
        let mut weight = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            weight[i * dim + i] = 1.0;
        }
        let bias = vec![0.0_f64; dim];
        Self {
            dim,
            num_timesteps,
            weight,
            bias,
        }
    }

    /// Apply the consistency function: `f_θ(x_t, t) = W · x_t + b` (toy linear).
    pub fn forward(&self, x_t: &[f64], _t: usize) -> Result<Vec<f64>, TensorError> {
        if x_t.len() != self.dim {
            return Err(make_err("ConsistencyModel: dimension mismatch"));
        }
        Ok(linear_fwd(x_t, &self.weight, &self.bias, self.dim))
    }

    /// Consistency loss between two adjacent timestep predictions.
    pub fn consistency_loss(
        &self,
        x_t: &[f64],
        t: usize,
        x_t_minus: &[f64],
        t_minus: usize,
    ) -> Result<f64, TensorError> {
        let f_t = self.forward(x_t, t)?;
        let f_t_minus = self.forward(x_t_minus, t_minus)?;
        let mse = f_t
            .iter()
            .zip(f_t_minus.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / self.dim as f64;
        Ok(mse)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// ODE integrator method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegratorMethod {
    /// Simple first-order Euler method.
    Euler,
    /// Heun's method (2nd-order predictor-corrector).
    Heun,
    /// DPM-Solver inspired 2nd-order step.
    DpmSolver,
}

/// Flow Matching ODE integrator for inference.
///
/// Integrates from t=1 (noise) to t=0 (data) using a learned velocity field.
#[derive(Debug, Clone)]
pub struct FlowMatchingIntegrator {
    /// Number of ODE steps.
    pub num_steps: usize,
    /// Integration method.
    pub method: IntegratorMethod,
}

impl FlowMatchingIntegrator {
    /// Create a new integrator.
    pub fn new(num_steps: usize, method: IntegratorMethod) -> Self {
        Self { num_steps, method }
    }

    /// Integrate from t=1 to t=0.
    ///
    /// `velocity_fn`: closure returning `v_θ(x, t)` given `x` and `t ∈ [0, 1]`.
    pub fn integrate<F>(&self, x_init: &[f64], mut velocity_fn: F) -> Result<Vec<f64>, TensorError>
    where
        F: FnMut(&[f64], f64) -> Vec<f64>,
    {
        if x_init.is_empty() {
            return Err(make_err("FlowMatchingIntegrator: empty input"));
        }
        let mut x = x_init.to_vec();
        let dt = -1.0 / self.num_steps as f64;

        for step in 0..self.num_steps {
            let t = 1.0 + step as f64 * dt;
            match self.method {
                IntegratorMethod::Euler => {
                    let v = velocity_fn(&x, t);
                    x = axpy(&x, dt, &v);
                }
                IntegratorMethod::Heun => {
                    let v1 = velocity_fn(&x, t);
                    let x_pred = axpy(&x, dt, &v1);
                    let t_next = (t + dt).max(0.0);
                    let v2 = velocity_fn(&x_pred, t_next);
                    let v_avg: Vec<f64> = v1
                        .iter()
                        .zip(v2.iter())
                        .map(|(&a, &b)| 0.5 * (a + b))
                        .collect();
                    x = axpy(&x, dt, &v_avg);
                }
                IntegratorMethod::DpmSolver => {
                    let v1 = velocity_fn(&x, t);
                    let t_mid = (t + 0.5 * dt).max(0.0);
                    let x_mid = axpy(&x, 0.5 * dt, &v1);
                    let v2 = velocity_fn(&x_mid, t_mid);
                    x = axpy(&x, dt, &v2);
                }
            }
        }
        Ok(x)
    }

    /// Single Euler step (convenience wrapper).
    pub fn euler_step(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64> {
        axpy(x, dt, v)
    }
}
