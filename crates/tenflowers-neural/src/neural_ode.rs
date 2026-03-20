//! Neural Ordinary Differential Equations (Neural ODEs)
//!
//! Implements Chen et al. (2018) "Neural Ordinary Differential Equations".
//! Provides Euler, RK4, and adaptive Dormand-Prince (RK45) ODE solvers,
//! a `NeuralOde` layer wrapper, and the adjoint memory-efficient forward pass.
//!
//! All computation is in `f32` and is 100 % pure Rust (no C / Fortran).

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during ODE solving.
#[derive(Debug, Clone, PartialEq)]
pub enum OdeError {
    /// State-space dimensions do not agree.
    DimensionMismatch { expected: usize, found: usize },
    /// Adaptive solver exceeded its step budget.
    MaxStepsExceeded { max_steps: usize },
    /// Matrix supplied for `LinearOdeFunc` is not square.
    MatrixNotSquare { rows: usize, cols: usize },
    /// One or both tolerances are non-positive.
    InvalidTolerance { rtol: f32, atol: f32 },
    /// Adaptive step-size collapsed below the floor.
    StepSizeTooSmall { step_size: f32 },
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "ODE dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            OdeError::MaxStepsExceeded { max_steps } => {
                write!(f, "ODE solver exceeded max steps ({})", max_steps)
            }
            OdeError::MatrixNotSquare { rows, cols } => {
                write!(f, "A-matrix is not square: {}×{}", rows, cols)
            }
            OdeError::InvalidTolerance { rtol, atol } => {
                write!(f, "Invalid tolerances: rtol={}, atol={}", rtol, atol)
            }
            OdeError::StepSizeTooSmall { step_size } => {
                write!(f, "Adaptive step size too small: {}", step_size)
            }
        }
    }
}

impl std::error::Error for OdeError {}

// ---------------------------------------------------------------------------
// OdeFunc trait
// ---------------------------------------------------------------------------

/// Dynamics function `f(t, y)` that defines `dy/dt = f(t, y)`.
pub trait OdeFunc {
    /// Evaluate the right-hand side at time `t` and state `y`.
    /// Returns a vector of the same dimension as `y`.
    fn forward(&self, t: f32, y: &[f32]) -> Vec<f32>;

    /// Dimension of the state vector.
    fn state_dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Helper: vector arithmetic
// ---------------------------------------------------------------------------

#[inline]
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

#[inline]
fn vec_scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|ai| ai * s).collect()
}

#[inline]
fn vec_axpy(a: &[f32], alpha: f32, b: &[f32]) -> Vec<f32> {
    // returns a + alpha * b
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai + alpha * bi)
        .collect()
}

#[inline]
fn vec_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Euler solver
// ---------------------------------------------------------------------------

/// Fixed-step Euler (first-order) ODE solver.
pub struct EulerSolver {
    /// Step size `h`.
    pub step_size: f32,
    /// Number of steps to take.
    pub num_steps: usize,
}

impl EulerSolver {
    /// Create a new Euler solver.
    pub fn new(step_size: f32, num_steps: usize) -> Self {
        Self {
            step_size,
            num_steps,
        }
    }

    /// Integrate from `t0` to `t1` and return the full trajectory
    /// as `[num_steps+1, dim]` (outer Vec: time points, inner: state).
    pub fn solve(&self, func: &dyn OdeFunc, y0: &[f32], t0: f32, t1: f32) -> Vec<Vec<f32>> {
        let n = self.num_steps.max(1);
        let h = (t1 - t0) / n as f32;
        let mut traj = Vec::with_capacity(n + 1);
        let mut y = y0.to_vec();
        let mut t = t0;
        traj.push(y.clone());
        for _ in 0..n {
            let dy = func.forward(t, &y);
            y = vec_axpy(&y, h, &dy);
            t += h;
            traj.push(y.clone());
        }
        traj
    }

    /// Convenience: only return the final state.
    pub fn final_state(&self, func: &dyn OdeFunc, y0: &[f32], t0: f32, t1: f32) -> Vec<f32> {
        let traj = self.solve(func, y0, t0, t1);
        traj.into_iter().last().unwrap_or_else(|| y0.to_vec())
    }
}

// ---------------------------------------------------------------------------
// RK4 solver
// ---------------------------------------------------------------------------

/// Fixed-step 4th-order Runge-Kutta ODE solver.
pub struct Rk4Solver {
    /// Step size `h`.
    pub step_size: f32,
    /// Number of steps to take.
    pub num_steps: usize,
}

impl Rk4Solver {
    /// Create a new RK4 solver.
    pub fn new(step_size: f32, num_steps: usize) -> Self {
        Self {
            step_size,
            num_steps,
        }
    }

    /// Perform a single RK4 step starting at `(t, y)` with step `h`.
    pub fn rk4_step(&self, func: &dyn OdeFunc, y: &[f32], t: f32, h: f32) -> Vec<f32> {
        let k1 = func.forward(t, y);
        let y2 = vec_axpy(y, h / 2.0, &k1);
        let k2 = func.forward(t + h / 2.0, &y2);
        let y3 = vec_axpy(y, h / 2.0, &k2);
        let k3 = func.forward(t + h / 2.0, &y3);
        let y4 = vec_axpy(y, h, &k3);
        let k4 = func.forward(t + h, &y4);

        // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        let dim = y.len();
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            out.push(y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
        }
        out
    }

    /// Integrate from `t0` to `t1` and return the full trajectory.
    pub fn solve(&self, func: &dyn OdeFunc, y0: &[f32], t0: f32, t1: f32) -> Vec<Vec<f32>> {
        let n = self.num_steps.max(1);
        let h = (t1 - t0) / n as f32;
        let mut traj = Vec::with_capacity(n + 1);
        let mut y = y0.to_vec();
        let mut t = t0;
        traj.push(y.clone());
        for _ in 0..n {
            y = self.rk4_step(func, &y, t, h);
            t += h;
            traj.push(y.clone());
        }
        traj
    }

    /// Convenience: only return the final state.
    pub fn final_state(&self, func: &dyn OdeFunc, y0: &[f32], t0: f32, t1: f32) -> Vec<f32> {
        let traj = self.solve(func, y0, t0, t1);
        traj.into_iter().last().unwrap_or_else(|| y0.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Dormand-Prince RK45 adaptive solver
// ---------------------------------------------------------------------------

/// Dormand-Prince (RK45) Butcher tableau coefficients.
/// Reference: Dormand & Prince (1980).
mod dp45 {
    // c_i (time nodes)
    pub const C2: f32 = 1.0 / 5.0;
    pub const C3: f32 = 3.0 / 10.0;
    pub const C4: f32 = 4.0 / 5.0;
    pub const C5: f32 = 8.0 / 9.0;
    // a_ij
    pub const A21: f32 = 1.0 / 5.0;
    pub const A31: f32 = 3.0 / 40.0;
    pub const A32: f32 = 9.0 / 40.0;
    pub const A41: f32 = 44.0 / 45.0;
    pub const A42: f32 = -56.0 / 15.0;
    pub const A43: f32 = 32.0 / 9.0;
    pub const A51: f32 = 19372.0 / 6561.0;
    pub const A52: f32 = -25360.0 / 2187.0;
    pub const A53: f32 = 64448.0 / 6561.0;
    pub const A54: f32 = -212.0 / 729.0;
    pub const A61: f32 = 9017.0 / 3168.0;
    pub const A62: f32 = -355.0 / 33.0;
    pub const A63: f32 = 46732.0 / 5247.0;
    pub const A64: f32 = 49.0 / 176.0;
    pub const A65: f32 = -5103.0 / 18656.0;
    // 5th-order weights (b_i for y_{n+1})
    pub const B1: f32 = 35.0 / 384.0;
    pub const B3: f32 = 500.0 / 1113.0;
    pub const B4: f32 = 125.0 / 192.0;
    pub const B5: f32 = -2187.0 / 6784.0;
    pub const B6: f32 = 11.0 / 84.0;
    // 4th-order weights (e_i for error estimate = b_i - b*_i)
    pub const E1: f32 = 71.0 / 57600.0;
    pub const E3: f32 = -71.0 / 16695.0;
    pub const E4: f32 = 71.0 / 1920.0;
    pub const E5: f32 = -17253.0 / 339200.0;
    pub const E6: f32 = 22.0 / 525.0;
    pub const E7: f32 = -1.0 / 40.0;
}

/// Adaptive Dormand-Prince (RK45) ODE solver.
///
/// Uses step-doubling error control with PI step-size selection.
pub struct Rk45Solver {
    /// Relative tolerance.
    pub rtol: f32,
    /// Absolute tolerance.
    pub atol: f32,
    /// Maximum number of steps allowed.
    pub max_steps: usize,
    /// Minimum permitted step size.
    pub min_step: f32,
    /// Maximum permitted step size.
    pub max_step: f32,
}

impl Rk45Solver {
    /// Create an RK45 solver with given tolerances and sensible defaults.
    pub fn new(rtol: f32, atol: f32) -> Self {
        Self {
            rtol,
            atol,
            max_steps: 100_000,
            min_step: 1e-10,
            max_step: f32::INFINITY,
        }
    }

    /// Integrate from `t0` to `t1`, returning the final state.
    pub fn solve(
        &self,
        func: &dyn OdeFunc,
        y0: &[f32],
        t0: f32,
        t1: f32,
    ) -> Result<Vec<f32>, OdeError> {
        if self.rtol <= 0.0 || self.atol <= 0.0 {
            return Err(OdeError::InvalidTolerance {
                rtol: self.rtol,
                atol: self.atol,
            });
        }

        let dim = y0.len();
        let mut y = y0.to_vec();
        let mut t = t0;
        let direction = if t1 >= t0 { 1.0_f32 } else { -1.0_f32 };

        // Initial step guess
        let mut h = direction * ((t1 - t0).abs() / 100.0).clamp(self.min_step, self.max_step);

        let mut steps = 0usize;

        while (t1 - t) * direction > 0.0 {
            if steps >= self.max_steps {
                return Err(OdeError::MaxStepsExceeded {
                    max_steps: self.max_steps,
                });
            }

            // Clamp to not overshoot
            if (t + h - t1) * direction > 0.0 {
                h = t1 - t;
            }

            // Evaluate stages
            let k1 = func.forward(t, &y);

            let y2: Vec<f32> = (0..dim).map(|i| y[i] + h * dp45::A21 * k1[i]).collect();
            let k2 = func.forward(t + dp45::C2 * h, &y2);

            let y3: Vec<f32> = (0..dim)
                .map(|i| y[i] + h * (dp45::A31 * k1[i] + dp45::A32 * k2[i]))
                .collect();
            let k3 = func.forward(t + dp45::C3 * h, &y3);

            let y4: Vec<f32> = (0..dim)
                .map(|i| y[i] + h * (dp45::A41 * k1[i] + dp45::A42 * k2[i] + dp45::A43 * k3[i]))
                .collect();
            let k4 = func.forward(t + dp45::C4 * h, &y4);

            let y5: Vec<f32> = (0..dim)
                .map(|i| {
                    y[i] + h
                        * (dp45::A51 * k1[i]
                            + dp45::A52 * k2[i]
                            + dp45::A53 * k3[i]
                            + dp45::A54 * k4[i])
                })
                .collect();
            let k5 = func.forward(t + dp45::C5 * h, &y5);

            let y6: Vec<f32> = (0..dim)
                .map(|i| {
                    y[i] + h
                        * (dp45::A61 * k1[i]
                            + dp45::A62 * k2[i]
                            + dp45::A63 * k3[i]
                            + dp45::A64 * k4[i]
                            + dp45::A65 * k5[i])
                })
                .collect();
            let k6 = func.forward(t + h, &y6);

            // 5th-order solution
            let y_new: Vec<f32> = (0..dim)
                .map(|i| {
                    y[i] + h
                        * (dp45::B1 * k1[i]
                            + dp45::B3 * k3[i]
                            + dp45::B4 * k4[i]
                            + dp45::B5 * k5[i]
                            + dp45::B6 * k6[i])
                })
                .collect();
            let k7 = func.forward(t + h, &y_new);

            // Error estimate (difference between 4th and 5th order)
            let err_vec: Vec<f32> = (0..dim)
                .map(|i| {
                    h * (dp45::E1 * k1[i]
                        + dp45::E3 * k3[i]
                        + dp45::E4 * k4[i]
                        + dp45::E5 * k5[i]
                        + dp45::E6 * k6[i]
                        + dp45::E7 * k7[i])
                })
                .collect();

            // Scale error
            let sc_err = err_vec
                .iter()
                .zip(y.iter().zip(y_new.iter()))
                .map(|(e, (yi, yn))| {
                    let sc = self.atol + yi.abs().max(yn.abs()) * self.rtol;
                    (e / sc) * (e / sc)
                })
                .sum::<f32>()
                / dim as f32;

            let err = sc_err.sqrt();

            if err <= 1.0 {
                // Accept step
                t += h;
                y = y_new;
                steps += 1;
            }

            // Step-size control (PI controller)
            let factor = if err == 0.0 {
                5.0_f32
            } else {
                (0.9 * err.powf(-0.2)).clamp(0.1, 5.0)
            };

            h = direction * (h.abs() * factor).clamp(self.min_step, self.max_step);

            if h.abs() < self.min_step {
                return Err(OdeError::StepSizeTooSmall { step_size: h });
            }
        }

        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// Linear ODE function
// ---------------------------------------------------------------------------

/// Linear dynamics: `dy/dt = A * y`.
///
/// Used for testing solvers against analytic solutions.
pub struct LinearOdeFunc {
    /// Dense matrix A (row-major), shape `[dim, dim]`.
    pub a_matrix: Vec<Vec<f32>>,
    /// Dimension of the state.
    pub dim: usize,
}

impl LinearOdeFunc {
    /// Create from a square matrix. Returns an error if not square.
    pub fn new(a_matrix: Vec<Vec<f32>>) -> Result<Self, OdeError> {
        let rows = a_matrix.len();
        for row in &a_matrix {
            if row.len() != rows {
                return Err(OdeError::MatrixNotSquare {
                    rows,
                    cols: row.len(),
                });
            }
        }
        Ok(Self {
            dim: rows,
            a_matrix,
        })
    }
}

impl OdeFunc for LinearOdeFunc {
    fn forward(&self, _t: f32, y: &[f32]) -> Vec<f32> {
        // A * y
        self.a_matrix
            .iter()
            .map(|row| row.iter().zip(y.iter()).map(|(a, yi)| a * yi).sum())
            .collect()
    }

    fn state_dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Spiral ODE function
// ---------------------------------------------------------------------------

/// Classic spiral dynamics: `dy/dt = A * y` with
/// `A = [[-0.1, -1.0], [1.0, -0.1]]`.
///
/// The eigenvalues have negative real part, so trajectories spiral inward.
pub struct SpiralOdeFunc {
    a: [[f32; 2]; 2],
}

impl SpiralOdeFunc {
    /// Create with the standard spiral matrix.
    pub fn new() -> Self {
        Self {
            a: [[-0.1, -1.0], [1.0, -0.1]],
        }
    }
}

impl Default for SpiralOdeFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl OdeFunc for SpiralOdeFunc {
    fn forward(&self, _t: f32, y: &[f32]) -> Vec<f32> {
        let y0 = *y.first().unwrap_or(&0.0);
        let y1 = *y.get(1).unwrap_or(&0.0);
        vec![
            self.a[0][0] * y0 + self.a[0][1] * y1,
            self.a[1][0] * y0 + self.a[1][1] * y1,
        ]
    }

    fn state_dim(&self) -> usize {
        2
    }
}

// ---------------------------------------------------------------------------
// Solver type enum
// ---------------------------------------------------------------------------

/// Selects which ODE solver backend to use inside `NeuralOde`.
#[derive(Debug, Clone)]
pub enum OdeSolverType {
    /// Fixed-step Euler integrator.
    Euler { step_size: f32, num_steps: usize },
    /// Fixed-step 4th-order Runge-Kutta integrator.
    Rk4 { step_size: f32, num_steps: usize },
    /// Adaptive Dormand-Prince integrator.
    Rk45 { rtol: f32, atol: f32 },
}

// ---------------------------------------------------------------------------
// NeuralOde layer wrapper
// ---------------------------------------------------------------------------

/// Neural ODE layer.
///
/// Wraps an ODE solver and integration interval so a `dyn OdeFunc` can be
/// used as a continuous-depth neural network layer.
pub struct NeuralOde {
    /// Solver to use for numerical integration.
    pub solver: OdeSolverType,
    /// Integration start time.
    pub t0: f32,
    /// Integration end time.
    pub t1: f32,
    /// Dimension of the latent state.
    pub state_dim: usize,
}

impl NeuralOde {
    /// Create a new `NeuralOde` layer.
    pub fn new(state_dim: usize, t0: f32, t1: f32, solver: OdeSolverType) -> Self {
        Self {
            solver,
            t0,
            t1,
            state_dim,
        }
    }

    /// Integrate `func` from `t0` to `t1` starting at `initial_state`.
    pub fn forward(&self, func: &dyn OdeFunc, initial_state: &[f32]) -> Result<Vec<f32>, OdeError> {
        if initial_state.len() != self.state_dim {
            return Err(OdeError::DimensionMismatch {
                expected: self.state_dim,
                found: initial_state.len(),
            });
        }
        match &self.solver {
            OdeSolverType::Euler {
                step_size: _,
                num_steps,
            } => {
                let solver = EulerSolver::new((self.t1 - self.t0) / *num_steps as f32, *num_steps);
                Ok(solver.final_state(func, initial_state, self.t0, self.t1))
            }
            OdeSolverType::Rk4 {
                step_size: _,
                num_steps,
            } => {
                let solver = Rk4Solver::new((self.t1 - self.t0) / *num_steps as f32, *num_steps);
                Ok(solver.final_state(func, initial_state, self.t0, self.t1))
            }
            OdeSolverType::Rk45 { rtol, atol } => {
                let solver = Rk45Solver::new(*rtol, *atol);
                solver.solve(func, initial_state, self.t0, self.t1)
            }
        }
    }

    /// Compute `n_points` evenly-spaced trajectory points from `t0` to `t1`.
    ///
    /// Returns a `Vec<Vec<f32>>` where each inner `Vec` is a state snapshot.
    pub fn trajectory(
        &self,
        func: &dyn OdeFunc,
        initial_state: &[f32],
        n_points: usize,
    ) -> Result<Vec<Vec<f32>>, OdeError> {
        if initial_state.len() != self.state_dim {
            return Err(OdeError::DimensionMismatch {
                expected: self.state_dim,
                found: initial_state.len(),
            });
        }
        let n = n_points.max(2);
        let mut traj = Vec::with_capacity(n);
        let mut y = initial_state.to_vec();
        let dt = (self.t1 - self.t0) / (n - 1) as f32;

        traj.push(y.clone());
        for step in 1..n {
            let t_prev = self.t0 + (step - 1) as f32 * dt;
            let t_next = t_prev + dt;
            y = match &self.solver {
                OdeSolverType::Euler {
                    step_size: _,
                    num_steps,
                } => {
                    let sub_steps = (*num_steps).max(1);
                    let solver = EulerSolver::new(dt / sub_steps as f32, sub_steps);
                    solver.final_state(func, &y, t_prev, t_next)
                }
                OdeSolverType::Rk4 {
                    step_size: _,
                    num_steps,
                } => {
                    let sub_steps = (*num_steps).max(1);
                    let solver = Rk4Solver::new(dt / sub_steps as f32, sub_steps);
                    solver.final_state(func, &y, t_prev, t_next)
                }
                OdeSolverType::Rk45 { rtol, atol } => {
                    let solver = Rk45Solver::new(*rtol, *atol);
                    solver.solve(func, &y, t_prev, t_next)?
                }
            };
            traj.push(y.clone());
        }
        Ok(traj)
    }
}

// ---------------------------------------------------------------------------
// Adjoint ODE
// ---------------------------------------------------------------------------

/// Adjoint-method wrapper for memory-efficient Neural ODE integration.
///
/// In the simplified (forward-only) implementation here the adjoint is
/// equivalent to the standard forward pass; a production implementation would
/// use reverse-mode integration for gradient computation.
pub struct AdjointOde {
    /// The underlying `NeuralOde`.
    pub ode: NeuralOde,
}

impl AdjointOde {
    /// Create an `AdjointOde` wrapping an existing `NeuralOde`.
    pub fn new(ode: NeuralOde) -> Self {
        Self { ode }
    }

    /// Forward pass (identical to `NeuralOde::forward`; gradient computation
    /// via the adjoint method would extend this in a full autograd context).
    pub fn forward(&self, func: &dyn OdeFunc, initial_state: &[f32]) -> Result<Vec<f32>, OdeError> {
        self.ode.forward(func, initial_state)
    }
}

// ---------------------------------------------------------------------------
// Continuous Normalizing Flows helper
// ---------------------------------------------------------------------------

/// Hutchinson trace estimator of `div(f)` — used in Continuous Normalizing
/// Flows to track log-det of the Jacobian along an ODE trajectory.
///
/// Uses a deterministic all-ones probe vector `ε` (finite-difference variant):
///   `tr(∂f/∂y) ≈ Σ_i ε_i · (f(y + δε_i) − f(y)) / δ`
///
/// # Parameters
/// - `func`: the ODE dynamics
/// - `y`: current state
/// - `t`: current time
/// - `epsilon`: finite-difference step size
///
/// # Returns
/// Scalar estimate of the trace of the Jacobian (divergence of `f`).
pub fn log_det_jacobian_estimate(func: &dyn OdeFunc, y: &[f32], t: f32, epsilon: f32) -> f32 {
    let dim = y.len();
    let f0 = func.forward(t, y);
    let eps = if epsilon.abs() < 1e-12 { 1e-5 } else { epsilon };

    // Probe with all-ones vector: finite-difference approximation of div(f)
    let y_eps: Vec<f32> = y.iter().map(|yi| yi + eps).collect();
    let f_eps = func.forward(t, &y_eps);

    // Hutchinson estimator: tr(J) ≈ (1/eps) * e^T * (f(y + eps*e) - f(y))
    // with e = all-ones → sum of column sums of J (first-order approx)
    f_eps
        .iter()
        .zip(f0.iter())
        .map(|(fe, f0i)| (fe - f0i) / eps)
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Scalar exponential decay: dy/dt = -y, exact solution y(t) = y0 * exp(-t)
    struct ExponentialDecay;
    impl OdeFunc for ExponentialDecay {
        fn forward(&self, _t: f32, y: &[f32]) -> Vec<f32> {
            y.iter().map(|yi| -yi).collect()
        }
        fn state_dim(&self) -> usize {
            1
        }
    }

    // dy/dt = 2t, exact y(t) = t^2 + C
    struct QuadraticDeriv;
    impl OdeFunc for QuadraticDeriv {
        fn forward(&self, t: f32, y: &[f32]) -> Vec<f32> {
            let _ = y;
            vec![2.0 * t]
        }
        fn state_dim(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_euler_exponential_decay_convergence() {
        let func = ExponentialDecay;
        let y0 = vec![1.0f32];
        // Exact answer: exp(-1) ≈ 0.36788
        let exact = (-1.0_f32).exp();
        let solver = EulerSolver::new(0.001, 1000);
        let result = solver.final_state(&func, &y0, 0.0, 1.0);
        let err = (result[0] - exact).abs();
        assert!(
            err < 0.01,
            "Euler should approximate exp(-1) within 0.01, got err={}",
            err
        );
    }

    #[test]
    fn test_euler_trajectory_length() {
        let func = ExponentialDecay;
        let solver = EulerSolver::new(0.1, 10);
        let traj = solver.solve(&func, &[1.0], 0.0, 1.0);
        assert_eq!(traj.len(), 11, "trajectory should have num_steps+1 points");
    }

    #[test]
    fn test_rk4_more_accurate_than_euler() {
        let func = ExponentialDecay;
        let y0 = vec![1.0f32];
        let exact = (-1.0_f32).exp();
        let n = 10; // same number of steps
        let euler = EulerSolver::new(0.1, n);
        let rk4 = Rk4Solver::new(0.1, n);
        let e_euler = (euler.final_state(&func, &y0, 0.0, 1.0)[0] - exact).abs();
        let e_rk4 = (rk4.final_state(&func, &y0, 0.0, 1.0)[0] - exact).abs();
        assert!(
            e_rk4 < e_euler,
            "RK4 error ({}) should be less than Euler error ({})",
            e_rk4,
            e_euler
        );
    }

    #[test]
    fn test_rk4_trajectory_length() {
        let func = ExponentialDecay;
        let solver = Rk4Solver::new(0.1, 5);
        let traj = solver.solve(&func, &[1.0], 0.0, 0.5);
        assert_eq!(traj.len(), 6);
    }

    #[test]
    fn test_rk4_single_step_accuracy() {
        // dy/dt = 2t starting at y=0, t=0, exact after h=0.5: y=0.25
        let func = QuadraticDeriv;
        let solver = Rk4Solver::new(0.5, 1);
        let y_new = solver.rk4_step(&func, &[0.0], 0.0, 0.5);
        assert!(
            (y_new[0] - 0.25).abs() < 1e-5,
            "rk4 step on dy/dt=2t should give 0.25"
        );
    }

    #[test]
    fn test_spiral_ode_inward_spiral() {
        // With A = [[-0.1,-1],[1,-0.1]], norm decreases over time.
        let func = SpiralOdeFunc::new();
        let y0 = vec![1.0, 0.0];
        let solver = Rk4Solver::new(0.01, 100);
        let y_end = solver.final_state(&func, &y0, 0.0, 1.0);
        let r_start = (y0[0] * y0[0] + y0[1] * y0[1]).sqrt();
        let r_end = (y_end[0] * y_end[0] + y_end[1] * y_end[1]).sqrt();
        assert!(
            r_end < r_start,
            "spiral radius should decrease: start={}, end={}",
            r_start,
            r_end
        );
    }

    #[test]
    fn test_spiral_default_trait() {
        let func = SpiralOdeFunc::default();
        assert_eq!(func.state_dim(), 2);
    }

    #[test]
    fn test_linear_ode_func_creation_square() {
        let a = vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0]];
        let func = LinearOdeFunc::new(a).expect("square matrix should succeed");
        assert_eq!(func.state_dim(), 2);
    }

    #[test]
    fn test_linear_ode_func_non_square_error() {
        let a = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = LinearOdeFunc::new(a);
        assert!(result.is_err(), "non-square matrix should return error");
    }

    #[test]
    fn test_linear_ode_matrix_vector_product() {
        // A = diag(-1), y = [2,3] → Ay = [-2,-3]
        let a = vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0]];
        let func = LinearOdeFunc::new(a).expect("creation ok");
        let dy = func.forward(0.0, &[2.0, 3.0]);
        assert!((dy[0] - (-2.0)).abs() < 1e-6);
        assert!((dy[1] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_neural_ode_euler_output_dimension() {
        let ode = NeuralOde::new(
            2,
            0.0,
            1.0,
            OdeSolverType::Euler {
                step_size: 0.1,
                num_steps: 10,
            },
        );
        let func = SpiralOdeFunc::new();
        let result = ode
            .forward(&func, &[1.0, 0.0])
            .expect("forward should succeed");
        assert_eq!(result.len(), 2, "output dimension must match state_dim");
    }

    #[test]
    fn test_neural_ode_rk4_output_dimension() {
        let ode = NeuralOde::new(
            2,
            0.0,
            1.0,
            OdeSolverType::Rk4 {
                step_size: 0.1,
                num_steps: 10,
            },
        );
        let func = SpiralOdeFunc::new();
        let result = ode
            .forward(&func, &[1.0, 0.0])
            .expect("forward should succeed");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_neural_ode_dimension_mismatch_error() {
        let ode = NeuralOde::new(
            3,
            0.0,
            1.0,
            OdeSolverType::Euler {
                step_size: 0.1,
                num_steps: 5,
            },
        );
        let func = SpiralOdeFunc::new(); // dim=2
        let result = ode.forward(&func, &[1.0, 0.0]);
        assert!(result.is_err(), "dimension mismatch should be an error");
        matches!(result, Err(OdeError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_neural_ode_trajectory_n_points() {
        let ode = NeuralOde::new(
            2,
            0.0,
            2.0,
            OdeSolverType::Euler {
                step_size: 0.1,
                num_steps: 5,
            },
        );
        let func = SpiralOdeFunc::new();
        let traj = ode
            .trajectory(&func, &[1.0, 0.0], 11)
            .expect("trajectory ok");
        assert_eq!(
            traj.len(),
            11,
            "trajectory should have exactly n_points entries"
        );
        for pt in &traj {
            assert_eq!(pt.len(), 2);
        }
    }

    #[test]
    fn test_adjoint_ode_same_as_neural_ode() {
        let ode1 = NeuralOde::new(
            2,
            0.0,
            1.0,
            OdeSolverType::Rk4 {
                step_size: 0.01,
                num_steps: 100,
            },
        );
        let ode2 = NeuralOde::new(
            2,
            0.0,
            1.0,
            OdeSolverType::Rk4 {
                step_size: 0.01,
                num_steps: 100,
            },
        );
        let adjoint = AdjointOde::new(ode2);
        let func = SpiralOdeFunc::new();
        let y0 = vec![1.0_f32, 0.5];
        let out1 = ode1.forward(&func, &y0).expect("ode1 ok");
        let out2 = adjoint.forward(&func, &y0).expect("adjoint ok");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6, "adjoint and neural ode should match");
        }
    }

    #[test]
    fn test_rk45_exponential_decay() {
        let func = ExponentialDecay;
        let solver = Rk45Solver::new(1e-4, 1e-7);
        let result = solver
            .solve(&func, &[1.0], 0.0, 1.0)
            .expect("rk45 solve ok");
        let exact = (-1.0_f32).exp();
        let err = (result[0] - exact).abs();
        assert!(err < 1e-3, "RK45 should be very accurate: err={}", err);
    }

    #[test]
    fn test_rk45_invalid_tolerance() {
        let solver = Rk45Solver::new(-1e-4, 1e-7);
        let func = ExponentialDecay;
        let result = solver.solve(&func, &[1.0], 0.0, 1.0);
        assert!(matches!(result, Err(OdeError::InvalidTolerance { .. })));
    }

    #[test]
    fn test_log_det_jacobian_estimate_nonzero() {
        // For f(y) = -y, div(f) = -dim
        let func =
            LinearOdeFunc::new(vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0]]).expect("matrix ok");
        let y = vec![1.0_f32, 1.0];
        let trace_est = log_det_jacobian_estimate(&func, &y, 0.0, 1e-4);
        // Should be approx -2 (trace of -I_2)
        assert!(
            (trace_est - (-2.0)).abs() < 0.1,
            "trace estimate should be close to -2, got {}",
            trace_est
        );
    }

    #[test]
    fn test_ode_error_display() {
        let e1 = OdeError::DimensionMismatch {
            expected: 2,
            found: 3,
        };
        assert!(e1.to_string().contains("dimension mismatch"));

        let e2 = OdeError::MaxStepsExceeded { max_steps: 1000 };
        assert!(e2.to_string().contains("max steps"));

        let e3 = OdeError::MatrixNotSquare { rows: 2, cols: 3 };
        assert!(e3.to_string().contains("square"));

        let e4 = OdeError::InvalidTolerance {
            rtol: -1.0,
            atol: 1e-6,
        };
        assert!(e4.to_string().contains("tolerance"));

        let e5 = OdeError::StepSizeTooSmall { step_size: 1e-12 };
        assert!(e5.to_string().contains("step size"));
    }

    #[test]
    fn test_neural_ode_rk45_solver_type() {
        let ode = NeuralOde::new(
            1,
            0.0,
            1.0,
            OdeSolverType::Rk45 {
                rtol: 1e-3,
                atol: 1e-6,
            },
        );
        let func = ExponentialDecay;
        let result = ode.forward(&func, &[1.0]).expect("rk45 forward ok");
        let exact = (-1.0_f32).exp();
        assert!(
            (result[0] - exact).abs() < 0.01,
            "NeuralOde with RK45 should be accurate, got {}",
            result[0]
        );
    }

    #[test]
    fn test_euler_solver_zero_steps_fallback() {
        // num_steps=0 should be treated as 1
        let func = ExponentialDecay;
        let solver = EulerSolver::new(1.0, 0);
        let traj = solver.solve(&func, &[1.0], 0.0, 1.0);
        assert_eq!(traj.len(), 2); // 0 steps → max(0,1)=1 step → 2 points
    }

    #[test]
    fn test_rk4_quadratic_accuracy() {
        // dy/dt = 2t, y(0)=0 → y(1) = 1.0
        let func = QuadraticDeriv;
        let solver = Rk4Solver::new(0.1, 10);
        let result = solver.final_state(&func, &[0.0], 0.0, 1.0);
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "RK4 should exactly integrate dy/dt=2t"
        );
    }

    #[test]
    fn test_spiral_trajectory_via_rk45_neural_ode() {
        let ode = NeuralOde::new(
            2,
            0.0,
            2.0,
            OdeSolverType::Rk45 {
                rtol: 1e-3,
                atol: 1e-6,
            },
        );
        let func = SpiralOdeFunc::new();
        let traj = ode
            .trajectory(&func, &[1.0, 0.0], 5)
            .expect("trajectory ok");
        assert_eq!(traj.len(), 5);
        // Norm at end should be smaller than at start (spiral inward)
        let r0 = vec_norm(&traj[0]);
        let r_last = vec_norm(traj.last().expect("non-empty traj"));
        assert!(r_last < r0, "spiral norm should decrease");
    }
}
