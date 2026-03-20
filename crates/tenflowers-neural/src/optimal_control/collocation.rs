//! Direct Collocation trajectory optimization.

use super::dynamics::DynamicsModel;
use tenflowers_core::{Result, TensorError};

// Direct Collocation (Trajectory Optimization)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for direct collocation trajectory optimization.
#[derive(Clone, Debug)]
pub struct TrajectoryOptConfig {
    /// Number of time steps in the trajectory.
    pub horizon: usize,
    /// Maximum number of gradient descent iterations.
    pub max_iter: usize,
    /// Gradient descent step size.
    pub lr: f64,
    /// Convergence tolerance on objective change.
    pub tolerance: f64,
}

impl Default for TrajectoryOptConfig {
    fn default() -> Self {
        Self {
            horizon: 50,
            max_iter: 500,
            lr: 1e-3,
            tolerance: 1e-6,
        }
    }
}

/// Result of direct collocation trajectory optimization.
#[derive(Clone, Debug)]
pub struct TrajectoryOptResult {
    /// Optimized state trajectory (horizon + 1 states).
    pub states: Vec<Vec<f64>>,
    /// Optimized control sequence (horizon controls).
    pub controls: Vec<Vec<f64>>,
    /// Total trajectory cost at convergence.
    pub total_cost: f64,
    /// L2 norm of defect constraints `||x_{t+1} - f(x_t, u_t)||`.
    pub defect_norm: f64,
}

/// Direct Collocation: optimize states and controls jointly via gradient descent.
///
/// The objective is:
/// ```text
/// minimize  Σ_t l(x_t, u_t) + l_f(x_T)  +  w_defect * ||x_{t+1} - f(x_t, u_t)||²
/// ```
/// where defect constraints enforce the dynamics.
pub struct DirectCollocation;

impl DirectCollocation {
    /// Create a new `DirectCollocation` optimizer.
    pub fn new() -> Self {
        Self
    }

    /// Optimize a trajectory starting from `x_init` states and `u_init` controls.
    ///
    /// The state at `t=0` is fixed to `x_init[0]`. States at `t > 0` and controls
    /// are jointly optimized via gradient descent.
    pub fn optimize<F, G>(
        &self,
        x_init: Vec<Vec<f64>>,
        u_init: Vec<Vec<f64>>,
        cost_fn: &F,
        terminal_fn: &G,
        dynamics: &dyn DynamicsModel,
        config: &TrajectoryOptConfig,
    ) -> Result<TrajectoryOptResult>
    where
        F: Fn(&[f64], &[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let h = config.horizon;
        let n = dynamics.state_dim();
        let m = dynamics.action_dim();
        let w_defect = 10.0; // defect penalty weight

        if x_init.len() != h + 1 {
            return Err(TensorError::invalid_argument(format!(
                "DirectCollocation: x_init length {} != horizon+1 = {}",
                x_init.len(),
                h + 1
            )));
        }
        if u_init.len() != h {
            return Err(TensorError::invalid_argument(format!(
                "DirectCollocation: u_init length {} != horizon = {}",
                u_init.len(),
                h
            )));
        }

        let mut states = x_init;
        let mut controls = u_init;
        let x0 = states[0].clone(); // Fixed initial state

        let mut prev_obj = f64::INFINITY;
        let finite_diff_h = 1e-5;

        for _iter in 0..config.max_iter {
            // ── Compute objective ──────────────────────────────────────────
            let mut obj = 0.0;
            for t in 0..h {
                obj += cost_fn(&states[t], &controls[t]);
            }
            obj += terminal_fn(&states[h]);

            // Defect penalty
            let mut defect_sq = 0.0;
            for t in 0..h {
                let x_next_pred = dynamics.step(&states[t], &controls[t]);
                for i in 0..n {
                    let d = states[t + 1][i] - x_next_pred[i];
                    defect_sq += d * d;
                }
            }
            obj += w_defect * defect_sq;

            // Check convergence
            if (prev_obj - obj).abs() < config.tolerance {
                break;
            }
            prev_obj = obj;

            // ── Gradient descent step via numerical gradients ──────────────
            // Gradient w.r.t. states[t] for t in 1..=h
            for t in 1..=h {
                for i in 0..n {
                    let mut sp = states.clone();
                    let mut sm = states.clone();
                    sp[t][i] += finite_diff_h;
                    sm[t][i] -= finite_diff_h;

                    // Recompute objective with perturbed state
                    let obj_p =
                        Self::eval_obj(&sp, &controls, cost_fn, terminal_fn, dynamics, h, w_defect);
                    let obj_m =
                        Self::eval_obj(&sm, &controls, cost_fn, terminal_fn, dynamics, h, w_defect);
                    let grad = (obj_p - obj_m) / (2.0 * finite_diff_h);
                    states[t][i] -= config.lr * grad;
                }
                // Keep x0 fixed
                states[0] = x0.clone();
            }

            // Gradient w.r.t. controls[t]
            for t in 0..h {
                for j in 0..m {
                    let mut cp = controls.clone();
                    let mut cm = controls.clone();
                    cp[t][j] += finite_diff_h;
                    cm[t][j] -= finite_diff_h;
                    let obj_p =
                        Self::eval_obj(&states, &cp, cost_fn, terminal_fn, dynamics, h, w_defect);
                    let obj_m =
                        Self::eval_obj(&states, &cm, cost_fn, terminal_fn, dynamics, h, w_defect);
                    let grad = (obj_p - obj_m) / (2.0 * finite_diff_h);
                    controls[t][j] -= config.lr * grad;
                }
            }

            states[0] = x0.clone();
        }

        // Compute final metrics
        let mut total_cost = 0.0;
        for t in 0..h {
            total_cost += cost_fn(&states[t], &controls[t]);
        }
        total_cost += terminal_fn(&states[h]);

        let mut defect_norm = 0.0;
        for t in 0..h {
            let x_next_pred = dynamics.step(&states[t], &controls[t]);
            for i in 0..n {
                let d = states[t + 1][i] - x_next_pred[i];
                defect_norm += d * d;
            }
        }
        defect_norm = defect_norm.sqrt();

        Ok(TrajectoryOptResult {
            states,
            controls,
            total_cost,
            defect_norm,
        })
    }

    /// Internal helper: evaluate total objective (cost + defect penalty).
    fn eval_obj<F, G>(
        states: &[Vec<f64>],
        controls: &[Vec<f64>],
        cost_fn: &F,
        terminal_fn: &G,
        dynamics: &dyn DynamicsModel,
        h: usize,
        w_defect: f64,
    ) -> f64
    where
        F: Fn(&[f64], &[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let n = dynamics.state_dim();
        let mut obj = 0.0;
        for t in 0..h {
            obj += cost_fn(&states[t], &controls[t]);
        }
        obj += terminal_fn(&states[h]);
        let mut defect_sq = 0.0;
        for t in 0..h {
            let x_next_pred = dynamics.step(&states[t], &controls[t]);
            for i in 0..n {
                let d = states[t + 1][i] - x_next_pred[i];
                defect_sq += d * d;
            }
        }
        obj + w_defect * defect_sq
    }
}

impl Default for DirectCollocation {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
