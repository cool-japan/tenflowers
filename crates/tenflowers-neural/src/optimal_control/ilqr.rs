//! Iterative LQR (iLQR) with numerical differentiation helpers.

use super::dynamics::DynamicsModel;
use super::utils::{
    mat_add, mat_mul, mat_scale, mat_sub, mat_sym_pos_def_inv, mat_transpose, mat_vec_mul, vec_add,
    vec_dot, vec_scale,
};
use tenflowers_core::{Result, TensorError};

// iLQR (iterative LQR)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the iterative LQR solver.
#[derive(Clone, Debug)]
pub struct IlqrConfig {
    /// Planning horizon (number of control steps).
    pub horizon: usize,
    /// Maximum number of iLQR outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative cost decrease).
    pub tolerance: f64,
    /// Number of backtracking line search steps tried: α ∈ {1, 0.5, 0.25, 0.125, ...}.
    pub line_search_steps: usize,
    /// Tikhonov regularization added to Q_uu for numerical invertibility.
    pub mu: f64,
}

impl Default for IlqrConfig {
    fn default() -> Self {
        Self {
            horizon: 50,
            max_iter: 100,
            tolerance: 1e-6,
            line_search_steps: 10,
            mu: 1e-6,
        }
    }
}

/// Result of the iLQR optimizer.
#[derive(Clone, Debug)]
pub struct IlqrResult {
    /// Optimal state trajectory (horizon + 1 states).
    pub trajectory: Vec<Vec<f64>>,
    /// Optimal control sequence (horizon controls).
    pub controls: Vec<Vec<f64>>,
    /// Total trajectory cost at convergence.
    pub total_cost: f64,
    /// Number of outer iterations executed.
    pub n_iter: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

/// Configuration for a quadratic cost function `(x - goal)^T Q (x - goal) + u^T R u`.
#[derive(Clone, Debug)]
pub struct CostConfig {
    /// State deviation cost matrix Q (state_dim × state_dim).
    pub q: Vec<Vec<f64>>,
    /// Control cost matrix R (action_dim × action_dim).
    pub r: Vec<Vec<f64>>,
    /// Terminal state cost matrix (state_dim × state_dim).
    pub q_terminal: Vec<Vec<f64>>,
    /// Goal state (used for state deviation).
    pub goal: Vec<f64>,
}

/// Create a quadratic running-cost function `(x - goal)^T Q (x - goal) + u^T R u`.
pub fn quadratic_cost(config: &CostConfig) -> impl Fn(&[f64], &[f64]) -> f64 + '_ {
    move |x: &[f64], u: &[f64]| -> f64 {
        // State deviation cost
        let dx: Vec<f64> = x
            .iter()
            .zip(config.goal.iter())
            .map(|(xi, gi)| xi - gi)
            .collect();
        let q_dx = mat_vec_mul(&config.q, &dx);
        let state_cost = vec_dot(&dx, &q_dx);

        // Control cost
        let r_u = mat_vec_mul(&config.r, u);
        let ctrl_cost = vec_dot(u, &r_u);

        state_cost + ctrl_cost
    }
}

/// Create a quadratic terminal-cost function `(x - goal)^T Q_terminal (x - goal)`.
fn quadratic_terminal_cost(config: &CostConfig) -> impl Fn(&[f64]) -> f64 + '_ {
    move |x: &[f64]| -> f64 {
        let dx: Vec<f64> = x
            .iter()
            .zip(config.goal.iter())
            .map(|(xi, gi)| xi - gi)
            .collect();
        let qt_dx = mat_vec_mul(&config.q_terminal, &dx);
        vec_dot(&dx, &qt_dx)
    }
}

/// Simulate a trajectory given dynamics, initial state, and control sequence.
pub fn forward_pass(
    dynamics: &dyn DynamicsModel,
    x0: &[f64],
    controls: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let h = controls.len();
    let mut traj = Vec::with_capacity(h + 1);
    let mut x = x0.to_vec();
    traj.push(x.clone());
    for u in controls {
        x = dynamics.step(&x, u);
        traj.push(x.clone());
    }
    traj
}

/// Compute the total trajectory cost.
pub fn total_cost<F, G>(
    traj: &[Vec<f64>],
    controls: &[Vec<f64>],
    cost_fn: &F,
    terminal_fn: &G,
) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    let h = controls.len();
    let mut cost = 0.0;
    for t in 0..h {
        cost += cost_fn(&traj[t], &controls[t]);
    }
    cost += terminal_fn(&traj[h]);
    cost
}

/// Iterative LQR solver for nonlinear trajectory optimization.
pub struct IlqrSolver;

impl IlqrSolver {
    /// Create a new `IlqrSolver`.
    pub fn new() -> Self {
        Self
    }

    /// Solve the iLQR problem.
    ///
    /// # Arguments
    /// - `dynamics`: The discrete-time dynamical system.
    /// - `x0`: Initial state.
    /// - `u_init`: Initial control sequence of length `config.horizon`.
    /// - `config`: Solver configuration.
    /// - `cost_fn`: Running cost `l(x, u)`.
    /// - `terminal_cost_fn`: Terminal cost `lf(xT)`.
    pub fn solve<F, G>(
        &self,
        dynamics: &dyn DynamicsModel,
        x0: &[f64],
        u_init: Vec<Vec<f64>>,
        config: &IlqrConfig,
        cost_fn: F,
        terminal_cost_fn: G,
    ) -> Result<IlqrResult>
    where
        F: Fn(&[f64], &[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let h = config.horizon;
        let n = dynamics.state_dim();
        let m = dynamics.action_dim();

        if u_init.len() != h {
            return Err(TensorError::invalid_argument(format!(
                "iLQR: u_init length {} != horizon {}",
                u_init.len(),
                h
            )));
        }

        let mut controls = u_init;
        let mut traj = forward_pass(dynamics, x0, &controls);
        let mut current_cost = total_cost(&traj, &controls, &cost_fn, &terminal_cost_fn);
        let mut converged = false;
        let mut n_iter = 0;

        for _iter in 0..config.max_iter {
            n_iter += 1;

            // ── Backward pass ──────────────────────────────────────────────
            // Terminal value function gradient and Hessian
            // We use numerical second derivatives for the cost function via finite differences.
            let h_eps = 1e-4;

            let x_t = &traj[h];
            let vx = numerical_grad_state(x_t, &terminal_cost_fn, h_eps);
            let vxx = numerical_hess_state(x_t, &terminal_cost_fn, h_eps);

            let mut vx_curr = vx;
            let mut vxx_curr = vxx;

            let mut k_feedforward: Vec<Vec<f64>> = Vec::with_capacity(h); // each: m-vector
            let mut k_feedback: Vec<Vec<Vec<f64>>> = Vec::with_capacity(h); // each: m × n

            for t in (0..h).rev() {
                let xt = &traj[t];
                let ut = &controls[t];

                // Linearize dynamics at (xt, ut)
                let (a, b) = dynamics.linearize(xt, ut);

                // Numerical cost gradient: l_x (n-vector), l_u (m-vector)
                let lx = numerical_grad_state_action_x(xt, ut, &cost_fn, h_eps);
                let lu = numerical_grad_state_action_u(xt, ut, &cost_fn, h_eps);

                // Numerical cost Hessians: l_xx (n×n), l_uu (m×m), l_ux (m×n)
                let lxx = numerical_hess_state_xx(xt, ut, &cost_fn, h_eps);
                let luu = numerical_hess_state_uu(xt, ut, &cost_fn, h_eps);
                let lux = numerical_hess_state_ux(xt, ut, &cost_fn, h_eps);

                let at = mat_transpose(&a);
                let bt = mat_transpose(&b);

                // Q-function approximation (Gauss-Newton style):
                // Q_x  = l_x  + A^T V_x
                // Q_u  = l_u  + B^T V_x
                // Q_xx = l_xx + A^T V_xx A
                // Q_uu = l_uu + B^T V_xx B  + μI  (regularization)
                // Q_ux = l_ux + B^T V_xx A
                let q_x = vec_add(&lx, &mat_vec_mul(&at, &vx_curr));
                let q_u = vec_add(&lu, &mat_vec_mul(&bt, &vx_curr));

                let at_vxx = mat_mul(&at, &vxx_curr);
                let q_xx = mat_add(&lxx, &mat_mul(&at_vxx, &a));

                let bt_vxx = mat_mul(&bt, &vxx_curr);
                let mut q_uu = mat_add(&luu, &mat_mul(&bt_vxx, &b));
                // Add Tikhonov regularization
                for i in 0..m {
                    q_uu[i][i] += config.mu;
                }
                let q_ux = mat_add(&lux, &mat_mul(&bt_vxx, &a));

                // Compute Q_uu inverse
                let q_uu_inv = mat_sym_pos_def_inv(&q_uu)?;

                // Feedback gain: K = -Q_uu^{-1} Q_ux  (m × n)
                let k_t = mat_scale(&mat_mul(&q_uu_inv, &q_ux), -1.0);
                // Feedforward: k = -Q_uu^{-1} Q_u  (m-vector)
                let kf_t = vec_scale(&mat_vec_mul(&q_uu_inv, &q_u), -1.0);

                // Update value function:
                // V_x  = Q_x  + Q_ux^T k
                // V_xx = Q_xx + Q_ux^T K
                let q_ux_t = mat_transpose(&q_ux);
                let q_uxt_k = mat_vec_mul(&q_ux_t, &kf_t);
                vx_curr = vec_add(&q_x, &q_uxt_k);

                let q_uxt_kmat = mat_mul(&q_ux_t, &k_t);
                vxx_curr = mat_add(&q_xx, &q_uxt_kmat);

                k_feedforward.push(kf_t);
                k_feedback.push(k_t);
            }

            k_feedforward.reverse();
            k_feedback.reverse();

            // ── Line search ────────────────────────────────────────────────
            let mut alpha = 1.0;
            let mut best_controls = controls.clone();
            let mut best_cost = current_cost;
            let mut improved = false;

            for _ls in 0..config.line_search_steps {
                let mut new_controls = Vec::with_capacity(h);
                let mut x_sim = x0.to_vec();

                for t in 0..h {
                    let dx: Vec<f64> = x_sim
                        .iter()
                        .zip(traj[t].iter())
                        .map(|(xi, ti)| xi - ti)
                        .collect();
                    let k_dx = mat_vec_mul(&k_feedback[t], &dx);
                    let delta_u = vec_add(
                        &vec_scale(&k_feedforward[t], alpha),
                        &vec_scale(&k_dx, alpha),
                    );
                    let u_new = vec_add(&controls[t], &delta_u);
                    x_sim = dynamics.step(&x_sim, &u_new);
                    new_controls.push(u_new);
                }

                let new_traj = forward_pass(dynamics, x0, &new_controls);
                let new_cost = total_cost(&new_traj, &new_controls, &cost_fn, &terminal_cost_fn);

                if new_cost < best_cost {
                    best_cost = new_cost;
                    best_controls = new_controls;
                    improved = true;
                    break;
                }

                alpha *= 0.5;
            }

            if improved {
                let cost_decrease = (current_cost - best_cost).abs();
                controls = best_controls;
                traj = forward_pass(dynamics, x0, &controls);
                current_cost = best_cost;

                if cost_decrease < config.tolerance * (1.0 + current_cost.abs()) {
                    converged = true;
                    break;
                }
            } else {
                // No improvement found: declare convergence
                converged = true;
                break;
            }
        }

        Ok(IlqrResult {
            trajectory: traj,
            controls,
            total_cost: current_cost,
            n_iter,
            converged,
        })
    }
}

impl Default for IlqrSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical Differentiation Helpers (for iLQR backward pass)
// ─────────────────────────────────────────────────────────────────────────────

pub fn numerical_grad_state<F: Fn(&[f64]) -> f64>(x: &[f64], f: &F, h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0f64; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * h);
    }
    grad
}

fn numerical_hess_state<F: Fn(&[f64]) -> f64>(x: &[f64], f: &F, h: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x);
    let mut hess = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            if i == j {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[i] += h;
                xm[i] -= h;
                hess[i][i] = (f(&xp) - 2.0 * f0 + f(&xm)) / (h * h);
            } else {
                let mut xpp = x.to_vec();
                let mut xpm = x.to_vec();
                let mut xmp = x.to_vec();
                let mut xmm = x.to_vec();
                xpp[i] += h;
                xpp[j] += h;
                xpm[i] += h;
                xpm[j] -= h;
                xmp[i] -= h;
                xmp[j] += h;
                xmm[i] -= h;
                xmm[j] -= h;
                let val = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h);
                hess[i][j] = val;
                hess[j][i] = val;
            }
        }
    }
    hess
}

fn numerical_grad_state_action_x<F: Fn(&[f64], &[f64]) -> f64>(
    x: &[f64],
    u: &[f64],
    f: &F,
    h: f64,
) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            (f(&xp, u) - f(&xm, u)) / (2.0 * h)
        })
        .collect()
}

fn numerical_grad_state_action_u<F: Fn(&[f64], &[f64]) -> f64>(
    x: &[f64],
    u: &[f64],
    f: &F,
    h: f64,
) -> Vec<f64> {
    let m = u.len();
    (0..m)
        .map(|i| {
            let mut up = u.to_vec();
            let mut um = u.to_vec();
            up[i] += h;
            um[i] -= h;
            (f(x, &up) - f(x, &um)) / (2.0 * h)
        })
        .collect()
}

fn numerical_hess_state_xx<F: Fn(&[f64], &[f64]) -> f64>(
    x: &[f64],
    u: &[f64],
    f: &F,
    h: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x, u);
    let mut hess = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            if i == j {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[i] += h;
                xm[i] -= h;
                hess[i][i] = (f(&xp, u) - 2.0 * f0 + f(&xm, u)) / (h * h);
            } else {
                let mut xpp = x.to_vec();
                let mut xpm = x.to_vec();
                let mut xmp = x.to_vec();
                let mut xmm = x.to_vec();
                xpp[i] += h;
                xpp[j] += h;
                xpm[i] += h;
                xpm[j] -= h;
                xmp[i] -= h;
                xmp[j] += h;
                xmm[i] -= h;
                xmm[j] -= h;
                let val = (f(&xpp, u) - f(&xpm, u) - f(&xmp, u) + f(&xmm, u)) / (4.0 * h * h);
                hess[i][j] = val;
                hess[j][i] = val;
            }
        }
    }
    hess
}

fn numerical_hess_state_uu<F: Fn(&[f64], &[f64]) -> f64>(
    x: &[f64],
    u: &[f64],
    f: &F,
    h: f64,
) -> Vec<Vec<f64>> {
    let m = u.len();
    let f0 = f(x, u);
    let mut hess = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in i..m {
            if i == j {
                let mut up = u.to_vec();
                let mut um = u.to_vec();
                up[i] += h;
                um[i] -= h;
                hess[i][i] = (f(x, &up) - 2.0 * f0 + f(x, &um)) / (h * h);
            } else {
                let mut upp = u.to_vec();
                let mut upm = u.to_vec();
                let mut ump = u.to_vec();
                let mut umm = u.to_vec();
                upp[i] += h;
                upp[j] += h;
                upm[i] += h;
                upm[j] -= h;
                ump[i] -= h;
                ump[j] += h;
                umm[i] -= h;
                umm[j] -= h;
                let val = (f(x, &upp) - f(x, &upm) - f(x, &ump) + f(x, &umm)) / (4.0 * h * h);
                hess[i][j] = val;
                hess[j][i] = val;
            }
        }
    }
    hess
}

fn numerical_hess_state_ux<F: Fn(&[f64], &[f64]) -> f64>(
    x: &[f64],
    u: &[f64],
    f: &F,
    h: f64,
) -> Vec<Vec<f64>> {
    // Returns m × n matrix: d^2 l / (du_i dx_j)
    let m = u.len();
    let n = x.len();
    let mut hess = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut upp_xp = (u.to_vec(), x.to_vec());
            let mut upp_xm = (u.to_vec(), x.to_vec());
            let mut upm_xp = (u.to_vec(), x.to_vec());
            let mut upm_xm = (u.to_vec(), x.to_vec());
            upp_xp.0[i] += h;
            upp_xp.1[j] += h;
            upp_xm.0[i] += h;
            upp_xm.1[j] -= h;
            upm_xp.0[i] -= h;
            upm_xp.1[j] += h;
            upm_xm.0[i] -= h;
            upm_xm.1[j] -= h;
            hess[i][j] =
                (f(&upp_xp.1, &upp_xp.0) - f(&upp_xm.1, &upp_xm.0) - f(&upm_xp.1, &upm_xp.0)
                    + f(&upm_xm.1, &upm_xm.0))
                    / (4.0 * h * h);
        }
    }
    hess
}

// ─────────────────────────────────────────────────────────────────────────────
