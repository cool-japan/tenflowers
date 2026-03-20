//! Linear Quadratic Regulator (LQR).

use super::dynamics::DynamicsModel;
use super::utils::{
    mat_add, mat_frob_norm, mat_mul, mat_scale, mat_sub, mat_sym_pos_def_inv, mat_transpose,
    mat_vec_mul, vec_norm, vec_scale,
};
use tenflowers_core::{Result, TensorError};

// LQR
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Linear Quadratic Regulator.
#[derive(Clone, Debug)]
pub struct LqrConfig {
    /// State cost matrix Q (state_dim × state_dim), positive semi-definite.
    pub q: Vec<Vec<f64>>,
    /// Control cost matrix R (action_dim × action_dim), positive definite.
    pub r: Vec<Vec<f64>>,
    /// Terminal state cost matrix (state_dim × state_dim), positive semi-definite.
    pub q_terminal: Vec<Vec<f64>>,
    /// Planning horizon (number of time steps).
    pub horizon: usize,
}

/// Solution returned by the LQR solver.
#[derive(Clone, Debug)]
pub struct LqrSolution {
    /// Feedback gain matrices K\[t\] of shape (action_dim × state_dim) for each time step.
    /// For the infinite-horizon case, all entries are identical.
    pub k: Vec<Vec<Vec<f64>>>,
    /// Cost-to-go (Riccati) matrices P\[t\] of shape (state_dim × state_dim).
    pub p: Vec<Vec<Vec<f64>>>,
    /// Flattened convenience access: `gains[t]` = `K[t]`.
    pub gains: Vec<Vec<Vec<f64>>>,
}

impl LqrSolution {
    /// Compute the optimal action at time `t` for the given state: `u = -K[t] * state`.
    pub fn optimal_action(&self, t: usize, state: &[f64]) -> Vec<f64> {
        let t_clamped = t.min(self.k.len().saturating_sub(1));
        let kt = &self.k[t_clamped];
        // u = -K * x
        vec_scale(&mat_vec_mul(kt, state), -1.0)
    }
}

/// LQR solver implementing DARE (discrete algebraic Riccati equation) for infinite horizon
/// and backward Riccati recursion for finite horizon.
pub struct LqrSolver;

impl LqrSolver {
    /// Create a new `LqrSolver`.
    pub fn new() -> Self {
        Self
    }

    /// Solve the infinite-horizon LQR via iterative Lyapunov recursion (DARE).
    ///
    /// Iterates `P_{k+1} = Q + A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A`
    /// until convergence (max 1000 iterations, tolerance 1e-10).
    pub fn solve_infinite_horizon(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        q: &[Vec<f64>],
        r: &[Vec<f64>],
    ) -> Result<LqrSolution> {
        let n = a.len();
        let m = r.len();

        // Initial P = Q
        let mut p = q.to_vec();
        let at = mat_transpose(a);
        let bt = mat_transpose(b);

        for _iter in 0..1000 {
            // S = R + B^T P B  (m × m)
            let pb = mat_mul(&p, b);
            let bt_pb = mat_mul(&bt, &pb);
            let s = mat_add(r, &bt_pb);

            // S_inv = S^{-1} (SPD because R is PD and B^T P B is PSD)
            let s_inv = mat_sym_pos_def_inv(&s)?;

            // K = S^{-1} B^T P A  (m × n)
            let bt_p = mat_mul(&bt, &p);
            let bt_pa = mat_mul(&bt_p, a);
            let k = mat_mul(&s_inv, &bt_pa);

            // P_new = Q + A^T P A - A^T P B K
            let at_p = mat_mul(&at, &p);
            let at_pa = mat_mul(&at_p, a);
            let at_pb = mat_mul(&at_p, b);
            let at_pb_k = mat_mul(&at_pb, &k);
            let p_new = mat_sub(&mat_add(q, &at_pa), &at_pb_k);

            let diff = mat_frob_norm(&mat_sub(&p_new, &p));
            p = p_new;

            if diff < 1e-10 {
                break;
            }
        }

        // Compute final gain K = S^{-1} B^T P A
        let pb = mat_mul(&p, b);
        let bt_pb = mat_mul(&bt, &pb);
        let s = mat_add(r, &bt_pb);
        let s_inv = mat_sym_pos_def_inv(&s)?;
        let bt_p = mat_mul(&bt, &p);
        let bt_pa = mat_mul(&bt_p, a);
        let k_mat = mat_mul(&s_inv, &bt_pa);

        // Infinite horizon: same K for every time step; we store a single-step solution.
        Ok(LqrSolution {
            k: vec![k_mat.clone()],
            p: vec![p],
            gains: vec![k_mat],
        })
    }

    /// Solve the finite-horizon LQR via backward Riccati recursion.
    ///
    /// Starting from `P_T = Q_terminal`, recursively computes:
    /// `P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A`
    pub fn solve_finite_horizon(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        config: &LqrConfig,
    ) -> Result<LqrSolution> {
        let n = a.len();
        let m = config.r.len();
        let h = config.horizon;

        let at = mat_transpose(a);
        let bt = mat_transpose(b);

        // Storage: P[0] = terminal, P[h] will be the cost from the start.
        // We build from t = h down to t = 0.
        let mut p_list = Vec::with_capacity(h + 1);
        let mut k_list = Vec::with_capacity(h);

        p_list.push(config.q_terminal.clone());

        for t in (0..h).rev() {
            let p_next = p_list
                .last()
                .expect("p_list cannot be empty during recursion");

            // S = R + B^T P B
            let pb = mat_mul(p_next, b);
            let bt_pb = mat_mul(&bt, &pb);
            let s = mat_add(&config.r, &bt_pb);
            let s_inv = mat_sym_pos_def_inv(&s)?;

            // K = S^{-1} B^T P A
            let bt_p = mat_mul(&bt, p_next);
            let bt_pa = mat_mul(&bt_p, a);
            let k = mat_mul(&s_inv, &bt_pa);

            // P_t = Q + A^T P A - A^T P B K
            let at_p = mat_mul(&at, p_next);
            let at_pa = mat_mul(&at_p, a);
            let at_pb = mat_mul(&at_p, b);
            let at_pb_k = mat_mul(&at_pb, &k);
            let p_t = mat_sub(&mat_add(&config.q, &at_pa), &at_pb_k);

            k_list.push(k);
            p_list.push(p_t);
        }

        // Reverse so that index 0 = first time step
        k_list.reverse();
        p_list.reverse();
        // p_list[0] = P at t=0 (beginning), p_list[h] = terminal cost
        // k_list[t] = gain at step t

        let gains = k_list.clone();
        Ok(LqrSolution {
            k: k_list,
            p: p_list,
            gains,
        })
    }
}

impl Default for LqrSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// LQR controller that wraps an `LqrSolution` and provides a convenient interface.
pub struct LqrController {
    /// The precomputed LQR solution (gain matrices).
    pub solution: LqrSolution,
}

impl LqrController {
    /// Create a new `LqrController` from an `LqrSolution`.
    pub fn new(solution: LqrSolution) -> Self {
        Self { solution }
    }

    /// Compute the optimal control action at time `t` for state `x`.
    pub fn control(&self, t: usize, state: &[f64]) -> Vec<f64> {
        self.solution.optimal_action(t, state)
    }

    /// Simulate a trajectory of length `steps` starting from `x0` using the LQR controller.
    ///
    /// Returns a vector of states visited: `[x0, x1, ..., x_steps]`.
    pub fn rollout(
        &self,
        dynamics: &dyn DynamicsModel,
        x0: &[f64],
        steps: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let mut traj = Vec::with_capacity(steps + 1);
        let mut x = x0.to_vec();
        traj.push(x.clone());
        for t in 0..steps {
            let u = self.control(t, &x);
            x = dynamics.step(&x, &u);
            traj.push(x.clone());
        }
        Ok(traj)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
