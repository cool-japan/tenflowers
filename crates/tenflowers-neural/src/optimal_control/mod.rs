//! # Optimal Control and Model Predictive Control (MPC)
//!
//! Comprehensive optimal control algorithms implemented in pure Rust:
//! - **LQR** (Linear Quadratic Regulator): infinite-horizon DARE and finite-horizon Riccati
//! - **iLQR** (iterative LQR): nonlinear trajectory optimization with line search
//! - **MPC variants**: CEM, MPPI, Random Shooting
//! - **Direct Collocation**: gradient-based trajectory optimization with defect constraints
//! - **Built-in dynamics**: LinearDynamics, DoubleIntegrator, CartPole, PendulumDynamics
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::optimal_control::{
//!     DoubleIntegrator, LqrSolver, LqrConfig, LqrController,
//! };
//!
//! let sys = DoubleIntegrator { dt: 0.05 };
//! let (a, b) = sys.linearize(&[0.0, 0.0], &[0.0]);
//! let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let r = vec![vec![0.01]];
//! let sol = LqrSolver::new().solve_infinite_horizon(&a, &b, &q, &r)?;
//! let ctrl = LqrController::new(sol);
//! let traj = ctrl.rollout(&sys, &[1.0, 0.0], 50)?;
//! ```

pub mod collocation;
pub mod dynamics;
pub mod ilqr;
pub mod lqr;
pub mod mpc;
pub mod utils;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use collocation::{DirectCollocation, TrajectoryOptConfig, TrajectoryOptResult};
pub use dynamics::{CartPole, DoubleIntegrator, DynamicsModel, LinearDynamics, PendulumDynamics};
pub use ilqr::{
    forward_pass, quadratic_cost, total_cost, CostConfig, IlqrConfig, IlqrResult, IlqrSolver,
};
pub use lqr::{LqrConfig, LqrController, LqrSolution, LqrSolver};
pub use mpc::{CrossEntropyMpc, MpcConfig, MpcPlan, MppiConfig, MppiController, RandomShootingMpc};
pub use utils::{
    mat_add, mat_mul, mat_scale, mat_sub, mat_sym_pos_def_inv, mat_transpose, mat_vec_mul, vec_add,
    vec_scale,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::collocation::{DirectCollocation, TrajectoryOptConfig};
    use super::dynamics::{
        CartPole, DoubleIntegrator, DynamicsModel, LinearDynamics, PendulumDynamics,
    };
    use super::ilqr::numerical_grad_state;
    use super::ilqr::{
        forward_pass, quadratic_cost, total_cost, CostConfig, IlqrConfig, IlqrSolver,
    };
    use super::lqr::{LqrConfig, LqrController, LqrSolver};
    use super::mpc::{CrossEntropyMpc, MpcConfig, MppiConfig, MppiController, RandomShootingMpc};
    use super::utils::{
        mat_identity, mat_mul, mat_sym_pos_def_inv, mat_transpose, mat_vec_mul, vec_dot, vec_norm,
    };

    // ── Matrix / vector utilities ─────────────────────────────────────────────

    #[test]
    fn test_mat_mul_2x2() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[0][1] - 22.0).abs() < 1e-10);
        assert!((c[1][0] - 43.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = mat_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][0] - 3.0).abs() < 1e-10);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_sym_pos_def_inv_identity() {
        let i2 = mat_identity(2);
        let inv = mat_sym_pos_def_inv(&i2).expect("Identity is SPD");
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[0][1]).abs() < 1e-10);
        assert!((inv[1][0]).abs() < 1e-10);
        assert!((inv[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_sym_pos_def_inv_correctness() {
        // A = [[4, 2], [2, 3]] is SPD.
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let inv = mat_sym_pos_def_inv(&a).expect("A is SPD");
        let ai = mat_mul(&a, &inv);
        assert!((ai[0][0] - 1.0).abs() < 1e-8, "ai[0][0] = {}", ai[0][0]);
        assert!((ai[0][1]).abs() < 1e-8, "ai[0][1] = {}", ai[0][1]);
        assert!((ai[1][0]).abs() < 1e-8, "ai[1][0] = {}", ai[1][0]);
        assert!((ai[1][1] - 1.0).abs() < 1e-8, "ai[1][1] = {}", ai[1][1]);
    }

    #[test]
    fn test_mat_sym_pos_def_inv_not_pd() {
        let a = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];
        assert!(mat_sym_pos_def_inv(&a).is_err());
    }

    // ── LinearDynamics ─────────────────────────────────────────────────────────

    #[test]
    fn test_linear_dynamics_step() {
        let a = vec![vec![1.0, 0.5], vec![0.0, 1.0]];
        let b = vec![vec![0.0], vec![1.0]];
        let sys = LinearDynamics { a, b };
        let state = vec![1.0, 2.0];
        let action = vec![3.0];
        let next = sys.step(&state, &action);
        assert!((next[0] - 2.0).abs() < 1e-10);
        assert!((next[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_dynamics_linearize_returns_ab() {
        let a = vec![vec![1.0, 0.5], vec![0.0, 1.0]];
        let b = vec![vec![0.0], vec![1.0]];
        let sys = LinearDynamics {
            a: a.clone(),
            b: b.clone(),
        };
        let (al, bl) = sys.linearize(&[0.0, 0.0], &[0.0]);
        assert!((al[0][0] - a[0][0]).abs() < 1e-10);
        assert!((bl[1][0] - b[1][0]).abs() < 1e-10);
    }

    // ── DoubleIntegrator ───────────────────────────────────────────────────────

    #[test]
    fn test_double_integrator_step() {
        let sys = DoubleIntegrator { dt: 0.1 };
        let state = vec![0.0, 1.0];
        let action = vec![2.0];
        let next = sys.step(&state, &action);
        assert!((next[0] - 0.1).abs() < 1e-10);
        assert!((next[1] - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_double_integrator_dims() {
        let sys = DoubleIntegrator { dt: 0.05 };
        assert_eq!(sys.state_dim(), 2);
        assert_eq!(sys.action_dim(), 1);
    }

    #[test]
    fn test_double_integrator_linearize_shape() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let (a, b) = sys.linearize(&[0.0, 0.0], &[0.0]);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].len(), 1);
        assert!((a[0][0] - 1.0).abs() < 1e-10);
        assert!((a[0][1] - 0.05).abs() < 1e-10);
        assert!((b[1][0] - 0.05).abs() < 1e-10);
    }

    // ── PendulumDynamics ───────────────────────────────────────────────────────

    #[test]
    fn test_pendulum_step_zero_damping_energy_conservation() {
        let sys = PendulumDynamics {
            dt: 0.001,
            gravity: 9.81,
            length: 1.0,
            damping: 0.0,
        };
        let action = vec![0.0f64];
        let mut s = vec![0.1f64, 0.0f64];
        let energy0 = 0.5 * s[1] * s[1] - (sys.gravity / sys.length) * s[0].cos();
        for _ in 0..1000 {
            s = sys.step(&s, &action);
        }
        let energy1 = 0.5 * s[1] * s[1] - (sys.gravity / sys.length) * s[0].cos();
        let energy_err = (energy1 - energy0).abs() / (energy0.abs() + 1e-10);
        assert!(energy_err < 0.05, "Energy err = {}", energy_err);
    }

    #[test]
    fn test_pendulum_damping_dissipates_energy() {
        let sys = PendulumDynamics {
            dt: 0.01,
            gravity: 9.81,
            length: 1.0,
            damping: 0.5,
        };
        let mut s = vec![0.5, 1.0];
        let action = vec![0.0];
        let ke0 = 0.5 * s[1] * s[1];
        for _ in 0..200 {
            s = sys.step(&s, &action);
        }
        let ke1 = 0.5 * s[1] * s[1];
        assert!(ke1 < ke0, "ke0={} ke1={}", ke0, ke1);
    }

    // ── CartPole ───────────────────────────────────────────────────────────────

    #[test]
    fn test_cartpole_linearize_at_equilibrium_shape() {
        let cp = CartPole {
            dt: 0.02,
            pole_length: 0.5,
            mass_cart: 1.0,
            mass_pole: 0.1,
        };
        let (a, b) = cp.linearize_at_equilibrium();
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].len(), 4);
        assert_eq!(b.len(), 4);
        assert_eq!(b[0].len(), 1);
    }

    #[test]
    fn test_cartpole_dims() {
        let cp = CartPole {
            dt: 0.02,
            pole_length: 0.5,
            mass_cart: 1.0,
            mass_pole: 0.1,
        };
        assert_eq!(cp.state_dim(), 4);
        assert_eq!(cp.action_dim(), 1);
    }

    // ── LQR ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_lqr_infinite_horizon_shape() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let (a, b) = sys.linearize(&[0.0, 0.0], &[0.0]);
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![0.01]];
        let sol = LqrSolver::new()
            .solve_infinite_horizon(&a, &b, &q, &r)
            .expect("LQR infinite horizon should succeed");
        assert_eq!(sol.k.len(), 1);
        assert_eq!(sol.k[0].len(), 1);
        assert_eq!(sol.k[0][0].len(), 2);
    }

    #[test]
    fn test_lqr_rollout_converges_to_zero() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let (a, b) = sys.linearize(&[0.0, 0.0], &[0.0]);
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![0.01]];
        let sol = LqrSolver::new()
            .solve_infinite_horizon(&a, &b, &q, &r)
            .expect("LQR infinite horizon should succeed");
        let ctrl = LqrController::new(sol);
        let x0 = vec![2.0, 0.0];
        let traj = ctrl
            .rollout(&sys, &x0, 200)
            .expect("rollout should succeed");
        let final_state = traj.last().expect("trajectory not empty");
        let final_norm = vec_norm(final_state);
        assert!(final_norm < 0.1, "final norm = {}", final_norm);
    }

    #[test]
    fn test_lqr_finite_horizon_shape() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let (a, b) = sys.linearize(&[0.0, 0.0], &[0.0]);
        let h = 20;
        let config = LqrConfig {
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![0.01]],
            q_terminal: vec![vec![10.0, 0.0], vec![0.0, 10.0]],
            horizon: h,
        };
        let sol = LqrSolver::new()
            .solve_finite_horizon(&a, &b, &config)
            .expect("finite horizon LQR should succeed");
        assert_eq!(sol.k.len(), h);
        assert_eq!(sol.k[0].len(), 1);
        assert_eq!(sol.k[0][0].len(), 2);
    }

    #[test]
    fn test_lqr_finite_horizon_gain_decreases_near_horizon() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let (a, b) = sys.linearize(&[0.0, 0.0], &[0.0]);
        let h = 30;
        let config = LqrConfig {
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![1.0]],
            q_terminal: vec![vec![100.0, 0.0], vec![0.0, 100.0]],
            horizon: h,
        };
        let sol = LqrSolver::new()
            .solve_finite_horizon(&a, &b, &config)
            .expect("finite horizon LQR should succeed");
        let gain_first = vec_norm(&sol.k[0][0]);
        let gain_last = vec_norm(&sol.k[h - 1][0]);
        assert!(
            gain_last >= gain_first,
            "last={} first={}",
            gain_last,
            gain_first
        );
    }

    // ── iLQR ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_ilqr_cost_decreases() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![1.0, 0.0];
        let h = 20;
        let u_init = vec![vec![0.5]; h];
        let cost_cfg = CostConfig {
            q: vec![vec![1.0, 0.0], vec![0.0, 0.1]],
            r: vec![vec![0.01]],
            q_terminal: vec![vec![10.0, 0.0], vec![0.0, 1.0]],
            goal: vec![0.0, 0.0],
        };
        let cost_fn = quadratic_cost(&cost_cfg);
        let terminal_fn = |x: &[f64]| -> f64 {
            let dx: Vec<f64> = x
                .iter()
                .zip(cost_cfg.goal.iter())
                .map(|(xi, gi)| xi - gi)
                .collect();
            let qt_dx = mat_vec_mul(&cost_cfg.q_terminal, &dx);
            vec_dot(&dx, &qt_dx)
        };
        let init_traj = forward_pass(&sys, &x0, &u_init);
        let init_cost = total_cost(&init_traj, &u_init, &cost_fn, &terminal_fn);
        let config = IlqrConfig {
            horizon: h,
            max_iter: 50,
            tolerance: 1e-8,
            line_search_steps: 8,
            mu: 1e-4,
        };
        let result = IlqrSolver::new()
            .solve(&sys, &x0, u_init, &config, cost_fn, terminal_fn)
            .expect("iLQR should succeed");
        assert!(
            result.total_cost < init_cost,
            "iLQR cost {} vs initial {}",
            result.total_cost,
            init_cost
        );
    }

    #[test]
    fn test_ilqr_converges_linear_system() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![0.5, 0.0];
        let h = 15;
        let u_init = vec![vec![0.0]; h];
        let cost_cfg = CostConfig {
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![0.1]],
            q_terminal: vec![vec![5.0, 0.0], vec![0.0, 5.0]],
            goal: vec![0.0, 0.0],
        };
        let cost_fn = quadratic_cost(&cost_cfg);
        let terminal_fn = |x: &[f64]| -> f64 {
            let dx: Vec<f64> = x
                .iter()
                .zip(cost_cfg.goal.iter())
                .map(|(xi, gi)| xi - gi)
                .collect();
            let qt_dx = mat_vec_mul(&cost_cfg.q_terminal, &dx);
            vec_dot(&dx, &qt_dx)
        };
        let config = IlqrConfig {
            horizon: h,
            max_iter: 100,
            tolerance: 1e-8,
            line_search_steps: 10,
            mu: 1e-6,
        };
        let result = IlqrSolver::new()
            .solve(&sys, &x0, u_init, &config, cost_fn, terminal_fn)
            .expect("iLQR should succeed");
        assert!(result.converged, "iLQR should converge on a linear system");
    }

    #[test]
    fn test_ilqr_trajectory_length() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let h = 10;
        let x0 = vec![1.0, 0.0];
        let u_init = vec![vec![0.0]; h];
        let cost_fn =
            |x: &[f64], u: &[f64]| -> f64 { x[0] * x[0] + x[1] * x[1] + 0.01 * u[0] * u[0] };
        let terminal_fn = |x: &[f64]| -> f64 { 5.0 * (x[0] * x[0] + x[1] * x[1]) };
        let config = IlqrConfig {
            horizon: h,
            ..Default::default()
        };
        let result = IlqrSolver::new()
            .solve(&sys, &x0, u_init, &config, cost_fn, terminal_fn)
            .expect("iLQR should succeed");
        assert_eq!(result.trajectory.len(), h + 1);
        assert_eq!(result.controls.len(), h);
    }

    // ── quadratic_cost ────────────────────────────────────────────────────────

    #[test]
    fn test_quadratic_cost_zero_at_goal() {
        let cfg = CostConfig {
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![1.0]],
            q_terminal: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            goal: vec![2.0, 3.0],
        };
        let cost = quadratic_cost(&cfg);
        let val = cost(&[2.0, 3.0], &[0.0]);
        assert!(val.abs() < 1e-10, "cost at goal = {}", val);
    }

    #[test]
    fn test_quadratic_cost_positive_away_from_goal() {
        let cfg = CostConfig {
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![0.1]],
            q_terminal: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            goal: vec![0.0, 0.0],
        };
        let cost = quadratic_cost(&cfg);
        let val = cost(&[1.0, 1.0], &[0.5]);
        assert!(
            val > 0.0,
            "quadratic cost should be positive away from goal"
        );
    }

    // ── CrossEntropyMpc ────────────────────────────────────────────────────────

    #[test]
    fn test_cem_plan_horizon_length() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![1.0, 0.0];
        let cfg = MpcConfig {
            horizon: 10,
            n_samples: 50,
            noise_std: 0.3,
            top_k: 10,
            n_iter: 3,
            seed: 42,
        };
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + x[1] * x[1] + 0.01 * u[0] * u[0];
        let plan = CrossEntropyMpc::new()
            .plan(&sys, &x0, &cost_fn, &cfg)
            .expect("CEM plan ok");
        assert_eq!(plan.actions.len(), cfg.horizon);
    }

    #[test]
    fn test_cem_plan_cost_finite() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![0.5, 0.5];
        let cfg = MpcConfig {
            horizon: 5,
            n_samples: 30,
            noise_std: 0.5,
            top_k: 5,
            n_iter: 2,
            seed: 7,
        };
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + x[1] * x[1] + 0.01 * u[0] * u[0];
        let plan = CrossEntropyMpc::new()
            .plan(&sys, &x0, &cost_fn, &cfg)
            .expect("CEM plan ok");
        assert!(plan.expected_cost.is_finite());
    }

    #[test]
    fn test_cem_step_returns_single_action() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![1.0, 0.0];
        let cfg = MpcConfig {
            horizon: 8,
            n_samples: 40,
            noise_std: 0.4,
            top_k: 8,
            n_iter: 2,
            seed: 1,
        };
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + u[0] * u[0];
        let action = CrossEntropyMpc::new()
            .step(&sys, &x0, &cost_fn, &cfg)
            .expect("CEM step ok");
        assert_eq!(action.len(), sys.action_dim());
    }

    // ── MppiController ─────────────────────────────────────────────────────────

    #[test]
    fn test_mppi_plan_returns_valid_plan() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![1.0, 0.0];
        let mppi_cfg = MppiConfig {
            horizon: 10,
            n_samples: 100,
            temperature: 1.0,
            noise_std: 0.3,
            seed: 42,
        };
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + x[1] * x[1] + 0.01 * u[0] * u[0];
        let mut ctrl = MppiController::new(mppi_cfg.clone());
        let plan = ctrl.plan(&sys, &x0, &cost_fn).expect("MPPI plan ok");
        assert_eq!(plan.actions.len(), mppi_cfg.horizon);
        assert_eq!(plan.actions[0].len(), sys.action_dim());
    }

    #[test]
    fn test_mppi_plan_cost_finite() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![0.5, 0.0];
        let mppi_cfg = MppiConfig {
            horizon: 5,
            n_samples: 50,
            temperature: 0.5,
            noise_std: 0.2,
            seed: 99,
        };
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + 0.1 * u[0] * u[0];
        let mut ctrl = MppiController::new(mppi_cfg);
        let plan = ctrl.plan(&sys, &x0, &cost_fn).expect("MPPI plan ok");
        assert!(plan.expected_cost.is_finite());
    }

    // ── RandomShootingMpc ──────────────────────────────────────────────────────

    #[test]
    fn test_random_shooting_best_leq_average() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![1.0, 0.5];
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + x[1] * x[1] + 0.01 * u[0] * u[0];
        let plan = RandomShootingMpc::new()
            .plan(&sys, &x0, &cost_fn, 100, 10, 0.5, 42)
            .expect("random shooting ok");
        assert!(plan.expected_cost.is_finite());
        assert_eq!(plan.actions.len(), 10);
    }

    #[test]
    fn test_random_shooting_plan_horizon() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let x0 = vec![2.0, 0.0];
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + 0.1 * u[0] * u[0];
        let plan = RandomShootingMpc::new()
            .plan(&sys, &x0, &cost_fn, 50, 15, 1.0, 7)
            .expect("random shooting ok");
        assert_eq!(plan.actions.len(), 15);
    }

    // ── DirectCollocation ──────────────────────────────────────────────────────

    #[test]
    fn test_direct_collocation_defect_norm_decreases() {
        let sys = DoubleIntegrator { dt: 0.1 };
        let h = 5;
        let n = sys.state_dim();
        let states: Vec<Vec<f64>> = (0..=h).map(|t| vec![t as f64 * 0.1, 0.0]).collect();
        let controls: Vec<Vec<f64>> = vec![vec![0.0]; h];
        let mut init_defect = 0.0;
        for t in 0..h {
            let x_next = sys.step(&states[t], &controls[t]);
            for i in 0..n {
                let d = states[t + 1][i] - x_next[i];
                init_defect += d * d;
            }
        }
        let init_defect = init_defect.sqrt();
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + 0.01 * u[0] * u[0];
        let terminal_fn = |x: &[f64]| 5.0 * x[0] * x[0];
        let config = TrajectoryOptConfig {
            horizon: h,
            max_iter: 30,
            lr: 1e-3,
            tolerance: 1e-8,
        };
        let result = DirectCollocation::new()
            .optimize(states, controls, &cost_fn, &terminal_fn, &sys, &config)
            .expect("direct collocation ok");
        assert!(
            result.defect_norm <= init_defect + 1e-6,
            "defect norm {} > initial {}",
            result.defect_norm,
            init_defect
        );
    }

    #[test]
    fn test_direct_collocation_result_structure() {
        let sys = DoubleIntegrator { dt: 0.05 };
        let h = 4;
        let states: Vec<Vec<f64>> = (0..=h).map(|_| vec![0.0, 0.0]).collect();
        let controls: Vec<Vec<f64>> = vec![vec![0.0]; h];
        let cost_fn = |x: &[f64], u: &[f64]| x[0] * x[0] + u[0] * u[0];
        let terminal_fn = |x: &[f64]| x[0] * x[0];
        let config = TrajectoryOptConfig {
            horizon: h,
            max_iter: 5,
            lr: 1e-4,
            tolerance: 1e-10,
        };
        let result = DirectCollocation::new()
            .optimize(states, controls, &cost_fn, &terminal_fn, &sys, &config)
            .expect("direct collocation ok");
        assert_eq!(result.states.len(), h + 1);
        assert_eq!(result.controls.len(), h);
    }

    // ── Numerical differentiation helpers ─────────────────────────────────────

    #[test]
    fn test_numerical_grad_quadratic() {
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let x = vec![1.0, 1.0];
        let g = numerical_grad_state(&x, &f, 1e-5);
        assert!((g[0] - 2.0).abs() < 1e-5, "grad[0] = {}", g[0]);
        assert!((g[1] - 4.0).abs() < 1e-5, "grad[1] = {}", g[1]);
    }
}
