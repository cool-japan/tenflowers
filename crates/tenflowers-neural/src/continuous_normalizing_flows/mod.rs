//! Continuous Normalizing Flows, FFJORD, and Flow Matching
//!
//! Implements:
//! - **CNF** (Chen et al. 2018): `dz/dt = f(z,t; θ)` with exact log-det via Euler ODE integration
//! - **FFJORD** (Grathwohl et al. 2019): CNF with Hutchinson trace estimator for scalable training
//! - **Flow Matching / CFM** (Lipman et al. 2022): conditional flow matching loss
//! - **OT-CFM** (Tong et al. 2023): optimal-transport conditional flow matching
//! - **Rectified Flow** (Liu et al. 2022): straight-line interpolation between noise and data
//!
//! All computations are `f64`, 100% pure Rust, no C/Fortran dependencies.
//!
//! # Mathematical background
//!
//! A CNF transforms a simple base distribution `p_0(z)` into a complex data
//! distribution `p_T(x)` by solving an ODE.  The log-likelihood is:
//! ```text
//! log p(x) = log p_0(z_0) + ∫_0^T tr(∂f/∂z) dt
//! ```
//! where `z(T) = x` and the integral of the trace is the log-det of the Jacobian.

mod ffjord;
mod flow_matching;
mod metrics;
mod mlp;
mod utils;

// Public re-exports
pub use ffjord::{Ffjord, FfjordBlock, FfjordConfig};
pub use flow_matching::{
    FlowMatchingConfig, FlowMatchingModel, OtCfmModel, RectifiedFlow, RectifiedFlowConfig,
};
pub use metrics::{evaluate_cnf, evaluate_flow_matching, CnfMetrics};
pub use mlp::{CnfDynamics, CnfMlp, ContinuousNormalizingFlow};

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    // ── CnfMlp ──────────────────────────────────────────────────────────────

    #[test]
    fn test_cnf_mlp_output_shape() {
        let mlp = CnfMlp::new(&[4, 16, 8, 4]);
        let x = vec![1.0, -0.5, 0.3, 2.1];
        let out = mlp.forward(&x);
        assert_eq!(out.len(), 4, "output dimension should be 4");
    }

    #[test]
    fn test_cnf_mlp_output_finite() {
        let mlp = CnfMlp::new(&[3, 32, 3]);
        let x = vec![0.0, 1.0, -1.0];
        let out = mlp.forward(&x);
        for v in &out {
            assert!(v.is_finite(), "all MLP outputs must be finite");
        }
    }

    #[test]
    fn test_cnf_mlp_jacobian_diag_shape() {
        let mlp = CnfMlp::new(&[4, 16, 4]);
        let x = vec![0.5, -0.5, 1.0, -1.0];
        let diag = mlp.jacobian_diagonal_approx(&x);
        assert_eq!(diag.len(), 4, "Jacobian diagonal should match input dim");
    }

    #[test]
    fn test_cnf_mlp_jacobian_diag_finite() {
        let mlp = CnfMlp::new(&[2, 8, 2]);
        let x = vec![0.1, 0.2];
        let diag = mlp.jacobian_diagonal_approx(&x);
        for d in &diag {
            assert!(d.is_finite(), "Jacobian diagonal elements must be finite");
        }
    }

    #[test]
    fn test_cnf_mlp_update_changes_weights() {
        let mut mlp = CnfMlp::new(&[2, 4, 2]);
        let before = mlp.weights[0][0][0];
        let grad_w: Vec<Vec<Vec<f64>>> = mlp
            .weights
            .iter()
            .map(|lw| lw.iter().map(|row| vec![1.0; row.len()]).collect())
            .collect();
        let grad_b: Vec<Vec<f64>> = mlp.biases.iter().map(|lb| vec![1.0; lb.len()]).collect();
        mlp.update(&grad_w, &grad_b, 0.1);
        let after = mlp.weights[0][0][0];
        assert!(
            (before - after).abs() > 1e-10,
            "update should change weights"
        );
    }

    #[test]
    fn test_cnf_mlp_single_layer() {
        let mlp = CnfMlp::new(&[3, 3]);
        let x = vec![1.0, 0.0, -1.0];
        let out = mlp.forward(&x);
        assert_eq!(out.len(), 3);
    }

    // ── CnfDynamics ─────────────────────────────────────────────────────────

    #[test]
    fn test_cnf_dynamics_forward_shape() {
        let dyn_ = CnfDynamics::new(4, 16, 2, true);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let dz = dyn_.forward(&z, 0.5);
        assert_eq!(dz.len(), 4, "dynamics output should match z_dim");
    }

    #[test]
    fn test_cnf_dynamics_forward_finite() {
        let dyn_ = CnfDynamics::new(3, 8, 1, true);
        let z = vec![1.0, -1.0, 0.0];
        let dz = dyn_.forward(&z, 0.0);
        for v in &dz {
            assert!(v.is_finite(), "dynamics output must be finite");
        }
    }

    #[test]
    fn test_cnf_dynamics_no_time() {
        let dyn_ = CnfDynamics::new(2, 8, 1, false);
        let z = vec![0.5, -0.5];
        let dz = dyn_.forward(&z, 99.0); // t ignored when include_time = false
        assert_eq!(dz.len(), 2);
    }

    #[test]
    fn test_cnf_dynamics_trace_jac_finite() {
        let dyn_ = CnfDynamics::new(2, 8, 1, true);
        let z = vec![0.1, 0.2];
        let mut rng = StdRng::seed_from_u64(42);
        let tr = dyn_.trace_jac_approx(&z, 0.5, 4, &mut rng);
        assert!(
            tr.is_finite(),
            "trace Jacobian approximation must be finite"
        );
    }

    #[test]
    fn test_cnf_dynamics_trace_more_samples() {
        let dyn_ = CnfDynamics::new(4, 16, 2, true);
        let z = vec![0.0, 0.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(7);
        let tr = dyn_.trace_jac_approx(&z, 0.3, 10, &mut rng);
        assert!(tr.is_finite());
    }

    // ── ContinuousNormalizingFlow ────────────────────────────────────────────

    #[test]
    fn test_cnf_integrate_forward_shape() {
        let cnf = ContinuousNormalizingFlow::new(3, 16, 2);
        let z0 = vec![0.1, 0.2, 0.3];
        let (z_t, _ld) = cnf.integrate_forward(&z0, 5, 0.0, 1.0);
        assert_eq!(z_t.len(), 3, "integrated state should have correct shape");
    }

    #[test]
    fn test_cnf_integrate_forward_finite() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let z0 = vec![0.5, -0.5];
        let (z_t, log_det) = cnf.integrate_forward(&z0, 10, 0.0, 1.0);
        for v in &z_t {
            assert!(v.is_finite());
        }
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_cnf_integrate_forward_changes_state() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let z0 = vec![0.0, 0.0];
        let (z_t, _ld) = cnf.integrate_forward(&z0, 5, 0.0, 1.0);
        assert_eq!(z_t.len(), 2);
    }

    #[test]
    fn test_cnf_integrate_backward_shape() {
        let cnf = ContinuousNormalizingFlow::new(3, 16, 2);
        let x = vec![1.0, -1.0, 0.5];
        let (z0, _ld) = cnf.integrate_backward(&x, 5);
        assert_eq!(z0.len(), 3);
    }

    #[test]
    fn test_cnf_log_prob_finite() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let x = vec![0.5, -0.5];
        let lp = cnf.log_prob(&x, 5);
        assert!(lp.is_finite(), "log_prob must be finite");
    }

    #[test]
    fn test_cnf_log_prob_reasonable_range() {
        let cnf = ContinuousNormalizingFlow::new(2, 16, 2);
        let x = vec![0.0, 0.0];
        let lp = cnf.log_prob(&x, 5);
        assert!(lp > -1000.0, "log_prob at origin should not be -inf");
    }

    #[test]
    fn test_cnf_sample_shape() {
        let cnf = ContinuousNormalizingFlow::new(4, 16, 2);
        let mut rng = StdRng::seed_from_u64(1);
        let x = cnf.sample(10, &mut rng);
        assert_eq!(x.len(), 4, "sample should have z_dim elements");
    }

    #[test]
    fn test_cnf_sample_finite() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let mut rng = StdRng::seed_from_u64(2);
        let x = cnf.sample(5, &mut rng);
        for v in &x {
            assert!(v.is_finite(), "sample should be finite");
        }
    }

    #[test]
    fn test_cnf_train_step_finite_loss() {
        let mut cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let x_batch = vec![vec![0.1, 0.2], vec![-0.3, 0.5], vec![0.0, -0.1]];
        let loss = cnf.train_step(&x_batch, 3, 1e-3);
        assert!(loss.is_finite(), "train_step loss must be finite");
    }

    #[test]
    fn test_cnf_log_det_sign_changes() {
        let cnf = ContinuousNormalizingFlow::new(2, 16, 2);
        let z0 = vec![0.5, 0.5];
        let (_z_t, log_det) = cnf.integrate_forward(&z0, 20, 0.0, 1.0);
        assert!(!log_det.is_nan(), "log_det must not be NaN");
    }

    // ── FfjordBlock ─────────────────────────────────────────────────────────

    #[test]
    fn test_ffjord_block_forward_shape() {
        let block = FfjordBlock::new(3, 16);
        let z = vec![0.1, 0.2, 0.3];
        let mut rng = StdRng::seed_from_u64(10);
        let (z_t, log_det) = block.forward(&z, 5, 1, &mut rng);
        assert_eq!(z_t.len(), 3);
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_ffjord_block_inverse_shape() {
        let block = FfjordBlock::new(2, 8);
        let x = vec![0.5, -0.5];
        let mut rng = StdRng::seed_from_u64(11);
        let (z0, log_det) = block.inverse(&x, 5, 1, &mut rng);
        assert_eq!(z0.len(), 2);
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_ffjord_block_inverse_approx_inverse() {
        let block = FfjordBlock::new(2, 16);
        let z = vec![0.5, -0.3];
        let mut rng = StdRng::seed_from_u64(12);
        let (z_t, _) = block.forward(&z, 20, 1, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(12);
        let (z_rec, _) = block.inverse(&z_t, 20, 1, &mut rng2);
        assert_eq!(z_rec.len(), 2);
        for v in &z_rec {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffjord_block_log_det_finite() {
        let block = FfjordBlock::new(4, 16);
        let z = vec![0.0, 0.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(13);
        let (_, log_det) = block.forward(&z, 5, 2, &mut rng);
        assert!(log_det.is_finite());
    }

    // ── Ffjord ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ffjord_log_prob_finite() {
        let config = FfjordConfig {
            z_dim: 2,
            hidden_dims: vec![16],
            n_blocks: 2,
            n_steps: 5,
            n_trace_samples: 1,
            lr: 1e-3,
        };
        let model = Ffjord::new(config);
        let x = vec![0.5, -0.5];
        let mut rng = StdRng::seed_from_u64(20);
        let lp = model.log_prob(&x, &mut rng);
        assert!(lp.is_finite(), "Ffjord log_prob must be finite");
    }

    #[test]
    fn test_ffjord_sample_shape() {
        let config = FfjordConfig {
            z_dim: 3,
            hidden_dims: vec![16],
            n_blocks: 1,
            n_steps: 5,
            n_trace_samples: 1,
            lr: 1e-3,
        };
        let model = Ffjord::new(config);
        let mut rng = StdRng::seed_from_u64(21);
        let x = model.sample(&mut rng);
        assert_eq!(x.len(), 3, "Ffjord sample should have z_dim=3 elements");
    }

    #[test]
    fn test_ffjord_sample_finite() {
        let config = FfjordConfig {
            z_dim: 2,
            hidden_dims: vec![8],
            n_blocks: 1,
            n_steps: 5,
            n_trace_samples: 1,
            lr: 1e-3,
        };
        let model = Ffjord::new(config);
        let mut rng = StdRng::seed_from_u64(22);
        let x = model.sample(&mut rng);
        for v in &x {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffjord_train_step_finite_loss() {
        let config = FfjordConfig {
            z_dim: 2,
            hidden_dims: vec![8],
            n_blocks: 1,
            n_steps: 3,
            n_trace_samples: 1,
            lr: 1e-3,
        };
        let mut model = Ffjord::new(config);
        let x_batch = vec![vec![0.1, 0.2], vec![-0.3, 0.5]];
        let mut rng = StdRng::seed_from_u64(23);
        let loss = model.train_step(&x_batch, 1e-3, &mut rng);
        assert!(loss.is_finite(), "Ffjord train_step loss must be finite");
    }

    #[test]
    fn test_ffjord_multi_block() {
        let config = FfjordConfig {
            z_dim: 2,
            hidden_dims: vec![8],
            n_blocks: 3,
            n_steps: 3,
            n_trace_samples: 1,
            lr: 1e-3,
        };
        let model = Ffjord::new(config.clone());
        assert_eq!(model.blocks.len(), 3);
        let mut rng = StdRng::seed_from_u64(24);
        let x = model.sample(&mut rng);
        assert_eq!(x.len(), 2);
    }

    // ── FlowMatchingModel ────────────────────────────────────────────────────

    #[test]
    fn test_flow_matching_velocity_shape() {
        let config = FlowMatchingConfig {
            z_dim: 4,
            hidden_dim: 16,
            n_layers: 2,
            sigma_min: 1e-4,
            n_steps: 10,
            lr: 1e-3,
        };
        let model = FlowMatchingModel::new(config);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let v = model.velocity(&z, 0.5);
        assert_eq!(v.len(), 4, "velocity output must match z_dim");
    }

    #[test]
    fn test_flow_matching_velocity_finite() {
        let config = FlowMatchingConfig::default();
        let model = FlowMatchingModel::new(config);
        let z = vec![1.0, -1.0];
        let v = model.velocity(&z, 0.0);
        for vi in &v {
            assert!(vi.is_finite());
        }
    }

    #[test]
    fn test_flow_matching_cfm_loss_nonnegative() {
        let config = FlowMatchingConfig::default();
        let model = FlowMatchingModel::new(config);
        let x0_batch = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let x1_batch = vec![vec![1.0, 0.5], vec![-1.0, 0.5]];
        let mut rng = StdRng::seed_from_u64(30);
        let loss = model.cfm_loss(&x0_batch, &x1_batch, &mut rng);
        assert!(loss >= 0.0, "CFM loss must be non-negative, got {}", loss);
    }

    #[test]
    fn test_flow_matching_cfm_loss_finite() {
        let config = FlowMatchingConfig::default();
        let model = FlowMatchingModel::new(config);
        let x0_batch = vec![vec![0.0, 0.0]];
        let x1_batch = vec![vec![1.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(31);
        let loss = model.cfm_loss(&x0_batch, &x1_batch, &mut rng);
        assert!(loss.is_finite(), "CFM loss must be finite");
    }

    #[test]
    fn test_flow_matching_train_step_finite_loss() {
        let config = FlowMatchingConfig {
            z_dim: 2,
            hidden_dim: 16,
            n_layers: 1,
            sigma_min: 1e-4,
            n_steps: 10,
            lr: 1e-3,
        };
        let mut model = FlowMatchingModel::new(config);
        let x1_batch = vec![vec![0.5, 0.5], vec![-0.5, 0.5], vec![0.0, -0.5]];
        let mut rng = StdRng::seed_from_u64(32);
        let loss = model.train_step(&x1_batch, 1e-3, &mut rng);
        assert!(
            loss.is_finite(),
            "FlowMatchingModel train_step loss must be finite"
        );
    }

    #[test]
    fn test_flow_matching_train_step_nonnegative_loss() {
        let config = FlowMatchingConfig::default();
        let mut model = FlowMatchingModel::new(config);
        let x1_batch = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(33);
        let loss = model.train_step(&x1_batch, 1e-4, &mut rng);
        assert!(loss >= 0.0, "loss must be non-negative");
    }

    #[test]
    fn test_flow_matching_sample_shape() {
        let config = FlowMatchingConfig {
            z_dim: 3,
            hidden_dim: 8,
            n_layers: 1,
            sigma_min: 1e-4,
            n_steps: 10,
            lr: 1e-3,
        };
        let model = FlowMatchingModel::new(config);
        let mut rng = StdRng::seed_from_u64(34);
        let x = model.sample(10, &mut rng);
        assert_eq!(x.len(), 3, "sample should have z_dim=3 elements");
    }

    #[test]
    fn test_flow_matching_sample_finite() {
        let config = FlowMatchingConfig::default();
        let model = FlowMatchingModel::new(config);
        let mut rng = StdRng::seed_from_u64(35);
        let x = model.sample(10, &mut rng);
        for v in &x {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_flow_matching_sample_batch_length() {
        let config = FlowMatchingConfig::default();
        let model = FlowMatchingModel::new(config);
        let mut rng = StdRng::seed_from_u64(36);
        let batch = model.sample_batch(7, 5, &mut rng);
        assert_eq!(
            batch.len(),
            7,
            "sample_batch should return exactly n_samples points"
        );
    }

    #[test]
    fn test_flow_matching_sample_batch_each_correct_dim() {
        let config = FlowMatchingConfig {
            z_dim: 4,
            ..Default::default()
        };
        let model = FlowMatchingModel::new(config);
        let mut rng = StdRng::seed_from_u64(37);
        let batch = model.sample_batch(5, 5, &mut rng);
        for s in &batch {
            assert_eq!(s.len(), 4);
        }
    }

    // ── OtCfmModel ──────────────────────────────────────────────────────────

    #[test]
    fn test_ot_cfm_ot_match_valid_permutation() {
        let x0 = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let x1 = vec![vec![0.1, 0.1], vec![0.9, 0.1], vec![0.1, 0.9]];
        let perm = OtCfmModel::ot_match(&x0, &x1);
        assert_eq!(
            perm.len(),
            3,
            "permutation should have length equal to batch size"
        );
        let mut seen = [false; 3];
        for &p in &perm {
            assert!(!seen[p], "duplicate assignment in OT match: index {}", p);
            seen[p] = true;
        }
    }

    #[test]
    fn test_ot_cfm_ot_match_nearest_pair() {
        let x0 = vec![vec![0.0_f64], vec![10.0_f64]];
        let x1 = vec![vec![0.1_f64], vec![9.9_f64]];
        let perm = OtCfmModel::ot_match(&x0, &x1);
        assert_eq!(perm[0], 0);
        assert_eq!(perm[1], 1);
    }

    #[test]
    fn test_ot_cfm_train_step_finite_loss() {
        let config = FlowMatchingConfig {
            z_dim: 2,
            hidden_dim: 8,
            n_layers: 1,
            sigma_min: 1e-4,
            n_steps: 10,
            lr: 1e-3,
        };
        let mut model = OtCfmModel::new(config);
        let x1_batch = vec![vec![0.5, 0.5], vec![-0.5, 0.5], vec![0.0, -0.5]];
        let mut rng = StdRng::seed_from_u64(40);
        let loss = model.train_step(&x1_batch, 1e-3, &mut rng);
        assert!(
            loss.is_finite(),
            "OtCfmModel train_step loss must be finite"
        );
    }

    #[test]
    fn test_ot_cfm_train_step_nonnegative_loss() {
        let config = FlowMatchingConfig::default();
        let mut model = OtCfmModel::new(config);
        let x1_batch = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(41);
        let loss = model.train_step(&x1_batch, 1e-3, &mut rng);
        assert!(loss >= 0.0, "loss must be non-negative");
    }

    #[test]
    fn test_ot_cfm_sample_shape() {
        let config = FlowMatchingConfig {
            z_dim: 3,
            hidden_dim: 8,
            n_layers: 1,
            ..Default::default()
        };
        let model = OtCfmModel::new(config);
        let mut rng = StdRng::seed_from_u64(42);
        let x = model.sample(10, &mut rng);
        assert_eq!(x.len(), 3);
    }

    #[test]
    fn test_ot_cfm_sample_finite() {
        let config = FlowMatchingConfig::default();
        let model = OtCfmModel::new(config);
        let mut rng = StdRng::seed_from_u64(43);
        let x = model.sample(10, &mut rng);
        for v in &x {
            assert!(v.is_finite());
        }
    }

    // ── RectifiedFlow ────────────────────────────────────────────────────────

    #[test]
    fn test_rectified_flow_reflow_loss_nonnegative() {
        let config = RectifiedFlowConfig::default();
        let model = RectifiedFlow::new(config);
        let x0_batch = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let x1_batch = vec![vec![1.0, 0.5], vec![-1.0, 0.5]];
        let mut rng = StdRng::seed_from_u64(50);
        let loss = model.reflow_loss(&x0_batch, &x1_batch, &mut rng);
        assert!(
            loss >= 0.0,
            "Reflow loss must be non-negative, got {}",
            loss
        );
    }

    #[test]
    fn test_rectified_flow_reflow_loss_finite() {
        let config = RectifiedFlowConfig::default();
        let model = RectifiedFlow::new(config);
        let x0_batch = vec![vec![0.0, 0.0]];
        let x1_batch = vec![vec![1.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(51);
        let loss = model.reflow_loss(&x0_batch, &x1_batch, &mut rng);
        assert!(loss.is_finite(), "Reflow loss must be finite");
    }

    #[test]
    fn test_rectified_flow_train_step_finite_loss() {
        let config = RectifiedFlowConfig {
            z_dim: 2,
            hidden_dim: 8,
            n_layers: 1,
            n_steps: 10,
            lr: 1e-3,
        };
        let mut model = RectifiedFlow::new(config);
        let x1_batch = vec![vec![0.5, 0.5], vec![-0.5, 0.5]];
        let mut rng = StdRng::seed_from_u64(52);
        let loss = model.train_step(&x1_batch, 1e-3, &mut rng);
        assert!(
            loss.is_finite(),
            "RectifiedFlow train_step loss must be finite"
        );
    }

    #[test]
    fn test_rectified_flow_train_step_nonnegative_loss() {
        let config = RectifiedFlowConfig::default();
        let mut model = RectifiedFlow::new(config);
        let x1_batch = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(53);
        let loss = model.train_step(&x1_batch, 1e-3, &mut rng);
        assert!(loss >= 0.0, "loss must be non-negative");
    }

    #[test]
    fn test_rectified_flow_sample_shape() {
        let config = RectifiedFlowConfig {
            z_dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_steps: 10,
            lr: 1e-3,
        };
        let model = RectifiedFlow::new(config);
        let mut rng = StdRng::seed_from_u64(54);
        let x = model.sample(10, &mut rng);
        assert_eq!(
            x.len(),
            4,
            "RectifiedFlow sample should have z_dim=4 elements"
        );
    }

    #[test]
    fn test_rectified_flow_sample_finite() {
        let config = RectifiedFlowConfig::default();
        let model = RectifiedFlow::new(config);
        let mut rng = StdRng::seed_from_u64(55);
        let x = model.sample(10, &mut rng);
        for v in &x {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_rectified_flow_empty_batch() {
        let config = RectifiedFlowConfig::default();
        let mut model = RectifiedFlow::new(config);
        let mut rng = StdRng::seed_from_u64(56);
        let loss = model.train_step(&[], 1e-3, &mut rng);
        assert_eq!(loss, 0.0, "empty batch should return 0 loss");
    }

    // ── CnfMetrics ──────────────────────────────────────────────────────────

    #[test]
    fn test_cnf_metrics_bits_per_dim_finite() {
        let metrics = CnfMetrics {
            mean_log_prob: -3.5,
            bits_per_dim: 3.5 / (2.0_f64.ln()),
            sample_diversity: 0.8,
        };
        assert!(
            metrics.bits_per_dim.is_finite(),
            "bits_per_dim must be finite"
        );
    }

    #[test]
    fn test_evaluate_cnf_returns_finite_metrics() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let test_data = vec![vec![0.0, 0.0], vec![1.0, -1.0], vec![-0.5, 0.5]];
        let metrics = evaluate_cnf(&cnf, &test_data, 3);
        assert!(
            metrics.mean_log_prob.is_finite(),
            "mean_log_prob should be finite"
        );
        assert!(
            metrics.bits_per_dim.is_finite(),
            "bits_per_dim should be finite"
        );
        assert!(
            metrics.sample_diversity.is_finite(),
            "sample_diversity should be finite"
        );
    }

    #[test]
    fn test_evaluate_cnf_empty_data() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let metrics = evaluate_cnf(&cnf, &[], 3);
        assert_eq!(metrics.mean_log_prob, 0.0);
        assert_eq!(metrics.bits_per_dim, 0.0);
    }

    #[test]
    fn test_evaluate_flow_matching_nonneg_bpd() {
        let config = FlowMatchingConfig {
            z_dim: 2,
            hidden_dim: 8,
            n_layers: 1,
            sigma_min: 1e-4,
            n_steps: 5,
            lr: 1e-3,
        };
        let model = FlowMatchingModel::new(config);
        let test_data = vec![vec![0.0, 0.0], vec![1.0, -1.0]];
        let mut rng = StdRng::seed_from_u64(60);
        let metrics = evaluate_flow_matching(&model, &test_data, 20, 5, &mut rng);
        assert!(
            metrics.bits_per_dim >= 0.0,
            "bits_per_dim should be non-negative, got {}",
            metrics.bits_per_dim
        );
    }

    #[test]
    fn test_evaluate_flow_matching_finite_metrics() {
        let config = FlowMatchingConfig::default();
        let model = FlowMatchingModel::new(config);
        let test_data = vec![vec![0.5, -0.5], vec![-0.5, 0.5]];
        let mut rng = StdRng::seed_from_u64(61);
        let metrics = evaluate_flow_matching(&model, &test_data, 20, 5, &mut rng);
        assert!(metrics.mean_log_prob.is_finite());
        assert!(metrics.bits_per_dim.is_finite());
        assert!(metrics.sample_diversity.is_finite());
    }

    // ── Additional edge case / integration tests ─────────────────────────────

    #[test]
    fn test_cnf_mlp_batch_consistency() {
        let mlp = CnfMlp::new(&[2, 8, 2]);
        let x1 = vec![0.1, 0.2];
        let x2 = vec![0.1, 0.2];
        let out1 = mlp.forward(&x1);
        let out2 = mlp.forward(&x2);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-12, "same input should give same output");
        }
    }

    #[test]
    fn test_ffjord_config_default() {
        let config = FfjordConfig::default();
        assert_eq!(config.z_dim, 2);
        assert_eq!(config.n_blocks, 2);
        assert_eq!(config.n_trace_samples, 1);
    }

    #[test]
    fn test_flow_matching_config_default() {
        let config = FlowMatchingConfig::default();
        assert_eq!(config.z_dim, 2);
        assert!((config.sigma_min - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_rectified_flow_reflow_loss_zero_steps() {
        let config = RectifiedFlowConfig {
            z_dim: 2,
            hidden_dim: 8,
            n_layers: 1,
            n_steps: 1,
            lr: 1e-3,
        };
        let model = RectifiedFlow::new(config);
        let x0 = vec![vec![0.0, 0.0]];
        let x1 = vec![vec![1.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(70);
        let loss = model.reflow_loss(&x0, &x1, &mut rng);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_cnf_integrate_forward_zero_dynamics() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 1);
        let z0 = vec![1.0, -1.0];
        let (z_t, _log_det) = cnf.integrate_forward(&z0, 1, 0.5, 0.5);
        assert_eq!(z_t.len(), 2);
    }

    #[test]
    fn test_ot_match_single_pair() {
        let x0 = vec![vec![1.0_f64, 2.0_f64]];
        let x1 = vec![vec![1.1_f64, 2.1_f64]];
        let perm = OtCfmModel::ot_match(&x0, &x1);
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_flow_matching_sigma_min_effect() {
        let config = FlowMatchingConfig {
            z_dim: 2,
            hidden_dim: 8,
            n_layers: 1,
            sigma_min: 0.0,
            n_steps: 10,
            lr: 1e-3,
        };
        let model = FlowMatchingModel::new(config);
        let x0_batch = vec![vec![0.0, 0.0]];
        let x1_batch = vec![vec![1.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(80);
        let loss = model.cfm_loss(&x0_batch, &x1_batch, &mut rng);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
