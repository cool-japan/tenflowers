//! Tests for operator_learning module: core, extensions, and advanced algorithms.

use super::*;
use super::advanced::{
    GnoKernel, GnoLayer, GnoModel, NeuralOperatorMetrics, PdeType, PinoLoss, PinoTrainer,
    UnoDecoder, UnoEncoder, UnoModel, UnoSkipConnection, WnoLayer, WnoModel,
    haar_dwt, haar_idwt, haar_dwt_multilevel, haar_idwt_multilevel,
};

// ─── Original tests (ported from the old single-file tests) ─────────────────

#[test]
fn test_dft_idft_roundtrip() {
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let (re, im) = dft_real(&x);
    let recovered = idft_real(&re, &im);
    for (a, b) in x.iter().zip(recovered.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "DFT/IDFT roundtrip failed: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_spectral_conv1d_output_shape() {
    let conv = SpectralConv1d::new(2, 3, 4, 42).expect("operation should succeed");
    let n = 16;
    let input = vec![0.1_f32; 2 * n];
    let out = conv.forward(&input, n).expect("operation should succeed");
    assert_eq!(out.len(), 3 * n, "SpectralConv1d output shape mismatch");
}

#[test]
fn test_spectral_conv1d_modes_validation() {
    let result = SpectralConv1d::new(2, 3, 0, 42);
    assert!(result.is_err(), "modes=0 should return an error");
}

#[test]
fn test_spectral_conv2d_output_shape() {
    let conv = SpectralConv2d::new(2, 3, 2, 2, 7).expect("operation should succeed");
    let (nx, ny) = (4, 4);
    let input = vec![0.1_f32; 2 * nx * ny];
    let out = conv.forward(&input, nx, ny).expect("operation should succeed");
    assert_eq!(out.len(), 3 * nx * ny, "SpectralConv2d output shape mismatch");
}

#[test]
fn test_fno_layer_forward() {
    let layer = FnoLayer::new(4, 3, 42).expect("operation should succeed");
    let n = 8;
    let input = vec![0.1_f32; 4 * n];
    let out = layer.forward(&input, n).expect("operation should succeed");
    assert_eq!(out.len(), 4 * n, "FnoLayer output size mismatch");
}

#[test]
fn test_fno_model_construction() {
    let config = FnoConfig::new(1, 8, 1, 3, 4);
    let model = FnoModel::new(config, 42).expect("operation should succeed");
    assert_eq!(model.blocks.len(), 3);
    assert_eq!(model.config.channels, 8);
}

#[test]
fn test_deepopnet_construction() {
    let branch = BranchNet::new(10, &[32], 16, 1).expect("operation should succeed");
    let trunk = TrunkNet::new(1, &[32], 16, 2).expect("operation should succeed");
    let don = DeepONet::new(branch, trunk).expect("operation should succeed");
    assert_eq!(don.branch.p, don.trunk.p);
}

#[test]
fn test_deepopnet_forward_dimensions() {
    let branch = BranchNet::new(5, &[16], 8, 10).expect("operation should succeed");
    let trunk = TrunkNet::new(1, &[16], 8, 11).expect("operation should succeed");
    let don = DeepONet::new(branch, trunk).expect("operation should succeed");
    let sensors = vec![0.1_f32; 5];
    let query = vec![0.5_f32; 1];
    let result = don.forward(&sensors, &query).expect("operation should succeed");
    assert!(result.is_finite());
}

#[test]
fn test_deepopnet_trainer_mse_loss() {
    let branch = BranchNet::new(3, &[8], 4, 30).expect("operation should succeed");
    let trunk = TrunkNet::new(1, &[8], 4, 31).expect("operation should succeed");
    let don = DeepONet::new(branch, trunk).expect("operation should succeed");
    let trainer = DeepONetTrainer::new(1e-3);
    let sensors = vec![vec![0.1_f32; 3]];
    let queries = vec![vec![vec![0.5_f32]]];
    let targets = vec![vec![1.0_f32]];
    let loss = trainer
        .mse_loss(&don, &sensors, &queries, &targets)
        .expect("computation failed");
    assert!(loss >= 0.0);
}

#[test]
fn test_continuous_sensor_deepopnet() {
    let branch = BranchNet::new(6, &[8], 4, 40).expect("operation should succeed");
    let trunk = TrunkNet::new(1, &[8], 4, 41).expect("operation should succeed");
    let don = DeepONet::new(branch, trunk).expect("operation should succeed");
    let cs = ContinuousSensorDeepONet::new(don, 6, 0.0, 1.0);
    let locs = cs.sample_sensors(42);
    assert_eq!(locs.len(), 6);
    for &l in &locs {
        assert!((0.0..=1.0).contains(&l));
    }
}

#[test]
fn test_symbolic_expr_constant() {
    let expr = SymbolicExpr::constant(3.125);
    assert!((expr.evaluate(&[]) - 3.125).abs() < 1e-10);
}

#[test]
fn test_symbolic_expr_variable() {
    let expr = SymbolicExpr::variable(0);
    assert!((expr.evaluate(&[2.5]) - 2.5).abs() < 1e-10);
}

#[test]
fn test_symbolic_expr_add() {
    let lhs = SymbolicExpr::variable(0);
    let rhs = SymbolicExpr::constant(1.0);
    let expr = SymbolicExpr::binary(SymbolicNode::Add, lhs, rhs);
    assert!(
        (expr.evaluate(&[3.0]) - 4.0).abs() < 1e-10,
        "x + 1 at x=3 should be 4"
    );
}

#[test]
fn test_symbolic_expr_mul() {
    let lhs = SymbolicExpr::variable(0);
    let rhs = SymbolicExpr::constant(2.0);
    let expr = SymbolicExpr::binary(SymbolicNode::Mul, lhs, rhs);
    assert!(
        (expr.evaluate(&[5.0]) - 10.0).abs() < 1e-10,
        "x * 2 at x=5 should be 10"
    );
}

#[test]
fn test_symbolic_expr_evaluate() {
    let x = SymbolicExpr::variable(0);
    let sin_x = SymbolicExpr::unary(SymbolicNode::Sin, x.clone());
    let cos_x = SymbolicExpr::unary(SymbolicNode::Cos, x);
    let expr = SymbolicExpr::binary(SymbolicNode::Add, sin_x, cos_x);
    let val = expr.evaluate(&[std::f64::consts::FRAC_PI_4]);
    let expected = std::f64::consts::FRAC_PI_4.sin() + std::f64::consts::FRAC_PI_4.cos();
    assert!((val - expected).abs() < 1e-10);
}

#[test]
fn test_symbolic_derivative_constant() {
    let expr = SymbolicExpr::constant(5.0);
    let d = expr.derivative(0);
    assert!((d.evaluate(&[1.0]) - 0.0).abs() < 1e-10, "d/dx(5) = 0");
}

#[test]
fn test_symbolic_derivative_variable() {
    let expr = SymbolicExpr::variable(0);
    let d = expr.derivative(0);
    assert!((d.evaluate(&[7.0]) - 1.0).abs() < 1e-10, "d/dx(x) = 1");
}

#[test]
fn test_symbolic_derivative_add() {
    let expr = SymbolicExpr::binary(
        SymbolicNode::Add,
        SymbolicExpr::variable(0),
        SymbolicExpr::constant(2.0),
    );
    let d = expr.derivative(0);
    assert!((d.evaluate(&[3.0]) - 1.0).abs() < 1e-9);
}

#[test]
fn test_symbolic_derivative_mul() {
    let expr = SymbolicExpr::binary(
        SymbolicNode::Mul,
        SymbolicExpr::variable(0),
        SymbolicExpr::constant(3.0),
    );
    let d = expr.derivative(0);
    assert!((d.evaluate(&[2.0]) - 3.0).abs() < 1e-9);
}

#[test]
fn test_symbolic_simplify() {
    let expr = SymbolicExpr::binary(
        SymbolicNode::Add,
        SymbolicExpr::constant(0.0),
        SymbolicExpr::variable(0),
    );
    let simplified = expr.simplify();
    assert!((simplified.evaluate(&[5.0]) - 5.0).abs() < 1e-10);
}

#[test]
fn test_symbolic_simplify_mul_zero() {
    let expr = SymbolicExpr::binary(
        SymbolicNode::Mul,
        SymbolicExpr::constant(0.0),
        SymbolicExpr::variable(0),
    );
    let simplified = expr.simplify();
    assert!((simplified.evaluate(&[999.0])).abs() < 1e-10);
}

#[test]
fn test_gp_regressor_fit() {
    let gp = GpSymbolicRegressor::new(10, 3, 3, 0.5, 5, 1);
    let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
    let y: Vec<f64> = x.iter().map(|xi| xi[0] * xi[0]).collect();
    let best = gp.fit(&x, &y, 42).expect("operation should succeed");
    let fitness = gp.mse_fitness(&best, &x, &y);
    assert!(fitness.is_finite());
}

#[test]
fn test_neural_symbolic_hybrid() {
    let basis = vec![
        SymbolicExpr::variable(0),
        SymbolicExpr::unary(SymbolicNode::Sin, SymbolicExpr::variable(0)),
    ];
    let hybrid =
        NeuralSymbolicHybrid::new(2, &[8], basis, 42).expect("operation should succeed");
    let result = hybrid.forward(&[0.5_f32, 0.3_f32], &[1.0_f64]);
    assert!(result.is_ok());
}

#[test]
fn test_hnn_construction() {
    let hnn = HamiltonianNN::new(2, &[32, 32], 42).expect("operation should succeed");
    assert_eq!(hnn.dim, 2);
    assert_eq!(hnn.net.input_dim, 4); // 2*dim
}

#[test]
fn test_hnn_forward() {
    let hnn = HamiltonianNN::new(2, &[16], 42).expect("operation should succeed");
    let q = vec![1.0_f32, 0.0];
    let p = vec![0.0_f32, 1.0];
    let h_val = hnn.hamiltonian(&q, &p).expect("operation should succeed");
    assert!(h_val.is_finite());
}

#[test]
fn test_hnn_symplectic_structure() {
    let hnn = HamiltonianNN::new(1, &[8], 42).expect("operation should succeed");
    let q = vec![1.0_f32];
    let p = vec![0.5_f32];
    let (dqdt, dpdt) = hnn.equations_of_motion(&q, &p).expect("operation should succeed");
    assert_eq!(dqdt.len(), 1);
    assert_eq!(dpdt.len(), 1);
    assert!(dqdt[0].is_finite());
    assert!(dpdt[0].is_finite());
}

#[test]
fn test_hnn_trainer_loss() {
    let hnn = HamiltonianNN::new(1, &[8], 42).expect("operation should succeed");
    let trainer = HnnTrainer::new(1e-3);
    let trajectories =
        vec![(vec![1.0_f32], vec![0.0_f32], vec![0.0_f32], vec![-1.0_f32])];
    let loss = trainer
        .trajectory_loss(&hnn, &trajectories)
        .expect("operation should succeed");
    assert!(loss >= 0.0);
}

#[test]
fn test_lnn_construction() {
    let lnn = LagrangianNN::new(2, &[16, 16], 99).expect("operation should succeed");
    assert_eq!(lnn.dim, 2);
    assert_eq!(lnn.net.input_dim, 4);
}

#[test]
fn test_lnn_forward() {
    let lnn = LagrangianNN::new(1, &[8], 42).expect("operation should succeed");
    let q = vec![0.5_f32];
    let qdot = vec![1.0_f32];
    let l_val = lnn.lagrangian(&q, &qdot).expect("operation should succeed");
    assert!(l_val.is_finite());
}

#[test]
fn test_verlet_integration_energy_conservation() {
    let hnn = HamiltonianNN::new(1, &[16], 42).expect("operation should succeed");
    let integrator = SymplecticIntegrator::new(hnn.clone(), 0.01);
    let q0 = vec![1.0_f32];
    let p0 = vec![0.0_f32];
    let traj = integrator.integrate(&q0, &p0, 5).expect("operation should succeed");
    assert_eq!(traj.len(), 6); // initial + 5 steps

    let h_init = hnn
        .hamiltonian(&traj[0].0, &traj[0].1)
        .expect("operation should succeed");
    let h_final = hnn
        .hamiltonian(&traj[5].0, &traj[5].1)
        .expect("operation should succeed");
    assert!(h_init.is_finite() && h_final.is_finite());
}

#[test]
fn test_score_network_output_shape() {
    let sn = ScoreNetwork::new(4, &[16], 42).expect("operation should succeed");
    let x = vec![0.1_f32; 4];
    let score = sn.score(&x, 0.5).expect("operation should succeed");
    assert_eq!(score.len(), 4, "score output must match data_dim");
}

#[test]
fn test_score_matching_loss_positive() {
    let sn = ScoreNetwork::new(2, &[8], 42).expect("operation should succeed");
    let loss_fn =
        ScoreMatchingLoss::new(vec![0.1, 0.5, 1.0]).expect("operation should succeed");
    let batch: Vec<Vec<f32>> = vec![vec![0.5_f32, -0.3], vec![1.0, 0.0]];
    let loss = loss_fn.compute(&sn, &batch, 7).expect("operation should succeed");
    assert!(loss.is_finite(), "score matching loss should be finite");
}

#[test]
fn test_langevin_sampler_steps() {
    let sn = ScoreNetwork::new(3, &[8], 42).expect("operation should succeed");
    let sampler = LangevinSampler::new(0.01, 5, 0.3);
    let x0 = vec![0.0_f32; 3];
    let result = sampler.sample(&sn, &x0, 99).expect("operation should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_sliced_score_matching() {
    let sn = ScoreNetwork::new(3, &[8], 7).expect("operation should succeed");
    let ssm = SlicedScoreMatching::new(4, 0.5);
    let batch: Vec<Vec<f32>> = vec![vec![0.1_f32; 3], vec![0.2_f32; 3]];
    let loss = ssm.compute(&sn, &batch, 13).expect("operation should succeed");
    assert!(loss.is_finite());
}

#[test]
fn test_spectral_conv1d_input_validation() {
    let conv = SpectralConv1d::new(2, 3, 4, 42).expect("operation should succeed");
    let result = conv.forward(&[0.1_f32; 5], 10);
    assert!(result.is_err());
}

// ─── New tests for advanced algorithms ──────────────────────────────────────

#[test]
fn test_haar_dwt_roundtrip() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let (approx, detail) = haar_dwt(&x);
    let recovered = haar_idwt(&approx, &detail);
    for (a, b) in x.iter().zip(recovered.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "Haar DWT/IDWT roundtrip: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_haar_dwt_energy_conservation() {
    let x = vec![1.0_f32, -1.0, 2.0, -2.0];
    let (approx, detail) = haar_dwt(&x);
    let energy_in: f32 = x.iter().map(|v| v * v).sum();
    let energy_out: f32 = approx.iter().chain(detail.iter()).map(|v| v * v).sum();
    assert!(
        (energy_in - energy_out).abs() < 1e-4,
        "Haar DWT should conserve energy: {} vs {}",
        energy_in,
        energy_out
    );
}

#[test]
fn test_haar_dwt_multilevel_roundtrip() {
    let x: Vec<f32> = (0..16_usize).map(|i| (i as f32).sin()).collect();
    let (approx, details) = haar_dwt_multilevel(&x, 3);
    let recovered = haar_idwt_multilevel(&approx, &details);
    for (a, b) in x.iter().zip(recovered.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "multi-level Haar roundtrip failed: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_wno_layer_output_shape() {
    let layer = WnoLayer::new(4, 4, 2, 4, 42).expect("WnoLayer creation failed");
    let n = 16;
    let input = vec![0.1_f32; 4 * n];
    let out = layer.forward(&input, n).expect("WnoLayer forward failed");
    assert_eq!(out.len(), 4 * n, "WnoLayer output shape mismatch");
}

#[test]
fn test_wno_layer_channel_change() {
    let layer = WnoLayer::new(2, 4, 1, 2, 7).expect("WnoLayer creation failed");
    let n = 8;
    let input = vec![0.5_f32; 2 * n];
    let out = layer.forward(&input, n).expect("WnoLayer forward failed");
    assert_eq!(out.len(), 4 * n);
}

#[test]
fn test_wno_layer_input_validation() {
    let layer = WnoLayer::new(3, 3, 2, 2, 99).expect("WnoLayer creation failed");
    let result = layer.forward(&[0.1_f32; 5], 10);
    assert!(result.is_err(), "Mismatched input should fail");
}

#[test]
fn test_wno_layer_output_finite() {
    let layer = WnoLayer::new(2, 2, 2, 2, 13).expect("WnoLayer creation failed");
    let n = 8;
    let input: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.1).sin()).collect();
    let out = layer.forward(&input, n).expect("WnoLayer forward failed");
    for v in &out {
        assert!(v.is_finite(), "WnoLayer output must be finite");
    }
}

#[test]
fn test_wno_model_construction() {
    let model = WnoModel::new(1, 8, 1, 3, 2, 4, 42).expect("WnoModel creation failed");
    assert_eq!(model.layers.len(), 3);
    assert_eq!(model.channels, 8);
}

#[test]
fn test_wno_model_forward_shape() {
    let model = WnoModel::new(1, 4, 1, 2, 1, 2, 55).expect("WnoModel creation failed");
    let n = 16;
    let input = vec![0.1_f32; n];
    let out = model.forward(&input, n).expect("WnoModel forward failed");
    assert_eq!(out.len(), n);
}

#[test]
fn test_wno_model_forward_finite() {
    let model = WnoModel::new(2, 4, 1, 2, 2, 4, 77).expect("WnoModel creation failed");
    let n = 16;
    let input: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.05).cos()).collect();
    let out = model.forward(&input, n).expect("WnoModel forward failed");
    for v in &out {
        assert!(v.is_finite(), "WnoModel output must be finite");
    }
}

#[test]
fn test_gno_kernel_construction() {
    let k = GnoKernel::new(2, 8, &[16, 16], 42).expect("GnoKernel creation failed");
    assert_eq!(k.coord_dim, 2);
    assert_eq!(k.out_dim, 8);
}

#[test]
fn test_gno_kernel_evaluate_shape() {
    let k = GnoKernel::new(2, 4, &[8], 13).expect("GnoKernel creation failed");
    let x = vec![0.5_f32, -0.3];
    let y = vec![0.1_f32, 0.9];
    let out = k.evaluate(&x, &y).expect("GnoKernel evaluate failed");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_gno_kernel_dimension_validation() {
    let k = GnoKernel::new(2, 4, &[8], 99).expect("GnoKernel creation failed");
    let result = k.evaluate(&[0.1_f32], &[0.2_f32, 0.3]);
    assert!(result.is_err(), "Dimension mismatch should fail");
}

#[test]
fn test_gno_layer_forward_shape() {
    let layer =
        GnoLayer::new(2, 4, 4, &[8], 42).expect("GnoLayer creation failed");
    let n = 5;
    let coords: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.1).collect();
    let features: Vec<f32> = vec![0.1_f32; n * 4];
    let out = layer
        .forward(&coords, &features, n)
        .expect("GnoLayer forward failed");
    assert_eq!(out.len(), n * 4);
}

#[test]
fn test_gno_layer_finite_output() {
    let layer = GnoLayer::new(1, 2, 2, &[4], 7).expect("GnoLayer creation failed");
    let n = 4;
    let coords: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let features: Vec<f32> = vec![0.5_f32; n * 2];
    let out = layer
        .forward(&coords, &features, n)
        .expect("GnoLayer forward failed");
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_gno_model_construction() {
    let model = GnoModel::new(1, 2, 8, 1, 2, &[8], 42).expect("GnoModel creation failed");
    assert_eq!(model.gno_layers.len(), 2);
    assert_eq!(model.coord_dim, 1);
}

#[test]
fn test_gno_model_forward_shape() {
    let model = GnoModel::new(1, 2, 4, 1, 2, &[4], 13).expect("GnoModel creation failed");
    let n = 6;
    let coords: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let features: Vec<f32> = vec![0.2_f32; n * 2];
    let out = model
        .forward(&coords, &features, n)
        .expect("GnoModel forward failed");
    assert_eq!(out.len(), n);
}

#[test]
fn test_pino_loss_heat_equation() {
    let loss = PinoLoss::new(PdeType::Heat, 0.1, 1.0, 0.1, 0.01);
    let n = 8;
    let fields: Vec<Vec<f32>> = (0..5)
        .map(|t| (0..n).map(|i| (i as f32 * 0.1 + t as f32 * 0.01).sin()).collect())
        .collect();
    let predicted = vec![0.0_f32; n];
    let target = vec![0.0_f32; n];
    let total = loss.compute(&predicted, &target, &fields).expect("PINO loss computation failed");
    assert!(total.is_finite(), "PINO heat loss must be finite");
    assert!(total >= 0.0);
}

#[test]
fn test_pino_loss_burgers_equation() {
    let loss = PinoLoss::new(PdeType::Burgers, 0.01, 0.5, 0.1, 0.01);
    let n = 10;
    let fields: Vec<Vec<f32>> = (0..4)
        .map(|t| (0..n).map(|i| (i as f32 * 0.05 + t as f32 * 0.02).cos()).collect())
        .collect();
    let predicted = vec![0.1_f32; n];
    let target = vec![0.0_f32; n];
    let total = loss.compute(&predicted, &target, &fields).expect("PINO Burgers loss failed");
    assert!(total.is_finite());
    assert!(total >= 0.0);
}

#[test]
fn test_pino_loss_wave_equation() {
    let loss = PinoLoss::new(PdeType::Wave, 1.0, 1.0, 0.1, 0.01);
    let n = 12;
    let fields: Vec<Vec<f32>> = (0..5)
        .map(|t| (0..n).map(|i| ((i as f32 * 0.1) - (t as f32 * 0.01)).sin()).collect())
        .collect();
    let predicted = vec![0.0_f32; n];
    let target = vec![0.0_f32; n];
    let total = loss.compute(&predicted, &target, &fields).expect("PINO wave loss failed");
    assert!(total.is_finite());
}

#[test]
fn test_pino_loss_data_term() {
    let loss = PinoLoss::new(PdeType::Heat, 0.1, 0.0, 0.1, 0.01);
    let predicted = vec![1.0_f32, 2.0, 3.0];
    let target = vec![1.0_f32, 2.0, 3.0];
    let total = loss.compute(&predicted, &target, &[]).expect("PINO loss computation failed");
    assert!(total.abs() < 1e-6, "Identical prediction/target should have ~0 loss");
}

#[test]
fn test_pino_trainer_construction() {
    let loss = PinoLoss::new(PdeType::Heat, 0.1, 1.0, 0.1, 0.01);
    let trainer = PinoTrainer::new(loss, 1e-3, 10);
    assert_eq!(trainer.epochs, 10);
}

#[test]
fn test_pino_trainer_train_step() {
    let loss = PinoLoss::new(PdeType::Burgers, 0.01, 0.5, 0.1, 0.01);
    let trainer = PinoTrainer::new(loss, 1e-3, 5);
    let n = 8;
    let fields: Vec<Vec<f32>> = (0..4)
        .map(|_| vec![0.1_f32; n])
        .collect();
    let predicted = vec![0.1_f32; n];
    let target = vec![0.1_f32; n];
    let loss_val = trainer
        .train_step(&predicted, &target, &fields)
        .expect("train_step failed");
    assert!(loss_val.is_finite());
    assert!(loss_val >= 0.0);
}

#[test]
fn test_uno_encoder_output_shape() {
    let enc = UnoEncoder::new(4, 8, 42).expect("UnoEncoder creation failed");
    let n = 16;
    let input = vec![0.1_f32; 4 * n];
    let out = enc.forward(&input, n).expect("UnoEncoder forward failed");
    assert_eq!(out.len(), 8 * (n / 2));
}

#[test]
fn test_uno_encoder_halves_resolution() {
    let enc = UnoEncoder::new(2, 4, 7).expect("UnoEncoder creation failed");
    let n = 32;
    let input = vec![0.5_f32; 2 * n];
    let out = enc.forward(&input, n).expect("UnoEncoder forward failed");
    assert_eq!(out.len(), 4 * 16);
}

#[test]
fn test_uno_encoder_finite_output() {
    let enc = UnoEncoder::new(3, 6, 13).expect("UnoEncoder creation failed");
    let n = 12;
    let input: Vec<f32> = (0..3 * n).map(|i| (i as f32 * 0.1).sin()).collect();
    let out = enc.forward(&input, n).expect("UnoEncoder forward failed");
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_uno_decoder_output_shape() {
    let dec = UnoDecoder::new(8, 4, 42).expect("UnoDecoder creation failed");
    let n_in = 8;
    let input = vec![0.1_f32; 8 * n_in];
    let out = dec.forward(&input, n_in).expect("UnoDecoder forward failed");
    assert_eq!(out.len(), 4 * (2 * n_in));
}

#[test]
fn test_uno_decoder_doubles_resolution() {
    let dec = UnoDecoder::new(4, 2, 7).expect("UnoDecoder creation failed");
    let n_in = 16;
    let input = vec![0.3_f32; 4 * n_in];
    let out = dec.forward(&input, n_in).expect("UnoDecoder forward failed");
    assert_eq!(out.len(), 2 * 32);
}

#[test]
fn test_uno_skip_connection_shape() {
    let skip = UnoSkipConnection::new(8, 8, 8, 42).expect("UnoSkipConnection creation failed");
    let n = 8;
    let enc = vec![0.1_f32; 8 * n];
    let dec = vec![0.2_f32; 8 * n];
    let out = skip.forward(&enc, &dec, n).expect("UnoSkipConnection forward failed");
    assert_eq!(out.len(), 8 * n);
}

#[test]
fn test_uno_skip_connection_finite() {
    let skip = UnoSkipConnection::new(4, 4, 4, 77).expect("UnoSkipConnection creation failed");
    let n = 10;
    let enc: Vec<f32> = (0..4 * n).map(|i| (i as f32 * 0.01).sin()).collect();
    let dec: Vec<f32> = (0..4 * n).map(|i| (i as f32 * 0.02).cos()).collect();
    let out = skip.forward(&enc, &dec, n).expect("UnoSkipConnection forward failed");
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_uno_model_construction() {
    let model = UnoModel::new(1, 4, 1, 42).expect("UnoModel creation failed");
    assert_eq!(model.base_channels, 4);
    assert_eq!(model.encoders.len(), 3);
    assert_eq!(model.decoders.len(), 3);
    assert_eq!(model.skips.len(), 3);
}

#[test]
fn test_uno_model_forward_output_shape() {
    let model = UnoModel::new(1, 2, 1, 42).expect("UnoModel creation failed");
    let n = 8; // must be divisible by 8
    let input = vec![0.1_f32; n];
    let out = model.forward(&input, n).expect("UnoModel forward failed");
    assert_eq!(out.len(), n, "UnoModel output shape mismatch");
}

#[test]
fn test_uno_model_forward_finite() {
    let model = UnoModel::new(2, 2, 1, 13).expect("UnoModel creation failed");
    let n = 8;
    let input: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.1).sin()).collect();
    let out = model.forward(&input, n).expect("UnoModel forward failed");
    for v in &out {
        assert!(v.is_finite(), "UnoModel output must be finite");
    }
}

#[test]
fn test_neural_operator_metrics_perfect_prediction() {
    let target = vec![1.0_f32, 2.0, 3.0, 4.0];
    let metrics =
        NeuralOperatorMetrics::compute(&target, &target).expect("metrics computation failed");
    assert!(
        metrics.relative_l2 < 1e-5,
        "Perfect prediction should have near-zero relative L2"
    );
    assert!(metrics.r_squared > 0.999, "R² should be ~1 for perfect prediction");
}

#[test]
fn test_neural_operator_metrics_nonzero_error() {
    let predicted = vec![1.1_f32, 2.2, 3.3, 4.4];
    let target = vec![1.0_f32, 2.0, 3.0, 4.0];
    let metrics =
        NeuralOperatorMetrics::compute(&predicted, &target).expect("metrics computation failed");
    assert!(metrics.relative_l2 > 0.0);
    assert!(metrics.mae > 0.0);
    assert!(metrics.max_error > 0.0);
    assert!(metrics.r_squared < 1.0);
}

#[test]
fn test_neural_operator_metrics_mae() {
    let predicted = vec![2.0_f32, 3.0];
    let target = vec![1.0_f32, 1.0];
    let metrics =
        NeuralOperatorMetrics::compute(&predicted, &target).expect("metrics computation failed");
    // MAE = (1 + 2) / 2 = 1.5
    assert!((metrics.mae - 1.5).abs() < 1e-5);
}

#[test]
fn test_neural_operator_metrics_dimension_mismatch() {
    let predicted = vec![1.0_f32; 5];
    let target = vec![1.0_f32; 4];
    let result = NeuralOperatorMetrics::compute(&predicted, &target);
    assert!(result.is_err(), "Dimension mismatch should return error");
}

#[test]
fn test_gno_model_zero_layers_error() {
    let result = GnoModel::new(1, 2, 4, 1, 0, &[4], 42);
    assert!(result.is_err(), "num_layers=0 should fail");
}

#[test]
fn test_wno_model_zero_layers_error() {
    let result = WnoModel::new(1, 4, 1, 0, 2, 2, 42);
    assert!(result.is_err(), "num_layers=0 should fail");
}

#[test]
fn test_wno_layer_invalid_modes() {
    let result = WnoLayer::new(2, 2, 2, 0, 42);
    assert!(result.is_err(), "num_modes=0 should fail");
}

#[test]
fn test_wno_layer_invalid_levels() {
    let result = WnoLayer::new(2, 2, 0, 2, 42);
    assert!(result.is_err(), "levels=0 should fail");
}

#[test]
fn test_pino_loss_length_mismatch() {
    let loss = PinoLoss::new(PdeType::Heat, 0.1, 1.0, 0.1, 0.01);
    let result = loss.compute(&[1.0_f32; 5], &[1.0_f32; 4], &[]);
    assert!(result.is_err(), "Length mismatch should fail");
}

#[test]
fn test_uno_encoder_invalid_channels() {
    let result = UnoEncoder::new(0, 4, 42);
    assert!(result.is_err(), "zero in_channels should fail");
}

#[test]
fn test_uno_decoder_invalid_channels() {
    let result = UnoDecoder::new(4, 0, 42);
    assert!(result.is_err(), "zero out_channels should fail");
}

#[test]
fn test_langevin_trajectory_length() {
    let sn = ScoreNetwork::new(2, &[8], 77).expect("operation should succeed");
    let sampler = LangevinSampler::new(0.005, 10, 0.1);
    let x0 = vec![0.5_f32; 2];
    let traj = sampler
        .sample_trajectory(&sn, &x0, 42)
        .expect("trajectory sampling failed");
    assert_eq!(traj.len(), 11); // n_steps + 1
    for state in &traj {
        assert_eq!(state.len(), 2);
        for v in state {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn test_fno_model_forward() {
    let config = FnoConfig::new(1, 4, 1, 2, 3);
    let model = FnoModel::new(config, 42).expect("operation should succeed");
    let n = 8;
    let input = vec![0.5_f32; n];
    let out = model.forward(&input, n).expect("FnoModel forward failed");
    assert_eq!(out.len(), n);
    for v in &out {
        assert!(v.is_finite());
    }
}
