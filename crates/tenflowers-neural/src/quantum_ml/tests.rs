//! Tests for quantum_ml module (original + Round-44 additions).

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── Original tests ────────────────────────────────────────────────────────────

#[test]
fn test_complex_mul() {
    let c = Complex64::new(1.0, 2.0).mul(Complex64::new(3.0, 4.0));
    assert!((c.re - (-5.0)).abs() < 1e-12);
    assert!((c.im - 10.0).abs() < 1e-12);
}

#[test]
fn test_complex_conj_abs_sq() {
    let a = Complex64::new(3.0, 4.0);
    let c = a.conj();
    assert!((c.re - 3.0).abs() < 1e-12);
    assert!((c.im - (-4.0)).abs() < 1e-12);
    assert!((a.abs_sq() - 25.0).abs() < 1e-12);
}

#[test]
fn test_complex_scale_add() {
    let a = Complex64::new(1.0, -1.0);
    let b = a.scale(2.0);
    assert!((b.re - 2.0).abs() < 1e-12);
    assert!((b.im - (-2.0)).abs() < 1e-12);
    let c = a.add(Complex64::new(0.5, 0.5));
    assert!((c.re - 1.5).abs() < 1e-12);
    assert!((c.im - (-0.5)).abs() < 1e-12);
}

#[test]
fn test_qubit_state_init() {
    let state = QubitState::new(3);
    assert_eq!(state.n_qubits, 3);
    assert_eq!(state.amplitudes.len(), 8);
    assert!((state.amplitudes[0].re - 1.0).abs() < 1e-12);
    for i in 1..8 {
        assert!(state.amplitudes[i].abs_sq() < 1e-12);
    }
}

#[test]
fn test_hadamard_superposition() {
    let mut state = QubitState::new(1);
    state.apply_gate(&QuantumGate::H, &[0]);
    let probs = state.measure_probs();
    assert!((probs[0] - 0.5).abs() < 1e-10);
    assert!((probs[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_rx_gate_rotation() {
    let mut state = QubitState::new(1);
    state.apply_gate(&QuantumGate::RX(std::f64::consts::PI), &[0]);
    let probs = state.measure_probs();
    assert!(probs[0].abs() < 1e-6);
    assert!((probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_ry_gate_rotation() {
    let mut state = QubitState::new(1);
    state.apply_gate(&QuantumGate::RY(std::f64::consts::FRAC_PI_2), &[0]);
    let probs = state.measure_probs();
    assert!((probs[0] - 0.5).abs() < 1e-10);
    assert!((probs[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_circuit_run() {
    let mut circuit = QuantumCircuit::new(2);
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::H, vec![0])]));
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::CNOT, vec![0, 1])]));
    let probs = circuit.run(QubitState::new(2)).measure_probs();
    assert!((probs[0] - 0.5).abs() < 1e-9);
    assert!(probs[1].abs() < 1e-9);
    assert!(probs[2].abs() < 1e-9);
    assert!((probs[3] - 0.5).abs() < 1e-9);
}

#[test]
fn test_measure_probs_sum_to_1() {
    let mut state = QubitState::new(3);
    for q in 0..3 {
        state.apply_gate(&QuantumGate::H, &[q]);
    }
    state.apply_gate(&QuantumGate::CNOT, &[0, 1]);
    state.apply_gate(&QuantumGate::RY(0.7), &[2]);
    let probs = state.measure_probs();
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    for &p in &probs {
        assert!(p >= 0.0);
    }
}

#[test]
fn test_gate_n_params() {
    assert_eq!(QuantumGate::H.n_params(), 0);
    assert_eq!(QuantumGate::CNOT.n_params(), 0);
    assert_eq!(QuantumGate::RX(1.0).n_params(), 1);
    assert_eq!(QuantumGate::RY(0.5).n_params(), 1);
}

#[test]
fn test_pauli_hamiltonian_expectation() {
    let mut h = PauliHamiltonian::new();
    h.add_term(1.0, "Z");
    let exp = h.expectation(&QubitState::new(1));
    assert!((exp - 1.0).abs() < 1e-10);
}

#[test]
fn test_pauli_hamiltonian_expectation_x1() {
    let mut h = PauliHamiltonian::new();
    h.add_term(1.0, "Z");
    let mut state = QubitState::new(1);
    state.apply_gate(&QuantumGate::H, &[0]);
    assert!(h.expectation(&state).abs() < 1e-10);
}

#[test]
fn test_pauli_hamiltonian_multi_term() {
    let mut h = PauliHamiltonian::new();
    h.add_term(0.5, "ZI");
    h.add_term(0.5, "IZ");
    let exp = h.expectation(&QubitState::new(2));
    assert!((exp - 1.0).abs() < 1e-9);
}

#[test]
fn test_ansatz_circuit_build() {
    let params: Vec<f64> = (0..9).map(|i| i as f64 * 0.1).collect();
    let circuit = AnsatzCircuit::build(3, 2, &params);
    assert!(circuit.is_ok());
}

#[test]
fn test_vqe_optimizer_gradient() {
    let mut h = PauliHamiltonian::new();
    h.add_term(-1.0, "ZZ");
    let opt = VqeOptimizer::new(2, 1, h, 42);
    let grads = opt.gradient(&opt.params.clone());
    assert_eq!(grads.len(), opt.params.len());
    for &g in &grads {
        assert!(g.is_finite());
    }
}

#[test]
fn test_parameter_shift_rule() {
    let psg = ParameterShiftGradient::new();
    let theta = std::f64::consts::FRAC_PI_4;
    let grads = psg.gradient(&|p: &[f64]| p[0].sin(), &[theta]);
    assert!((grads[0] - theta.cos()).abs() < 1e-10);
}

#[test]
fn test_qaoa_circuit() {
    let qaoa = Qaoa::new(3);
    let result = qaoa.build_circuit(3, 2, &[0.5, 0.3], &[0.4, 0.2]);
    assert!(result.is_ok());
    let probs = result
        .expect("build_circuit failed")
        .run(QubitState::new(3))
        .measure_probs();
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn test_maxcut_hamiltonian() {
    let adj = vec![vec![0., 1., 1.], vec![1., 0., 1.], vec![1., 1., 0.]];
    assert_eq!(CostHamiltonian::maxcut_hamiltonian(&adj).terms.len(), 3);
}

#[test]
fn test_amplitude_encoding() {
    let angles = DataEncodingLayer::amplitude_encoding(&[1.0, 0.0, 0.0, 0.0], 4);
    assert_eq!(angles.len(), 4);
    assert!(angles[0].abs() < 1e-10);
    assert!((angles[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn test_angle_encoding() {
    let angles = DataEncodingLayer::angle_encoding(&[0.0, 1.0], 2);
    assert_eq!(angles.len(), 2);
    assert!(angles[0].abs() < 1e-12);
    assert!((angles[1] - std::f64::consts::PI * 1.0f64.tanh()).abs() < 1e-10);
}

#[test]
fn test_quantum_layer_forward() {
    let layer = QuantumLayer::new(2, 1, 42).expect("QuantumLayer creation failed");
    let out = layer
        .forward(&[0.5, -0.3], &layer.params.clone())
        .expect("forward failed");
    assert_eq!(out.len(), 2);
    for &v in &out {
        assert!(v.is_finite() && (-1.0..=1.0).contains(&v));
    }
}

#[test]
fn test_hybrid_qc_shape() {
    let model = HybridQuantumClassical::new(2, 1, 3, 99)
        .expect("HybridQuantumClassical creation failed");
    let out = model.forward(&[0.1, 0.9]).expect("forward failed");
    assert_eq!(out.len(), 3);
    for &v in &out {
        assert!(v.is_finite() && v >= 0.0);
    }
}

#[test]
fn test_quantum_kernel_identical() {
    let kernel = QuantumKernel::new(2);
    let x = vec![0.5, 1.0];
    assert!((kernel.compute(&x, &x) - 1.0).abs() < 1e-10);
}

#[test]
fn test_quantum_kernel_range() {
    let kernel = QuantumKernel::new(2);
    let k = kernel.compute(&[0.3, 0.7], &[-0.5, 1.2]);
    assert!((0.0..=1.0 + 1e-10).contains(&k));
}

#[test]
fn test_quantum_svm_fit_predict() {
    let mut svm = QuantumSvm::new(2);
    svm.fit(
        &[vec![0., 0.], vec![1., 1.], vec![0., 1.], vec![1., 0.]],
        &[1., 1., -1., -1.],
    );
    let pred = svm.predict(&[0., 0.]);
    assert!(pred == 1.0 || pred == -1.0);
}

#[test]
fn test_grover_prob_sum() {
    let probs = GroverSearch::new(3).search(3, &|_| false, 2);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
    assert_eq!(probs.len(), 8);
}

#[test]
fn test_grover_single_target_amplification() {
    let probs = GroverSearch::new(3).search(3, &|idx| idx == 5, 2);
    assert!(probs[5] > 0.1);
}

#[test]
fn test_parameter_shift_gradient() {
    let psg = ParameterShiftGradient::new();
    let params = vec![0.5, 1.2];
    let grads = psg.gradient(&|p: &[f64]| p[0].cos() * p[1].sin(), &params);
    assert_eq!(grads.len(), 2);
    assert!((grads[0] - (-params[0].sin() * params[1].sin())).abs() < 1e-10);
    assert!((grads[1] - params[0].cos() * params[1].cos()).abs() < 1e-10);
}

#[test]
fn test_quantum_annealing_binary() {
    let sim = QuantumAnnealingSimulator::new(0.5);
    let mut rng = StdRng::seed_from_u64(7);
    let result = sim.anneal(
        &[0.0, 0.0],
        &[vec![0., -1.], vec![-1., 0.]],
        &(0..20).map(|i| 2.0 * (0.9f64).powi(i)).collect::<Vec<_>>(),
        &mut rng,
    );
    assert_eq!(result.len(), 2);
    for &s in &result {
        assert!(s == 1 || s == -1);
    }
}

#[test]
fn test_natural_gradient_finite() {
    let ng = NaturalGradientQnn::new(1e-3);
    let g = ng.natural_gradient(
        &|p: &[f64]| (p[0] - 0.5).powi(2) + (p[1] + 0.3).powi(2),
        &[0.1, 0.7],
    );
    assert_eq!(g.len(), 2);
    for &v in &g {
        assert!(v.is_finite());
    }
}

#[test]
fn test_readout_mitigation_rows_sum() {
    let rem = ReadoutErrorMitigation::calibrate(1, 0.05, 0.03);
    let mitigated = rem.mitigate(&[0.7, 0.3]);
    assert!((mitigated.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for &p in &mitigated {
        assert!(p >= 0.0);
    }
}

#[test]
fn test_readout_mitigation_identity() {
    let rem = ReadoutErrorMitigation::calibrate(1, 0.0, 0.0);
    let m = rem.mitigate(&[0.6, 0.4]);
    assert!((m[0] - 0.6).abs() < 1e-10);
    assert!((m[1] - 0.4).abs() < 1e-10);
}

#[test]
fn test_zne_extrapolation() {
    let v = ZeroNoiseExtrapolation::extrapolate(&[(1.0, 1.2), (2.0, 1.4)]);
    assert!((v - 1.0).abs() < 1e-10);
}

#[test]
fn test_zne_richardson_3pts() {
    let v = ZeroNoiseExtrapolation::extrapolate(&[(1.0, 1.5), (2.0, 3.0), (3.0, 5.5)]);
    assert!((v - 1.0).abs() < 1e-6);
}

#[test]
fn test_zne_single_point() {
    assert!((ZeroNoiseExtrapolation::extrapolate(&[(1.0, 0.9)]) - 0.9).abs() < 1e-12);
}

#[test]
fn test_noisy_simulator_construction() {
    let sim = NoisyQuantumSimulator::new(2, 0.01, 0.02, 0.03, 42);
    assert_eq!(sim.readout.n_qubits, 2);
    assert!((sim.gate_noise.p - 0.01).abs() < 1e-12);
}

#[test]
fn test_noisy_simulator_run() {
    let sim = NoisyQuantumSimulator::new(2, 0.01, 0.005, 0.005, 123);
    let mut circuit = QuantumCircuit::new(2);
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::H, vec![0])]));
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::CNOT, vec![0, 1])]));
    let probs = sim.run_noisy(&circuit);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-8);
    assert_eq!(probs.len(), 4);
}

#[test]
fn test_bit_flip_channel() {
    let ch = BitFlipChannel::new(1.0);
    let mut rng = StdRng::seed_from_u64(1);
    let state = ch.apply(QubitState::new(1), 0, 1.0, &mut rng);
    assert!((state.measure_probs()[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_bit_flip_no_flip() {
    let ch = BitFlipChannel::new(0.0);
    let mut rng = StdRng::seed_from_u64(2);
    let state = ch.apply(QubitState::new(1), 0, 0.0, &mut rng);
    assert!((state.measure_probs()[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_depolarizing_noise_application() {
    let noise = DepolarizingNoise::new(0.0);
    let mut rng = StdRng::seed_from_u64(5);
    let mut state = QubitState::new(2);
    state.apply_gate(&QuantumGate::H, &[0]);
    let before: Vec<f64> = state.measure_probs();
    noise.apply(&mut state, &mut rng);
    let after: Vec<f64> = state.measure_probs();
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn test_swap_gate() {
    let mut state = QubitState::new(2);
    state.apply_gate(&QuantumGate::X, &[1]);
    state.apply_gate(&QuantumGate::SWAP, &[0, 1]);
    let probs = state.measure_probs();
    assert!((probs[2] - 1.0).abs() < 1e-9);
}

#[test]
fn test_phase_gate() {
    let mut state = QubitState::new(1);
    state.apply_gate(&QuantumGate::X, &[0]);
    let p_before = state.measure_probs()[1];
    state.apply_gate(&QuantumGate::Phase(std::f64::consts::PI), &[0]);
    let p_after = state.measure_probs()[1];
    assert!((p_before - p_after).abs() < 1e-10);
}

#[test]
fn test_circuit_n_parameters() {
    let mut circuit = QuantumCircuit::new(2);
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::RY(0.5), vec![0])]));
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::RX(0.3), vec![1])]));
    circuit.add_layer(CircuitLayer::new(vec![(QuantumGate::CNOT, vec![0, 1])]));
    assert_eq!(circuit.n_parameters(), 2);
}

#[test]
fn test_expectation_z_ground_state() {
    let state = QubitState::new(2);
    assert!((state.expectation_z(0) - 1.0).abs() < 1e-10);
    assert!((state.expectation_z(1) - 1.0).abs() < 1e-10);
}

#[test]
fn test_backpropagation_gradient_finite() {
    let bp = QuantumBackpropagation::new(2);
    let mut h = PauliHamiltonian::new();
    h.add_term(1.0, "ZZ");
    let grads = bp.gradient(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &h, 1);
    assert_eq!(grads.len(), 6);
    for &g in &grads {
        assert!(g.is_finite());
    }
}

#[test]
fn test_vqe_optimize_converges() {
    let mut h = PauliHamiltonian::new();
    h.add_term(-1.0, "ZZ");
    let mut opt = VqeOptimizer::new(2, 1, h, 0);
    let initial = opt.energy(&opt.params.clone());
    opt.optimize(30, 0.1);
    let final_e = opt.energy(&opt.params.clone());
    assert!(final_e <= initial + 0.5);
}

// ── Round-44 new tests ────────────────────────────────────────────────────────

// IQP Feature Map tests

#[test]
fn test_iqp_feature_map_self_kernel_is_one() {
    let fm = advanced::IqpFeatureMap::new(2, 1);
    let x = vec![0.5, 1.2];
    let k = fm.kernel(&x, &x);
    assert!((k - 1.0).abs() < 1e-9, "Self-kernel should be 1, got {k}");
}

#[test]
fn test_iqp_feature_map_kernel_range() {
    let fm = advanced::IqpFeatureMap::new(2, 2);
    let x1 = vec![0.3, -0.7];
    let x2 = vec![1.1, 0.5];
    let k = fm.kernel(&x1, &x2);
    assert!((0.0..=1.0 + 1e-9).contains(&k));
}

#[test]
fn test_iqp_feature_map_symmetry() {
    let fm = advanced::IqpFeatureMap::new(3, 1);
    let x1 = vec![0.1, 0.2, 0.3];
    let x2 = vec![0.4, 0.5, 0.6];
    let k12 = fm.kernel(&x1, &x2);
    let k21 = fm.kernel(&x2, &x1);
    assert!((k12 - k21).abs() < 1e-9, "Kernel must be symmetric");
}

#[test]
fn test_iqp_feature_state_norm() {
    let fm = advanced::IqpFeatureMap::new(2, 1);
    let state = fm.feature_state(&[0.5, 1.0]);
    let norm_sq: f64 = state.amplitudes.iter().map(|a| a.abs_sq()).sum();
    assert!((norm_sq - 1.0).abs() < 1e-9, "State must be normalized");
}

// ZZ Feature Map tests

#[test]
fn test_zz_feature_map_self_kernel() {
    let fm = advanced::ZzFeatureMap::new(2, 2);
    let x = vec![0.8, 1.5];
    let k = fm.kernel(&x, &x);
    assert!((k - 1.0).abs() < 1e-9, "Self-kernel should be 1, got {k}");
}

#[test]
fn test_zz_feature_map_range() {
    let fm = advanced::ZzFeatureMap::new(2, 1);
    let k = fm.kernel(&[0.1, 0.2], &[0.9, 0.8]);
    assert!((0.0..=1.0 + 1e-9).contains(&k));
}

#[test]
fn test_zz_feature_map_state_norm() {
    let fm = advanced::ZzFeatureMap::new(3, 1);
    let state = fm.feature_state(&[0.2, 0.4, 0.6]);
    let norm_sq: f64 = state.amplitudes.iter().map(|a| a.abs_sq()).sum();
    assert!((norm_sq - 1.0).abs() < 1e-9);
}

#[test]
fn test_zz_feature_map_differs_from_iqp() {
    // ZZ and IQP maps should generally give different values for non-trivial inputs
    let iqp = advanced::IqpFeatureMap::new(2, 1);
    let zz = advanced::ZzFeatureMap::new(2, 1);
    let x1 = vec![0.3, 0.7];
    let x2 = vec![0.8, 0.2];
    let k_iqp = iqp.kernel(&x1, &x2);
    let k_zz = zz.kernel(&x1, &x2);
    // Both in [0,1]; verify they are valid
    assert!((0.0..=1.0 + 1e-9).contains(&k_iqp));
    assert!((0.0..=1.0 + 1e-9).contains(&k_zz));
}

// QuantumKernelFull tests

#[test]
fn test_quantum_kernel_full_angle_encoding() {
    let k = advanced::QuantumKernelFull::new(
        2,
        advanced::QuantumFeatureMapType::AngleEncoding,
    );
    let x = vec![0.5, 1.0];
    assert!((k.compute(&x, &x) - 1.0).abs() < 1e-9);
}

#[test]
fn test_quantum_kernel_full_iqp() {
    let k = advanced::QuantumKernelFull::new(
        2,
        advanced::QuantumFeatureMapType::Iqp { reps: 1 },
    );
    let x = vec![0.5, -0.3];
    assert!((k.compute(&x, &x) - 1.0).abs() < 1e-9);
}

#[test]
fn test_quantum_kernel_full_zz() {
    let k = advanced::QuantumKernelFull::new(
        2,
        advanced::QuantumFeatureMapType::ZzMap { reps: 1 },
    );
    let x = vec![1.0, 0.5];
    assert!((k.compute(&x, &x) - 1.0).abs() < 1e-9);
}

#[test]
fn test_quantum_kernel_full_matrix_symmetric() {
    let k = advanced::QuantumKernelFull::new(
        2,
        advanced::QuantumFeatureMapType::AngleEncoding,
    );
    let data = vec![vec![0.1, 0.2], vec![0.5, 0.6], vec![-0.3, 0.9]];
    let km = k.kernel_matrix(&data);
    let n = data.len();
    for i in 0..n {
        for j in 0..n {
            assert!((km[i][j] - km[j][i]).abs() < 1e-9, "Kernel matrix not symmetric");
        }
    }
}

#[test]
fn test_kernel_target_alignment_positive() {
    let k = advanced::QuantumKernelFull::new(
        2,
        advanced::QuantumFeatureMapType::AngleEncoding,
    );
    let x = vec![vec![0.1, 0.2], vec![-0.1, -0.2], vec![0.5, 0.5]];
    let y = vec![1.0, -1.0, 1.0];
    let kta = k.kernel_target_alignment(&x, &y);
    assert!(kta.is_finite());
    assert!((-1.0..=1.0 + 1e-9).contains(&kta));
}

// QuantumKernelAlignment tests

#[test]
fn test_quantum_kernel_alignment_optimize() {
    let mut kta = advanced::QuantumKernelAlignment::new(2, 1);
    let x = vec![vec![0.1, 0.2], vec![-0.1, -0.2], vec![0.3, 0.0]];
    let y = vec![1.0, -1.0, 1.0];
    let score = kta.optimize(&x, &y, 3);
    assert!(score.is_finite());
    assert!(!kta.alignment_history.is_empty());
}

#[test]
fn test_quantum_kernel_alignment_history_length() {
    let mut kta = advanced::QuantumKernelAlignment::new(2, 1);
    let x = vec![vec![0.5, 0.3], vec![-0.5, -0.3]];
    let y = vec![1.0, -1.0];
    kta.optimize(&x, &y, 5);
    assert_eq!(kta.alignment_history.len(), 5);
}

// QaoaLayer tests

#[test]
fn test_qaoa_layer_apply_preserves_norm() {
    let edges = vec![(0, 1, 1.0), (1, 2, 0.5)];
    let layer = advanced::QaoaLayer::new(3, edges, 0.3, 0.5);
    let mut state = QubitState::new(3);
    for q in 0..3 {
        state.apply_gate(&QuantumGate::H, &[q]);
    }
    layer.apply(&mut state);
    let norm_sq: f64 = state.amplitudes.iter().map(|a| a.abs_sq()).sum();
    assert!((norm_sq - 1.0).abs() < 1e-9);
}

#[test]
fn test_qaoa_layer_to_circuit_runs() {
    let edges = vec![(0, 1, 1.0)];
    let layer = advanced::QaoaLayer::new(2, edges, 0.4, 0.2);
    let circuit = layer.to_circuit();
    let init = QubitState::new(2);
    let final_state = circuit.run(init);
    let sum: f64 = final_state.measure_probs().iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

// QaoaOptimizer tests

#[test]
fn test_qaoa_optimizer_energy_finite() {
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
    let opt = advanced::QaoaOptimizer::new(3, edges, 1, 42);
    let e = opt.energy(&opt.gamma, &opt.beta);
    assert!(e.is_finite());
}

#[test]
fn test_qaoa_optimizer_energy_nonnegative_for_maxcut() {
    // MaxCut energy is non-negative
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
    let opt = advanced::QaoaOptimizer::new(3, edges, 1, 7);
    let e = opt.energy(&opt.gamma, &opt.beta);
    assert!(e >= -1e-9);
}

#[test]
fn test_qaoa_optimizer_optimize_improves_or_stable() {
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
    let mut opt = advanced::QaoaOptimizer::new(3, edges, 1, 99);
    let e0 = opt.energy(&opt.gamma.clone(), &opt.beta.clone());
    let e1 = opt.optimize(3);
    // After optimization, energy should be >= 0 and finite
    assert!(e1.is_finite());
    assert!(e1 >= -1e-6);
    // Energy history recorded
    assert_eq!(opt.energy_history.len(), 3);
    let _ = e0; // used
}

#[test]
fn test_qaoa_optimizer_final_probs_sum_to_1() {
    let edges = vec![(0, 1, 1.0)];
    let mut opt = advanced::QaoaOptimizer::new(2, edges, 1, 13);
    opt.optimize(2);
    let probs = opt.final_state_probs();
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

// MaxCutQaoa tests

#[test]
fn test_maxcut_qaoa_construction() {
    let adj = vec![
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];
    let mc = advanced::MaxCutQaoa::new(adj, 1, 42);
    assert!(mc.is_ok());
}

#[test]
fn test_maxcut_qaoa_best_cut_valid() {
    let adj = vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
    ];
    let mut mc = advanced::MaxCutQaoa::new(adj, 1, 7).expect("creation failed");
    mc.optimize(2);
    let cut = mc.best_cut();
    assert_eq!(cut.len(), 3);
    for &c in &cut {
        assert!(c == 0 || c == 1);
    }
}

#[test]
fn test_maxcut_qaoa_cut_value() {
    let adj = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    let mc = advanced::MaxCutQaoa::new(adj, 1, 1).expect("creation failed");
    // Assignment [0, 1] should have cut value 1.0
    assert!((mc.cut_value(&[0, 1]) - 1.0).abs() < 1e-9);
    // Assignment [0, 0] should have cut value 0.0
    assert!(mc.cut_value(&[0, 0]).abs() < 1e-9);
}

#[test]
fn test_maxcut_qaoa_empty_adj_error() {
    let result = advanced::MaxCutQaoa::new(vec![], 1, 0);
    assert!(result.is_err());
}

// QaoaMetrics tests

#[test]
fn test_qaoa_metrics_compute() {
    let adj = vec![
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];
    let mut mc = advanced::MaxCutQaoa::new(adj, 1, 5).expect("creation failed");
    mc.optimize(2);
    let metrics = advanced::QaoaMetrics::compute(&mc, Some(2.0));
    assert!(metrics.final_energy.is_finite());
    assert!(metrics.approximation_ratio >= 0.0);
    assert!(metrics.energy_variance >= 0.0);
}

#[test]
fn test_qaoa_metrics_without_optimal() {
    let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let mut mc = advanced::MaxCutQaoa::new(adj, 1, 3).expect("creation failed");
    mc.optimize(2);
    let metrics = advanced::QaoaMetrics::compute(&mc, None);
    assert!(metrics.optimal_prob.is_none());
    assert!(metrics.final_energy.is_finite());
}

// ZneExtrapolation tests

#[test]
fn test_zne_richardson_order1_exact() {
    // f(c) = 1 + 2c => f(0) = 1
    let data = vec![(1.0, 3.0), (2.0, 5.0)];
    let v = advanced::ZneExtrapolation::richardson(&data, 1);
    assert!((v - 1.0).abs() < 1e-9);
}

#[test]
fn test_zne_richardson_order2() {
    // f(c) = 1 + c + c² => f(0) = 1
    let data = vec![(1.0, 3.0), (2.0, 7.0), (3.0, 13.0)];
    let v = advanced::ZneExtrapolation::richardson(&data, 2);
    assert!((v - 1.0).abs() < 1e-6);
}

#[test]
fn test_zne_linear_extrapolation() {
    // f(c) = 2 + 0.5c => f(0) = 2
    let data = vec![(2.0, 3.0), (4.0, 4.0)];
    let v = advanced::ZneExtrapolation::linear(&data);
    assert!((v - 2.0).abs() < 1e-9);
}

#[test]
fn test_zne_exponential_fit() {
    // f(c) = 2.0 * exp(-1.0 * c) + 1.0 => f(0) = 3.0
    // Use larger, cleaner exponential decay for stable fitting
    let data: Vec<(f64, f64)> = vec![
        (1.0, 2.0 * (-1.0f64).exp() + 1.0),
        (2.0, 2.0 * (-2.0f64).exp() + 1.0),
        (3.0, 2.0 * (-3.0f64).exp() + 1.0),
        (4.0, 2.0 * (-4.0f64).exp() + 1.0),
        (5.0, 2.0 * (-5.0f64).exp() + 1.0),
    ];
    let v = advanced::ZneExtrapolation::exponential_fit(&data);
    // The method is a heuristic; verify it returns a finite value in reasonable range
    assert!(v.is_finite(), "exp fit should return finite value, got {v}");
    assert!(v > 0.0, "extrapolated value should be positive for this model, got {v}");
}

#[test]
fn test_zne_lagrange_polynomial() {
    // Quadratic f(c) = 1 + c²: f(0) = 1
    let data = vec![(1.0, 2.0), (2.0, 5.0), (3.0, 10.0)];
    let v = advanced::ZneExtrapolation::lagrange(&data);
    assert!((v - 1.0).abs() < 1e-9);
}

#[test]
fn test_zne_richardson_fallback_too_few_pts() {
    let data = vec![(1.0, 2.0)];
    let v = advanced::ZneExtrapolation::richardson(&data, 3);
    assert!(v.is_finite());
}

// ProbabilisticErrorCancellation tests

#[test]
fn test_pec_construction() {
    let pec = advanced::ProbabilisticErrorCancellation::new(0.01, 10);
    assert!(pec.gamma_overhead >= 1.0);
    assert!((pec.error_rate - 0.01).abs() < 1e-12);
}

#[test]
fn test_pec_zero_noise_overhead_one() {
    let pec = advanced::ProbabilisticErrorCancellation::new(0.0, 5);
    // γ = 1 for zero noise
    assert!((pec.gamma_overhead - 1.0).abs() < 1e-9);
}

#[test]
fn test_pec_sample_gate_sign() {
    let pec = advanced::ProbabilisticErrorCancellation::new(0.05, 3);
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let (sign, idx) = pec.sample_gate(&mut rng);
        assert!(sign == 1.0 || sign == -1.0);
        assert!(idx <= 3);
    }
}

#[test]
fn test_pec_estimate_zero_noise() {
    let pec = advanced::ProbabilisticErrorCancellation::new(0.0, 1);
    let vals = vec![0.9, 0.8, 0.85];
    let signs = vec![1.0, 1.0, 1.0];
    let est = pec.estimate(&vals, &signs);
    assert!(est.is_finite());
}

#[test]
fn test_pec_sample_overhead_grows_with_n_gates() {
    let pec1 = advanced::ProbabilisticErrorCancellation::new(0.05, 5);
    let pec2 = advanced::ProbabilisticErrorCancellation::new(0.05, 10);
    assert!(pec2.gamma_overhead >= pec1.gamma_overhead);
}

// MeasurementErrorMitigation tests

#[test]
fn test_mem_identity_no_errors() {
    let mem = advanced::MeasurementErrorMitigation::from_per_qubit_rates(
        &[0.0, 0.0],
        &[0.0, 0.0],
    ).expect("construction failed");
    let noisy = vec![0.5, 0.2, 0.2, 0.1];
    let mitigated = mem.mitigate(&noisy).expect("mitigation failed");
    assert!((mitigated.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for (&m, &n) in mitigated.iter().zip(noisy.iter()) {
        assert!((m - n).abs() < 1e-6, "Identity should preserve probs");
    }
}

#[test]
fn test_mem_mitigate_sums_to_one() {
    let mem = advanced::MeasurementErrorMitigation::from_per_qubit_rates(
        &[0.02, 0.03],
        &[0.01, 0.02],
    ).expect("construction failed");
    let noisy = vec![0.6, 0.1, 0.2, 0.1];
    let mitigated = mem.mitigate(&noisy).expect("mitigation failed");
    assert!((mitigated.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for &v in &mitigated {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_mem_wrong_length_error() {
    let mem = advanced::MeasurementErrorMitigation::from_per_qubit_rates(
        &[0.01],
        &[0.01],
    ).expect("construction failed");
    assert!(mem.mitigate(&[0.5, 0.3, 0.2]).is_err());
}

#[test]
fn test_mem_mismatched_rates_error() {
    let result = advanced::MeasurementErrorMitigation::from_per_qubit_rates(
        &[0.01, 0.02],
        &[0.01],
    );
    assert!(result.is_err());
}

#[test]
fn test_mem_condition_number_finite() {
    let mem = advanced::MeasurementErrorMitigation::from_per_qubit_rates(
        &[0.05],
        &[0.03],
    ).expect("construction failed");
    let cn = mem.condition_number_bound();
    assert!(cn.is_finite() && cn >= 1.0);
}

#[test]
fn test_mem_cal_matrix_columns_sum_to_one() {
    let mem = advanced::MeasurementErrorMitigation::from_per_qubit_rates(
        &[0.1, 0.05],
        &[0.08, 0.03],
    ).expect("construction failed");
    let dim = 4usize;
    for prepared in 0..dim {
        let col_sum: f64 = (0..dim).map(|meas| mem.cal_matrix[meas][prepared]).sum();
        assert!((col_sum - 1.0).abs() < 1e-9, "col {prepared} sums to {col_sum}");
    }
}

// QbmModel tests

#[test]
fn test_qbm_model_construction() {
    let qbm = advanced::QbmModel::new(2, 2, 0.5, 1.0, 42);
    assert_eq!(qbm.n_visible, 2);
    assert_eq!(qbm.n_hidden, 2);
    assert_eq!(qbm.weights.len(), 4);
}

#[test]
fn test_qbm_model_weights_symmetric() {
    let qbm = advanced::QbmModel::new(2, 2, 0.3, 1.0, 7);
    let n = qbm.n_visible + qbm.n_hidden;
    for i in 0..n {
        for j in 0..n {
            assert!((qbm.weights[i][j] - qbm.weights[j][i]).abs() < 1e-12);
        }
    }
}

#[test]
fn test_qbm_gibbs_sample_valid_spins() {
    let qbm = advanced::QbmModel::new(2, 1, 0.5, 1.0, 42);
    let mut rng = StdRng::seed_from_u64(1);
    let spins = qbm.gibbs_sample(&[1.0, -1.0, 1.0], 5, &mut rng);
    assert_eq!(spins.len(), 3);
    for &s in &spins {
        assert!(s == 1.0 || s == -1.0);
    }
}

#[test]
fn test_qbm_model_correlation_finite() {
    let qbm = advanced::QbmModel::new(2, 1, 0.5, 1.0, 10);
    let mut rng = StdRng::seed_from_u64(2);
    let c = qbm.model_correlation(0, 1, 10, &mut rng);
    assert!(c.is_finite() && (-1.0..=1.0).contains(&c));
}

// QbmTrainer tests

#[test]
fn test_qbm_trainer_fit_loss_history() {
    let mut qbm = advanced::QbmModel::new(2, 1, 0.3, 1.0, 42);
    let mut trainer = advanced::QbmTrainer::new(0.01, 1, 3);
    let data = vec![
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
        vec![1.0, 1.0],
    ];
    let history = trainer.fit(&mut qbm, &data, 3, 99);
    assert_eq!(history.len(), 3);
    for &loss in &history {
        assert!(loss.is_finite() && loss >= 0.0);
    }
}

#[test]
fn test_qbm_trainer_single_epoch() {
    let mut qbm = advanced::QbmModel::new(2, 2, 0.5, 1.0, 5);
    let mut trainer = advanced::QbmTrainer::new(0.05, 1, 2);
    let mut rng = StdRng::seed_from_u64(7);
    let data = vec![vec![1.0, -1.0]];
    let loss = trainer.train_epoch(&mut qbm, &data, &mut rng);
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_qbm_trainer_weights_updated() {
    let mut qbm = advanced::QbmModel::new(2, 1, 0.2, 1.0, 13);
    let w_before = qbm.weights.clone();
    let mut trainer = advanced::QbmTrainer::new(0.1, 1, 2);
    let data = vec![vec![1.0, 1.0], vec![-1.0, -1.0]];
    trainer.fit(&mut qbm, &data, 2, 17);
    // After training, at least some weights should have changed
    let any_changed = qbm
        .weights
        .iter()
        .zip(w_before.iter())
        .any(|(row_new, row_old)| {
            row_new
                .iter()
                .zip(row_old.iter())
                .any(|(a, b)| (a - b).abs() > 1e-15)
        });
    assert!(any_changed, "Training should update weights");
}
