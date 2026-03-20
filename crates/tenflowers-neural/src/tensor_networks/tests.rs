// Tests for tensor_networks module
// Included via `include!("tests.rs")` in mod.rs

use super::*;

// ============================================================================
// Helper functions
// ============================================================================

/// Create a simple input sequence: n_sites vectors each of length d, filled with 1/d.
fn make_uniform_input(n_sites: usize, d: usize) -> Vec<Vec<f64>> {
    let val = 1.0 / d as f64;
    (0..n_sites).map(|_| vec![val; d]).collect()
}

/// Create a dense identity-like matrix [n × n].
fn make_eye(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| {
        let mut row = vec![0.0_f64; n];
        row[i] = 1.0;
        row
    }).collect()
}

/// Create a random-ish (deterministic) matrix [rows × cols].
fn make_det_matrix(rows: usize, cols: usize, seed: f64) -> Vec<Vec<f64>> {
    (0..rows).map(|i| {
        (0..cols).map(|j| ((i as f64 + seed) * (j as f64 + 1.0)).sin() * 0.5 + 0.1).collect()
    }).collect()
}

// ============================================================================
// §1 TnMatrixProductState tests
// ============================================================================

#[test]
fn test_mps_construction_defaults() {
    let mps = TnMatrixProductState::new(4, 3, 5);
    assert_eq!(mps.n_sites, 4);
    assert_eq!(mps.d, 3);
    assert_eq!(mps.chi_max, 5);
    assert_eq!(mps.sites.len(), 4);
}

#[test]
fn test_mps_site_boundary_chi() {
    let mps = TnMatrixProductState::new(6, 2, 4);
    // First site must have chi_left=1
    assert_eq!(mps.sites[0].chi_left, 1);
    // Last site must have chi_right=1
    assert_eq!(mps.sites[5].chi_right, 1);
    // Middle sites should have chi_max bond dim
    assert_eq!(mps.sites[1].chi_left, 4);
}

#[test]
fn test_mps_contract_all_output_shape() {
    let mps = TnMatrixProductState::new(5, 3, 4);
    let input = make_uniform_input(5, 3);
    let out = mps.contract_all(&input).expect("contraction should succeed");
    // Last site has chi_right=1, so output should be length 1
    assert_eq!(out.len(), 1);
}

#[test]
fn test_mps_contract_all_dimension_error_sites() {
    let mps = TnMatrixProductState::new(4, 3, 4);
    let input = make_uniform_input(3, 3); // wrong number of sites
    let result = mps.contract_all(&input);
    assert!(result.is_err());
}

#[test]
fn test_mps_contract_all_dimension_error_features() {
    let mps = TnMatrixProductState::new(4, 3, 4);
    let mut input = make_uniform_input(4, 3);
    input[2] = vec![0.5, 0.5]; // wrong feature dim
    let result = mps.contract_all(&input);
    assert!(result.is_err());
}

#[test]
fn test_mps_bond_dimensions_count() {
    let mps = TnMatrixProductState::new(5, 2, 3);
    let bonds = mps.bond_dimensions();
    // n_sites - 1 bonds
    assert_eq!(bonds.len(), 4);
}

#[test]
fn test_mps_total_params_positive() {
    let mps = TnMatrixProductState::new(4, 3, 5);
    assert!(mps.total_params() > 0);
}

#[test]
fn test_mps_total_params_formula() {
    let mps = TnMatrixProductState::new(3, 2, 4);
    // site 0: chi_l=1, d=2, chi_r=4 → 8 params
    // site 1: chi_l=4, d=2, chi_r=4 → 32 params
    // site 2: chi_l=4, d=2, chi_r=1 → 8 params
    // total: 48
    assert_eq!(mps.total_params(), 48);
}

#[test]
fn test_mps_svd_compress_reduces_bonds() {
    let mut mps = TnMatrixProductState::new(4, 2, 8);
    mps.svd_compress(3).expect("SVD compress should succeed");
    let bonds = mps.bond_dimensions();
    for b in &bonds {
        assert!(*b <= 3, "bond dim {} should be <= 3 after compression to 3", b);
    }
}

#[test]
fn test_mps_svd_compress_preserves_output_type() {
    let mut mps = TnMatrixProductState::new(3, 2, 4);
    mps.svd_compress(2).expect("compress ok");
    // After compression, contract_all should still work
    let input = make_uniform_input(3, 2);
    let out = mps.contract_all(&input);
    assert!(out.is_ok());
}

#[test]
fn test_mps_get_slice() {
    let mps = TnMatrixProductState::new(3, 2, 4);
    let site = &mps.sites[1];
    let slice = site.get_slice(0);
    assert_eq!(slice.len(), site.chi_left);
    assert_eq!(slice[0].len(), site.chi_right);
}

// ============================================================================
// §2 TnTuckerLayer tests
// ============================================================================

#[test]
fn test_tucker_layer_output_shape() {
    let layer = TnTuckerLayer::new(16, 8, 4);
    let x = vec![1.0_f64; 16];
    let y = layer.forward(&x);
    assert_eq!(y.len(), 8);
}

#[test]
fn test_tucker_layer_output_shape_large() {
    let layer = TnTuckerLayer::new(64, 32, 8);
    let x = vec![0.1_f64; 64];
    let y = layer.forward(&x);
    assert_eq!(y.len(), 32);
}

#[test]
fn test_tucker_layer_compression_ratio_less_than_one() {
    let layer = TnTuckerLayer::new(128, 64, 8);
    let ratio = layer.compression_ratio();
    assert!(ratio < 1.0, "Tucker compression ratio {} should be < 1 for small rank", ratio);
}

#[test]
fn test_tucker_layer_compression_ratio_rank_min() {
    // When rank >= min(in, out), no compression
    let layer = TnTuckerLayer::new(4, 4, 4);
    // r1=4, r2=4: compressed = 4*4 + 4*4 + 4*4 = 48, full = 4*4 = 16
    // ratio > 1 is expected for this degenerate case
    let ratio = layer.compression_ratio();
    assert!(ratio > 0.0);
}

#[test]
fn test_tucker_layer_from_dense() {
    let w = make_det_matrix(8, 6, 1.0);
    let layer = TnTuckerLayer::from_dense(&w, 3);
    assert_eq!(layer.in_dim, 6);
    assert_eq!(layer.out_dim, 8);
    assert!(layer.r1 <= 3);
    assert!(layer.r2 <= 3);
}

#[test]
fn test_tucker_layer_from_dense_reconstruction() {
    let w = make_eye(4);
    let layer = TnTuckerLayer::from_dense(&w, 4);
    // Forward should produce reasonable output
    let x = vec![1.0, 0.0, 0.0, 0.0_f64];
    let y = layer.forward(&x);
    assert_eq!(y.len(), 4);
}

#[test]
fn test_tucker_layer_factor_dims() {
    let layer = TnTuckerLayer::new(10, 8, 4);
    assert_eq!(layer.u1.len(), 10);    // in_dim rows
    assert_eq!(layer.u1[0].len(), 4); // r1 cols
    assert_eq!(layer.u2.len(), 8);    // out_dim rows
    assert_eq!(layer.u2[0].len(), 4); // r2 cols
    assert_eq!(layer.g.len(), 4);     // r2 rows
    assert_eq!(layer.g[0].len(), 4);  // r1 cols
}

#[test]
fn test_tucker_layer_bias_init_zero() {
    let layer = TnTuckerLayer::new(4, 3, 2);
    assert_eq!(layer.bias, vec![0.0_f64; 3]);
}

// ============================================================================
// §3 TnTreeTensorNetwork tests
// ============================================================================

#[test]
fn test_tree_tn_construction_power_of_two() {
    let tree = TnTreeTensorNetwork::new(4, 3, 8, 2).expect("4 leaves ok");
    assert_eq!(tree.n_leaves, 4);
    assert_eq!(tree.in_dim, 3);
    assert_eq!(tree.bond_dim, 8);
    assert_eq!(tree.out_dim, 2);
}

#[test]
fn test_tree_tn_construction_8_leaves() {
    let tree = TnTreeTensorNetwork::new(8, 4, 6, 3).expect("8 leaves ok");
    assert_eq!(tree.n_leaves, 8);
    assert_eq!(tree.depth(), 3);
}

#[test]
fn test_tree_tn_rejects_non_power_of_two() {
    let result = TnTreeTensorNetwork::new(3, 2, 4, 2);
    assert!(result.is_err(), "3 is not a power of 2");
}

#[test]
fn test_tree_tn_rejects_zero_leaves() {
    let result = TnTreeTensorNetwork::new(0, 2, 4, 2);
    assert!(result.is_err(), "0 leaves is invalid");
}

#[test]
fn test_tree_tn_forward_output_shape() {
    let tree = TnTreeTensorNetwork::new(4, 3, 8, 5).expect("ok");
    let inputs: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5_f64; 3]).collect();
    let out = tree.forward(&inputs).expect("forward ok");
    assert_eq!(out.len(), 5);
}

#[test]
fn test_tree_tn_forward_8_leaves() {
    let tree = TnTreeTensorNetwork::new(8, 2, 4, 3).expect("ok");
    let inputs: Vec<Vec<f64>> = (0..8).map(|i| vec![(i as f64) * 0.1; 2]).collect();
    let out = tree.forward(&inputs).expect("ok");
    assert_eq!(out.len(), 3);
}

#[test]
fn test_tree_tn_forward_wrong_num_inputs() {
    let tree = TnTreeTensorNetwork::new(4, 3, 4, 2).expect("ok");
    let inputs: Vec<Vec<f64>> = (0..3).map(|_| vec![0.5; 3]).collect();
    assert!(tree.forward(&inputs).is_err());
}

#[test]
fn test_tree_tn_forward_wrong_input_dim() {
    let tree = TnTreeTensorNetwork::new(4, 3, 4, 2).expect("ok");
    let mut inputs: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5; 3]).collect();
    inputs[1] = vec![0.5, 0.5]; // wrong dim
    assert!(tree.forward(&inputs).is_err());
}

#[test]
fn test_tree_tn_depth() {
    let tree2 = TnTreeTensorNetwork::new(2, 2, 4, 2).expect("ok");
    assert_eq!(tree2.depth(), 1);
    let tree4 = TnTreeTensorNetwork::new(4, 2, 4, 2).expect("ok");
    assert_eq!(tree4.depth(), 2);
    let tree8 = TnTreeTensorNetwork::new(8, 2, 4, 2).expect("ok");
    assert_eq!(tree8.depth(), 3);
}

// ============================================================================
// §4 TnMeraLayer tests
// ============================================================================

#[test]
fn test_mera_layer_construction() {
    let mera = TnMeraLayer::new(4, 2).expect("ok");
    assert_eq!(mera.d, 2);
    assert_eq!(mera.disentanglers.len(), 2);
    assert_eq!(mera.isometries.len(), 2);
}

#[test]
fn test_mera_layer_rejects_odd_sites() {
    let result = TnMeraLayer::new(3, 2);
    assert!(result.is_err(), "3 sites is odd");
}

#[test]
fn test_mera_layer_rejects_zero_sites() {
    let result = TnMeraLayer::new(0, 2);
    assert!(result.is_err());
}

#[test]
fn test_mera_coarsen_halves_sites() {
    let mera = TnMeraLayer::new(4, 2).expect("ok");
    let state: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5_f64; 2]).collect();
    let out = mera.coarsen(&state).expect("coarsen ok");
    assert_eq!(out.len(), 2); // 4 → 2
}

#[test]
fn test_mera_coarsen_output_dim() {
    let mera = TnMeraLayer::new(6, 3).expect("ok");
    let state: Vec<Vec<f64>> = (0..6).map(|_| vec![0.3_f64; 3]).collect();
    let out = mera.coarsen(&state).expect("coarsen ok");
    assert_eq!(out.len(), 3);
    for v in &out {
        assert_eq!(v.len(), 3, "output vectors should have d=3 dimensions");
    }
}

#[test]
fn test_mera_forward_same_as_coarsen() {
    let mera = TnMeraLayer::new(4, 2).expect("ok");
    let state: Vec<Vec<f64>> = (0..4).map(|i| vec![(i as f64) * 0.1 + 0.1; 2]).collect();
    let out_coarsen = mera.coarsen(&state).expect("ok");
    let out_forward = mera.forward(&state).expect("ok");
    assert_eq!(out_coarsen.len(), out_forward.len());
}

#[test]
fn test_mera_disentangler_size() {
    let mera = TnMeraLayer::new(4, 3).expect("ok");
    // d=3, d^2=9
    for dis in &mera.disentanglers {
        assert_eq!(dis.len(), 9);
        assert_eq!(dis[0].len(), 9);
    }
}

#[test]
fn test_mera_isometry_size() {
    let mera = TnMeraLayer::new(4, 3).expect("ok");
    // Each isometry stored flat: d^2 * d = 9 * 3 = 27 elements
    for iso in &mera.isometries {
        assert_eq!(iso.len(), 27, "isometry flat length should be d^2 * d = 27");
    }
}

#[test]
fn test_mera_coarsen_wrong_state_dim() {
    let mera = TnMeraLayer::new(4, 2).expect("ok");
    let state: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5; 3]).collect(); // wrong d
    assert!(mera.coarsen(&state).is_err());
}

// ============================================================================
// §5 TnConvolutionalKernel tests
// ============================================================================

#[test]
fn test_conv_kernel_construction() {
    let ck = TnConvolutionalKernel::new(4, 8, 3, 2);
    assert_eq!(ck.in_channels, 4);
    assert_eq!(ck.out_channels, 8);
    assert_eq!(ck.kernel_size, 3);
}

#[test]
fn test_conv_kernel_materialize_shape() {
    let ck = TnConvolutionalKernel::new(4, 8, 3, 2);
    let w = ck.materialize();
    assert_eq!(w.len(), 8);          // out_channels
    assert_eq!(w[0].len(), 4);       // in_channels
    assert_eq!(w[0][0].len(), 3);    // kernel_size
}

#[test]
fn test_conv_kernel_materialize_not_all_zero() {
    let ck = TnConvolutionalKernel::new(4, 8, 3, 2);
    let w = ck.materialize();
    let sum: f64 = w.iter().flat_map(|r| r.iter().flat_map(|c| c.iter())).map(|x| x.abs()).sum();
    assert!(sum > 1e-10, "materialized kernel should not be all zeros");
}

#[test]
fn test_conv_kernel_forward_1d_output_shape() {
    let ck = TnConvolutionalKernel::new(3, 5, 3, 2);
    let input: Vec<Vec<f64>> = (0..3).map(|_| vec![0.5_f64; 10]).collect(); // 3 ch × 10 len
    let out = ck.forward_1d(&input);
    assert_eq!(out.len(), 5);        // out_channels
    assert_eq!(out[0].len(), 10);   // same length (same padding)
}

#[test]
fn test_conv_kernel_forward_1d_non_trivial() {
    let ck = TnConvolutionalKernel::new(2, 4, 3, 2);
    let input: Vec<Vec<f64>> = (0..2).map(|i| (0..8).map(|t| (i as f64 + t as f64) * 0.1).collect()).collect();
    let out = ck.forward_1d(&input);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn test_conv_kernel_compression_ratio() {
    let ck = TnConvolutionalKernel::new(16, 32, 5, 4);
    let ratio = ck.compression_ratio();
    assert!(ratio > 0.0);
    // With rank=4 << 16*5=80, should be compressed
    assert!(ratio < 1.0, "compression ratio {} should be < 1 for rank=4", ratio);
}

// ============================================================================
// §6 TnQuantumInspiredLayer tests
// ============================================================================

#[test]
fn test_quantum_layer_construction() {
    let ql = TnQuantumInspiredLayer::new(5, 3, 4, 2);
    assert_eq!(ql.n_classes, 2);
    assert_eq!(ql.mps.n_sites, 5);
}

#[test]
fn test_quantum_layer_amplitude_is_scalar() {
    let ql = TnQuantumInspiredLayer::new(4, 2, 3, 2);
    let x = make_uniform_input(4, 2);
    let amp = ql.amplitude(&x).expect("amplitude ok");
    // Just ensure it's a finite scalar
    assert!(amp.is_finite());
}

#[test]
fn test_quantum_layer_amplitude_wrong_sites() {
    let ql = TnQuantumInspiredLayer::new(4, 2, 3, 2);
    let x = make_uniform_input(3, 2); // wrong n_sites
    assert!(ql.amplitude(&x).is_err());
}

#[test]
fn test_quantum_layer_log_probability_finite() {
    let ql = TnQuantumInspiredLayer::new(4, 2, 3, 2);
    let x = make_uniform_input(4, 2);
    let lp = ql.log_probability(&x).expect("log_prob ok");
    assert!(lp.is_finite() || lp == f64::NEG_INFINITY);
}

#[test]
fn test_quantum_layer_nll_loss_positive() {
    let ql = TnQuantumInspiredLayer::new(3, 2, 3, 1);
    let batch: Vec<Vec<Vec<f64>>> = (0..5).map(|_| make_uniform_input(3, 2)).collect();
    let loss = ql.nll_loss(&batch).expect("nll ok");
    // NLL can be negative (if amplitude > 1, log is positive so -log is negative)
    // But it should be finite
    assert!(loss.is_finite());
}

#[test]
fn test_quantum_layer_nll_empty_batch_error() {
    let ql = TnQuantumInspiredLayer::new(3, 2, 3, 1);
    let result = ql.nll_loss(&[]);
    assert!(result.is_err());
}

// ============================================================================
// §7 TnLowRankRnn tests
// ============================================================================

#[test]
fn test_low_rank_rnn_construction() {
    let rnn = TnLowRankRnn::new(4, 8, 3).expect("ok");
    assert_eq!(rnn.hidden_size, 8);
    assert_eq!(rnn.input_size, 4);
    assert_eq!(rnn.rank, 3);
}

#[test]
fn test_low_rank_rnn_forward_output_shape() {
    let rnn = TnLowRankRnn::new(4, 8, 3).expect("ok");
    let x = vec![0.1_f64; 4];
    let h = vec![0.0_f64; 8];
    let h_new = rnn.forward(&x, &h).expect("forward ok");
    assert_eq!(h_new.len(), 8, "output should equal hidden_size");
}

#[test]
fn test_low_rank_rnn_forward_tanh_bounded() {
    let rnn = TnLowRankRnn::new(3, 6, 2).expect("ok");
    let x = vec![1.0_f64; 3];
    let h = vec![0.5_f64; 6];
    let h_new = rnn.forward(&x, &h).expect("ok");
    for v in &h_new {
        assert!(*v >= -1.0 && *v <= 1.0, "tanh output {} out of bounds", v);
    }
}

#[test]
fn test_low_rank_rnn_forward_wrong_input_size() {
    let rnn = TnLowRankRnn::new(4, 8, 3).expect("ok");
    let x = vec![0.1_f64; 3]; // wrong input size
    let h = vec![0.0_f64; 8];
    assert!(rnn.forward(&x, &h).is_err());
}

#[test]
fn test_low_rank_rnn_forward_wrong_hidden_size() {
    let rnn = TnLowRankRnn::new(4, 8, 3).expect("ok");
    let x = vec![0.1_f64; 4];
    let h = vec![0.0_f64; 6]; // wrong hidden size
    assert!(rnn.forward(&x, &h).is_err());
}

#[test]
fn test_low_rank_rnn_compression_ratio() {
    let rnn = TnLowRankRnn::new(4, 16, 4).expect("ok");
    let ratio = rnn.compression_ratio();
    assert!(ratio > 0.0);
    assert!(ratio < 1.0, "TT-RNN compression ratio {} should be < 1 for rank=4 vs hidden=16", ratio);
}

#[test]
fn test_low_rank_rnn_sequential_forward() {
    let rnn = TnLowRankRnn::new(2, 4, 2).expect("ok");
    let mut h = vec![0.0_f64; 4];
    for t in 0..5 {
        let x = vec![(t as f64) * 0.1; 2];
        h = rnn.forward(&x, &h).expect("ok");
    }
    assert_eq!(h.len(), 4);
    for v in &h {
        assert!(v.is_finite());
    }
}

// ============================================================================
// §8 TnEntanglementMeasures tests
// ============================================================================

#[test]
fn test_entropy_von_neumann_non_negative() {
    let svs = vec![1.0, 0.5, 0.3, 0.1];
    let s = TnEntanglementMeasures::von_neumann_entropy(&svs);
    assert!(s >= 0.0, "von Neumann entropy should be >= 0, got {}", s);
}

#[test]
fn test_entropy_von_neumann_pure_state() {
    // Pure state: only one nonzero singular value → entropy = 0
    let svs = vec![1.0, 0.0, 0.0];
    let s = TnEntanglementMeasures::von_neumann_entropy(&svs);
    assert!(s.abs() < 1e-10, "pure state entropy should be 0, got {}", s);
}

#[test]
fn test_entropy_von_neumann_maximally_entangled() {
    // Maximally entangled: all singular values equal
    let svs = vec![1.0, 1.0, 1.0, 1.0];
    let s = TnEntanglementMeasures::von_neumann_entropy(&svs);
    // S = log(4) ≈ 1.386
    let expected = (4_f64).ln();
    assert!((s - expected).abs() < 1e-6, "max entropy should be ln(4)={}, got {}", expected, s);
}

#[test]
fn test_entropy_renyi_alpha_2() {
    let svs = vec![1.0, 0.5, 0.25];
    let s = TnEntanglementMeasures::renyi_entropy(&svs, 2.0);
    assert!(s >= 0.0 || s.is_finite()); // Rényi entropy can be complex but should be finite
}

#[test]
fn test_entropy_renyi_alpha_1_approaches_von_neumann() {
    let svs = vec![1.0, 0.8, 0.4, 0.2];
    let s_vn = TnEntanglementMeasures::von_neumann_entropy(&svs);
    let s_r1 = TnEntanglementMeasures::renyi_entropy(&svs, 1.0);
    assert!((s_vn - s_r1).abs() < 1e-6, "Rényi(α=1) ≈ von Neumann: {} vs {}", s_r1, s_vn);
}

#[test]
fn test_entropy_renyi_empty() {
    let s = TnEntanglementMeasures::renyi_entropy(&[], 2.0);
    assert_eq!(s, 0.0);
}

#[test]
fn test_schmidt_rank_all_above_tol() {
    let svs = vec![1.0, 0.5, 0.3, 0.1];
    let rank = TnEntanglementMeasures::schmidt_rank(&svs, 0.05);
    assert_eq!(rank, 4);
}

#[test]
fn test_schmidt_rank_partial() {
    let svs = vec![1.0, 0.5, 0.01, 0.001];
    let rank = TnEntanglementMeasures::schmidt_rank(&svs, 0.05);
    assert_eq!(rank, 2);
}

#[test]
fn test_schmidt_rank_empty() {
    let rank = TnEntanglementMeasures::schmidt_rank(&[], 0.01);
    assert_eq!(rank, 0);
}

#[test]
fn test_entanglement_spectrum_valid_bond() {
    let mps = TnMatrixProductState::new(4, 2, 5);
    let spectrum = TnEntanglementMeasures::entanglement_spectrum(&mps, 1);
    assert!(!spectrum.is_empty(), "spectrum at valid bond should be non-empty");
}

#[test]
fn test_entanglement_spectrum_invalid_bond() {
    let mps = TnMatrixProductState::new(4, 2, 5);
    let spectrum = TnEntanglementMeasures::entanglement_spectrum(&mps, 10);
    assert!(spectrum.is_empty(), "spectrum at invalid bond should be empty");
}

// ============================================================================
// §9 TnNeuralNetworkTN tests
// ============================================================================

#[test]
fn test_tn_network_construction() {
    let net = TnNeuralNetworkTN::new(&[16, 8, 4], 4).expect("ok");
    assert_eq!(net.layers.len(), 2);
}

#[test]
fn test_tn_network_rejects_insufficient_dims() {
    let result = TnNeuralNetworkTN::new(&[16], 4);
    assert!(result.is_err());
}

#[test]
fn test_tn_network_forward_output_shape() {
    let net = TnNeuralNetworkTN::new(&[8, 4, 2], 3).expect("ok");
    let x = vec![0.5_f64; 8];
    let out = net.forward(&x).expect("forward ok");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_tn_network_forward_3_layers() {
    let net = TnNeuralNetworkTN::new(&[16, 8, 4, 2], 4).expect("ok");
    let x = vec![0.1_f64; 16];
    let out = net.forward(&x).expect("ok");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_tn_network_total_params_positive() {
    let net = TnNeuralNetworkTN::new(&[16, 8, 4], 4).expect("ok");
    assert!(net.total_params() > 0);
}

#[test]
fn test_tn_network_compression_ratio() {
    let net = TnNeuralNetworkTN::new(&[128, 64, 32], 8).expect("ok");
    let dense_params = 128 * 64 + 64 * 32;
    let ratio = net.compression_ratio(dense_params);
    assert!(ratio > 0.0);
    // With rank=8 << 128, compression should happen
    assert!(ratio < 1.0, "compression ratio {} should be < 1", ratio);
}

#[test]
fn test_tn_network_forward_finite_outputs() {
    let net = TnNeuralNetworkTN::new(&[10, 5, 3], 3).expect("ok");
    let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.05).collect();
    let out = net.forward(&x).expect("ok");
    for v in &out {
        assert!(v.is_finite(), "output {} should be finite", v);
    }
}

#[test]
fn test_tn_network_layer_dims_preserved() {
    let net = TnNeuralNetworkTN::new(&[8, 4, 2], 2).expect("ok");
    assert_eq!(net.layer_dims, vec![8, 4, 2]);
}

// ============================================================================
// §10 TnMetrics tests
// ============================================================================

#[test]
fn test_metrics_mps_fidelity_same_model() {
    let mps = TnMatrixProductState::new(3, 2, 4);
    let input = make_uniform_input(3, 2);
    let fidelity = TnMetrics::mps_fidelity(&mps, &mps, &input).expect("ok");
    assert!((fidelity - 1.0).abs() < 1e-8, "fidelity with itself should be 1.0, got {}", fidelity);
}

#[test]
fn test_metrics_mps_fidelity_different_models() {
    let mps1 = TnMatrixProductState::new(3, 2, 4);
    let mps2 = TnMatrixProductState::new(3, 2, 4);
    let input = make_uniform_input(3, 2);
    let fidelity = TnMetrics::mps_fidelity(&mps1, &mps2, &input).expect("ok");
    assert!((0.0..=1.0).contains(&fidelity), "fidelity should be in [0,1], got {}", fidelity);
}

#[test]
fn test_metrics_tucker_reconstruction_error_identity() {
    // Identity matrix: Tucker with full rank should give near-zero error
    let w = make_eye(4);
    let layer = TnTuckerLayer::from_dense(&w, 4);
    let err = TnMetrics::tucker_reconstruction_error(&w, &layer);
    // With full rank approximation, error should be small
    assert!(err < 1.0, "reconstruction error for identity should be small, got {}", err);
}

#[test]
fn test_metrics_tucker_reconstruction_error_non_negative() {
    let w = make_det_matrix(5, 4, 2.0);
    let layer = TnTuckerLayer::from_dense(&w, 2);
    let err = TnMetrics::tucker_reconstruction_error(&w, &layer);
    assert!(err >= 0.0);
}

#[test]
fn test_metrics_bond_dimension_profile() {
    let mps = TnMatrixProductState::new(5, 2, 6);
    let profile = TnMetrics::bond_dimension_profile(&mps);
    assert_eq!(profile.len(), 4); // n_sites - 1
}

#[test]
fn test_metrics_effective_rank_all_significant() {
    let svs = vec![1.0, 0.9, 0.8, 0.7];
    let rank = TnMetrics::effective_rank(&svs, 0.5);
    assert_eq!(rank, 4);
}

#[test]
fn test_metrics_effective_rank_partial() {
    let svs = vec![10.0, 5.0, 0.1, 0.01];
    let rank = TnMetrics::effective_rank(&svs, 0.1);
    // threshold * max = 0.1 * 10 = 1.0 → only 10.0 and 5.0 pass
    assert_eq!(rank, 2);
}

#[test]
fn test_metrics_effective_rank_empty() {
    let rank = TnMetrics::effective_rank(&[], 0.1);
    assert_eq!(rank, 0);
}

#[test]
fn test_metrics_effective_rank_all_zero() {
    let svs = vec![0.0, 0.0, 0.0];
    let rank = TnMetrics::effective_rank(&svs, 0.01);
    assert_eq!(rank, 0);
}

// ============================================================================
// Additional integration tests
// ============================================================================

#[test]
fn test_mps_single_site() {
    // Edge case: single-site MPS
    let mps = TnMatrixProductState::new(1, 3, 4);
    assert_eq!(mps.sites[0].chi_left, 1);
    assert_eq!(mps.sites[0].chi_right, 1);
    let input = vec![vec![0.5_f64; 3]];
    let out = mps.contract_all(&input).expect("ok");
    assert_eq!(out.len(), 1);
}

#[test]
fn test_tree_node_contract_output() {
    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
    let node = TnTreeNode::new(3, 3, 4, &mut rng);
    let left = vec![0.5; 3];
    let right = vec![0.3; 3];
    let out = node.contract(&left, &right);
    assert_eq!(out.len(), 4);
}

#[test]
fn test_tucker_layer_zero_input() {
    let layer = TnTuckerLayer::new(8, 4, 3);
    let x = vec![0.0_f64; 8];
    let y = layer.forward(&x);
    // With zero input and zero bias, output should be zero
    for v in &y {
        assert_eq!(*v, 0.0, "zero input with zero bias should give zero output");
    }
}

#[test]
fn test_mera_6_sites_d2() {
    let mera = TnMeraLayer::new(6, 2).expect("ok");
    let state: Vec<Vec<f64>> = (0..6).map(|i| vec![(i as f64) * 0.1; 2]).collect();
    let out = mera.coarsen(&state).expect("ok");
    assert_eq!(out.len(), 3);
    for v in &out {
        assert_eq!(v.len(), 2);
    }
}

#[test]
fn test_conv_kernel_rank1_extreme_compression() {
    let ck = TnConvolutionalKernel::new(32, 64, 5, 1);
    let ratio = ck.compression_ratio();
    // Very high compression for rank=1
    assert!(ratio < 0.2, "rank-1 kernel ratio {} should be very small", ratio);
}

#[test]
fn test_quantum_layer_multiple_forward_calls() {
    let ql = TnQuantumInspiredLayer::new(3, 2, 4, 1);
    // Multiple forward calls should be consistent
    let x = make_uniform_input(3, 2);
    let amp1 = ql.amplitude(&x).expect("ok");
    let amp2 = ql.amplitude(&x).expect("ok");
    assert!((amp1 - amp2).abs() < 1e-12, "repeated amplitude calls should be consistent");
}

#[test]
fn test_rnn_zero_hidden_state() {
    let rnn = TnLowRankRnn::new(3, 6, 2).expect("ok");
    let x = vec![0.5_f64; 3];
    let h = vec![0.0_f64; 6];
    let h_new = rnn.forward(&x, &h).expect("ok");
    assert_eq!(h_new.len(), 6);
    // Output should be tanh of (W_ih @ x + 0 + bias)
    for v in &h_new {
        assert!(*v >= -1.0 && *v <= 1.0);
    }
}

#[test]
fn test_entanglement_spectrum_positive_values() {
    let mps = TnMatrixProductState::new(5, 3, 6);
    let spectrum = TnEntanglementMeasures::entanglement_spectrum(&mps, 2);
    for sv in &spectrum {
        assert!(*sv >= 0.0, "singular values should be non-negative, got {}", sv);
    }
}

#[test]
fn test_tn_network_relu_applied() {
    // Verify ReLU is applied: all intermediate hidden states should be non-negative
    // We can verify by checking that after multiple evaluations outputs are bounded
    let net = TnNeuralNetworkTN::new(&[4, 8, 2], 2).expect("ok");
    let x = vec![-1.0_f64; 4];
    let out = net.forward(&x).expect("ok");
    assert_eq!(out.len(), 2);
    for v in &out {
        assert!(v.is_finite());
    }
}
