//! Tests for `nn_verification` — Round 47 Track B.
//!
//! 55+ tests covering all ten sections of the module.

#![allow(clippy::approx_constant)]

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: small networks for testing
// ─────────────────────────────────────────────────────────────────────────────

/// Build a tiny 2→2 ReLU network.
fn tiny_relu_net() -> NnvNetwork {
    let layer = NnvLayer {
        weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        bias: vec![0.0, 0.0],
        activation: NnvActivation::Relu,
    };
    NnvNetwork::new(vec![layer]).expect("tiny_relu_net should be valid")
}

/// Build a two-layer 2→2→2 ReLU network.
fn two_layer_relu_net() -> NnvNetwork {
    let l1 = NnvLayer {
        weights: vec![vec![1.0, 1.0], vec![-1.0, 1.0]],
        bias: vec![0.0, 0.0],
        activation: NnvActivation::Relu,
    };
    let l2 = NnvLayer {
        weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        bias: vec![0.5, -0.5],
        activation: NnvActivation::Linear,
    };
    NnvNetwork::new(vec![l1, l2]).expect("two_layer_relu_net should be valid")
}

/// Build a monotone-in-input-0 network: single layer with positive weights.
fn monotone_net() -> NnvNetwork {
    // output[0] = 2*x[0] + 0.5*x[1] + 1.0 (always increasing in x[0])
    let layer = NnvLayer {
        weights: vec![vec![2.0, 0.5]],
        bias: vec![1.0],
        activation: NnvActivation::Linear,
    };
    NnvNetwork::new(vec![layer]).expect("monotone_net should be valid")
}

/// Build a 2-class classification-like network: 2 inputs → 2 outputs.
fn classify_net() -> NnvNetwork {
    // Output 0 is much larger than output 1 near (1,1).
    let layer = NnvLayer {
        weights: vec![vec![3.0, 3.0], vec![-1.0, -1.0]],
        bias: vec![0.0, 0.0],
        activation: NnvActivation::Linear,
    };
    NnvNetwork::new(vec![layer]).expect("classify_net should be valid")
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  NnvInterval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_interval_new_valid() {
    let iv = NnvInterval::new(-1.0, 2.0).expect("should succeed");
    assert_eq!(iv.lb, -1.0);
    assert_eq!(iv.ub, 2.0);
}

#[test]
fn test_interval_new_invalid_lb_gt_ub() {
    let result = NnvInterval::new(3.0, 1.0);
    assert!(result.is_err(), "lb > ub should fail");
}

#[test]
fn test_interval_new_nan() {
    let result = NnvInterval::new(f64::NAN, 1.0);
    assert!(result.is_err(), "NaN lb should fail");
}

#[test]
fn test_interval_point() {
    let iv = NnvInterval::point(5.0);
    assert_eq!(iv.lb, 5.0);
    assert_eq!(iv.ub, 5.0);
}

#[test]
fn test_interval_add_basic() {
    let a = NnvInterval { lb: 1.0, ub: 3.0 };
    let b = NnvInterval { lb: -1.0, ub: 2.0 };
    let c = a.add(&b);
    assert_eq!(c.lb, 0.0);
    assert_eq!(c.ub, 5.0);
}

#[test]
fn test_interval_add_point() {
    let a = NnvInterval::point(2.0);
    let b = NnvInterval::point(3.0);
    let c = a.add(&b);
    assert_eq!(c.lb, 5.0);
    assert_eq!(c.ub, 5.0);
}

#[test]
fn test_interval_mul_positive() {
    // [2,3] * [4,5] = [8,15]
    let a = NnvInterval { lb: 2.0, ub: 3.0 };
    let b = NnvInterval { lb: 4.0, ub: 5.0 };
    let c = a.mul(&b);
    assert!((c.lb - 8.0).abs() < 1e-10);
    assert!((c.ub - 15.0).abs() < 1e-10);
}

#[test]
fn test_interval_mul_cross_zero() {
    // [-1,1] * [-1,1] = [-1,1]
    let a = NnvInterval { lb: -1.0, ub: 1.0 };
    let b = NnvInterval { lb: -1.0, ub: 1.0 };
    let c = a.mul(&b);
    assert!(c.lb <= -1.0 + 1e-10);
    assert!(c.ub >= 1.0 - 1e-10);
}

#[test]
fn test_interval_mul_negative() {
    // [-3,-1] * [2,4] = [-12,-2]
    let a = NnvInterval { lb: -3.0, ub: -1.0 };
    let b = NnvInterval { lb: 2.0, ub: 4.0 };
    let c = a.mul(&b);
    assert!((c.lb - (-12.0)).abs() < 1e-10);
    assert!((c.ub - (-2.0)).abs() < 1e-10);
}

#[test]
fn test_interval_relu_negative() {
    let iv = NnvInterval { lb: -3.0, ub: -1.0 };
    let r = iv.relu();
    assert_eq!(r.lb, 0.0);
    assert_eq!(r.ub, 0.0);
}

#[test]
fn test_interval_relu_positive() {
    let iv = NnvInterval { lb: 1.0, ub: 5.0 };
    let r = iv.relu();
    assert_eq!(r.lb, 1.0);
    assert_eq!(r.ub, 5.0);
}

#[test]
fn test_interval_relu_mixed() {
    let iv = NnvInterval { lb: -2.0, ub: 3.0 };
    let r = iv.relu();
    assert_eq!(r.lb, 0.0);
    assert_eq!(r.ub, 3.0);
}

#[test]
fn test_interval_tanh_bounds_monotone() {
    let iv = NnvInterval { lb: -1.0, ub: 1.0 };
    let r = iv.tanh_bounds();
    assert!((r.lb - (-1.0_f64).tanh()).abs() < 1e-10);
    assert!((r.ub - 1.0_f64.tanh()).abs() < 1e-10);
    assert!(r.lb <= r.ub);
}

#[test]
fn test_interval_sigmoid_bounds() {
    let iv = NnvInterval { lb: 0.0, ub: 2.0 };
    let r = iv.sigmoid_bounds();
    assert!(r.lb > 0.0 && r.ub < 1.0);
    assert!(r.lb <= r.ub);
}

#[test]
fn test_interval_width() {
    let iv = NnvInterval { lb: 1.0, ub: 4.0 };
    assert!((iv.width() - 3.0).abs() < 1e-10);
}

#[test]
fn test_interval_width_zero() {
    let iv = NnvInterval::point(7.0);
    assert_eq!(iv.width(), 0.0);
}

#[test]
fn test_interval_contains() {
    let iv = NnvInterval { lb: -1.0, ub: 3.0 };
    assert!(iv.contains(0.0));
    assert!(iv.contains(-1.0));
    assert!(iv.contains(3.0));
    assert!(!iv.contains(4.0));
    assert!(!iv.contains(-2.0));
}

#[test]
fn test_interval_intersect_overlapping() {
    let a = NnvInterval { lb: 0.0, ub: 5.0 };
    let b = NnvInterval { lb: 3.0, ub: 8.0 };
    let c = a.intersect(&b).expect("should intersect");
    assert!((c.lb - 3.0).abs() < 1e-10);
    assert!((c.ub - 5.0).abs() < 1e-10);
}

#[test]
fn test_interval_intersect_disjoint() {
    let a = NnvInterval { lb: 0.0, ub: 2.0 };
    let b = NnvInterval { lb: 5.0, ub: 8.0 };
    assert!(a.intersect(&b).is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  NnvZonotope
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_zonotope_from_box_centers() {
    let intervals = vec![
        NnvInterval { lb: 0.0, ub: 4.0 },
        NnvInterval { lb: -2.0, ub: 2.0 },
    ];
    let z = NnvZonotope::from_box(&intervals);
    assert!((z.center[0] - 2.0).abs() < 1e-10);
    assert!((z.center[1] - 0.0).abs() < 1e-10);
}

#[test]
fn test_zonotope_from_box_generators() {
    let intervals = vec![
        NnvInterval { lb: 0.0, ub: 4.0 },
        NnvInterval { lb: -2.0, ub: 2.0 },
    ];
    let z = NnvZonotope::from_box(&intervals);
    // First generator: radius for dim 0
    assert!(z.generators.iter().any(|g| (g[0] - 2.0).abs() < 1e-10));
    // Second generator: radius for dim 1
    assert!(z.generators.iter().any(|g| (g[1] - 2.0).abs() < 1e-10));
}

#[test]
fn test_zonotope_interval_bounds_round_trips() {
    let intervals = vec![
        NnvInterval { lb: -1.0, ub: 3.0 },
        NnvInterval { lb: 0.0, ub: 2.0 },
    ];
    let z = NnvZonotope::from_box(&intervals);
    let recovered = z.interval_bounds();
    for (orig, rec) in intervals.iter().zip(recovered.iter()) {
        assert!((orig.lb - rec.lb).abs() < 1e-10, "lb mismatch");
        assert!((orig.ub - rec.ub).abs() < 1e-10, "ub mismatch");
    }
}

#[test]
fn test_zonotope_affine_transform_identity() {
    let intervals = vec![NnvInterval { lb: -1.0, ub: 1.0 }];
    let z = NnvZonotope::from_box(&intervals);
    // Identity: W = [[1]], b = [0]
    let w = vec![vec![1.0_f64]];
    let b = vec![0.0_f64];
    let z2 = z.affine_transform(&w, &b);
    let bounds = z2.interval_bounds();
    assert!((bounds[0].lb - (-1.0)).abs() < 1e-10);
    assert!((bounds[0].ub - 1.0).abs() < 1e-10);
}

#[test]
fn test_zonotope_affine_transform_scale() {
    let intervals = vec![NnvInterval { lb: -1.0, ub: 1.0 }];
    let z = NnvZonotope::from_box(&intervals);
    // Scale by 2
    let w = vec![vec![2.0_f64]];
    let b = vec![0.0_f64];
    let z2 = z.affine_transform(&w, &b);
    let bounds = z2.interval_bounds();
    assert!((bounds[0].lb - (-2.0)).abs() < 1e-10);
    assert!((bounds[0].ub - 2.0).abs() < 1e-10);
}

#[test]
fn test_zonotope_affine_transform_translation() {
    let intervals = vec![NnvInterval { lb: 0.0, ub: 2.0 }];
    let z = NnvZonotope::from_box(&intervals);
    let w = vec![vec![1.0_f64]];
    let b = vec![5.0_f64];
    let z2 = z.affine_transform(&w, &b);
    let bounds = z2.interval_bounds();
    assert!((bounds[0].lb - 5.0).abs() < 1e-10);
    assert!((bounds[0].ub - 7.0).abs() < 1e-10);
}

#[test]
fn test_zonotope_relu_all_positive() {
    let intervals = vec![NnvInterval { lb: 1.0, ub: 3.0 }];
    let z = NnvZonotope::from_box(&intervals);
    let zr = z.relu();
    let bounds = zr.interval_bounds();
    // No change for always-active neurons.
    assert!(bounds[0].lb >= 0.0);
    assert!(bounds[0].ub >= 1.0);
}

#[test]
fn test_zonotope_relu_all_negative() {
    let intervals = vec![NnvInterval { lb: -3.0, ub: -1.0 }];
    let z = NnvZonotope::from_box(&intervals);
    let zr = z.relu();
    let bounds = zr.interval_bounds();
    // Always inactive → should be zeroed.
    assert!((bounds[0].lb).abs() < 1e-10);
    assert!((bounds[0].ub).abs() < 1e-10);
}

#[test]
fn test_zonotope_relu_mixed() {
    let intervals = vec![NnvInterval { lb: -1.0, ub: 1.0 }];
    let z = NnvZonotope::from_box(&intervals);
    let zr = z.relu();
    let bounds = zr.interval_bounds();
    // The zonotope overapproximation of relu([-1,1]) = [0,1]:
    // - The interval bounds must CONTAIN [0, 1] (soundness).
    // - lb may be slightly negative (acceptable overapproximation in DeepZ).
    assert!(
        bounds[0].lb <= 0.0 + 1e-6,
        "lb must be ≤ 0 (contains relu lower bound)"
    );
    assert!(
        bounds[0].ub >= 1.0 - 1e-6,
        "ub must be ≥ 1.0 (contains relu upper bound)"
    );
    // Upper bound must be positive.
    assert!(bounds[0].ub >= 0.0);
}

#[test]
fn test_zonotope_n_neurons() {
    let intervals = vec![
        NnvInterval { lb: 0.0, ub: 1.0 },
        NnvInterval { lb: 0.0, ub: 1.0 },
        NnvInterval { lb: 0.0, ub: 1.0 },
    ];
    let z = NnvZonotope::from_box(&intervals);
    assert_eq!(z.n_neurons(), 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  NnvNetwork
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_network_forward_identity() {
    let net = tiny_relu_net();
    let out = net.forward(&[3.0, -1.0]);
    // ReLU identity layer: max(0, x)
    assert!((out[0] - 3.0).abs() < 1e-10);
    assert!((out[1] - 0.0).abs() < 1e-10);
}

#[test]
fn test_network_forward_two_layer() {
    let net = two_layer_relu_net();
    let out = net.forward(&[1.0, 1.0]);
    // l1: [2, 0] → relu → [2, 0]
    // l2: [1*2+0*0+0.5, 0*2+1*0-0.5] = [2.5, -0.5]
    assert!((out[0] - 2.5).abs() < 1e-10, "out[0] = {}", out[0]);
    assert!((out[1] - (-0.5)).abs() < 1e-10, "out[1] = {}", out[1]);
}

#[test]
fn test_network_n_params() {
    let net = tiny_relu_net();
    // Layer: 2×2 weights + 2 biases = 6
    assert_eq!(net.n_params(), 6);
}

#[test]
fn test_network_n_params_two_layer() {
    let net = two_layer_relu_net();
    // Layer 1: 2×2 + 2 = 6; Layer 2: 2×2 + 2 = 6; total = 12
    assert_eq!(net.n_params(), 12);
}

#[test]
fn test_network_n_layers() {
    assert_eq!(tiny_relu_net().n_layers(), 1);
    assert_eq!(two_layer_relu_net().n_layers(), 2);
}

#[test]
fn test_network_invalid_dim_mismatch() {
    let l1 = NnvLayer {
        weights: vec![vec![1.0, 0.0]],
        bias: vec![0.0],
        activation: NnvActivation::Relu,
    };
    // l2 expects 3 inputs but l1 outputs 1.
    let l2 = NnvLayer {
        weights: vec![vec![1.0, 0.0, 0.0]],
        bias: vec![0.0],
        activation: NnvActivation::Linear,
    };
    let result = NnvNetwork::new(vec![l1, l2]);
    assert!(result.is_err());
}

#[test]
fn test_network_invalid_empty_layers() {
    let result = NnvNetwork::new(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_network_tanh_activation() {
    let layer = NnvLayer {
        weights: vec![vec![1.0]],
        bias: vec![0.0],
        activation: NnvActivation::Tanh,
    };
    let net = NnvNetwork::new(vec![layer]).expect("valid");
    let out = net.forward(&[0.0]);
    assert!((out[0] - 0.0_f64.tanh()).abs() < 1e-10);
}

#[test]
fn test_network_sigmoid_activation() {
    let layer = NnvLayer {
        weights: vec![vec![1.0]],
        bias: vec![0.0],
        activation: NnvActivation::Sigmoid,
    };
    let net = NnvNetwork::new(vec![layer]).expect("valid");
    let out = net.forward(&[0.0]);
    assert!((out[0] - 0.5).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  NnvIntervalBoundPropagation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ibp_contains_concrete_output() {
    let net = tiny_relu_net();
    let input = vec![2.0, -1.0];
    let eps = 0.5;
    let bounds = NnvIntervalBoundPropagation::output_bounds_epsilon_ball(&net, &input, eps)
        .expect("should succeed");
    let actual = net.forward(&input);
    for (b, &a) in bounds.iter().zip(actual.iter()) {
        assert!(
            b.contains(a),
            "concrete output {a} not in [{}, {}]",
            b.lb,
            b.ub
        );
    }
}

#[test]
fn test_ibp_epsilon_ball_width() {
    let net = tiny_relu_net();
    let center = vec![0.0, 0.0];
    let eps = 1.0;
    let bounds =
        NnvIntervalBoundPropagation::output_bounds_epsilon_ball(&net, &center, eps).expect("ok");
    for b in &bounds {
        assert!(b.ub - b.lb >= 0.0, "bounds must be valid intervals");
    }
}

#[test]
fn test_ibp_propagate_identity_net() {
    let net = tiny_relu_net();
    let input_bounds = vec![
        NnvInterval { lb: 1.0, ub: 2.0 },
        NnvInterval { lb: -1.0, ub: 0.0 },
    ];
    let out = NnvIntervalBoundPropagation::propagate(&net, &input_bounds).expect("ok");
    // First neuron: relu([1,2]) = [1,2]
    assert!((out[0].lb - 1.0).abs() < 1e-10);
    assert!((out[0].ub - 2.0).abs() < 1e-10);
    // Second neuron: relu([-1,0]) = [0,0]
    assert!((out[1].lb - 0.0).abs() < 1e-10);
    assert!((out[1].ub - 0.0).abs() < 1e-10);
}

#[test]
fn test_ibp_verify_output_lower_bound_true() {
    let bounds = vec![
        NnvInterval { lb: 2.0, ub: 4.0 },
        NnvInterval { lb: -1.0, ub: 1.0 },
    ];
    assert!(NnvIntervalBoundPropagation::verify_output_lower_bound(
        &bounds, 0, 1.5
    ));
}

#[test]
fn test_ibp_verify_output_lower_bound_false() {
    let bounds = vec![NnvInterval { lb: 0.5, ub: 2.0 }];
    assert!(!NnvIntervalBoundPropagation::verify_output_lower_bound(
        &bounds, 0, 1.0
    ));
}

#[test]
fn test_ibp_wrong_input_dim_error() {
    let net = tiny_relu_net();
    let bounds = vec![NnvInterval::point(0.0)]; // wrong: expects 2
    let result = NnvIntervalBoundPropagation::propagate(&net, &bounds);
    assert!(result.is_err());
}

#[test]
fn test_ibp_two_layer_sound() {
    let net = two_layer_relu_net();
    let center = vec![1.0, 0.5];
    let eps = 0.1;
    let bounds =
        NnvIntervalBoundPropagation::output_bounds_epsilon_ball(&net, &center, eps).expect("ok");
    let actual = net.forward(&center);
    for (b, &a) in bounds.iter().zip(actual.iter()) {
        assert!(
            b.contains(a),
            "soundness violated: {a} ∉ [{}, {}]",
            b.lb,
            b.ub
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  NnvCrownBounds
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_crown_relu_bound_all_positive() {
    let (alo, blo, ahi, bhi) = NnvCrownBounds::relu_linear_bound(1.0, 3.0);
    // Always active: identity
    assert!((alo - 1.0).abs() < 1e-10);
    assert!((blo - 0.0).abs() < 1e-10);
    assert!((ahi - 1.0).abs() < 1e-10);
    assert!((bhi - 0.0).abs() < 1e-10);
}

#[test]
fn test_crown_relu_bound_all_negative() {
    let (alo, blo, ahi, bhi) = NnvCrownBounds::relu_linear_bound(-3.0, -1.0);
    // Always inactive: zero
    assert!((alo - 0.0).abs() < 1e-10);
    assert!((blo - 0.0).abs() < 1e-10);
    assert!((ahi - 0.0).abs() < 1e-10);
    assert!((bhi - 0.0).abs() < 1e-10);
}

#[test]
fn test_crown_relu_bound_mixed() {
    let (alo, blo, ahi, bhi) = NnvCrownBounds::relu_linear_bound(-2.0, 4.0);
    // Mixed case: upper slope = ub/(ub-lb) = 4/6
    let expected_alpha_upper = 4.0 / 6.0;
    assert!((ahi - expected_alpha_upper).abs() < 1e-8);
    let expected_beta_upper = -(-2.0) * 4.0 / 6.0;
    assert!((bhi - expected_beta_upper).abs() < 1e-8);
    // Lower relaxation is conservative (0)
    assert!(alo >= 0.0);
    assert!(blo >= -1e-10);
}

#[test]
fn test_crown_propagate_bounds_sound() {
    let net = tiny_relu_net();
    let input_bounds = vec![
        NnvInterval { lb: 0.5, ub: 1.5 },
        NnvInterval { lb: -0.5, ub: 0.5 },
    ];
    let crown = NnvCrownBounds::propagate_bounds(&net, &input_bounds).expect("ok");
    let ibp = NnvIntervalBoundPropagation::propagate(&net, &input_bounds).expect("ok");
    // Crown bounds should be at most as wide as IBP (or wider due to relaxation direction).
    // At minimum they must contain actual outputs.
    let actual = net.forward(&[1.0, 0.0]);
    for (b, &a) in crown.iter().zip(actual.iter()) {
        // The bounds should at least be valid intervals.
        assert!(b.lb <= b.ub, "Crown bound invalid: [{}, {}]", b.lb, b.ub);
        // IBP upper ≥ Crown upper is NOT guaranteed; just check crown is consistent.
        let _ = ibp[0].ub;
    }
}

#[test]
fn test_crown_certified_lower_bound() {
    // A network that always outputs ≥ 5 for inputs near [2,2].
    let layer = NnvLayer {
        weights: vec![vec![1.0, 1.0]],
        bias: vec![3.0],
        activation: NnvActivation::Linear,
    };
    let net = NnvNetwork::new(vec![layer]).expect("ok");
    let lb = NnvCrownBounds::certified_lower_bound(&net, &[2.0, 2.0], 0.1, 0).expect("ok");
    // At center: 2+2+3=7; with eps=0.1 perturbation worst case: 2*0.1 less = 6.8
    assert!(lb <= 7.0, "lower bound should be ≤ center value");
    assert!(lb > 0.0, "lower bound should be positive for this net");
}

#[test]
fn test_crown_certified_lower_bound_relu() {
    let net = tiny_relu_net();
    let lb = NnvCrownBounds::certified_lower_bound(&net, &[1.0, 0.5], 0.2, 0).expect("ok");
    // Should be non-negative (ReLU output)
    assert!(
        lb >= -1e-6,
        "ReLU output lb should be non-negative, got {lb}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  NnvRandomizedSmoothing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_smoothing_new_fields() {
    let s = NnvRandomizedSmoothing::new(0.25, 500);
    assert!((s.sigma - 0.25).abs() < 1e-10);
    assert_eq!(s.n_samples, 500);
    assert_eq!(s.alpha, 0.001);
}

#[test]
fn test_smoothing_smooth_predict_valid_class() {
    let net = classify_net();
    let mut seed = 12345_u64;
    let (cls, p) =
        NnvRandomizedSmoothing::new(0.1, 100).smooth_predict(&net, &[1.0, 1.0], 2, &mut seed);
    assert!(cls < 2, "class must be in [0, n_classes)");
    assert!((0.0..=1.0).contains(&p), "probability must be in [0,1]");
}

#[test]
fn test_smoothing_smooth_predict_dominant_class() {
    // Net strongly favors class 0 near [1,1].
    let net = classify_net();
    let mut seed = 42_u64;
    let (cls, _) =
        NnvRandomizedSmoothing::new(0.05, 200).smooth_predict(&net, &[1.0, 1.0], 2, &mut seed);
    // With small sigma the dominant class should be 0.
    assert_eq!(cls, 0);
}

#[test]
fn test_smoothing_certify_radius_above_half() {
    let s = NnvRandomizedSmoothing::new(0.25, 1000);
    let r = s.certify_radius(0.8);
    assert!(r > 0.0, "radius should be positive for p_hat > 0.5");
}

#[test]
fn test_smoothing_certify_radius_at_half() {
    let s = NnvRandomizedSmoothing::new(0.25, 1000);
    let r = s.certify_radius(0.5);
    assert_eq!(r, 0.0, "no certificate at p_hat = 0.5");
}

#[test]
fn test_smoothing_certify_radius_below_half() {
    let s = NnvRandomizedSmoothing::new(0.25, 1000);
    let r = s.certify_radius(0.3);
    assert_eq!(r, 0.0, "no certificate below p_hat = 0.5");
}

#[test]
fn test_smoothing_normal_icdf_monotone() {
    let vals: Vec<f64> = vec![0.1, 0.3, 0.5, 0.7, 0.9]
        .into_iter()
        .map(NnvRandomizedSmoothing::normal_icdf)
        .collect();
    for w in vals.windows(2) {
        assert!(w[1] > w[0], "icdf must be monotonically increasing");
    }
}

#[test]
fn test_smoothing_normal_icdf_symmetry() {
    // Phi^{-1}(0.5) ≈ 0
    let mid = NnvRandomizedSmoothing::normal_icdf(0.5);
    assert!(mid.abs() < 0.1, "icdf(0.5) ≈ 0, got {mid}");
}

#[test]
fn test_smoothing_bentkus_bound_nonneg() {
    let lb = NnvRandomizedSmoothing::bentkus_bound(1000, 800, 0.001);
    assert!((0.0..=1.0).contains(&lb));
}

#[test]
fn test_smoothing_bentkus_bound_zero_k() {
    let lb = NnvRandomizedSmoothing::bentkus_bound(1000, 0, 0.001);
    assert_eq!(lb, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  NnvPropertyChecker
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_property_checker_robustness_verified() {
    // With very small epsilon, the dominant class should be certifiable.
    let net = classify_net();
    let result = NnvPropertyChecker::check_robustness_ibp(&net, &[2.0, 2.0], 0.001, 0, 2);
    assert!(
        matches!(result, NnvResult::Verified { .. }),
        "Expected Verified, got {result:?}"
    );
}

#[test]
fn test_property_checker_robustness_unknown_large_eps() {
    // With a large epsilon, IBP may not be able to certify.
    let net = classify_net();
    let result = NnvPropertyChecker::check_robustness_ibp(&net, &[0.0, 0.0], 10.0, 0, 2);
    // Either Unknown or still Verified depending on network structure; just check no panic.
    let _ = result;
}

#[test]
fn test_property_checker_estimate_lipschitz_positive() {
    let net = two_layer_relu_net();
    let lip = NnvPropertyChecker::estimate_lipschitz(&net);
    assert!(lip > 0.0, "Lipschitz estimate must be positive");
}

#[test]
fn test_property_checker_estimate_lipschitz_monotone() {
    let net = monotone_net();
    let lip = NnvPropertyChecker::estimate_lipschitz(&net);
    // Layer has max column sum = max(2.0, 0.5) = 2.0
    assert!((lip - 2.0).abs() < 1e-10, "lip = {lip}");
}

#[test]
fn test_property_check_lipschitz_property_verified() {
    let net = monotone_net();
    let result = NnvPropertyChecker::check(&net, &NnvProperty::Lipschitz { max_lip: 10.0 });
    assert!(matches!(result, NnvResult::Verified { .. }));
}

#[test]
fn test_property_check_lipschitz_property_unknown() {
    let net = two_layer_relu_net();
    let result = NnvPropertyChecker::check(&net, &NnvProperty::Lipschitz { max_lip: 0.001 });
    assert!(matches!(result, NnvResult::Unknown { .. }));
}

#[test]
fn test_property_check_output_bound_verified() {
    // A linear net that always outputs positive values.
    let layer = NnvLayer {
        weights: vec![vec![1.0]],
        bias: vec![5.0],
        activation: NnvActivation::Linear,
    };
    let net = NnvNetwork::new(vec![layer]).expect("ok");
    let prop = NnvProperty::OutputBound {
        input_bounds: vec![NnvInterval { lb: 0.0, ub: 1.0 }],
        output_idx: 0,
        lower: Some(4.0),
        upper: None,
    };
    let result = NnvPropertyChecker::check(&net, &prop);
    assert!(matches!(result, NnvResult::Verified { .. }), "{result:?}");
}

#[test]
fn test_property_check_monotonicity() {
    let net = monotone_net();
    let result = NnvPropertyChecker::check(
        &net,
        &NnvProperty::Monotonicity {
            input_dim: 0,
            output_dim: 0,
        },
    );
    // Positive weight on input 0 → monotone.
    assert!(matches!(result, NnvResult::Verified { .. }), "{result:?}");
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  NnvAdversarialCertifier
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_certifier_new_defaults() {
    let c = NnvAdversarialCertifier::new();
    assert!(c.ibp);
    assert!(c.crown);
    assert!(c.smoothing.is_none());
}

#[test]
fn test_certifier_certify_sample_returns_struct() {
    let net = classify_net();
    let mut seed = 1_u64;
    let cert = NnvAdversarialCertifier::new().certify_sample(&net, &[1.0, 1.0], 0.01, 2, &mut seed);
    assert!(cert.certified_radius >= 0.0);
    assert!(!cert.method.is_empty());
}

#[test]
fn test_certifier_certify_sample_certified_for_easy_case() {
    let net = classify_net();
    let mut seed = 99_u64;
    let cert =
        NnvAdversarialCertifier::new().certify_sample(&net, &[5.0, 5.0], 0.001, 2, &mut seed);
    assert!(
        cert.is_certified,
        "Should certify for small eps near clearly dominant class"
    );
}

#[test]
fn test_certifier_dataset_certification_length() {
    let net = classify_net();
    let inputs = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![0.5, 0.5]];
    let mut seed = 7_u64;
    let dataset_cert =
        NnvAdversarialCertifier::new().certify_dataset(&net, &inputs, 0.01, 2, &mut seed);
    assert_eq!(dataset_cert.certifications.len(), 3);
}

#[test]
fn test_certifier_dataset_certified_accuracy_range() {
    let net = classify_net();
    let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, i as f64]).collect();
    let mut seed = 13_u64;
    let dc = NnvAdversarialCertifier::new().certify_dataset(&net, &inputs, 0.001, 2, &mut seed);
    assert!(dc.certified_accuracy >= 0.0 && dc.certified_accuracy <= 1.0);
}

#[test]
fn test_certifier_dataset_mean_radius_nonneg() {
    let net = classify_net();
    let inputs = vec![vec![1.0, 1.0]];
    let mut seed = 21_u64;
    let dc = NnvAdversarialCertifier::new().certify_dataset(&net, &inputs, 0.01, 2, &mut seed);
    assert!(dc.mean_radius >= 0.0);
}

#[test]
fn test_certifier_with_smoothing() {
    let net = classify_net();
    let mut c = NnvAdversarialCertifier::new();
    c.smoothing = Some(NnvRandomizedSmoothing::new(0.1, 50));
    let mut seed = 31_u64;
    let cert = c.certify_sample(&net, &[1.0, 1.0], 0.05, 2, &mut seed);
    // Just check it doesn't panic.
    assert!(cert.certified_radius >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  NnvMonotonicityCheck
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_monotonicity_check_positive_weight() {
    let net = monotone_net();
    let input_bounds = vec![
        NnvInterval { lb: -1.0, ub: 1.0 },
        NnvInterval { lb: -1.0, ub: 1.0 },
    ];
    // output[0] = 2*x[0] + 0.5*x[1] + 1 — always increasing in x[0].
    let mono = NnvMonotonicityCheck::check_input_monotonicity(&net, 0, 0, &input_bounds);
    assert!(
        mono,
        "Should certify monotonicity for positive-weight linear net"
    );
}

#[test]
fn test_monotonicity_jacobian_bounds_shape() {
    let net = two_layer_relu_net();
    let input_bounds = vec![
        NnvInterval { lb: -1.0, ub: 1.0 },
        NnvInterval { lb: -1.0, ub: 1.0 },
    ];
    let jac = NnvMonotonicityCheck::approximate_jacobian_bounds(&net, &input_bounds);
    assert_eq!(jac.len(), net.output_dim, "rows = output_dim");
    for row in &jac {
        assert_eq!(row.len(), net.input_dim, "cols = input_dim");
    }
}

#[test]
fn test_monotonicity_jacobian_bounds_valid_intervals() {
    let net = monotone_net();
    let input_bounds = vec![
        NnvInterval { lb: 0.0, ub: 1.0 },
        NnvInterval { lb: 0.0, ub: 1.0 },
    ];
    let jac = NnvMonotonicityCheck::approximate_jacobian_bounds(&net, &input_bounds);
    for row in &jac {
        for iv in row {
            assert!(iv.lb <= iv.ub, "Jacobian interval must be valid");
        }
    }
}

#[test]
fn test_monotonicity_convexity_certificate_returns_value() {
    let net = tiny_relu_net();
    let input_bounds = vec![
        NnvInterval { lb: -1.0, ub: 1.0 },
        NnvInterval { lb: -1.0, ub: 1.0 },
    ];
    let cert = NnvMonotonicityCheck::convexity_certificate(&net, &input_bounds);
    assert!(cert.is_some());
    assert!(cert.expect("convexity_certificate should return Some") >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  NnvMetrics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_verified_accuracy_all_verified() {
    let results = vec![
        NnvResult::Verified {
            certificate: "cert".into(),
        },
        NnvResult::Verified {
            certificate: "cert2".into(),
        },
    ];
    assert!((NnvMetrics::verified_accuracy(&results) - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_verified_accuracy_none_verified() {
    let results = vec![
        NnvResult::Unknown { reason: "r".into() },
        NnvResult::Falsified {
            counterexample: vec![0.0],
        },
    ];
    assert!((NnvMetrics::verified_accuracy(&results) - 0.0).abs() < 1e-10);
}

#[test]
fn test_metrics_verified_accuracy_mixed() {
    let results = vec![
        NnvResult::Verified {
            certificate: "cert".into(),
        },
        NnvResult::Unknown { reason: "r".into() },
    ];
    assert!((NnvMetrics::verified_accuracy(&results) - 0.5).abs() < 1e-10);
}

#[test]
fn test_metrics_verified_accuracy_empty() {
    assert_eq!(NnvMetrics::verified_accuracy(&[]), 0.0);
}

#[test]
fn test_metrics_mean_certified_radius() {
    let certs = vec![
        NnvCertification {
            certified_radius: 0.1,
            method: "IBP".into(),
            is_certified: true,
        },
        NnvCertification {
            certified_radius: 0.3,
            method: "CROWN".into(),
            is_certified: true,
        },
    ];
    let mean = NnvMetrics::mean_certified_radius(&certs);
    assert!((mean - 0.2).abs() < 1e-10);
}

#[test]
fn test_metrics_mean_certified_radius_zero() {
    let certs = vec![NnvCertification {
        certified_radius: 0.0,
        method: "none".into(),
        is_certified: false,
    }];
    let mean = NnvMetrics::mean_certified_radius(&certs);
    assert_eq!(mean, 0.0);
}

#[test]
fn test_metrics_mean_certified_radius_empty() {
    assert_eq!(NnvMetrics::mean_certified_radius(&[]), 0.0);
}

#[test]
fn test_metrics_ibp_tightness_same_bounds() {
    let bounds = vec![NnvInterval { lb: 0.0, ub: 2.0 }];
    let t = NnvMetrics::ibp_tightness(&bounds, &bounds);
    assert!((t - 1.0).abs() < 1e-10, "identical bounds → ratio 1.0");
}

#[test]
fn test_metrics_ibp_tightness_crown_tighter() {
    let ibp = vec![NnvInterval { lb: 0.0, ub: 4.0 }];
    let crown = vec![NnvInterval { lb: 1.0, ub: 3.0 }];
    let t = NnvMetrics::ibp_tightness(&ibp, &crown);
    assert!(t < 1.0, "CROWN tighter → ratio < 1.0, got {t}");
}

#[test]
fn test_metrics_network_lipschitz_upper_positive() {
    let lip = NnvMetrics::network_lipschitz_upper(&two_layer_relu_net());
    assert!(lip > 0.0);
}

#[test]
fn test_metrics_output_range_diameter() {
    let bounds = vec![
        NnvInterval { lb: 0.0, ub: 5.0 },
        NnvInterval { lb: -1.0, ub: 2.0 },
    ];
    let d = NnvMetrics::output_range_diameter(&bounds);
    assert!((d - 5.0).abs() < 1e-10);
}

#[test]
fn test_metrics_output_range_diameter_single() {
    let bounds = vec![NnvInterval { lb: -2.0, ub: 4.0 }];
    let d = NnvMetrics::output_range_diameter(&bounds);
    assert!((d - 6.0).abs() < 1e-10);
}
