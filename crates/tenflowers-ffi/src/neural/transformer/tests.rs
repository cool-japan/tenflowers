use super::*;
use crate::implicit_autograd::{
    record_and_link_unary as record_unary_for_test, run_backward, UnaryOpKind as UnaryKindForTest,
};

fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
    let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.05 - 0.5).collect()
}

/// Sum every element of `tensor` into a scalar `PyTensor`, recording the
/// reduction on the implicit tape. Re-derived locally (see
/// `normalization.rs`'s own private test helper of the same name — that
/// module's `mod tests` is private, so this cannot be imported cross-module).
fn tape_sum_to_scalar(tensor: &PyTensor) -> PyTensor {
    let raw = tenflowers_core::ops::sum(&tensor.tensor, None, false).expect("sum must succeed");
    let scalar = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_unary_for_test(
        UnaryKindForTest::Sum {
            axes: None,
            keepdims: false,
        },
        tensor,
        &scalar,
    )
    .expect("recording sum must succeed");
    scalar
}

/// `sum(tensor * distinct_per_element_weights)`, reduced to a scalar —
/// real, tape-tracked ops (`Mul` then `Sum`), exactly mirroring
/// `normalization.rs`'s own weighted-loss idiom (see that module's
/// `batch_norm_training_mode_gamma_beta_gradients_are_correct` test for the
/// full rationale this is copied from).
///
/// A PLAIN `sum(tensor)` loss makes `grad_output` uniformly `1` everywhere,
/// which — independent of any implementation bug — can make certain
/// parameters' gradients exactly (or numerically indistinguishably) zero as
/// a pure algebraic identity for SOME random weight initialisations (caught
/// intermittently during development: `PyTransformerEncoderLayer`'s `ff_w1`
/// and `PyTransformerDecoderLayer`'s self-attention `q_proj` both hit this
/// on different runs, with different random `Tensor::randn` seeds). Weighting
/// each output element by a distinct, non-uniform scalar before summing makes
/// `grad_output` non-uniform, which breaks this class of symmetric
/// cancellation without weakening what is actually being tested — the
/// gradient is still a real gradient of a real (if arbitrary) scalar loss.
fn weighted_sum_to_scalar(tensor: &PyTensor) -> PyTensor {
    let size = tensor.tensor.size();
    let weight_data: Vec<f32> = (0..size).map(|i| 0.7 + 0.13 * (i as f32)).collect();
    let weight_shape = tensor.tensor.shape().dims().to_vec();
    let weight_tensor =
        Tensor::from_vec(weight_data, &weight_shape).expect("weight tensor construction");
    let weight = PyTensor {
        tensor: Arc::new(weight_tensor),
        requires_grad: false,
        is_pinned: false,
    };

    let weighted_raw = tenflowers_core::ops::mul(&tensor.tensor, &weight.tensor)
        .expect("weighted mul must succeed");
    let weighted = PyTensor {
        tensor: Arc::new(weighted_raw),
        requires_grad: true,
        is_pinned: false,
    };
    crate::implicit_autograd::record_and_link_binary(
        crate::implicit_autograd::BinaryOpKind::Mul,
        tensor,
        &weight,
        &weighted,
    )
    .expect("recording weighted mul must succeed");

    tape_sum_to_scalar(&weighted)
}

/// Overwrite every one of `params`' values with a distinct, deterministic,
/// well-conditioned ramp pattern (via `.set_data()`), eliminating
/// `Tensor::randn`-sourced randomness from a gradient test entirely.
///
/// # Why this is necessary in addition to `weighted_sum_to_scalar`
///
/// `weighted_sum_to_scalar` alone reduces (but, empirically, over ~50
/// stress-test iterations during development, does not eliminate) the
/// chance of an all-zero gradient for SOME parameter: with genuinely random
/// (`Tensor::randn`-initialised) weights, a small-shaped layer (`d_model=4`,
/// `head_dim=2`) has a real, if low-probability, chance that an entire
/// attention head's contribution to the loss is coincidentally orthogonal
/// to whatever fixed weighting pattern the test loss applies — e.g., if
/// that head's softmax happens to saturate to a near-uniform distribution
/// for this specific random draw AND its value output happens to land in
/// the null space of the loss's weighting for this specific random draw.
/// (Different parameters — `q_proj`, `ff_w1`, `ff_w2` — were each observed
/// to independently trigger this on different runs during development,
/// confirming it is a property of *randomized weights meeting a small
/// shape*, not a bug isolated to any one parameter or op.) Seeding every
/// parameter with a FIXED, deterministic, distinctly-varying-per-element
/// ramp (rather than relying on `Tensor::randn`'s per-run randomness to
/// "probably" avoid every possible degenerate configuration) removes this
/// residual flake source at its actual origin instead of only reducing its
/// probability.
fn seed_all_parameters_with_ramp(py: Python<'_>, params: &[Py<PyParameter>]) {
    for (i, handle) in params.iter().enumerate() {
        let shape = handle.borrow(py).shape();
        let size: usize = shape.iter().product();
        // Distinct per-parameter phase offset (`i` term) so no two
        // parameters ever receive numerically-related values.
        let data: Vec<f32> = (0..size)
            .map(|j| 0.15 + 0.037 * (j as f32) + 0.011 * (i as f32))
            .collect();
        let tensor = Tensor::from_vec(data, &shape).expect("seed tensor construction");
        handle
            .borrow(py)
            .set_data(tensor)
            .expect("set_data must succeed while seeding");
    }
}

/// Central-difference numerical gradient of `f` at each element of
/// `tensor_data`, used to independently verify the analytic gradients
/// `.backward()` populates. `f` must return a scalar loss.
fn finite_difference_grad(tensor_data: &[f32], h: f32, f: impl Fn(&[f32]) -> f32) -> Vec<f32> {
    let mut grad = vec![0.0f32; tensor_data.len()];
    for i in 0..tensor_data.len() {
        let mut plus = tensor_data.to_vec();
        plus[i] += h;
        let mut minus = tensor_data.to_vec();
        minus[i] -= h;
        let f_plus = f(&plus);
        let f_minus = f(&minus);
        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

// -----------------------------------------------------------------------
// Preserved forward-pass tests (pre-existing, shape/masking correctness).
// -----------------------------------------------------------------------

#[test]
fn positional_encoding_modifies_input() {
    let pe = PyPositionalEncoding::new(4, None, None).expect("pe construction");
    let input = make_tensor(vec![1.0; 3 * 2 * 4], &[3, 2, 4]);
    let out = pe.forward(&input, None).expect("forward");

    assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 4]);
    let in_vec = input.tensor.to_vec().expect("in vec");
    let out_vec = out.tensor.to_vec().expect("out vec");
    assert!(
        in_vec
            .iter()
            .zip(out_vec.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "positional encoding must change the input"
    );
}

#[test]
fn transformer_encoder_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let enc =
            PyTransformerEncoderLayer::new(py, 8, 2, None, None, None, None, None).expect("enc");
        let src = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        let out = enc.forward(py, &src, None, None).expect("forward");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "encoder output must not be all zeros"
        );
    });
}

#[test]
fn transformer_decoder_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let dec =
            PyTransformerDecoderLayer::new(py, 8, 2, None, None, None, None, None).expect("dec");
        let tgt = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        let memory = make_tensor(ramp(4 * 2 * 8), &[4, 2, 8]);
        let out = dec
            .forward(py, &tgt, &memory, None, None, None, None)
            .expect("forward");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "decoder output must not be all zeros"
        );
    });
}

#[test]
fn transformer_encoder_forward_with_masks_succeeds() {
    Python::initialize();
    Python::attach(|py| {
        // Providing real masks previously returned an error; it must now succeed.
        let enc =
            PyTransformerEncoderLayer::new(py, 8, 2, None, None, None, None, None).expect("enc");
        // Default batch_first=false: src is [seq=3, batch=2, d_model=8].
        let src = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        // Causal [seq, seq] = [3, 3] additive attention mask.
        let src_mask = generate_square_subsequent_mask(3).expect("square mask");
        // key padding [batch, seq] = [2, 3]; mask position 2 for batch 0 only.
        // (No row becomes fully masked, so softmax stays finite.)
        let src_kpm = make_tensor(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 3]);

        let out = enc
            .forward(py, &src, Some(&src_mask), Some(&src_kpm))
            .expect("masked encoder forward must now succeed");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().all(|&x| x.is_finite()),
            "masked encoder output must be finite (no NaN/inf)"
        );
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "masked encoder output must not be all zeros"
        );
    });
}

#[test]
fn transformer_decoder_forward_with_masks_succeeds() {
    Python::initialize();
    Python::attach(|py| {
        // All four decoder mask parameters provided together must succeed.
        let dec =
            PyTransformerDecoderLayer::new(py, 8, 2, None, None, None, None, None).expect("dec");
        // Default batch_first=false: tgt [tgt=3, batch=2, 8], memory [src=4, batch=2, 8].
        let tgt = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        let memory = make_tensor(ramp(4 * 2 * 8), &[4, 2, 8]);
        let tgt_mask = generate_square_subsequent_mask(3).expect("tgt mask"); // [3, 3]
        let memory_mask = make_tensor(vec![0.0; 3 * 4], &[3, 4]); // [tgt=3, src=4]
        let tgt_kpm = make_tensor(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 3]); // [batch, tgt]
                                                                                // [batch, src=4]; mask memory position 3 for batch 0 only.
        let memory_kpm = make_tensor(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], &[2, 4]);

        let out = dec
            .forward(
                py,
                &tgt,
                &memory,
                Some(&tgt_mask),
                Some(&memory_mask),
                Some(&tgt_kpm),
                Some(&memory_kpm),
            )
            .expect("masked decoder forward must now succeed");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().all(|&x| x.is_finite()),
            "masked decoder output must be finite (no NaN/inf)"
        );
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "masked decoder output must not be all zeros"
        );
    });
}

// -----------------------------------------------------------------------
// batch_first=true forward correctness (the pre-existing tests above only
// exercise the default batch_first=false layout).
// -----------------------------------------------------------------------

#[test]
fn transformer_encoder_forward_batch_first_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let enc = PyTransformerEncoderLayer::new(py, 8, 2, None, None, None, Some(true), None)
            .expect("enc");
        // batch_first=true: src is [batch=2, seq=3, d_model=8].
        let src = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);
        let out = enc.forward(py, &src, None, None).expect("forward");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().all(|v| v.is_finite()),
            "batch_first encoder output must be finite"
        );
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "batch_first encoder output must not be all zeros"
        );
    });
}

// -----------------------------------------------------------------------
// Gradient-flow tests: forward -> loss(sum) -> backward -> assert every
// parameter's gradient is populated, non-zero, finite, and (for a spot
// checked parameter) matches a finite-difference oracle.
// -----------------------------------------------------------------------

#[test]
fn positional_encoding_gradient_flows_through_unchanged() {
    // `pe` itself has no gradient, but `x`'s gradient must flow through the
    // add unchanged: d(sum(x + pe))/dx = 1 everywhere.
    Python::initialize();
    Python::attach(|py| {
        let pe_layer = PyPositionalEncoding::new(4, None, None).expect("pe construction");
        let x_data = ramp(3 * 2 * 4);
        let x_raw = Tensor::from_vec(x_data.clone(), &[3, 2, 4]).expect("x tensor");
        let x = PyTensor {
            tensor: Arc::new(x_raw),
            requires_grad: true,
            is_pinned: false,
        };
        crate::implicit_autograd::mark_leaf(&x);

        let out = pe_layer.forward(&x, None).expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let grad = crate::implicit_autograd::get_grad(&x)
            .expect("x gradient must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        assert_eq!(grad.len(), x_data.len());
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                (g - 1.0).abs() < 1e-5,
                "grad[{i}] must be exactly 1.0 (add's gradient is identity), got {g}"
            );
        }
    });
}

#[test]
fn transformer_encoder_all_parameters_receive_finite_nonzero_gradients() {
    Python::initialize();
    Python::attach(|py| {
        // Small shapes: d_model=4, nhead=2, dim_feedforward=6, batch=2, seq=2.
        // Default activation ("relu").
        let enc = PyTransformerEncoderLayer::new(py, 4, 2, Some(6), None, None, Some(true), None)
            .expect("enc construction");
        let src = make_tensor(ramp(2 * 2 * 4), &[2, 2, 4]);

        // Deterministically seed every parameter (see
        // `seed_all_parameters_with_ramp`'s doc for why this — not just
        // `weighted_sum_to_scalar`'s non-uniform loss weighting alone — is
        // needed to fully eliminate a real, if low-probability,
        // `Tensor::randn`-sourced flake risk with these small shapes).
        let params = enc.parameters(py);
        seed_all_parameters_with_ramp(py, &params);

        let out = enc.forward(py, &src, None, None).expect("forward");
        let loss = weighted_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        // self_attn: 5 params (q,k,v,out,bias) + ff: 4 params + layer_norm: 2 params = 11.
        assert_eq!(
            params.len(),
            11,
            "expected 5 (self_attn) + 4 (ff) + 2 (layer_norm)"
        );

        for (i, handle) in params.iter().enumerate() {
            let grad = handle
                .borrow(py)
                .grad()
                .unwrap_or_else(|e| panic!("param[{i}] gradient must be populated: {e}"));
            let grad_vec = grad.tensor.to_vec().expect("grad readable");
            assert!(
                grad_vec.iter().any(|&g| g != 0.0),
                "param[{i}] grad must be non-zero: {grad_vec:?}"
            );
            assert!(
                grad_vec.iter().all(|g| g.is_finite()),
                "param[{i}] grad must be finite: {grad_vec:?}"
            );
        }
    });
}

#[test]
fn transformer_encoder_ff_w2_gradient_matches_finite_difference() {
    Python::initialize();
    Python::attach(|py| {
        // d_model=4, nhead=2, dim_feedforward=3 (smallest feedforward that
        // still has a real hidden layer), batch=1, seq=2.
        let enc = PyTransformerEncoderLayer::new(py, 4, 2, Some(3), None, None, Some(true), None)
            .expect("enc construction");
        let src_data = ramp(2 * 4);
        let src = make_tensor(src_data.clone(), &[1, 2, 4]);

        let out = enc.forward(py, &src, None, None).expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let params = enc.parameters(py);
        // Index 5 = ff_w2 ([self_attn x5, ff_w1, ff_b1, ff_w2, ff_b2, gamma, beta]).
        let ff_w2_handle = &params[7];
        let grad_ff_w2 = ff_w2_handle
            .borrow(py)
            .grad()
            .expect("ff_w2 grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let ff_w2_shape = ff_w2_handle.borrow(py).shape();
        let ff_w2_vals = ff_w2_handle
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");

        // Snapshot every other parameter's current value.
        let snapshots: Vec<Tensor<f32>> = params
            .iter()
            .map(|p| {
                p.borrow(py)
                    .to_tensor()
                    .expect("to_tensor")
                    .tensor
                    .as_ref()
                    .clone()
            })
            .collect();

        let loss_fn = |ff_w2_flat: &[f32]| -> f32 {
            Python::attach(|py2| {
                let probe = PyTransformerEncoderLayer::new(
                    py2,
                    4,
                    2,
                    Some(3),
                    None,
                    None,
                    Some(true),
                    None,
                )
                .expect("probe construction");
                let probe_params = probe.parameters(py2);
                assert_eq!(probe_params.len(), snapshots.len());
                for (handle, snap) in probe_params.iter().zip(snapshots.iter()) {
                    handle.borrow(py2).set_data(snap.clone()).expect("set_data");
                }
                let ff_w2_tensor =
                    Tensor::from_vec(ff_w2_flat.to_vec(), &ff_w2_shape).expect("ff_w2 tensor");
                probe_params[7]
                    .borrow(py2)
                    .set_data(ff_w2_tensor)
                    .expect("set_data ff_w2");

                let probe_src = make_tensor(src_data.clone(), &[1, 2, 4]);
                let probe_out = probe
                    .forward(py2, &probe_src, None, None)
                    .expect("probe forward");
                probe_out.tensor.to_vec().expect("readable").iter().sum()
            })
        };

        let fd_grad = finite_difference_grad(&ff_w2_vals, 1e-3, loss_fn);
        assert_close(
            &grad_ff_w2,
            &fd_grad,
            5e-2,
            "transformer_encoder ff_w2 grad vs finite-difference",
        );
    });
}

#[test]
fn transformer_decoder_all_parameters_receive_finite_nonzero_gradients() {
    Python::initialize();
    Python::attach(|py| {
        let dec = PyTransformerDecoderLayer::new(py, 4, 2, Some(6), None, None, Some(true), None)
            .expect("dec construction");
        let tgt = make_tensor(ramp(2 * 4), &[1, 2, 4]);
        let memory = make_tensor(ramp(3 * 4), &[1, 3, 4]);

        // Deterministically seed every parameter — see
        // `seed_all_parameters_with_ramp`'s doc for why this is needed in
        // addition to `weighted_sum_to_scalar`'s non-uniform loss weighting
        // to fully eliminate a real, if low-probability,
        // `Tensor::randn`-sourced flake risk with these small shapes
        // (caught intermittently during development on more than one
        // different parameter across different runs).
        let params = dec.parameters(py);
        seed_all_parameters_with_ramp(py, &params);

        let out = dec
            .forward(py, &tgt, &memory, None, None, None, None)
            .expect("forward");
        let loss = weighted_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        // self_attn: 5 + cross_attn: 5 + ff: 4 + layer_norm: 2 = 16.
        assert_eq!(
            params.len(),
            16,
            "expected 5 (self_attn) + 5 (cross_attn) + 4 (ff) + 2 (layer_norm)"
        );

        for (i, handle) in params.iter().enumerate() {
            let grad = handle
                .borrow(py)
                .grad()
                .unwrap_or_else(|e| panic!("param[{i}] gradient must be populated: {e}"));
            let grad_vec = grad.tensor.to_vec().expect("grad readable");
            assert!(
                grad_vec.iter().any(|&g| g != 0.0),
                "param[{i}] grad must be non-zero: {grad_vec:?}"
            );
            assert!(
                grad_vec.iter().all(|g| g.is_finite()),
                "param[{i}] grad must be finite: {grad_vec:?}"
            );
        }
    });
}

#[test]
fn transformer_decoder_ff_b2_gradient_matches_finite_difference() {
    Python::initialize();
    Python::attach(|py| {
        let dec = PyTransformerDecoderLayer::new(py, 4, 2, Some(3), None, None, Some(true), None)
            .expect("dec construction");
        let tgt_data = ramp(2 * 4);
        let memory_data = ramp(2 * 4);
        let tgt = make_tensor(tgt_data.clone(), &[1, 2, 4]);
        let memory = make_tensor(memory_data.clone(), &[1, 2, 4]);

        let out = dec
            .forward(py, &tgt, &memory, None, None, None, None)
            .expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let params = dec.parameters(py);
        // Index 13 = ff_b2 ([self_attn x5, cross_attn x5, ff_w1, ff_b1, ff_w2, ff_b2, gamma, beta]).
        let ff_b2_handle = &params[13];
        let grad_ff_b2 = ff_b2_handle
            .borrow(py)
            .grad()
            .expect("ff_b2 grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let ff_b2_shape = ff_b2_handle.borrow(py).shape();
        let ff_b2_vals = ff_b2_handle
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");

        let snapshots: Vec<Tensor<f32>> = params
            .iter()
            .map(|p| {
                p.borrow(py)
                    .to_tensor()
                    .expect("to_tensor")
                    .tensor
                    .as_ref()
                    .clone()
            })
            .collect();

        let loss_fn = |ff_b2_flat: &[f32]| -> f32 {
            Python::attach(|py2| {
                let probe = PyTransformerDecoderLayer::new(
                    py2,
                    4,
                    2,
                    Some(3),
                    None,
                    None,
                    Some(true),
                    None,
                )
                .expect("probe construction");
                let probe_params = probe.parameters(py2);
                assert_eq!(probe_params.len(), snapshots.len());
                for (handle, snap) in probe_params.iter().zip(snapshots.iter()) {
                    handle.borrow(py2).set_data(snap.clone()).expect("set_data");
                }
                let ff_b2_tensor =
                    Tensor::from_vec(ff_b2_flat.to_vec(), &ff_b2_shape).expect("ff_b2 tensor");
                probe_params[13]
                    .borrow(py2)
                    .set_data(ff_b2_tensor)
                    .expect("set_data ff_b2");

                let probe_tgt = make_tensor(tgt_data.clone(), &[1, 2, 4]);
                let probe_memory = make_tensor(memory_data.clone(), &[1, 2, 4]);
                let probe_out = probe
                    .forward(py2, &probe_tgt, &probe_memory, None, None, None, None)
                    .expect("probe forward");
                probe_out.tensor.to_vec().expect("readable").iter().sum()
            })
        };

        let fd_grad = finite_difference_grad(&ff_b2_vals, 1e-3, loss_fn);
        assert_close(
            &grad_ff_b2,
            &fd_grad,
            5e-2,
            "transformer_decoder ff_b2 grad vs finite-difference",
        );
    });
}

// -----------------------------------------------------------------------
// `.parameters()` identity-sharing tests, mirroring `layers.rs`'s
// `parameters_share_identity_with_forward_across_an_optimizer_update_cycle`.
// -----------------------------------------------------------------------

#[test]
fn encoder_parameters_share_identity_with_forward_and_survive_set_data() {
    Python::initialize();
    Python::attach(|py| {
        let enc = PyTransformerEncoderLayer::new(py, 4, 2, Some(6), None, None, Some(true), None)
            .expect("enc construction");
        let src = make_tensor(ramp(2 * 4), &[1, 2, 4]);

        // Deterministically seed every parameter (see
        // `seed_all_parameters_with_ramp`'s doc): with genuinely random
        // (`Tensor::randn`) weights, this test's premise — "zeroing ff_w2
        // must change the output" — has a real, if low-probability, chance
        // of being spuriously false for a random draw where `ff_w2`'s
        // contribution to `out1` was ALREADY exactly zero even before being
        // explicitly zeroed (e.g. the whole feed-forward hidden layer
        // landing post-ReLU-zero for this input) — caught intermittently
        // during development. A fixed, well-conditioned seed removes that
        // possibility rather than merely reducing its probability.
        let params_before = enc.parameters(py);
        seed_all_parameters_with_ramp(py, &params_before);
        let ids_before: Vec<usize> = params_before.iter().map(|p| p.borrow(py).id()).collect();

        let out1 = enc.forward(py, &src, None, None).expect("first forward");
        let out1_vec = out1.tensor.to_vec().expect("out1 readable");

        // Zero out ff_w2 (index 7) via a `.parameters()` handle.
        let ff_w2_handle = &params_before[7];
        let ff_w2_shape = ff_w2_handle.borrow(py).shape();
        ff_w2_handle
            .borrow(py)
            .set_data(Tensor::<f32>::zeros(&ff_w2_shape))
            .expect("set_data");

        assert_eq!(
            ff_w2_handle.borrow(py).id(),
            ids_before[7],
            "ff_w2 parameter id must be stable across set_data()"
        );

        let out2 = enc.forward(py, &src, None, None).expect("second forward");
        let out2_vec = out2.tensor.to_vec().expect("out2 readable");

        assert_eq!(out1_vec.len(), out2_vec.len());
        let changed = out1_vec
            .iter()
            .zip(out2_vec.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            changed,
            "zeroing ff_w2 via a parameters() handle must change forward()'s output \
             on this SAME layer instance"
        );

        let params_after = enc.parameters(py);
        let ids_after: Vec<usize> = params_after.iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(ids_before, ids_after, "every parameter id must be stable");
    });
}

#[test]
fn decoder_parameters_share_identity_with_forward_and_survive_set_data() {
    Python::initialize();
    Python::attach(|py| {
        let dec = PyTransformerDecoderLayer::new(py, 4, 2, Some(6), None, None, Some(true), None)
            .expect("dec construction");
        let tgt = make_tensor(ramp(2 * 4), &[1, 2, 4]);
        let memory = make_tensor(ramp(3 * 4), &[1, 3, 4]);

        // Deterministically seed every parameter — see
        // `seed_all_parameters_with_ramp`'s doc and
        // `encoder_parameters_share_identity_with_forward_and_survive_set_data`'s
        // identical comment for the full rationale (removes a real,
        // low-probability `Tensor::randn`-sourced flake risk with this
        // test's "zeroing ff_w2 must change the output" premise).
        let params_before = dec.parameters(py);
        seed_all_parameters_with_ramp(py, &params_before);
        let ids_before: Vec<usize> = params_before.iter().map(|p| p.borrow(py).id()).collect();

        let out1 = dec
            .forward(py, &tgt, &memory, None, None, None, None)
            .expect("first forward");
        let out1_vec = out1.tensor.to_vec().expect("out1 readable");

        // Zero out ff_w2 (index 12: [self_attn x5, cross_attn x5, ff_w1, ff_b1, ff_w2, ...]).
        let ff_w2_handle = &params_before[12];
        let ff_w2_shape = ff_w2_handle.borrow(py).shape();
        ff_w2_handle
            .borrow(py)
            .set_data(Tensor::<f32>::zeros(&ff_w2_shape))
            .expect("set_data");

        assert_eq!(
            ff_w2_handle.borrow(py).id(),
            ids_before[12],
            "ff_w2 parameter id must be stable across set_data()"
        );

        let out2 = dec
            .forward(py, &tgt, &memory, None, None, None, None)
            .expect("second forward");
        let out2_vec = out2.tensor.to_vec().expect("out2 readable");

        assert_eq!(out1_vec.len(), out2_vec.len());
        let changed = out1_vec
            .iter()
            .zip(out2_vec.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            changed,
            "zeroing ff_w2 via a parameters() handle must change forward()'s output \
             on this SAME layer instance"
        );

        let params_after = dec.parameters(py);
        let ids_after: Vec<usize> = params_after.iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(ids_before, ids_after, "every parameter id must be stable");
    });
}

#[test]
fn encoder_clone_produces_independent_parameter_identities() {
    Python::initialize();
    Python::attach(|py| {
        let enc = PyTransformerEncoderLayer::new(py, 4, 2, Some(6), None, None, Some(true), None)
            .expect("enc construction");
        let cloned = enc.clone();

        let original_ids: Vec<usize> = enc
            .parameters(py)
            .iter()
            .map(|p| p.borrow(py).id())
            .collect();
        let cloned_ids: Vec<usize> = cloned
            .parameters(py)
            .iter()
            .map(|p| p.borrow(py).id())
            .collect();

        assert_eq!(original_ids.len(), cloned_ids.len());
        for (o, c) in original_ids.iter().zip(cloned_ids.iter()) {
            assert_ne!(
                o, c,
                "Clone must produce a genuinely independent layer with fresh parameter ids"
            );
        }
    });
}
