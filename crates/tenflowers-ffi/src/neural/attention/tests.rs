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
/// reduction on the implicit tape — the `loss = sum(output)` step every
/// gradient test in this module needs before calling `run_backward`. Copied
/// from `normalization.rs`'s own private test helper of the same name (that
/// module's `mod tests` is private, so this must be re-derived locally
/// rather than imported cross-module).
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
fn multihead_attention_is_real_with_normalized_weights() {
    Python::initialize();
    Python::attach(|py| {
        // embed_dim = 8, num_heads = 2, batch_first.
        let mha =
            PyMultiheadAttention::new(py, 8, 2, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        let q = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);
        let k = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);
        let v = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);

        let (output, weights) = mha
            .forward(py, &q, &k, &v, None, Some(true), None, None)
            .expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![2, 3, 8]);
        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "attention output must not be all zeros"
        );

        let weights = weights.expect("attention weights present");
        assert_eq!(weights.tensor.shape().dims().to_vec(), vec![2, 3, 3]);
        let w = weights.tensor.to_vec().expect("weights vec");
        // Weights shape [batch=2, tgt=3, src=3]: every row over `src` sums to ~1.
        let src = 3usize;
        for row in 0..(2 * 3) {
            let sum: f32 = (0..src).map(|j| w[row * src + j]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "attention weight row must sum to ~1, got {}",
                sum
            );
        }
    });
}

#[test]
fn scaled_dot_product_attention_is_real() {
    let q = make_tensor(ramp(8), &[1, 2, 4]);
    let k = make_tensor(ramp(8), &[1, 2, 4]);
    let v = make_tensor(vec![2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], &[1, 2, 4]);

    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("sdpa");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4]);
    let out_vec = out.tensor.to_vec().expect("vec");
    assert!(
        out_vec.iter().any(|&x| x != 0.0),
        "SDPA output must not be all zeros"
    );
}

/// 2x2 identity matrix used to make projections a no-op so that pre-softmax
/// scores are exactly `Q·K^T / sqrt(d_k)` and can be reasoned about by hand.
fn identity_2x2() -> Tensor<f32> {
    Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).expect("identity")
}

#[test]
fn multihead_attention_key_padding_mask_zeros_padded_weight() {
    Python::initialize();
    Python::attach(|py| {
        // embed_dim=2, num_heads=1, batch_first: single head sees the full embed.
        let mha =
            PyMultiheadAttention::new(py, 2, 1, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        // Identity projections => scores are exactly Q·K^T / sqrt(2).
        mha.q_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        mha.k_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        mha.v_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        mha.out_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        if let Some(ref bias_param) = mha.bias_param {
            bias_param
                .borrow(py)
                .set_data(Tensor::zeros(&[2]))
                .expect("set_data");
        }

        // batch=1, tgt_len=1, src_len=3, embed=2.
        // q=[1,0]; k0=[2,0], k1=[1,0], k2=[0,0] => raw scores [2,1,0].
        let q = make_tensor(vec![1.0, 0.0], &[1, 1, 2]);
        let k = make_tensor(vec![2.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1, 3, 2]);
        // v0=[1,0], v1=[0,1], v2=[5,5]: masking v2 (large) changes the output a lot.
        let v = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 5.0, 5.0], &[1, 3, 2]);

        // Unmasked: attention spreads across all three source positions.
        let (out_unmasked, w_unmasked) = mha
            .forward(py, &q, &k, &v, None, Some(true), None, None)
            .expect("unmasked forward");
        let wu = w_unmasked
            .expect("weights present")
            .tensor
            .to_vec()
            .expect("wu");
        assert_eq!(wu.len(), 3);
        assert!(
            wu[2] > 0.01,
            "without a mask, padded pos 2 must carry real weight, got {}",
            wu[2]
        );

        // Masked: key_padding_mask marks source position 2 as padding (nonzero).
        let kpm = make_tensor(vec![0.0, 0.0, 1.0], &[1, 3]);
        let (out_masked, w_masked) = mha
            .forward(py, &q, &k, &v, Some(&kpm), Some(true), None, None)
            .expect("masked forward");
        let wm = w_masked
            .expect("weights present")
            .tensor
            .to_vec()
            .expect("wm");

        // (b) padded position weight is ~0, remaining weights still sum to ~1.
        assert!(wm[2] < 1e-3, "padded weight must be ~0, got {}", wm[2]);
        assert!(
            (wm[0] + wm[1] - 1.0).abs() < 1e-3,
            "surviving weights must sum to ~1, got {}",
            wm[0] + wm[1]
        );
        assert!(
            wm[0] > wm[1],
            "pos 0 has the higher score so must retain more weight ({} vs {})",
            wm[0],
            wm[1]
        );

        // (a) the attention output changes vs the unmasked case.
        let ou = out_unmasked.tensor.to_vec().expect("ou");
        let om = out_masked.tensor.to_vec().expect("om");
        let changed = ou.iter().zip(om.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(changed, "masking a key must change the attention output");
    });
}

#[test]
fn multihead_attention_combines_attn_and_key_padding_masks() {
    Python::initialize();
    Python::attach(|py| {
        let mha =
            PyMultiheadAttention::new(py, 2, 1, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        mha.q_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        mha.k_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        mha.v_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        mha.out_proj_param
            .borrow(py)
            .set_data(identity_2x2())
            .expect("set_data");
        if let Some(ref bias_param) = mha.bias_param {
            bias_param
                .borrow(py)
                .set_data(Tensor::zeros(&[2]))
                .expect("set_data");
        }

        let q = make_tensor(vec![1.0, 0.0], &[1, 1, 2]);
        let k = make_tensor(vec![2.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1, 3, 2]);
        let v = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 5.0, 5.0], &[1, 3, 2]);

        // attn_mask [tgt=1, src=3] blocks source position 0;
        // key_padding_mask [batch=1, src=3] blocks a DIFFERENT position (2).
        let attn_mask = make_tensor(vec![-1.0e9, 0.0, 0.0], &[1, 3]);
        let kpm = make_tensor(vec![0.0, 0.0, 1.0], &[1, 3]);

        let (_out, weights) = mha
            .forward(
                py,
                &q,
                &k,
                &v,
                Some(&kpm),
                Some(true),
                Some(&attn_mask),
                None,
            )
            .expect("masked forward");
        let w = weights
            .expect("weights present")
            .tensor
            .to_vec()
            .expect("w");

        // Both masked positions must be ~0 simultaneously; only pos 1 survives.
        assert!(w[0] < 1e-3, "attn-masked pos 0 must be ~0, got {}", w[0]);
        assert!(
            w[2] < 1e-3,
            "key-padding-masked pos 2 must be ~0, got {}",
            w[2]
        );
        assert!(
            (w[1] - 1.0).abs() < 1e-3,
            "the single surviving pos 1 must keep ~all the mass, got {}",
            w[1]
        );
    });
}

// -----------------------------------------------------------------------
// Gradient-flow tests: forward -> loss(sum) -> backward -> assert every
// parameter's gradient is populated, non-zero, finite, and (for a spot
// checked parameter) matches a finite-difference oracle.
// -----------------------------------------------------------------------

#[test]
fn multihead_attention_all_parameters_receive_finite_nonzero_gradients() {
    Python::initialize();
    Python::attach(|py| {
        // Small shapes to keep the finite-difference oracle cheap:
        // embed_dim=4, num_heads=2, batch=2, seq=2 (self-attention: q=k=v).
        let mha =
            PyMultiheadAttention::new(py, 4, 2, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        let x = make_tensor(ramp(2 * 2 * 4), &[2, 2, 4]);

        let (out, _weights) = mha
            .forward(py, &x, &x, &x, None, Some(false), None, None)
            .expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let params = mha.parameters(py);
        assert_eq!(
            params.len(),
            5,
            "expected [q_proj, k_proj, v_proj, out_proj, bias] since bias=true by default"
        );

        for (label, handle) in ["q_proj", "k_proj", "v_proj", "out_proj", "bias"]
            .into_iter()
            .zip(params.iter())
        {
            let grad = handle
                .borrow(py)
                .grad()
                .unwrap_or_else(|e| panic!("{label} gradient must be populated: {e}"));
            let grad_vec = grad.tensor.to_vec().expect("grad readable");
            assert!(
                grad_vec.iter().any(|&g| g != 0.0),
                "{label} grad must be non-zero: {grad_vec:?}"
            );
            assert!(
                grad_vec.iter().all(|g| g.is_finite()),
                "{label} grad must be finite: {grad_vec:?}"
            );
        }
    });
}

#[test]
fn multihead_attention_out_proj_gradient_matches_finite_difference() {
    Python::initialize();
    Python::attach(|py| {
        // embed_dim=4, num_heads=2, batch=1, seq=2: the smallest shape that
        // still exercises every head and both batch/seq positions.
        let mha =
            PyMultiheadAttention::new(py, 4, 2, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        let x = make_tensor(ramp(2 * 4), &[1, 2, 4]);

        let (out, _weights) = mha
            .forward(py, &x, &x, &x, None, Some(false), None, None)
            .expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let params = mha.parameters(py);
        let out_proj_handle = &params[3]; // [q_proj, k_proj, v_proj, out_proj, bias]
        let grad_out_proj = out_proj_handle
            .borrow(py)
            .grad()
            .expect("out_proj grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");

        // Snapshot every OTHER parameter's current value so the
        // finite-difference oracle can hold them fixed while perturbing only
        // out_proj_weight, re-running the REAL forward computation for each
        // perturbation (not a simplified stand-in).
        let q_vals = params[0].borrow(py).to_tensor().expect("to_tensor");
        let k_vals = params[1].borrow(py).to_tensor().expect("to_tensor");
        let v_vals = params[2].borrow(py).to_tensor().expect("to_tensor");
        let out_proj_vals = out_proj_handle
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let bias_vals = params[4].borrow(py).to_tensor().expect("to_tensor");

        let out_proj_shape = out_proj_handle.borrow(py).shape();
        let x_data = x.tensor.to_vec().expect("x readable");

        // Re-run the REAL forward pass (a fresh `PyMultiheadAttention` sharing
        // every parameter except a perturbed out_proj_weight) for each
        // finite-difference probe.
        let loss_fn = |out_proj_flat: &[f32]| -> f32 {
            Python::attach(|py2| {
                let probe = PyMultiheadAttention::new(
                    py2,
                    4,
                    2,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(true),
                )
                .expect("probe construction");
                probe
                    .q_proj_param
                    .borrow(py2)
                    .set_data((*q_vals.tensor).clone())
                    .expect("set_data");
                probe
                    .k_proj_param
                    .borrow(py2)
                    .set_data((*k_vals.tensor).clone())
                    .expect("set_data");
                probe
                    .v_proj_param
                    .borrow(py2)
                    .set_data((*v_vals.tensor).clone())
                    .expect("set_data");
                let out_proj_tensor =
                    Tensor::from_vec(out_proj_flat.to_vec(), &out_proj_shape).expect("out_proj");
                probe
                    .out_proj_param
                    .borrow(py2)
                    .set_data(out_proj_tensor)
                    .expect("set_data");
                if let Some(ref bp) = probe.bias_param {
                    bp.borrow(py2)
                        .set_data((*bias_vals.tensor).clone())
                        .expect("set_data");
                }

                let probe_x = make_tensor(x_data.clone(), &[1, 2, 4]);
                let (probe_out, _) = probe
                    .forward(
                        py2,
                        &probe_x,
                        &probe_x,
                        &probe_x,
                        None,
                        Some(false),
                        None,
                        None,
                    )
                    .expect("probe forward");
                probe_out.tensor.to_vec().expect("readable").iter().sum()
            })
        };

        let fd_grad = finite_difference_grad(&out_proj_vals, 1e-3, loss_fn);
        assert_close(
            &grad_out_proj,
            &fd_grad,
            5e-2,
            "multihead_attention out_proj grad vs finite-difference",
        );
    });
}

#[test]
fn multihead_attention_gradients_flow_with_non_batch_first_layout() {
    Python::initialize();
    Python::attach(|py| {
        // Same as the all-parameters test above but with batch_first=false,
        // exercising the seq-first slicing/stacking branch of forward().
        let mha =
            PyMultiheadAttention::new(py, 4, 2, None, None, None, None, None, None, Some(false))
                .expect("mha construction");
        // batch_first=false: x is [seq=2, batch=2, embed=4].
        let x = make_tensor(ramp(2 * 2 * 4), &[2, 2, 4]);

        let (out, _weights) = mha
            .forward(py, &x, &x, &x, None, Some(false), None, None)
            .expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 2, 4]);

        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        for handle in mha.parameters(py) {
            let grad = handle.borrow(py).grad().expect("grad must be populated");
            let grad_vec = grad.tensor.to_vec().expect("grad readable");
            assert!(grad_vec.iter().any(|&g| g != 0.0), "grad must be non-zero");
            assert!(
                grad_vec.iter().all(|g| g.is_finite()),
                "grad must be finite"
            );
        }
    });
}

// -----------------------------------------------------------------------
// `.parameters()` identity-sharing test, mirroring `layers.rs`'s
// `parameters_share_identity_with_forward_across_an_optimizer_update_cycle`.
// -----------------------------------------------------------------------

#[test]
fn parameters_share_identity_with_forward_and_survive_set_data() {
    Python::initialize();
    Python::attach(|py| {
        let mha =
            PyMultiheadAttention::new(py, 4, 2, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        let x = make_tensor(ramp(2 * 4), &[1, 2, 4]);

        let params_before = mha.parameters(py);
        assert_eq!(params_before.len(), 5);
        let ids_before: Vec<usize> = params_before.iter().map(|p| p.borrow(py).id()).collect();

        let (out1, _) = mha
            .forward(py, &x, &x, &x, None, Some(false), None, None)
            .expect("first forward");
        let out1_vec = out1.tensor.to_vec().expect("out1 readable");

        // Zero out the out_proj weight via `.set_data()` through a
        // `.parameters()` handle (index 3: [q, k, v, out_proj, bias]).
        let out_proj_handle = &params_before[3];
        let out_proj_shape = out_proj_handle.borrow(py).shape();
        let zeroed = Tensor::<f32>::zeros(&out_proj_shape);
        out_proj_handle
            .borrow(py)
            .set_data(zeroed)
            .expect("set_data must succeed");

        // The parameter's own id must be unchanged across set_data — this is
        // the actual property under test.
        assert_eq!(
            out_proj_handle.borrow(py).id(),
            ids_before[3],
            "out_proj parameter id must be stable across set_data()"
        );

        // forward() on the SAME layer instance must now reflect the update:
        // with out_proj zeroed, the final output projection collapses to
        // just the (shared) bias, broadcast identically across every
        // position — so the output must differ from the pre-update run.
        let (out2, _) = mha
            .forward(py, &x, &x, &x, None, Some(false), None, None)
            .expect("second forward (after set_data)");
        let out2_vec = out2.tensor.to_vec().expect("out2 readable");

        assert_eq!(out1_vec.len(), out2_vec.len());
        let changed = out1_vec
            .iter()
            .zip(out2_vec.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            changed,
            "zeroing out_proj via a parameters() handle must change forward()'s output \
             on this SAME layer instance: out1={out1_vec:?}, out2={out2_vec:?}"
        );

        // With out_proj zeroed, every output position must now equal the
        // shared bias exactly (out_proj contributes nothing).
        let bias_vals = params_before[4]
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        for pos in 0..(out2_vec.len() / bias_vals.len()) {
            let row = &out2_vec[pos * bias_vals.len()..(pos + 1) * bias_vals.len()];
            assert_close(
                row,
                &bias_vals,
                1e-4,
                "with out_proj zeroed, every output row must equal the shared bias exactly",
            );
        }

        // Every parameter id is stable across the whole cycle.
        let params_after = mha.parameters(py);
        let ids_after: Vec<usize> = params_after.iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(ids_before, ids_after, "every parameter id must be stable");
    });
}

#[test]
fn clone_produces_independent_parameter_identities() {
    Python::initialize();
    Python::attach(|py| {
        let mha =
            PyMultiheadAttention::new(py, 4, 2, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        let cloned = mha.clone();

        let original_ids: Vec<usize> = mha
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
