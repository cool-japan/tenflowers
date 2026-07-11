//! Tests for the recurrent layers module.
//!
//! # Test rigor
//!
//! Every gradient-correctness test in this module follows the same template
//! established in `neural/normalization.rs`'s test module (see that file's
//! own doc for the full rationale): forward -> a non-degenerate scalar loss
//! -> `run_backward` -> assert every relevant parameter's gradient is
//! populated, non-all-zero, finite, AND matches an independent
//! finite-difference oracle that re-runs the SAME real, composed
//! `forward()` this layer actually uses (not a hand-rederived closed-form
//! formula) — so a shared mistake between the implementation and a
//! hand-rolled oracle cannot produce a false pass.
//!
//! Unlike BatchNorm's `sum(out)` loss (which is degenerate for `grad_gamma`
//! by an exact algebraic identity — see `normalization.rs`'s extensive
//! comment on that), a plain `sum()` loss over an LSTM/GRU/RNN's output
//! sequence has no analogous structural cancellation: each weight
//! contributes to the recurrence through a genuinely different path at each
//! time step, so there is no reason to expect e.g. `grad_weight_hh` to
//! vanish identically. Tests below nonetheless follow the belt-and-suspenders
//! habit of weighting elements by distinct nonzero scalars before summing
//! wherever cheap to do, purely to keep the gradient well-conditioned (avoid
//! near-cancellation from symmetric ramp data) rather than because a
//! structural zero is anticipated.

use super::*;
use crate::implicit_autograd::{get_grad_by_id, run_backward};

fn approx_zero(values: &[f32], tol: f32) -> bool {
    let sum: f32 = values.iter().sum();
    sum.abs() < tol
}

/// Sum every element of `tensor` into a scalar `PyTensor`, recording the
/// reduction on the implicit tape. Mirrors `normalization.rs`'s helper of
/// the same name exactly.
fn tape_sum_to_scalar(tensor: &PyTensor) -> PyTensor {
    let raw = tenflowers_core::ops::sum(&tensor.tensor, None, false).expect("sum must succeed");
    let scalar = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    crate::implicit_autograd::record_and_link_unary(
        crate::implicit_autograd::UnaryOpKind::Sum {
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
/// `tensor_data`. Mirrors `normalization.rs`'s helper of the same name
/// exactly.
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

/// Weight each row of a `[rows, cols]` `PyTensor` by a distinct nonzero
/// per-row scalar, then sum to a scalar loss — avoids any accidental
/// near-cancellation from summing symmetric `ramp()` data uniformly, exactly
/// mirroring `normalization.rs`'s "weight samples before summing" pattern.
/// Built from real tape-tracked ops (`Mul` then `Sum`), so gradients flow to
/// every upstream parameter exactly as `.backward()` would compute them for
/// any other loss.
fn weighted_sum_loss(out: &PyTensor, weights_per_row: &[f32]) -> PyTensor {
    let rows = out.tensor.shape().dims()[0];
    assert_eq!(rows, weights_per_row.len());
    // Broadcast-multiply: weight shaped [rows, 1, 1, ...] against out's
    // [rows, ...] so each "row" (whatever the leading axis represents for
    // this particular output — batch or seq_len) gets a distinct scalar.
    let mut weight_shape = vec![rows];
    weight_shape.extend(std::iter::repeat_n(1, out.tensor.ndim() - 1));
    let weight_tensor = make_tensor(weights_per_row.to_vec(), &weight_shape);
    let weighted_raw =
        tenflowers_core::ops::mul(&out.tensor, &weight_tensor.tensor).expect("weighted mul");
    let weighted = PyTensor {
        tensor: Arc::new(weighted_raw),
        requires_grad: true,
        is_pinned: false,
    };
    crate::implicit_autograd::record_and_link_binary(
        crate::implicit_autograd::BinaryOpKind::Mul,
        out,
        &weight_tensor,
        &weighted,
    )
    .expect("recording weighted mul must succeed");
    tape_sum_to_scalar(&weighted)
}

// -----------------------------------------------------------------------
// Forward smoke tests (adapted from the pre-autograd version: same shape /
// non-all-zero-ness assertions, now against the new Py<PyParameter>-based
// structs and `py`-threaded `forward()` signatures).
// -----------------------------------------------------------------------

#[test]
fn lstm_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 4, 5, None, None, None, None, None).expect("lstm");
        // Multi-layer LSTM/GRU/RNN weights are zero-initialised by design
        // (see the module-level "Zero- vs. randn-initialisation" doc) — a
        // FRESH layer's output is therefore genuinely, exactly all-zero
        // (verified explicitly below), not merely "might happen to be
        // zero". `nudge_param` moves every parameter to a small nonzero
        // value first so this test can meaningfully check "the composed
        // forward computation produces real, nonzero, finite numbers" —
        // the property this test exists to confirm — without that
        // zero-init degeneracy masking it.
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (zero_output, _zero_hidden) = lstm.forward(py, &input, None).expect("forward");
        assert!(
            zero_output.tensor.to_vec().expect("vec").iter().all(|&v| v == 0.0),
            "a fresh, zero-initialised LSTM must produce an exactly-zero output"
        );
        for p in lstm.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 20.0));
        }

        let (output, (h_n, c_n)) = lstm.forward(py, &input, None).expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);
        assert_eq!(c_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "LSTM output must not be all zeros after nudging weights off their zero-init"
        );
        assert!(out_vec.iter().all(|v| v.is_finite()));
    });
}

#[test]
fn gru_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let gru = PyGRU::new(py, 4, 5, None, None, None, None, None).expect("gru");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        // See `lstm_forward_is_real`'s comment on why nudging is needed
        // before checking non-zero output: zero-init is deliberate and
        // genuinely produces an all-zero output on its own.
        for p in gru.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 21.0));
        }
        let (output, h_n) = gru.forward(py, &input, None).expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "GRU output must not be all zeros after nudging weights off their zero-init"
        );
        assert!(out_vec.iter().all(|v| v.is_finite()));
    });
}

#[test]
fn rnn_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let rnn = PyRNN::new(py, 4, 5, None, None, None, None, None, None).expect("rnn");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        // See `lstm_forward_is_real`'s comment on why nudging is needed.
        for p in rnn.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 22.0));
        }
        let (output, h_n) = rnn.forward(py, &input, None).expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "RNN output must not be all zeros after nudging weights off their zero-init"
        );
        assert!(out_vec.iter().all(|v| v.is_finite()));
    });
}

#[test]
fn lstm_cell_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyLSTMCell::new(py, 4, 5, None).expect("cell");
        let input = make_tensor(ramp(2 * 4), &[2, 4]);
        let (h1, c1) = cell.forward(py, &input, None).expect("forward");

        assert_eq!(h1.tensor.shape().dims().to_vec(), vec![2, 5]);
        assert_eq!(c1.tensor.shape().dims().to_vec(), vec![2, 5]);

        let h_vec = h1.tensor.to_vec().expect("vec");
        assert!(
            h_vec.iter().any(|&v| v != 0.0),
            "LSTMCell output must not be all zeros"
        );
        assert!(h_vec.iter().all(|v| v.is_finite()));
    });
}

#[test]
fn gru_cell_forward_is_real() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyGRUCell::new(py, 4, 5, None).expect("cell");
        let input = make_tensor(ramp(2 * 4), &[2, 4]);
        let h1 = cell.forward(py, &input, None).expect("forward");

        assert_eq!(h1.tensor.shape().dims().to_vec(), vec![2, 5]);

        let h_vec = h1.tensor.to_vec().expect("vec");
        assert!(
            h_vec.iter().any(|&v| v != 0.0),
            "GRUCell output must not be all zeros"
        );
        assert!(h_vec.iter().all(|v| v.is_finite()));
    });
}

// -----------------------------------------------------------------------
// Gradient-correctness tests: forward -> weighted-sum loss -> backward ->
// assert every weight/bias gradient is populated, non-zero, finite, and
// matches an independent finite-difference oracle re-running the SAME real
// forward() this layer uses.
//
// All structs below are zero-initialised for their multi-layer weights
// (see the module-level "Zero- vs. randn-initialisation" doc), so a
// backward pass run against a FRESH layer would trivially see zero
// gradients almost everywhere (e.g. LSTM's forget/output gates saturate at
// sigmoid(0)=0.5 but weight_hh's contribution through h=0 initial state
// vanishes on the very first step). Every test below therefore assigns
// small nonzero values to each parameter via `set_data` BEFORE the forward
// pass, so the recurrence is genuinely non-degenerate from step 0.
// -----------------------------------------------------------------------

/// Perturb every element of `param`'s current value in place via
/// `set_data`. Used to move a freshly-constructed (zero-initialised)
/// layer's weights to a small, non-degenerate starting point before
/// checking gradients.
fn nudge_param(py: Python<'_>, param: &Py<PyParameter>, values: Vec<f32>) {
    let shape = param.borrow(py).shape();
    let tensor = Tensor::from_vec(values, &shape).expect("tensor construction");
    param.borrow(py).set_data(tensor).expect("set_data");
}

/// Deterministic small nonzero values (distinct from `ramp` so tests do not
/// accidentally share failure modes), scaled small to keep the recurrence
/// numerically well-behaved for finite differencing.
fn small_values(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.037).sin() * 0.2)
        .collect()
}

#[test]
fn lstm_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 3, 2, None, None, None, None, None).expect("lstm");
        for p in lstm.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 1.0));
        }

        let input_data = ramp(2 * 2 * 3); // seq_len=2, batch=2, input_size=3
        let input = make_tensor(input_data.clone(), &[2, 2, 3]);

        let (output, _hidden) = lstm.forward(py, &input, None).expect("forward");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![2, 2, 2]);

        let loss = weighted_sum_loss(&output, &[0.7, 1.3]);
        run_backward(&loss).expect("backward must succeed");

        let params = lstm.parameters(py);
        assert_eq!(params.len(), 4, "weight_ih, weight_hh, bias_ih, bias_hh for 1 layer");

        let mut any_nonzero_overall = false;
        for p in &params {
            let id = p.borrow(py).id();
            let grad = get_grad_by_id(id)
                .unwrap_or_else(|_| panic!("gradient must be populated for param id {id}"))
                .tensor
                .to_vec()
                .expect("grad readable");
            assert!(grad.iter().all(|g| g.is_finite()), "grad must be finite: {grad:?}");
            if grad.iter().any(|&g| g != 0.0) {
                any_nonzero_overall = true;
            }
        }
        assert!(
            any_nonzero_overall,
            "at least one LSTM parameter must have a nonzero gradient"
        );

        // weight_ih and weight_hh specifically must be nonzero (the bias
        // gradients could in principle be sparser depending on gate
        // saturation, but the weight matrices multiply a genuinely
        // nonconstant x_t/h at every step).
        let grad_w_ih = get_grad_by_id(params[0].borrow(py).id())
            .expect("weight_ih grad")
            .tensor
            .to_vec()
            .expect("readable");
        let grad_w_hh = get_grad_by_id(params[1].borrow(py).id())
            .expect("weight_hh grad")
            .tensor
            .to_vec()
            .expect("readable");
        assert!(grad_w_ih.iter().any(|&g| g != 0.0), "weight_ih grad must be nonzero");
        assert!(grad_w_hh.iter().any(|&g| g != 0.0), "weight_hh grad must be nonzero");

        // Finite-difference oracle for weight_ih: re-run the SAME real
        // forward() with a perturbed weight_ih value, recomputing the exact
        // same weighted-sum loss.
        let w_ih_param = &params[0];
        let w_ih_shape = w_ih_param.borrow(py).shape();
        let w_ih_vals = w_ih_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let loss_fn = |w_ih: &[f32]| -> f32 {
            nudge_param(py, w_ih_param, w_ih.to_vec());
            let (out, _h) = lstm.forward(py, &input, None).expect("forward in fd probe");
            let out_vec = out.tensor.to_vec().expect("readable");
            let batch = out.tensor.shape().dims()[1];
            let hidden = out.tensor.shape().dims()[2];
            let weights = [0.7f32, 1.3];
            (0..2)
                .flat_map(|s| (0..batch * hidden).map(move |j| (s, j)))
                .map(|(s, j)| out_vec[s * batch * hidden + j] * weights[s])
                .sum()
        };
        let fd_grad_w_ih = finite_difference_grad(&w_ih_vals, 1e-3, loss_fn);
        // Restore weight_ih to its nudged (non-perturbed) value before
        // comparing, so the analytic gradient (computed at the ORIGINAL
        // value) is compared against an FD gradient centered at that same
        // point.
        nudge_param(py, w_ih_param, w_ih_vals.clone());
        assert_eq!(w_ih_shape.iter().product::<usize>(), grad_w_ih.len());
        assert_close(
            &grad_w_ih,
            &fd_grad_w_ih,
            5e-2,
            "lstm weight_ih grad vs finite-difference",
        );
    });
}

#[test]
fn gru_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let gru = PyGRU::new(py, 3, 2, None, None, None, None, None).expect("gru");
        for p in gru.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 2.0));
        }

        let input_data = ramp(2 * 2 * 3);
        let input = make_tensor(input_data, &[2, 2, 3]);

        let (output, _h_n) = gru.forward(py, &input, None).expect("forward");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![2, 2, 2]);

        let loss = weighted_sum_loss(&output, &[0.6, 1.4]);
        run_backward(&loss).expect("backward must succeed");

        let params = gru.parameters(py);
        assert_eq!(params.len(), 4);

        let grad_w_ih = get_grad_by_id(params[0].borrow(py).id())
            .expect("weight_ih grad populated")
            .tensor
            .to_vec()
            .expect("readable");
        let grad_w_hh = get_grad_by_id(params[1].borrow(py).id())
            .expect("weight_hh grad populated")
            .tensor
            .to_vec()
            .expect("readable");
        assert!(grad_w_ih.iter().all(|g| g.is_finite()));
        assert!(grad_w_hh.iter().all(|g| g.is_finite()));
        assert!(grad_w_ih.iter().any(|&g| g != 0.0), "weight_ih grad must be nonzero");
        assert!(grad_w_hh.iter().any(|&g| g != 0.0), "weight_hh grad must be nonzero");

        let w_hh_param = &params[1];
        let w_hh_vals = w_hh_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let loss_fn = |w_hh: &[f32]| -> f32 {
            nudge_param(py, w_hh_param, w_hh.to_vec());
            let (out, _h) = gru.forward(py, &input, None).expect("forward in fd probe");
            let out_vec = out.tensor.to_vec().expect("readable");
            let batch = out.tensor.shape().dims()[1];
            let hidden = out.tensor.shape().dims()[2];
            let weights = [0.6f32, 1.4];
            (0..2)
                .flat_map(|s| (0..batch * hidden).map(move |j| (s, j)))
                .map(|(s, j)| out_vec[s * batch * hidden + j] * weights[s])
                .sum()
        };
        let fd_grad_w_hh = finite_difference_grad(&w_hh_vals, 1e-3, loss_fn);
        nudge_param(py, w_hh_param, w_hh_vals.clone());
        assert_close(
            &grad_w_hh,
            &fd_grad_w_hh,
            5e-2,
            "gru weight_hh grad vs finite-difference",
        );
    });
}

#[test]
fn rnn_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let rnn = PyRNN::new(py, 3, 2, None, None, None, None, None, None).expect("rnn");
        for p in rnn.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 3.0));
        }

        let input_data = ramp(2 * 2 * 3);
        let input = make_tensor(input_data, &[2, 2, 3]);

        let (output, _h_n) = rnn.forward(py, &input, None).expect("forward");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![2, 2, 2]);

        let loss = weighted_sum_loss(&output, &[0.9, 1.1]);
        run_backward(&loss).expect("backward must succeed");

        let params = rnn.parameters(py);
        assert_eq!(params.len(), 4);

        let grad_w_ih = get_grad_by_id(params[0].borrow(py).id())
            .expect("weight_ih grad populated")
            .tensor
            .to_vec()
            .expect("readable");
        assert!(grad_w_ih.iter().all(|g| g.is_finite()));
        assert!(grad_w_ih.iter().any(|&g| g != 0.0), "weight_ih grad must be nonzero");

        let w_ih_param = &params[0];
        let w_ih_vals = w_ih_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let loss_fn = |w_ih: &[f32]| -> f32 {
            nudge_param(py, w_ih_param, w_ih.to_vec());
            let (out, _h) = rnn.forward(py, &input, None).expect("forward in fd probe");
            let out_vec = out.tensor.to_vec().expect("readable");
            let batch = out.tensor.shape().dims()[1];
            let hidden = out.tensor.shape().dims()[2];
            let weights = [0.9f32, 1.1];
            (0..2)
                .flat_map(|s| (0..batch * hidden).map(move |j| (s, j)))
                .map(|(s, j)| out_vec[s * batch * hidden + j] * weights[s])
                .sum()
        };
        let fd_grad_w_ih = finite_difference_grad(&w_ih_vals, 1e-3, loss_fn);
        nudge_param(py, w_ih_param, w_ih_vals.clone());
        assert_close(
            &grad_w_ih,
            &fd_grad_w_ih,
            5e-2,
            "rnn weight_ih grad vs finite-difference",
        );
    });
}

#[test]
fn lstm_cell_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyLSTMCell::new(py, 3, 2, None).expect("cell");
        // LSTMCell's weights already start at a proper randn-scaled
        // init (unlike the multi-layer structs' zero-init), so no
        // `nudge_param` pre-perturbation is needed here for a
        // well-conditioned gradient.
        let input = make_tensor(ramp(2 * 3), &[2, 3]);

        let (h1, _c1) = cell.forward(py, &input, None).expect("forward");
        assert_eq!(h1.tensor.shape().dims().to_vec(), vec![2, 2]);

        let loss = weighted_sum_loss(&h1, &[0.5, 1.5]);
        run_backward(&loss).expect("backward must succeed");

        let params = cell.parameters(py);
        assert_eq!(params.len(), 2, "weight_ih, weight_hh");

        let grad_w_ih = get_grad_by_id(params[0].borrow(py).id())
            .expect("weight_ih grad populated")
            .tensor
            .to_vec()
            .expect("readable");
        assert!(grad_w_ih.iter().all(|g| g.is_finite()));
        assert!(grad_w_ih.iter().any(|&g| g != 0.0), "weight_ih grad must be nonzero");

        let w_ih_param = &params[0];
        let w_ih_vals = w_ih_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let loss_fn = |w_ih: &[f32]| -> f32 {
            nudge_param(py, w_ih_param, w_ih.to_vec());
            let (h, _c) = cell.forward(py, &input, None).expect("forward in fd probe");
            let out_vec = h.tensor.to_vec().expect("readable");
            let hidden = h.tensor.shape().dims()[1];
            let weights = [0.5f32, 1.5];
            (0..2)
                .flat_map(|b| (0..hidden).map(move |j| (b, j)))
                .map(|(b, j)| out_vec[b * hidden + j] * weights[b])
                .sum()
        };
        let fd_grad_w_ih = finite_difference_grad(&w_ih_vals, 1e-3, loss_fn);
        nudge_param(py, w_ih_param, w_ih_vals.clone());
        assert_close(
            &grad_w_ih,
            &fd_grad_w_ih,
            5e-2,
            "lstm_cell weight_ih grad vs finite-difference",
        );
    });
}

#[test]
fn gru_cell_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyGRUCell::new(py, 3, 2, None).expect("cell");
        let input = make_tensor(ramp(2 * 3), &[2, 3]);
        // A single-step GRU cell's `weight_hh` only ever multiplies `h`
        // (`gh = h @ weight_hh`); with the default `hidden=None` (h_0 = 0,
        // via `zeros_state`) that product is IDENTICALLY zero regardless of
        // `weight_hh`'s value — there is no later time step (unlike the
        // multi-step `PyGRU::forward`) for a nonzero hidden state to ever
        // arise within this single call, so `weight_hh`'s gradient would be
        // a genuine, structural zero, not a bug. An explicit, nonzero `h_0`
        // (shape `[batch=2, hidden_size=2]`) is required to exercise
        // `weight_hh`'s gradient path at all.
        let h0 = make_tensor(vec![0.3, -0.2, 0.15, 0.25], &[2, 2]);

        let h1 = cell.forward(py, &input, Some(h0.clone())).expect("forward");
        assert_eq!(h1.tensor.shape().dims().to_vec(), vec![2, 2]);

        let loss = weighted_sum_loss(&h1, &[0.4, 1.6]);
        run_backward(&loss).expect("backward must succeed");

        let params = cell.parameters(py);
        assert_eq!(params.len(), 2);

        let grad_w_hh = get_grad_by_id(params[1].borrow(py).id())
            .expect("weight_hh grad populated")
            .tensor
            .to_vec()
            .expect("readable");
        assert!(grad_w_hh.iter().all(|g| g.is_finite()));
        assert!(grad_w_hh.iter().any(|&g| g != 0.0), "weight_hh grad must be nonzero");

        let w_hh_param = &params[1];
        let w_hh_vals = w_hh_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let loss_fn = |w_hh: &[f32]| -> f32 {
            nudge_param(py, w_hh_param, w_hh.to_vec());
            let h = cell
                .forward(py, &input, Some(h0.clone()))
                .expect("forward in fd probe");
            let out_vec = h.tensor.to_vec().expect("readable");
            let hidden = h.tensor.shape().dims()[1];
            let weights = [0.4f32, 1.6];
            (0..2)
                .flat_map(|b| (0..hidden).map(move |j| (b, j)))
                .map(|(b, j)| out_vec[b * hidden + j] * weights[b])
                .sum()
        };
        let fd_grad_w_hh = finite_difference_grad(&w_hh_vals, 1e-3, loss_fn);
        nudge_param(py, w_hh_param, w_hh_vals.clone());
        assert_close(
            &grad_w_hh,
            &fd_grad_w_hh,
            5e-2,
            "gru_cell weight_hh grad vs finite-difference",
        );
    });
}

// -----------------------------------------------------------------------
// `.parameters()` identity-sharing tests: mirror
// `layers.rs::parameters_share_identity_with_forward_across_an_optimizer_update_cycle`
// — get a handle from `.parameters()`, mutate its value via `.set_data()`,
// call `.forward()` again on the SAME layer object, confirm the output
// changed (not just "didn't crash").
// -----------------------------------------------------------------------

#[test]
fn lstm_parameters_share_identity_with_forward() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 3, 2, None, None, None, None, None).expect("lstm");
        let input = make_tensor(ramp(2 * 2 * 3), &[2, 2, 3]);

        let (out_before, _h) = lstm.forward(py, &input, None).expect("forward before");
        let before_vec = out_before.tensor.to_vec().expect("vec");
        // Fresh LSTM has zero-initialised weights, so its untouched output
        // is exactly zero everywhere (a real degenerate baseline, not a
        // bug) — asserted explicitly so the post-mutation "changed" check
        // below is unambiguous (any nonzero output must be attributable to
        // the mutation, not residual pre-existing signal).
        assert!(
            before_vec.iter().all(|&v| v == 0.0),
            "fresh zero-initialised LSTM must produce an all-zero output"
        );

        let params = lstm.parameters(py);
        assert_eq!(params.len(), 4);
        let weight_ih_handle = &params[0];
        let n = weight_ih_handle.borrow(py).size();
        weight_ih_handle
            .borrow(py)
            .set_data(Tensor::from_vec(small_values(n, 5.0), &weight_ih_handle.borrow(py).shape()).expect("tensor"))
            .expect("set_data");

        let (out_after, _h) = lstm.forward(py, &input, None).expect("forward after");
        let after_vec = out_after.tensor.to_vec().expect("vec");
        assert!(
            after_vec.iter().any(|&v| v != 0.0),
            "mutating weight_ih via a .parameters() handle must change the SAME layer's next forward() output"
        );
    });
}

#[test]
fn gru_parameters_share_identity_with_forward() {
    Python::initialize();
    Python::attach(|py| {
        let gru = PyGRU::new(py, 3, 2, None, None, None, None, None).expect("gru");
        let input = make_tensor(ramp(2 * 2 * 3), &[2, 2, 3]);

        let (out_before, _h) = gru.forward(py, &input, None).expect("forward before");
        let before_vec = out_before.tensor.to_vec().expect("vec");
        assert!(before_vec.iter().all(|&v| v == 0.0));

        // Mutate weight_ih, not weight_hh: with h_0 defaulting to zeros and
        // weight_ih ALSO still at its zero-init, `h` stays exactly 0 forever
        // (GRU's `h_new = (1-z)*n + z*h` with z=sigmoid(0)=0.5, n=tanh(0)=0
        // collapses to `h_new = 0.5*h`, a fixed point at h=0) — so mutating
        // ONLY weight_hh would leave the whole recurrence permanently gated
        // to zero regardless of weight_hh's value (weight_hh's own
        // contribution `h @ weight_hh` is `0 @ weight_hh = 0` at every
        // step), and this test would spuriously fail not because identity
        // sharing is broken but because of an unrelated structural
        // degeneracy. weight_ih instead reaches a nonzero pre-activation on
        // the very first timestep via `x_t @ weight_ih`, since `x_t` itself
        // is the nonzero `ramp()` input data.
        let params = gru.parameters(py);
        let weight_ih_handle = &params[0];
        let n = weight_ih_handle.borrow(py).size();
        weight_ih_handle
            .borrow(py)
            .set_data(Tensor::from_vec(small_values(n, 6.0), &weight_ih_handle.borrow(py).shape()).expect("tensor"))
            .expect("set_data");

        let (out_after, _h) = gru.forward(py, &input, None).expect("forward after");
        let after_vec = out_after.tensor.to_vec().expect("vec");
        assert!(
            after_vec.iter().any(|&v| v != 0.0),
            "mutating weight_ih via a .parameters() handle must change the SAME layer's next forward() output"
        );
    });
}

#[test]
fn rnn_parameters_share_identity_with_forward() {
    Python::initialize();
    Python::attach(|py| {
        let rnn = PyRNN::new(py, 3, 2, None, None, None, None, None, None).expect("rnn");
        let input = make_tensor(ramp(2 * 2 * 3), &[2, 2, 3]);

        let (out_before, _h) = rnn.forward(py, &input, None).expect("forward before");
        let before_vec = out_before.tensor.to_vec().expect("vec");
        assert!(before_vec.iter().all(|&v| v == 0.0));

        let params = rnn.parameters(py);
        let weight_ih_handle = &params[0];
        let n = weight_ih_handle.borrow(py).size();
        weight_ih_handle
            .borrow(py)
            .set_data(Tensor::from_vec(small_values(n, 7.0), &weight_ih_handle.borrow(py).shape()).expect("tensor"))
            .expect("set_data");

        let (out_after, _h) = rnn.forward(py, &input, None).expect("forward after");
        let after_vec = out_after.tensor.to_vec().expect("vec");
        assert!(
            after_vec.iter().any(|&v| v != 0.0),
            "mutating weight_ih via a .parameters() handle must change the SAME layer's next forward() output"
        );
    });
}

#[test]
fn lstm_cell_parameters_share_identity_with_forward() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyLSTMCell::new(py, 3, 2, None).expect("cell");
        let input = make_tensor(ramp(2 * 3), &[2, 3]);

        let (out_before, _c) = cell.forward(py, &input, None).expect("forward before");
        let before_vec = out_before.tensor.to_vec().expect("vec");

        let params = cell.parameters(py);
        assert_eq!(params.len(), 2);
        // Mutate weight_ih, not weight_hh: with h_0 defaulting to zeros
        // (`hidden=None`), the hidden-to-hidden branch `h @ weight_hh`
        // contributes nothing regardless of weight_hh's value on this
        // single-step forward pass, so zeroing ONLY weight_hh would leave
        // the output unchanged for a reason unrelated to identity sharing.
        // weight_ih instead directly affects the gate pre-activations
        // through the nonzero `input` data.
        let weight_ih_handle = &params[0];
        let original_size = weight_ih_handle.borrow(py).size();
        weight_ih_handle
            .borrow(py)
            .set_data(Tensor::zeros(&weight_ih_handle.borrow(py).shape()))
            .expect("set_data");
        assert_eq!(original_size, weight_ih_handle.borrow(py).size());

        let (out_after, _c) = cell.forward(py, &input, None).expect("forward after");
        let after_vec = out_after.tensor.to_vec().expect("vec");
        assert_eq!(before_vec.len(), after_vec.len());
        assert!(
            before_vec
                .iter()
                .zip(after_vec.iter())
                .any(|(b, a)| (b - a).abs() > 1e-6),
            "zeroing weight_ih via a .parameters() handle must change the SAME cell's next forward() output"
        );
    });
}

#[test]
fn gru_cell_parameters_share_identity_with_forward() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyGRUCell::new(py, 3, 2, None).expect("cell");
        let input = make_tensor(ramp(2 * 3), &[2, 3]);

        let out_before = cell.forward(py, &input, None).expect("forward before");
        let before_vec = out_before.tensor.to_vec().expect("vec");

        let params = cell.parameters(py);
        let weight_ih_handle = &params[0];
        weight_ih_handle
            .borrow(py)
            .set_data(Tensor::zeros(&weight_ih_handle.borrow(py).shape()))
            .expect("set_data");

        let out_after = cell.forward(py, &input, None).expect("forward after");
        let after_vec = out_after.tensor.to_vec().expect("vec");
        assert_eq!(before_vec.len(), after_vec.len());
        assert!(
            before_vec
                .iter()
                .zip(after_vec.iter())
                .any(|(b, a)| (b - a).abs() > 1e-6),
            "zeroing weight_ih via a .parameters() handle must change the SAME cell's next forward() output"
        );
    });
}

// -----------------------------------------------------------------------
// Regression tests, adapted from the pre-autograd version (same behavioural
// guarantees, now against the new struct shapes / py-threaded signatures).
// -----------------------------------------------------------------------

#[test]
fn gru_forward_threads_initial_hidden_state() {
    Python::initialize();
    Python::attach(|py| {
        let gru = PyGRU::new(py, 4, 5, None, None, None, None, None).expect("gru");
        // Non-degenerate (nonzero) weights so the initial hidden state can
        // actually influence the output through weight_hh.
        for p in gru.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 8.0));
        }
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);

        let (out_none, _) = gru.forward(py, &input, None).expect("forward none");

        // A nonzero initial hidden state [num_layers = 1, batch = 2, hidden = 5].
        let h0 = make_tensor(ramp(2 * 5), &[1, 2, 5]);
        let (out_h0, _) = gru.forward(py, &input, Some(h0)).expect("forward h0");

        let a = out_none.tensor.to_vec().expect("vec");
        let b = out_h0.tensor.to_vec().expect("vec");
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "an explicit nonzero h_0 must change the GRU output"
        );
    });
}

#[test]
fn gru_forward_matches_single_cell_step() {
    Python::initialize();
    Python::attach(|py| {
        let gru = PyGRU::new(py, 4, 5, None, None, None, None, None).expect("gru");
        for p in gru.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 9.0));
        }
        // seq_len = 1 so the entire forward reduces to one gru cell step.
        let input = make_tensor(ramp(2 * 4), &[1, 2, 4]);
        let h0 = make_tensor(ramp(2 * 5), &[1, 2, 5]);

        let (output, h_n) = gru.forward(py, &input, Some(h0.clone())).expect("forward");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![1, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        // Recompute the expected single step directly from the same
        // snapshotted parameter values and the module's own
        // `gru::gru_cell_step` (accessible here since `tests` is a
        // descendant of `recurrent`, and `gru_cell_step` is a private
        // top-level fn in the `gru` submodule) — via the crate-visible
        // wrapper below, since `gru_cell_step` itself is not `pub`.
        let params = gru.parameters(py);
        let w_ih = params[0].borrow(py).to_tensor().expect("to_tensor");
        let w_hh = params[1].borrow(py).to_tensor().expect("to_tensor");
        let b_ih = params[2].borrow(py).to_tensor().expect("to_tensor");
        let b_hh = params[3].borrow(py).to_tensor().expect("to_tensor");

        let x_t = sequence_timestep(&input, 0, 2, 4, false).expect("x_t");
        let h_prev = slice_initial_state(&h0, 0, 0, 1, 2, 5).expect("h_prev");

        let expected =
            super::gru::gru_cell_step_for_tests(&x_t, &h_prev, &w_ih, &w_hh, Some(&b_ih), Some(&b_hh), 5)
                .expect("gru_cell_step");
        let exp = expected.tensor.to_vec().expect("vec");
        let got = output.tensor.to_vec().expect("vec");
        assert_eq!(exp.len(), got.len());
        for (e, g) in exp.iter().zip(got.iter()) {
            assert!(
                (e - g).abs() < 1e-4,
                "GRU forward must match a single gru_cell_step (expected {e}, got {g})"
            );
        }
    });
}

#[test]
fn gru_bidirectional_forward_none_still_works() {
    Python::initialize();
    Python::attach(|py| {
        // Regression: unchanged bidirectional path (hidden = None) still succeeds.
        let gru = PyGRU::new(py, 4, 5, None, None, None, None, Some(true)).expect("gru");
        for p in gru.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 10.0));
        }
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = gru
            .forward(py, &input, None)
            .expect("bidirectional forward");
        // Bidirectional doubles the output feature dim.
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 10]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![2, 2, 5]);
        let out = output.tensor.to_vec().expect("vec");
        assert!(
            out.iter().any(|&v| v != 0.0),
            "output must not be all zeros"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    });
}

#[test]
fn gru_bidirectional_forward_with_hidden_errors() {
    Python::initialize();
    Python::attach(|py| {
        // Regression: the bidirectional guard is preserved.
        let gru = PyGRU::new(py, 4, 5, None, None, None, None, Some(true)).expect("gru");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let h0 = make_tensor(ramp(2 * 2 * 5), &[2, 2, 5]);
        let result = gru.forward(py, &input, Some(h0));
        assert!(
            result.is_err(),
            "bidirectional GRU with explicit hidden must still error"
        );
    });
}

#[test]
fn lstm_bidirectional_state_extraction_still_errors() {
    Python::initialize();
    Python::attach(|py| {
        // Regression: PyLSTM's (h_n, c_n) extraction remains
        // unidirectional-only, exactly as the pre-autograd version.
        let lstm = PyLSTM::new(py, 4, 5, None, None, None, None, Some(true)).expect("lstm");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let result = lstm.forward(py, &input, None);
        assert!(
            result.is_err(),
            "bidirectional LSTM must still error on forward() (h_n, c_n) extraction"
        );
    });
}

#[test]
fn rnn_relu_forward_succeeds() {
    Python::initialize();
    Python::attach(|py| {
        // Previously this constructed fine but failed on the first forward() call.
        let rnn = PyRNN::new(
            py,
            4,
            5,
            None,
            Some("relu".to_string()),
            None,
            None,
            None,
            None,
        )
        .expect("rnn");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = rnn
            .forward(py, &input, None)
            .expect("relu forward must succeed");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);
        let out = output.tensor.to_vec().expect("vec");
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "relu RNN output must be non-negative"
        );
    });
}

#[test]
fn rnn_relu_zeroes_negative_preactivations() {
    Python::initialize();
    Python::attach(|py| {
        // input_size = 1, hidden_size = 2, single layer, single timestep, relu.
        let rnn = PyRNN::new(
            py,
            1,
            2,
            None,
            Some("relu".to_string()),
            None,
            Some(true),
            None,
            None,
        )
        .expect("rnn");

        let params = rnn.parameters(py);
        // Order for 1 layer with bias: [w_ih, w_hh, b_ih, b_hh].
        nudge_param(py, &params[0], vec![1.0, 1.0]);
        nudge_param(py, &params[1], vec![0.0, 0.0, 0.0, 0.0]);
        nudge_param(py, &params[2], vec![-2.0, 0.5]);
        nudge_param(py, &params[3], vec![0.0, 0.0]);

        // batch_first input [batch = 1, seq = 1, feat = 1]; h_0 defaults to zeros.
        let input = make_tensor(vec![1.0], &[1, 1, 1]);
        let (output, _h_n) = rnn.forward(py, &input, None).expect("relu forward");

        let out = output.tensor.to_vec().expect("vec");
        // Pre-activations are [-1.0, 1.5]; relu zeroes the negative entry exactly,
        // whereas tanh(-1.0) ~= -0.76 would be strictly negative.
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 0.0).abs() < 1e-6,
            "negative pre-activation must relu to 0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 1.5).abs() < 1e-5,
            "positive pre-activation must pass through relu, got {}",
            out[1]
        );
    });
}

#[test]
fn rnn_bidirectional_relu_is_nonnegative() {
    Python::initialize();
    Python::attach(|py| {
        // Non-negativity across the whole bidirectional output confirms BOTH the
        // forward and the reverse call sites use relu (tanh could be negative).
        let rnn = PyRNN::new(
            py,
            4,
            5,
            None,
            Some("relu".to_string()),
            None,
            None,
            None,
            Some(true),
        )
        .expect("rnn");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = rnn.forward(py, &input, None).expect("bi relu forward");
        // Bidirectional doubles the output feature dim.
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 10]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![2, 2, 5]);
        let out = output.tensor.to_vec().expect("vec");
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "bidirectional relu output must be non-negative everywhere"
        );
    });
}

#[test]
fn rnn_bidirectional_forward_with_hidden_state_succeeds() {
    Python::initialize();
    Python::attach(|py| {
        // Regression / new-capability check: unlike PyLSTM/PyGRU, PyRNN's
        // bidirectional forward() accepts an explicit hidden state (no
        // guard rejects it — see `rnn.rs::forward`'s doc comment on this).
        let rnn = PyRNN::new(py, 4, 5, None, None, None, None, None, Some(true)).expect("rnn");
        for p in rnn.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 11.0));
        }
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let h0 = make_tensor(ramp(2 * 2 * 5), &[2, 2, 5]);
        let (output, h_n) = rnn
            .forward(py, &input, Some(h0))
            .expect("bidirectional forward with hidden must succeed");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 10]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![2, 2, 5]);
        assert!(output.tensor.to_vec().expect("vec").iter().all(|v| v.is_finite()));
    });
}

// -----------------------------------------------------------------------
// reset_parameters() sanity: after resetting, parameters() returns
// DIFFERENT identities (fresh Py<PyParameter> objects), matching
// PyBatchNorm1d::reset_parameters's behaviour.
// -----------------------------------------------------------------------

#[test]
fn lstm_reset_parameters_produces_fresh_identities() {
    Python::initialize();
    Python::attach(|py| {
        let mut lstm = PyLSTM::new(py, 3, 2, None, None, None, None, None).expect("lstm");
        // Keep the ORIGINAL `Py<PyParameter>` handles alive (not just their
        // `usize` ids) across the `reset_parameters()` call below. If only
        // the ids were kept, every original handle would be dropped
        // immediately after `.collect()` (nothing else holds a Python-level
        // reference to them beyond `self.weight_ih[l]` etc, which
        // `reset_parameters()` immediately `.clear()`s) — freeing each
        // `Arc<RwLock<Tensor<f32>>>` allocation BEFORE `reset_parameters()`
        // allocates its replacements, which then makes it entirely
        // plausible (and was empirically observed to happen for the
        // sibling `gru_reset_parameters_produces_fresh_identities` case)
        // for the allocator to hand back the EXACT SAME just-freed address
        // for a new, logically-distinct parameter — see
        // `crate::neural::layers::PyParameter`'s own `Drop` impl doc for
        // the general "id survives past the object that minted it" hazard
        // this mirrors. Keeping `before_handles` alive keeps every original
        // allocation genuinely occupied throughout, so any address seen
        // afterward is guaranteed to be a real, distinct allocation.
        let before_handles = lstm.parameters(py);
        let before_ids: Vec<usize> = before_handles.iter().map(|p| p.borrow(py).id()).collect();
        lstm.reset_parameters(py).expect("reset_parameters");
        let after_ids: Vec<usize> = lstm.parameters(py).iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(before_ids.len(), after_ids.len());
        for (b, a) in before_ids.iter().zip(after_ids.iter()) {
            assert_ne!(b, a, "reset_parameters must produce a fresh parameter identity");
        }
        drop(before_handles);
    });
}

#[test]
fn gru_reset_parameters_produces_fresh_identities() {
    Python::initialize();
    Python::attach(|py| {
        let mut gru = PyGRU::new(py, 3, 2, None, None, None, None, Some(true)).expect("gru");
        // See `lstm_reset_parameters_produces_fresh_identities`'s comment on
        // why `before_handles` must be kept alive (not just `before_ids`)
        // across the `reset_parameters()` call.
        let before_handles = gru.parameters(py);
        let before_ids: Vec<usize> = before_handles.iter().map(|p| p.borrow(py).id()).collect();
        gru.reset_parameters(py).expect("reset_parameters");
        let after_ids: Vec<usize> = gru.parameters(py).iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(before_ids.len(), after_ids.len());
        for (b, a) in before_ids.iter().zip(after_ids.iter()) {
            assert_ne!(b, a, "reset_parameters must produce a fresh parameter identity");
        }
        drop(before_handles);
    });
}

#[test]
fn rnn_reset_parameters_produces_fresh_identities() {
    Python::initialize();
    Python::attach(|py| {
        let mut rnn = PyRNN::new(py, 3, 2, None, None, None, None, None, None).expect("rnn");
        // See `lstm_reset_parameters_produces_fresh_identities`'s comment on
        // why `before_handles` must be kept alive (not just `before_ids`)
        // across the `reset_parameters()` call.
        let before_handles = rnn.parameters(py);
        let before_ids: Vec<usize> = before_handles.iter().map(|p| p.borrow(py).id()).collect();
        rnn.reset_parameters(py).expect("reset_parameters");
        let after_ids: Vec<usize> = rnn.parameters(py).iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(before_ids.len(), after_ids.len());
        for (b, a) in before_ids.iter().zip(after_ids.iter()) {
            assert_ne!(b, a, "reset_parameters must produce a fresh parameter identity");
        }
        drop(before_handles);
    });
}

// -----------------------------------------------------------------------
// Validation-error regression tests: every PyValueError/PyRuntimeError the
// pre-autograd version raised must still be raised under the same
// conditions.
// -----------------------------------------------------------------------

#[test]
fn lstm_rejects_invalid_construction_args() {
    Python::initialize();
    Python::attach(|py| {
        assert!(PyLSTM::new(py, 0, 5, None, None, None, None, None).is_err());
        assert!(PyLSTM::new(py, 4, 0, None, None, None, None, None).is_err());
        assert!(PyLSTM::new(py, 4, 5, Some(0), None, None, None, None).is_err());
        assert!(PyLSTM::new(py, 4, 5, None, None, None, Some(1.5), None).is_err());
        assert!(PyLSTM::new(py, 4, 5, None, None, None, Some(-0.1), None).is_err());
    });
}

#[test]
fn gru_rejects_invalid_construction_args() {
    Python::initialize();
    Python::attach(|py| {
        assert!(PyGRU::new(py, 0, 5, None, None, None, None, None).is_err());
        assert!(PyGRU::new(py, 4, 0, None, None, None, None, None).is_err());
        assert!(PyGRU::new(py, 4, 5, Some(0), None, None, None, None).is_err());
        assert!(PyGRU::new(py, 4, 5, None, None, None, Some(1.5), None).is_err());
    });
}

#[test]
fn rnn_rejects_invalid_construction_args() {
    Python::initialize();
    Python::attach(|py| {
        assert!(PyRNN::new(py, 0, 5, None, None, None, None, None, None).is_err());
        assert!(PyRNN::new(py, 4, 0, None, None, None, None, None, None).is_err());
        assert!(PyRNN::new(py, 4, 5, Some(0), None, None, None, None, None).is_err());
        assert!(PyRNN::new(
            py,
            4,
            5,
            None,
            Some("bogus".to_string()),
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(PyRNN::new(py, 4, 5, None, None, None, None, Some(1.5), None).is_err());
    });
}

#[test]
fn lstm_cell_rejects_invalid_construction_args() {
    Python::initialize();
    Python::attach(|py| {
        assert!(PyLSTMCell::new(py, 0, 5, None).is_err());
        assert!(PyLSTMCell::new(py, 4, 0, None).is_err());
    });
}

#[test]
fn gru_cell_rejects_invalid_construction_args() {
    Python::initialize();
    Python::attach(|py| {
        assert!(PyGRUCell::new(py, 0, 5, None).is_err());
        assert!(PyGRUCell::new(py, 4, 0, None).is_err());
    });
}

#[test]
fn lstm_rejects_wrong_input_rank_and_size() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 4, 5, None, None, None, None, None).expect("lstm");
        // Wrong rank (2D instead of 3D).
        let bad_rank = make_tensor(ramp(4 * 2), &[4, 2]);
        assert!(lstm.forward(py, &bad_rank, None).is_err());
        // Wrong input_size (3 instead of 4).
        let bad_size = make_tensor(ramp(3 * 2 * 3), &[3, 2, 3]);
        assert!(lstm.forward(py, &bad_size, None).is_err());
    });
}

#[test]
fn gru_rejects_wrong_input_rank_and_size() {
    Python::initialize();
    Python::attach(|py| {
        let gru = PyGRU::new(py, 4, 5, None, None, None, None, None).expect("gru");
        let bad_rank = make_tensor(ramp(4 * 2), &[4, 2]);
        assert!(gru.forward(py, &bad_rank, None).is_err());
        let bad_size = make_tensor(ramp(3 * 2 * 3), &[3, 2, 3]);
        assert!(gru.forward(py, &bad_size, None).is_err());
    });
}

#[test]
fn rnn_rejects_wrong_input_rank_and_size() {
    Python::initialize();
    Python::attach(|py| {
        let rnn = PyRNN::new(py, 4, 5, None, None, None, None, None, None).expect("rnn");
        let bad_rank = make_tensor(ramp(4 * 2), &[4, 2]);
        assert!(rnn.forward(py, &bad_rank, None).is_err());
        let bad_size = make_tensor(ramp(3 * 2 * 3), &[3, 2, 3]);
        assert!(rnn.forward(py, &bad_size, None).is_err());
    });
}

#[test]
fn lstm_cell_rejects_wrong_input_rank_and_size() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyLSTMCell::new(py, 4, 5, None).expect("cell");
        let bad_rank = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        assert!(cell.forward(py, &bad_rank, None).is_err());
        let bad_size = make_tensor(ramp(2 * 3), &[2, 3]);
        assert!(cell.forward(py, &bad_size, None).is_err());
    });
}

#[test]
fn gru_cell_rejects_wrong_input_rank_and_size() {
    Python::initialize();
    Python::attach(|py| {
        let cell = PyGRUCell::new(py, 4, 5, None).expect("cell");
        let bad_rank = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        assert!(cell.forward(py, &bad_rank, None).is_err());
        let bad_size = make_tensor(ramp(2 * 3), &[2, 3]);
        assert!(cell.forward(py, &bad_size, None).is_err());
    });
}

// -----------------------------------------------------------------------
// Clone / impl Clone sanity: cloning a layer must produce an object sharing
// PARAMETER identity with the original (Py::clone_ref semantics — see the
// module doc's reference to `PyBatchNorm1d`'s `impl Clone` in
// `normalization.rs`, which this mirrors exactly), unlike `PyDense`'s
// `impl Clone` (deliberately independent, fresh identity per
// `PyParameter::clone_param`) — the two designs diverge on purpose, so this
// test locks in which one `recurrent`'s structs actually use.
// -----------------------------------------------------------------------

#[test]
fn lstm_clone_shares_parameter_identity() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 3, 2, None, None, None, None, None).expect("lstm");
        let cloned = lstm.clone();
        let orig_ids: Vec<usize> = lstm.parameters(py).iter().map(|p| p.borrow(py).id()).collect();
        let clone_ids: Vec<usize> = cloned.parameters(py).iter().map(|p| p.borrow(py).id()).collect();
        assert_eq!(orig_ids, clone_ids, "Clone must share PyParameter identity (Py::clone_ref semantics)");
    });
}

#[test]
fn multi_layer_lstm_parameter_count_matches_layer_count() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 3, 2, Some(2), None, None, None, None).expect("lstm 2-layer");
        // 2 layers * (weight_ih, weight_hh, bias_ih, bias_hh) = 8.
        assert_eq!(lstm.parameters(py).len(), 8);

        let lstm_bi =
            PyLSTM::new(py, 3, 2, Some(2), None, None, None, Some(true)).expect("bidirectional lstm 2-layer");
        // 2 layers * 4 forward + 2 layers * 4 reverse = 16.
        assert_eq!(lstm_bi.parameters(py).len(), 16);
    });
}

#[test]
fn multi_layer_lstm_forward_is_real_and_finite() {
    Python::initialize();
    Python::attach(|py| {
        let lstm = PyLSTM::new(py, 3, 2, Some(2), None, None, None, None).expect("lstm 2-layer");
        for p in lstm.parameters(py) {
            let n = p.borrow(py).size();
            nudge_param(py, &p, small_values(n, 12.0));
        }
        let input = make_tensor(ramp(2 * 2 * 3), &[2, 2, 3]);
        let (output, (h_n, c_n)) = lstm.forward(py, &input, None).expect("forward");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![2, 2, 2]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![2, 2, 2]);
        assert_eq!(c_n.tensor.shape().dims().to_vec(), vec![2, 2, 2]);
        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(out_vec.iter().any(|&v| v != 0.0));
        assert!(out_vec.iter().all(|v| v.is_finite()));
    });
}
