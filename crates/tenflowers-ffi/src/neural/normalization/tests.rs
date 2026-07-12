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

fn approx_zero(values: &[f32], tol: f32) -> bool {
    let sum: f32 = values.iter().sum();
    sum.abs() < tol
}

/// Sum every element of `tensor` into a scalar `PyTensor`, recording the
/// reduction on the implicit tape — the `loss = sum(output)` step every
/// gradient test in this module needs before calling `run_backward`.
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

#[test]
fn batch_norm_normalizes_per_channel() {
    Python::initialize();
    Python::attach(|py| {
        let mut bn = PyBatchNorm1d::new(py, 3, None, None, None).expect("bn construction");
        // (N=4, C=3) with non-constant columns.
        let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let input = make_tensor(data, &[4, 3]);
        let out = bn.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![4, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
        // Each channel (column) is mean-centred after training-mode BatchNorm.
        for c in 0..3 {
            let col: Vec<f32> = (0..4).map(|n| out_vec[n * 3 + c]).collect();
            assert!(approx_zero(&col, 1e-3), "channel {c} should have mean 0");
        }
    });
}

#[test]
fn batch_norm_rejects_4d_input() {
    Python::initialize();
    Python::attach(|py| {
        let mut bn = PyBatchNorm1d::new(py, 2, None, None, None).expect("bn construction");
        let input = make_tensor(vec![0.0; 16], &[2, 2, 2, 2]);
        assert!(bn.forward(py, &input).is_err());
    });
}

#[test]
fn layer_norm_centres_last_dim() {
    Python::initialize();
    Python::attach(|py| {
        let ln = PyLayerNorm::new(py, vec![3], None).expect("ln construction");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = ln.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
        for row in 0..2 {
            let r: Vec<f32> = (0..3).map(|i| out_vec[row * 3 + i]).collect();
            assert!(approx_zero(&r, 1e-3), "row {row} should have mean 0");
        }
    });
}

#[test]
fn group_norm_forward_real_output() {
    Python::initialize();
    Python::attach(|py| {
        let gn = PyGroupNorm::new(py, 2, 4, None).expect("gn construction");
        // (N=1, C=4, L=2)
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let input = make_tensor(data, &[1, 4, 2]);
        let out = gn.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 4, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
    });
}

#[test]
fn instance_norm_centres_each_channel() {
    Python::initialize();
    Python::attach(|py| {
        let inorm = PyInstanceNorm1d::new(py, 2, None).expect("in construction");
        // (N=1, C=2, L=4)
        let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let input = make_tensor(data, &[1, 2, 4]);
        let out = inorm.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
        // Each channel is normalised over its own L dimension -> mean 0.
        assert!(approx_zero(&out_vec[0..4], 1e-3));
        assert!(approx_zero(&out_vec[4..8], 1e-3));
    });
}

// -----------------------------------------------------------------
// Gradient flow tests: forward -> loss(sum) -> backward -> assert
// gamma AND beta gradients are populated, non-zero, and numerically
// correct against a finite-difference oracle.
// -----------------------------------------------------------------

#[test]
fn batch_norm_training_mode_gamma_beta_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let mut bn = PyBatchNorm1d::new(py, 2, None, None, None).expect("bn construction");
        bn.train();
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = make_tensor(input_data.clone(), &[4, 2]);

        let out = bn.forward(py, &input).expect("forward");

        // NOTE on loss function: `loss = sum(out)` (this test's original
        // loss) makes training-mode grad_gamma IDENTICALLY ZERO as a pure
        // algebraic identity, independent of any implementation detail:
        // `grad_output` is then uniformly 1 across the batch per channel,
        // so `grad_gamma_c = sum_n(grad_output[n,c] * normalized[n,c])
        // = sum_n(normalized[n,c])`, and `normalized[n,c] =
        // (x[n,c] - batch_mean_c)/std_c` where `batch_mean_c` is exactly
        // `mean_n(x[n,c])` — so `sum_n(x[n,c] - batch_mean_c)
        // = sum_n(x[n,c]) - N*batch_mean_c = 0` exactly, for every
        // channel. Confirmed empirically (not just algebraically) via a
        // standalone finite-difference probe against the real,
        // unmodified `tenflowers_core::ops::batch_norm` kernel: FD
        // grad_gamma under `sum(out)` came back as
        // `[-5.960464e-5, -5.960464e-5]` (pure fp noise at the 1e-3
        // finite-difference step size), matching the tape's exact
        // `[0.0, 0.0]` output — this is a real degeneracy of the loss
        // choice, not a backward-pass bug.
        //
        // To exercise a genuinely non-zero, well-conditioned gamma
        // gradient, weight each sample by a distinct per-sample scalar
        // before summing: `loss = sum(out * per_sample_weight)` with
        // weight = [0.5, 1.0, 1.5, 2.0] (one per batch row, broadcast
        // over both channels). This makes `grad_output[n, c] =
        // weight[n]`, non-uniform in `n`, so the cancellation above no
        // longer applies. Built as REAL tape-tracked ops (an elementwise
        // `Mul` recorded via `record_and_link_binary`, then the existing
        // `tape_sum_to_scalar` reduction) so gradients flow through to
        // gamma/beta exactly as `.backward()` would compute them for any
        // other loss — not a change to the oracle math alone.
        let weight = make_tensor(vec![0.5, 1.0, 1.5, 2.0], &[4, 1]);
        let weighted_raw = tenflowers_core::ops::mul(&out.tensor, &weight.tensor)
            .expect("weighted mul must succeed");
        let weighted = PyTensor {
            tensor: Arc::new(weighted_raw),
            requires_grad: true,
            is_pinned: false,
        };
        crate::implicit_autograd::record_and_link_binary(
            crate::implicit_autograd::BinaryOpKind::Mul,
            &out,
            &weight,
            &weighted,
        )
        .expect("recording weighted mul must succeed");

        let loss = tape_sum_to_scalar(&weighted);
        run_backward(&loss).expect("backward must succeed");

        let gamma_id = bn.gamma_param.borrow(py).id();
        let beta_id = bn.beta_param.borrow(py).id();
        let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
            .expect("gamma grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
            .expect("beta grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");

        assert_eq!(grad_gamma.len(), 2);
        assert_eq!(grad_beta.len(), 2);
        assert!(
            grad_gamma.iter().any(|&g| g != 0.0),
            "gamma grad must be non-zero: {grad_gamma:?}"
        );
        assert!(
            grad_beta.iter().any(|&g| g != 0.0),
            "beta grad must be non-zero: {grad_beta:?}"
        );
        assert!(grad_gamma.iter().all(|g| g.is_finite()));
        assert!(grad_beta.iter().all(|g| g.is_finite()));

        // beta's gradient for `loss = sum((gamma*normalized+beta) *
        // weight)` is:
        //   d(loss)/d(beta_c) = sum_n(weight[n] * d(out[n,c])/d(beta_c))
        //                     = sum_n(weight[n] * 1) = sum(weight)
        //                     = 0.5 + 1.0 + 1.5 + 2.0 = 5.0
        // — independent of gamma/beta/normalized's own values (so, unlike
        // the sum-of-squares alternative, this stays non-degenerate even
        // at the layer's fresh beta=0 initialisation), for every channel.
        assert_close(&grad_beta, &[5.0, 5.0], 1e-3, "batch_norm train grad_beta");

        // gamma's gradient closed form:
        //   d(loss)/d(gamma_c) = sum_n(weight[n] * normalized[n,c])
        // With input columns c0=[1,3,5,7] (mean=4) and c1=[2,4,6,8]
        // (mean=5), both have batch variance
        // mean((x-mean)^2) = (9+1+1+9)/4 = 5, so std = sqrt(5+1e-5), and
        // both channels normalise to the SAME values (only shifted by a
        // constant per-channel mean, which cancels):
        // normalized = [-3,-1,1,3] / std ~= [-1.341641, -0.447214,
        // 0.447214, 1.341641] for both channels. So:
        //   grad_gamma_c = 0.5*(-1.341641) + 1.0*(-0.447214)
        //                + 1.5*(0.447214) + 2.0*(1.341641)
        //                ~= 2.236068
        // for both channels (matches the FD cross-check below and the
        // standalone empirical probe's `[2.2361279, 2.2361279]`).
        let std = (5.0f32 + 1e-5).sqrt();
        let normalized = [-3.0f32, -1.0, 1.0, 3.0].map(|v| v / std);
        let weights = [0.5f32, 1.0, 1.5, 2.0];
        let expected_gamma: f32 = normalized
            .iter()
            .zip(weights.iter())
            .map(|(&n, &w)| n * w)
            .sum();
        assert_close(
            &grad_gamma,
            &[expected_gamma, expected_gamma],
            1e-2,
            "batch_norm train grad_gamma vs closed-form oracle",
        );

        // gamma's gradient must ALSO match a finite-difference oracle
        // that re-runs the SAME real forward kernel this layer uses (and
        // the SAME weighted-sum loss), so this test cannot pass on a
        // fudged/oracle-mismatched formula shared by both the closed
        // form and the tape-based computation above.
        let gamma_vals = bn
            .gamma_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let beta_vals = bn
            .beta_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        // batch_norm hard-requires exactly 4D (NCHW) input; the native
        // [4, 2] input must be reshaped up to [4, 2, 1, 1] before calling
        // the op directly here (mirroring `tape_reshape_up`'s own
        // reshape call, see that function above, but without tape
        // recording — this closure only computes a raw scalar loss for
        // finite-differencing).
        let input_arc_4d = Arc::new(
            tenflowers_core::ops::reshape(
                &Tensor::<f32>::from_vec(input_data.clone(), &[4, 2]).expect("input"),
                &[4, 2, 1, 1],
            )
            .expect("reshape to 4D"),
        );
        let weight_vals = [0.5f32, 1.0, 1.5, 2.0];
        let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
            let gamma_t = Tensor::from_vec(gamma.to_vec(), &[2]).expect("gamma");
            let beta_t = Tensor::from_vec(beta.to_vec(), &[2]).expect("beta");
            let running_mean = Tensor::<f32>::zeros(&[2]);
            let running_var = Tensor::<f32>::ones(&[2]);
            let out = tenflowers_core::ops::batch_norm(
                &input_arc_4d,
                &gamma_t,
                &beta_t,
                &running_mean,
                &running_var,
                1e-5,
                true,
            )
            .expect("batch_norm");
            let out_vec = out.to_vec().expect("readable");
            // out_vec is NCHW-flattened (N=4, C=2, H=1, W=1): index
            // n*2 + c for (n, c).
            (0..4)
                .flat_map(|n| (0..2).map(move |c| (n, c)))
                .map(|(n, c)| out_vec[n * 2 + c] * weight_vals[n])
                .sum()
        };
        let fd_grad_gamma = finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
        assert_close(
            &grad_gamma,
            &fd_grad_gamma,
            5e-2,
            "batch_norm train grad_gamma vs finite-difference",
        );
    });
}

#[test]
fn batch_norm_eval_mode_gamma_beta_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let mut bn = PyBatchNorm1d::new(py, 2, None, None, None).expect("bn construction");
        // Give eval mode non-trivial running statistics to normalize
        // against, distinct from the freshly-initialised (0, 1) prior,
        // so this test cannot spuriously pass via a degenerate default.
        bn.running_mean = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("running_mean");
        bn.running_var = Tensor::from_vec(vec![4.0, 9.0], &[2]).expect("running_var");
        bn.eval();

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = make_tensor(input_data.clone(), &[4, 2]);

        let out = bn.forward(py, &input).expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let gamma_id = bn.gamma_param.borrow(py).id();
        let beta_id = bn.beta_param.borrow(py).id();
        let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
            .expect("gamma grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
            .expect("beta grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");

        assert_eq!(grad_gamma.len(), 2);
        assert_eq!(grad_beta.len(), 2);
        assert!(
            grad_gamma.iter().any(|&g| g != 0.0),
            "gamma grad must be non-zero: {grad_gamma:?}"
        );
        assert!(
            grad_beta.iter().any(|&g| g != 0.0),
            "beta grad must be non-zero: {grad_beta:?}"
        );
        assert!(grad_gamma.iter().all(|g| g.is_finite()));
        assert!(grad_beta.iter().all(|g| g.is_finite()));
        assert_close(&grad_beta, &[4.0, 4.0], 1e-3, "batch_norm eval grad_beta");

        // Eval-mode gamma gradient oracle:
        // d(sum(gamma*(x-running_mean)/std+beta))/d(gamma) =
        // sum((x - running_mean) / std) over the batch, per channel —
        // exactly the eval-mode formula documented on `batch_norm_backward`.
        let std0 = (4.0f32 + 1e-5).sqrt();
        let std1 = (9.0f32 + 1e-5).sqrt();
        let expected_gamma_c0: f32 = [1.0, 3.0, 5.0, 7.0].iter().map(|&x| (x - 1.0) / std0).sum();
        let expected_gamma_c1: f32 = [2.0, 4.0, 6.0, 8.0].iter().map(|&x| (x - 2.0) / std1).sum();
        assert_close(
            &grad_gamma,
            &[expected_gamma_c0, expected_gamma_c1],
            1e-2,
            "batch_norm eval grad_gamma vs closed-form oracle",
        );

        // Cross-check the closed-form oracle above against an
        // independent finite-difference oracle over the SAME real eval
        // forward kernel, so a coincidental agreement between two
        // hand-derived formulas cannot mask a shared mistake.
        let gamma_vals = bn
            .gamma_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let beta_vals = bn
            .beta_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        // batch_norm hard-requires exactly 4D (NCHW) input; the native
        // [4, 2] input must be reshaped up to [4, 2, 1, 1] before calling
        // the op directly here (mirroring `tape_reshape_up`'s own
        // reshape call, see that function above, but without tape
        // recording — this closure only computes a raw scalar loss for
        // finite-differencing).
        let input_arc = Arc::new(
            tenflowers_core::ops::reshape(
                &Tensor::<f32>::from_vec(input_data, &[4, 2]).expect("input"),
                &[4, 2, 1, 1],
            )
            .expect("reshape to 4D"),
        );
        let running_mean = bn.running_mean.clone();
        let running_var = bn.running_var.clone();
        let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
            let gamma_t = Tensor::from_vec(gamma.to_vec(), &[2]).expect("gamma");
            let beta_t = Tensor::from_vec(beta.to_vec(), &[2]).expect("beta");
            let out = tenflowers_core::ops::batch_norm(
                &input_arc,
                &gamma_t,
                &beta_t,
                &running_mean,
                &running_var,
                1e-5,
                false,
            )
            .expect("batch_norm");
            out.to_vec().expect("readable").iter().sum()
        };
        let fd_grad_gamma = finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
        assert_close(
            &grad_gamma,
            &fd_grad_gamma,
            5e-2,
            "batch_norm eval grad_gamma vs finite-difference",
        );
    });
}

#[test]
fn layer_norm_gamma_beta_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let ln = PyLayerNorm::new(py, vec![3], None).expect("ln construction");
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 9.0];
        let input = make_tensor(input_data.clone(), &[2, 3]);

        let out = ln.forward(py, &input).expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let gamma_id = ln.gamma_param.borrow(py).id();
        let beta_id = ln.beta_param.borrow(py).id();
        let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
            .expect("gamma grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
            .expect("beta grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");

        assert_eq!(grad_gamma.len(), 3);
        assert_eq!(grad_beta.len(), 3);
        assert!(
            grad_gamma.iter().any(|&g| g != 0.0),
            "gamma grad must be non-zero"
        );
        assert!(grad_beta.iter().all(|v| v.is_finite()));
        assert!(grad_gamma.iter().all(|v| v.is_finite()));
        // d(sum(gamma*normalized+beta))/d(beta) = row count = 2 for every element.
        assert_close(&grad_beta, &[2.0, 2.0, 2.0], 1e-3, "layer_norm grad_beta");

        let gamma_vals = ln
            .gamma_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let beta_vals = ln
            .beta_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let input_arc = Arc::new(Tensor::<f32>::from_vec(input_data, &[2, 3]).expect("input"));
        let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
            let gamma_t = Tensor::from_vec(gamma.to_vec(), &[3]).expect("gamma");
            let beta_t = Tensor::from_vec(beta.to_vec(), &[3]).expect("beta");
            let out = tenflowers_core::ops::layer_norm(&input_arc, &gamma_t, &beta_t, &[3], 1e-5)
                .expect("layer_norm");
            out.to_vec().expect("readable").iter().sum()
        };
        let fd_grad_gamma = finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
        assert_close(
            &grad_gamma,
            &fd_grad_gamma,
            5e-2,
            "layer_norm grad_gamma vs finite-difference",
        );
    });
}

#[test]
fn group_norm_gamma_beta_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let gn = PyGroupNorm::new(py, 2, 4, None).expect("gn construction");
        // (N=1, C=4, L=2)
        let input_data: Vec<f32> = vec![1.0, 5.0, 2.0, 3.0, 8.0, 1.0, 4.0, 6.0];
        let input = make_tensor(input_data.clone(), &[1, 4, 2]);

        let out = gn.forward(py, &input).expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let gamma_id = gn.gamma_param.borrow(py).id();
        let beta_id = gn.beta_param.borrow(py).id();
        let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
            .expect("gamma grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
            .expect("beta grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");

        assert_eq!(grad_gamma.len(), 4);
        assert_eq!(grad_beta.len(), 4);
        assert!(
            grad_gamma.iter().any(|&g| g != 0.0),
            "gamma grad must be non-zero"
        );
        assert!(grad_beta.iter().all(|v| v.is_finite()));
        assert!(grad_gamma.iter().all(|v| v.is_finite()));
        // d(sum(...))/d(beta[c]) = number of positions beta[c] is added
        // at = N * L = 1 * 2 = 2, for every channel.
        assert_close(
            &grad_beta,
            &[2.0, 2.0, 2.0, 2.0],
            1e-3,
            "group_norm grad_beta",
        );

        let gamma_vals = gn
            .gamma_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let beta_vals = gn
            .beta_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        // Finite-difference oracle re-runs the exact same reshape-to-4D
        // + group_norm forward path this layer's forward() uses.
        let input_arc_4d =
            Arc::new(Tensor::<f32>::from_vec(input_data, &[1, 4, 2, 1]).expect("input"));
        let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
            let gamma_t = Tensor::from_vec(gamma.to_vec(), &[4]).expect("gamma");
            let beta_t = Tensor::from_vec(beta.to_vec(), &[4]).expect("beta");
            let out = tenflowers_core::ops::group_norm(&input_arc_4d, &gamma_t, &beta_t, 2, 1e-5)
                .expect("group_norm");
            out.to_vec().expect("readable").iter().sum()
        };
        let fd_grad_gamma = finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
        assert_close(
            &grad_gamma,
            &fd_grad_gamma,
            5e-2,
            "group_norm grad_gamma vs finite-difference",
        );
    });
}

#[test]
fn instance_norm_gamma_beta_gradients_are_correct() {
    Python::initialize();
    Python::attach(|py| {
        let inorm = PyInstanceNorm1d::new(py, 2, None).expect("in construction");
        // (N=1, C=2, L=4)
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let input = make_tensor(input_data.clone(), &[1, 2, 4]);

        let out = inorm.forward(py, &input).expect("forward");
        let loss = tape_sum_to_scalar(&out);
        run_backward(&loss).expect("backward must succeed");

        let gamma_id = inorm.gamma_param.borrow(py).id();
        let beta_id = inorm.beta_param.borrow(py).id();
        let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
            .expect("gamma grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");
        let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
            .expect("beta grad must be populated")
            .tensor
            .to_vec()
            .expect("grad readable");

        assert_eq!(grad_gamma.len(), 2);
        assert_eq!(grad_beta.len(), 2);
        assert!(
            grad_gamma.iter().any(|&g| g != 0.0),
            "gamma grad must be non-zero"
        );
        assert!(grad_beta.iter().all(|v| v.is_finite()));
        assert!(grad_gamma.iter().all(|v| v.is_finite()));
        // d(sum(...))/d(beta[c]) = N * L = 1 * 4 = 4, for every channel.
        assert_close(&grad_beta, &[4.0, 4.0], 1e-3, "instance_norm grad_beta");

        let gamma_vals = inorm
            .gamma_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let beta_vals = inorm
            .beta_param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor")
            .tensor
            .to_vec()
            .expect("readable");
        let input_arc_4d =
            Arc::new(Tensor::<f32>::from_vec(input_data, &[1, 2, 4, 1]).expect("input"));
        let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
            let gamma_t = Tensor::from_vec(gamma.to_vec(), &[2]).expect("gamma");
            let beta_t = Tensor::from_vec(beta.to_vec(), &[2]).expect("beta");
            let out = tenflowers_core::ops::group_norm(&input_arc_4d, &gamma_t, &beta_t, 2, 1e-5)
                .expect("group_norm (instance_norm equivalent)");
            out.to_vec().expect("readable").iter().sum()
        };
        let fd_grad_gamma = finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
        assert_close(
            &grad_gamma,
            &fd_grad_gamma,
            5e-2,
            "instance_norm grad_gamma vs finite-difference",
        );
    });
}

#[test]
fn parameters_returns_gamma_and_beta_with_matching_identity() {
    Python::initialize();
    Python::attach(|py| {
        let bn = PyBatchNorm1d::new(py, 3, None, None, None).expect("bn construction");
        let params = bn.parameters(py);
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].borrow(py).id(), bn.gamma_param.borrow(py).id());
        assert_eq!(params[1].borrow(py).id(), bn.beta_param.borrow(py).id());

        let ln = PyLayerNorm::new(py, vec![4], None).expect("ln construction");
        let ln_params = ln.parameters(py);
        assert_eq!(ln_params.len(), 2);
        assert_eq!(ln_params[0].borrow(py).id(), ln.gamma_param.borrow(py).id());
        assert_eq!(ln_params[1].borrow(py).id(), ln.beta_param.borrow(py).id());

        let gn = PyGroupNorm::new(py, 2, 4, None).expect("gn construction");
        let gn_params = gn.parameters(py);
        assert_eq!(gn_params.len(), 2);
        assert_eq!(gn_params[0].borrow(py).id(), gn.gamma_param.borrow(py).id());
        assert_eq!(gn_params[1].borrow(py).id(), gn.beta_param.borrow(py).id());

        let inorm = PyInstanceNorm1d::new(py, 3, None).expect("in construction");
        let in_params = inorm.parameters(py);
        assert_eq!(in_params.len(), 2);
        assert_eq!(
            in_params[0].borrow(py).id(),
            inorm.gamma_param.borrow(py).id()
        );
        assert_eq!(
            in_params[1].borrow(py).id(),
            inorm.beta_param.borrow(py).id()
        );
    });
}
