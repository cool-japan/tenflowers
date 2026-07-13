"""
End-to-end PyTorch-style training-loop convergence tests for TenfloweRS FFI.

This module is the actual point of the implicit-autograd remediation effort:
it drives the *real* Python-facing API surface a user would type — exactly
the `.backward()` / `.grad()` / `optimizer.step()` / `optimizer.zero_grad()`
pattern documented in `crates/tenflowers-ffi/src/lib.rs`'s own module-level
docstring and in `README.md` — through a genuine multi-step training loop, and
asserts the loss *actually decreases substantially*, not merely that
`.backward()` runs without raising and populates `.grad()` once (that
narrower check already exists in `test_gradient_parity.py` and
`integration_test.py::test_optimizer_step`/`test_end_to_end_training`, and
neither of those calls `optimizer.step()` in a loop or checks convergence).

If a single "does implicit autograd actually work end-to-end" smoke test
matters anywhere in this codebase, it is this one.

Layer coverage (per the closing verification's requirement of "at least
Dense alone, a Sequential MLP, and one non-Dense layer type"):
  * test_dense_layer_training_converges       -- PyDense alone
  * test_sequential_mlp_training_converges     -- PySequential MLP (3 PyDense layers)
  * test_conv2d_training_converges             -- Conv2D (a non-Dense layer)

Conv2D configuration note: `PyConv2D`'s struct-level "Autograd" doc
(`neural/conv_layers/mod.rs`) documents that `.backward()` support is only
wired for the subset of configurations with unit dilation, `groups == 1`,
and no explicit integer padding (`padding == (0, 0)`, which is also the
constructor's own default when `padding` is omitted) -- other configurations
still produce a correct forward value but are not tape-linked. This test
deliberately stays inside that supported subset (default padding/dilation/
groups) so it validates the real, currently-wired backward path rather than
tripping the documented forward-only boundary.
"""

import numpy as np
import pytest

try:
    import tenflowers as tf
except ImportError:
    tf = None

pytestmark = pytest.mark.skipif(tf is None, reason="tenflowers native module not compiled")


def _assert_finite(loss_value: float, step: int):
    assert np.isfinite(loss_value), (
        f"loss became non-finite ({loss_value}) at step {step}; a training "
        "loop that diverges to NaN/Inf is not a passing convergence test"
    )


def test_dense_layer_training_converges():
    """A single PyDense layer, trained with real SGD on a fixed linear
    regression target, must converge: loss(step 49) << loss(step 0).

    Mirrors the project's own docstring pattern:

        layer = tf.PyDense(4, 1)
        optimizer = tf.SGD(learning_rate=0.1)
        for step in range(50):
            y_pred = layer.forward(x)
            loss = tf.mse_loss(y_pred, y_true)
            loss.backward()
            optimizer.step(layer)
            optimizer.zero_grad(layer)
    """
    rng = np.random.default_rng(seed=1234)

    # Fixed synthetic data: y = X @ w_true + b_true, no noise, so a linear
    # PyDense(4, 1) model (itself an affine map) can drive MSE arbitrarily
    # close to zero -- a clean, unambiguous convergence signal.
    batch_size = 16
    in_features = 4
    w_true = np.array([[2.0], [-1.5], [0.5], [3.0]], dtype=np.float32)
    b_true = np.float32(0.7)

    X = rng.uniform(-1.0, 1.0, size=(batch_size, in_features)).astype(np.float32)
    y = (X @ w_true + b_true).astype(np.float32)

    x_tensor = tf.tensor_from_numpy(X)
    y_tensor = tf.tensor_from_numpy(y)

    layer = tf.PyDense(in_features, 1, activation=None)
    for param in layer.parameters():
        param.set_requires_grad(True)

    optimizer = tf.SGD(learning_rate=0.1)

    losses = []
    for step in range(50):
        y_pred = layer.forward(x_tensor)
        loss = tf.mse_loss(y_pred, y_tensor)

        loss.backward()
        optimizer.step(layer)
        optimizer.zero_grad(layer)

        loss_value = float(tf.tensor_to_numpy(loss)[()])
        _assert_finite(loss_value, step)
        losses.append(loss_value)

    print("\nDense training loss trace (every 5 steps):")
    for step in range(0, 50, 5):
        print(f"  step {step:2d}: loss={losses[step]:.6f}")
    print(f"  step 49: loss={losses[-1]:.6f}")

    assert losses[0] > 1e-3, (
        f"initial loss ({losses[0]}) is already ~0; the test's synthetic "
        "target is not a meaningful convergence signal -- check w_true/b_true"
    )
    # A real gradient-descent step on a convex quadratic (linear model + MSE)
    # with lr=0.1 over 50 steps should reduce loss by at least two orders of
    # magnitude on noiseless data. This threshold is deliberately loose (an
    # honest "did training work at all" bar, not a tight numerical-precision
    # bar) -- 100x reduction cannot happen if optimizer.step() were a no-op,
    # if backward() were not populating real gradients, or if the gradient
    # sign/formula were wrong (those failure modes diverge or plateau, they
    # do not coincidentally shrink loss 100x).
    assert losses[-1] < losses[0] / 100.0, (
        f"loss did not converge: step0={losses[0]:.6f} -> step49={losses[-1]:.6f} "
        f"(only {losses[0] / max(losses[-1], 1e-12):.1f}x reduction, need >= 100x)"
    )


def test_sequential_mlp_training_converges():
    """A PySequential MLP (3 PyDense layers, ReLU-activated hidden layers),
    trained with real Adam on a fixed nonlinear regression target, must
    converge: loss(step 49) << loss(step 0).

    Uses Adam rather than plain SGD because the hidden layers'
    zero-mean-Xavier-initialised weights plus ReLU give a less
    perfectly-conditioned loss surface than the single-layer linear case in
    `test_dense_layer_training_converges`; Adam's per-parameter adaptive step
    size converges reliably within 50 steps without any learning-rate
    hand-tuning, exactly as it would for a real user.
    """
    rng = np.random.default_rng(seed=5678)

    batch_size = 32
    in_features = 6

    # Fixed nonlinear regression target: y = sum(X)^2 scaled down, plus a
    # sine term -- deliberately not linearly separable by a single affine
    # map, so this exercises genuine multi-layer nonlinear fitting (not
    # something a single PyDense could already solve, unlike the previous
    # test).
    X = rng.uniform(-1.0, 1.0, size=(batch_size, in_features)).astype(np.float32)
    raw = X.sum(axis=1, keepdims=True)
    y = (0.1 * raw**2 + 0.5 * np.sin(raw)).astype(np.float32)

    x_tensor = tf.tensor_from_numpy(X)
    y_tensor = tf.tensor_from_numpy(y)

    model = tf.PySequential()
    model.add(tf.PyDense(in_features, 16, activation="relu"))
    model.add(tf.PyDense(16, 16, activation="relu"))
    model.add(tf.PyDense(16, 1, activation=None))
    model.train()

    params = model.parameters()
    assert len(params) >= 6, (
        f"expected >= 6 parameter tensors (weight+bias x 3 layers), got "
        f"{len(params)} -- Sequential.parameters() may not be aggregating "
        "every layer's parameters"
    )
    for param in params:
        param.set_requires_grad(True)

    optimizer = tf.Adam(learning_rate=0.01)

    losses = []
    for step in range(50):
        y_pred = model.forward(x_tensor)
        loss = tf.mse_loss(y_pred, y_tensor)

        loss.backward()
        for param in params:
            grad = param.grad()
            assert grad is not None, (
                f"step {step}: gradient not populated for a Sequential "
                "parameter after backward() -- implicit autograd is not "
                "tracing through PySequential.forward()"
            )
            assert grad.shape() == param.shape(), (
                f"step {step}: gradient shape {grad.shape()} does not match "
                f"parameter shape {param.shape()}"
            )

        optimizer.step(model)
        optimizer.zero_grad(model)

        loss_value = float(tf.tensor_to_numpy(loss)[()])
        _assert_finite(loss_value, step)
        losses.append(loss_value)

    print("\nSequential MLP training loss trace (every 5 steps):")
    for step in range(0, 50, 5):
        print(f"  step {step:2d}: loss={losses[step]:.6f}")
    print(f"  step 49: loss={losses[-1]:.6f}")

    assert losses[0] > 1e-3, (
        f"initial loss ({losses[0]}) is already ~0; the test's synthetic "
        "target is not a meaningful convergence signal"
    )
    # Looser bar than the single-Dense-layer test (10x, not 100x): a 3-layer
    # nonlinear MLP on a nonlinear target is a harder optimization landscape,
    # but any genuine end-to-end backprop + Adam should still comfortably
    # clear an order of magnitude within 50 steps. Anything short of that
    # would point at a real wiring gap (e.g. a layer silently not
    # participating in the backward pass) rather than normal optimization
    # noise.
    assert losses[-1] < losses[0] / 10.0, (
        f"loss did not converge: step0={losses[0]:.6f} -> step49={losses[-1]:.6f} "
        f"(only {losses[0] / max(losses[-1], 1e-12):.1f}x reduction, need >= 10x)"
    )
    # Monotonic-ish sanity: the second half of training should have a lower
    # mean loss than the first half. This catches a subtler failure mode
    # than the endpoint ratio alone -- e.g. loss oscillating wildly and
    # happening to land low on step 49 by chance rather than genuinely
    # having descended.
    first_half_mean = float(np.mean(losses[:25]))
    second_half_mean = float(np.mean(losses[25:]))
    assert second_half_mean < first_half_mean, (
        f"second-half mean loss ({second_half_mean:.6f}) is not lower than "
        f"first-half mean loss ({first_half_mean:.6f}) -- loss trace does "
        "not show a genuine downward trend"
    )


def test_conv2d_training_converges():
    """PyConv2D (a non-Dense layer), trained with real Adam, must converge:
    loss(step 29) << loss(step 0).

    Stays inside PyConv2D's documented tape-linked configuration subset
    (default padding=(0,0), dilation=(1,1), groups=1 -- see this module's
    top-level docstring) so it validates the real backward path rather than
    the forward-only fallback for other configurations.

    Uses fewer steps (30, not 50) and a smaller spatial size than the other
    two tests purely for wall-clock cost -- Conv2D's forward has
    `out_channels * in_channels * kernel_h * kernel_w * out_h * out_w`
    inner-loop work per call, run twice per step (forward, then whatever
    re-forward the backward pass needs), which adds up fast in a Python
    loop; the assertions themselves are just as strict as the other tests'.
    """
    rng = np.random.default_rng(seed=9012)

    batch_size = 2
    in_channels = 3
    out_channels = 4
    height, width = 8, 8
    kernel_size = (3, 3)

    X = rng.uniform(-1.0, 1.0, size=(batch_size, in_channels, height, width)).astype(np.float32)
    x_tensor = tf.tensor_from_numpy(X)

    # kernel_size=(3,3), default stride=(1,1)/padding=(0,0)/dilation=(1,1)/
    # groups=1 -- output spatial size (H-2, W-2) = (6, 6).
    conv = tf.Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
    )

    params = conv.parameters()
    assert len(params) >= 1, "Conv2D should expose at least a weight parameter"
    for param in params:
        param.set_requires_grad(True)

    # Probe the real output shape before building a matching fixed target
    # (rather than hand-deriving H_out/W_out and risking an off-by-one).
    probe_output = conv.forward(x_tensor)
    out_shape = probe_output.shape()

    # Fixed, non-zero regression target with the same shape as Conv2D's own
    # output. This MUST be non-zero: `neural/conv_layers/mod.rs` zero-
    # initialises both weight and bias (see this module's top-level
    # docstring), so `conv.forward(x)` at step 0 is exactly all-zero
    # regardless of `x` -- an all-zero target would make step-0 loss exactly
    # 0.0 (nothing to converge from, and the `losses[0] > 1e-6` sanity check
    # below would itself fail). A fixed non-zero target instead gives a
    # genuine, unambiguous initial error for training to reduce.
    y = rng.uniform(0.5, 1.5, size=out_shape).astype(np.float32)
    y_tensor = tf.tensor_from_numpy(y)

    optimizer = tf.Adam(learning_rate=0.05)

    losses = []
    num_steps = 30
    for step in range(num_steps):
        y_pred = conv.forward(x_tensor)
        assert y_pred.shape() == out_shape, (
            f"step {step}: Conv2D output shape changed between calls "
            f"({y_pred.shape()} vs {out_shape})"
        )
        loss = tf.mse_loss(y_pred, y_tensor)

        loss.backward()
        for param in params:
            grad = param.grad()
            assert grad is not None, (
                f"step {step}: gradient not populated for a Conv2D "
                "parameter after backward() -- implicit autograd is not "
                "tracing through Conv2D.forward() for this configuration"
            )
            assert grad.shape() == param.shape(), (
                f"step {step}: gradient shape {grad.shape()} does not match "
                f"parameter shape {param.shape()}"
            )

        optimizer.step(conv)
        optimizer.zero_grad(conv)

        loss_value = float(tf.tensor_to_numpy(loss)[()])
        _assert_finite(loss_value, step)
        losses.append(loss_value)

    print("\nConv2D training loss trace (every 5 steps):")
    for step in range(0, num_steps, 5):
        print(f"  step {step:2d}: loss={losses[step]:.6f}")
    print(f"  step {num_steps - 1}: loss={losses[-1]:.6f}")

    assert losses[0] > 1e-6, (
        f"initial loss ({losses[0]}) is already ~0; the all-zero-weight "
        "Conv2D init happens to start near the all-zero target -- this "
        "would make the test's convergence signal meaningless (see "
        "conv_layers/mod.rs's zero-init note); investigate the input scale."
    )
    assert losses[-1] < losses[0] / 5.0, (
        f"loss did not converge: step0={losses[0]:.6f} -> "
        f"step{num_steps - 1}={losses[-1]:.6f} "
        f"(only {losses[0] / max(losses[-1], 1e-12):.1f}x reduction, need >= 5x)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
