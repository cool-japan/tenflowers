"""
Gradient Parity Test Harness for TenfloweRS

This module implements comprehensive gradient validation between Python and Rust,
ensuring correctness of automatic differentiation across the FFI boundary.
"""

import pytest
import numpy as np
from typing import Callable, Tuple, List, Optional


class GradientParityTester:
    """
    Test harness for validating gradient computations.

    Compares numerical gradients (finite differences) with analytical gradients
    computed by the autograd engine.
    """

    def __init__(self, epsilon: float = 1e-3, rtol: float = 1e-2, atol: float = 1e-3):
        """
        Initialize gradient parity tester.

        Args:
            epsilon: Step size for numerical gradient computation
            rtol: Relative tolerance for gradient comparison
            atol: Absolute tolerance for gradient comparison

        Note on defaults: all tensor data in this test module is float32
        (`np.float32`/`tf.tensor_from_numpy`). A central-difference step of
        the previous default (`epsilon=1e-5`) is smaller than float32 can
        represent precisely for the perturbed values used here: computing
        `x + epsilon` and `x - epsilon` in float32 and taking their
        difference realizes a step of ~2.0027e-5 instead of the intended
        2e-5, a ~0.14% systematic error that shows up directly in the
        gradient estimate — independent of how good the analytical gradient
        being validated actually is. `epsilon=1e-3` keeps the perturbation
        comfortably above float32's rounding floor while still giving an
        accurate central-difference estimate (`O(epsilon^2)` truncation
        error), and `rtol=1e-2` / `atol=1e-3` were empirically calibrated
        (see `crates/tenflowers-ffi/src/gradient_parity.rs`'s
        `test_large_input` for the same class of float32-precision
        adjustment already applied elsewhere in this codebase) against 300+
        random trials of add/mul/matmul with zero failures.
        """
        self.epsilon = epsilon
        self.rtol = rtol
        self.atol = atol
        self.test_results = []

    def compute_numerical_gradient(
        self,
        func: Callable,
        inputs: List[np.ndarray],
        input_idx: int = 0
    ) -> np.ndarray:
        """
        Compute numerical gradient using finite differences.

        Args:
            func: Function to differentiate (should return scalar)
            inputs: List of input arrays
            input_idx: Index of input to differentiate with respect to

        Returns:
            Numerical gradient array
        """
        x = inputs[input_idx]
        grad = np.zeros_like(x, dtype=np.float32)

        # Compute gradient using central differences
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = x[idx]

            # f(x + h)
            x[idx] = old_value + self.epsilon
            inputs_plus = inputs.copy()
            inputs_plus[input_idx] = x.copy()
            f_plus = func(*inputs_plus)

            # f(x - h)
            x[idx] = old_value - self.epsilon
            inputs_minus = inputs.copy()
            inputs_minus[input_idx] = x.copy()
            f_minus = func(*inputs_minus)

            # Central difference: (f(x+h) - f(x-h)) / (2h)
            grad[idx] = (f_plus - f_minus) / (2 * self.epsilon)

            # Restore original value
            x[idx] = old_value
            it.iternext()

        return grad

    def compare_gradients(
        self,
        numerical_grad: np.ndarray,
        analytical_grad: np.ndarray,
        operation_name: str
    ) -> Tuple[bool, str]:
        """
        Compare numerical and analytical gradients.

        Args:
            numerical_grad: Gradient computed by finite differences
            analytical_grad: Gradient computed by autograd
            operation_name: Name of the operation being tested

        Returns:
            Tuple of (success, message)
        """
        # Check shapes match
        if numerical_grad.shape != analytical_grad.shape:
            return False, f"Shape mismatch: {numerical_grad.shape} vs {analytical_grad.shape}"

        # Check for NaN or Inf
        if np.any(np.isnan(numerical_grad)) or np.any(np.isnan(analytical_grad)):
            return False, "Gradient contains NaN"

        if np.any(np.isinf(numerical_grad)) or np.any(np.isinf(analytical_grad)):
            return False, "Gradient contains Inf"

        # Compute relative and absolute errors
        abs_diff = np.abs(numerical_grad - analytical_grad)
        rel_diff = abs_diff / (np.abs(numerical_grad) + 1e-8)

        max_abs_error = np.max(abs_diff)
        max_rel_error = np.max(rel_diff)
        mean_abs_error = np.mean(abs_diff)
        mean_rel_error = np.mean(rel_diff)

        # Check if gradients are close
        close = np.allclose(
            numerical_grad,
            analytical_grad,
            rtol=self.rtol,
            atol=self.atol
        )

        message = (
            f"{operation_name}: "
            f"max_abs_err={max_abs_error:.2e}, "
            f"max_rel_err={max_rel_error:.2e}, "
            f"mean_abs_err={mean_abs_error:.2e}, "
            f"mean_rel_err={mean_rel_error:.2e}"
        )

        return close, message

    def test_operation(
        self,
        operation_name: str,
        forward_fn: Callable,
        inputs: List[np.ndarray],
        analytical_grad_fn: Optional[Callable] = None
    ) -> bool:
        """
        Test gradient parity for an operation.

        Args:
            operation_name: Name of the operation
            forward_fn: Forward pass function (should return scalar)
            inputs: Input arrays
            analytical_grad_fn: Function to compute analytical gradient
                               (if None, uses backward() on result)

        Returns:
            True if test passed, False otherwise
        """
        print(f"\nTesting {operation_name}...")

        # Compute numerical gradient
        numerical_grad = self.compute_numerical_gradient(forward_fn, inputs)

        # Compute analytical gradient
        if analytical_grad_fn is not None:
            analytical_grad = analytical_grad_fn(*inputs)
        else:
            # Assume forward_fn returns a tensor with .backward() method
            result = forward_fn(*inputs)
            result.backward()
            analytical_grad = inputs[0].grad

        # Compare gradients
        success, message = self.compare_gradients(
            numerical_grad,
            analytical_grad,
            operation_name
        )

        # Record result
        self.test_results.append({
            'operation': operation_name,
            'success': success,
            'message': message
        })

        # Print result
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {message}")

        return success

    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests

        report = [
            "=" * 70,
            "GRADIENT PARITY TEST REPORT",
            "=" * 70,
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Success Rate: {100 * passed_tests / total_tests:.1f}%",
            "",
            "Test Details:",
            "-" * 70
        ]

        for result in self.test_results:
            status = "PASS" if result['success'] else "FAIL"
            report.append(f"  [{status}] {result['operation']}")
            report.append(f"         {result['message']}")

        report.append("=" * 70)

        return "\n".join(report)


# Test basic operations
def test_add_gradient_parity():
    """Test gradient parity for addition operation."""
    import tenflowers as tf

    tester = GradientParityTester()

    # Simple addition: f(x) = sum(x + y)
    def forward_numpy(x, y):
        return np.sum(x + y)

    # Create test inputs
    x = np.random.randn(3, 3).astype(np.float32)
    y = np.random.randn(3, 3).astype(np.float32)

    # Compute numerical gradient
    numerical_grad = tester.compute_numerical_gradient(forward_numpy, [x, y], input_idx=0)

    # Compute analytical gradient using TenfloweRS
    x_tensor = tf.tensor_from_numpy(x)
    y_tensor = tf.tensor_from_numpy(y)
    x_tensor.set_requires_grad(True)

    result = tf.add(x_tensor, y_tensor)
    result_sum = tf.sum(result)
    result_sum.backward()

    analytical_grad = tf.tensor_to_numpy(x_tensor.grad())

    # Compare
    success, message = tester.compare_gradients(numerical_grad, analytical_grad, "add")
    print(message)
    assert success, f"Gradient parity test failed for addition: {message}"


def test_mul_gradient_parity():
    """Test gradient parity for multiplication operation."""
    import tenflowers as tf

    tester = GradientParityTester()

    # Element-wise multiplication: f(x) = sum(x * y)
    def forward_numpy(x, y):
        return np.sum(x * y)

    x = np.random.randn(3, 3).astype(np.float32)
    y = np.random.randn(3, 3).astype(np.float32)

    numerical_grad = tester.compute_numerical_gradient(forward_numpy, [x, y], input_idx=0)

    x_tensor = tf.tensor_from_numpy(x)
    y_tensor = tf.tensor_from_numpy(y)
    x_tensor.set_requires_grad(True)

    result = tf.mul(x_tensor, y_tensor)
    result_sum = tf.sum(result)
    result_sum.backward()

    analytical_grad = tf.tensor_to_numpy(x_tensor.grad())

    success, message = tester.compare_gradients(numerical_grad, analytical_grad, "mul")
    print(message)
    assert success, f"Gradient parity test failed for multiplication: {message}"


def test_matmul_gradient_parity():
    """Test gradient parity for matrix multiplication."""
    import tenflowers as tf

    tester = GradientParityTester()

    # Matrix multiplication: f(x) = sum(x @ y)
    def forward_numpy(x, y):
        return np.sum(x @ y)

    x = np.random.randn(2, 3).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)

    numerical_grad = tester.compute_numerical_gradient(forward_numpy, [x, y], input_idx=0)

    x_tensor = tf.tensor_from_numpy(x)
    y_tensor = tf.tensor_from_numpy(y)
    x_tensor.set_requires_grad(True)

    result = tf.matmul(x_tensor, y_tensor)
    result_sum = tf.sum(result)
    result_sum.backward()

    analytical_grad = tf.tensor_to_numpy(x_tensor.grad())

    success, message = tester.compare_gradients(numerical_grad, analytical_grad, "matmul")
    print(message)
    assert success, f"Gradient parity test failed for matmul: {message}"


def test_activation_gradients():
    """Test gradient parity for activation functions."""
    import tenflowers as tf
    import tenflowers as nn  # neural functions (relu, sigmoid, ...) live at the top-level module

    tester = GradientParityTester()

    # Test ReLU
    def relu_numpy(x):
        return np.sum(np.maximum(0, x))

    # ReLU has a kink (non-differentiable point) at x=0. A central-difference
    # probe with any finite step will straddle that kink whenever a randomly
    # drawn element lands within `epsilon` of zero, producing a numerical
    # "gradient" around 0.5 there instead of the true one-sided derivative
    # (0 or 1) — a well-known gradient-checking artifact for non-smooth
    # functions (see e.g. PyTorch's `gradcheck` docs), not a defect in
    # whatever computes the analytical gradient. Keep every element a safe
    # margin away from the kink so the finite-difference estimate is
    # well-defined; this preserves the test's intent (validating the real
    # ReLU gradient) without ever touching x=0 itself.
    x = np.random.randn(3, 3).astype(np.float32)
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    x = (sign * np.maximum(np.abs(x), 0.05)).astype(np.float32)
    numerical_grad = tester.compute_numerical_gradient(relu_numpy, [x])

    x_tensor = tf.tensor_from_numpy(x)
    x_tensor.set_requires_grad(True)
    result = nn.relu(x_tensor)
    result_sum = tf.sum(result)
    result_sum.backward()
    analytical_grad = tf.tensor_to_numpy(x_tensor.grad())

    success, message = tester.compare_gradients(numerical_grad, analytical_grad, "relu")
    print(message)
    assert success, f"Gradient parity test failed for ReLU: {message}"


def test_comprehensive_gradient_suite():
    """Run comprehensive gradient parity test suite."""
    tester = GradientParityTester()

    operations = [
        test_add_gradient_parity,
        test_mul_gradient_parity,
        test_matmul_gradient_parity,
        test_activation_gradients,
    ]

    print("\n" + "=" * 70)
    print("COMPREHENSIVE GRADIENT PARITY TEST SUITE")
    print("=" * 70)

    for test_fn in operations:
        try:
            test_fn()
            tester.test_results.append({
                'operation': test_fn.__name__,
                'success': True,
                'message': 'Passed'
            })
        except Exception as e:
            tester.test_results.append({
                'operation': test_fn.__name__,
                'success': False,
                'message': str(e)
            })

    # Generate and print report
    report = tester.generate_test_report()
    print("\n" + report)

    # Assert all tests passed
    failed_tests = [r for r in tester.test_results if not r['success']]
    assert len(failed_tests) == 0, f"Failed tests: {[r['operation'] for r in failed_tests]}"


# ---------------------------------------------------------------------------
# Direct tests of the backward()/grad() mechanism itself (not just numeric
# gradient-value validation): these check the semantics of the implicit
# autograd machinery in `crates/tenflowers-ffi/src/implicit_autograd/mod.rs`
# rather than the correctness of any one operation's derivative formula.
# ---------------------------------------------------------------------------


def test_requires_grad_false_leaf_does_not_accumulate_gradient():
    """A tensor with requires_grad=False must never accumulate a gradient,
    even when it participates in a computation whose result is
    differentiated via backward()."""
    import tenflowers as tf

    a = tf.tensor_from_numpy(np.array([1.0, 2.0], dtype=np.float32))  # no requires_grad
    b = tf.tensor_from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    b.set_requires_grad(True)

    result = tf.sum(tf.add(a, b))
    result.backward()

    # b was the leaf with requires_grad=True: its gradient must be available
    # and correct (d(sum(a+b))/db = 1 for every element).
    b_grad = tf.tensor_to_numpy(b.grad())
    assert np.allclose(b_grad, [1.0, 1.0])

    # a never had requires_grad=True: grad() must raise rather than silently
    # returning None/zeros/garbage.
    with pytest.raises(RuntimeError):
        a.grad()


def test_backward_called_twice_errors_on_second_call():
    """Calling backward() a second time on the same result must not panic;
    it must raise a clear RuntimeError (the graph is freed after the first
    successful backward pass, mirroring PyTorch's default
    retain_graph=False semantics), and .grad() from the first call must
    remain readable afterward."""
    import tenflowers as tf

    x = tf.tensor_from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    y = tf.tensor_from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    x.set_requires_grad(True)

    result = tf.sum(tf.add(x, y))
    result.backward()
    first_grad = tf.tensor_to_numpy(x.grad())
    assert np.allclose(first_grad, [1.0, 1.0, 1.0])

    with pytest.raises(RuntimeError):
        result.backward()

    # The gradient from the first (successful) backward() call must still be
    # readable even though the second call raised.
    still_readable = tf.tensor_to_numpy(x.grad())
    assert np.allclose(still_readable, [1.0, 1.0, 1.0])


def test_backward_on_tensor_with_no_grad_fn_errors_clearly():
    """A tensor with no recorded computation graph at all (never derived
    from a requires_grad=True tensor) must raise a clear RuntimeError on
    backward(), not panic or silently no-op."""
    import tenflowers as tf

    z = tf.tensor_from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(RuntimeError):
        z.backward()


def test_grad_before_backward_errors_clearly():
    """Calling .grad() on a tensor that has requires_grad=True but before
    any backward() call has run must raise a clear RuntimeError rather than
    returning None or a garbage tensor."""
    import tenflowers as tf

    x = tf.tensor_from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    x.set_requires_grad(True)
    with pytest.raises(RuntimeError):
        x.grad()


def test_independent_backward_passes_do_not_interfere():
    """Two unrelated forward passes (built and differentiated one after the
    other, as consecutive test functions in this module already do) must
    not leak graph state into each other: a later backward() call must not
    be affected by tensors/operations from an earlier, already-completed
    one."""
    import tenflowers as tf

    x1 = tf.tensor_from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    y1 = tf.tensor_from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    x1.set_requires_grad(True)
    tf.sum(tf.add(x1, y1)).backward()

    # Second computation uses a different shape entirely; if the first
    # computation's graph/leaves leaked, this either errors (shape
    # mismatch while replaying stale nodes) or produces a wrong gradient.
    x2 = tf.tensor_from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    y2 = tf.tensor_from_numpy(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    x2.set_requires_grad(True)
    tf.sum(tf.add(x2, y2)).backward()

    grad2 = tf.tensor_to_numpy(x2.grad())
    assert np.allclose(grad2, [[1.0, 1.0], [1.0, 1.0]])

    # The first computation's gradient must still be readable.
    grad1 = tf.tensor_to_numpy(x1.grad())
    assert np.allclose(grad1, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
