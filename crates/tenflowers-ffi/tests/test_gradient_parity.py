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

    def __init__(self, epsilon: float = 1e-5, rtol: float = 1e-4, atol: float = 1e-6):
        """
        Initialize gradient parity tester.

        Args:
            epsilon: Step size for numerical gradient computation
            rtol: Relative tolerance for gradient comparison
            atol: Absolute tolerance for gradient comparison
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
    x_tensor.requires_grad = True

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
    x_tensor.requires_grad = True

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
    x_tensor.requires_grad = True

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
    import tenflowers.neural as nn

    tester = GradientParityTester()

    # Test ReLU
    def relu_numpy(x):
        return np.sum(np.maximum(0, x))

    x = np.random.randn(3, 3).astype(np.float32)
    numerical_grad = tester.compute_numerical_gradient(relu_numpy, [x])

    x_tensor = tf.tensor_from_numpy(x)
    x_tensor.requires_grad = True
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
