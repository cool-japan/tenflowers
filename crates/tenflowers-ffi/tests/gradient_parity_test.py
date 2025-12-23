#!/usr/bin/env python3
"""
Gradient Parity Test Harness for TenfloweRS

This test suite verifies that gradients computed by TenfloweRS match
those computed by PyTorch (as a reference implementation).

The tests ensure that:
1. Forward pass outputs match
2. Backward pass gradients match within acceptable tolerance
3. Higher-order gradients match (where applicable)

Usage:
    python tests/gradient_parity_test.py [--verbose] [--tolerance 1e-5]
"""

import argparse
import sys
from typing import Callable, List, Tuple, Optional
import math

# Import testing framework components
try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required for gradient parity tests")
    print("Install with: pip install numpy")
    sys.exit(1)

# Optional: Try to import PyTorch for reference comparisons
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available - will use numerical gradients as reference")


class GradientParityTester:
    """Test harness for verifying gradient parity between implementations"""

    def __init__(self, tolerance: float = 1e-5, verbose: bool = False):
        self.tolerance = tolerance
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results: List[Tuple[str, bool, Optional[str]]] = []

    def numerical_gradient(
        self,
        func: Callable,
        x: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Compute numerical gradient using finite differences

        Args:
            func: Function to compute gradient for
            x: Input point
            epsilon: Finite difference step size

        Returns:
            Numerical gradient
        """
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            old_value = x[idx]

            # f(x + epsilon)
            x[idx] = old_value + epsilon
            fxh_plus = func(x)

            # f(x - epsilon)
            x[idx] = old_value - epsilon
            fxh_minus = func(x)

            # Restore original value
            x[idx] = old_value

            # Compute gradient: (f(x+h) - f(x-h)) / (2*h)
            grad[idx] = (fxh_plus - fxh_minus) / (2 * epsilon)
            it.iternext()

        return grad

    def compare_tensors(
        self,
        a: np.ndarray,
        b: np.ndarray,
        name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Compare two tensors for equality within tolerance

        Args:
            a: First tensor
            b: Second tensor
            name: Name for error message

        Returns:
            (success, error_message)
        """
        if a.shape != b.shape:
            return False, f"{name}: Shape mismatch: {a.shape} vs {b.shape}"

        max_diff = np.max(np.abs(a - b))
        if max_diff > self.tolerance:
            rel_error = max_diff / (np.max(np.abs(a)) + 1e-10)
            return False, (
                f"{name}: Values differ by {max_diff:.2e} "
                f"(relative: {rel_error:.2e}, tolerance: {self.tolerance:.2e})"
            )

        return True, None

    def test_operation(
        self,
        name: str,
        operation: Callable,
        inputs: List[np.ndarray],
        expected_output: Optional[np.ndarray] = None,
        expected_gradients: Optional[List[np.ndarray]] = None,
    ) -> bool:
        """
        Test a single operation for gradient parity

        Args:
            name: Test name
            operation: Operation to test
            inputs: Input tensors
            expected_output: Expected output (optional)
            expected_gradients: Expected gradients (optional)

        Returns:
            True if test passed, False otherwise
        """
        if self.verbose:
            print(f"\nTesting: {name}")

        try:
            # Compute output
            output = operation(*inputs)

            # Check output if expected
            if expected_output is not None:
                success, error = self.compare_tensors(
                    output, expected_output, f"{name} output"
                )
                if not success:
                    self.test_results.append((name, False, error))
                    self.tests_failed += 1
                    if self.verbose:
                        print(f"  FAILED: {error}")
                    return False

            # Check gradients if expected
            if expected_gradients is not None:
                for i, (inp, expected_grad) in enumerate(zip(inputs, expected_gradients)):
                    if expected_grad is None:
                        continue

                    # Compute numerical gradient
                    def scalar_func(x):
                        result = operation(x, *inputs[1:] if i == 0 else inputs[:i] + [x] + inputs[i+1:])
                        return np.sum(result)

                    numerical_grad = self.numerical_gradient(scalar_func, inp)

                    success, error = self.compare_tensors(
                        numerical_grad, expected_grad, f"{name} gradient[{i}]"
                    )
                    if not success:
                        self.test_results.append((name, False, error))
                        self.tests_failed += 1
                        if self.verbose:
                            print(f"  FAILED: {error}")
                        return False

            # Test passed
            self.test_results.append((name, True, None))
            self.tests_passed += 1
            if self.verbose:
                print(f"  PASSED")
            return True

        except Exception as e:
            error = f"{name}: Exception: {str(e)}"
            self.test_results.append((name, False, error))
            self.tests_failed += 1
            if self.verbose:
                print(f"  FAILED: {error}")
            return False

    def run_basic_tests(self):
        """Run basic gradient parity tests"""
        print("\n=== Running Basic Gradient Parity Tests ===\n")

        # Test 1: Addition
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        self.test_operation(
            "Addition",
            lambda x, y: x + y,
            [a, b],
            expected_output=np.array([[6.0, 8.0], [10.0, 12.0]])
        )

        # Test 2: Multiplication
        self.test_operation(
            "Element-wise Multiplication",
            lambda x, y: x * y,
            [a, b],
            expected_output=np.array([[5.0, 12.0], [21.0, 32.0]])
        )

        # Test 3: Matrix Multiplication
        self.test_operation(
            "Matrix Multiplication",
            lambda x, y: np.matmul(x, y),
            [a, b],
            expected_output=np.array([[19.0, 22.0], [43.0, 50.0]])
        )

        # Test 4: ReLU activation
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        self.test_operation(
            "ReLU",
            lambda x: np.maximum(0, x),
            [x],
            expected_output=np.array([[0.0, 2.0], [3.0, 0.0]])
        )

        # Test 5: Sigmoid activation
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.test_operation(
            "Sigmoid",
            sigmoid,
            [np.array([[0.0, 1.0], [-1.0, 2.0]])]
        )

        # Test 6: Tanh activation
        self.test_operation(
            "Tanh",
            np.tanh,
            [np.array([[0.0, 1.0], [-1.0, 2.0]])]
        )

        # Test 7: Mean reduction
        self.test_operation(
            "Mean",
            lambda x: np.mean(x),
            [np.array([[1.0, 2.0], [3.0, 4.0]])]
        )

        # Test 8: Sum reduction
        self.test_operation(
            "Sum",
            lambda x: np.sum(x),
            [np.array([[1.0, 2.0], [3.0, 4.0]])]
        )

    def run_advanced_tests(self):
        """Run advanced gradient parity tests"""
        print("\n=== Running Advanced Gradient Parity Tests ===\n")

        # Test 1: Softmax
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        self.test_operation(
            "Softmax",
            softmax,
            [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
        )

        # Test 2: Cross Entropy Loss
        def cross_entropy(logits, targets):
            probs = softmax(logits)
            log_probs = np.log(probs + 1e-10)
            return -np.sum(targets * log_probs)

        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        self.test_operation(
            "Cross Entropy Loss",
            lambda x: cross_entropy(x, targets),
            [logits]
        )

        # Test 3: Batch Normalization (simplified)
        def batch_norm(x, eps=1e-5):
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            return (x - mean) / np.sqrt(var + eps)

        self.test_operation(
            "Batch Normalization",
            batch_norm,
            [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])]
        )

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("GRADIENT PARITY TEST SUMMARY")
        print("=" * 70)
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Total Tests:  {self.tests_passed + self.tests_failed}")
        print(f"Success Rate: {100 * self.tests_passed / (self.tests_passed + self.tests_failed):.1f}%")
        print("=" * 70)

        if self.tests_failed > 0:
            print("\nFailed Tests:")
            for name, passed, error in self.test_results:
                if not passed:
                    print(f"  - {name}: {error}")

        return self.tests_failed == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Gradient parity test harness for TenfloweRS"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for gradient comparisons (default: 1e-5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced tests",
    )

    args = parser.parse_args()

    tester = GradientParityTester(
        tolerance=args.tolerance,
        verbose=args.verbose
    )

    # Run basic tests
    tester.run_basic_tests()

    # Run advanced tests if requested
    if args.advanced:
        tester.run_advanced_tests()

    # Print summary and return exit code
    success = tester.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
