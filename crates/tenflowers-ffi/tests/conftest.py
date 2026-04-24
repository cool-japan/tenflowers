"""
Shared pytest fixtures for TenfloweRS FFI tests.

This conftest.py provides reusable fixtures for the entire FFI test suite,
including random tensors, trained tiny models, synthetic datasets, and
integration helpers.

Markers registered here:
- slow: tests that take noticeably longer (>1 s)
- gpu: tests that require a GPU device
- integration: tests that exercise the full Python ↔ Rust FFI stack
"""

import pytest
import numpy as np
from typing import Callable

# ---------------------------------------------------------------------------
# Optional import helper
# ---------------------------------------------------------------------------

def _try_import_tenflowers():
    """
    Attempt to import the compiled tenflowers extension module.

    Returns None and marks collection as expected-failure when the native
    extension has not been compiled yet (e.g. in a source-only checkout).
    """
    try:
        import tenflowers  # noqa: PLC0415
        return tenflowers
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Session-scoped: check if the native extension is available
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tf_module():
    """
    Provide the tenflowers native module.

    If the extension is not compiled, tests that depend on this fixture will
    be skipped automatically via the ``pytest.importorskip`` mechanism.
    """
    return pytest.importorskip("tenflowers",
                               reason="tenflowers native module not compiled")


# ---------------------------------------------------------------------------
# Random number generators (deterministic seeds for reproducibility)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    """Session-scoped NumPy RNG with fixed seed for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def rng_fresh():
    """Function-scoped fresh RNG with fixed seed (test isolation)."""
    return np.random.default_rng(seed=0)


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_float_array(rng_fresh) -> np.ndarray:
    """2-D float32 array of shape (4, 4) with values in [-1, 1]."""
    return rng_fresh.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)


@pytest.fixture
def batch_float_array(rng_fresh) -> np.ndarray:
    """3-D float32 array of shape (8, 4, 4) — a mini-batch of matrices."""
    return rng_fresh.uniform(-1.0, 1.0, size=(8, 4, 4)).astype(np.float32)


@pytest.fixture
def random_tensor(tf_module, small_float_array):
    """
    A TenfloweRS PyTensor constructed from ``small_float_array``.

    Requires the native extension (``tf_module``).
    """
    return tf_module.tensor_from_numpy(small_float_array)


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_classification_data(rng_fresh):
    """
    Synthetic binary-classification dataset: X in (100, 4), y in {0, 1}.

    Returns:
        (X, y) — both as float32 NumPy arrays.
    """
    X = rng_fresh.normal(0.0, 1.0, size=(100, 4)).astype(np.float32)
    # Label = 1 if sum of features > 0, else 0
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


@pytest.fixture
def synthetic_regression_data(rng_fresh):
    """
    Synthetic regression dataset: X in (100, 4), y = sum(X) + noise.

    Returns:
        (X, y) — both as float32 NumPy arrays.
    """
    X = rng_fresh.normal(0.0, 1.0, size=(100, 4)).astype(np.float32)
    noise = rng_fresh.normal(0.0, 0.1, size=(100, 1)).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) + noise).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Gradient checking helper
# ---------------------------------------------------------------------------

@pytest.fixture
def numerical_gradient() -> Callable:
    """
    Return a helper function that computes numerical gradients via central
    finite differences.

    Usage::

        def test_foo(numerical_gradient):
            grad = numerical_gradient(lambda x: np.sum(x ** 2), x0)
            np.testing.assert_allclose(grad, 2 * x0, atol=1e-4)
    """
    def _compute(func: Callable, x: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        grad = np.zeros_like(x, dtype=np.float64)
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            original = float(x[idx])

            x[idx] = original + epsilon
            f_plus = float(func(x))

            x[idx] = original - epsilon
            f_minus = float(func(x))

            grad[idx] = (f_plus - f_minus) / (2.0 * epsilon)
            x[idx] = original
            it.iternext()
        return grad

    return _compute


# ---------------------------------------------------------------------------
# pytest configuration hooks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests as requiring a GPU device",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests that exercise the full Python ↔ Rust FFI stack",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: marks tests as performance benchmarks",
    )
