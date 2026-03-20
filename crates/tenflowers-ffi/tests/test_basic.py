"""
Basic functionality tests for TenfloweRS Python bindings.
"""

import pytest
import numpy as np


def test_import():
    """Test that tenflowers can be imported."""
    import tenflowers as tf
    assert tf.__version__ is not None


def test_tensor_creation():
    """Test basic tensor creation functions."""
    import tenflowers as tf

    # Test zeros
    t = tf.zeros([2, 3])
    assert t.shape() == [2, 3]

    # Test ones
    t = tf.ones([3, 3])
    assert t.shape() == [3, 3]

    # Test rand
    t = tf.rand([2, 2])
    assert t.shape() == [2, 2]

    # Test randn
    t = tf.randn([4, 4])
    assert t.shape() == [4, 4]


def test_basic_operations():
    """Test basic tensor operations."""
    import tenflowers as tf

    a = tf.ones([2, 2])
    b = tf.ones([2, 2])

    # Addition
    c = tf.add(a, b)
    assert c.shape() == [2, 2]

    # Multiplication
    c = tf.mul(a, b)
    assert c.shape() == [2, 2]

    # Subtraction
    c = tf.sub(a, b)
    assert c.shape() == [2, 2]

    # Division
    c = tf.div(a, b)
    assert c.shape() == [2, 2]


def test_matmul():
    """Test matrix multiplication."""
    import tenflowers as tf

    a = tf.ones([2, 3])
    b = tf.ones([3, 4])
    c = tf.matmul(a, b)

    assert c.shape() == [2, 4]


def test_transpose():
    """Test tensor transpose."""
    import tenflowers as tf

    a = tf.ones([2, 3])
    b = tf.transpose(a)

    assert b.shape() == [3, 2]


def test_reshape():
    """Test tensor reshape."""
    import tenflowers as tf

    a = tf.ones([2, 3])
    b = tf.reshape(a, [3, 2])

    assert b.shape() == [3, 2]


def test_mathematical_operations():
    """Test mathematical operations."""
    import tenflowers as tf

    a = tf.ones([2, 2])

    # Test exp
    result = tf.exp(a)
    assert result.shape() == [2, 2]

    # Test log
    result = tf.log(a)
    assert result.shape() == [2, 2]

    # Test sqrt
    result = tf.sqrt(a)
    assert result.shape() == [2, 2]

    # Test abs
    result = tf.abs(a)
    assert result.shape() == [2, 2]


def test_reduction_operations():
    """Test reduction operations."""
    import tenflowers as tf

    a = tf.ones([3, 3])

    # Sum
    result = tf.sum(a)
    assert result.shape() == []  # Scalar

    # Mean
    result = tf.mean(a)
    assert result.shape() == []

    # Max
    result = tf.max(a)
    assert result.shape() == []

    # Min
    result = tf.min(a)
    assert result.shape() == []


def test_tensor_manipulation():
    """Test tensor manipulation operations."""
    import tenflowers as tf

    a = tf.ones([2, 3])
    b = tf.ones([2, 3])

    # Cat
    c = tf.cat([a, b], dim=0)
    assert c.shape() == [4, 3]

    # Stack
    c = tf.stack([a, b])
    assert c.shape() == [2, 2, 3]

    # Squeeze (add and remove dimension)
    a_unsqueezed = tf.unsqueeze(a, dim=0)
    assert a_unsqueezed.shape() == [1, 2, 3]

    a_squeezed = tf.squeeze(a_unsqueezed)
    assert a_squeezed.shape() == [2, 3]

    # Flatten
    a_flat = tf.flatten(a)
    assert a_flat.shape() == [6]


def test_numpy_interop():
    """Test NumPy interoperability."""
    import tenflowers as tf
    import numpy as np

    # NumPy to tensor
    np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = tf.tensor_from_numpy(np_array)
    assert tensor.shape() == [2, 2]

    # Tensor to NumPy
    back_to_numpy = tf.tensor_to_numpy(tensor)
    assert back_to_numpy.shape == (2, 2)
    assert np.allclose(back_to_numpy, np_array)


def test_device_management():
    """Test device management."""
    import tenflowers as tf

    # Get default device
    device = tf.get_default_device()
    assert device in ["cpu", "gpu:0", "gpu:1"]  # Depends on availability

    # Set default device
    tf.set_default_device("cpu")
    assert tf.get_default_device() == "cpu"


def test_gradient_management():
    """Test gradient management."""
    import tenflowers as tf

    # Check gradient state
    initial_state = tf.is_grad_enabled()

    # Disable gradients
    tf.set_grad_enabled(False)
    assert not tf.is_grad_enabled()

    # Enable gradients
    tf.set_grad_enabled(True)
    assert tf.is_grad_enabled()

    # Restore initial state
    tf.set_grad_enabled(initial_state)


def test_memory_profiling():
    """Test memory profiling."""
    import tenflowers as tf

    # Enable profiling
    tf.enable_memory_profiling()

    # Create some tensors
    a = tf.ones([100, 100])
    b = tf.ones([100, 100])
    c = tf.matmul(a, b)

    # Get memory info
    current_mem, peak_mem = tf.get_memory_info()
    assert current_mem >= 0
    assert peak_mem >= 0

    # Disable profiling
    tf.disable_memory_profiling()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
