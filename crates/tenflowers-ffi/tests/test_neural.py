"""
Neural network functionality tests for TenfloweRS Python bindings.
"""

import pytest


def test_activation_functions():
    """Test neural network activation functions."""
    import tenflowers.neural as nn
    import tenflowers as tf

    x = tf.ones([2, 2])

    # ReLU
    result = nn.relu(x)
    assert result.shape() == [2, 2]

    # Sigmoid
    result = nn.sigmoid(x)
    assert result.shape() == [2, 2]

    # Tanh
    result = nn.tanh(x)
    assert result.shape() == [2, 2]


def test_gelu_activation():
    """Test GELU activation function."""
    import tenflowers.neural as nn
    import tenflowers as tf

    x = tf.randn([3, 3])
    result = nn.gelu(x)
    assert result.shape() == [3, 3]


def test_swish_activation():
    """Test Swish activation function."""
    import tenflowers.neural as nn
    import tenflowers as tf

    x = tf.randn([3, 3])
    result = nn.swish(x)
    assert result.shape() == [3, 3]


def test_mish_activation():
    """Test Mish activation function."""
    import tenflowers.neural as nn
    import tenflowers as tf

    x = tf.randn([3, 3])
    result = nn.mish(x)
    assert result.shape() == [3, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
