#!/usr/bin/env python3
"""
TenfloweRS Basic Tensor Operations Example

This example demonstrates the fundamental tensor operations available
in TenfloweRS, including creation, manipulation, and mathematical operations.
"""

import tenflowers as tf
import numpy as np


def example_tensor_creation():
    """Demonstrate various ways to create tensors"""
    print("\n=== Tensor Creation ===\n")

    # Create tensors with specific values
    zeros = tf.zeros([3, 4])
    print(f"Zeros tensor shape: {zeros.shape()}")

    ones = tf.ones([2, 3])
    print(f"Ones tensor shape: {ones.shape()}")

    # Create random tensors
    random_uniform = tf.rand([3, 3])
    print(f"Random uniform tensor shape: {random_uniform.shape()}")

    random_normal = tf.randn([2, 2])
    print(f"Random normal tensor shape: {random_normal.shape()}")

    # Create from numpy array
    np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    from_numpy = tf.tensor_from_numpy(np_array)
    print(f"From numpy tensor shape: {from_numpy.shape()}")


def example_basic_operations():
    """Demonstrate basic tensor operations"""
    print("\n=== Basic Operations ===\n")

    # Create sample tensors
    a = tf.ones([3, 3])
    b = tf.ones([3, 3])

    # Arithmetic operations
    c = tf.add(a, b)
    print(f"Addition result shape: {c.shape()}")

    d = tf.mul(a, b)
    print(f"Multiplication result shape: {d.shape()}")

    e = tf.sub(a, b)
    print(f"Subtraction result shape: {e.shape()}")

    # Matrix operations
    x = tf.ones([2, 3])
    y = tf.ones([3, 4])
    z = tf.matmul(x, y)
    print(f"Matrix multiplication result shape: {z.shape()}")

    # Transpose
    t = tf.transpose(x)
    print(f"Transpose result shape: {t.shape()}")


def example_shape_manipulation():
    """Demonstrate shape manipulation operations"""
    print("\n=== Shape Manipulation ===\n")

    # Create a tensor
    tensor = tf.ones([2, 3, 4])
    print(f"Original shape: {tensor.shape()}")

    # Reshape
    reshaped = tf.reshape(tensor, [6, 4])
    print(f"Reshaped: {reshaped.shape()}")

    # Squeeze and unsqueeze
    squeezed = tf.squeeze(tf.ones([1, 3, 1, 4]))
    print(f"Squeezed shape: {squeezed.shape()}")

    unsqueezed = tf.unsqueeze(tf.ones([3, 4]), 0)
    print(f"Unsqueezed shape: {unsqueezed.shape()}")

    # Concatenation
    a = tf.ones([2, 3])
    b = tf.ones([2, 3])
    cat_result = tf.cat([a, b], axis=0)
    print(f"Concatenated shape: {cat_result.shape()}")


def example_reduction_operations():
    """Demonstrate reduction operations"""
    print("\n=== Reduction Operations ===\n")

    tensor = tf.rand([3, 4])

    # Various reductions
    sum_result = tf.sum(tensor)
    print(f"Sum: {sum_result}")

    mean_result = tf.mean(tensor)
    print(f"Mean: {mean_result}")

    max_result = tf.max(tensor)
    print(f"Max: {max_result}")

    min_result = tf.min(tensor)
    print(f"Min: {min_result}")


def example_mathematical_functions():
    """Demonstrate mathematical functions"""
    print("\n=== Mathematical Functions ===\n")

    tensor = tf.ones([2, 2])

    # Exponential and logarithm
    exp_result = tf.exp(tensor)
    print(f"Exp shape: {exp_result.shape()}")

    log_result = tf.log(tf.ones([2, 2]) * 2.0)  # log(2)
    print(f"Log shape: {log_result.shape()}")

    # Trigonometric
    sin_result = tf.sin(tensor)
    print(f"Sin shape: {sin_result.shape()}")

    cos_result = tf.cos(tensor)
    print(f"Cos shape: {cos_result.shape()}")

    # Other
    sqrt_result = tf.sqrt(tensor)
    print(f"Sqrt shape: {sqrt_result.shape()}")

    abs_result = tf.abs(tensor)
    print(f"Abs shape: {abs_result.shape()}")


def example_dtype_system():
    """Demonstrate the dtype system"""
    print("\n=== DType System ===\n")

    # Access dtype constants
    print(f"Float32: {tf.float32}")
    print(f"Float64: {tf.float64}")
    print(f"Int32: {tf.int32}")
    print(f"BFloat16: {tf.bfloat16}")

    # Check dtype properties
    print(f"\nFloat32 is floating point: {tf.float32.is_floating_point()}")
    print(f"Int32 is integer: {tf.int32.is_integer()}")
    print(f"Float32 is supported: {tf.float32.is_supported()}")

    # Type promotion
    result_dtype = tf.result_type(tf.float32, tf.float64)
    print(f"\nResult type of float32 + float64: {result_dtype}")

    # Safe casting
    is_safe = tf.is_safe_cast_py(tf.float32, tf.float64)
    print(f"Is safe to cast float32 -> float64: {is_safe}")


def main():
    """Run all examples"""
    print("=" * 70)
    print("TenfloweRS Basic Tensor Operations Examples")
    print("=" * 70)

    example_tensor_creation()
    example_basic_operations()
    example_shape_manipulation()
    example_reduction_operations()
    example_mathematical_functions()
    example_dtype_system()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
