#!/usr/bin/env python3
"""
TenfloweRS Neural Network Example

This example demonstrates how to build and train neural networks using
TenfloweRS, including layers, optimizers, and the gradient tape.
"""

import tenflowers as tf
import numpy as np


def example_activation_functions():
    """Demonstrate various activation functions"""
    print("\n=== Activation Functions ===\n")

    input_tensor = tf.randn([2, 4])

    # Basic activations
    relu_out = tf.relu(input_tensor)
    print(f"ReLU output shape: {relu_out.shape()}")

    sigmoid_out = tf.sigmoid(input_tensor)
    print(f"Sigmoid output shape: {sigmoid_out.shape()}")

    tanh_out = tf.tanh(input_tensor)
    print(f"Tanh output shape: {tanh_out.shape()}")

    # Advanced activations
    gelu_out = tf.gelu(input_tensor)
    print(f"GELU output shape: {gelu_out.shape()}")

    swish_out = tf.swish(input_tensor)
    print(f"Swish output shape: {swish_out.shape()}")

    # Softmax
    logits = tf.randn([2, 10])
    softmax_out = tf.softmax(logits)
    print(f"Softmax output shape: {softmax_out.shape()}")


def example_layers():
    """Demonstrate neural network layers"""
    print("\n=== Neural Network Layers ===\n")

    # Dense layer
    dense = tf.PyDense(input_dim=10, output_dim=5)
    input_data = tf.randn([3, 10])
    output = dense.forward(input_data)
    print(f"Dense layer output shape: {output.shape()}")

    # Batch normalization
    batch_norm = tf.PyBatchNorm1d(num_features=5)
    bn_output = batch_norm.forward(output)
    print(f"BatchNorm output shape: {bn_output.shape()}")

    # Layer normalization
    layer_norm = tf.PyLayerNorm(normalized_shape=[5])
    ln_output = layer_norm.forward(output)
    print(f"LayerNorm output shape: {ln_output.shape()}")

    # Group normalization
    group_norm = tf.PyGroupNorm(num_groups=1, num_channels=5)
    # Reshape for group norm (needs 3D: batch, channels, length)
    reshaped = tf.reshape(output, [3, 5, 1])
    gn_output = group_norm.forward(reshaped)
    print(f"GroupNorm output shape: {gn_output.shape()}")


def example_convolutional_layers():
    """Demonstrate convolutional layers"""
    print("\n=== Convolutional Layers ===\n")

    # Conv1D
    conv1d = tf.PyConv1D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    )
    input_1d = tf.randn([2, 3, 10])  # (batch, channels, length)
    output_1d = conv1d.forward(input_1d)
    print(f"Conv1D output shape: {output_1d.shape()}")

    # Conv2D
    conv2d = tf.PyConv2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    )
    input_2d = tf.randn([2, 3, 28, 28])  # (batch, channels, height, width)
    output_2d = conv2d.forward(input_2d)
    print(f"Conv2D output shape: {output_2d.shape()}")

    # Max pooling
    max_pool = tf.PyMaxPool2D(kernel_size=2, stride=2)
    pooled = max_pool.forward(output_2d)
    print(f"MaxPool2D output shape: {pooled.shape()}")


def example_recurrent_layers():
    """Demonstrate recurrent layers"""
    print("\n=== Recurrent Layers ===\n")

    # LSTM
    lstm = tf.PyLSTM(input_size=10, hidden_size=20, num_layers=2)
    input_seq = tf.randn([5, 3, 10])  # (seq_len, batch, input_size)
    output, (hidden, cell) = lstm.forward(input_seq)
    print(f"LSTM output shape: {output.shape()}")
    print(f"LSTM hidden shape: {hidden.shape()}")

    # GRU
    gru = tf.PyGRU(input_size=10, hidden_size=20, num_layers=2)
    output, hidden = gru.forward(input_seq)
    print(f"GRU output shape: {output.shape()}")
    print(f"GRU hidden shape: {hidden.shape()}")


def example_attention():
    """Demonstrate attention mechanisms"""
    print("\n=== Attention Mechanisms ===\n")

    # Multi-head attention
    mha = tf.PyMultiheadAttention(embed_dim=512, num_heads=8)

    # Create query, key, value tensors
    seq_len, batch_size, embed_dim = 10, 2, 512
    query = tf.randn([seq_len, batch_size, embed_dim])
    key = tf.randn([seq_len, batch_size, embed_dim])
    value = tf.randn([seq_len, batch_size, embed_dim])

    output, weights = mha.forward(query, key, value)
    print(f"MultiheadAttention output shape: {output.shape()}")
    if weights is not None:
        print(f"Attention weights shape: {weights.shape()}")


def example_mamba_ssm():
    """Demonstrate Mamba/State Space Model layers"""
    print("\n=== Mamba/SSM Layers ===\n")

    # Mamba layer
    mamba = tf.PyMamba(d_model=256, d_state=16, expand_factor=2)
    input_seq = tf.randn([2, 10, 256])  # (batch, seq_len, d_model)
    output, hidden = mamba.forward(input_seq)
    print(f"Mamba output shape: {output.shape()}")
    print(f"Mamba hidden state shape: {hidden.shape()}")

    # Simple State Space Model
    ssm = tf.PyStateSpaceModel(d_model=128, d_state=16)
    input_seq = tf.randn([2, 10, 128])
    output, hidden = ssm.forward(input_seq)
    print(f"SSM output shape: {output.shape()}")
    print(f"SSM hidden state shape: {hidden.shape()}")


def example_optimizers():
    """Demonstrate various optimizers"""
    print("\n=== Optimizers ===\n")

    # Adam optimizer
    adam = tf.PyAdam(learning_rate=0.001)
    print(f"Adam optimizer: {adam}")
    print(f"Learning rate: {adam.get_learning_rate()}")

    # AdamW optimizer
    adamw = tf.PyAdamW(learning_rate=0.001)
    print(f"\nAdamW optimizer: {adamw}")

    # SGD with momentum
    sgd = tf.PySGD.with_momentum(learning_rate=0.01, momentum=0.9)
    print(f"\nSGD with momentum: {sgd}")

    # RMSprop
    rmsprop = tf.PyRMSprop(learning_rate=0.001)
    print(f"\nRMSprop optimizer: {rmsprop}")

    # Extended optimizers
    adabelief = tf.PyAdaBelief(learning_rate=0.001)
    print(f"\nAdaBelief optimizer: {adabelief}")

    radam = tf.PyRAdam(learning_rate=0.001)
    print(f"\nRAdam optimizer: {radam}")

    nadam = tf.PyNadam(learning_rate=0.001)
    print(f"\nNadam optimizer: {nadam}")


def example_learning_rate_schedulers():
    """Demonstrate learning rate schedulers"""
    print("\n=== Learning Rate Schedulers ===\n")

    # Step LR
    step_lr = tf.PyStepLR(initial_lr=0.1, step_size=10, gamma=0.1)
    print(f"StepLR initial: {step_lr.get_lr()}")
    step_lr.step()
    print(f"StepLR after step: {step_lr.get_lr()}")

    # Exponential LR
    exp_lr = tf.PyExponentialLR(initial_lr=0.1, gamma=0.95)
    print(f"\nExponentialLR initial: {exp_lr.get_lr()}")
    exp_lr.step()
    print(f"ExponentialLR after step: {exp_lr.get_lr()}")

    # Cosine Annealing LR
    cosine_lr = tf.PyCosineAnnealingLR(initial_lr=0.1, T_max=100)
    print(f"\nCosineAnnealingLR initial: {cosine_lr.get_lr()}")


def example_loss_functions():
    """Demonstrate loss functions"""
    print("\n=== Loss Functions ===\n")

    # Mean Squared Error
    predictions = tf.randn([10, 1])
    targets = tf.randn([10, 1])
    mse = tf.mse_loss(predictions, targets)
    print(f"MSE Loss shape: {mse.shape()}")

    # Binary Cross Entropy
    predictions = tf.sigmoid(tf.randn([10, 1]))
    targets = tf.rand([10, 1])
    bce = tf.binary_cross_entropy(predictions, targets)
    print(f"BCE Loss shape: {bce.shape()}")

    # Cross Entropy
    logits = tf.randn([10, 5])
    targets = tf.rand([10, 5])
    ce = tf.cross_entropy(logits, targets)
    print(f"CE Loss shape: {ce.shape()}")


def example_gradient_tape():
    """Demonstrate gradient computation"""
    print("\n=== Gradient Tape ===\n")

    # Create gradient tape
    tape = tf.create_gradient_tape()
    print("Created gradient tape")

    # Note: Actual gradient computation requires full integration
    # This demonstrates the API structure
    print("Gradient tape API ready for use")


def main():
    """Run all examples"""
    print("=" * 70)
    print("TenfloweRS Neural Network Examples")
    print("=" * 70)

    example_activation_functions()
    example_layers()
    example_convolutional_layers()
    example_recurrent_layers()
    example_attention()
    example_mamba_ssm()
    example_optimizers()
    example_learning_rate_schedulers()
    example_loss_functions()
    example_gradient_tape()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
