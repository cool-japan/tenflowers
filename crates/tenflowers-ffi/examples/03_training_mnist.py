#!/usr/bin/env python3
"""
TenfloweRS MNIST Training Example

This example demonstrates how to train a simple neural network on the MNIST dataset
using TenfloweRS. It showcases:
- Building a sequential model
- Training loop with optimizer
- Loss computation
- Accuracy evaluation
"""

import tenflowers as tf
import numpy as np


class SimpleMNISTModel:
    """Simple feedforward neural network for MNIST"""

    def __init__(self):
        # Build a simple 3-layer network
        self.layer1 = tf.PyDense(input_dim=784, output_dim=128)
        self.layer2 = tf.PyDense(input_dim=128, output_dim=64)
        self.layer3 = tf.PyDense(input_dim=64, output_dim=10)

        # Normalization layers
        self.bn1 = tf.PyBatchNorm1d(num_features=128)
        self.bn2 = tf.PyBatchNorm1d(num_features=64)

        # Dropout for regularization
        self.dropout1 = tf.PyDropout(p=0.2)
        self.dropout2 = tf.PyDropout(p=0.2)

    def forward(self, x):
        """Forward pass through the network"""
        # Layer 1
        x = self.layer1.forward(x)
        x = self.bn1.forward(x)
        x = tf.relu(x)
        x = self.dropout1.forward(x)

        # Layer 2
        x = self.layer2.forward(x)
        x = self.bn2.forward(x)
        x = tf.relu(x)
        x = self.dropout2.forward(x)

        # Layer 3 (output)
        x = self.layer3.forward(x)
        return x

    def train_mode(self):
        """Set model to training mode"""
        self.bn1.train()
        self.bn2.train()
        self.dropout1.train()
        self.dropout2.train()

    def eval_mode(self):
        """Set model to evaluation mode"""
        self.bn1.eval()
        self.bn2.eval()
        self.dropout1.eval()
        self.dropout2.eval()


def generate_dummy_mnist_data(num_samples=1000):
    """
    Generate dummy MNIST-like data for demonstration
    In practice, you would load real MNIST data
    """
    # Random images (28x28 flattened to 784)
    images = np.random.randn(num_samples, 784).astype(np.float32)
    # Random labels (one-hot encoded, 10 classes)
    labels_idx = np.random.randint(0, 10, num_samples)
    labels = np.zeros((num_samples, 10), dtype=np.float32)
    labels[np.arange(num_samples), labels_idx] = 1.0

    return images, labels


def compute_accuracy(predictions, targets):
    """Compute classification accuracy"""
    # Convert tensors to numpy for evaluation
    pred_np = tf.tensor_to_numpy(predictions)
    target_np = tf.tensor_to_numpy(targets)

    # Get predicted classes
    pred_classes = np.argmax(pred_np, axis=1)
    target_classes = np.argmax(target_np, axis=1)

    # Compute accuracy
    accuracy = np.mean(pred_classes == target_classes)
    return accuracy


def train_epoch(model, optimizer, train_data, train_labels, batch_size=32):
    """Train for one epoch"""
    model.train_mode()

    num_samples = len(train_data)
    num_batches = num_samples // batch_size

    total_loss = 0.0
    total_accuracy = 0.0

    for i in range(num_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]

        # Convert to tensors
        inputs = tf.tensor_from_numpy(batch_data)
        targets = tf.tensor_from_numpy(batch_labels)

        # Forward pass
        outputs = model.forward(inputs)

        # Compute loss
        loss = tf.cross_entropy(outputs, targets)

        # Compute accuracy
        accuracy = compute_accuracy(outputs, targets)

        total_loss += loss
        total_accuracy += accuracy

        # In a full implementation, we would:
        # 1. Compute gradients with gradient tape
        # 2. Apply optimizer step
        # 3. Zero gradients
        # For now, we demonstrate the structure

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def evaluate(model, test_data, test_labels, batch_size=32):
    """Evaluate model on test data"""
    model.eval_mode()

    num_samples = len(test_data)
    num_batches = num_samples // batch_size

    total_loss = 0.0
    total_accuracy = 0.0

    for i in range(num_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_data = test_data[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]

        # Convert to tensors
        inputs = tf.tensor_from_numpy(batch_data)
        targets = tf.tensor_from_numpy(batch_labels)

        # Forward pass
        outputs = model.forward(inputs)

        # Compute loss and accuracy
        loss = tf.cross_entropy(outputs, targets)
        accuracy = compute_accuracy(outputs, targets)

        total_loss += loss
        total_accuracy += accuracy

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def train_model():
    """Main training function"""
    print("=" * 70)
    print("TenfloweRS MNIST Training Example")
    print("=" * 70)

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Generate dummy data
    print("\nGenerating dummy training data...")
    train_data, train_labels = generate_dummy_mnist_data(num_samples=1000)
    test_data, test_labels = generate_dummy_mnist_data(num_samples=200)
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create model
    print("\nInitializing model...")
    model = SimpleMNISTModel()
    print("Model architecture:")
    print("  Input: 784 (28x28 flattened)")
    print("  Hidden 1: 128 + BatchNorm + ReLU + Dropout(0.2)")
    print("  Hidden 2: 64 + BatchNorm + ReLU + Dropout(0.2)")
    print("  Output: 10 (classes)")

    # Create optimizer
    print(f"\nInitializing optimizer (Adam, lr={learning_rate})...")
    optimizer = tf.PyAdam(learning_rate=learning_rate)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, optimizer, train_data, train_labels, batch_size
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_data, test_labels, batch_size
        )

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


def demonstrate_advanced_features():
    """Demonstrate advanced training features"""
    print("\n" + "=" * 70)
    print("Advanced Features")
    print("=" * 70)

    # Learning rate scheduling
    print("\n1. Learning Rate Scheduling:")
    scheduler = tf.PyStepLR(initial_lr=0.1, step_size=5, gamma=0.5)
    print(f"   Initial LR: {scheduler.get_lr():.4f}")
    for i in range(10):
        scheduler.step()
        if (i + 1) % 5 == 0:
            print(f"   After {i + 1} steps: {scheduler.get_lr():.4f}")

    # Different optimizers
    print("\n2. Optimizer Variants:")
    optimizers = [
        ("Adam", tf.PyAdam(learning_rate=0.001)),
        ("AdamW", tf.PyAdamW(learning_rate=0.001)),
        ("SGD+Momentum", tf.PySGD.with_momentum(0.01, 0.9)),
        ("RMSprop", tf.PyRMSprop(learning_rate=0.001)),
        ("AdaBelief", tf.PyAdaBelief(learning_rate=0.001)),
        ("RAdam", tf.PyRAdam(learning_rate=0.001)),
    ]

    for name, opt in optimizers:
        print(f"   {name}: {opt}")

    # Early stopping
    print("\n3. Early Stopping:")
    early_stop = tf.PyEarlyStopping(patience=5, min_delta=0.001)
    print(f"   Patience: {early_stop.patience}")
    print(f"   Min delta: {early_stop.min_delta}")

    # Metrics tracking
    print("\n4. Metrics Tracking:")
    metrics = tf.PyMetricsTracker()
    metrics.add_metric("loss", 0.5)
    metrics.add_metric("accuracy", 0.85)
    print("   Metrics tracked successfully")


def main():
    """Run the complete example"""
    try:
        # Run training
        train_model()

        # Demonstrate advanced features
        demonstrate_advanced_features()

        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during training: {e}")
        print("Note: This is a demonstration of the API structure.")
        print("Full training requires gradient tape integration.")


if __name__ == "__main__":
    main()
