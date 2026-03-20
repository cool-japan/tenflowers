use tenflowers_core::{Tensor, Device, TensorError};
use tenflowers_neural::{
    Sequential, Dense, Adam, Model, Optimizer,
    categorical_cross_entropy
};
use tenflowers_dataset::{TensorDataset, Dataset};
use tenflowers_autograd::{GradientTape, TrackedTensor};
use scirs2_core::random::Random;

/// Generate synthetic MNIST-like data for demonstration
fn generate_synthetic_mnist(samples: usize) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = rand::thread_rng();
    
    // Generate random 28x28 images (flattened to 784)
    let mut x_data = vec![0.0f32; samples * 784];
    for i in 0..samples {
        for j in 0..784 {
            // Create simple patterns that correlate with digit classes
            let class = i % 10;
            let base_pattern = (class as f32) / 10.0;
            x_data[i * 784 + j] = base_pattern + rng.gen::<f32>() * 0.1;
        }
    }
    
    // Generate one-hot encoded labels
    let mut y_data = vec![0.0f32; samples * 10];
    for i in 0..samples {
        let class = i % 10; // Cycle through classes 0-9
        y_data[i * 10 + class] = 1.0;
    }
    
    let x_tensor = Tensor::from_vec(x_data, &[samples, 784]);
    let y_tensor = Tensor::from_vec(y_data, &[samples, 10]);
    
    (x_tensor, y_tensor)
}

/// Custom loss function that integrates with autograd
fn custom_categorical_crossentropy_loss(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> tenflowers_core::Result<Tensor<f32>> {
    categorical_cross_entropy(predictions, targets)
}

/// Simple training loop with basic dataset iteration
fn train_simple<M: Model<f32>>(
    model: &mut M,
    optimizer: &mut Adam<f32>,
    train_dataset: &TensorDataset<f32>,
    epochs: usize,
    batch_size: usize,
) -> tenflowers_core::Result<()> {
    let tape = GradientTape::new();
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;
        let dataset_len = train_dataset.len();
        
        // Simple batch iteration
        for batch_start in (0..dataset_len).step_by(batch_size) {
            batch_count += 1;
            let batch_end = (batch_start + batch_size).min(dataset_len);
            
            // Get a single sample from the batch (simplified)
            let (sample_x, sample_y) = train_dataset.get(batch_start)?;
            
            // Convert tensors to tracked tensors
            let x_tracked = tape.watch(sample_x);
            
            // Forward pass
            let predictions = model.forward(&x_tracked.tensor)?;
            let loss = custom_categorical_crossentropy_loss(&predictions, &sample_y)?;
            
            // Get scalar loss value
            let loss_value: f32 = *loss.get(&[]).unwrap_or(&0.0);
            epoch_loss += loss_value;
            
            // Compute gradients (simplified for demonstration)
            // In a real implementation, we would compute gradients with respect to model parameters
            // For now, we'll use the existing optimizer step method
            model.zero_grad();
            optimizer.step(model)?;
            
            if batch_count % 10 == 0 {
                println!("Epoch {}, Batch {}, Loss: {:.4}", epoch + 1, batch_count, loss_value);
            }
        }
        
        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f32 } else { 0.0 };
        println!("Epoch {} completed. Average Loss: {:.4}", epoch + 1, avg_loss);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS MNIST Example - Eager Mode with Training");
    println!("====================================================");

    // Create a simple feedforward network for MNIST
    let mut model = Sequential::new(vec![
        Box::new(Dense::new(784, 128, true).with_activation("relu".to_string())),
        Box::new(Dense::new(128, 64, true).with_activation("relu".to_string())),
        Box::new(Dense::new(64, 10, true).with_activation("softmax".to_string())),
    ]);

    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    println!("Generating synthetic MNIST-like data...");
    
    // Generate synthetic training and validation data
    let (x_train, y_train) = generate_synthetic_mnist(1000);
    let (x_val, y_val) = generate_synthetic_mnist(200);

    println!("Training data shape: {:?}", x_train.shape());
    println!("Training labels shape: {:?}", y_train.shape());

    // Create datasets
    let train_dataset = TensorDataset::new(x_train, y_train);
    let val_dataset = TensorDataset::new(x_val, y_val);

    println!("Model created successfully!");
    println!("Starting training...");

    // Training configuration
    let epochs = 5;
    let batch_size = 32;
    
    // Use simple training loop
    train_simple(&mut model, &mut optimizer, &train_dataset, epochs, batch_size)?;

    println!("Training completed!");
    
    // Simple validation
    println!("Running validation...");
    model.set_training(false);
    
    let mut val_loss = 0.0f32;
    let mut val_samples = 0;
    let val_dataset_len = val_dataset.len();
    
    // Validate on a subset of validation data
    for i in (0..val_dataset_len).step_by(batch_size).take(10) {
        let (val_x, val_y) = val_dataset.get(i)?;
        let predictions = model.forward(&val_x)?;
        let loss = custom_categorical_crossentropy_loss(&predictions, &val_y)?;
        val_loss += *loss.get(&[]).unwrap_or(&0.0);
        val_samples += 1;
    }
    
    let avg_val_loss = if val_samples > 0 { val_loss / val_samples as f32 } else { 0.0 };
    println!("Validation Loss: {:.4}", avg_val_loss);
    
    println!("Example completed successfully!");
    
    Ok(())
}