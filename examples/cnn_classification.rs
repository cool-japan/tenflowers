use tenflowers_core::{Tensor, Device};
use tenflowers_neural::{
    Sequential, Dense, Conv2D, MaxPool2D, Adam, Model, Optimizer,
    categorical_cross_entropy, ReLU, Dropout
};
use tenflowers_dataset::{TensorDataset, Dataset};
use scirs2_core::random::Random;

/// Generate synthetic image classification data
fn generate_synthetic_image_data(samples: usize) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = rand::thread_rng();
    
    // Generate random 32x32 RGB images (NCHW format: [batch, channels, height, width])
    let mut x_data = vec![0.0f32; samples * 3 * 32 * 32];
    for i in 0..samples {
        let class = i % 5; // 5 classes
        
        // Create class-specific patterns
        for c in 0..3 {
            for h in 0..32 {
                for w in 0..32 {
                    let idx = i * (3 * 32 * 32) + c * (32 * 32) + h * 32 + w;
                    
                    // Create simple patterns based on class
                    let base_value = match class {
                        0 => if c == 0 { 0.8 } else { 0.2 }, // Red-ish
                        1 => if c == 1 { 0.8 } else { 0.2 }, // Green-ish
                        2 => if c == 2 { 0.8 } else { 0.2 }, // Blue-ish
                        3 => 0.7, // Gray-ish
                        _ => 0.3,  // Dark
                    };
                    
                    // Add some spatial patterns
                    let spatial_pattern = if (h / 4 + w / 4) % 2 == 0 { 0.1 } else { -0.1 };
                    
                    x_data[idx] = base_value + spatial_pattern + rng.gen::<f32>() * 0.1;
                }
            }
        }
    }
    
    // Generate one-hot encoded labels
    let mut y_data = vec![0.0f32; samples * 5];
    for i in 0..samples {
        let class = i % 5;
        y_data[i * 5 + class] = 1.0;
    }
    
    let x_tensor = Tensor::from_vec(x_data, &[samples, 3, 32, 32]);
    let y_tensor = Tensor::from_vec(y_data, &[samples, 5]);
    
    (x_tensor, y_tensor)
}

/// Simple CNN architecture for image classification
fn create_cnn_model() -> Sequential<f32> {
    Sequential::new(vec![
        // First convolutional block
        Box::new(Conv2D::new(3, 32, 3, 1, "same".to_string(), true)), // 3->32 channels, 3x3 kernel
        Box::new(ReLU::new()),
        Box::new(MaxPool2D::new(2, 2)), // 32x32 -> 16x16
        
        // Second convolutional block
        Box::new(Conv2D::new(32, 64, 3, 1, "same".to_string(), true)), // 32->64 channels
        Box::new(ReLU::new()),
        Box::new(MaxPool2D::new(2, 2)), // 16x16 -> 8x8
        
        // Third convolutional block
        Box::new(Conv2D::new(64, 128, 3, 1, "same".to_string(), true)), // 64->128 channels
        Box::new(ReLU::new()),
        Box::new(MaxPool2D::new(2, 2)), // 8x8 -> 4x4
        
        // Flatten and fully connected layers
        // Note: In practice, we'd need a Flatten layer here
        // For this example, we'll assume the model handles this internally
        Box::new(Dense::new(128 * 4 * 4, 256, true).with_activation("relu".to_string())),
        Box::new(Dropout::new(0.5)),
        Box::new(Dense::new(256, 128, true).with_activation("relu".to_string())),
        Box::new(Dropout::new(0.3)),
        Box::new(Dense::new(128, 5, true).with_activation("softmax".to_string())), // 5 classes
    ])
}

/// Training loop for CNN
fn train_cnn<M: Model<f32>>(
    model: &mut M,
    optimizer: &mut Adam<f32>,
    train_dataset: &TensorDataset<f32>,
    epochs: usize,
    batch_size: usize,
) -> tenflowers_core::Result<()> {
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;
        let dataset_len = train_dataset.len();
        
        model.set_training(true);
        
        // Training loop
        for batch_start in (0..dataset_len).step_by(batch_size) {
            batch_count += 1;
            
            // Get sample from dataset
            let (sample_x, sample_y) = train_dataset.get(batch_start)?;
            
            // Forward pass
            let predictions = model.forward(&sample_x)?;
            let loss = categorical_cross_entropy(&predictions, &sample_y)?;
            
            // Get scalar loss value
            let loss_value: f32 = *loss.get(&[]).unwrap_or(&0.0);
            epoch_loss += loss_value;
            
            // Backward pass and optimization
            model.zero_grad();
            optimizer.step(model)?;
            
            if batch_count % 20 == 0 {
                println!("Epoch {}, Batch {}, Loss: {:.4}", epoch + 1, batch_count, loss_value);
            }
        }
        
        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f32 } else { 0.0 };
        println!("Epoch {} completed. Average Loss: {:.4}", epoch + 1, avg_loss);
    }
    
    Ok(())
}

/// Calculate accuracy on dataset
fn evaluate_accuracy<M: Model<f32>>(
    model: &mut M,
    dataset: &TensorDataset<f32>,
    num_samples: usize,
) -> tenflowers_core::Result<f32> {
    model.set_training(false);
    
    let mut correct = 0;
    let mut total = 0;
    let dataset_len = dataset.len().min(num_samples);
    
    for i in 0..dataset_len {
        let (sample_x, sample_y) = dataset.get(i)?;
        let predictions = model.forward(&sample_x)?;
        
        // Find predicted class (argmax)
        let pred_data = predictions.data();
        let pred_class = pred_data.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        // Find true class
        let true_data = sample_y.data();
        let true_class = true_data.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        if pred_class == true_class {
            correct += 1;
        }
        total += 1;
    }
    
    Ok(correct as f32 / total as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS CNN Classification Example");
    println!("=====================================");

    // Generate synthetic data
    println!("Generating synthetic image data...");
    let (x_train, y_train) = generate_synthetic_image_data(1000);
    let (x_val, y_val) = generate_synthetic_image_data(200);

    println!("Training data shape: {:?}", x_train.shape());
    println!("Training labels shape: {:?}", y_train.shape());

    // Create CNN model
    println!("Creating CNN model...");
    let mut model = create_cnn_model();

    // Create optimizer
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Create datasets
    let train_dataset = TensorDataset::new(x_train, y_train);
    let val_dataset = TensorDataset::new(x_val, y_val);

    println!("Starting training...");
    
    // Training configuration
    let epochs = 10;
    let batch_size = 16; // Smaller batch size for image data
    
    // Train the model
    train_cnn(&mut model, &mut optimizer, &train_dataset, epochs, batch_size)?;

    println!("Training completed!");
    
    // Evaluate accuracy
    println!("Evaluating model...");
    
    let train_accuracy = evaluate_accuracy(&mut model, &train_dataset, 100)?;
    println!("Training Accuracy (100 samples): {:.2}%", train_accuracy * 100.0);
    
    let val_accuracy = evaluate_accuracy(&mut model, &val_dataset, 100)?;
    println!("Validation Accuracy (100 samples): {:.2}%", val_accuracy * 100.0);
    
    println!("CNN Classification example completed successfully!");
    
    Ok(())
}