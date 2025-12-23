use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::time::Instant;
/// Advanced Autograd Training Demo using TenfloweRS
///
/// This example demonstrates:
/// - Complete autograd integration with GradientTape
/// - Real neural network training with automatic differentiation  
/// - Multiple optimizers (Adam, SGD) comparison
/// - Training/validation split with proper metrics
/// - Callback system (early stopping, learning rate reduction)
/// - Loss function integration with autograd
use tenflowers_core::{Result as TensorResult, Tensor};
use tenflowers_neural::{
    layers::{Dense, Layer},
    loss::mse,
    model::{Model, Sequential},
    optimizers::{Adam, Optimizer, SGD},
    trainer::{EarlyStopping, LearningRateReduction, Trainer, TrainingMetrics},
};

/// Generate synthetic regression dataset
/// Creates a dataset where: y = 0.5*x1 + 0.3*x2 + 0.1*x1*x2 + noise
fn generate_regression_data(
    num_samples: usize,
    add_noise: bool,
) -> TensorResult<(Tensor<f32>, Tensor<f32>)> {
    let mut x_data = Vec::with_capacity(num_samples * 2);
    let mut y_data = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        // Generate 2 input features
        let x1 = ((i as f32) / (num_samples as f32)) * 4.0 - 2.0; // Range [-2, 2]
        let x2 = ((i * 3 + 1) as f32).sin() * 1.5; // Range ~[-1.5, 1.5]

        x_data.extend([x1, x2]);

        // Target function: y = 0.5*x1 + 0.3*x2 + 0.1*x1*x2 + noise
        let mut y = 0.5 * x1 + 0.3 * x2 + 0.1 * x1 * x2;

        // Add noise for more realistic training
        if add_noise {
            let noise = ((i * 7 + 13) as f32).sin() * 0.1;
            y += noise;
        }

        y_data.push(y);
    }

    let x_tensor = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[num_samples, 2]), x_data)?);
    let y_tensor = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[num_samples, 1]), y_data)?);

    Ok((x_tensor, y_tensor))
}

/// Create a multi-layer regression model
fn create_regression_model() -> TensorResult<Sequential<f32>> {
    let layers: Vec<Box<dyn Layer<f32>>> = vec![
        Box::new(Dense::new(2, 8, true)),  // Input: 2 -> 8
        Box::new(Dense::new(8, 16, true)), // Hidden: 8 -> 16
        Box::new(Dense::new(16, 8, true)), // Hidden: 16 -> 8
        Box::new(Dense::new(8, 1, true)),  // Output: 8 -> 1
    ];

    Ok(Sequential::new(layers))
}

/// Create data iterator for training
struct DataIterator {
    inputs: Tensor<f32>,
    targets: Tensor<f32>,
    batch_size: usize,
    current_idx: usize,
}

impl DataIterator {
    fn new(inputs: Tensor<f32>, targets: Tensor<f32>, batch_size: usize) -> Self {
        Self {
            inputs,
            targets,
            batch_size,
            current_idx: 0,
        }
    }

    fn len(&self) -> usize {
        self.inputs.shape().dims()[0]
    }
}

impl Iterator for DataIterator {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let total_samples = self.inputs.shape().dims()[0];

        if self.current_idx >= total_samples {
            return None;
        }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, total_samples);

        // Create batch slices
        let batch_inputs = self
            .inputs
            .slice(&[self.current_idx..end_idx, 0..self.inputs.shape().dims()[1]])
            .ok()?;
        let batch_targets = self
            .targets
            .slice(&[self.current_idx..end_idx, 0..self.targets.shape().dims()[1]])
            .ok()?;

        self.current_idx = end_idx;

        Some((batch_inputs, batch_targets))
    }
}

impl Clone for DataIterator {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            targets: self.targets.clone(),
            batch_size: self.batch_size,
            current_idx: 0, // Reset for new epoch
        }
    }
}

/// Mean Absolute Error loss function for comparison
fn mae_loss(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> TensorResult<Tensor<f32>> {
    let diff = predictions.sub(targets)?;
    let abs_diff = diff.abs()?;
    abs_diff.mean(None, false)
}

/// Compute R² coefficient of determination
fn compute_r_squared(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> TensorResult<f32> {
    let pred_vec = predictions.to_vec().map_err(|_| {
        tenflowers_core::TensorError::invalid_argument(
            "Failed to convert predictions to vec".to_string(),
        )
    })?;
    let target_vec = targets.to_vec().map_err(|_| {
        tenflowers_core::TensorError::invalid_argument(
            "Failed to convert targets to vec".to_string(),
        )
    })?;

    // Compute means
    let pred_mean: f32 = pred_vec.iter().sum::<f32>() / pred_vec.len() as f32;
    let target_mean: f32 = target_vec.iter().sum::<f32>() / target_vec.len() as f32;

    // Compute sums of squares
    let mut ss_tot = 0.0f32;
    let mut ss_res = 0.0f32;

    for i in 0..pred_vec.len() {
        ss_tot += (target_vec[i] - target_mean).powi(2);
        ss_res += (target_vec[i] - pred_vec[i]).powi(2);
    }

    let r_squared = 1.0 - (ss_res / ss_tot);
    Ok(r_squared)
}

fn main() -> TensorResult<()> {
    println!("🌻 TenfloweRS Advanced Autograd Training Demo 🌻");
    println!("==================================================");

    // Generate training and validation datasets
    println!("🔢 Generating synthetic regression dataset...");
    let (train_x, train_y) = generate_regression_data(800, true)?;
    let (val_x, val_y) = generate_regression_data(200, false)?;

    println!("  Training data: {} samples", train_x.shape().dims()[0]);
    println!("  Validation data: {} samples", val_x.shape().dims()[0]);
    println!("  Features: {} dimensions", train_x.shape().dims()[1]);

    // Create model architecture
    println!("\n🏗️ Creating neural network model...");
    let mut model = create_regression_model()?;

    println!("  Model architecture:");
    println!("    Input layer:    2 -> 8   (with bias)");
    println!("    Hidden layer 1: 8 -> 16  (with bias)");
    println!("    Hidden layer 2: 16 -> 8  (with bias)");
    println!("    Output layer:   8 -> 1   (with bias)");

    let param_count: usize = model
        .parameters()
        .iter()
        .map(|p| p.shape().dims().iter().product::<usize>())
        .sum();
    println!("    Total parameters: {}", param_count);

    // Test initial forward pass
    println!("\n🧪 Testing initial model predictions...");
    let sample_input = train_x.slice(&[0..3, 0..train_x.shape().dims()[1]])?;
    let sample_target = train_y.slice(&[0..3, 0..train_y.shape().dims()[1]])?;
    let initial_pred = model.forward(&sample_input)?;

    println!(
        "  Sample inputs: {:?}",
        sample_input.to_vec().unwrap_or_default()
    );
    println!(
        "  Target outputs: {:?}",
        sample_target.to_vec().unwrap_or_default()
    );
    println!(
        "  Initial predictions: {:?}",
        initial_pred.to_vec().unwrap_or_default()
    );

    // Train with Adam optimizer
    println!("\n🚀 Training with Adam optimizer...");
    println!("=====================================");

    {
        // Create fresh model for Adam
        let mut model_adam = create_regression_model()?;
        let mut adam_optimizer = Adam::new(0.01);

        // Create trainer with callbacks
        let mut trainer = Trainer::new();

        // Add early stopping callback
        let early_stopping = EarlyStopping::new(
            15,    // patience: 15 epochs
            0.001, // min_delta: 0.001
            "val_loss".to_string(),
            "min".to_string(),
        );
        trainer.add_callback(Box::new(early_stopping));

        // Add learning rate reduction callback
        let lr_reduction = LearningRateReduction::with_defaults("val_loss".to_string())
            .factor(0.5)
            .patience(8)
            .verbose(true)
            .min_lr(1e-6);
        trainer.add_callback(Box::new(lr_reduction));

        // Create data iterators
        let batch_size = 32;
        let train_data = DataIterator::new(train_x.clone(), train_y.clone(), batch_size);
        let val_data = DataIterator::new(val_x.clone(), val_y.clone(), batch_size);

        println!("  Batch size: {}", batch_size);
        println!(
            "  Training batches per epoch: {}",
            (train_data.len() + batch_size - 1) / batch_size
        );
        println!(
            "  Validation batches per epoch: {}",
            (val_data.len() + batch_size - 1) / batch_size
        );

        // Train model using autograd with Adam
        let start_time = Instant::now();
        let training_state = trainer.fit(
            &mut model_adam,
            &mut adam_optimizer,
            train_data,
            Some(val_data),
            50,  // max epochs
            mse, // Mean squared error loss function
        )?;

        let training_time = start_time.elapsed();

        println!(
            "\n📊 Adam Training completed in {:.2}s",
            training_time.as_secs_f64()
        );
        println!("  Total epochs: {}", training_state.epoch + 1);
        println!(
            "  Final training loss: {:.6}",
            training_state.history.last().map(|m| m.loss).unwrap_or(0.0)
        );
        println!(
            "  Final validation loss: {:.6}",
            training_state
                .val_history
                .last()
                .map(|m| m.loss)
                .unwrap_or(0.0)
        );

        // Evaluate final model performance
        println!("\n🎯 Adam Final model evaluation:");
        model_adam.set_training(false);

        // Full validation set predictions
        let val_predictions = model_adam.forward(&val_x)?;
        let final_mse = mse(&val_predictions, &val_y)?;
        let final_mae = mae_loss(&val_predictions, &val_y)?;
        let r_squared = compute_r_squared(&val_predictions, &val_y)?;

        let mse_value = final_mse.get(&[]).unwrap_or(0.0);
        let mae_value = final_mae.get(&[]).unwrap_or(0.0);

        println!("  Mean Squared Error: {:.6}", mse_value);
        println!("  Mean Absolute Error: {:.6}", mae_value);
        println!("  R² Score: {:.6}", r_squared);
        println!("  RMSE: {:.6}", mse_value.sqrt());

        // Sample predictions vs targets
        let sample_preds = val_predictions.slice(&[0..5, 0..val_predictions.shape().dims()[1]])?;
        let sample_targets = val_y.slice(&[0..5, 0..val_y.shape().dims()[1]])?;

        println!("\n  Sample predictions vs targets:");
        let pred_vals = sample_preds.to_vec().unwrap_or_default();
        let target_vals = sample_targets.to_vec().unwrap_or_default();

        for i in 0..std::cmp::min(pred_vals.len(), target_vals.len()) {
            println!(
                "    Prediction: {:.4}, Target: {:.4}, Error: {:.4}",
                pred_vals[i],
                target_vals[i],
                (pred_vals[i] - target_vals[i]).abs()
            );
        }
    }

    // Train with SGD optimizer
    println!("\n🚀 Training with SGD optimizer...");
    println!("====================================");

    {
        // Create fresh model for SGD
        let mut model_sgd = create_regression_model()?;
        let mut sgd_optimizer = SGD::new(0.1).with_momentum(0.9);

        // Simple training without callbacks for comparison
        let mut trainer = Trainer::new();

        let batch_size = 32;
        let train_data = DataIterator::new(train_x.clone(), train_y.clone(), batch_size);
        let val_data = DataIterator::new(val_x.clone(), val_y.clone(), batch_size);

        // Train model using autograd with SGD
        let start_time = Instant::now();
        let training_state = trainer.fit(
            &mut model_sgd,
            &mut sgd_optimizer,
            train_data,
            Some(val_data),
            30, // fewer epochs for SGD
            mse,
        )?;

        let training_time = start_time.elapsed();

        println!(
            "\n📊 SGD Training completed in {:.2}s",
            training_time.as_secs_f64()
        );
        println!("  Total epochs: {}", training_state.epoch + 1);
        println!(
            "  Final training loss: {:.6}",
            training_state.history.last().map(|m| m.loss).unwrap_or(0.0)
        );
        println!(
            "  Final validation loss: {:.6}",
            training_state
                .val_history
                .last()
                .map(|m| m.loss)
                .unwrap_or(0.0)
        );

        // Evaluate SGD model
        model_sgd.set_training(false);
        let val_predictions = model_sgd.forward(&val_x)?;
        let final_mse = mse(&val_predictions, &val_y)?;
        let mse_value = final_mse.get(&[]).unwrap_or(0.0);
        println!("  SGD Final MSE: {:.6}", mse_value);
    }

    println!("\n✅ Autograd training demonstration completed successfully!");
    println!("🌻 TenfloweRS autograd system provides:");
    println!("  • Automatic gradient computation via GradientTape");
    println!("  • Seamless integration with neural network training");
    println!("  • Support for complex model architectures");
    println!("  • Multiple optimizer algorithms with proper gradient application");
    println!("  • Advanced training features (callbacks, early stopping, LR scheduling)");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() -> TensorResult<()> {
        let (x, y) = generate_regression_data(100, false)?;
        assert_eq!(x.shape().dims(), &[100, 2]);
        assert_eq!(y.shape().dims(), &[100, 1]);
        Ok(())
    }

    #[test]
    fn test_model_creation() -> TensorResult<()> {
        let model = create_regression_model()?;
        let input = Tensor::ones(&[1, 2]);
        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn test_data_iterator() -> TensorResult<()> {
        let (x, y) = generate_regression_data(10, false)?;
        let mut iter = DataIterator::new(x, y, 3);

        let batch1 = iter.next().unwrap();
        assert_eq!(batch1.0.shape().dims()[0], 3);

        let batch2 = iter.next().unwrap();
        assert_eq!(batch2.0.shape().dims()[0], 3);

        let batch3 = iter.next().unwrap();
        assert_eq!(batch3.0.shape().dims()[0], 3);

        let batch4 = iter.next().unwrap();
        assert_eq!(batch4.0.shape().dims()[0], 1); // Last batch

        assert!(iter.next().is_none());
        Ok(())
    }

    #[test]
    fn test_r_squared_computation() -> TensorResult<()> {
        // Perfect predictions should give R² = 1.0
        let targets =
            Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![1.0, 2.0, 3.0])?);
        let perfect_preds = targets.clone();
        let r2 = compute_r_squared(&perfect_preds, &targets)?;
        assert!((r2 - 1.0).abs() < 1e-6);
        Ok(())
    }
}
