use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::time::Instant;
/// Simple Neural Network Training Demo using TenfloweRS
///
/// This example demonstrates:
/// - Sequential model creation with dense layers
/// - Basic forward pass computation
/// - Model evaluation with synthetic data
/// - Adam optimizer configuration
/// - Learning rate scheduling setup
use tenflowers_core::{Device, Result as TensorResult, Tensor};
use tenflowers_neural::{
    layers::{Dense, Layer},
    loss::mse,
    model::{Model, Sequential},
    optimizers::{Adam, Optimizer},
    scheduler::{LearningRateScheduler, StepLR},
};

/// Create a simple feedforward neural network
fn create_model() -> TensorResult<Sequential<f32>> {
    let layers: Vec<Box<dyn Layer<f32>>> = vec![
        Box::new(Dense::new(4, 8, true)), // Input layer: 4 -> 8
        Box::new(Dense::new(8, 6, true)), // Hidden layer: 8 -> 6
        Box::new(Dense::new(6, 3, true)), // Output layer: 6 -> 3
    ];

    Ok(Sequential::new(layers))
}

/// Generate simple synthetic regression data
fn generate_data(num_samples: usize) -> TensorResult<(Tensor<f32>, Tensor<f32>)> {
    let mut x_data = Vec::with_capacity(num_samples * 4);
    let mut y_data = Vec::with_capacity(num_samples * 3);

    for i in 0..num_samples {
        // Generate 4 input features
        let x1 = (i as f32) / 100.0;
        let x2 = ((i + 1) as f32).sin() / 2.0;
        let x3 = ((i + 2) as f32).cos() / 2.0;
        let x4 = (i as f32 % 10.0) / 10.0;

        x_data.extend([x1, x2, x3, x4]);

        // Generate 3 target outputs (simple function of inputs)
        let y1 = x1 + x2;
        let y2 = x3 * x4;
        let y3 = (x1 + x3) / 2.0;

        y_data.extend([y1, y2, y3]);
    }

    // Create tensors
    let x_tensor = Tensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[num_samples, 4]), x_data).map_err(|_| {
            tenflowers_core::TensorError::invalid_shape_simple(
                "Invalid shape for input data".to_string(),
            )
        })?,
    );

    let y_tensor = Tensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[num_samples, 3]), y_data).map_err(|_| {
            tenflowers_core::TensorError::invalid_shape_simple(
                "Invalid shape for target data".to_string(),
            )
        })?,
    );

    Ok((x_tensor, y_tensor))
}

/// Simple model evaluation
fn evaluate_model(model: &Sequential<f32>, x: &Tensor<f32>, y: &Tensor<f32>) -> TensorResult<f32> {
    let predictions = model.forward(x)?;
    let loss = mse(&predictions, y)?;

    // Calculate a simple scalar loss measure (mean of all elements)
    let loss_data = loss.to_vec()?;
    let mean_loss = loss_data.iter().sum::<f32>() / loss_data.len() as f32;

    Ok(mean_loss)
}

/// Demonstrate model capabilities
fn demo_model_features(model: &Sequential<f32>) -> TensorResult<()> {
    println!("\n🔍 Model Analysis:");
    println!("  Number of layers: 3"); // Hard-coded since layers() is private

    // Test with a small batch
    println!("\n🧪 Testing with sample input...");
    let test_input = Tensor::from_array(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), // 2 samples, 4 features each
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        )
        .map_err(|_| {
            tenflowers_core::TensorError::invalid_shape_simple(
                "Invalid test input shape".to_string(),
            )
        })?,
    );

    let output = model.forward(&test_input)?;
    println!("  Input shape:  {:?}", test_input.shape());
    println!("  Output shape: {:?}", output.shape());
    let output_vec = output.to_vec()?;
    println!(
        "  Sample output: {:?}",
        &output_vec[..3.min(output_vec.len())]
    ); // Show first 3 values

    Ok(())
}

/// Demonstrate optimizer configuration
fn demo_optimizer_config() -> TensorResult<()> {
    println!("\n⚙️  Optimizer Configuration:");

    // Adam optimizer with custom settings
    let optimizer: Adam<f32> = Adam::new(0.001).with_weight_decay(1e-4);

    println!("  Optimizer: Adam");
    println!("  Learning rate: 0.001");
    println!("  Weight decay: 1e-4");

    // Learning rate scheduler
    let scheduler = StepLR::new(0.001, 10, 0.8);
    println!("  Scheduler: StepLR (step_size=10, gamma=0.8)");

    // Show learning rate progression
    println!("  LR Schedule:");
    for epoch in [0, 10, 20, 30, 40, 50] {
        println!("    Epoch {}: LR = {:.6}", epoch, scheduler.get_lr(epoch));
    }

    Ok(())
}

/// Main training simulation (without actual gradient computation)
fn simulate_training() -> TensorResult<()> {
    println!("\n🎯 Training Simulation:");

    let (train_x, train_y) = generate_data(100)?;
    let (val_x, val_y) = generate_data(20)?;

    let model = create_model()?;

    println!("  Training data: {} samples", train_x.shape()[0]);
    println!("  Validation data: {} samples", val_x.shape()[0]);

    // Simulate training epochs
    for epoch in 0..5 {
        let start_time = Instant::now();

        // Forward pass
        let train_loss = evaluate_model(&model, &train_x, &train_y)?;
        let val_loss = evaluate_model(&model, &val_x, &val_y)?;

        let epoch_time = start_time.elapsed().as_millis();

        println!(
            "  Epoch {}: Train Loss = {:.6}, Val Loss = {:.6} ({} ms)",
            epoch + 1,
            train_loss,
            val_loss,
            epoch_time
        );
    }

    println!("  Note: This is a simulation - actual training requires autograd integration");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌻 TenfloweRS Simple Training Demo 🌻");
    println!("{}", "=".repeat(45));

    let device = Device::default();
    println!("Using device: {:?}", device);

    // Create and analyze model
    println!("\n🏗️  Creating neural network model...");
    let model = create_model()?;
    println!("Model created successfully!");

    demo_model_features(&model)?;
    demo_optimizer_config()?;
    simulate_training()?;

    // Demonstrate data generation
    println!("\n📊 Data Generation Demo:");
    let (x, y) = generate_data(5)?;
    println!("  Generated {} samples", x.shape()[0]);
    println!("  Input features: {:?}", x.shape());
    println!("  Target outputs: {:?}", y.shape());
    let x_vec = x.to_vec()?;
    let y_vec = y.to_vec()?;
    println!("  Sample input: {:?}", &x_vec[..4.min(x_vec.len())]);
    println!("  Sample target: {:?}", &y_vec[..3.min(y_vec.len())]);

    println!("\n💡 Next Steps:");
    println!("  • Integrate with tenflowers-autograd for actual training");
    println!("  • Add more complex layer types (Conv2D, BatchNorm, etc.)");
    println!("  • Implement real loss functions and metrics");
    println!("  • Add model checkpointing and serialization");
    println!("  • Experiment with different architectures");

    println!("\n✅ Demo completed successfully!");
    println!("🌻 TenfloweRS neural network framework is ready! 🌻");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() -> TensorResult<()> {
        let _model = create_model()?;
        // Model creation successful if no error
        Ok(())
    }

    #[test]
    fn test_data_generation() -> TensorResult<()> {
        let (x, y) = generate_data(10)?;
        assert_eq!(x.shape(), &[10, 4]);
        assert_eq!(y.shape(), &[10, 3]);
        Ok(())
    }

    #[test]
    fn test_forward_pass() -> TensorResult<()> {
        let model = create_model()?;
        let (x, _y) = generate_data(3)?;

        let output = model.forward(&x)?;
        assert_eq!(output.shape(), &[3, 3]);
        Ok(())
    }

    #[test]
    fn test_evaluation() -> TensorResult<()> {
        let model = create_model()?;
        let (x, y) = generate_data(5)?;

        let loss = evaluate_model(&model, &x, &y)?;
        assert!(loss > 0.0); // Loss should be positive
        assert!(loss.is_finite()); // Loss should be finite
        Ok(())
    }

    #[test]
    fn test_scheduler() {
        let scheduler = StepLR::new(0.001, 5, 0.5);

        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(4), 0.001);
        assert_eq!(scheduler.get_lr(5), 0.0005); // 0.001 * 0.5
        assert_eq!(scheduler.get_lr(10), 0.00025); // 0.001 * 0.5^2
    }
}
