use tenflowers_dataset::{RealMnistBuilder, Dataset, DataLoader, DataLoaderBuilder};
use tenflowers_neural::{Sequential, Dense, Model, ActivationFn, OptimizerTrait, optimizers::SGD, loss::MSELoss};
use tenflowers_autograd::GradientTape;
use tenflowers_core::{Tensor, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 TenfloweRS Real MNIST Example");
    println!("================================");
    
    // Set up device
    let device = Device::CPU;
    
    // Create real MNIST dataset with automatic downloading
    println!("📥 Loading MNIST dataset...");
    let train_dataset = RealMnistBuilder::new()
        .root("./data")  // Data will be downloaded to ./data/MNIST/raw/
        .train(true)     // Use training set
        .download(true)  // Enable automatic downloading
        .build::<f32>()?;
    
    let test_dataset = RealMnistBuilder::new()
        .root("./data")
        .train(false)    // Use test set
        .download(true)
        .build::<f32>()?;
    
    println!("✅ MNIST dataset loaded successfully!");
    println!("   Training samples: {}", train_dataset.len());
    println!("   Test samples: {}", test_dataset.len());
    
    // Create data loaders
    let train_loader = DataLoaderBuilder::new(train_dataset)
        .batch_size(32)
        .shuffle(true)
        .build();
    
    let test_loader = DataLoaderBuilder::new(test_dataset)
        .batch_size(32)
        .shuffle(false)
        .build();
    
    // Create a simple neural network
    println!("\n🧠 Creating neural network...");
    let mut model = Sequential::new()
        .add(Dense::new(784, 128).unwrap())  // Input: 28*28 = 784 pixels
        .add(ActivationFn::ReLU)
        .add(Dense::new(128, 64).unwrap())
        .add(ActivationFn::ReLU)
        .add(Dense::new(64, 10).unwrap())    // Output: 10 classes (digits 0-9)
        .build::<f32>(&device)?;
    
    // Set up optimizer and loss
    let mut optimizer = SGD::new(0.01);
    let loss_fn = MSELoss::new();
    
    println!("   Model created with {} parameters", model.parameter_count());
    
    // Training loop
    println!("\n🚀 Training model...");
    let epochs = 5;
    
    for epoch in 0..epochs {
        println!("\nEpoch {}/{}", epoch + 1, epochs);
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Training loop
        model.train();
        for batch in train_loader.clone().take(100) { // Take only 100 batches for quick demo
            let mut tape = GradientTape::new();
            
            // Prepare batch data
            let (images, labels) = prepare_batch(batch)?;
            
            // Convert labels to one-hot encoding
            let one_hot_labels = create_one_hot(&labels, 10)?;
            
            // Forward pass
            tape.watch(&model.parameters());
            let logits = model.forward(&images)?;
            let loss = loss_fn.compute(&logits, &one_hot_labels)?;
            
            // Backward pass
            let gradients = tape.gradient(&loss, &model.parameters())?;
            
            // Update parameters
            optimizer.step(&model.parameters(), &gradients)?;
            
            total_loss += loss.as_slice().unwrap()[0];
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("   Average training loss: {:.4}", avg_loss);
        
        // Validation
        model.eval();
        let mut correct = 0;
        let mut total = 0;
        
        for batch in test_loader.clone().take(50) { // Take only 50 batches for quick demo
            let (images, labels) = prepare_batch(batch)?;
            
            let predictions = model.forward(&images)?;
            let predicted_classes = argmax(&predictions)?;
            
            let labels_data = labels.as_slice().unwrap();
            let predicted_data = predicted_classes.as_slice().unwrap();
            
            for (true_label, pred_label) in labels_data.iter().zip(predicted_data.iter()) {
                if (*true_label as u8) == (*pred_label as u8) {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        let accuracy = correct as f32 / total as f32 * 100.0;
        println!("   Validation accuracy: {:.2}%", accuracy);
    }
    
    println!("\n🎉 Training completed!");
    
    // Test on a few samples
    println!("\n🔍 Testing on sample images...");
    let (sample_image, sample_label) = train_dataset.get(0)?;
    println!("   Sample 0: True label = {:.0}", sample_label.as_slice().unwrap()[0]);
    
    let prediction = model.forward(&sample_image.unsqueeze(0)?)?;
    let predicted_class = argmax(&prediction)?;
    println!("   Sample 0: Predicted label = {:.0}", predicted_class.as_slice().unwrap()[0]);
    
    Ok(())
}

fn prepare_batch(batch: Vec<(Tensor<f32>, Tensor<f32>)>) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    let batch_size = batch.len();
    
    // Collect all images and labels
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();
    
    for (image, label) in batch {
        all_images.extend_from_slice(image.as_slice().unwrap());
        all_labels.push(label.as_slice().unwrap()[0]);
    }
    
    let images = Tensor::from_vec(all_images, &[batch_size, 784])?;
    let labels = Tensor::from_vec(all_labels, &[batch_size])?;
    
    Ok((images, labels))
}

fn create_one_hot(labels: &Tensor<f32>, num_classes: usize) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let labels_data = labels.as_slice().unwrap();
    let batch_size = labels_data.len();
    
    let mut one_hot_data = vec![0.0; batch_size * num_classes];
    
    for (i, &label) in labels_data.iter().enumerate() {
        let class_index = label as usize;
        if class_index < num_classes {
            one_hot_data[i * num_classes + class_index] = 1.0;
        }
    }
    
    Tensor::from_vec(one_hot_data, &[batch_size, num_classes]).map_err(Into::into)
}

fn argmax(tensor: &Tensor<f32>) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let data = tensor.as_slice().unwrap();
    let shape = tensor.shape();
    let (batch_size, num_classes) = (shape.dims()[0], shape.dims()[1]);
    
    let mut argmax_data = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let start_idx = i * num_classes;
        let end_idx = start_idx + num_classes;
        let slice = &data[start_idx..end_idx];
        
        let max_idx = slice.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        argmax_data.push(max_idx as f32);
    }
    
    Tensor::from_vec(argmax_data, &[batch_size]).map_err(Into::into)
}

// Extension trait to add unsqueeze method
trait TensorExt<T> {
    fn unsqueeze(&self, dim: usize) -> Result<Tensor<T>, Box<dyn std::error::Error>>;
}

impl<T: Clone + Default + num_traits::Zero + Send + Sync + 'static> TensorExt<T> for Tensor<T> {
    fn unsqueeze(&self, dim: usize) -> Result<Tensor<T>, Box<dyn std::error::Error>> {
        let current_shape = self.shape().dims();
        let mut new_shape = current_shape.to_vec();
        new_shape.insert(dim, 1);
        tenflowers_core::ops::reshape(self, &new_shape).map_err(Into::into)
    }
}