use tenflowers_core::Tensor;
use tenflowers_neural::model::Model;
use tenflowers_neural::optimizers::{Adam, AdamW, Optimizer, RMSprop, SGD};

struct SimpleModel {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl SimpleModel {
    fn new() -> Self {
        Self {
            weight: Tensor::ones(&[2, 2]),
            bias: Tensor::zeros(&[2]),
        }
    }
}

impl Model<f32> for SimpleModel {
    fn forward(&self, input: &Tensor<f32>) -> tenflowers_core::Result<Tensor<f32>> {
        // Simple forward pass for testing - just return input
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn set_training(&mut self, _training: bool) {
        // No-op for testing
    }

    fn zero_grad(&mut self) {
        // Set gradients to None - simplified for testing
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn test_adamw_optimizer() {
    let mut model = SimpleModel::new();
    let mut optimizer = AdamW::new(0.001, 0.01);

    // Set some dummy gradients
    model.weight.set_grad(Some(Tensor::ones(&[2, 2])));
    model.bias.set_grad(Some(Tensor::ones(&[2])));

    // Save initial weights
    let initial_weight = model.weight.get(&[0, 0]).unwrap();
    let initial_bias = model.bias.get(&[0]).unwrap();

    // Perform optimization step
    optimizer
        .step(&mut model)
        .expect("Optimization step failed");

    // Check that parameters changed
    let new_weight = model.weight.get(&[0, 0]).unwrap();
    let new_bias = model.bias.get(&[0]).unwrap();

    assert_ne!(initial_weight, new_weight, "Weight should have changed");
    assert_ne!(initial_bias, new_bias, "Bias should have changed");
    assert!(
        new_weight < initial_weight,
        "Weight should have decreased due to gradient descent"
    );
    assert!(
        new_bias < initial_bias,
        "Bias should have decreased due to gradient descent"
    );
}

#[test]
fn test_rmsprop_optimizer() {
    let mut model = SimpleModel::new();
    let mut optimizer = RMSprop::new(0.01);

    // Set some dummy gradients
    model.weight.set_grad(Some(Tensor::ones(&[2, 2])));
    model.bias.set_grad(Some(Tensor::ones(&[2])));

    // Save initial weights
    let initial_weight = model.weight.get(&[0, 0]).unwrap();
    let initial_bias = model.bias.get(&[0]).unwrap();

    // Perform optimization step
    optimizer
        .step(&mut model)
        .expect("Optimization step failed");

    // Check that parameters changed
    let new_weight = model.weight.get(&[0, 0]).unwrap();
    let new_bias = model.bias.get(&[0]).unwrap();

    assert_ne!(initial_weight, new_weight, "Weight should have changed");
    assert_ne!(initial_bias, new_bias, "Bias should have changed");
    assert!(
        new_weight < initial_weight,
        "Weight should have decreased due to gradient descent"
    );
    assert!(
        new_bias < initial_bias,
        "Bias should have decreased due to gradient descent"
    );
}

#[test]
fn test_rmsprop_with_momentum() {
    let mut model = SimpleModel::new();
    let mut optimizer = RMSprop::new(0.01).with_momentum(0.9);

    // Set some dummy gradients
    model.weight.set_grad(Some(Tensor::ones(&[2, 2])));
    model.bias.set_grad(Some(Tensor::ones(&[2])));

    // Save initial weights
    let initial_weight = model.weight.get(&[0, 0]).unwrap();

    // Perform optimization step
    optimizer
        .step(&mut model)
        .expect("Optimization step failed");

    // Check that parameters changed
    let new_weight = model.weight.get(&[0, 0]).unwrap();
    assert_ne!(
        initial_weight, new_weight,
        "Weight should have changed with momentum"
    );
}

#[test]
fn test_adamw_weight_decay() {
    let mut model = SimpleModel::new();
    let mut optimizer_no_decay = AdamW::new(0.001, 0.0);
    let mut optimizer_with_decay = AdamW::new(0.001, 0.01);

    // Create two identical models
    let mut model1 = SimpleModel::new();
    let mut model2 = SimpleModel::new();

    // Set identical gradients
    model1.weight.set_grad(Some(Tensor::ones(&[2, 2])));
    model1.bias.set_grad(Some(Tensor::ones(&[2])));
    model2.weight.set_grad(Some(Tensor::ones(&[2, 2])));
    model2.bias.set_grad(Some(Tensor::ones(&[2])));

    // Apply optimizers
    optimizer_no_decay.step(&mut model1).expect("Step failed");
    optimizer_with_decay.step(&mut model2).expect("Step failed");

    // With weight decay, the parameter should decrease more
    let weight_no_decay = model1.weight.get(&[0, 0]).unwrap();
    let weight_with_decay = model2.weight.get(&[0, 0]).unwrap();

    assert!(
        weight_with_decay < weight_no_decay,
        "Weight decay should cause more parameter reduction"
    );
}

#[test]
fn test_optimizer_configurations() {
    // Test AdamW with custom betas
    let adamw_custom: AdamW<f32> = AdamW::with_betas(0.001, 0.01, 0.8, 0.995);
    assert_eq!(adamw_custom.beta1, 0.8);
    assert_eq!(adamw_custom.beta2, 0.995);
    assert_eq!(adamw_custom.learning_rate, 0.001);
    assert_eq!(adamw_custom.weight_decay, 0.01);

    // Test AdamW with epsilon
    let adamw_epsilon: AdamW<f32> = AdamW::new(0.001, 0.01).with_epsilon(1e-6);
    assert_eq!(adamw_epsilon.epsilon, 1e-6);

    // Test RMSprop with custom alpha and weight decay
    let rmsprop_custom = RMSprop::new(0.01)
        .with_alpha(0.95)
        .with_weight_decay(0.001)
        .with_momentum(0.9);
    assert_eq!(rmsprop_custom.alpha, 0.95);
    assert_eq!(rmsprop_custom.weight_decay, 0.001);
    assert_eq!(rmsprop_custom.momentum, Some(0.9));

    // Test default optimizers
    let adamw_default: AdamW<f32> = AdamW::default();
    assert_eq!(adamw_default.learning_rate, 0.001);
    assert_eq!(adamw_default.weight_decay, 0.01);

    let rmsprop_default = RMSprop::default();
    assert_eq!(rmsprop_default.learning_rate, 0.01);
}
