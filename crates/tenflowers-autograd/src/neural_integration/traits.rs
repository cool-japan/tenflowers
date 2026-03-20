//! Traits for neural network integration with autograd

use tenflowers_core::{Result, Tensor};

/// Trait for neural network layers that can be integrated with autograd
///
/// This trait provides the interface for neural network layers to work seamlessly
/// with the automatic differentiation system. It uses traits to avoid circular
/// dependencies between autograd and neural crates.
pub trait NeuralLayer<T> {
    /// Forward pass through the layer
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>>;

    /// Get references to the layer's parameters
    fn parameters(&self) -> Vec<&Tensor<T>>;

    /// Get mutable references to the layer's parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>>;

    /// Set training mode
    fn set_training(&mut self, training: bool);

    /// Check if layer is in training mode
    fn is_training(&self) -> bool;
}

/// Trait for models that can be integrated with autograd
///
/// This trait provides the interface for complete neural network models to work
/// with the automatic differentiation system.
pub trait NeuralModel<T> {
    /// Forward pass through the model
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>>;

    /// Get all parameters from the model
    fn parameters(&self) -> Vec<&Tensor<T>>;

    /// Get all mutable parameters from the model
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>>;

    /// Set training mode for the entire model
    fn set_training(&mut self, training: bool);

    /// Check if model is in training mode
    fn is_training(&self) -> bool;
}

/// Trait for loss functions that can be integrated with autograd
pub trait NeuralLoss<T> {
    /// Compute the loss given predictions and targets
    fn compute_loss(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>;

    /// Get the name of the loss function
    fn name(&self) -> &str;
}

/// Trait for metrics that can be computed during training
pub trait NeuralMetric<T> {
    /// Compute the metric given predictions and targets
    fn compute(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<T>;

    /// Get the name of the metric
    fn name(&self) -> &str;

    /// Reset any internal state (for accumulated metrics)
    fn reset(&mut self);
}
