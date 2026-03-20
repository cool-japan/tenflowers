//! Neural network integration layer for autograd
//!
//! This module provides seamless integration between TenfloweRS neural network layers
//! and the autograd system, enabling automatic gradient computation and optimization.
//!
//! The integration uses traits to avoid circular dependencies between autograd and neural crates.

pub mod autograd_layer;
pub mod optimizer;
pub mod trainer;
pub mod traits;

pub use autograd_layer::AutogradLayer;
pub use optimizer::{AutogradOptimizer, OptimizerType};
pub use trainer::{AutogradTrainer, TrainingMetrics};
pub use traits::{NeuralLayer, NeuralLoss, NeuralMetric, NeuralModel};

// Re-export for backward compatibility
pub use autograd_layer::AutogradLayer as AutogradLayerWrapper;
pub use optimizer::AutogradOptimizer as OptimizationEngine;
pub use trainer::AutogradTrainer as TrainingEngine;
