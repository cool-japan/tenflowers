//! Optimization algorithms for neural network training.
//!
//! This module provides a comprehensive collection of optimization algorithms (optimizers)
//! for training neural networks. Each optimizer implements different strategies for updating
//! model parameters based on computed gradients.
//!
//! # Available Optimizers
//!
//! ## First-Order Adaptive Optimizers
//!
//! - [`Adam`]: Adaptive Moment Estimation - most popular general-purpose optimizer
//! - [`AdamW`]: Adam with decoupled weight decay regularization
//! - [`RAdam`]: Rectified Adam with variance warmup
//! - [`Nadam`]: Adam with Nesterov momentum
//! - [`AdaBelief`]: Adapts step size based on gradient prediction error
//! - [`Adagrad`]: Adaptive gradient with per-parameter learning rates
//! - [`Adadelta`]: Extension of Adagrad with adaptive learning rate
//! - [`RMSprop`]: Root Mean Square Propagation
//!
//! ## Momentum-Based Optimizers
//!
//! - [`SGD`]: Stochastic Gradient Descent with optional momentum
//! - [`Lion`]: Evolved optimizer with sign updates and momentum
//!
//! ## Second-Order Optimizers
//!
//! - [`LBFGS`]: Limited-memory Broyden-Fletcher-Goldfarb-Shanno
//! - [`Sophia`]: Second-order clipped stochastic optimizer
//!
//! ## Large-Scale Training Optimizers
//!
//! - [`LAMB`]: Layer-wise Adaptive Moments for Batch training
//! - [`SAMOptimizer`]: Sharpness-Aware Minimization
//! - [`Soap`]: Shampoo-based optimizer for large models
//!
//! ## Meta-Optimizers and Wrappers
//!
//! - [`Lookahead`]: Wraps any optimizer with slow and fast weights
//! - [`SWA`]: Stochastic Weight Averaging for better generalization
//! - [`GradientCentralizationWrapper`]: Adds gradient centralization to any optimizer
//! - [`OptimizerWithAccumulation`]: Gradient accumulation for large batch training
//! - [`ParameterGroupOptimizer`]: Different learning rates for parameter groups
//!
//! # Basic Usage
//!
//! ## Training with Adam
//!
//! ```rust,no_run
//! use tenflowers_neural::{Sequential, Dense, Adam};
//! use tenflowers_neural::loss::mse;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut model = Sequential::new();
//! model.add(Dense::new(10, 64)?);
//! model.add(Dense::new(64, 1)?);
//!
//! let mut optimizer = Adam::new(0.001); // learning rate
//!
//! // Training loop
//! for epoch in 0..10 {
//!     let x = Tensor::zeros(&[32, 10]);
//!     let y = Tensor::zeros(&[32, 1]);
//!
//!     let pred = model.forward(&x)?;
//!     let loss = mse(&pred, &y)?;
//!
//!     // Backward pass and update
//!     optimizer.zero_grad(&mut model);
//!     // loss.backward()?; // Would compute gradients
//!     optimizer.step(&mut model)?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## AdamW with Weight Decay
//!
//! ```rust,no_run
//! use tenflowers_neural::{AdamW, Sequential};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let mut optimizer = AdamW::new(0.001)
//!     .with_weight_decay(0.01)  // L2 regularization
//!     .with_betas(0.9, 0.999)
//!     .with_eps(1e-8);
//! # Ok(())
//! # }
//! ```
//!
//! ## Gradient Accumulation for Large Batches
//!
//! ```rust,no_run
//! use tenflowers_neural::{Adam, OptimizerWithAccumulation, Sequential};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let base_optimizer = Adam::new(0.001);
//! let mut optimizer = OptimizerWithAccumulation::new(
//!     Box::new(base_optimizer),
//!     4,  // accumulate over 4 steps
//! );
//!
//! // Effective batch size = physical_batch_size * 4
//! for step in 0..4 {
//!     // Forward pass with smaller batch
//!     // loss.backward()?;
//!     optimizer.step(&mut model)?;  // Only updates every 4 steps
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Parameter Groups with Different Learning Rates
//!
//! ```rust,no_run
//! use tenflowers_neural::optimizers::{ParameterGroup, ParameterGroupOptimizer};
//! use tenflowers_neural::{Sequential, Adam};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//!
//! let groups = vec![
//!     ParameterGroup::new(vec![], 0.001),  // First layers
//!     ParameterGroup::new(vec![], 0.0001), // Last layers (fine-tuning)
//! ];
//!
//! let mut optimizer = ParameterGroupOptimizer::new(
//!     Box::new(Adam::new(0.001)),
//!     groups,
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Lookahead Optimizer
//!
//! ```rust,no_run
//! use tenflowers_neural::{Adam, Lookahead, Sequential};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let base = Adam::new(0.001);
//! let mut optimizer = Lookahead::new(
//!     Box::new(base),
//!     0.5,  // alpha (slow weights update rate)
//!     5,    // k (update slow weights every 5 steps)
//! );
//! # Ok(())
//! # }
//! ```
//!
//! ## Stochastic Weight Averaging
//!
//! ```rust,no_run
//! use tenflowers_neural::{Adam, SWA, Sequential};
//! use tenflowers_neural::optimizers::SwaConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let base = Adam::new(0.001);
//! let config = SwaConfig {
//!     swa_start: 100,   // Start averaging after 100 epochs
//!     swa_freq: 5,      // Update average every 5 epochs
//!     swa_lr: 0.0001,   // Learning rate during SWA
//! };
//! let mut optimizer = SWA::new(Box::new(base), config);
//! # Ok(())
//! # }
//! ```
//!
//! ## Gradient Centralization
//!
//! ```rust,no_run
//! use tenflowers_neural::{Adam, Sequential};
//! use tenflowers_neural::optimizers::{
//!     GradientCentralizationWrapper,
//!     GradientCentralizationConfig,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let base = Adam::new(0.001);
//! let config = GradientCentralizationConfig {
//!     use_gc: true,
//!     gc_conv_only: false,  // Apply to all layers
//! };
//! let mut optimizer = GradientCentralizationWrapper::new(
//!     Box::new(base),
//!     config,
//! );
//! # Ok(())
//! # }
//! ```
//!
//! # Optimizer Selection Guide
//!
//! ## General Purpose
//! - **Adam**: Best starting point for most tasks
//! - **AdamW**: Adam with better weight decay (preferred for transformers)
//!
//! ## Computer Vision
//! - **SGD with momentum**: Often better final performance for CNNs
//! - **LAMB**: Efficient for large batch training
//!
//! ## Natural Language Processing
//! - **AdamW**: Standard for transformer models
//! - **Lion**: More memory efficient than Adam
//!
//! ## Large-Scale Training
//! - **LAMB**: Maintains accuracy with large batches
//! - **SAM**: Improves generalization through sharpness minimization
//! - **Gradient Accumulation**: Simulate large batches with limited memory
//!
//! ## Fine-Tuning
//! - **AdamW with low learning rate**: Standard approach
//! - **Parameter Groups**: Different rates for different layers
//! - **Lookahead**: More stable convergence
//!
//! ## Research/Experimental
//! - **Sophia**: Second-order optimizer for LLMs
//! - **Soap**: Advanced preconditioned optimizer
//! - **AdaBelief**: Adapts to gradient prediction error

pub mod adabelief;
pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod cosine_scheduler;
pub mod enhanced_accumulation;
pub mod gradient_centralization;
pub mod gradient_clipping;
pub mod lamb;
pub mod lbfgs;
pub mod lion;
pub mod lookahead;
pub mod nadam;
pub mod optimizer_with_accumulation;
pub mod parameter_groups;
pub mod radam;
pub mod rmsprop;
pub mod sam;
pub mod sgd;
pub mod soap;
pub mod sophia;
pub mod swa;

pub use adabelief::{AdaBelief, AdaBeliefConfig};
pub use adadelta::Adadelta;
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use cosine_scheduler::{
    create_cosine_schedule_for_epochs, CosineScheduler, CosineSchedulerConfig, SnapshotEnsemble,
};
pub use enhanced_accumulation::{
    AccumulationProgress, AccumulationStrategy, EnhancedGradientAccumulator,
    EnhancedOptimizerWithAccumulation, MemoryConfig, MemoryStats,
};
pub use gradient_centralization::{
    apply_gradient_centralization, GradientCentralizationConfig, GradientCentralizationWrapper,
    WithGradientCentralization,
};
pub use gradient_clipping::{
    clip_gradients_adaptive, clip_gradients_by_global_norm, clip_gradients_by_norm,
    clip_gradients_by_value,
};
pub use lamb::LAMB;
pub use lbfgs::LBFGS;
pub use lion::Lion;
pub use lookahead::Lookahead;
pub use nadam::Nadam;
pub use optimizer_with_accumulation::OptimizerWithAccumulation;
pub use parameter_groups::{ParameterGroup, ParameterGroupConfig, ParameterGroupOptimizer};
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use sam::SAMOptimizer;
pub use sgd::SGD;
pub use soap::Soap;
pub use sophia::Sophia;
pub use swa::{ensemble_predict, SwaConfig, SWA};

use crate::model::Model;
use tenflowers_core::Result;

pub trait Optimizer<T> {
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()>;
    fn zero_grad(&self, model: &mut dyn Model<T>);
    fn set_learning_rate(&mut self, learning_rate: f32);
    fn get_learning_rate(&self) -> f32;
}
