//! Gradient computation modules
//!
//! This module contains the refactored gradient computation logic, organized by operation type:
//!
//! - `core`: Core gradient computation algorithms and operation dispatch
//! - `basic_ops`: Basic arithmetic operations (add, mul, sub, div, matmul)
//! - `tensor_ops`: Tensor manipulation operations (transpose, reshape, sum, mean)
//! - `activation_ops`: Activation functions (relu, sigmoid, tanh, softmax, etc.)
//! - `neural_ops`: Neural network operations (conv2d, batchnorm, layernorm, etc.)

pub mod activation_ops;
pub mod basic_ops;
pub mod core;
pub mod neural_ops;
pub mod tensor_ops;

// Re-export the main GradientTape gradient computation method
// (Note: Currently no external usage of core exports)
