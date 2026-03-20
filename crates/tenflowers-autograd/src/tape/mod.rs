//! Automatic differentiation tape for gradient computation
//!
//! This module provides a comprehensive automatic differentiation system that records
//! computational operations in a tape and enables backward gradient computation through
//! the computational graph.

// Re-export specialized modules
pub mod gradient_computation;
pub mod gradient_tape;
pub mod helpers;
pub mod higher_order_derivatives;
pub mod operations;
pub mod structures;
pub mod tracked_tensor;
pub mod utils;

// Re-export commonly used types
pub use operations::Operation;
pub use structures::{GradientTape, TapeNode, TrackedTensor};
pub use utils::*;

/// Unique identifier for tensors in the computation graph
pub type TensorId = usize;
