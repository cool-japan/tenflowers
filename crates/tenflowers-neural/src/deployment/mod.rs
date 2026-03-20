pub mod fusion;
pub mod mobile;
/// Model optimization and deployment utilities for mobile/edge devices.
///
/// This module provides various optimization techniques to reduce model size,
/// improve inference speed, and enable deployment on resource-constrained devices.
pub mod optimization;
pub mod pruning;
pub mod quantization;

pub use fusion::*;
pub use mobile::*;
pub use optimization::*;
pub use pruning::*;
pub use quantization::*;
