//! # TenfloweRS - Pure Rust Deep Learning Framework
//!
//! TenfloweRS is a comprehensive machine learning framework implemented in pure Rust,
//! providing TensorFlow-compatible APIs with Rust's safety and performance guarantees.
//! Built on the robust SciRS2 scientific computing ecosystem, TenfloweRS offers:
//!
//! - **Production-Ready**: Full-featured neural networks, training, and deployment
//! - **High Performance**: GPU acceleration, SIMD optimization, mixed precision
//! - **Type Safety**: Rust's type system prevents common ML bugs at compile time
//! - **Cross-Platform**: CPU, GPU (CUDA, Metal, Vulkan), and WebGPU support
//! - **Ecosystem Integration**: Seamless integration with SciRS2, NumRS2, and OptiRS
//!
//! ## Quick Start
//!
//! ### Basic Tensor Operations
//!
//! ```rust,no_run
//! use tenflowers::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create tensors
//! let a = Tensor::<f32>::zeros(&[2, 3]);
//! let b = Tensor::<f32>::ones(&[2, 3]);
//!
//! // Arithmetic operations
//! let c = ops::add(&a, &b)?;
//! let d = ops::mul(&a, &b)?;
//!
//! // Matrix multiplication
//! let x = Tensor::<f32>::ones(&[2, 3]);
//! let y = Tensor::<f32>::ones(&[3, 4]);
//! let z = ops::matmul(&x, &y)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Building Neural Networks
//!
//! ```rust,no_run
//! use tenflowers::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simple feedforward network
//! let mut model = Sequential::new();
//! model.add(Dense::new(784, 128)?);
//! model.add_activation(ActivationFunction::ReLU);
//! model.add(Dense::new(128, 10)?);
//! model.add_activation(ActivationFunction::Softmax);
//!
//! // Forward pass
//! let input = Tensor::zeros(&[32, 784]);
//! let output = model.forward(&input)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Training Models
//!
//! ```rust,no_run
//! use tenflowers::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut model = Sequential::new();
//! # let x_train = Tensor::zeros(&[100, 10]);
//! # let y_train = Tensor::zeros(&[100, 3]);
//! // Quick training
//! let results = quick_train(
//!     model,
//!     &x_train,
//!     &y_train,
//!     Box::new(SGD::new(0.01)),
//!     categorical_cross_entropy,
//!     10,  // epochs
//!     32,  // batch_size
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ### GPU Acceleration
//!
//! ```rust,no_run
//! use tenflowers::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "gpu")]
//! # {
//! // Move computation to GPU
//! let device = Device::gpu(0)?;
//! let gpu_tensor = Tensor::<f32>::zeros(&[1000, 1000]).to_device(&device)?;
//! let result = ops::matmul(&gpu_tensor, &gpu_tensor)?;
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! ### Automatic Differentiation
//!
//! ```rust,no_run
//! use tenflowers::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut tape = GradientTape::new();
//!
//! // Create tracked tensors
//! let x = tape.watch(Tensor::<f32>::ones(&[2, 2]));
//! let y = tape.watch(Tensor::<f32>::ones(&[2, 2]));
//!
//! // Compute gradients
//! let z = tape.watch(Tensor::<f32>::ones(&[2, 2]));
//! let gradients = tape.gradient(&[z], &[x, y])?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Data Loading
//!
//! ```rust,no_run
//! use tenflowers::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load dataset
//! let dataset = CsvDatasetBuilder::new("data.csv")
//!     .has_header(true)
//!     .build()?;
//!
//! // Create data loader with batching
//! let loader = DataLoaderBuilder::new(dataset)
//!     .batch_size(32)
//!     .shuffle(true)
//!     .num_workers(4)
//!     .build()?;
//!
//! // Iterate through batches
//! for batch in loader.iter() {
//!     let (features, labels) = batch?;
//!     // Training step...
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! TenfloweRS is organized into several focused crates:
//!
//! - [`core`]: Tensor operations and device management
//! - [`autograd`]: Automatic differentiation engine
//! - [`neural`]: Neural network layers and models
//! - [`dataset`]: Data loading and preprocessing
//!
//! ## Feature Flags
//!
//! ### Default Features
//! - `std`: Standard library support
//! - `parallel`: Parallel execution via Rayon
//!
//! ### GPU Acceleration
//! - `gpu`: GPU acceleration via WGPU (Metal, Vulkan, DirectX, WebGPU)
//! - `cuda`: CUDA support (Linux/Windows only)
//! - `cudnn`: cuDNN support (requires CUDA)
//! - `opencl`: OpenCL support
//! - `metal`: Metal support (macOS only)
//! - `rocm`: ROCm support (AMD GPUs)
//! - `nccl`: NCCL for distributed GPU training
//!
//! ### BLAS Acceleration
//! - `blas`: Generic BLAS support
//! - `blas-openblas`: OpenBLAS acceleration
//! - `blas-mkl`: Intel MKL acceleration
//! - `blas-accelerate`: Apple Accelerate framework (macOS only)
//!
//! ### Performance & Optimization
//! - `simd`: SIMD vectorization optimizations
//!
//! ### Serialization & I/O
//! - `serialize`: Serialization support (JSON, MessagePack)
//! - `compression`: Compression support for checkpoints
//! - `onnx`: ONNX model import/export
//!
//! ### Platform Support
//! - `wasm`: WebAssembly support
//!
//! ### Development
//! - `autograd`: Automatic differentiation support
//! - `benchmark`: Benchmarking utilities
//!
//! ### Language Bindings
//! - `python`: Python bindings via PyO3
//!
//! ### Convenience
//! - `full`: Enable most features (gpu, blas-openblas, simd, serialize, compression, onnx, autograd, python)
//!
//! ## SciRS2 Integration
//!
//! TenfloweRS is built on top of the SciRS2 ecosystem:
//!
//! ```
//! TenfloweRS (Deep Learning Framework)
//!     ↓ builds upon
//! OptiRS (ML Optimization)
//!     ↓ builds upon
//! SciRS2 (Scientific Computing Foundation)
//! ```
//!
//! This integration provides:
//! - Advanced numerical operations via `scirs2-core`
//! - Automatic differentiation via `scirs2-autograd`
//! - Neural network abstractions via `scirs2-neural`
//! - Optimized algorithms via `optirs`

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]

// Re-export all public APIs from subcrates
pub use tenflowers_core as core;
pub use tenflowers_autograd as autograd;
pub use tenflowers_neural as neural;
pub use tenflowers_dataset as dataset;

#[cfg(feature = "python")]
pub use tenflowers_ffi as ffi;

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and traits,
/// allowing users to get started quickly with a single glob import:
///
/// ```rust
/// use tenflowers::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use crate::core::{Tensor, Device, dtype};
    pub use crate::core::ops;

    // Autograd
    pub use crate::autograd::{GradientTape, TrackedTensor};

    // Neural network layers
    pub use crate::neural::layers::{Dense, Conv2D, MaxPool2D, Dropout, BatchNorm};
    pub use crate::neural::ActivationFunction;

    // Models
    pub use crate::neural::{Sequential, Model};

    // Optimizers
    pub use crate::neural::{SGD, Adam, AdamW};

    // Loss functions
    pub use crate::neural::{
        mse, categorical_cross_entropy, binary_cross_entropy,
    };

    // Training utilities
    pub use crate::neural::{quick_train, Trainer};

    // Callbacks
    pub use crate::neural::{EarlyStopping, ModelCheckpoint};

    // Dataset
    pub use crate::dataset::{
        DataLoader, DataLoaderBuilder,
        CsvDataset, CsvDatasetBuilder,
        ImageFolderDataset, ImageFolderDatasetBuilder,
    };

    // Common trait re-exports
    pub use crate::neural::Layer;
    pub use crate::dataset::Dataset;
}

/// Common types and utilities
///
/// This module provides type aliases and utility functions that are
/// commonly used throughout TenfloweRS applications.
pub mod common {
    /// Result type using TenfloweRS error types
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    /// Shape type for tensor dimensions
    pub type Shape = Vec<usize>;
}

// Version information
/// The version of the TenfloweRS framework
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the version string of TenfloweRS
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
        assert_eq!(version(), VERSION);
    }
}
