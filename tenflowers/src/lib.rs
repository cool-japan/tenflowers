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
//! let model = Sequential::<f32>::new(vec![])
//!     .add(Box::new(Dense::new(784, 128, true).with_activation("relu".to_string())))
//!     .add(Box::new(Dense::new(128, 10, true).with_activation("sigmoid".to_string())));
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
//! // Create model and data
//! let model = Sequential::<f32>::new(vec![])
//!     .add(Box::new(Dense::new(10, 64, true).with_activation("relu".to_string())))
//!     .add(Box::new(Dense::new(64, 3, true)));
//! let x_train = Tensor::<f32>::zeros(&[100, 10]);
//! let y_train = Tensor::<f32>::zeros(&[100, 3]);
//!
//! // Create optimizer and loss function
//! let optimizer = SGD::<f32>::new(0.01);
//! // Training loop would go here using Trainer
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
//! let device = Device::try_gpu(0)?;
//! let gpu_tensor = Tensor::<f32>::zeros(&[1000, 1000]).to_device(device)?;
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
//! use tenflowers::dataset::RandomSampler;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load dataset
//! let dataset: CsvDataset<f32> = CsvDatasetBuilder::new()
//!     .from_path("data.csv")
//!     .has_header(true)
//!     .build()?;
//!
//! // Create data loader with batching and shuffling
//! let loader = DataLoaderBuilder::new(dataset)
//!     .batch_size(32)
//!     .num_workers(4)
//!     .build(RandomSampler::new());
//!
//! // Iterate through batches
//! for batch in loader.iter() {
//!     let (features, labels) = batch?.into_collated()?;
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
//! ```text
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
pub use tenflowers_autograd as autograd;
pub use tenflowers_core as core;
pub use tenflowers_dataset as dataset;
pub use tenflowers_neural as neural;

// Declarative macros (tensor![], etc.)
pub mod macros;

// #[cfg(feature = "python")]
// pub use tenflowers_ffi as ffi;

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
    pub use crate::core::ops;
    pub use crate::core::{dtype, Device, Tensor};

    // Autograd
    pub use crate::autograd::{GradientTape, TrackedTensor};

    // Neural network layers
    pub use crate::neural::layers::{BatchNorm, Conv2D, Dense, Dropout, MaxPool2D};
    pub use crate::neural::ActivationFunction;

    // Models
    pub use crate::neural::{Model, Sequential};

    // Optimizers
    pub use crate::neural::{Adam, AdamW, SGD};

    // Loss functions
    pub use crate::neural::{binary_cross_entropy, categorical_cross_entropy, mse};

    // Training utilities
    pub use crate::neural::{quick_train, Trainer};

    // Callbacks
    pub use crate::neural::{EarlyStopping, ModelCheckpoint};

    // Dataset
    pub use crate::dataset::{
        CsvDataset, CsvDatasetBuilder, DataLoader, DataLoaderBuilder, ImageFolderDataset,
        ImageFolderDatasetBuilder,
    };

    // Common trait re-exports
    pub use crate::dataset::Dataset;
    pub use crate::neural::Layer;
}

/// Neural network layers, activations, and models
///
/// Provides a convenient `tenflowers::nn` alias for the most commonly used
/// layer types and neural network building blocks from `tenflowers_neural`.
///
/// # Example
///
/// ```rust
/// use tenflowers::nn::Dense;
/// let layer = Dense::<f32>::new(4, 2, true);
/// ```
pub mod nn {
    pub use tenflowers_neural::layers::{BatchNorm, MaxPool2D};
    pub use tenflowers_neural::{
        ActivationFunction, Conv2D, Dense, Dropout, Layer, Model, MultiHeadAttention, RMSNorm,
        Sequential, TransformerDecoder, TransformerEncoder, GRU, LSTM, RNN,
    };
}

/// Optimization algorithms
///
/// Provides a convenient `tenflowers::optim` alias for the optimizer types
/// exported from `tenflowers_neural`.
///
/// # Example
///
/// ```rust
/// use tenflowers::optim::Adam;
/// let opt = Adam::<f32>::new(0.001);
/// ```
pub mod optim {
    pub use tenflowers_neural::{
        Adadelta, Adagrad, Adam, AdamW, Lion, Lookahead, Nadam, Optimizer, ParameterGroup,
        ParameterGroupOptimizer, RAdam, RMSprop, LAMB, SGD,
    };
}

/// Data pipeline and dataset utilities
///
/// Provides a convenient `tenflowers::data` alias for the dataset types
/// from `tenflowers_dataset`.
///
/// # Example
///
/// ```rust
/// use tenflowers::data::Dataset;
/// ```
pub mod data {
    pub use tenflowers_dataset::{
        CsvDataset, CsvDatasetBuilder, DataLoader, DataLoaderBuilder, Dataset, ImageFolderDataset,
        ImageFolderDatasetBuilder, RandomSampler,
    };
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

/// Structured version metadata for the TenfloweRS framework.
///
/// Returned by [`version_info()`]; contains the version string, package name,
/// and a short human-readable description populated at compile time via
/// `env!()` macros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionInfo {
    /// Semver version string (e.g. `"0.1.0"`).
    pub version: &'static str,
    /// Crate / package name (always `"tenflowers"`).
    pub pkg_name: &'static str,
    /// One-line description from `Cargo.toml`.
    pub description: &'static str,
}

impl std::fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} v{} — {}",
            self.pkg_name, self.version, self.description
        )
    }
}

/// Returns structured version metadata populated at compile time.
///
/// # Example
///
/// ```rust
/// let info = tenflowers::version_info();
/// assert!(!info.version.is_empty());
/// assert_eq!(info.pkg_name, "tenflowers");
/// ```
pub fn version_info() -> VersionInfo {
    VersionInfo {
        version: env!("CARGO_PKG_VERSION"),
        pkg_name: env!("CARGO_PKG_NAME"),
        description: env!("CARGO_PKG_DESCRIPTION"),
    }
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
