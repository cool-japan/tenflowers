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
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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

// Deprecation helpers for wrapping pub-use items with #[deprecated].
pub mod deprecations;

// Common-shape type aliases (Tensor1D, Tensor2D, Vector, Matrix, …)
pub mod type_aliases;
pub use type_aliases::*;

// TensorError at crate root; CoreResult preserves the TensorError-based alias.
pub use tenflowers_core::Result as CoreResult;
pub use tenflowers_core::TensorError;

// Interoperability utilities (ndarray conversions, etc.)
pub mod interop;

// High-level I/O helpers (save/load tensors and models).
pub mod io;

/// Unified error types for the TenfloweRS framework.
///
/// See [`error::FrameworkError`] and [`error::Result`].
pub mod error;

/// Top-level `Result` alias using [`error::FrameworkError`].
///
/// ```rust
/// use tenflowers::Result;
///
/// fn ok_val() -> Result<u32> { Ok(1) }
/// assert_eq!(ok_val().unwrap(), 1);
/// ```
pub use error::Result;

/// Subcrate version consistency checking.
///
/// See [`version_check::check_version_consistency`] and
/// [`version_check::assert_versions_consistent`].
pub mod version_check;

/// Unified logging and diagnostic interface.
///
/// See [`logging::set_log_level`], [`logging::init_from_env`], and the
/// `log_info!`, `log_warn!`, `log_error!`, `log_debug!`, `log_trace!` macros.
pub mod logging;

/// Common utility functions.
///
/// See [`utils::softmax`], [`utils::sigmoid`], [`utils::bytes_to_human_readable`], etc.
pub mod utils;

// ONNX import/export helpers (re-exports tenflowers-neural's ONNX surface).
#[cfg(feature = "onnx")]
pub mod onnx {
    //! ONNX model import/export helpers.
    //!
    //! This module re-exports the ONNX surface from `tenflowers-neural`.
    //! Enable it with the `onnx` Cargo feature:
    //!
    //! ```toml
    //! tenflowers = { features = ["onnx"] }
    //! ```
    //!
    //! ## Supported operations
    //!
    //! - [`OnnxImport`] / [`OnnxExport`] traits for model-level import and export.
    //! - [`OnnxModel`]: parsed ONNX graph representation.
    //! - [`OnnxGraph`], [`OnnxNode`], [`OnnxTensor`], [`OnnxValueInfo`]: graph primitives.
    //! - [`OnnxDataType`], [`OnnxFormat`], [`OnnxError`]: type and error enums.
    //!
    //! ## Example
    //!
    //! ```rust,ignore
    //! use tenflowers::onnx::{OnnxExport, OnnxFormat};
    //!
    //! // Export a Sequential model to an ONNX file
    //! let model = /* ... */;
    //! model.export_onnx("model.onnx", OnnxFormat::Protobuf)?;
    //! ```
    //!
    //! See [`tenflowers_neural::onnx`] for the full API surface.
    pub use tenflowers_neural::onnx::{
        OnnxAttribute, OnnxDataType, OnnxError, OnnxExport, OnnxFormat, OnnxGraph, OnnxImport,
        OnnxModel, OnnxNode, OnnxTensor, OnnxValueInfo,
    };
}

// #[cfg(feature = "python")]
// pub use tenflowers_ffi as ffi;

/// Prelude module for convenient imports.
///
/// Import the entire prelude with a single glob import to get everything you
/// need for a typical training loop without extra `use` statements:
///
/// ```rust
/// use tenflowers::prelude::*;
/// ```
///
/// See the `docs/PRELUDE_STABILITY.md` file in the meta crate for the stability
/// policy (additions allowed in minor releases; removals require a major version bump).
///
/// # Example — minimal model construction
///
/// ```rust,no_run
/// use tenflowers::prelude::*;
///
/// // Build a two-layer feedforward network
/// let model = Sequential::<f32>::new(vec![])
///     .add(Box::new(Dense::new(784, 128, true).with_activation("relu".to_string())))
///     .add(Box::new(Dense::new(128, 10, true)));
///
/// // Create an Adam optimizer
/// let _optimizer = Adam::<f32>::new(1e-3);
/// ```
///
/// # Example — MNIST-style classification train loop
///
/// ```rust,ignore
/// use tenflowers::prelude::*;
///
/// // 1. Build network: 784-input → 128 ReLU → 10-output
/// let model = Sequential::<f32>::new(vec![])
///     .add(Box::new(Dense::new(784, 128, true).with_activation("relu".to_string())))
///     .add(Box::new(Dense::new(128, 10, true)));
///
/// // 2. Dummy data (replace with a real DataLoader in production)
/// let x_train = Tensor::<f32>::zeros(&[128, 784]); // batch=128, features=784
/// let y_train = Tensor::<f32>::zeros(&[128, 10]);   // one-hot labels
///
/// // 3. Set up optimiser and trainer
/// let optimizer = SGD::<f32>::new(0.01);
/// let trainer = Trainer::new(model, optimizer);
///
/// // 4. Run 5 epochs (Trainer handles forward / backward / parameter update)
/// for epoch in 0..5 {
///     let loss = trainer.train_step(&x_train, &y_train, categorical_cross_entropy)?;
///     println!("Epoch {}: loss = {:.4}", epoch, loss);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example — regression with Adam
///
/// ```rust,ignore
/// use tenflowers::prelude::*;
///
/// // 1. Small regression network: 10-input → 64 ReLU → 1-output
/// let model = Sequential::<f32>::new(vec![])
///     .add(Box::new(Dense::new(10, 64, true).with_activation("relu".to_string())))
///     .add(Box::new(Dense::new(64, 1, true)));
///
/// // 2. Synthetic data
/// let x = Tensor::<f32>::zeros(&[200, 10]);
/// let y = Tensor::<f32>::zeros(&[200, 1]);
///
/// // 3. Adam optimiser with default hyper-parameters
/// let optimizer = Adam::<f32>::new(1e-3);
/// let trainer = Trainer::new(model, optimizer);
///
/// // 4. Training loop with MSE loss
/// for epoch in 0..20 {
///     let loss = trainer.train_step(&x, &y, mse)?;
///     println!("Epoch {}: MSE = {:.6}", epoch, loss);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example — inference from a loaded model
///
/// ```rust,ignore
/// use tenflowers::prelude::*;
///
/// // 1. Re-create the architecture (must match saved weights)
/// let mut model = Sequential::<f32>::new(vec![])
///     .add(Box::new(Dense::new(784, 128, true).with_activation("relu".to_string())))
///     .add(Box::new(Dense::new(128, 10, true)));
///
/// // 2. Load weights (oxicode / JSON backend, gated by `serialize` feature)
/// // model.load_weights(std::path::Path::new("checkpoint.json"))?;
///
/// // 3. Run inference on a single sample (shape [1, 784])
/// let sample = Tensor::<f32>::zeros(&[1, 784]);
/// let logits = model.forward(&sample)?;
///
/// // 4. Apply softmax to get probabilities (operations via the `ops` module)
/// let probs = ops::softmax(&logits, 1)?;
/// println!("Class probabilities: {:?}", probs.shape().dims());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub mod prelude {
    // Core types
    pub use crate::core::ops;
    pub use crate::core::{dtype, DType, Device, Tensor};

    // Error handling (CoreResult = TensorError-based; Result = FrameworkError-based)
    pub use crate::core::TensorError;
    pub use crate::error::{FrameworkError, Result};
    pub use crate::CoreResult;

    // Autograd
    pub use crate::autograd::{GradientTape, TrackedTensor};

    // Neural network layers
    pub use crate::neural::layers::{BatchNorm, Conv2D, Dense, Dropout, MaxPool2D};
    pub use crate::neural::ActivationFunction;

    // Transformer / attention / recurrent layers
    pub use crate::neural::{
        MultiHeadAttention, RMSNorm, TransformerDecoder, TransformerEncoder, GRU, LSTM, RNN,
    };

    // Models
    pub use crate::neural::{Model, Sequential};

    // Optimizers (trait + common types)
    pub use crate::neural::{Adam, AdamW, Optimizer, ParameterGroup, SGD};

    // Loss functions
    pub use crate::neural::{binary_cross_entropy, categorical_cross_entropy, mse};

    // Training utilities
    pub use crate::neural::{quick_train, Trainer};

    // Callbacks
    pub use crate::neural::{EarlyStopping, ModelCheckpoint};

    // Dataset
    pub use crate::dataset::{
        CsvDataset, CsvDatasetBuilder, DataLoader, DataLoaderBuilder, ImageFolderDataset,
        ImageFolderDatasetBuilder, RandomSampler,
    };

    // Common trait re-exports
    pub use crate::dataset::Dataset;
    pub use crate::neural::Layer;

    // Type aliases (Tensor1D, Tensor2D, Tensor3D, Tensor4D, Vector, Matrix, …)
    pub use crate::type_aliases::*;
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

/// Common types and utilities.
///
/// This module provides type aliases and utility functions that are
/// commonly used throughout TenfloweRS applications.
pub mod common {
    /// Shape type for tensor dimensions.
    pub type Shape = Vec<usize>;

    /// Re-export the unified framework `Result` for convenience.
    pub use crate::error::Result;
}

/// Experimental and preview APIs. Not covered by stability guarantees.
///
/// Enable with the `experimental` Cargo feature:
/// ```toml
/// tenflowers = { features = ["experimental"] }
/// ```
#[cfg(feature = "experimental")]
pub mod experimental {
    // Future experimental re-exports go here.
    // Example: pub use some_crate::UnstableType;
}

/// Platform detection and SIMD capability introspection.
///
/// See [`platform::current_platform`], [`platform::detect_simd_capabilities`].
pub mod platform;

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
    /// Semver version string (e.g. `"0.2.1"`).
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
