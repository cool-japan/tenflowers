//! Interoperability utilities for TenfloweRS.
//!
//! This module provides conversion helpers between TenfloweRS tensors and
//! external array / data representations, enabling seamless data exchange
//! with the broader Rust scientific computing ecosystem.
//!
//! ## Modules
//!
//! - [`ndarray`]: Conversions between [`Tensor<T>`](tenflowers_core::Tensor) and
//!   [`ndarray::ArrayD<T>`](scirs2_core::ndarray::ArrayD) from the `scirs2-core`
//!   re-export (the canonical ndarray path in this project).
//!
//! ## Design notes
//!
//! Conversions are **safe copies** for this initial release. Both TenfloweRS
//! `Tensor<T>` and ndarray `ArrayD<T>` own their heap data, so a zero-copy path
//! would require lifetime entanglement or `Arc`-based sharing. A zero-copy variant
//! is tracked as a follow-up (see `TODO.md`).
//!
//! ## Example
//!
//! ```rust,no_run
//! use tenflowers::interop::ndarray::{from_ndarray, to_ndarray};
//! use scirs2_core::ndarray::Array2;
//!
//! // ndarray -> Tensor
//! let arr = Array2::<f32>::zeros((3, 4)).into_dyn();
//! let tensor = from_ndarray(arr);
//! assert_eq!(tensor.shape().dims(), &[3, 4]);
//!
//! // Tensor -> ndarray (to_ndarray returns Result<ArrayD<T>>)
//! let round_tripped = to_ndarray(&tensor).unwrap();
//! assert_eq!(round_tripped.shape(), &[3, 4]);
//! ```

/// Conversion helpers between TenfloweRS tensors and `ndarray::ArrayD`.
pub mod ndarray;
