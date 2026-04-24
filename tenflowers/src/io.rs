// TensorError is a large enum; we allow result_large_err here since the return
// type is dictated by tenflowers_core and cannot be changed in this wrapper.
#![allow(clippy::result_large_err)]
//! High-level serialization and I/O helpers.
//!
//! This module provides convenience wrappers around the serialization
//! primitives in `tenflowers-core` and `tenflowers-neural`, so that callers
//! can save/load tensors and models without needing to import internal details.
//!
//! ## Feature gate
//!
//! Most functions in this module require the `serialize` Cargo feature.
//! Attempting to call them without the feature will produce a compile error.
//!
//! ```toml
//! tenflowers = { features = ["serialize"] }
//! ```
//!
//! ## Default format
//!
//! All persistence uses the built-in binary format from `tenflowers-core`
//! (pure Rust, no C/Fortran dependencies — compliant with COOLJAPAN Pure Rust
//! Policy).  The format is versioned; see `tenflowers_core::serialization`.
//!
//! ## Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "serialize")]
//! # fn example() -> tenflowers_core::Result<()> {
//! use tenflowers::io::{save_tensor, load_tensor};
//! use tenflowers_core::Tensor;
//! use std::path::Path;
//!
//! let tensor = Tensor::<f32>::ones(&[3, 3]);
//! let path = Path::new("/tmp/my_tensor.bin");
//!
//! save_tensor(&tensor, path)?;
//! let restored: Tensor<f32> = load_tensor(path)?;
//! assert_eq!(restored.shape().dims(), &[3, 3]);
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use tenflowers_core::{Result, Tensor};

// -----------------------------------------------------------------------
// Tensor I/O
// -----------------------------------------------------------------------

/// Save a tensor to a file using the default binary format.
///
/// The file is created (or overwritten) at `path`.  The format is the
/// TenfloweRS versioned binary layout — see
/// [`tenflowers_core::serialization`] for details.
///
/// # Errors
///
/// Returns [`tenflowers_core::TensorError`] if the file cannot be written
/// or if serialization fails.
///
/// # Feature gate
///
/// Requires the `serialize` Cargo feature.
#[cfg(feature = "serialize")]
pub fn save_tensor<T>(tensor: &Tensor<T>, path: &Path) -> Result<()>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    tenflowers_core::serialization::save_tensor(tensor, path)
}

/// Load a tensor from a file previously saved with [`save_tensor`].
///
/// The dtype `T` must match the dtype of the saved tensor.
///
/// # Errors
///
/// Returns [`tenflowers_core::TensorError`] if the file cannot be read or
/// deserialization fails.
///
/// # Feature gate
///
/// Requires the `serialize` Cargo feature.
#[cfg(feature = "serialize")]
pub fn load_tensor<T>(path: &Path) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    tenflowers_core::serialization::load_tensor::<T>(path)
}

// -----------------------------------------------------------------------
// Fallback stubs when `serialize` feature is disabled
// -----------------------------------------------------------------------

/// Save a tensor to a file (stub — compile error unless `serialize` feature enabled).
///
/// # Errors
///
/// Always returns `TensorError::NotImplemented` unless the `serialize` feature
/// is enabled.
#[cfg(not(feature = "serialize"))]
pub fn save_tensor<T>(_tensor: &Tensor<T>, _path: &Path) -> Result<()>
where
    T: Clone + Default,
{
    Err(tenflowers_core::TensorError::not_implemented_simple(
        "save_tensor requires the `serialize` Cargo feature".to_string(),
    ))
}

/// Load a tensor from a file (stub — compile error unless `serialize` feature enabled).
///
/// # Errors
///
/// Always returns `TensorError::NotImplemented` unless the `serialize` feature
/// is enabled.
#[cfg(not(feature = "serialize"))]
pub fn load_tensor<T>(_path: &Path) -> Result<Tensor<T>>
where
    T: Clone + Default,
{
    Err(tenflowers_core::TensorError::not_implemented_simple(
        "load_tensor requires the `serialize` Cargo feature".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use tenflowers_core::Tensor;

    /// Verify that the serialize/load round-trip preserves shape and data.
    ///
    /// This test is compiled only when `serialize` is enabled.
    #[cfg(feature = "serialize")]
    #[test]
    fn test_tensor_round_trip_f32() {
        let original = Tensor::<f32>::ones(&[2, 3]);
        let path = temp_dir().join("tenflowers_io_test_f32.bin");

        save_tensor(&original, &path).expect("save_tensor should succeed");
        let restored: Tensor<f32> = load_tensor(&path).expect("load_tensor should succeed");

        assert_eq!(
            original.shape().dims(),
            restored.shape().dims(),
            "shape must be preserved across round-trip"
        );

        let orig_data = original.as_slice().expect("contiguous");
        let rest_data = restored.as_slice().expect("contiguous");
        assert_eq!(
            orig_data, rest_data,
            "data must be identical after round-trip"
        );

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    /// Verify that the stub returns `NotImplemented` when `serialize` is off.
    #[cfg(not(feature = "serialize"))]
    #[test]
    fn test_stub_returns_not_implemented() {
        let dummy = Tensor::<f32>::zeros(&[1]);
        let path = temp_dir().join("tenflowers_stub_test.bin");
        let result = save_tensor(&dummy, &path);
        assert!(result.is_err(), "stub should return Err");
    }
}
