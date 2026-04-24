// TensorError is a large enum; allow result_large_err since the return type is
// dictated by tenflowers_core.
#![allow(clippy::result_large_err)]
//! Conversion helpers between [`Tensor<T>`](tenflowers_core::Tensor) and
//! [`ndarray::ArrayD<T>`](scirs2_core::ndarray::ArrayD).
//!
//! # Ownership model
//!
//! Both `Tensor<T>` and `ArrayD<T>` own their heap allocation.  All conversions
//! here are **safe copies** — the data is cloned into a fresh allocation.
//!
//! A zero-copy variant (shared `Arc` storage) is tracked as a follow-up item.
//!
//! # Feature gate
//!
//! This module is unconditionally available because `scirs2-core` (which owns
//! the ndarray type alias) is already a transitive dependency of `tenflowers-core`.
//! No extra Cargo feature is required.
//!
//! # Example
//!
//! ```rust,no_run
//! use tenflowers::interop::ndarray::{from_ndarray, to_ndarray};
//! use scirs2_core::ndarray::Array2;
//!
//! let arr = Array2::<f32>::zeros((3, 4)).into_dyn();
//! let tensor = from_ndarray(arr);
//! assert_eq!(tensor.shape().dims(), &[3, 4]);
//!
//! let roundtrip = to_ndarray(&tensor).expect("conversion should succeed");
//! assert_eq!(roundtrip.shape(), &[3, 4]);
//! ```

use scirs2_core::ndarray::{ArrayD, IxDyn};
use tenflowers_core::tensor::TensorStorage;
use tenflowers_core::{Result, Tensor, TensorError};

/// Convert an [`ndarray::ArrayD<T>`](ArrayD) into a TenfloweRS [`Tensor<T>`].
///
/// The array data is **copied** into a new CPU tensor.  The resulting tensor is
/// placed on the CPU device and does not require gradient computation.
///
/// # Errors
///
/// Returns [`TensorError`] if the array is not contiguous in memory and the
/// copy cannot be performed.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers::interop::ndarray::from_ndarray;
/// use scirs2_core::ndarray::Array2;
///
/// let arr = Array2::<f32>::from_elem((2, 3), 1.0_f32).into_dyn();
/// let tensor = from_ndarray(arr);
/// assert_eq!(tensor.shape().dims(), &[2, 3]);
/// ```
pub fn from_ndarray<T>(array: ArrayD<T>) -> Tensor<T>
where
    T: Clone + Default,
{
    Tensor::from_array(array)
}

/// Convert a TenfloweRS [`Tensor<T>`] into an `ndarray::ArrayD<T>`.
///
/// Only CPU tensors are supported.  The data is **copied** into a fresh
/// dynamically-shaped ndarray.
///
/// # Errors
///
/// Returns [`TensorError::UnsupportedDevice`] if the tensor resides on a
/// non-CPU device.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers::interop::ndarray::to_ndarray;
/// use tenflowers_core::Tensor;
///
/// let tensor = Tensor::<f32>::zeros(&[2, 3]);
/// let arr = to_ndarray(&tensor).expect("conversion should succeed");
/// assert_eq!(arr.shape(), &[2, 3]);
/// ```
pub fn to_ndarray<T>(tensor: &Tensor<T>) -> Result<ArrayD<T>>
where
    T: Clone + Default,
{
    match &tensor.storage {
        TensorStorage::Cpu(arr) => {
            // Clone the underlying ArrayD — this is a safe copy.
            Ok(arr.clone())
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => Err(TensorError::unsupported_device(
            "to_ndarray",
            "gpu",
            true, // fallback available: call .to_cpu() first
        )),
    }
}

/// Round-trip helper: convert [`Tensor<T>`] → [`ArrayD<T>`] → [`Tensor<T>`].
///
/// Primarily useful for testing that `from_ndarray(to_ndarray(t))` preserves
/// shape and data.
///
/// # Errors
///
/// Propagates any error from [`to_ndarray`].
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers::interop::ndarray::round_trip;
/// use tenflowers_core::Tensor;
///
/// let original = Tensor::<f32>::ones(&[4, 4]);
/// let copied = round_trip(&original).expect("round-trip should succeed");
/// assert_eq!(original.shape().dims(), copied.shape().dims());
/// ```
pub fn round_trip<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default,
{
    let array = to_ndarray(tensor)?;
    Ok(from_ndarray(array))
}

/// Attempt to convert a slice of raw data with an explicit shape directly into
/// a [`Tensor<T>`] via ndarray's `from_shape_vec`.
///
/// This is a convenience wrapper that avoids the caller needing to import
/// ndarray directly.
///
/// # Errors
///
/// Returns [`TensorError::InvalidShape`] if `data.len()` does not equal the
/// product of `shape`.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers::interop::ndarray::from_slice_with_shape;
///
/// let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let tensor = from_slice_with_shape(data, &[2, 3]).expect("shape is valid");
/// assert_eq!(tensor.shape().dims(), &[2, 3]);
/// ```
pub fn from_slice_with_shape<T>(data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default,
{
    let total: usize = shape.iter().product();
    if data.len() != total {
        return Err(TensorError::invalid_shape_simple(format!(
            "data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            total
        )));
    }
    let array = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
    Ok(Tensor::from_array(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ndarray_shape_preserved() {
        use scirs2_core::ndarray::Array2;
        let arr = Array2::<f32>::zeros((3, 4)).into_dyn();
        let tensor = from_ndarray(arr);
        assert_eq!(tensor.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_to_ndarray_shape_preserved() {
        let tensor = Tensor::<f32>::ones(&[2, 5]);
        let arr = to_ndarray(&tensor).expect("to_ndarray should succeed on CPU tensor");
        assert_eq!(arr.shape(), &[2, 5]);
    }

    #[test]
    fn test_to_ndarray_values_preserved() {
        use scirs2_core::ndarray::Array1;
        let arr = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let tensor = from_ndarray(arr);
        let back = to_ndarray(&tensor).expect("to_ndarray should succeed");
        assert_eq!(back.as_slice().expect("contiguous"), &[1.0_f64, 2.0, 3.0]);
    }

    #[test]
    fn test_round_trip_f32() {
        let original = Tensor::<f32>::zeros(&[2, 3]);
        let copied = round_trip(&original).expect("round_trip should succeed");
        assert_eq!(original.shape().dims(), copied.shape().dims());
    }

    #[test]
    fn test_round_trip_f64() {
        let original = Tensor::<f64>::ones(&[5, 5]);
        let copied = round_trip(&original).expect("round_trip should succeed");
        assert_eq!(original.shape().dims(), copied.shape().dims());
    }

    #[test]
    fn test_from_slice_with_shape_valid() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = from_slice_with_shape(data, &[2, 3]).expect("valid shape");
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_from_slice_with_shape_mismatch_errors() {
        let data = vec![1.0_f32, 2.0];
        let result = from_slice_with_shape(data, &[2, 3]);
        assert!(result.is_err(), "mismatched length should return Err");
    }

    #[test]
    fn test_from_slice_with_shape_i32() {
        let data: Vec<i32> = (0..12).collect();
        let tensor = from_slice_with_shape(data, &[3, 4]).expect("valid shape");
        assert_eq!(tensor.shape().dims(), &[3, 4]);
    }
}
