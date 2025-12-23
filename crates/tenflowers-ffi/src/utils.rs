//! Utility functions for TenfloweRS FFI
//!
//! This module provides convenient utility functions for common operations,
//! tensor manipulation, model inspection, and debugging.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Get tensor information as a dictionary
///
/// # Arguments
///
/// * `tensor` - Tensor to inspect
///
/// # Returns
///
/// Dictionary containing:
/// - shape: List of dimensions
/// - ndim: Number of dimensions
/// - numel: Total number of elements
/// - dtype: Data type (currently always float32)
/// - requires_grad: Whether gradients are tracked
/// - is_pinned: Whether memory is pinned
#[pyfunction]
pub fn tensor_info(py: Python, tensor: &PyTensor) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    // Shape information
    let shape = tensor.shape();
    dict.set_item("shape", shape.clone())?;
    dict.set_item("ndim", shape.len())?;

    // Calculate number of elements
    let numel: usize = shape.iter().product();
    dict.set_item("numel", numel)?;

    // Data type (currently only f32)
    dict.set_item("dtype", "float32")?;

    // Gradient tracking
    dict.set_item("requires_grad", tensor.requires_grad)?;

    // Memory pinning
    dict.set_item("is_pinned", tensor.is_pinned)?;

    Ok(dict.unbind())
}

/// Check if two tensors have the same shape
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// True if shapes match, False otherwise
#[pyfunction]
pub fn same_shape(a: &PyTensor, b: &PyTensor) -> bool {
    a.shape() == b.shape()
}

/// Check if a tensor is scalar (0-dimensional or shape [1])
///
/// # Arguments
///
/// * `tensor` - Tensor to check
///
/// # Returns
///
/// True if tensor is scalar, False otherwise
#[pyfunction]
pub fn is_scalar(tensor: &PyTensor) -> bool {
    let shape = tensor.shape();
    shape.is_empty() || (shape.len() == 1 && shape[0] == 1)
}

/// Check if a tensor is a vector (1-dimensional)
///
/// # Arguments
///
/// * `tensor` - Tensor to check
///
/// # Returns
///
/// True if tensor is a vector, False otherwise
#[pyfunction]
pub fn is_vector(tensor: &PyTensor) -> bool {
    tensor.shape().len() == 1
}

/// Check if a tensor is a matrix (2-dimensional)
///
/// # Arguments
///
/// * `tensor` - Tensor to check
///
/// # Returns
///
/// True if tensor is a matrix, False otherwise
#[pyfunction]
pub fn is_matrix(tensor: &PyTensor) -> bool {
    tensor.shape().len() == 2
}

/// Get the total number of elements in a tensor
///
/// # Arguments
///
/// * `tensor` - Tensor to inspect
///
/// # Returns
///
/// Total number of elements
#[pyfunction]
pub fn numel(tensor: &PyTensor) -> usize {
    tensor.shape().iter().product()
}

/// Validate tensor shapes for binary operations
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `operation` - Name of the operation for error messages
///
/// # Returns
///
/// Ok(()) if shapes are compatible, Err otherwise
#[pyfunction]
pub fn validate_shapes(a: &PyTensor, b: &PyTensor, operation: &str) -> PyResult<()> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a != shape_b {
        return Err(PyValueError::new_err(format!(
            "{}: shape mismatch: {:?} vs {:?}",
            operation, shape_a, shape_b
        )));
    }

    Ok(())
}

/// Create a summary string for a tensor
///
/// # Arguments
///
/// * `tensor` - Tensor to summarize
/// * `name` - Optional name for the tensor
///
/// # Returns
///
/// Formatted string with tensor information
#[pyfunction]
#[pyo3(signature = (tensor, name=None))]
pub fn tensor_summary(tensor: &PyTensor, name: Option<&str>) -> String {
    let shape = tensor.shape();
    let numel: usize = shape.iter().product();

    let name_str = name.unwrap_or("Tensor");
    let grad_str = if tensor.requires_grad {
        ", requires_grad=True"
    } else {
        ""
    };
    let pinned_str = if tensor.is_pinned {
        ", pinned=True"
    } else {
        ""
    };

    format!(
        "{}(shape={:?}, numel={}, dtype=float32{}{})",
        name_str, shape, numel, grad_str, pinned_str
    )
}

/// Check if all tensors in a list have the same shape
///
/// # Arguments
///
/// * `tensors` - List of tensors
///
/// # Returns
///
/// True if all shapes match, False otherwise
#[pyfunction]
pub fn all_same_shape(tensors: Vec<PyTensor>) -> bool {
    if tensors.is_empty() {
        return true;
    }

    let first_shape = tensors[0].shape();
    tensors.iter().all(|t| t.shape() == first_shape)
}

/// Get the broadcast shape for two tensor shapes
///
/// # Arguments
///
/// * `shape_a` - First shape
/// * `shape_b` - Second shape
///
/// # Returns
///
/// Broadcast shape if compatible, None otherwise
#[pyfunction]
pub fn broadcast_shape(shape_a: Vec<usize>, shape_b: Vec<usize>) -> PyResult<Vec<usize>> {
    let ndim_a = shape_a.len();
    let ndim_b = shape_b.len();
    let max_ndim = ndim_a.max(ndim_b);

    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let dim_a = if i < ndim_a {
            shape_a[ndim_a - 1 - i]
        } else {
            1
        };
        let dim_b = if i < ndim_b {
            shape_b[ndim_b - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            return Err(PyValueError::new_err(format!(
                "Shapes {:?} and {:?} are not broadcastable",
                shape_a, shape_b
            )));
        }
    }

    result.reverse();
    Ok(result)
}

/// Check if two shapes are broadcastable
///
/// # Arguments
///
/// * `shape_a` - First shape
/// * `shape_b` - Second shape
///
/// # Returns
///
/// True if shapes are broadcastable, False otherwise
#[pyfunction]
pub fn is_broadcastable(shape_a: Vec<usize>, shape_b: Vec<usize>) -> bool {
    broadcast_shape(shape_a, shape_b).is_ok()
}

/// Calculate memory usage of a tensor in bytes
///
/// # Arguments
///
/// * `tensor` - Tensor to calculate memory for
///
/// # Returns
///
/// Memory usage in bytes
#[pyfunction]
pub fn tensor_memory_bytes(tensor: &PyTensor) -> usize {
    let numel: usize = tensor.shape().iter().product();
    // Currently only f32 (4 bytes per element)
    numel * 4
}

/// Calculate memory usage of a tensor in human-readable format
///
/// # Arguments
///
/// * `tensor` - Tensor to calculate memory for
///
/// # Returns
///
/// Memory usage as a human-readable string (e.g., "1.2 MB")
#[pyfunction]
pub fn tensor_memory_str(tensor: &PyTensor) -> String {
    let bytes = tensor_memory_bytes(tensor);
    format_bytes(bytes)
}

/// Format bytes as human-readable string
///
/// # Arguments
///
/// * `bytes` - Number of bytes
///
/// # Returns
///
/// Formatted string (e.g., "1.2 MB")
#[pyfunction]
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Print tensor information to console
///
/// # Arguments
///
/// * `tensor` - Tensor to print
/// * `name` - Optional name for the tensor
#[pyfunction]
#[pyo3(signature = (tensor, name=None))]
pub fn print_tensor_info(tensor: &PyTensor, name: Option<&str>) {
    println!("{}", tensor_summary(tensor, name));
}

/// Validate that a dimension index is valid for a tensor
///
/// # Arguments
///
/// * `tensor` - Tensor to validate against
/// * `dim` - Dimension index
///
/// # Returns
///
/// Ok(()) if valid, Err otherwise
#[pyfunction]
pub fn validate_dimension(tensor: &PyTensor, dim: i32) -> PyResult<()> {
    let ndim = tensor.shape().len() as i32;

    if dim < -ndim || dim >= ndim {
        return Err(PyValueError::new_err(format!(
            "Dimension {} out of range for {}-D tensor (valid range: {} to {})",
            dim,
            ndim,
            -ndim,
            ndim - 1
        )));
    }

    Ok(())
}

/// Normalize a dimension index to be positive
///
/// # Arguments
///
/// * `tensor` - Tensor to normalize dimension for
/// * `dim` - Dimension index (can be negative)
///
/// # Returns
///
/// Normalized positive dimension index
#[pyfunction]
pub fn normalize_dimension(tensor: &PyTensor, dim: i32) -> PyResult<usize> {
    validate_dimension(tensor, dim)?;

    let ndim = tensor.shape().len() as i32;
    let normalized = if dim < 0 { ndim + dim } else { dim };

    Ok(normalized as usize)
}

/// Create a range tensor (similar to arange in NumPy)
///
/// # Arguments
///
/// * `start` - Start value (inclusive)
/// * `end` - End value (exclusive)
/// * `step` - Step size (default: 1.0)
///
/// # Returns
///
/// 1D tensor containing range of values
#[pyfunction]
#[pyo3(signature = (start, end, step=None))]
pub fn arange(start: f32, end: f32, step: Option<f32>) -> PyResult<Vec<f32>> {
    let step = step.unwrap_or(1.0);

    if step == 0.0 {
        return Err(PyValueError::new_err("Step cannot be zero"));
    }

    if (end - start) * step < 0.0 {
        return Err(PyValueError::new_err(
            "Step has wrong sign for given start and end",
        ));
    }

    let mut result = Vec::new();
    let mut current = start;

    while (step > 0.0 && current < end) || (step < 0.0 && current > end) {
        result.push(current);
        current += step;
    }

    Ok(result)
}

/// Create a linearly spaced tensor
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value
/// * `num` - Number of samples
///
/// # Returns
///
/// 1D tensor containing linearly spaced values
#[pyfunction]
pub fn linspace(start: f32, end: f32, num: usize) -> PyResult<Vec<f32>> {
    if num == 0 {
        return Err(PyValueError::new_err("Number of samples must be positive"));
    }

    if num == 1 {
        return Ok(vec![start]);
    }

    let step = (end - start) / (num - 1) as f32;
    let result: Vec<f32> = (0..num).map(|i| start + step * i as f32).collect();

    Ok(result)
}

/// Get device information as a string
///
/// # Returns
///
/// String describing current default device
#[pyfunction]
pub fn get_device_info() -> String {
    // This would integrate with the actual device management
    "CPU (default)".to_string()
}

/// Check if GPU is available
///
/// # Returns
///
/// True if GPU is available, False otherwise
#[pyfunction]
pub fn is_gpu_available() -> bool {
    // This would check actual GPU availability
    #[cfg(feature = "gpu")]
    {
        true
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Get TenfloweRS version
///
/// # Returns
///
/// Version string
#[pyfunction]
pub fn version() -> &'static str {
    "0.1.0-alpha.2"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tenflowers_core::Tensor;

    fn create_test_tensor(shape: Vec<usize>) -> PyTensor {
        let tensor = Tensor::zeros(&shape);
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    #[test]
    fn test_is_scalar() {
        assert!(is_scalar(&create_test_tensor(vec![])));
        assert!(is_scalar(&create_test_tensor(vec![1])));
        assert!(!is_scalar(&create_test_tensor(vec![2])));
    }

    #[test]
    fn test_is_vector() {
        assert!(is_vector(&create_test_tensor(vec![5])));
        assert!(!is_vector(&create_test_tensor(vec![2, 3])));
    }

    #[test]
    fn test_is_matrix() {
        assert!(is_matrix(&create_test_tensor(vec![2, 3])));
        assert!(!is_matrix(&create_test_tensor(vec![5])));
    }

    #[test]
    fn test_numel() {
        assert_eq!(numel(&create_test_tensor(vec![2, 3, 4])), 24);
        assert_eq!(numel(&create_test_tensor(vec![5])), 5);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(100), "100 B");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }

    #[test]
    fn test_arange() {
        let result = arange(0.0, 5.0, Some(1.0)).unwrap();
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let result = arange(1.0, 2.0, Some(0.25)).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_linspace() {
        let result = linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[4], 1.0);
    }

    #[test]
    fn test_broadcast_shape() {
        let result = broadcast_shape(vec![3, 1, 5], vec![1, 4, 5]).unwrap();
        assert_eq!(result, vec![3, 4, 5]);

        let result = broadcast_shape(vec![1], vec![3, 4, 5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_broadcastable() {
        assert!(is_broadcastable(vec![3, 1, 5], vec![1, 4, 5]));
        assert!(!is_broadcastable(vec![3, 2], vec![3, 3]));
    }
}
