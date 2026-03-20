//! Error mapping module for TenfloweRS FFI
//!
//! This module provides comprehensive error mapping from Rust errors to Python exceptions,
//! implementing a unified error taxonomy for better error handling across the FFI boundary.

use pyo3::exceptions::{PyException, PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::fmt;

/// Custom exception for shape mismatch errors
pyo3::create_exception!(tenflowers, ShapeError, PyValueError);

/// Custom exception for device placement errors
pyo3::create_exception!(tenflowers, DeviceError, PyRuntimeError);

/// Custom exception for gradient computation errors
pyo3::create_exception!(tenflowers, GradientError, PyRuntimeError);

/// Custom exception for numerical stability issues
pyo3::create_exception!(tenflowers, NumericalError, PyRuntimeError);

/// Custom exception for memory allocation errors
pyo3::create_exception!(tenflowers, MemoryError, PyException);

/// Custom exception for tensor operation errors
pyo3::create_exception!(tenflowers, TensorOpError, PyRuntimeError);

/// Custom exception for layer configuration errors
pyo3::create_exception!(tenflowers, LayerConfigError, PyValueError);

/// Custom exception for optimizer errors
pyo3::create_exception!(tenflowers, OptimizerError, PyRuntimeError);

/// Custom exception for serialization errors
pyo3::create_exception!(tenflowers, SerializationError, PyRuntimeError);

/// Custom exception for data loading errors
pyo3::create_exception!(tenflowers, DataLoadError, PyRuntimeError);

/// Custom exception for graph compilation errors
pyo3::create_exception!(tenflowers, GraphCompileError, PyRuntimeError);

/// Custom exception for checkpoint errors
pyo3::create_exception!(tenflowers, CheckpointError, PyRuntimeError);

/// Unified error type for TenfloweRS FFI operations
#[derive(Debug, Clone)]
pub enum TenflowersError {
    /// Shape mismatch in tensor operations
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },
    /// Invalid dimension or axis specification
    InvalidDimension {
        dim: i32,
        ndim: usize,
        operation: String,
    },
    /// Device placement or transfer error
    DevicePlacement {
        source_device: String,
        target_device: String,
        reason: String,
    },
    /// Gradient computation error
    GradientComputation {
        tensor_name: Option<String>,
        reason: String,
    },
    /// Numerical instability detected
    NumericalInstability {
        operation: String,
        value_info: String,
    },
    /// Memory allocation failure
    MemoryAllocation {
        requested_bytes: usize,
        available_bytes: Option<usize>,
    },
    /// Invalid tensor operation
    InvalidOperation { operation: String, reason: String },
    /// Type conversion error
    TypeConversion {
        from_type: String,
        to_type: String,
        reason: String,
    },
    /// Layer configuration error
    LayerConfiguration {
        layer_type: String,
        parameter: String,
        reason: String,
    },
    /// Optimizer step error
    OptimizerStep {
        optimizer_type: String,
        reason: String,
    },
    /// Index out of bounds
    IndexOutOfBounds {
        index: i64,
        size: usize,
        axis: Option<usize>,
    },
    /// Serialization error
    Serialization { format: String, reason: String },
    /// Deserialization error
    Deserialization { format: String, reason: String },
    /// Data loading error
    DataLoad { source: String, reason: String },
    /// Graph compilation error
    GraphCompile { pass_name: String, reason: String },
    /// Model checkpoint error
    Checkpoint {
        operation: String,
        path: String,
        reason: String,
    },
    /// Dtype mismatch error
    DtypeMismatch {
        expected: String,
        actual: String,
        operation: String,
    },
    /// Not implemented error
    NotImplemented { feature: String, reason: String },
    /// Generic error with message
    Generic { message: String },
}

impl fmt::Display for TenflowersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TenflowersError::ShapeMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    operation, expected, actual
                )
            }
            TenflowersError::InvalidDimension {
                dim,
                ndim,
                operation,
            } => {
                write!(
                    f,
                    "Invalid dimension {} for {}-D tensor in operation: {}",
                    dim, ndim, operation
                )
            }
            TenflowersError::DevicePlacement {
                source_device,
                target_device,
                reason,
            } => {
                write!(
                    f,
                    "Device placement error: cannot move from {} to {} ({})",
                    source_device, target_device, reason
                )
            }
            TenflowersError::GradientComputation {
                tensor_name,
                reason,
            } => {
                if let Some(name) = tensor_name {
                    write!(f, "Gradient computation failed for '{}': {}", name, reason)
                } else {
                    write!(f, "Gradient computation failed: {}", reason)
                }
            }
            TenflowersError::NumericalInstability {
                operation,
                value_info,
            } => {
                write!(
                    f,
                    "Numerical instability detected in {}: {}",
                    operation, value_info
                )
            }
            TenflowersError::MemoryAllocation {
                requested_bytes,
                available_bytes,
            } => {
                if let Some(available) = available_bytes {
                    write!(
                        f,
                        "Memory allocation failed: requested {} bytes, only {} bytes available",
                        requested_bytes, available
                    )
                } else {
                    write!(
                        f,
                        "Memory allocation failed: requested {} bytes",
                        requested_bytes
                    )
                }
            }
            TenflowersError::InvalidOperation { operation, reason } => {
                write!(f, "Invalid operation '{}': {}", operation, reason)
            }
            TenflowersError::TypeConversion {
                from_type,
                to_type,
                reason,
            } => {
                write!(
                    f,
                    "Type conversion error from {} to {}: {}",
                    from_type, to_type, reason
                )
            }
            TenflowersError::LayerConfiguration {
                layer_type,
                parameter,
                reason,
            } => {
                write!(
                    f,
                    "Layer configuration error in {}: parameter '{}' - {}",
                    layer_type, parameter, reason
                )
            }
            TenflowersError::OptimizerStep {
                optimizer_type,
                reason,
            } => {
                write!(f, "Optimizer step failed in {}: {}", optimizer_type, reason)
            }
            TenflowersError::IndexOutOfBounds { index, size, axis } => {
                if let Some(ax) = axis {
                    write!(
                        f,
                        "Index {} out of bounds for axis {} (size: {})",
                        index, ax, size
                    )
                } else {
                    write!(f, "Index {} out of bounds (size: {})", index, size)
                }
            }
            TenflowersError::Serialization { format, reason } => {
                write!(f, "Serialization error ({} format): {}", format, reason)
            }
            TenflowersError::Deserialization { format, reason } => {
                write!(f, "Deserialization error ({} format): {}", format, reason)
            }
            TenflowersError::DataLoad { source, reason } => {
                write!(f, "Data loading error from '{}': {}", source, reason)
            }
            TenflowersError::GraphCompile { pass_name, reason } => {
                write!(
                    f,
                    "Graph compilation failed at pass '{}': {}",
                    pass_name, reason
                )
            }
            TenflowersError::Checkpoint {
                operation,
                path,
                reason,
            } => {
                write!(
                    f,
                    "Checkpoint {} failed for '{}': {}",
                    operation, path, reason
                )
            }
            TenflowersError::DtypeMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Dtype mismatch in {}: expected {}, got {}",
                    operation, expected, actual
                )
            }
            TenflowersError::NotImplemented { feature, reason } => {
                write!(f, "Feature '{}' not implemented: {}", feature, reason)
            }
            TenflowersError::Generic { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for TenflowersError {}

/// Convert TenflowersError to appropriate Python exception
impl From<TenflowersError> for PyErr {
    fn from(err: TenflowersError) -> PyErr {
        match err {
            TenflowersError::ShapeMismatch { .. } => ShapeError::new_err(err.to_string()),
            TenflowersError::InvalidDimension { .. } => PyIndexError::new_err(err.to_string()),
            TenflowersError::DevicePlacement { .. } => DeviceError::new_err(err.to_string()),
            TenflowersError::GradientComputation { .. } => GradientError::new_err(err.to_string()),
            TenflowersError::NumericalInstability { .. } => {
                NumericalError::new_err(err.to_string())
            }
            TenflowersError::MemoryAllocation { .. } => MemoryError::new_err(err.to_string()),
            TenflowersError::InvalidOperation { .. } => TensorOpError::new_err(err.to_string()),
            TenflowersError::TypeConversion { .. } => PyTypeError::new_err(err.to_string()),
            TenflowersError::LayerConfiguration { .. } => {
                LayerConfigError::new_err(err.to_string())
            }
            TenflowersError::OptimizerStep { .. } => OptimizerError::new_err(err.to_string()),
            TenflowersError::IndexOutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
            TenflowersError::Serialization { .. } => SerializationError::new_err(err.to_string()),
            TenflowersError::Deserialization { .. } => SerializationError::new_err(err.to_string()),
            TenflowersError::DataLoad { .. } => DataLoadError::new_err(err.to_string()),
            TenflowersError::GraphCompile { .. } => GraphCompileError::new_err(err.to_string()),
            TenflowersError::Checkpoint { .. } => CheckpointError::new_err(err.to_string()),
            TenflowersError::DtypeMismatch { .. } => ShapeError::new_err(err.to_string()),
            TenflowersError::NotImplemented { .. } => PyRuntimeError::new_err(err.to_string()),
            TenflowersError::Generic { .. } => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

/// Helper function to create shape mismatch error
pub fn shape_mismatch_error(
    operation: &str,
    expected: Vec<usize>,
    actual: Vec<usize>,
) -> TenflowersError {
    TenflowersError::ShapeMismatch {
        expected,
        actual,
        operation: operation.to_string(),
    }
}

/// Helper function to create invalid dimension error
pub fn invalid_dimension_error(operation: &str, dim: i32, ndim: usize) -> TenflowersError {
    TenflowersError::InvalidDimension {
        dim,
        ndim,
        operation: operation.to_string(),
    }
}

/// Helper function to create device placement error
pub fn device_placement_error(
    source_device: &str,
    target_device: &str,
    reason: &str,
) -> TenflowersError {
    TenflowersError::DevicePlacement {
        source_device: source_device.to_string(),
        target_device: target_device.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create gradient computation error
pub fn gradient_computation_error(tensor_name: Option<String>, reason: &str) -> TenflowersError {
    TenflowersError::GradientComputation {
        tensor_name,
        reason: reason.to_string(),
    }
}

/// Helper function to create numerical instability error
pub fn numerical_instability_error(operation: &str, value_info: &str) -> TenflowersError {
    TenflowersError::NumericalInstability {
        operation: operation.to_string(),
        value_info: value_info.to_string(),
    }
}

/// Helper function to create memory allocation error
pub fn memory_allocation_error(
    requested_bytes: usize,
    available_bytes: Option<usize>,
) -> TenflowersError {
    TenflowersError::MemoryAllocation {
        requested_bytes,
        available_bytes,
    }
}

/// Helper function to create invalid operation error
pub fn invalid_operation_error(operation: &str, reason: &str) -> TenflowersError {
    TenflowersError::InvalidOperation {
        operation: operation.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create type conversion error
pub fn type_conversion_error(from_type: &str, to_type: &str, reason: &str) -> TenflowersError {
    TenflowersError::TypeConversion {
        from_type: from_type.to_string(),
        to_type: to_type.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create layer configuration error
pub fn layer_configuration_error(
    layer_type: &str,
    parameter: &str,
    reason: &str,
) -> TenflowersError {
    TenflowersError::LayerConfiguration {
        layer_type: layer_type.to_string(),
        parameter: parameter.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create optimizer step error
pub fn optimizer_step_error(optimizer_type: &str, reason: &str) -> TenflowersError {
    TenflowersError::OptimizerStep {
        optimizer_type: optimizer_type.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create index out of bounds error
pub fn index_out_of_bounds_error(index: i64, size: usize, axis: Option<usize>) -> TenflowersError {
    TenflowersError::IndexOutOfBounds { index, size, axis }
}

/// Helper function to create serialization error
pub fn serialization_error(format: &str, reason: &str) -> TenflowersError {
    TenflowersError::Serialization {
        format: format.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create deserialization error
pub fn deserialization_error(format: &str, reason: &str) -> TenflowersError {
    TenflowersError::Deserialization {
        format: format.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create data load error
pub fn data_load_error(source: &str, reason: &str) -> TenflowersError {
    TenflowersError::DataLoad {
        source: source.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create graph compile error
pub fn graph_compile_error(pass_name: &str, reason: &str) -> TenflowersError {
    TenflowersError::GraphCompile {
        pass_name: pass_name.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create checkpoint error
pub fn checkpoint_error(operation: &str, path: &str, reason: &str) -> TenflowersError {
    TenflowersError::Checkpoint {
        operation: operation.to_string(),
        path: path.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create dtype mismatch error
pub fn dtype_mismatch_error(operation: &str, expected: &str, actual: &str) -> TenflowersError {
    TenflowersError::DtypeMismatch {
        expected: expected.to_string(),
        actual: actual.to_string(),
        operation: operation.to_string(),
    }
}

/// Helper function to create not implemented error
pub fn not_implemented_error(feature: &str, reason: &str) -> TenflowersError {
    TenflowersError::NotImplemented {
        feature: feature.to_string(),
        reason: reason.to_string(),
    }
}

/// Helper function to create generic error
pub fn generic_error(message: &str) -> TenflowersError {
    TenflowersError::Generic {
        message: message.to_string(),
    }
}

/// Convert anyhow::Error to TenflowersError
pub fn from_anyhow_error(err: anyhow::Error) -> TenflowersError {
    TenflowersError::Generic {
        message: format!("{:#}", err),
    }
}

/// Convert tenflowers_core error to TenflowersError
pub fn from_core_error(err: impl std::fmt::Display) -> TenflowersError {
    TenflowersError::Generic {
        message: err.to_string(),
    }
}

/// Register custom exceptions with Python module
pub fn register_exceptions(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ShapeError", py.get_type::<ShapeError>())?;
    m.add("DeviceError", py.get_type::<DeviceError>())?;
    m.add("GradientError", py.get_type::<GradientError>())?;
    m.add("NumericalError", py.get_type::<NumericalError>())?;
    m.add("MemoryError", py.get_type::<MemoryError>())?;
    m.add("TensorOpError", py.get_type::<TensorOpError>())?;
    m.add("LayerConfigError", py.get_type::<LayerConfigError>())?;
    m.add("OptimizerError", py.get_type::<OptimizerError>())?;
    m.add("SerializationError", py.get_type::<SerializationError>())?;
    m.add("DataLoadError", py.get_type::<DataLoadError>())?;
    m.add("GraphCompileError", py.get_type::<GraphCompileError>())?;
    m.add("CheckpointError", py.get_type::<CheckpointError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_error() {
        let err = shape_mismatch_error("matmul", vec![3, 4], vec![4, 5]);
        let message = err.to_string();
        assert!(message.contains("Shape mismatch"));
        assert!(message.contains("matmul"));
        assert!(message.contains("[3, 4]"));
        assert!(message.contains("[4, 5]"));
    }

    #[test]
    fn test_invalid_dimension_error() {
        let err = invalid_dimension_error("sum", -3, 2);
        let message = err.to_string();
        assert!(message.contains("Invalid dimension"));
        assert!(message.contains("-3"));
        assert!(message.contains("2-D"));
    }

    #[test]
    fn test_device_placement_error() {
        let err = device_placement_error("CPU", "GPU", "GPU not available");
        let message = err.to_string();
        assert!(message.contains("Device placement error"));
        assert!(message.contains("CPU"));
        assert!(message.contains("GPU"));
        assert!(message.contains("not available"));
    }

    #[test]
    fn test_gradient_computation_error() {
        let err = gradient_computation_error(Some("loss".to_string()), "no backward graph");
        let message = err.to_string();
        assert!(message.contains("Gradient computation failed"));
        assert!(message.contains("loss"));
        assert!(message.contains("no backward graph"));
    }

    #[test]
    fn test_numerical_instability_error() {
        let err = numerical_instability_error("log", "input contains zeros");
        let message = err.to_string();
        assert!(message.contains("Numerical instability"));
        assert!(message.contains("log"));
        assert!(message.contains("zeros"));
    }

    #[test]
    fn test_memory_allocation_error() {
        let err = memory_allocation_error(1024 * 1024 * 1024, Some(512 * 1024 * 1024));
        let message = err.to_string();
        assert!(message.contains("Memory allocation failed"));
        assert!(message.contains("1073741824"));
        assert!(message.contains("536870912"));
    }

    #[test]
    fn test_layer_configuration_error() {
        let err = layer_configuration_error("Dense", "units", "must be positive");
        let message = err.to_string();
        assert!(message.contains("Layer configuration error"));
        assert!(message.contains("Dense"));
        assert!(message.contains("units"));
        assert!(message.contains("positive"));
    }

    #[test]
    fn test_optimizer_step_error() {
        let err = optimizer_step_error("Adam", "no parameters to optimize");
        let message = err.to_string();
        assert!(message.contains("Optimizer step failed"));
        assert!(message.contains("Adam"));
        assert!(message.contains("no parameters"));
    }
}
