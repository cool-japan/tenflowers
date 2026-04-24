//! Error mapping module for TenfloweRS FFI
//!
//! This module provides comprehensive error mapping from Rust errors to Python exceptions,
//! implementing a unified error taxonomy for better error handling across the FFI boundary.
//!
//! ## FFI Error Mapping Overview
//!
//! Every `tenflowers_core::TensorError` variant is structurally mapped to a `TenflowersError`
//! variant, then to a Python exception. The full table is in
//! `docs/FFI_ERROR_MAPPING.md` (crate-local) and reproduced here as a quick reference:
//!
//! | TensorError Variant     | TenflowersError Variant | Python Exception      |
//! |-------------------------|-------------------------|-----------------------|
//! | ShapeMismatch           | ShapeMismatch           | ShapeError (ValueError) |
//! | InvalidShape            | ShapeMismatch           | ShapeError (ValueError) |
//! | InvalidAxis             | ShapeMismatch           | ShapeError (ValueError) |
//! | DeviceMismatch          | DevicePlacement         | DeviceError (RuntimeError) |
//! | UnsupportedDevice       | DevicePlacement         | DeviceError (RuntimeError) |
//! | GpuError                | DevicePlacement         | DeviceError (RuntimeError) |
//! | DeviceError             | DevicePlacement         | DeviceError (RuntimeError) |
//! | GradientNotEnabled      | GradientComputation     | GradientError (RuntimeError) |
//! | NumericalError          | NumericalInstability    | NumericalError (RuntimeError) |
//! | AllocationError         | MemoryAllocation        | MemoryError (Exception) |
//! | ResourceExhausted       | MemoryAllocation        | MemoryError (Exception) |
//! | InvalidArgument         | InvalidOperation        | TensorOpError (RuntimeError) |
//! | UnsupportedOperation    | InvalidOperation        | TensorOpError (RuntimeError) |
//! | ComputeError            | InvalidOperation        | TensorOpError (RuntimeError) |
//! | BlasError               | InvalidOperation        | TensorOpError (RuntimeError) |
//! | InvalidOperation        | InvalidOperation        | TensorOpError (RuntimeError) |
//! | SerializationError      | Serialization           | SerializationError (RuntimeError) |
//! | IoError                 | DataLoad                | DataLoadError (RuntimeError) |
//! | NotImplemented          | NotImplemented          | PyNotImplementedError |
//! | BenchmarkError          | Generic                 | PyRuntimeError |
//! | Timeout                 | Generic                 | PyRuntimeError |
//! | CacheError              | Generic                 | PyRuntimeError |
//! | Other                   | Generic                 | PyRuntimeError |

use pyo3::exceptions::{
    PyException, PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use std::fmt;
use tenflowers_core::TensorError;

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

/// Unified error type for TenfloweRS FFI operations.
///
/// All variants carry `operation` (the name of the failing op) and structured
/// detail fields so that any caller can format a useful Python exception message.
#[derive(Debug, Clone)]
pub enum TenflowersError {
    /// Shape mismatch in tensor operations
    ShapeMismatch {
        operation: String,
        expected: String,
        got: String,
    },
    /// Invalid dimension or axis specification (mapped into ShapeMismatch category)
    InvalidDimension {
        dim: i32,
        ndim: usize,
        operation: String,
    },
    /// Device placement or transfer error
    DevicePlacement { operation: String, details: String },
    /// Gradient computation error
    GradientComputation { operation: String, details: String },
    /// Numerical instability detected
    NumericalInstability { operation: String, details: String },
    /// Memory allocation failure
    MemoryAllocation { operation: String, details: String },
    /// Invalid tensor operation
    InvalidOperation { operation: String, details: String },
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
    Serialization { operation: String, details: String },
    /// Deserialization error
    Deserialization { operation: String, details: String },
    /// Data loading error
    DataLoad { operation: String, details: String },
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
    NotImplemented { operation: String, details: String },
    /// Generic error with message
    Generic { operation: String, details: String },
}

impl fmt::Display for TenflowersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TenflowersError::ShapeMismatch {
                operation,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Shape mismatch in '{}': expected {}, got {}",
                    operation, expected, got
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
            TenflowersError::DevicePlacement { operation, details } => {
                write!(f, "Device error in '{}': {}", operation, details)
            }
            TenflowersError::GradientComputation { operation, details } => {
                write!(
                    f,
                    "Gradient computation failed in '{}': {}",
                    operation, details
                )
            }
            TenflowersError::NumericalInstability { operation, details } => {
                write!(
                    f,
                    "Numerical instability detected in '{}': {}",
                    operation, details
                )
            }
            TenflowersError::MemoryAllocation { operation, details } => {
                write!(
                    f,
                    "Memory allocation failed in '{}': {}",
                    operation, details
                )
            }
            TenflowersError::InvalidOperation { operation, details } => {
                write!(f, "Invalid operation '{}': {}", operation, details)
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
            TenflowersError::Serialization { operation, details } => {
                write!(f, "Serialization error in '{}': {}", operation, details)
            }
            TenflowersError::Deserialization { operation, details } => {
                write!(f, "Deserialization error in '{}': {}", operation, details)
            }
            TenflowersError::DataLoad { operation, details } => {
                write!(f, "Data loading error in '{}': {}", operation, details)
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
            TenflowersError::NotImplemented { operation, details } => {
                write!(f, "Not implemented '{}': {}", operation, details)
            }
            TenflowersError::Generic { operation, details } => {
                write!(f, "Error in '{}': {}", operation, details)
            }
        }
    }
}

impl std::error::Error for TenflowersError {}

/// Convert TenflowersError to appropriate Python exception.
///
/// Uses custom exception subclasses where available to allow fine-grained
/// Python-side `except` clauses.
impl From<TenflowersError> for PyErr {
    fn from(err: TenflowersError) -> PyErr {
        err.into_py_err()
    }
}

impl TenflowersError {
    /// Map a `tenflowers_core::TensorError` to a `TenflowersError`, preserving all
    /// structured information from the original error instead of stringifying.
    ///
    /// This is the primary entry point for converting core errors at the FFI boundary.
    /// The match is non-exhaustive-free: if `TensorError` gains a new variant, this
    /// function will fail to compile, which is the desired behaviour.
    pub fn from_core_error(err: TensorError) -> TenflowersError {
        match err {
            TensorError::ShapeMismatch {
                operation,
                expected,
                got,
                ..
            } => TenflowersError::ShapeMismatch {
                operation,
                expected,
                got,
            },

            TensorError::InvalidShape {
                operation,
                reason,
                shape,
                ..
            } => {
                let shape_str = shape
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_else(|| "unknown".to_string());
                TenflowersError::ShapeMismatch {
                    operation,
                    expected: "valid shape".to_string(),
                    got: format!("{}: {}", shape_str, reason),
                }
            }

            TensorError::InvalidAxis {
                operation,
                axis,
                ndim,
                ..
            } => TenflowersError::ShapeMismatch {
                operation,
                expected: format!("axis < {}", ndim),
                got: axis.to_string(),
            },

            TensorError::DeviceMismatch {
                operation,
                device1,
                device2,
                ..
            } => TenflowersError::DevicePlacement {
                operation,
                details: format!("{} vs {}", device1, device2),
            },

            TensorError::UnsupportedDevice {
                operation, device, ..
            } => TenflowersError::DevicePlacement {
                operation,
                details: format!("unsupported: {}", device),
            },

            #[cfg(feature = "gpu")]
            TensorError::GpuError {
                operation, details, ..
            } => TenflowersError::DevicePlacement { operation, details },

            TensorError::DeviceError {
                operation, details, ..
            } => TenflowersError::DevicePlacement { operation, details },

            TensorError::GradientNotEnabled { operation, .. } => {
                TenflowersError::GradientComputation {
                    operation,
                    details: "gradient not enabled".to_string(),
                }
            }

            TensorError::NumericalError {
                operation, details, ..
            } => TenflowersError::NumericalInstability { operation, details },

            TensorError::AllocationError {
                operation, details, ..
            } => TenflowersError::MemoryAllocation { operation, details },

            TensorError::ResourceExhausted {
                operation,
                resource,
                ..
            } => TenflowersError::MemoryAllocation {
                operation,
                details: resource,
            },

            TensorError::InvalidArgument {
                operation, reason, ..
            } => TenflowersError::InvalidOperation {
                operation,
                details: reason,
            },

            TensorError::UnsupportedOperation {
                operation, reason, ..
            } => TenflowersError::InvalidOperation {
                operation,
                details: reason,
            },

            TensorError::ComputeError {
                operation, details, ..
            } => TenflowersError::InvalidOperation { operation, details },

            #[cfg(feature = "blas")]
            TensorError::BlasError {
                operation, details, ..
            } => TenflowersError::InvalidOperation { operation, details },

            TensorError::InvalidOperation {
                operation, reason, ..
            } => TenflowersError::InvalidOperation {
                operation,
                details: reason,
            },

            TensorError::SerializationError {
                operation, details, ..
            } => TenflowersError::Serialization { operation, details },

            TensorError::IoError {
                operation, details, ..
            } => TenflowersError::DataLoad { operation, details },

            TensorError::NotImplemented {
                operation, details, ..
            } => TenflowersError::NotImplemented { operation, details },

            TensorError::BenchmarkError {
                operation, details, ..
            } => TenflowersError::Generic { operation, details },

            TensorError::Timeout {
                operation,
                duration_ms,
                ..
            } => TenflowersError::Generic {
                operation,
                details: format!("timeout after {}ms", duration_ms),
            },

            TensorError::CacheError {
                operation, details, ..
            } => TenflowersError::Generic { operation, details },

            TensorError::Other {
                operation, details, ..
            } => TenflowersError::Generic { operation, details },
        }
    }

    /// Convert this `TenflowersError` into a `PyErr` using the most specific
    /// exception class available for the variant.
    pub fn into_py_err(self) -> PyErr {
        let msg = self.to_string();
        match self {
            TenflowersError::ShapeMismatch { .. } => ShapeError::new_err(msg),
            TenflowersError::InvalidDimension { .. } => PyIndexError::new_err(msg),
            TenflowersError::DevicePlacement { .. } => DeviceError::new_err(msg),
            TenflowersError::GradientComputation { .. } => GradientError::new_err(msg),
            TenflowersError::NumericalInstability { .. } => NumericalError::new_err(msg),
            TenflowersError::MemoryAllocation { .. } => MemoryError::new_err(msg),
            TenflowersError::InvalidOperation { .. } => TensorOpError::new_err(msg),
            TenflowersError::TypeConversion { .. } => PyTypeError::new_err(msg),
            TenflowersError::LayerConfiguration { .. } => LayerConfigError::new_err(msg),
            TenflowersError::OptimizerStep { .. } => OptimizerError::new_err(msg),
            TenflowersError::IndexOutOfBounds { .. } => PyIndexError::new_err(msg),
            TenflowersError::Serialization { .. } | TenflowersError::Deserialization { .. } => {
                SerializationError::new_err(msg)
            }
            TenflowersError::DataLoad { .. } => DataLoadError::new_err(msg),
            TenflowersError::GraphCompile { .. } => GraphCompileError::new_err(msg),
            TenflowersError::Checkpoint { .. } => CheckpointError::new_err(msg),
            TenflowersError::DtypeMismatch { .. } => ShapeError::new_err(msg),
            TenflowersError::NotImplemented { .. } => PyNotImplementedError::new_err(msg),
            TenflowersError::Generic { .. } => PyRuntimeError::new_err(msg),
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Helper constructors (lower-level ergonomic API for FFI callers)
// ───────────────────────────────────────────────────────────────────

/// Helper function to create shape mismatch error
pub fn shape_mismatch_error(operation: &str, expected: &str, got: &str) -> TenflowersError {
    TenflowersError::ShapeMismatch {
        operation: operation.to_string(),
        expected: expected.to_string(),
        got: got.to_string(),
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
pub fn device_placement_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::DevicePlacement {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create gradient computation error
pub fn gradient_computation_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::GradientComputation {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create numerical instability error
pub fn numerical_instability_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::NumericalInstability {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create memory allocation error
pub fn memory_allocation_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::MemoryAllocation {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create invalid operation error
pub fn invalid_operation_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::InvalidOperation {
        operation: operation.to_string(),
        details: details.to_string(),
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
pub fn serialization_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::Serialization {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create deserialization error
pub fn deserialization_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::Deserialization {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create data load error
pub fn data_load_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::DataLoad {
        operation: operation.to_string(),
        details: details.to_string(),
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
pub fn not_implemented_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::NotImplemented {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Helper function to create generic error
pub fn generic_error(operation: &str, details: &str) -> TenflowersError {
    TenflowersError::Generic {
        operation: operation.to_string(),
        details: details.to_string(),
    }
}

/// Convert anyhow::Error to TenflowersError
pub fn from_anyhow_error(err: anyhow::Error) -> TenflowersError {
    TenflowersError::Generic {
        operation: "unknown".to_string(),
        details: format!("{:#}", err),
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
    use tenflowers_core::TensorError;

    // ────────────────────────────────────────────────────────────────
    // from_core_error variant coverage tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn shape_mismatch_maps_correctly() {
        let err = TensorError::shape_mismatch("matmul", "[2,3]", "[3,2]");
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::ShapeMismatch { operation, .. } if operation == "matmul"),
            "expected ShapeMismatch for matmul, got: {}",
            mapped
        );
        let msg = mapped.to_string();
        assert!(
            msg.contains("matmul"),
            "message should contain operation name"
        );
        assert!(
            msg.contains("[2,3]"),
            "message should contain expected shape"
        );
    }

    #[test]
    fn invalid_shape_maps_to_shape_mismatch() {
        let err = TensorError::invalid_shape("reshape", "[2,3]", "[4]");
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(mapped, TenflowersError::ShapeMismatch { .. }),
            "InvalidShape should map to ShapeMismatch"
        );
    }

    #[test]
    fn invalid_axis_maps_to_shape_mismatch() {
        let err = TensorError::InvalidAxis {
            operation: "sum".to_string(),
            axis: 5,
            ndim: 2,
            context: None,
        };
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::ShapeMismatch { expected, .. } if expected.contains("2")),
            "InvalidAxis should map to ShapeMismatch with ndim info"
        );
        let msg = mapped.to_string();
        assert!(msg.contains("5"), "message should contain axis value");
    }

    #[test]
    fn device_mismatch_maps_to_device_placement() {
        let err = TensorError::device_mismatch("add", "CPU", "GPU");
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::DevicePlacement { details, .. } if details.contains("CPU")),
            "DeviceMismatch should map to DevicePlacement with device info"
        );
    }

    #[test]
    fn unsupported_device_maps_to_device_placement() {
        let err = TensorError::unsupported_device("conv2d", "FPGA", false);
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::DevicePlacement { details, .. } if details.contains("FPGA")),
            "UnsupportedDevice should map to DevicePlacement"
        );
    }

    #[test]
    fn device_error_maps_to_device_placement() {
        let err = TensorError::DeviceError {
            operation: "transfer".to_string(),
            details: "driver failure".to_string(),
            device: "GPU:0".to_string(),
            context: None,
        };
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::DevicePlacement { details, .. } if details.contains("driver failure")),
            "DeviceError should map to DevicePlacement"
        );
    }

    #[test]
    fn gradient_not_enabled_maps_to_gradient_computation() {
        let err = TensorError::GradientNotEnabled {
            operation: "backward".to_string(),
            suggestion: "use requires_grad=True".to_string(),
            context: None,
        };
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::GradientComputation { details, .. } if details.contains("gradient not enabled")),
            "GradientNotEnabled should map to GradientComputation"
        );
    }

    #[test]
    fn numerical_error_maps_to_numerical_instability() {
        let err = TensorError::numerical_error("log", "input contains NaN", vec![]);
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::NumericalInstability { details, .. } if details.contains("NaN")),
            "NumericalError should map to NumericalInstability"
        );
    }

    #[test]
    fn allocation_error_maps_to_memory_allocation() {
        let err = TensorError::allocation_error("zeros", "out of memory", Some(1024 * 1024), None);
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::MemoryAllocation { details, .. } if details.contains("out of memory")),
            "AllocationError should map to MemoryAllocation"
        );
    }

    #[test]
    fn resource_exhausted_maps_to_memory_allocation() {
        let err = TensorError::resource_exhausted_simple("VRAM".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::MemoryAllocation { details, .. } if details.contains("VRAM")),
            "ResourceExhausted should map to MemoryAllocation"
        );
    }

    #[test]
    fn invalid_argument_maps_to_invalid_operation() {
        let err = TensorError::invalid_argument_op("slice", "step must be non-zero");
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::InvalidOperation { details, .. } if details.contains("non-zero")),
            "InvalidArgument should map to InvalidOperation"
        );
    }

    #[test]
    fn unsupported_operation_maps_to_invalid_operation() {
        let err =
            TensorError::unsupported_operation_simple("complex div not supported".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(mapped, TenflowersError::InvalidOperation { .. }),
            "UnsupportedOperation should map to InvalidOperation"
        );
    }

    #[test]
    fn compute_error_maps_to_invalid_operation() {
        let err = TensorError::compute_error_simple("numerical overflow".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(mapped, TenflowersError::InvalidOperation { .. }),
            "ComputeError should map to InvalidOperation"
        );
    }

    #[test]
    fn serialization_error_maps_correctly() {
        let err = TensorError::serialization_error_simple("corrupt header".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::Serialization { details, .. } if details.contains("corrupt header")),
            "SerializationError should map to Serialization"
        );
    }

    #[test]
    fn io_error_maps_to_data_load() {
        let err = TensorError::io_error_simple("file not found".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::DataLoad { details, .. } if details.contains("file not found")),
            "IoError should map to DataLoad"
        );
    }

    #[test]
    fn not_implemented_maps_correctly() {
        let err =
            TensorError::not_implemented_simple("complex matmul not yet implemented".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::NotImplemented { details, .. } if details.contains("not yet implemented")),
            "NotImplemented should map to NotImplemented"
        );
    }

    #[test]
    fn invalid_operation_maps_correctly() {
        let err = TensorError::invalid_operation_simple("division by zero".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::InvalidOperation { details, .. } if details.contains("division by zero")),
            "InvalidOperation should map to InvalidOperation"
        );
    }

    #[test]
    fn benchmark_error_maps_to_generic() {
        let err = TensorError::benchmark_error_simple("timer failed".to_string());
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(mapped, TenflowersError::Generic { .. }),
            "BenchmarkError should map to Generic"
        );
    }

    #[test]
    fn timeout_maps_to_generic_with_duration() {
        let err = TensorError::timeout_simple(5000);
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::Generic { details, .. } if details.contains("5000ms")),
            "Timeout should map to Generic with duration info"
        );
    }

    #[test]
    fn cache_error_maps_to_generic() {
        let err = TensorError::CacheError {
            operation: "cache_lookup".to_string(),
            details: "eviction policy failed".to_string(),
            recoverable: true,
            context: None,
        };
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(mapped, TenflowersError::Generic { .. }),
            "CacheError should map to Generic"
        );
    }

    #[test]
    fn other_maps_to_generic() {
        let err = TensorError::other_op("unknown_op", "some unexpected failure");
        let mapped = TenflowersError::from_core_error(err);
        assert!(
            matches!(&mapped, TenflowersError::Generic { details, .. } if details.contains("unexpected failure")),
            "Other should map to Generic"
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Display / message format tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn shape_mismatch_display_contains_expected_and_got() {
        let err = shape_mismatch_error("matmul", "[3, 4]", "[4, 5]");
        let msg = err.to_string();
        assert!(msg.contains("Shape mismatch"));
        assert!(msg.contains("matmul"));
        assert!(msg.contains("[3, 4]"));
        assert!(msg.contains("[4, 5]"));
    }

    #[test]
    fn invalid_dimension_display() {
        let err = invalid_dimension_error("sum", -3, 2);
        let msg = err.to_string();
        assert!(msg.contains("Invalid dimension"));
        assert!(msg.contains("-3"));
        assert!(msg.contains("2-D"));
    }

    #[test]
    fn device_placement_display() {
        let err = device_placement_error("transfer", "CPU vs GPU");
        let msg = err.to_string();
        assert!(msg.contains("Device error"));
        assert!(msg.contains("CPU vs GPU"));
    }

    #[test]
    fn gradient_computation_display() {
        let err = gradient_computation_error("backward", "no backward graph");
        let msg = err.to_string();
        assert!(msg.contains("Gradient computation failed"));
        assert!(msg.contains("no backward graph"));
    }

    #[test]
    fn memory_allocation_display() {
        let err = memory_allocation_error("zeros", "out of host memory");
        let msg = err.to_string();
        assert!(msg.contains("Memory allocation failed"));
        assert!(msg.contains("out of host memory"));
    }

    #[test]
    fn layer_configuration_display() {
        let err = layer_configuration_error("Dense", "units", "must be positive");
        let msg = err.to_string();
        assert!(msg.contains("Layer configuration error"));
        assert!(msg.contains("Dense"));
        assert!(msg.contains("units"));
        assert!(msg.contains("positive"));
    }

    #[test]
    fn optimizer_step_display() {
        let err = optimizer_step_error("Adam", "no parameters to optimize");
        let msg = err.to_string();
        assert!(msg.contains("Optimizer step failed"));
        assert!(msg.contains("Adam"));
        assert!(msg.contains("no parameters"));
    }
}
