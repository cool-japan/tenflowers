//! Data type abstraction module for TenfloweRS FFI
//!
//! This module provides dtype abstraction for supporting various numerical types
//! including f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64.
//!
//! Currently, the core tensor library supports f32 operations. This module provides
//! the infrastructure for future expansion to support additional data types.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::fmt;

/// Supported data types for tensors
#[pyclass(name = "DType")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyDType {
    /// 32-bit floating point (IEEE 754)
    Float32,
    /// 64-bit floating point (IEEE 754)
    Float64,
    /// 16-bit floating point (IEEE 754-2008)
    Float16,
    /// 16-bit brain floating point
    BFloat16,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// Boolean type
    Bool,
}

impl fmt::Display for PyDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyDType::Float32 => write!(f, "float32"),
            PyDType::Float64 => write!(f, "float64"),
            PyDType::Float16 => write!(f, "float16"),
            PyDType::BFloat16 => write!(f, "bfloat16"),
            PyDType::Int8 => write!(f, "int8"),
            PyDType::Int16 => write!(f, "int16"),
            PyDType::Int32 => write!(f, "int32"),
            PyDType::Int64 => write!(f, "int64"),
            PyDType::UInt8 => write!(f, "uint8"),
            PyDType::UInt16 => write!(f, "uint16"),
            PyDType::UInt32 => write!(f, "uint32"),
            PyDType::UInt64 => write!(f, "uint64"),
            PyDType::Bool => write!(f, "bool"),
        }
    }
}

#[pymethods]
impl PyDType {
    /// Create a DType from a string representation
    ///
    /// # Arguments
    ///
    /// * `dtype_str` - String representation of the dtype (e.g., "float32", "int64")
    ///
    /// # Returns
    ///
    /// DType instance
    #[staticmethod]
    pub fn from_string(dtype_str: &str) -> PyResult<Self> {
        match dtype_str.to_lowercase().as_str() {
            "float32" | "f32" => Ok(PyDType::Float32),
            "float64" | "f64" | "double" => Ok(PyDType::Float64),
            "float16" | "f16" | "half" => Ok(PyDType::Float16),
            "bfloat16" | "bf16" => Ok(PyDType::BFloat16),
            "int8" | "i8" => Ok(PyDType::Int8),
            "int16" | "i16" => Ok(PyDType::Int16),
            "int32" | "i32" => Ok(PyDType::Int32),
            "int64" | "i64" | "long" => Ok(PyDType::Int64),
            "uint8" | "u8" | "byte" => Ok(PyDType::UInt8),
            "uint16" | "u16" => Ok(PyDType::UInt16),
            "uint32" | "u32" => Ok(PyDType::UInt32),
            "uint64" | "u64" => Ok(PyDType::UInt64),
            "bool" | "boolean" => Ok(PyDType::Bool),
            _ => Err(PyValueError::new_err(format!(
                "Unknown dtype string: {}",
                dtype_str
            ))),
        }
    }

    /// Get the size in bytes of this dtype
    ///
    /// # Returns
    ///
    /// Size in bytes
    pub fn itemsize(&self) -> usize {
        match self {
            PyDType::Float32 => 4,
            PyDType::Float64 => 8,
            PyDType::Float16 => 2,
            PyDType::BFloat16 => 2,
            PyDType::Int8 => 1,
            PyDType::Int16 => 2,
            PyDType::Int32 => 4,
            PyDType::Int64 => 8,
            PyDType::UInt8 => 1,
            PyDType::UInt16 => 2,
            PyDType::UInt32 => 4,
            PyDType::UInt64 => 8,
            PyDType::Bool => 1,
        }
    }

    /// Check if this is a floating point dtype
    ///
    /// # Returns
    ///
    /// True if floating point, False otherwise
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            PyDType::Float32 | PyDType::Float64 | PyDType::Float16 | PyDType::BFloat16
        )
    }

    /// Check if this is an integer dtype
    ///
    /// # Returns
    ///
    /// True if integer, False otherwise
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            PyDType::Int8
                | PyDType::Int16
                | PyDType::Int32
                | PyDType::Int64
                | PyDType::UInt8
                | PyDType::UInt16
                | PyDType::UInt32
                | PyDType::UInt64
        )
    }

    /// Check if this is a signed dtype
    ///
    /// # Returns
    ///
    /// True if signed, False otherwise
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            PyDType::Float32
                | PyDType::Float64
                | PyDType::Float16
                | PyDType::BFloat16
                | PyDType::Int8
                | PyDType::Int16
                | PyDType::Int32
                | PyDType::Int64
        )
    }

    /// Check if this is an unsigned dtype
    ///
    /// # Returns
    ///
    /// True if unsigned, False otherwise
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            PyDType::UInt8 | PyDType::UInt16 | PyDType::UInt32 | PyDType::UInt64
        )
    }

    /// Get the name of this dtype
    ///
    /// # Returns
    ///
    /// String representation of the dtype
    pub fn name(&self) -> String {
        self.to_string()
    }

    /// Check if this dtype is currently supported by the backend
    ///
    /// # Returns
    ///
    /// True if supported, False otherwise
    pub fn is_supported(&self) -> bool {
        // Currently, only f32 is fully supported
        // This will be expanded in the future
        matches!(self, PyDType::Float32)
    }

    /// Get the corresponding numpy dtype string
    ///
    /// # Returns
    ///
    /// Numpy dtype string (e.g., "float32", "int64")
    pub fn numpy_dtype(&self) -> String {
        match self {
            PyDType::Float32 => "float32".to_string(),
            PyDType::Float64 => "float64".to_string(),
            PyDType::Float16 => "float16".to_string(),
            PyDType::BFloat16 => "bfloat16".to_string(),
            PyDType::Int8 => "int8".to_string(),
            PyDType::Int16 => "int16".to_string(),
            PyDType::Int32 => "int32".to_string(),
            PyDType::Int64 => "int64".to_string(),
            PyDType::UInt8 => "uint8".to_string(),
            PyDType::UInt16 => "uint16".to_string(),
            PyDType::UInt32 => "uint32".to_string(),
            PyDType::UInt64 => "uint64".to_string(),
            PyDType::Bool => "bool".to_string(),
        }
    }

    /// Check dtype compatibility for operations
    ///
    /// # Arguments
    ///
    /// * `other` - Another dtype to check compatibility with
    ///
    /// # Returns
    ///
    /// True if dtypes are compatible for operations, False otherwise
    pub fn is_compatible_with(&self, other: &PyDType) -> bool {
        // Same dtype is always compatible
        if self == other {
            return true;
        }

        // Floating point types can be mixed with appropriate casting
        if self.is_floating_point() && other.is_floating_point() {
            return true;
        }

        // Integer types can be mixed with appropriate casting
        if self.is_integer() && other.is_integer() {
            return true;
        }

        false
    }

    /// Get the result dtype for binary operations between two dtypes
    ///
    /// # Arguments
    ///
    /// * `other` - Another dtype for the binary operation
    ///
    /// # Returns
    ///
    /// Result dtype after promotion
    pub fn result_type(&self, other: &PyDType) -> PyResult<PyDType> {
        // Type promotion rules similar to NumPy
        if self == other {
            return Ok(*self);
        }

        // Floating point promotion
        if self.is_floating_point() && other.is_floating_point() {
            let self_size = self.itemsize();
            let other_size = other.itemsize();

            // Special case for bfloat16 - promote to float32
            if matches!(self, PyDType::BFloat16) || matches!(other, PyDType::BFloat16) {
                return Ok(PyDType::Float32);
            }

            return if self_size >= other_size {
                Ok(*self)
            } else {
                Ok(*other)
            };
        }

        // Integer promotion
        if self.is_integer() && other.is_integer() {
            let self_size = self.itemsize();
            let other_size = other.itemsize();

            // If one is signed and the other is unsigned, promote to signed with larger size
            if self.is_signed() != other.is_signed() {
                let max_size = self_size.max(other_size);
                return match max_size {
                    1 => Ok(PyDType::Int8),
                    2 => Ok(PyDType::Int16),
                    4 => Ok(PyDType::Int32),
                    8 => Ok(PyDType::Int64),
                    _ => Err(PyTypeError::new_err("Invalid dtype size")),
                };
            }

            // Same signedness - promote to larger size
            return if self_size >= other_size {
                Ok(*self)
            } else {
                Ok(*other)
            };
        }

        // Float + Integer -> Float
        if self.is_floating_point() && other.is_integer() {
            return Ok(*self);
        }
        if self.is_integer() && other.is_floating_point() {
            return Ok(*other);
        }

        Err(PyTypeError::new_err(format!(
            "Cannot determine result type for {} and {}",
            self, other
        )))
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __repr__(&self) -> String {
        format!("DType.{}", self)
    }

    fn __hash__(&self) -> u64 {
        *self as u64
    }

    fn __richcmp__(&self, other: &PyDType, op: pyo3::pyclass::CompareOp) -> PyResult<bool> {
        match op {
            pyo3::pyclass::CompareOp::Eq => Ok(self == other),
            pyo3::pyclass::CompareOp::Ne => Ok(self != other),
            _ => Err(PyTypeError::new_err(
                "DType only supports == and != comparisons",
            )),
        }
    }
}

/// DType constants module for convenience
pub mod dtypes {
    use super::PyDType;

    pub const FLOAT32: PyDType = PyDType::Float32;
    pub const FLOAT64: PyDType = PyDType::Float64;
    pub const FLOAT16: PyDType = PyDType::Float16;
    pub const BFLOAT16: PyDType = PyDType::BFloat16;
    pub const INT8: PyDType = PyDType::Int8;
    pub const INT16: PyDType = PyDType::Int16;
    pub const INT32: PyDType = PyDType::Int32;
    pub const INT64: PyDType = PyDType::Int64;
    pub const UINT8: PyDType = PyDType::UInt8;
    pub const UINT16: PyDType = PyDType::UInt16;
    pub const UINT32: PyDType = PyDType::UInt32;
    pub const UINT64: PyDType = PyDType::UInt64;
    pub const BOOL: PyDType = PyDType::Bool;
}

/// Check if a dtype conversion is safe (no precision loss)
pub fn is_safe_cast(from: &PyDType, to: &PyDType) -> bool {
    if from == to {
        return true;
    }

    match (from, to) {
        // Float upcasts
        (PyDType::Float16, PyDType::Float32 | PyDType::Float64) => true,
        (PyDType::BFloat16, PyDType::Float32 | PyDType::Float64) => true,
        (PyDType::Float32, PyDType::Float64) => true,

        // Integer upcasts (same sign)
        (PyDType::Int8, PyDType::Int16 | PyDType::Int32 | PyDType::Int64) => true,
        (PyDType::Int16, PyDType::Int32 | PyDType::Int64) => true,
        (PyDType::Int32, PyDType::Int64) => true,
        (PyDType::UInt8, PyDType::UInt16 | PyDType::UInt32 | PyDType::UInt64) => true,
        (PyDType::UInt16, PyDType::UInt32 | PyDType::UInt64) => true,
        (PyDType::UInt32, PyDType::UInt64) => true,

        // Unsigned to signed (if target is larger)
        (PyDType::UInt8, PyDType::Int16 | PyDType::Int32 | PyDType::Int64) => true,
        (PyDType::UInt16, PyDType::Int32 | PyDType::Int64) => true,
        (PyDType::UInt32, PyDType::Int64) => true,

        _ => false,
    }
}

/// Python function to check if a cast is safe
#[pyfunction]
pub fn is_safe_cast_py(from: &PyDType, to: &PyDType) -> bool {
    is_safe_cast(from, to)
}

/// Python function to get the result dtype for binary operations
#[pyfunction]
pub fn result_type(dtype1: &PyDType, dtype2: &PyDType) -> PyResult<PyDType> {
    dtype1.result_type(dtype2)
}

/// Python function to promote multiple dtypes to a common dtype
#[pyfunction]
pub fn promote_types(dtypes: Vec<PyDType>) -> PyResult<PyDType> {
    if dtypes.is_empty() {
        return Err(PyValueError::new_err("Cannot promote empty dtype list"));
    }

    let mut result = dtypes[0];
    for dtype in &dtypes[1..] {
        result = result.result_type(dtype)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(PyDType::Float32.itemsize(), 4);
        assert_eq!(PyDType::Float64.itemsize(), 8);
        assert_eq!(PyDType::Float16.itemsize(), 2);
        assert_eq!(PyDType::BFloat16.itemsize(), 2);
        assert_eq!(PyDType::Int32.itemsize(), 4);
        assert_eq!(PyDType::UInt8.itemsize(), 1);
    }

    #[test]
    fn test_dtype_properties() {
        assert!(PyDType::Float32.is_floating_point());
        assert!(!PyDType::Int32.is_floating_point());
        assert!(PyDType::Int32.is_integer());
        assert!(!PyDType::Float32.is_integer());
        assert!(PyDType::Int32.is_signed());
        assert!(!PyDType::UInt32.is_signed());
        assert!(PyDType::UInt32.is_unsigned());
        assert!(!PyDType::Int32.is_unsigned());
    }

    #[test]
    fn test_safe_cast() {
        assert!(is_safe_cast(&PyDType::Float32, &PyDType::Float64));
        assert!(!is_safe_cast(&PyDType::Float64, &PyDType::Float32));
        assert!(is_safe_cast(&PyDType::Int32, &PyDType::Int64));
        assert!(!is_safe_cast(&PyDType::Int64, &PyDType::Int32));
        assert!(is_safe_cast(&PyDType::UInt8, &PyDType::Int16));
    }

    #[test]
    fn test_result_type() {
        let result = PyDType::Float32.result_type(&PyDType::Float64).unwrap();
        assert_eq!(result, PyDType::Float64);

        let result = PyDType::Int32.result_type(&PyDType::Int64).unwrap();
        assert_eq!(result, PyDType::Int64);

        let result = PyDType::Float32.result_type(&PyDType::Int32).unwrap();
        assert_eq!(result, PyDType::Float32);
    }
}
