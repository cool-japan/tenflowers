//! Dtype Promotion Rules for TenfloweRS
//!
//! This module implements NumPy and PyTorch compatible dtype promotion rules
//! for mixed-precision tensor operations, addressing the TODO item for dtype
//! promotion when multi-dtype support is added.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use tenflowers_core::{DType, TensorError};

/// Python-accessible dtype promotion manager
#[pyclass]
pub struct PyDtypePromoter {
    inner: DtypePromoter,
}

/// Core dtype promotion logic
pub struct DtypePromoter {
    promotion_table: HashMap<(DType, DType), DType>,
    category_hierarchy: Vec<DtypeCategory>,
}

/// Dtype categories for promotion hierarchy
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DtypeCategory {
    Bool,
    UnsignedInteger,
    SignedInteger,
    FloatingPoint,
    Complex,
}

impl DtypeCategory {
    /// Get the category for a given dtype
    pub fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Bool => DtypeCategory::Bool,
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
                DtypeCategory::UnsignedInteger
            }
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 | DType::Int4 => {
                DtypeCategory::SignedInteger
            }
            DType::Float16 | DType::Float32 | DType::Float64 => DtypeCategory::FloatingPoint,
            DType::BFloat16 => DtypeCategory::FloatingPoint,
            DType::Complex32 | DType::Complex64 => DtypeCategory::Complex,
            DType::String => DtypeCategory::Complex, // Fallback - strings are complex
        }
    }
}

impl DtypePromoter {
    pub fn new() -> Self {
        let mut promotion_table = HashMap::new();
        let category_hierarchy = vec![
            DtypeCategory::Bool,
            DtypeCategory::UnsignedInteger,
            DtypeCategory::SignedInteger,
            DtypeCategory::FloatingPoint,
            DtypeCategory::Complex,
        ];

        // Initialize comprehensive promotion table following NumPy rules
        Self::init_promotion_table(&mut promotion_table);

        Self {
            promotion_table,
            category_hierarchy,
        }
    }

    /// Initialize the promotion table with NumPy-compatible rules
    fn init_promotion_table(table: &mut HashMap<(DType, DType), DType>) {
        // Same dtype promotes to itself
        for dtype in [
            DType::Bool,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::Float16,
            DType::Float32,
            DType::Float64,
            DType::BFloat16,
            DType::Complex32,
            DType::Complex64,
        ] {
            table.insert((dtype, dtype), dtype);
        }

        // Bool promotions - bool promotes to any other type
        for dtype in [
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::Float16,
            DType::Float32,
            DType::Float64,
            DType::BFloat16,
            DType::Complex32,
            DType::Complex64,
        ] {
            table.insert((DType::Bool, dtype), dtype);
            table.insert((dtype, DType::Bool), dtype);
        }

        // Integer promotions (unsigned)
        table.insert((DType::UInt8, DType::UInt16), DType::UInt16);
        table.insert((DType::UInt16, DType::UInt8), DType::UInt16);
        table.insert((DType::UInt8, DType::UInt32), DType::UInt32);
        table.insert((DType::UInt32, DType::UInt8), DType::UInt32);
        table.insert((DType::UInt8, DType::UInt64), DType::UInt64);
        table.insert((DType::UInt64, DType::UInt8), DType::UInt64);
        table.insert((DType::UInt16, DType::UInt32), DType::UInt32);
        table.insert((DType::UInt32, DType::UInt16), DType::UInt32);
        table.insert((DType::UInt16, DType::UInt64), DType::UInt64);
        table.insert((DType::UInt64, DType::UInt16), DType::UInt64);
        table.insert((DType::UInt32, DType::UInt64), DType::UInt64);
        table.insert((DType::UInt64, DType::UInt32), DType::UInt64);

        // Integer promotions (signed)
        table.insert((DType::Int8, DType::Int16), DType::Int16);
        table.insert((DType::Int16, DType::Int8), DType::Int16);
        table.insert((DType::Int8, DType::Int32), DType::Int32);
        table.insert((DType::Int32, DType::Int8), DType::Int32);
        table.insert((DType::Int8, DType::Int64), DType::Int64);
        table.insert((DType::Int64, DType::Int8), DType::Int64);
        table.insert((DType::Int16, DType::Int32), DType::Int32);
        table.insert((DType::Int32, DType::Int16), DType::Int32);
        table.insert((DType::Int16, DType::Int64), DType::Int64);
        table.insert((DType::Int64, DType::Int16), DType::Int64);
        table.insert((DType::Int32, DType::Int64), DType::Int64);
        table.insert((DType::Int64, DType::Int32), DType::Int64);

        // Mixed signed/unsigned integer promotions
        table.insert((DType::UInt8, DType::Int16), DType::Int16);
        table.insert((DType::Int16, DType::UInt8), DType::Int16);
        table.insert((DType::UInt16, DType::Int32), DType::Int32);
        table.insert((DType::Int32, DType::UInt16), DType::Int32);
        table.insert((DType::UInt32, DType::Int64), DType::Int64);
        table.insert((DType::Int64, DType::UInt32), DType::Int64);

        // Float promotions
        table.insert((DType::Float16, DType::Float32), DType::Float32);
        table.insert((DType::Float32, DType::Float16), DType::Float32);
        table.insert((DType::Float16, DType::Float64), DType::Float64);
        table.insert((DType::Float64, DType::Float16), DType::Float64);
        table.insert((DType::Float32, DType::Float64), DType::Float64);
        table.insert((DType::Float64, DType::Float32), DType::Float64);
        table.insert((DType::BFloat16, DType::Float32), DType::Float32);
        table.insert((DType::Float32, DType::BFloat16), DType::Float32);

        // Integer to float promotions
        for int_dtype in [
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
        ] {
            for float_dtype in [
                DType::Float16,
                DType::Float32,
                DType::Float64,
                DType::BFloat16,
            ] {
                table.insert((int_dtype, float_dtype), float_dtype);
                table.insert((float_dtype, int_dtype), float_dtype);
            }
        }

        // Complex promotions
        table.insert((DType::Complex32, DType::Complex64), DType::Complex64);
        table.insert((DType::Complex64, DType::Complex32), DType::Complex64);

        // Real to complex promotions
        for real_dtype in [
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::Float16,
            DType::Float32,
            DType::Float64,
            DType::BFloat16,
        ] {
            // Promote to appropriate complex type based on precision
            let complex_dtype = match real_dtype {
                DType::Float64 => DType::Complex64,
                _ => DType::Complex32,
            };
            table.insert((real_dtype, complex_dtype), complex_dtype);
            table.insert((complex_dtype, real_dtype), complex_dtype);
        }
    }

    /// Promote two dtypes according to NumPy rules
    #[allow(clippy::result_large_err)]
    pub fn promote(&self, dtype1: DType, dtype2: DType) -> Result<DType, TensorError> {
        // Check direct lookup first
        if let Some(&result) = self.promotion_table.get(&(dtype1, dtype2)) {
            return Ok(result);
        }

        // Fallback to category-based promotion
        self.promote_by_category(dtype1, dtype2)
    }

    /// Promote multiple dtypes to a common type
    #[allow(clippy::result_large_err)]
    pub fn promote_multiple(&self, dtypes: &[DType]) -> Result<DType, TensorError> {
        if dtypes.is_empty() {
            return Err(TensorError::InvalidOperation {
                operation: "promote_multiple".to_string(),
                reason: "Empty dtype list".to_string(),
                context: None,
            });
        }

        let mut result = dtypes[0];
        for &dtype in &dtypes[1..] {
            result = self.promote(result, dtype)?;
        }

        Ok(result)
    }

    /// Promote dtypes based on category hierarchy
    #[allow(clippy::result_large_err)]
    fn promote_by_category(&self, dtype1: DType, dtype2: DType) -> Result<DType, TensorError> {
        let cat1 = DtypeCategory::from_dtype(dtype1);
        let cat2 = DtypeCategory::from_dtype(dtype2);

        // Same category - promote to larger size
        if cat1 == cat2 {
            return Ok(self.promote_within_category(dtype1, dtype2));
        }

        // Different categories - promote to higher category
        let target_category = if cat1 > cat2 { cat1 } else { cat2 };

        match target_category {
            DtypeCategory::Bool => Ok(DType::Bool),
            DtypeCategory::UnsignedInteger => Ok(DType::UInt64),
            DtypeCategory::SignedInteger => Ok(DType::Int64),
            DtypeCategory::FloatingPoint => Ok(DType::Float64),
            DtypeCategory::Complex => Ok(DType::Complex64),
        }
    }

    /// Promote within the same category to larger size
    fn promote_within_category(&self, dtype1: DType, dtype2: DType) -> DType {
        match (dtype1, dtype2) {
            // Both unsigned integers
            (DType::UInt8, DType::UInt16) | (DType::UInt16, DType::UInt8) => DType::UInt16,
            (DType::UInt8, DType::UInt32) | (DType::UInt32, DType::UInt8) => DType::UInt32,
            (DType::UInt8, DType::UInt64) | (DType::UInt64, DType::UInt8) => DType::UInt64,
            (DType::UInt16, DType::UInt32) | (DType::UInt32, DType::UInt16) => DType::UInt32,
            (DType::UInt16, DType::UInt64) | (DType::UInt64, DType::UInt16) => DType::UInt64,
            (DType::UInt32, DType::UInt64) | (DType::UInt64, DType::UInt32) => DType::UInt64,

            // Both signed integers
            (DType::Int8, DType::Int16) | (DType::Int16, DType::Int8) => DType::Int16,
            (DType::Int8, DType::Int32) | (DType::Int32, DType::Int8) => DType::Int32,
            (DType::Int8, DType::Int64) | (DType::Int64, DType::Int8) => DType::Int64,
            (DType::Int16, DType::Int32) | (DType::Int32, DType::Int16) => DType::Int32,
            (DType::Int16, DType::Int64) | (DType::Int64, DType::Int16) => DType::Int64,
            (DType::Int32, DType::Int64) | (DType::Int64, DType::Int32) => DType::Int64,

            // Both floats
            (DType::Float16, DType::Float32) | (DType::Float32, DType::Float16) => DType::Float32,
            (DType::Float16, DType::Float64) | (DType::Float64, DType::Float16) => DType::Float64,
            (DType::Float32, DType::Float64) | (DType::Float64, DType::Float32) => DType::Float64,
            (DType::BFloat16, DType::Float32) | (DType::Float32, DType::BFloat16) => DType::Float32,

            // Both complex
            (DType::Complex32, DType::Complex64) | (DType::Complex64, DType::Complex32) => {
                DType::Complex64
            }

            // Default to the larger dtype if we can't determine
            _ => {
                if self.dtype_size(dtype1) >= self.dtype_size(dtype2) {
                    dtype1
                } else {
                    dtype2
                }
            }
        }
    }

    /// Get approximate size/precision ordering for dtypes
    fn dtype_size(&self, dtype: DType) -> u32 {
        match dtype {
            DType::Bool => 1,
            DType::UInt8 | DType::Int8 => 8,
            DType::Int4 => 4,
            DType::UInt16 | DType::Int16 | DType::Float16 | DType::BFloat16 => 16,
            DType::UInt32 | DType::Int32 | DType::Float32 => 32,
            DType::UInt64 | DType::Int64 | DType::Float64 | DType::Complex32 => 64,
            DType::Complex64 => 128,
            DType::String => 64, // Pointer size
        }
    }

    /// Check if dtype promotion is safe (no precision loss)
    pub fn is_safe_promotion(&self, from: DType, to: DType) -> bool {
        if from == to {
            return true;
        }

        let from_cat = DtypeCategory::from_dtype(from);
        let to_cat = DtypeCategory::from_dtype(to);

        // Promoting to higher category is generally safe
        if to_cat > from_cat {
            return true;
        }

        // Within same category, promoting to larger size is safe
        if from_cat == to_cat {
            return self.dtype_size(to) >= self.dtype_size(from);
        }

        false
    }
}

impl Default for DtypePromoter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PyDtypePromoter {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyDtypePromoter {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: DtypePromoter::new(),
        }
    }

    /// Promote two dtypes to their common type
    pub fn promote(&self, dtype1: &str, dtype2: &str) -> PyResult<String> {
        let dt1 = self.parse_dtype(dtype1)?;
        let dt2 = self.parse_dtype(dtype2)?;

        let result = self
            .inner
            .promote(dt1, dt2)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(self.dtype_to_string(result))
    }

    /// Promote multiple dtypes to their common type
    pub fn promote_multiple(&self, dtypes: Vec<String>) -> PyResult<String> {
        let parsed_dtypes: Result<Vec<DType>, _> =
            dtypes.iter().map(|s| self.parse_dtype(s)).collect();

        let parsed_dtypes = parsed_dtypes?;
        let result = self
            .inner
            .promote_multiple(&parsed_dtypes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(self.dtype_to_string(result))
    }

    /// Check if promotion from one dtype to another is safe
    pub fn is_safe_promotion(&self, from_dtype: &str, to_dtype: &str) -> PyResult<bool> {
        let from = self.parse_dtype(from_dtype)?;
        let to = self.parse_dtype(to_dtype)?;

        Ok(self.inner.is_safe_promotion(from, to))
    }

    /// Get all available dtype promotion rules as a dictionary
    pub fn get_promotion_table(&self, py: Python) -> PyResult<PyObject> {
        let py_dict = PyDict::new(py);

        for (&(dt1, dt2), &result) in &self.inner.promotion_table {
            let key = format!(
                "({}, {})",
                self.dtype_to_string(dt1),
                self.dtype_to_string(dt2)
            );
            py_dict.set_item(key, self.dtype_to_string(result))?;
        }

        Ok(py_dict.into())
    }

    /// Get dtype category hierarchy
    pub fn get_category_hierarchy(&self) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .category_hierarchy
            .iter()
            .map(|cat| format!("{:?}", cat))
            .collect())
    }

    /// Explain promotion decision
    pub fn explain_promotion(&self, dtype1: &str, dtype2: &str) -> PyResult<String> {
        let dt1 = self.parse_dtype(dtype1)?;
        let dt2 = self.parse_dtype(dtype2)?;

        let result = self
            .inner
            .promote(dt1, dt2)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let cat1 = DtypeCategory::from_dtype(dt1);
        let cat2 = DtypeCategory::from_dtype(dt2);

        let explanation = if dt1 == dt2 {
            format!("Same dtype: {} -> {}", dtype1, self.dtype_to_string(result))
        } else if cat1 == cat2 {
            format!(
                "Same category ({:?}): {} + {} -> {} (larger size)",
                cat1,
                dtype1,
                dtype2,
                self.dtype_to_string(result)
            )
        } else {
            format!(
                "Different categories ({:?} + {:?}): {} + {} -> {} (higher category)",
                cat1,
                cat2,
                dtype1,
                dtype2,
                self.dtype_to_string(result)
            )
        };

        Ok(explanation)
    }
}

// Helper implementation methods (not exposed to Python)
impl PyDtypePromoter {
    fn parse_dtype(&self, dtype_str: &str) -> Result<DType, PyErr> {
        match dtype_str.to_lowercase().as_str() {
            "bool" => Ok(DType::Bool),
            "u8" | "uint8" => Ok(DType::UInt8),
            "u16" | "uint16" => Ok(DType::UInt16),
            "u32" | "uint32" => Ok(DType::UInt32),
            "u64" | "uint64" => Ok(DType::UInt64),
            "i8" | "int8" => Ok(DType::Int8),
            "i16" | "int16" => Ok(DType::Int16),
            "i32" | "int32" => Ok(DType::Int32),
            "i64" | "int64" => Ok(DType::Int64),
            "f16" | "float16" | "half" => Ok(DType::Float16),
            "f32" | "float32" | "float" => Ok(DType::Float32),
            "f64" | "float64" | "double" => Ok(DType::Float64),
            "bf16" | "bfloat16" => Ok(DType::BFloat16),
            "c64" | "complex64" => Ok(DType::Complex32),
            "c128" | "complex128" => Ok(DType::Complex64),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown dtype: {}",
                dtype_str
            ))),
        }
    }

    fn dtype_to_string(&self, dtype: DType) -> String {
        match dtype {
            DType::Bool => "bool",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Int8 => "int8",
            DType::Int4 => "int4",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::BFloat16 => "bfloat16",
            DType::Complex32 => "complex64",
            DType::Complex64 => "complex128",
            DType::String => "string",
        }
        .to_string()
    }
}

/// Register dtype promotion functions with Python module
pub fn register_dtype_promotion_functions(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtypePromoter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_dtype_promotion() {
        let promoter = DtypePromoter::new();
        assert_eq!(
            promoter.promote(DType::Float32, DType::Float32).unwrap(),
            DType::Float32
        );
        assert_eq!(
            promoter.promote(DType::Int32, DType::Int32).unwrap(),
            DType::Int32
        );
    }

    #[test]
    fn test_int_to_float_promotion() {
        let promoter = DtypePromoter::new();
        assert_eq!(
            promoter.promote(DType::Int32, DType::Float32).unwrap(),
            DType::Float32
        );
        assert_eq!(
            promoter.promote(DType::Float32, DType::Int32).unwrap(),
            DType::Float32
        );
    }

    #[test]
    fn test_bool_promotion() {
        let promoter = DtypePromoter::new();
        assert_eq!(
            promoter.promote(DType::Bool, DType::Int32).unwrap(),
            DType::Int32
        );
        assert_eq!(
            promoter.promote(DType::Bool, DType::Float64).unwrap(),
            DType::Float64
        );
    }

    #[test]
    fn test_multiple_dtype_promotion() {
        let promoter = DtypePromoter::new();
        let dtypes = vec![DType::UInt8, DType::Int16, DType::Float32];
        assert_eq!(promoter.promote_multiple(&dtypes).unwrap(), DType::Float32);
    }

    #[test]
    fn test_safe_promotion() {
        let promoter = DtypePromoter::new();
        assert!(promoter.is_safe_promotion(DType::Int16, DType::Int32));
        assert!(promoter.is_safe_promotion(DType::Int32, DType::Float64));
        assert!(!promoter.is_safe_promotion(DType::Float64, DType::Int32));
    }
}
