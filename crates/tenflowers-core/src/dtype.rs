#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum DType {
    Float16,  // Half precision IEEE 754-2008
    BFloat16, // Brain floating point (Google's bfloat16)
    Float32,
    Float64,
    Int32,
    Int64,
    Int16,
    Int8,
    Int4,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    Bool,
    Complex32,
    Complex64,
    String, // Variable-length string (for structured arrays)
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Int16 => 2,
            DType::Int8 => 1,
            DType::Int4 => 1, // Packed, but minimum addressable unit is 1 byte
            DType::UInt64 => 8,
            DType::UInt32 => 4,
            DType::UInt16 => 2,
            DType::UInt8 => 1,
            DType::Bool => 1,
            DType::Complex32 => 8,  // 2 * f32
            DType::Complex64 => 16, // 2 * f64
            DType::String => 8,     // Pointer size for variable-length string
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Int16 => "int16",
            DType::Int8 => "int8",
            DType::Int4 => "int4",
            DType::UInt64 => "uint64",
            DType::UInt32 => "uint32",
            DType::UInt16 => "uint16",
            DType::UInt8 => "uint8",
            DType::Bool => "bool",
            DType::Complex32 => "complex32",
            DType::Complex64 => "complex64",
            DType::String => "string",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Get DType from Rust type
pub fn dtype_from_type<T: 'static>() -> DType {
    use crate::complex::{Complex32, Complex64};
    use half::{bf16, f16};

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>() {
        DType::Float16
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bf16>() {
        DType::BFloat16
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        DType::Float32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        DType::Float64
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        DType::Int32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        DType::Int64
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
        DType::Int16
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        DType::Int8
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        DType::UInt64
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        DType::UInt32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>() {
        DType::UInt16
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
        DType::UInt8
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
        DType::Bool
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
        DType::Complex32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        DType::Complex64
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<String>() {
        DType::String
    } else {
        // Default to Float32 for unknown types
        DType::Float32
    }
}
