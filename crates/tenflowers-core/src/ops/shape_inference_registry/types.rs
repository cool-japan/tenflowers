//! Shape inference registry types: metadata values, operation categories.

use crate::{Result, Shape, TensorError};
use std::collections::HashMap;

/// Metadata for an operation (e.g., axis, keepdims, transpose flags)
pub type OperationMetadata = HashMap<String, MetadataValue>;

/// Value types for operation metadata
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Bool(bool),
    Int(i64),
    UInt(usize),
    IntVec(Vec<i64>),
    UIntVec(Vec<usize>),
    String(String),
}

impl MetadataValue {
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_uint(&self) -> Option<usize> {
        if let Self::UInt(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_int_vec(&self) -> Option<&Vec<i64>> {
        if let Self::IntVec(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_uint_vec(&self) -> Option<&Vec<usize>> {
        if let Self::UIntVec(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// Shape inference function signature
pub type ShapeInferenceFn = fn(&[Shape], &OperationMetadata) -> Result<Shape>;

/// Operation category for organizing inference rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Element-wise binary operations (add, sub, mul, div)
    BinaryElementwise,
    /// Element-wise unary operations (abs, exp, log, sin, cos)
    UnaryElementwise,
    /// Matrix operations (matmul, dot, outer)
    MatrixOps,
    /// Reduction operations (sum, mean, max, min)
    Reduction,
    /// Manipulation operations (reshape, transpose, permute)
    Manipulation,
    /// Convolution operations
    Convolution,
    /// Pooling operations
    Pooling,
    /// Concatenation and stacking
    Concatenation,
    /// Padding operations
    Padding,
    /// Indexing and slicing
    Indexing,
    /// Comparison operations
    Comparison,
    /// Logical operations
    Logical,
    /// Other operations
    Other,
}

/// Registered operation with shape inference rules
pub(super) struct RegisteredOperation {
    pub(super) name: String,
    pub(super) category: OperationCategory,
    pub(super) inference_fn: ShapeInferenceFn,
    pub(super) description: String,
}
