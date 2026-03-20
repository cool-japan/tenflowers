//! Core type definitions for the operation registry.
//!
//! Contains all the data types, traits, and structural definitions used by the registry,
//! including operation metadata, kernel traits, and attribute types.

use crate::{DType, Device, Result, Shape, TensorError};
use scirs2_core::metrics::{Counter, Histogram, Timer};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

/// Operation version information
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OpVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl OpVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another version
    /// Compatible if major version matches and minor version is >= required
    pub fn is_compatible_with(&self, required: &OpVersion) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl std::fmt::Display for OpVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Default for OpVersion {
    fn default() -> Self {
        Self::new(1, 0, 0)
    }
}

/// Metadata for an operation
#[derive(Clone)]
pub struct OpDef {
    /// Operation name
    pub name: String,
    /// Operation version
    pub version: OpVersion,
    /// Input argument definitions
    pub inputs: Vec<ArgDef>,
    /// Output definitions
    pub outputs: Vec<ArgDef>,
    /// Operation attributes
    pub attrs: HashMap<String, AttrDef>,
    /// Shape inference function
    pub shape_fn: Option<ShapeFn>,
    /// Gradient function name (if differentiable)
    pub grad_fn: Option<String>,
    /// Documentation
    pub doc: String,
    /// Deprecated flag - marks if this version is deprecated
    pub deprecated: bool,
    /// If deprecated, message explaining deprecation
    pub deprecation_message: Option<String>,
}

/// Argument definition
#[derive(Debug, Clone)]
pub struct ArgDef {
    pub name: String,
    pub dtype: Option<DType>,
    pub shape: Option<Shape>,
    pub doc: String,
}

/// Attribute definition
#[derive(Debug, Clone)]
pub struct AttrDef {
    pub name: String,
    pub attr_type: AttrType,
    pub default: Option<AttrValue>,
    pub doc: String,
}

/// Attribute types
#[derive(Debug, Clone, PartialEq)]
pub enum AttrType {
    Int,
    Float,
    Bool,
    String,
    Shape,
    DType,
    IntList,
    FloatList,
}

/// Attribute values
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Shape(Shape),
    DType(DType),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
}

/// Shape inference function type
pub type ShapeFn =
    Arc<dyn Fn(&[&Shape], &HashMap<String, AttrValue>) -> Result<Vec<Shape>> + Send + Sync>;

/// Kernel implementation trait
pub trait Kernel: Send + Sync {
    /// Execute the kernel
    fn compute(
        &self,
        inputs: &[&dyn Any],
        attrs: &HashMap<String, AttrValue>,
    ) -> Result<Vec<Box<dyn Any>>>;

    /// Get supported device
    fn device(&self) -> Device;

    /// Get supported data type
    fn dtype(&self) -> DType;
}

/// Operation registry key (name + version)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct OpKey {
    pub(super) name: String,
    pub(super) version: OpVersion,
}

/// Kernel registry key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct KernelKey {
    pub(super) op: String,
    pub(super) version: OpVersion,
    pub(super) device: Device,
    pub(super) dtype: DType,
}

/// Ultra-performance registry metrics
pub(super) struct RegistryMetrics {
    /// Operation lookup counter
    pub(super) op_lookups: Counter,
    /// Kernel execution counter
    pub(super) kernel_executions: Counter,
    /// Cache hit ratio histogram
    pub(super) cache_hit_ratio: Histogram,
    /// Operation execution time
    pub(super) execution_timer: Timer,
    /// Batch processing metrics
    pub(super) batch_operations: Counter,
    /// SIMD acceleration usage
    pub(super) simd_accelerated_ops: Counter,
}

/// Batch operation for high-throughput processing
#[derive(Debug, Clone)]
pub struct BatchOperation {
    pub(super) op_name: String,
    #[allow(dead_code)]
    pub(super) inputs: Vec<String>, // Simplified for now
    #[allow(dead_code)]
    pub(super) attrs: HashMap<String, AttrValue>,
    pub(super) priority: u8,
    pub(super) estimated_cost: f64,
}

/// Ultra-performance kernel scheduler with predictive optimization
pub(super) struct UltraKernelScheduler {
    /// Execution history for performance prediction
    #[allow(dead_code)]
    pub(super) execution_history: HashMap<String, Vec<f64>>,
    /// Resource utilization tracking
    #[allow(dead_code)]
    pub(super) cpu_utilization: AtomicU64,
    #[allow(dead_code)]
    pub(super) gpu_utilization: AtomicU64,
    /// Adaptive batch size optimization
    #[allow(dead_code)]
    pub(super) optimal_batch_sizes: HashMap<String, usize>,
    /// Hot operation tracking
    pub(super) hot_operations: HashMap<String, AtomicU64>,
}

/// Global operation registry with ultra-performance optimizations
pub struct OpRegistry {
    pub(super) ops: RwLock<HashMap<OpKey, OpDef>>,
    pub(super) kernels: RwLock<HashMap<KernelKey, Arc<dyn Kernel>>>,
    /// Track latest version for each operation name
    pub(super) latest_versions: RwLock<HashMap<String, OpVersion>>,
    /// Ultra-fast lookup cache for frequently accessed operations
    pub(super) op_cache: RwLock<HashMap<String, Arc<OpDef>>>,
    /// Ultra-fast kernel cache with SIMD-optimized lookup
    pub(super) kernel_cache: RwLock<HashMap<String, Arc<dyn Kernel>>>,
    /// Performance metrics and analytics
    pub(super) metrics: RegistryMetrics,
    /// Batch operation queue for high-throughput processing
    #[allow(dead_code)]
    pub(super) batch_queue: RwLock<Vec<BatchOperation>>,
    /// Ultra-performance kernel scheduler
    pub(super) scheduler: RwLock<UltraKernelScheduler>,
}
