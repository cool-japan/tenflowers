//! Operation Registry
//!
//! Provides a centralized operation registry with ultra-performance optimizations,
//! including caching, versioning, SIMD acceleration, and batch processing.
//!
//! # Modules
//! - `types`: Core type definitions (OpDef, Kernel trait, OpRegistry struct, etc.)
//! - `core`: OpRegistry method implementations, analytics, global OP_REGISTRY
//! - `builtin`: Registration of standard built-in operations
//! - `ultra_extensions`: HFT-optimized and quantum-inspired registry variants
//! - `tests`: Unit tests for the registry

pub mod builtin;
pub mod core;
pub mod tests;
pub mod types;
pub mod ultra_extensions;

pub use core::{RegistryAnalytics, UltraKernel, OP_REGISTRY};
pub use types::{
    ArgDef, AttrDef, AttrType, AttrValue, BatchOperation, Kernel, OpDef, OpRegistry, OpVersion,
    ShapeFn,
};
pub use ultra_extensions::{HftRegistry, PrecisionMetrics, QuantumRegistry};
