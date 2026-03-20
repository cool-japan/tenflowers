//! Configuration structs for kernel fusion.

pub use super::kernel_types::SimdInstructionSet;

/// Memory access pattern optimization
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessPattern {
    /// Sequential access (cache-friendly)
    Sequential,
    /// Strided access with known pattern
    Strided { stride: usize },
    /// Random access (cache-unfriendly)
    Random,
    /// Tiled access for blocked algorithms
    Tiled { tile_size: (usize, usize) },
    /// Coalesced access for GPU optimization
    Coalesced { alignment: usize },
}

/// Advanced memory layout strategies
#[derive(Debug, Clone, Copy)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    TiledOptimal,
    AdaptiveCoalesced,
    UltraVectorized,
}

/// Precision requirements for ultra-sophisticated computations
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    Float16,
    Float32,
    Float64,
    Mixed,
    Adaptive,
}

/// SIMD vectorization configuration
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Vector width (e.g., 4, 8, 16)
    pub vector_width: usize,
    /// Enable auto-vectorization
    pub enable_vectorization: bool,
    /// Target SIMD instruction set
    pub instruction_set: SimdInstructionSet,
    /// Alignment requirements
    pub alignment: usize,
}

/// Ultra-sophisticated fusion constraints
#[derive(Debug, Clone)]
pub struct FusionConstraints {
    pub max_shared_memory_kb: u32,
    pub max_registers_per_thread: u32,
    pub max_workgroup_size: (u32, u32, u32),
    pub min_occupancy_percentage: f32,
    pub required_precision: Precision,
}

/// Multi precision modes for hardware acceleration
#[derive(Debug, Clone, PartialEq)]
pub enum MultiPrecisionMode {
    /// FP16 input, FP32 accumulator
    Fp16Fp32,
    /// BF16 input, FP32 accumulator
    Bf16Fp32,
    /// INT8 input, INT32 accumulator
    Int8Int32,
    /// FP8 input, FP16 accumulator (Hopper)
    Fp8Fp16,
    /// Dynamic precision selection
    Dynamic,
}

/// Hardware optimization configuration
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Multi precision mode (FP16, BF16, INT8, etc.)
    pub precision_mode: MultiPrecisionMode,
    /// Matrix tile sizes for Tensor Cores
    pub tile_size: (usize, usize, usize),
    /// Enable Tensor Core specific optimizations
    pub enable_optimizations: bool,
    /// Accumulator precision override
    pub accumulator_precision: Option<String>,
}
