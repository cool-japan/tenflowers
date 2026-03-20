//! Core kernel operation types: FusableOp, SimdInstructionSet, ParallelizationStrategy.

/// Types of fusable operations with latest GPU optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusableOp {
    Add,
    Mul,
    Sub,
    Div,
    ReLU,
    LeakyReLU,
    ELU,
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    SiLU,
    Mish,
    RMSNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    BatchNorm,
    MatMul,
    Conv2D,
    DepthwiseConv2D,
    GroupConv2D,
    ScaledDotProductAttention,
    MultiHeadAttention,
    Transpose,
    Reshape,
    Permute,
    Sum,
    Mean,
    Max,
    Min,
    Softmax,
    LogSoftmax,
    Quantize8,
    Quantize4,
    Dequantize8,
    Dequantize4,
    FP8MatMul,
    FP8Add,
    HardwareMatMul,
    SparseMatMul,
    BlockSparseMatMul,
    WinogradConv,
    FFTConv,
    RMSNormFused,
    GroupNormFused,
    LayerNormFused,
    InPlaceActivation,
    FusedResidual,
    FusedDropout,
    FlashAttention,
    ChunkedAttention,
    SparseAttention,
    WarpReduceSum,
    BlockReduceMax,
    TreeReduce,
    QuantizedMatMul,
    MultiPrecisionOp,
    TiledOperation,
    VectorizedOp,
    AsyncMemoryOp,
}

/// SIMD instruction set targets
#[derive(Debug, Clone, PartialEq)]
pub enum SimdInstructionSet {
    /// AVX-512 for high-end CPUs
    Avx512,
    /// AVX2 for modern CPUs
    Avx2,
    /// SSE4 for older CPUs
    Sse4,
    /// ARM NEON for ARM processors
    Neon,
    /// GPU wavefront/warp operations
    GpuWavefront,
}

/// Parallelization strategy for multi-GPU/multi-core
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelizationStrategy {
    /// No parallelization
    None,
    /// Data parallel across devices
    DataParallel { num_devices: usize },
    /// Model parallel with pipeline
    ModelParallel { pipeline_stages: usize },
    /// Hybrid data + model parallelism
    Hybrid {
        data_groups: usize,
        model_stages: usize,
    },
    /// Dynamic load balancing
    DynamicLoadBalancing,
}
