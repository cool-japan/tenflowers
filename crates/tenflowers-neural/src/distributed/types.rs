//! Distributed training types: structs, enums, error types

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tenflowers_core::{Device, Result, Tensor, TensorError};

/// Communication backend types for distributed training
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CommunicationBackend {
    /// NCCL backend for NVIDIA GPUs
    #[cfg(feature = "nccl")]
    Nccl,
    /// Gloo backend for general CPU/GPU communication
    Gloo,
    /// MPI backend for HPC environments
    #[cfg(feature = "mpi")]
    Mpi,
    /// Thread-based backend for single-node multi-GPU
    Thread,
    /// Custom user-defined backend
    Custom(String),
}

/// Communication group for collective operations
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    /// Group identifier
    pub group_id: String,
    /// Rank of this process in the group
    pub rank: usize,
    /// Total number of processes in the group
    pub world_size: usize,
    /// Devices participating in this group
    pub devices: Vec<Device>,
    /// Backend used for communication
    pub backend: CommunicationBackend,
}

/// Collective operation types
#[derive(Debug, Clone)]
pub enum CollectiveOp {
    /// All-reduce operation (sum, average, min, max)
    AllReduce { reduction_op: ReductionOp },
    /// All-gather operation
    AllGather,
    /// Reduce-scatter operation
    ReduceScatter { reduction_op: ReductionOp },
    /// Broadcast from root rank
    Broadcast { root_rank: usize },
    /// Point-to-point send
    Send { dest_rank: usize },
    /// Point-to-point receive
    Recv { src_rank: usize },
}

/// Reduction operations for collective ops
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Average,
    Min,
    Max,
    Product,
}

/// Communication performance metrics
#[derive(Debug, Default)]
pub struct CommunicationMetrics {
    /// Total bytes communicated
    pub total_bytes: u64,
    /// Number of operations performed
    pub operation_count: u64,
    /// Total communication time
    pub total_time: Duration,
    /// Average bandwidth (bytes/second)
    pub avg_bandwidth: f64,
    /// Per-operation metrics
    pub operation_metrics: HashMap<String, OperationMetrics>,
}

/// Metrics for specific operation types
#[derive(Debug, Default, Clone)]
pub struct OperationMetrics {
    pub count: u64,
    pub total_time: Duration,
    pub total_bytes: u64,
    pub avg_latency: Duration,
}

impl Clone for CommunicationMetrics {
    fn clone(&self) -> Self {
        Self {
            total_bytes: self.total_bytes,
            operation_count: self.operation_count,
            total_time: self.total_time,
            avg_bandwidth: self.avg_bandwidth,
            operation_metrics: self.operation_metrics.clone(),
        }
    }
}

/// Trait for communication backend implementations
/// Note: For simplicity, we use f32 tensors for now. This can be extended to support
/// multiple types using enum dispatch or other type erasure techniques.
pub trait CommunicationBackendImpl: Send + Sync {
    /// Initialize the backend
    fn initialize(&mut self, config: &BackendConfig) -> Result<()>;

    /// Create a communication group
    fn create_group(&mut self, group: &CommunicationGroup) -> Result<()>;

    /// Perform all-reduce operation with f32 tensors
    fn all_reduce_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>>;

    /// Perform all-gather operation with f32 tensors
    fn all_gather_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>>;

    /// Perform broadcast operation with f32 tensors
    fn broadcast_f32(
        &self,
        tensor: &Tensor<f32>,
        root_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>>;

    /// Send f32 tensor to specific rank
    fn send_f32(
        &self,
        tensor: &Tensor<f32>,
        dest_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<()>;

    /// Receive f32 tensor from specific rank
    fn recv_f32(
        &self,
        shape: &[usize],
        src_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>>;

    /// Finalize the backend
    fn finalize(&mut self) -> Result<()>;

    /// Get backend name
    fn name(&self) -> &str;
}

/// Configuration for communication backends
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend-specific options
    pub options: HashMap<String, String>,
    /// Timeout for operations
    pub timeout: Duration,
    /// Enable compression
    pub compression: bool,
    /// Compression algorithm if enabled
    pub compression_algo: CompressionAlgorithm,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            options: HashMap::new(),
            timeout: Duration::from_secs(30),
            compression: false,
            compression_algo: CompressionAlgorithm::None,
        }
    }
}

/// Compression algorithms for communication
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    None,
    /// Top-k sparsification
    TopK {
        k: usize,
    },
    /// Random sparsification
    Random {
        ratio: f32,
    },
    /// Quantization
    Quantization {
        bits: u8,
    },
    /// Custom compression
    Custom(String),
}

/// Result of collective operations with f32 tensors
#[derive(Debug)]
pub enum CollectiveResult<T> {
    /// Single tensor result
    Tensor(Tensor<T>),
    /// Multiple tensor result (e.g., from all-gather)
    TensorList(Vec<Tensor<T>>),
    /// No result (e.g., from send)
    None,
}

/// Communication runtime for managing distributed operations
pub struct CommunicationRuntime {
    /// Active communication groups
    pub(super) groups: HashMap<String, CommunicationGroup>,
    /// Default communication group
    pub(super) default_group: Option<String>,
    /// Backend implementations
    pub(super) backends: HashMap<CommunicationBackend, Box<dyn CommunicationBackendImpl>>,
    /// Performance metrics
    pub(super) metrics: Arc<Mutex<CommunicationMetrics>>,
}
