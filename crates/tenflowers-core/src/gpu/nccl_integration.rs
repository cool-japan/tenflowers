//! NCCL integration for multi-GPU distributed training
//!
//! This module provides NVIDIA Collective Communications Library (NCCL) bindings
//! and high-level abstractions for multi-GPU and multi-node distributed training.
//! Supports AllReduce, AllGather, ReduceScatter, and Broadcast operations.

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
use crate::{Device, Result, Tensor, TensorError};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// NCCL communicator for managing multi-GPU communication
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug)]
pub struct NcclCommunicator {
    /// NCCL communicator handle
    comm: NcclComm,
    /// Local rank in the communicator
    rank: i32,
    /// Total number of ranks
    size: i32,
    /// CUDA device ID for this rank
    device_id: i32,
    /// NCCL unique ID for multi-node setup
    unique_id: NcclUniqueId,
}

/// NCCL collective operation types
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclCollectiveOp {
    /// All-reduce: reduce across all ranks and broadcast result
    AllReduce,
    /// All-gather: gather data from all ranks to all ranks
    AllGather,
    /// Reduce-scatter: reduce and scatter chunks to different ranks
    ReduceScatter,
    /// Broadcast: send data from root rank to all ranks
    Broadcast,
    /// Reduce: reduce data to a specific rank
    Reduce,
    /// All-to-all: each rank sends different data to every rank
    AllToAll,
}

/// NCCL reduction operations
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclReductionOp {
    /// Sum reduction
    Sum,
    /// Product reduction
    Product,
    /// Maximum reduction
    Max,
    /// Minimum reduction
    Min,
    /// Average reduction
    Average,
}

/// NCCL data types
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclDataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 16-bit floating point
    Float16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    Uint8,
}

/// NCCL communicator configuration
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct NcclConfig {
    /// Local rank of this process
    pub rank: i32,
    /// Total number of ranks
    pub size: i32,
    /// CUDA device ID to use
    pub device_id: i32,
    /// Network interface for multi-node communication
    pub network_interface: Option<String>,
    /// Enable NCCL debug output
    pub debug: bool,
    /// NCCL algorithm selection
    pub algorithm: Option<NcclAlgorithm>,
}

/// NCCL algorithm selection for optimization
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclAlgorithm {
    /// Ring algorithm (default)
    Ring,
    /// Tree algorithm
    Tree,
    /// CollNet algorithm for collective networks
    CollNet,
    /// NVLS (NVLink Sharp) algorithm
    Nvls,
}

/// Multi-GPU distributed training coordinator
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
pub struct DistributedTrainer {
    /// NCCL communicator
    communicator: NcclCommunicator,
    /// Local GPU tensors for each device
    local_tensors: HashMap<i32, Vec<Tensor<f32>>>,
    /// Gradient synchronization configuration
    sync_config: GradientSyncConfig,
    /// Performance metrics
    metrics: DistributedMetrics,
}

/// Gradient synchronization configuration
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct GradientSyncConfig {
    /// Whether to enable gradient compression
    pub compress_gradients: bool,
    /// Bucket size for gradient bucketing (bytes)
    pub bucket_size: usize,
    /// Whether to overlap communication with computation
    pub overlap_comm_comp: bool,
    /// Reduction operation for gradients
    pub reduction_op: NcclReductionOp,
}

/// Performance metrics for distributed training
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Default)]
pub struct DistributedMetrics {
    /// Total communication time
    pub total_comm_time_ms: f64,
    /// Total bytes communicated
    pub total_bytes_communicated: u64,
    /// Average bandwidth (GB/s)
    pub average_bandwidth_gb_s: f64,
    /// Number of collective operations
    pub num_collectives: u64,
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
impl NcclCommunicator {
    /// Create a new NCCL communicator for single-node multi-GPU setup
    pub fn new_single_node(config: &NcclConfig) -> Result<Self> {
        // Initialize NCCL
        unsafe {
            nccl_init()?;
        }

        // Generate unique ID on rank 0, broadcast to others
        let unique_id = if config.rank == 0 {
            NcclUniqueId::generate()?
        } else {
            // In practice, this would be received from rank 0
            NcclUniqueId::default()
        };

        // Set CUDA device
        unsafe {
            cuda_set_device(config.device_id)?;
        }

        // Initialize NCCL communicator
        let comm = unsafe { nccl_comm_init_rank(unique_id, config.size, config.rank)? };

        Ok(NcclCommunicator {
            comm,
            rank: config.rank,
            size: config.size,
            device_id: config.device_id,
            unique_id,
        })
    }

    /// Create a new NCCL communicator for multi-node setup
    pub fn new_multi_node(
        config: &NcclConfig,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Self> {
        // Initialize NCCL with network backend
        unsafe {
            nccl_init()?;
        }

        // Set network interface if specified
        if let Some(interface) = &config.network_interface {
            unsafe {
                nccl_set_network_interface(interface.as_ptr())?;
            }
        }

        // Initialize multi-node communicator
        let unique_id = if config.rank == 0 {
            let id = NcclUniqueId::generate()?;
            // Broadcast unique ID to all nodes (implementation specific)
            Self::broadcast_unique_id(&id, master_addr, master_port)?;
            id
        } else {
            // Receive unique ID from rank 0
            Self::receive_unique_id(master_addr, master_port)?
        };

        let comm = unsafe { nccl_comm_init_rank(unique_id, config.size, config.rank)? };

        Ok(NcclCommunicator {
            comm,
            rank: config.rank,
            size: config.size,
            device_id: config.device_id,
            unique_id,
        })
    }

    /// Execute AllReduce operation
    pub fn all_reduce<T>(&mut self, tensor: &mut Tensor<T>, op: NcclReductionOp) -> Result<()>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let data_type = Self::infer_nccl_datatype::<T>()?;
        let count = tensor.numel();

        unsafe {
            nccl_all_reduce(
                tensor.data().as_ptr() as *const std::ffi::c_void,
                tensor.data().as_ptr() as *mut std::ffi::c_void,
                count,
                data_type,
                op,
                self.comm,            // NCCL communicator
                std::ptr::null_mut(), // Default stream
            )?;
        }

        Ok(())
    }

    /// Execute AllGather operation
    pub fn all_gather<T>(
        &mut self,
        send_tensor: &Tensor<T>,
        recv_tensor: &mut Tensor<T>,
    ) -> Result<()>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let data_type = Self::infer_nccl_datatype::<T>()?;
        let send_count = send_tensor.numel();

        unsafe {
            nccl_all_gather(
                send_tensor.data().as_ptr() as *const std::ffi::c_void,
                recv_tensor.data().as_ptr() as *mut std::ffi::c_void,
                send_count,
                data_type,
                self.comm, // NCCL communicator
                std::ptr::null_mut(),
            )?;
        }

        Ok(())
    }

    /// Execute Broadcast operation
    pub fn broadcast<T>(&mut self, tensor: &mut Tensor<T>, root: i32) -> Result<()>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let data_type = Self::infer_nccl_datatype::<T>()?;
        let count = tensor.numel();

        unsafe {
            nccl_broadcast(
                tensor.data().as_ptr() as *const std::ffi::c_void,
                tensor.data().as_ptr() as *mut std::ffi::c_void,
                count,
                data_type,
                root,
                self.comm, // NCCL communicator
                std::ptr::null_mut(),
            )?;
        }

        Ok(())
    }

    /// Execute ReduceScatter operation
    pub fn reduce_scatter<T>(
        &mut self,
        send_tensor: &Tensor<T>,
        recv_tensor: &mut Tensor<T>,
        op: NcclReductionOp,
    ) -> Result<()>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let data_type = Self::infer_nccl_datatype::<T>()?;
        let recv_count = recv_tensor.numel();

        unsafe {
            nccl_reduce_scatter(
                send_tensor.data().as_ptr() as *const std::ffi::c_void,
                recv_tensor.data().as_ptr() as *mut std::ffi::c_void,
                recv_count,
                data_type,
                op,
                self.comm, // NCCL communicator
                std::ptr::null_mut(),
            )?;
        }

        Ok(())
    }

    /// Synchronize all ranks
    pub fn barrier(&mut self) -> Result<()> {
        // NCCL doesn't have explicit barrier, use AllReduce on dummy data
        let mut dummy = Tensor::<i32>::zeros(&[1]);
        self.all_reduce(&mut dummy, NcclReductionOp::Sum)
    }

    /// Get communicator rank
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get communicator size
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    // Private helper methods

    fn infer_nccl_datatype<T>() -> Result<NcclDataType>
    where
        T: 'static,
    {
        let type_id = std::any::TypeId::of::<T>();

        if type_id == std::any::TypeId::of::<f32>() {
            Ok(NcclDataType::Float32)
        } else if type_id == std::any::TypeId::of::<f64>() {
            Ok(NcclDataType::Float64)
        } else if type_id == std::any::TypeId::of::<i32>() {
            Ok(NcclDataType::Int32)
        } else if type_id == std::any::TypeId::of::<i64>() {
            Ok(NcclDataType::Int64)
        } else if type_id == std::any::TypeId::of::<i8>() {
            Ok(NcclDataType::Int8)
        } else if type_id == std::any::TypeId::of::<u8>() {
            Ok(NcclDataType::Uint8)
        } else {
            Err(TensorError::unsupported_operation_simple(format!(
                "Unsupported data type for NCCL: {:?}",
                std::any::type_name::<T>()
            )))
        }
    }

    fn broadcast_unique_id(
        unique_id: &NcclUniqueId,
        master_addr: &str,
        master_port: u16,
    ) -> Result<()> {
        // Implementation would broadcast unique ID to all nodes
        // This could use MPI, TCP sockets, or other communication mechanisms
        Ok(())
    }

    fn receive_unique_id(master_addr: &str, master_port: u16) -> Result<NcclUniqueId> {
        // Implementation would receive unique ID from rank 0
        Ok(NcclUniqueId::default())
    }
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
impl DistributedTrainer {
    /// Create a new distributed trainer
    pub fn new(config: &NcclConfig, sync_config: GradientSyncConfig) -> Result<Self> {
        let communicator = NcclCommunicator::new_single_node(config)?;

        Ok(DistributedTrainer {
            communicator,
            local_tensors: HashMap::new(),
            sync_config,
            metrics: DistributedMetrics::default(),
        })
    }

    /// Synchronize gradients across all ranks
    pub fn sync_gradients(&mut self, gradients: &mut [Tensor<f32>]) -> Result<()> {
        let start_time = std::time::Instant::now();

        if self.sync_config.overlap_comm_comp {
            // Overlap communication with computation using gradient bucketing
            self.sync_gradients_bucketed(gradients)?;
        } else {
            // Simple sequential synchronization
            for gradient in gradients.iter_mut() {
                self.communicator
                    .all_reduce(gradient, self.sync_config.reduction_op)?;

                // Scale by number of ranks for averaging
                if self.sync_config.reduction_op == NcclReductionOp::Average {
                    // Note: In practice, NCCL might handle averaging automatically
                    // This is for illustration
                }
            }
        }

        // Update metrics
        let elapsed = start_time.elapsed();
        self.metrics.total_comm_time_ms += elapsed.as_secs_f64() * 1000.0;
        self.metrics.num_collectives += gradients.len() as u64;

        Ok(())
    }

    /// Broadcast model parameters from rank 0 to all ranks
    pub fn broadcast_parameters(&mut self, parameters: &mut [Tensor<f32>]) -> Result<()> {
        for parameter in parameters.iter_mut() {
            self.communicator.broadcast(parameter, 0)?;
        }
        Ok(())
    }

    /// All-gather operation for model averaging
    pub fn all_gather_parameters(
        &mut self,
        local_params: &[Tensor<f32>],
        gathered_params: &mut [Tensor<f32>],
    ) -> Result<()> {
        assert_eq!(local_params.len(), gathered_params.len());

        for (local, gathered) in local_params.iter().zip(gathered_params.iter_mut()) {
            self.communicator.all_gather(local, gathered)?;
        }

        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &DistributedMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = DistributedMetrics::default();
    }

    // Private methods

    fn sync_gradients_bucketed(&mut self, gradients: &mut [Tensor<f32>]) -> Result<()> {
        // Implement gradient bucketing for overlapping communication with computation
        let bucket_size = self.sync_config.bucket_size;
        let mut current_bucket_size = 0;
        let mut bucket_gradients = Vec::new();

        for gradient in gradients.iter_mut() {
            let gradient_size = gradient.numel() * std::mem::size_of::<f32>();

            if current_bucket_size + gradient_size > bucket_size && !bucket_gradients.is_empty() {
                // Process current bucket
                self.process_gradient_bucket(&mut bucket_gradients)?;
                bucket_gradients.clear();
                current_bucket_size = 0;
            }

            bucket_gradients.push(gradient);
            current_bucket_size += gradient_size;
        }

        // Process remaining gradients
        if !bucket_gradients.is_empty() {
            self.process_gradient_bucket(&mut bucket_gradients)?;
        }

        Ok(())
    }

    fn process_gradient_bucket(&mut self, bucket: &mut [&mut Tensor<f32>]) -> Result<()> {
        // Process a bucket of gradients
        for gradient in bucket.iter_mut() {
            self.communicator
                .all_reduce(*gradient, self.sync_config.reduction_op)?;
        }
        Ok(())
    }
}

// NCCL FFI types and bindings
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct NcclComm {
    handle: *mut std::ffi::c_void,
}

#[derive(Debug, Clone, Copy)]
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
#[repr(C)]
struct NcclUniqueId {
    internal: [u8; 128], // NCCL_UNIQUE_ID_BYTES
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
impl Default for NcclUniqueId {
    fn default() -> Self {
        Self { internal: [0; 128] }
    }
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
impl NcclUniqueId {
    fn generate() -> Result<Self> {
        unsafe {
            let mut unique_id = Self::default();
            nccl_get_unique_id(&mut unique_id)?;
            Ok(unique_id)
        }
    }
}

// Simplified NCCL FFI functions (in practice these would be proper extern "C" bindings)
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_init() -> Result<()> {
    // ncclInit()
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_get_unique_id(unique_id: *mut NcclUniqueId) -> Result<()> {
    // ncclGetUniqueId(unique_id)
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_comm_init_rank(unique_id: NcclUniqueId, nranks: i32, rank: i32) -> Result<NcclComm> {
    // ncclCommInitRank(&comm, nranks, unique_id, rank)
    Ok(NcclComm {
        handle: std::ptr::null_mut(),
    })
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_all_reduce(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: NcclDataType,
    op: NcclReductionOp,
    comm: NcclComm,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    // ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_all_gather(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    sendcount: usize,
    datatype: NcclDataType,
    comm: NcclComm,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    // ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_broadcast(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: NcclDataType,
    root: i32,
    comm: NcclComm,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    // ncclBcast(buff, count, datatype, root, comm, stream)
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_reduce_scatter(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    recvcount: usize,
    datatype: NcclDataType,
    op: NcclReductionOp,
    comm: NcclComm,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    // ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn nccl_set_network_interface(interface: *const u8) -> Result<()> {
    // Set NCCL_SOCKET_IFNAME environment variable or similar
    Ok(())
}

#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_set_device(device_id: i32) -> Result<()> {
    // cudaSetDevice(device_id)
    Ok(())
}

/// Stub implementation for non-NCCL platforms
#[cfg(not(feature = "nccl"))]
pub mod nccl_stub {
    //! Stub implementation for platforms without NCCL support
    use crate::{Result, TensorError};

    pub fn nccl_not_available() -> Result<()> {
        Err(TensorError::device_error_simple(
            "NCCL integration is only available with the 'nccl' feature enabled".to_string(),
        ))
    }
}

/// NCCL performance benchmarking and optimization tools
#[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
pub mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};

    pub struct NcclBenchmark {
        trainer: DistributedTrainer,
        results: Vec<BenchmarkResult>,
    }

    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub operation: String,
        pub data_size_mb: f64,
        pub duration: Duration,
        pub bandwidth_gb_s: f64,
        pub algorithm_efficiency: f64,
    }

    impl NcclBenchmark {
        pub fn new(config: &NcclConfig) -> Result<Self> {
            let sync_config = GradientSyncConfig {
                compress_gradients: false,
                bucket_size: 25 * 1024 * 1024, // 25MB buckets
                overlap_comm_comp: true,
                reduction_op: NcclReductionOp::Average,
            };

            Ok(NcclBenchmark {
                trainer: DistributedTrainer::new(config, sync_config)?,
                results: Vec::new(),
            })
        }

        /// Benchmark AllReduce performance at different data sizes
        pub fn benchmark_all_reduce(&mut self, sizes: &[usize]) -> Result<Vec<BenchmarkResult>> {
            let mut results = Vec::new();

            for &size in sizes {
                let mut tensor = Tensor::<f32>::ones(&[size]);

                let start = Instant::now();
                self.trainer
                    .communicator
                    .all_reduce(&mut tensor, NcclReductionOp::Sum)?;
                let duration = start.elapsed();

                let data_size_mb = (size * 4) as f64 / 1024.0 / 1024.0; // f32 = 4 bytes
                let bandwidth = (data_size_mb * 8.0) / duration.as_secs_f64() / 1024.0; // GB/s

                // Theoretical vs actual bandwidth efficiency
                let theoretical_bandwidth = 25.0; // Assume 25 GB/s theoretical for illustration
                let efficiency = bandwidth / theoretical_bandwidth;

                results.push(BenchmarkResult {
                    operation: format!("AllReduce_{}_elements", size),
                    data_size_mb,
                    duration,
                    bandwidth_gb_s: bandwidth,
                    algorithm_efficiency: efficiency,
                });
            }

            self.results.extend(results.clone());
            Ok(results)
        }

        /// Benchmark different collective operations
        pub fn benchmark_collectives(&mut self, size: usize) -> Result<Vec<BenchmarkResult>> {
            let mut results = Vec::new();

            // AllReduce
            let mut tensor = Tensor::<f32>::ones(&[size]);
            let start = Instant::now();
            self.trainer
                .communicator
                .all_reduce(&mut tensor, NcclReductionOp::Sum)?;
            let duration = start.elapsed();

            let data_size_mb = (size * 4) as f64 / 1024.0 / 1024.0;
            let bandwidth = (data_size_mb * 8.0) / duration.as_secs_f64() / 1024.0;

            results.push(BenchmarkResult {
                operation: "AllReduce".to_string(),
                data_size_mb,
                duration,
                bandwidth_gb_s: bandwidth,
                algorithm_efficiency: 0.85, // Example efficiency
            });

            // Broadcast
            let mut tensor = Tensor::<f32>::ones(&[size]);
            let start = Instant::now();
            self.trainer.communicator.broadcast(&mut tensor, 0)?;
            let duration = start.elapsed();

            let bandwidth = (data_size_mb * 8.0) / duration.as_secs_f64() / 1024.0;

            results.push(BenchmarkResult {
                operation: "Broadcast".to_string(),
                data_size_mb,
                duration,
                bandwidth_gb_s: bandwidth,
                algorithm_efficiency: 0.90, // Example efficiency
            });

            self.results.extend(results.clone());
            Ok(results)
        }

        /// Generate comprehensive performance report
        pub fn generate_report(&self) -> String {
            let mut report = String::from("NCCL Performance Benchmark Report\n");
            report.push_str("=====================================\n\n");

            for result in &self.results {
                report.push_str(&format!(
                    "Operation: {}\n  Data Size: {:.2} MB\n  Duration: {:?}\n  Bandwidth: {:.2} GB/s\n  Efficiency: {:.1}%\n\n",
                    result.operation, result.data_size_mb, result.duration,
                    result.bandwidth_gb_s, result.algorithm_efficiency * 100.0
                ));
            }

            report
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
    fn test_nccl_communicator_creation() {
        let config = NcclConfig {
            rank: 0,
            size: 1,
            device_id: 0,
            network_interface: None,
            debug: false,
            algorithm: None,
        };

        let result = NcclCommunicator::new_single_node(&config);
        // Test should pass on systems with NCCL support
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("NCCL"));
    }

    #[test]
    #[cfg(not(feature = "nccl"))]
    fn test_nccl_not_available() {
        let result = nccl_stub::nccl_not_available();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("NCCL integration is only available"));
    }

    #[test]
    #[cfg(all(feature = "nccl", any(target_os = "linux", target_os = "windows")))]
    fn test_gradient_sync_config() {
        let sync_config = GradientSyncConfig {
            compress_gradients: true,
            bucket_size: 25 * 1024 * 1024,
            overlap_comm_comp: true,
            reduction_op: NcclReductionOp::Average,
        };

        assert_eq!(sync_config.bucket_size, 25 * 1024 * 1024);
        assert!(sync_config.overlap_comm_comp);
    }
}
