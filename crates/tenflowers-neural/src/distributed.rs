#![allow(unexpected_cfgs)]
#![allow(unreachable_patterns)] // GPU/ROCM patterns unreachable when features are disabled

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tenflowers_core::ops::manipulation::slice;
use tenflowers_core::{Device, Result, Tensor, TensorError};

/// Element-wise minimum operation between two tensors
fn element_wise_min<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + scirs2_core::num_traits::Zero,
{
    use tenflowers_core::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
            let result = scirs2_core::ndarray::Zip::from(arr_a)
                .and(arr_b)
                .map_collect(|a_val, b_val| {
                    if a_val <= b_val {
                        a_val.clone()
                    } else {
                        b_val.clone()
                    }
                });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // Use the GPU min operation from tenflowers-core
            tenflowers_core::ops::binary::min(a, b)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::invalid_argument_op(
            "element_wise_min",
            "Tensors must be on the same device",
        )),
        #[cfg(not(feature = "gpu"))]
        _ => unreachable!("GPU variant should not exist without gpu feature"),
    }
}

/// Element-wise maximum operation between two tensors
fn element_wise_max<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + scirs2_core::num_traits::Zero,
{
    use tenflowers_core::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
            let result = scirs2_core::ndarray::Zip::from(arr_a)
                .and(arr_b)
                .map_collect(|a_val, b_val| {
                    if a_val >= b_val {
                        a_val.clone()
                    } else {
                        b_val.clone()
                    }
                });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // Use the GPU max operation from tenflowers-core
            tenflowers_core::ops::binary::max(a, b)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::invalid_argument_op(
            "element_wise_max",
            "Tensors must be on the same device",
        )),
        #[cfg(not(feature = "gpu"))]
        _ => unreachable!("GPU variant should not exist without gpu feature"),
    }
}

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

/// Communication runtime for managing distributed operations
pub struct CommunicationRuntime {
    /// Active communication groups
    groups: HashMap<String, CommunicationGroup>,
    /// Default communication group
    default_group: Option<String>,
    /// Backend implementations
    backends: HashMap<CommunicationBackend, Box<dyn CommunicationBackendImpl>>,
    /// Performance metrics
    metrics: Arc<Mutex<CommunicationMetrics>>,
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

impl Default for CommunicationRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunicationRuntime {
    /// Create new communication runtime
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            default_group: None,
            backends: HashMap::new(),
            metrics: Arc::new(Mutex::new(CommunicationMetrics::default())),
        }
    }

    /// Register a communication backend
    pub fn register_backend(
        &mut self,
        backend_type: CommunicationBackend,
        backend: Box<dyn CommunicationBackendImpl>,
    ) {
        self.backends.insert(backend_type, backend);
    }

    /// Initialize communication runtime with configuration
    pub fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        // Initialize all registered backends
        for (backend_type, backend) in &mut self.backends {
            backend.initialize(config)?;
        }
        Ok(())
    }

    /// Create a new communication group
    pub fn create_group(&mut self, group: CommunicationGroup) -> Result<()> {
        // Create group with appropriate backend
        if let Some(backend) = self.backends.get_mut(&group.backend) {
            backend.create_group(&group)?;
        } else {
            return Err(TensorError::unsupported_operation_simple(format!(
                "Backend {:?} not registered",
                group.backend
            )));
        }

        let group_id = group.group_id.clone();
        self.groups.insert(group_id.clone(), group);

        // Set as default group if none exists
        if self.default_group.is_none() {
            self.default_group = Some(group_id);
        }

        Ok(())
    }

    /// Perform collective operation with f32 tensors
    pub fn collective_op_f32(
        &self,
        op: CollectiveOp,
        tensor: &Tensor<f32>,
        group_id: Option<&str>,
    ) -> Result<CollectiveResult<f32>> {
        let group_id = group_id.or(self.default_group.as_deref()).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "collective_op_f32",
                "No communication group specified or available",
            )
        })?;

        let group = self.groups.get(group_id).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "collective_op_f32",
                &format!("Communication group '{group_id}' not found"),
            )
        })?;

        let backend = self.backends.get(&group.backend).ok_or_else(|| {
            TensorError::unsupported_operation_simple(format!(
                "Backend {:?} not available",
                group.backend
            ))
        })?;

        let start_time = Instant::now();

        let result = match op {
            CollectiveOp::AllReduce { reduction_op } => {
                let result_tensor = backend.all_reduce_f32(tensor, group, reduction_op)?;
                CollectiveResult::Tensor(result_tensor)
            }
            CollectiveOp::AllGather => {
                let result_tensors = backend.all_gather_f32(tensor, group)?;
                CollectiveResult::TensorList(result_tensors)
            }
            CollectiveOp::Broadcast { root_rank } => {
                let result_tensor = backend.broadcast_f32(tensor, root_rank, group)?;
                CollectiveResult::Tensor(result_tensor)
            }
            CollectiveOp::Send { dest_rank } => {
                backend.send_f32(tensor, dest_rank, group)?;
                CollectiveResult::None
            }
            CollectiveOp::Recv { src_rank } => {
                let result_tensor = backend.recv_f32(tensor.shape().dims(), src_rank, group)?;
                CollectiveResult::Tensor(result_tensor)
            }
            CollectiveOp::ReduceScatter { reduction_op } => {
                // Simplified implementation - in practice this would be more complex
                let gathered = backend.all_gather_f32(tensor, group)?;
                let reduced = self.reduce_tensors_f32(&gathered, reduction_op)?;
                let chunk_size = reduced.shape().dims()[0] / group.world_size;
                let start_idx = group.rank * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, reduced.shape().dims()[0]);
                // Slice the reduced tensor to get the chunk for this rank
                #[allow(clippy::single_range_in_vec_init)]
                let result_tensor = slice(&reduced, &[start_idx..end_idx])?;
                CollectiveResult::Tensor(result_tensor)
            }
        };

        let elapsed = start_time.elapsed();
        self.update_metrics(&op, tensor, elapsed);

        Ok(result)
    }

    /// Get communication group
    pub fn get_group(&self, group_id: &str) -> Option<&CommunicationGroup> {
        self.groups.get(group_id)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> CommunicationMetrics {
        self.metrics
            .lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }

    /// Finalize communication runtime
    pub fn finalize(&mut self) -> Result<()> {
        for backend in self.backends.values_mut() {
            backend.finalize()?;
        }
        Ok(())
    }

    /// Reduce multiple f32 tensors with specified operation
    fn reduce_tensors_f32(&self, tensors: &[Tensor<f32>], op: ReductionOp) -> Result<Tensor<f32>> {
        if tensors.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "reduce_tensors_f32",
                "No tensors to reduce",
            ));
        }

        let mut result = tensors[0].clone();

        for tensor in &tensors[1..] {
            result = match op {
                ReductionOp::Sum => result.add(tensor)?,
                ReductionOp::Average => result.add(tensor)?,
                ReductionOp::Min => element_wise_min(&result, tensor)?,
                ReductionOp::Max => element_wise_max(&result, tensor)?,
                ReductionOp::Product => result.mul(tensor)?,
            };
        }

        if matches!(op, ReductionOp::Average) {
            // For average, divide by number of tensors
            let len_f32 = tensors.len() as f32;
            let divisor = Tensor::from_scalar(len_f32);
            result = result.div(&divisor)?;
        }

        Ok(result)
    }

    /// Update performance metrics for f32 tensors
    fn update_metrics(&self, op: &CollectiveOp, tensor: &Tensor<f32>, elapsed: Duration) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.operation_count += 1;
            metrics.total_time += elapsed;

            // Estimate bytes transferred (simplified) - f32 is 4 bytes
            let tensor_size = tensor.shape().size() * 4;
            metrics.total_bytes += tensor_size as u64;

            // Update average bandwidth
            if metrics.total_time.as_secs_f64() > 0.0 {
                metrics.avg_bandwidth =
                    metrics.total_bytes as f64 / metrics.total_time.as_secs_f64();
            }

            // Update operation-specific metrics
            let op_name = match op {
                CollectiveOp::AllReduce { .. } => "all_reduce",
                CollectiveOp::AllGather => "all_gather",
                CollectiveOp::Broadcast { .. } => "broadcast",
                CollectiveOp::Send { .. } => "send",
                CollectiveOp::Recv { .. } => "recv",
                CollectiveOp::ReduceScatter { .. } => "reduce_scatter",
            };

            let op_metrics = metrics
                .operation_metrics
                .entry(op_name.to_string())
                .or_default();
            op_metrics.count += 1;
            op_metrics.total_time += elapsed;
            op_metrics.total_bytes += tensor_size as u64;
            op_metrics.avg_latency = op_metrics.total_time / op_metrics.count as u32;
        }
    }
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

/// High-level distributed model wrappers
pub mod models {
    use super::*;
    use crate::Model;
    use parking_lot::RwLock;
    use std::sync::Arc;

    /// Data Parallel model wrapper for single-node multi-GPU training
    pub struct DataParallel {
        /// Base model to replicate across devices
        base_model: Arc<RwLock<Box<dyn Model<f32>>>>,
        /// Device assignments for each replica
        device_replicas: Vec<Device>,
        /// Communication runtime for gradient synchronization
        comm_runtime: Arc<RwLock<CommunicationRuntime>>,
        /// Whether model is in training mode
        is_training: bool,
        /// Synchronization mode
        sync_mode: SynchronizationMode,
    }

    /// Distributed Data Parallel model wrapper for multi-node training
    pub struct DistributedDataParallel {
        /// Base model wrapped for distributed training
        base_model: Arc<RwLock<Box<dyn Model<f32>>>>,
        /// Communication group for this DDP instance
        process_group: Arc<CommunicationGroup>,
        /// Communication runtime
        comm_runtime: Arc<RwLock<CommunicationRuntime>>,
        /// Local device for this process
        device: Device,
        /// Whether to broadcast parameters from rank 0 on initialization
        broadcast_buffers: bool,
        /// Whether model is in training mode
        is_training: bool,
        /// Gradient bucket size for efficient communication
        bucket_size: usize,
        /// DDP-specific configuration
        ddp_config: DDPConfig,
    }

    /// Configuration for Distributed Data Parallel training
    #[derive(Debug, Clone)]
    pub struct DDPConfig {
        /// Find unused parameters to skip in gradient sync
        pub find_unused_parameters: bool,
        /// Gradient as bucket view for memory efficiency
        pub gradient_as_bucket_view: bool,
        /// Static computation graph optimization
        pub static_graph: bool,
        /// Delay all-reduce until backward is complete
        pub delay_all_reduce: bool,
    }

    /// Synchronization modes for DataParallel
    #[derive(Debug, Clone, Copy)]
    pub enum SynchronizationMode {
        /// Synchronous - wait for all replicas
        Synchronous,
        /// Asynchronous - don't wait for all replicas
        Asynchronous,
        /// Bounded staleness - allow limited staleness
        BoundedStaleness { max_staleness: u32 },
    }

    impl Default for DDPConfig {
        fn default() -> Self {
            Self {
                find_unused_parameters: false,
                gradient_as_bucket_view: false,
                static_graph: false,
                delay_all_reduce: true,
            }
        }
    }

    impl DataParallel {
        /// Create new DataParallel model wrapper
        pub fn new(
            model: Box<dyn Model<f32>>,
            devices: Vec<Device>,
            comm_runtime: Arc<RwLock<CommunicationRuntime>>,
        ) -> Result<Self> {
            if devices.is_empty() {
                return Err(TensorError::invalid_argument_op(
                    "DataParallel::new",
                    "No devices provided",
                ));
            }

            #[allow(clippy::arc_with_non_send_sync)]
            let base_model = Arc::new(RwLock::new(model));

            // Replicate model parameters to all devices
            Self::replicate_parameters(&base_model, &devices)?;

            Ok(Self {
                base_model,
                device_replicas: devices,
                comm_runtime,
                is_training: true,
                sync_mode: SynchronizationMode::Synchronous,
            })
        }

        /// Replicate model parameters across devices
        fn replicate_parameters(
            model: &Arc<RwLock<Box<dyn Model<f32>>>>,
            devices: &[Device],
        ) -> Result<()> {
            let model_read = model.read();
            let parameters = model_read.parameters();

            for param in parameters {
                for device in devices {
                    if *param.device() != *device {
                        // Transfer parameter to target device
                        param.to(device.clone())?;
                    }
                }
            }

            Ok(())
        }

        /// Perform forward pass with data parallelism
        pub fn forward_parallel(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
            if inputs.len() != self.device_replicas.len() {
                return Err(TensorError::invalid_argument_op(
                    "forward_parallel",
                    &format!(
                        "Expected {} inputs for {} devices",
                        self.device_replicas.len(),
                        inputs.len()
                    ),
                ));
            }

            let model = self.base_model.read();
            let mut outputs = Vec::with_capacity(inputs.len());

            // Run forward pass on each device replica
            for (input, device) in inputs.iter().zip(&self.device_replicas) {
                // Ensure input is on correct device
                let input_on_device = if *input.device() != *device {
                    input.to(device.clone())?
                } else {
                    input.clone()
                };

                let output = model.forward(&input_on_device)?;
                outputs.push(output);
            }

            Ok(outputs)
        }

        /// Synchronize gradients across all device replicas
        pub fn sync_gradients(&mut self) -> Result<()> {
            if !self.is_training {
                return Ok(()); // No gradient sync in eval mode
            }

            let mut model = self.base_model.write();
            let mut parameters = model.parameters_mut();

            // All-reduce gradients across devices
            for param in parameters.iter_mut() {
                if let Some(grad) = param.grad() {
                    let comm_runtime = self.comm_runtime.read();

                    // Perform all-reduce on gradient
                    let op = CollectiveOp::AllReduce {
                        reduction_op: ReductionOp::Average,
                    };

                    if let Ok(CollectiveResult::Tensor(synced_grad)) =
                        comm_runtime.collective_op_f32(op, grad, None)
                    {
                        param.set_grad(Some(synced_grad));
                    }
                }
            }

            Ok(())
        }

        /// Set synchronization mode
        pub fn set_sync_mode(&mut self, mode: SynchronizationMode) {
            self.sync_mode = mode;
        }

        /// Get device replicas
        pub fn devices(&self) -> &[Device] {
            &self.device_replicas
        }
    }

    impl Model<f32> for DataParallel {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            // For single input, replicate across devices and gather results
            let inputs: Vec<Tensor<f32>> = self
                .device_replicas
                .iter()
                .map(|device| input.to(device.clone()))
                .collect::<Result<Vec<_>>>()?;

            let outputs = self.forward_parallel(&inputs)?;

            // Gather outputs to primary device
            let primary_device = &self.device_replicas[0];
            let gathered_outputs: Vec<Tensor<f32>> = outputs
                .into_iter()
                .map(|output| output.to(primary_device.clone()))
                .collect::<Result<Vec<_>>>()?;

            // Average outputs from all replicas
            let mut result = gathered_outputs[0].clone();
            for output in &gathered_outputs[1..] {
                result = result.add(output)?;
            }

            let num_devices = gathered_outputs.len() as f32;
            let divisor = Tensor::from_scalar(num_devices);
            result.div(&divisor)
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            // We can't return references to parameters due to the RwLock guard lifetime
            // This is a limitation that would need a different design in practice
            vec![]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            // We can't return mutable references to parameters due to the RwLock guard lifetime
            // This is a limitation that would need a different design in practice
            vec![]
        }

        fn set_training(&mut self, training: bool) {
            self.is_training = training;
            self.base_model.write().set_training(training);
        }

        fn zero_grad(&mut self) {
            self.base_model.write().zero_grad();
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl DistributedDataParallel {
        /// Create new DistributedDataParallel model wrapper
        pub fn new(
            model: Box<dyn Model<f32>>,
            device: Device,
            process_group: Arc<CommunicationGroup>,
            comm_runtime: Arc<RwLock<CommunicationRuntime>>,
            config: DDPConfig,
        ) -> Result<Self> {
            #[allow(clippy::arc_with_non_send_sync)]
            let base_model = Arc::new(RwLock::new(model));

            let mut ddp = Self {
                base_model,
                process_group,
                comm_runtime,
                device,
                broadcast_buffers: true,
                is_training: true,
                bucket_size: 25 * 1024 * 1024, // 25MB default bucket size
                ddp_config: config,
            };

            // Broadcast parameters from rank 0 to ensure synchronization
            if ddp.broadcast_buffers {
                ddp.broadcast_parameters()?;
            }

            Ok(ddp)
        }

        /// Broadcast model parameters from rank 0 to all other ranks
        fn broadcast_parameters(&mut self) -> Result<()> {
            let model = self.base_model.read();
            let parameters = model.parameters();

            let comm_runtime = self.comm_runtime.read();

            for param in parameters {
                let op = CollectiveOp::Broadcast { root_rank: 0 };

                if let Ok(CollectiveResult::Tensor(synced_param)) =
                    comm_runtime.collective_op_f32(op, param, Some(&self.process_group.group_id))
                {
                    // Update parameter with broadcasted value
                    // Note: This is a simplified implementation
                    // In practice, we'd need mutable access to parameters
                }
            }

            Ok(())
        }

        /// Perform gradient synchronization using all-reduce
        pub fn sync_gradients(&mut self) -> Result<()> {
            if !self.is_training {
                return Ok(());
            }

            let mut model = self.base_model.write();
            let mut parameters = model.parameters_mut();

            // Group gradients into buckets for efficient communication
            let mut gradient_buckets = self.create_gradient_buckets(&mut parameters)?;

            let comm_runtime = self.comm_runtime.read();

            // All-reduce each gradient bucket
            for bucket in &gradient_buckets {
                for grad_tensor in bucket {
                    let op = CollectiveOp::AllReduce {
                        reduction_op: ReductionOp::Average,
                    };

                    // Note: In practice, we would need to update the actual gradient tensors
                    // This is a simplified implementation for demonstration
                    if let Ok(CollectiveResult::Tensor(_synced_grad)) = comm_runtime
                        .collective_op_f32(op, grad_tensor, Some(&self.process_group.group_id))
                    {
                        // In a complete implementation, we would update the parameter gradients
                        // with the synchronized gradients
                    }
                }
            }

            Ok(())
        }

        /// Create gradient buckets for efficient communication
        fn create_gradient_buckets<'a>(
            &self,
            parameters: &'a mut [&'a mut Tensor<f32>],
        ) -> Result<Vec<Vec<&'a Tensor<f32>>>> {
            let mut buckets = Vec::new();
            let mut current_bucket = Vec::new();
            let mut current_bucket_size = 0;

            for param in parameters {
                if let Some(grad) = param.grad() {
                    let grad_size = grad.shape().size() * std::mem::size_of::<f32>();

                    if current_bucket_size + grad_size > self.bucket_size
                        && !current_bucket.is_empty()
                    {
                        buckets.push(std::mem::take(&mut current_bucket));
                        current_bucket_size = 0;
                    }

                    current_bucket.push(grad);
                    current_bucket_size += grad_size;
                }
            }

            if !current_bucket.is_empty() {
                buckets.push(current_bucket);
            }

            Ok(buckets)
        }

        /// Get process group information
        pub fn process_group(&self) -> &CommunicationGroup {
            &self.process_group
        }

        /// Get local rank within process group
        pub fn local_rank(&self) -> usize {
            self.process_group.rank
        }

        /// Get world size (total number of processes)
        pub fn world_size(&self) -> usize {
            self.process_group.world_size
        }

        /// Set bucket size for gradient communication
        pub fn set_bucket_size(&mut self, size: usize) {
            self.bucket_size = size;
        }
    }

    impl Model<f32> for DistributedDataParallel {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            // Ensure input is on correct device
            let input_on_device = if *input.device() != self.device {
                input.to(self.device.clone())?
            } else {
                input.clone()
            };

            // Forward pass through base model
            self.base_model.read().forward(&input_on_device)
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            // We can't return references to parameters due to the RwLock guard lifetime
            // This is a limitation that would need a different design in practice
            vec![]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            // We can't return mutable references to parameters due to the RwLock guard lifetime
            // This is a limitation that would need a different design in practice
            vec![]
        }

        fn set_training(&mut self, training: bool) {
            self.is_training = training;
            self.base_model.write().set_training(training);
        }

        fn zero_grad(&mut self) {
            self.base_model.write().zero_grad();
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Utility functions for distributed models
    pub mod utils {
        use super::*;

        /// Initialize process group for distributed training
        pub fn init_process_group(
            backend: CommunicationBackend,
            rank: usize,
            world_size: usize,
        ) -> Result<(Arc<RwLock<CommunicationRuntime>>, Arc<CommunicationGroup>)> {
            let mut comm_runtime = CommunicationRuntime::new();

            // Register appropriate backend
            match backend {
                CommunicationBackend::Thread => {
                    comm_runtime.register_backend(
                        CommunicationBackend::Thread,
                        Box::new(crate::backends::thread::ThreadBackend::new()),
                    );
                }
                #[cfg(feature = "nccl")]
                CommunicationBackend::Nccl => {
                    comm_runtime.register_backend(
                        CommunicationBackend::Nccl,
                        Box::new(crate::backends::nccl::NcclBackend::new()),
                    );
                }
                _ => {
                    return Err(TensorError::unsupported_operation_simple(format!(
                        "Backend {backend:?} not supported"
                    )));
                }
            }

            let config = BackendConfig::default();
            comm_runtime.initialize(&config)?;

            // Create process group
            let devices = auto_detect_available_devices();
            let process_group = Arc::new(CommunicationGroup {
                group_id: "ddp_main".to_string(),
                rank,
                world_size,
                devices,
                backend,
            });

            let runtime = Arc::new(RwLock::new(comm_runtime));
            runtime.write().create_group((*process_group).clone())?;

            Ok((runtime, process_group))
        }

        /// Create DataParallel model wrapper with automatic device detection
        pub fn create_data_parallel(model: Box<dyn Model<f32>>) -> Result<DataParallel> {
            let devices = auto_detect_available_devices();
            let comm_runtime = crate::distributed::utils::init_distributed(0, devices.len(), None)?;
            let runtime = Arc::new(RwLock::new(comm_runtime));

            DataParallel::new(model, devices, runtime)
        }

        /// Create DistributedDataParallel model wrapper
        pub fn create_distributed_data_parallel(
            model: Box<dyn Model<f32>>,
            device: Device,
            backend: CommunicationBackend,
            rank: usize,
            world_size: usize,
        ) -> Result<DistributedDataParallel> {
            let (comm_runtime, process_group) = init_process_group(backend, rank, world_size)?;
            let config = DDPConfig::default();

            DistributedDataParallel::new(model, device, process_group, comm_runtime, config)
        }
    }
}

/// Utility functions for distributed communication
pub mod utils {
    use super::*;
    #[cfg(feature = "gloo")]
    use crate::backends::gloo::GlooBackend;
    #[cfg(feature = "nccl")]
    use crate::backends::nccl::NcclBackend;
    use crate::backends::thread::ThreadBackend;

    /// Initialize distributed environment with automatic backend selection
    pub fn init_distributed(
        rank: usize,
        world_size: usize,
        backend: Option<CommunicationBackend>,
    ) -> Result<CommunicationRuntime> {
        let mut runtime = CommunicationRuntime::new();

        // Auto-select backend if not specified
        let backend = backend.unwrap_or({
            #[cfg(feature = "nccl")]
            {
                // Check if NCCL is available and we have NVIDIA GPUs
                CommunicationBackend::Nccl
            }
            #[cfg(not(feature = "nccl"))]
            {
                CommunicationBackend::Thread
            }
        });

        // Register and initialize backend
        match backend {
            CommunicationBackend::Thread => {
                runtime
                    .register_backend(CommunicationBackend::Thread, Box::new(ThreadBackend::new()));
            }
            #[cfg(feature = "nccl")]
            CommunicationBackend::Nccl => {
                runtime.register_backend(CommunicationBackend::Nccl, Box::new(NcclBackend::new()));
            }
            #[cfg(feature = "gloo")]
            CommunicationBackend::Gloo => {
                runtime.register_backend(CommunicationBackend::Gloo, Box::new(GlooBackend::new()));
            }
            #[cfg(not(feature = "gloo"))]
            CommunicationBackend::Gloo => {
                return Err(TensorError::unsupported_operation_simple(
                    "Gloo backend not compiled in. Enable 'gloo' feature".to_string(),
                ));
            }
            _ => {
                return Err(TensorError::unsupported_operation_simple(format!(
                    "Backend {backend:?} not supported"
                )));
            }
        }

        let config = BackendConfig::default();
        runtime.initialize(&config)?;

        // Create default group
        let devices = auto_detect_available_devices();
        let group = CommunicationGroup {
            group_id: "default".to_string(),
            rank,
            world_size,
            devices,
            backend,
        };

        runtime.create_group(group)?;

        Ok(runtime)
    }

    /// Create a distributed data parallel group
    pub fn create_data_parallel_group(
        runtime: &mut CommunicationRuntime,
        devices: Vec<Device>,
        backend: CommunicationBackend,
    ) -> Result<String> {
        let group_id = "data_parallel".to_string();
        let world_size = devices.len();

        // Get actual rank from device count or environment variable
        let actual_rank = if let Ok(rank_str) = std::env::var("RANK") {
            rank_str.parse::<usize>().unwrap_or(0)
        } else if let Ok(local_rank_str) = std::env::var("LOCAL_RANK") {
            local_rank_str.parse::<usize>().unwrap_or(0)
        } else {
            // Default to rank 0 for single-device setups
            0
        };

        let group = CommunicationGroup {
            group_id: group_id.clone(),
            rank: actual_rank,
            world_size,
            devices,
            backend,
        };

        runtime.create_group(group)?;
        Ok(group_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::thread::ThreadBackend;

    #[test]
    fn test_communication_runtime_creation() {
        let runtime = CommunicationRuntime::new();
        assert!(runtime.groups.is_empty());
        assert!(runtime.default_group.is_none());
    }

    #[test]
    fn test_thread_backend() {
        let mut backend = ThreadBackend::new();
        let config = BackendConfig::default();

        assert!(backend.initialize(&config).is_ok());
        assert_eq!(backend.name(), "thread");
    }

    #[test]
    fn test_communication_group_creation() {
        let mut runtime = CommunicationRuntime::new();
        runtime.register_backend(CommunicationBackend::Thread, Box::new(ThreadBackend::new()));

        let config = BackendConfig::default();
        runtime.initialize(&config).unwrap();

        let group = CommunicationGroup {
            group_id: "test_group".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        assert!(runtime.create_group(group).is_ok());
        assert!(runtime.get_group("test_group").is_some());
    }

    #[test]
    fn test_all_reduce_operation() {
        let mut runtime = CommunicationRuntime::new();
        runtime.register_backend(CommunicationBackend::Thread, Box::new(ThreadBackend::new()));

        let config = BackendConfig::default();
        runtime.initialize(&config).unwrap();

        let group = CommunicationGroup {
            group_id: "test_group".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        runtime.create_group(group).unwrap();

        let tensor = Tensor::<f32>::ones(&[2, 3]);
        let op = CollectiveOp::AllReduce {
            reduction_op: ReductionOp::Sum,
        };

        let result = runtime
            .collective_op_f32(op, &tensor, Some("test_group"))
            .unwrap();

        assert!(
            matches!(result, CollectiveResult::Tensor(_)),
            "Expected tensor result from collective operation"
        );
    }
}

/// Auto-detect available devices for distributed training
fn auto_detect_available_devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu]; // CPU is always available

    #[cfg(feature = "gpu")]
    {
        // Try to detect GPU devices
        // In a real implementation, this would query the GPU runtime
        // For now, we'll use a simple heuristic

        // Check if CUDA_VISIBLE_DEVICES is set
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            // Parse CUDA_VISIBLE_DEVICES to get available GPU IDs
            if let Ok(cuda_devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
                for (i, device_id) in cuda_devices.split(',').enumerate() {
                    if let Ok(_id) = device_id.trim().parse::<u32>() {
                        devices.push(Device::Gpu(i));
                    }
                }
            }
        } else {
            // Default: assume one GPU is available if GPU feature is enabled
            devices.push(Device::Gpu(0));
        }
    }

    devices
}
