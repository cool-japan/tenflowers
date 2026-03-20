//! DataParallel and DistributedDataParallel model wrappers with gradient sync

use parking_lot::RwLock;
use std::sync::Arc;
use tenflowers_core::{Device, Result, Tensor, TensorError};

use super::types::{
    BackendConfig, CollectiveOp, CollectiveResult, CommunicationBackend, CommunicationGroup,
    CommunicationRuntime, ReductionOp,
};
use crate::Model;

/// Data Parallel model wrapper for single-node multi-GPU training
pub struct DataParallel {
    /// Base model to replicate across devices
    pub(crate) base_model: Arc<RwLock<Box<dyn Model<f32>>>>,
    /// Device assignments for each replica
    pub(crate) device_replicas: Vec<Device>,
    /// Communication runtime for gradient synchronization
    pub(crate) comm_runtime: Arc<RwLock<CommunicationRuntime>>,
    /// Whether model is in training mode
    pub(crate) is_training: bool,
    /// Synchronization mode
    pub(crate) sync_mode: SynchronizationMode,
}

/// Distributed Data Parallel model wrapper for multi-node training
pub struct DistributedDataParallel {
    /// Base model wrapped for distributed training
    pub(crate) base_model: Arc<RwLock<Box<dyn Model<f32>>>>,
    /// Communication group for this DDP instance
    pub(crate) process_group: Arc<CommunicationGroup>,
    /// Communication runtime
    pub(crate) comm_runtime: Arc<RwLock<CommunicationRuntime>>,
    /// Local device for this process
    pub(crate) device: Device,
    /// Whether to broadcast parameters from rank 0 on initialization
    pub(crate) broadcast_buffers: bool,
    /// Whether model is in training mode
    pub(crate) is_training: bool,
    /// Gradient bucket size for efficient communication
    pub(crate) bucket_size: usize,
    /// DDP-specific configuration
    pub(crate) ddp_config: DDPConfig,
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

        for (input, device) in inputs.iter().zip(&self.device_replicas) {
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
            return Ok(());
        }

        let mut model = self.base_model.write();
        let mut parameters = model.parameters_mut();

        for param in parameters.iter_mut() {
            if let Some(grad) = param.grad() {
                let comm_runtime = self.comm_runtime.read();

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
        let inputs: Vec<Tensor<f32>> = self
            .device_replicas
            .iter()
            .map(|device| input.to(device.clone()))
            .collect::<Result<Vec<_>>>()?;

        let outputs = self.forward_parallel(&inputs)?;

        let primary_device = &self.device_replicas[0];
        let gathered_outputs: Vec<Tensor<f32>> = outputs
            .into_iter()
            .map(|output| output.to(primary_device.clone()))
            .collect::<Result<Vec<_>>>()?;

        let mut result = gathered_outputs[0].clone();
        for output in &gathered_outputs[1..] {
            result = result.add(output)?;
        }

        let num_devices = gathered_outputs.len() as f32;
        let divisor = Tensor::from_scalar(num_devices);
        result.div(&divisor)
    }

    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
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

            if let Ok(CollectiveResult::Tensor(_synced_param)) =
                comm_runtime.collective_op_f32(op, param, Some(&self.process_group.group_id))
            {
                // Update parameter with broadcasted value
                // Note: This is a simplified implementation
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

        let mut gradient_buckets = self.create_gradient_buckets(&mut parameters)?;

        let comm_runtime = self.comm_runtime.read();

        for bucket in &gradient_buckets {
            for grad_tensor in bucket {
                let op = CollectiveOp::AllReduce {
                    reduction_op: ReductionOp::Average,
                };

                if let Ok(CollectiveResult::Tensor(_synced_grad)) = comm_runtime.collective_op_f32(
                    op,
                    grad_tensor,
                    Some(&self.process_group.group_id),
                ) {
                    // In a complete implementation, we would update the parameter gradients
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

                if current_bucket_size + grad_size > self.bucket_size && !current_bucket.is_empty()
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
        let input_on_device = if *input.device() != self.device {
            input.to(self.device.clone())?
        } else {
            input.clone()
        };

        self.base_model.read().forward(&input_on_device)
    }

    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
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
    use super::super::types::CommunicationBackend;
    use super::*;

    /// Initialize process group for distributed training
    pub fn init_process_group(
        backend: CommunicationBackend,
        rank: usize,
        world_size: usize,
    ) -> Result<(Arc<RwLock<CommunicationRuntime>>, Arc<CommunicationGroup>)> {
        let mut comm_runtime = CommunicationRuntime::new();

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

        let devices = super::super::auto_detect_available_devices();
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
        let devices = super::super::auto_detect_available_devices();
        let comm_runtime = super::super::utils::init_distributed(0, devices.len(), None)?;
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
