use crate::gradient_accumulation::GradientAccumulator;
use crate::tape::{GradientTape, TrackedTensor};
use futures::channel::oneshot;
use futures::Future;
use futures::FutureExt;
// HashMap, Arc, Mutex not currently used
use std::thread;
use std::time::Instant;
use tenflowers_core::{Device, Result, Tensor, TensorError};

/// Parallel gradient computation configuration
#[derive(Debug, Clone)]
pub struct ParallelGradientConfig {
    /// Number of worker threads for gradient computation
    pub num_workers: usize,
    /// Maximum batch size per worker
    pub max_batch_size: usize,
    /// Whether to enable asynchronous gradient computation
    pub async_gradients: bool,
    /// Communication backend for multi-GPU
    pub communication_backend: CommunicationBackend,
    /// Pipeline parallelism configuration
    pub pipeline_config: Option<PipelineConfig>,
}

/// Communication backend for multi-GPU training
#[derive(Debug, Clone)]
pub enum CommunicationBackend {
    /// NCCL-like ring allreduce
    RingAllReduce,
    /// Parameter server approach
    ParameterServer,
    /// Peer-to-peer communication
    PeerToPeer,
}

/// Pipeline parallelism configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Micro-batch size for pipeline
    pub micro_batch_size: usize,
    /// Number of devices per stage
    pub devices_per_stage: usize,
}

/// Parallel gradient computation engine
pub struct ParallelGradientEngine {
    config: ParallelGradientConfig,
    devices: Vec<Device>,
    #[allow(dead_code)]
    gradient_accumulator: GradientAccumulator,
    async_runtime: tokio::runtime::Runtime,
}

/// Gradient computation task for parallel execution
#[derive(Debug, Clone)]
pub struct GradientTask<T> {
    /// Tensor to compute gradients for
    pub target: TrackedTensor<T>,
    /// Sources to compute gradients with respect to
    pub sources: Vec<TrackedTensor<T>>,
    /// Device to execute on
    pub device: Device,
    /// Task ID for tracking
    pub task_id: u64,
}

/// Result of parallel gradient computation
#[derive(Debug)]
pub struct ParallelGradientResult<T> {
    /// Computed gradients
    pub gradients: Vec<Tensor<T>>,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Memory usage
    pub memory_usage: u64,
    /// Device used
    pub device: Device,
}

/// Asynchronous gradient computation handle
pub struct AsyncGradientHandle<T> {
    receiver: oneshot::Receiver<Result<ParallelGradientResult<T>>>,
}

impl<T> Future for AsyncGradientHandle<T> {
    type Output = Result<ParallelGradientResult<T>>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match self.receiver.poll_unpin(cx) {
            std::task::Poll::Ready(Ok(result)) => std::task::Poll::Ready(result),
            std::task::Poll::Ready(Err(_)) => std::task::Poll::Ready(Err(
                TensorError::compute_error_simple("Gradient computation failed".to_string()),
            )),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl ParallelGradientEngine {
    /// Create a new parallel gradient engine
    pub fn new(config: ParallelGradientConfig, devices: Vec<Device>) -> Result<Self> {
        let gradient_accumulator = GradientAccumulator::new(true);
        let async_runtime = tokio::runtime::Runtime::new().map_err(|e| {
            TensorError::compute_error_simple(format!("Failed to create async runtime: {}", e))
        })?;

        Ok(Self {
            config,
            devices,
            gradient_accumulator,
            async_runtime,
        })
    }

    /// Compute gradients in parallel across multiple devices
    pub fn compute_gradients_parallel<T>(
        &self,
        tasks: Vec<GradientTask<T>>,
    ) -> Result<Vec<ParallelGradientResult<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + num_traits::FromPrimitive
            + num_traits::Float
            + num_traits::Signed
            + PartialOrd
            + bytemuck::Pod,
    {
        let mut results = Vec::new();
        let mut handles = Vec::new();

        // Distribute tasks across devices
        for (i, task) in tasks.into_iter().enumerate() {
            let device = self.devices[i % self.devices.len()];
            let tape = GradientTape::new();

            let handle = thread::spawn(move || {
                let start_time = Instant::now();

                // Compute gradients on this device
                let target_on_device = tape.watch(task.target.tensor.to_device(device)?);
                let sources_on_device: Result<Vec<_>> = task
                    .sources
                    .iter()
                    .map(|s| Ok(tape.watch(s.tensor.to_device(device)?)))
                    .collect();
                let sources_on_device = sources_on_device?;
                let source_refs: Vec<_> = sources_on_device.iter().collect();

                let gradients = tape.gradient(&[target_on_device], &sources_on_device)?;

                let execution_time = start_time.elapsed();
                let memory_usage = gradients
                    .iter()
                    .map(|g| {
                        g.as_ref().map(|tensor| tensor.numel()).unwrap_or(0)
                            * std::mem::size_of::<T>()
                    })
                    .sum::<usize>() as u64;

                let unwrapped_gradients: Vec<Tensor<T>> =
                    gradients.into_iter().filter_map(|g| g).collect();

                Ok(ParallelGradientResult {
                    gradients: unwrapped_gradients,
                    execution_time,
                    memory_usage,
                    device,
                })
            });

            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let result: Result<ParallelGradientResult<T>> = handle
                .join()
                .map_err(|_| TensorError::compute_error_simple("Thread join failed".to_string()))?;
            results.push(result?);
        }

        // Aggregate gradients across devices if multi-GPU
        if self.devices.len() > 1 {
            self.aggregate_gradients_multi_gpu(&mut results)?;
        }

        Ok(results)
    }

    /// Compute gradients asynchronously
    pub fn compute_gradients_async<T>(
        &self,
        tasks: Vec<GradientTask<T>>,
    ) -> Vec<AsyncGradientHandle<T>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + num_traits::FromPrimitive
            + num_traits::Float
            + num_traits::Signed
            + PartialOrd
            + bytemuck::Pod,
    {
        let mut handles = Vec::new();

        for task in tasks {
            let (sender, receiver) = oneshot::channel();
            let _device = task.device;

            self.async_runtime.spawn(async move {
                let result = Self::compute_single_gradient_async(task).await;
                let _ = sender.send(result);
            });

            handles.push(AsyncGradientHandle { receiver });
        }

        handles
    }

    /// Compute gradients with pipeline parallelism
    pub fn compute_gradients_pipeline<T>(
        &self,
        tasks: Vec<GradientTask<T>>,
    ) -> Result<Vec<ParallelGradientResult<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + num_traits::FromPrimitive
            + num_traits::Float
            + num_traits::Signed
            + PartialOrd
            + bytemuck::Pod,
    {
        let pipeline_config = self.config.pipeline_config.as_ref().ok_or_else(|| {
            TensorError::invalid_argument("Pipeline config not provided".to_string())
        })?;

        let mut results = Vec::new();
        let stage_size = tasks.len() / pipeline_config.num_stages;

        // Process tasks in pipeline stages
        for stage in 0..pipeline_config.num_stages {
            let start_idx = stage * stage_size;
            let end_idx = if stage == pipeline_config.num_stages - 1 {
                tasks.len()
            } else {
                (stage + 1) * stage_size
            };

            let stage_tasks = tasks[start_idx..end_idx].to_vec();
            let stage_results = self.compute_stage_gradients(stage_tasks, stage)?;
            results.extend(stage_results);
        }

        Ok(results)
    }

    /// Aggregate gradients across multiple GPUs using simple averaging
    fn aggregate_gradients_multi_gpu<T>(
        &self,
        results: &mut [ParallelGradientResult<T>],
    ) -> Result<()>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Send
            + Sync
            + 'static
            + num_traits::FromPrimitive
            + num_traits::Zero
            + num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if results.is_empty() {
            return Ok(());
        }

        let num_devices = results.len();
        let gradient_len = results[0].gradients.len();

        // For each gradient position
        for grad_idx in 0..gradient_len {
            let mut sum_gradient: Option<Tensor<T>> = None;

            // Collect and sum tensors from all devices
            for result in results.iter() {
                if grad_idx < result.gradients.len() {
                    if let Some(ref mut sum) = sum_gradient {
                        *sum = sum.add(&result.gradients[grad_idx])?;
                    } else {
                        sum_gradient = Some(result.gradients[grad_idx].clone());
                    }
                }
            }

            // Average the gradients
            if let Some(sum) = sum_gradient {
                let scale = T::from_usize(num_devices).unwrap();
                let avg_gradient = sum.div(&Tensor::from_scalar(scale))?;

                // Update all results with averaged gradient
                for result in results.iter_mut() {
                    if grad_idx < result.gradients.len() {
                        result.gradients[grad_idx] = avg_gradient.clone();
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute gradients for a single stage in pipeline
    fn compute_stage_gradients<T>(
        &self,
        tasks: Vec<GradientTask<T>>,
        stage: usize,
    ) -> Result<Vec<ParallelGradientResult<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + num_traits::FromPrimitive
            + num_traits::Float
            + num_traits::Signed
            + PartialOrd
            + bytemuck::Pod,
    {
        let mut results = Vec::new();
        let pipeline_config = self.config.pipeline_config.as_ref().unwrap();

        // Select devices for this stage
        let stage_devices = &self.devices[stage * pipeline_config.devices_per_stage
            ..(stage + 1) * pipeline_config.devices_per_stage];

        // Process tasks with selected devices
        for (i, task) in tasks.into_iter().enumerate() {
            let device = &stage_devices[i % stage_devices.len()];
            let tape = GradientTape::new();

            let start_time = Instant::now();

            // Move tensors to stage device
            let target_on_device = tape.watch(task.target.tensor.to_device(*device)?);
            let sources_on_device: Result<Vec<_>> = task
                .sources
                .iter()
                .map(|s| Ok(tape.watch(s.tensor.to_device(*device)?)))
                .collect();
            let sources_on_device = sources_on_device?;
            let source_refs: Vec<_> = sources_on_device.iter().collect();

            let gradients = tape.gradient(&[target_on_device], &sources_on_device)?;

            let execution_time = start_time.elapsed();
            let memory_usage = gradients
                .iter()
                .map(|g| {
                    g.as_ref().map(|tensor| tensor.numel()).unwrap_or(0) * std::mem::size_of::<T>()
                })
                .sum::<usize>() as u64;

            let unwrapped_gradients: Vec<Tensor<T>> =
                gradients.into_iter().filter_map(|g| g).collect();

            results.push(ParallelGradientResult {
                gradients: unwrapped_gradients,
                execution_time,
                memory_usage,
                device: *device,
            });
        }

        Ok(results)
    }

    /// Async computation of a single gradient
    async fn compute_single_gradient_async<T>(
        task: GradientTask<T>,
    ) -> Result<ParallelGradientResult<T>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + num_traits::FromPrimitive
            + num_traits::Float
            + num_traits::Signed
            + PartialOrd
            + bytemuck::Pod,
    {
        let start_time = Instant::now();
        let tape = GradientTape::new();

        // Compute gradients
        let target_on_device = tape.watch(task.target.tensor.to_device(task.device)?);
        let sources_on_device: Result<Vec<_>> = task
            .sources
            .iter()
            .map(|s| Ok(tape.watch(s.tensor.to_device(task.device)?)))
            .collect();
        let sources_on_device = sources_on_device?;
        let source_refs: Vec<_> = sources_on_device.iter().collect();

        let gradients = tape.gradient(&[target_on_device], &sources_on_device)?;

        let execution_time = start_time.elapsed();
        let memory_usage = gradients
            .iter()
            .map(|g| {
                g.as_ref().map(|tensor| tensor.numel()).unwrap_or(0) * std::mem::size_of::<T>()
            })
            .sum::<usize>() as u64;

        let unwrapped_gradients: Vec<Tensor<T>> = gradients.into_iter().filter_map(|g| g).collect();

        Ok(ParallelGradientResult {
            gradients: unwrapped_gradients,
            execution_time,
            memory_usage,
            device: task.device,
        })
    }

    /// Estimate memory usage for gradients
    #[allow(dead_code)]
    fn estimate_memory_usage<T>(&self, gradients: &[Tensor<T>]) -> u64
    where
        T: Clone + Default + num_traits::Zero + num_traits::One + Send + Sync + 'static,
    {
        gradients
            .iter()
            .map(|g| g.numel() * std::mem::size_of::<T>())
            .sum::<usize>() as u64
    }
}

impl Default for ParallelGradientConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            max_batch_size: 32,
            async_gradients: true,
            communication_backend: CommunicationBackend::RingAllReduce,
            pipeline_config: None,
        }
    }
}

/// Builder for parallel gradient configuration
pub struct ParallelGradientConfigBuilder {
    config: ParallelGradientConfig,
}

impl ParallelGradientConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ParallelGradientConfig::default(),
        }
    }

    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }

    pub fn max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.config.max_batch_size = max_batch_size;
        self
    }

    pub fn async_gradients(mut self, async_gradients: bool) -> Self {
        self.config.async_gradients = async_gradients;
        self
    }

    pub fn communication_backend(mut self, backend: CommunicationBackend) -> Self {
        self.config.communication_backend = backend;
        self
    }

    pub fn pipeline_config(mut self, config: PipelineConfig) -> Self {
        self.config.pipeline_config = Some(config);
        self
    }

    pub fn build(self) -> ParallelGradientConfig {
        self.config
    }
}

impl Default for ParallelGradientConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::GradientTape;
    use tenflowers_core::{Device, Tensor};

    #[test]
    fn test_parallel_gradient_config_builder() {
        let config = ParallelGradientConfigBuilder::new()
            .num_workers(4)
            .max_batch_size(16)
            .async_gradients(false)
            .communication_backend(CommunicationBackend::ParameterServer)
            .build();

        assert_eq!(config.num_workers, 4);
        assert_eq!(config.max_batch_size, 16);
        assert!(!config.async_gradients);
        matches!(
            config.communication_backend,
            CommunicationBackend::ParameterServer
        );
    }

    #[test]
    fn test_parallel_gradient_engine_creation() {
        let config = ParallelGradientConfig::default();
        let devices = vec![Device::Cpu, Device::Cpu];

        let engine = ParallelGradientEngine::new(config, devices);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_async_gradient_computation() {
        let config = ParallelGradientConfig::default();
        let devices = vec![Device::Cpu];
        let engine = ParallelGradientEngine::new(config, devices).unwrap();

        let tape = GradientTape::new();
        let x = tape.watch(Tensor::<f32>::ones(&[2, 2]));
        let y = tape.watch(Tensor::<f32>::ones(&[2, 2]));
        let z = x.add(&y).unwrap();

        let task = GradientTask {
            target: z,
            sources: vec![x, y],
            device: Device::Cpu,
            task_id: 1,
        };

        let mut handles = engine.compute_gradients_async(vec![task]);
        let result = handles.pop().unwrap().await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.gradients.len(), 2);
    }

    #[test]
    fn test_pipeline_config() {
        let pipeline_config = PipelineConfig {
            num_stages: 4,
            micro_batch_size: 8,
            devices_per_stage: 2,
        };

        let config = ParallelGradientConfigBuilder::new()
            .pipeline_config(pipeline_config)
            .build();

        assert!(config.pipeline_config.is_some());
        let pipeline = config.pipeline_config.unwrap();
        assert_eq!(pipeline.num_stages, 4);
        assert_eq!(pipeline.micro_batch_size, 8);
        assert_eq!(pipeline.devices_per_stage, 2);
    }
}
