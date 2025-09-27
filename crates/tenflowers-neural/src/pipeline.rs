use crate::{model_parallel::PipelineConfig, Layer};
use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use tenflowers_core::{ops::concat, Device, Result, Tensor, TensorError};

/// Pipeline parallel model that splits a sequential model across multiple devices
pub struct PipelineParallelModel<T> {
    /// Original layers (referenced by stages)
    layers: Vec<Box<dyn Layer<T>>>,
    /// Stages of the pipeline (each stage contains layer indices)
    stages: Vec<PipelineStage>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Communication channels between stages
    stage_channels: Vec<StageChannel<T>>,
    /// Performance metrics
    metrics: Arc<Mutex<PipelineMetrics>>,
}

/// A single stage in the pipeline
pub struct PipelineStage {
    /// Layer indices for this stage (references to original layers)
    layer_indices: Vec<usize>,
    /// Device for this stage
    device: Device,
    /// Stage index
    stage_id: usize,
}

/// Communication channel between pipeline stages
struct StageChannel<T> {
    sender: mpsc::Sender<MicroBatch<T>>,
    receiver: mpsc::Receiver<MicroBatch<T>>,
}

/// Micro-batch for pipeline execution
#[derive(Debug)]
pub struct MicroBatch<T> {
    /// Batch data
    pub data: Tensor<T>,
    /// Micro-batch ID for tracking
    pub micro_batch_id: usize,
    /// Global batch ID
    pub batch_id: usize,
    /// Timestamp when micro-batch was created
    pub timestamp: Instant,
}

/// Pipeline execution metrics
#[derive(Debug, Default)]
pub struct PipelineMetrics {
    /// Total forward passes
    pub forward_passes: usize,
    /// Total backward passes
    pub backward_passes: usize,
    /// Average stage execution time
    pub avg_stage_time: HashMap<usize, Duration>,
    /// Pipeline bubble time (idle time)
    pub bubble_time: Duration,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Memory usage per stage
    pub memory_usage: HashMap<usize, usize>,
}

/// Pipeline execution scheduler
pub struct PipelineScheduler {
    /// Number of stages
    num_stages: usize,
    /// Micro-batch size
    micro_batch_size: usize,
    /// Number of micro-batches in flight
    num_micro_batches: usize,
    /// Current scheduling state
    state: SchedulerState,
    /// Metrics tracking
    metrics: Arc<Mutex<PipelineMetrics>>,
}

/// Pipeline scheduler state
#[derive(Debug, Clone)]
struct SchedulerState {
    /// Current global batch being processed
    current_batch: usize,
    /// Current micro-batch within the batch
    current_micro_batch: usize,
    /// Active micro-batches in pipeline
    active_micro_batches: VecDeque<usize>,
}

impl<T> PipelineParallelModel<T>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new pipeline parallel model
    pub fn new(layers: Vec<Box<dyn Layer<T>>>, config: PipelineConfig) -> Result<Self> {
        // Validate configuration
        if config.num_stages == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "PipelineParallelModel::new".to_string(),
                reason: "Pipeline must have at least one stage".to_string(),
                context: None,
            });
        }

        if config.stage_devices.len() != config.num_stages {
            return Err(TensorError::InvalidArgument {
                operation: "PipelineParallelModel::new".to_string(),
                reason: "Number of stage devices must match number of stages".to_string(),
                context: None,
            });
        }

        // Split layers into stages
        let layers_per_stage = layers.len() / config.num_stages;
        let mut stages = Vec::new();

        for stage_id in 0..config.num_stages {
            let start_idx = stage_id * layers_per_stage;
            let end_idx = if stage_id == config.num_stages - 1 {
                layers.len() // Last stage gets remaining layers
            } else {
                (stage_id + 1) * layers_per_stage
            };

            let layer_indices: Vec<usize> = (start_idx..end_idx).collect();

            let stage = PipelineStage {
                layer_indices,
                device: config.stage_devices[stage_id].clone(),
                stage_id,
            };

            stages.push(stage);
        }

        // Create communication channels between stages
        let mut stage_channels = Vec::new();
        for _ in 0..config.num_stages - 1 {
            let (sender, receiver) = mpsc::channel();
            stage_channels.push(StageChannel { sender, receiver });
        }

        Ok(Self {
            layers,
            stages,
            config,
            stage_channels,
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
        })
    }

    /// Execute forward pass through the pipeline
    pub fn forward_pipeline(&mut self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let scheduler = PipelineScheduler::new(
            self.config.num_stages,
            self.config.micro_batch_size,
            self.config.num_micro_batches,
            self.metrics.clone(),
        );

        // Split input into micro-batches
        let micro_batches = self.split_into_micro_batches(input)?;

        // Execute pipeline with micro-batches
        let results = self.execute_pipeline_forward(micro_batches, scheduler)?;

        // Combine results
        self.combine_micro_batch_results(results)
    }

    /// Split input tensor into micro-batches
    fn split_into_micro_batches(&self, input: &Tensor<T>) -> Result<Vec<MicroBatch<T>>> {
        let batch_size = input.shape().dims()[0];
        let num_micro_batches =
            (batch_size + self.config.micro_batch_size - 1) / self.config.micro_batch_size;

        let mut micro_batches = Vec::new();

        for i in 0..num_micro_batches {
            let start_idx = i * self.config.micro_batch_size;
            let end_idx = std::cmp::min(start_idx + self.config.micro_batch_size, batch_size);

            // Create slice of input for this micro-batch
            #[allow(clippy::single_range_in_vec_init)]
            let micro_batch_data = input.slice(&[start_idx..end_idx])?;

            let micro_batch = MicroBatch {
                data: micro_batch_data,
                micro_batch_id: i,
                batch_id: 0, // For now, single batch
                timestamp: Instant::now(),
            };

            micro_batches.push(micro_batch);
        }

        Ok(micro_batches)
    }

    /// Execute forward pass through pipeline stages
    fn execute_pipeline_forward(
        &mut self,
        micro_batches: Vec<MicroBatch<T>>,
        mut scheduler: PipelineScheduler,
    ) -> Result<Vec<MicroBatch<T>>> {
        let mut results = Vec::new();

        // For simplified implementation, execute sequentially for now
        // In a full implementation, this would use async execution across stages
        for micro_batch in micro_batches {
            let mut current_data = micro_batch.data;

            // Execute through each stage
            for stage in &self.stages {
                current_data = stage.forward(&current_data, &self.layers)?;
            }

            // Create result micro-batch
            let result = MicroBatch {
                data: current_data,
                micro_batch_id: micro_batch.micro_batch_id,
                batch_id: micro_batch.batch_id,
                timestamp: micro_batch.timestamp,
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Combine micro-batch results back into a single tensor
    fn combine_micro_batch_results(&self, mut results: Vec<MicroBatch<T>>) -> Result<Tensor<T>> {
        if results.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "combine_micro_batch_results".to_string(),
                reason: "No micro-batch results to combine".to_string(),
                context: None,
            });
        }

        // Sort by micro-batch ID to ensure correct order
        results.sort_by_key(|mb| mb.micro_batch_id);

        // Extract tensors
        let tensors: Vec<Tensor<T>> = results.into_iter().map(|mb| mb.data).collect();

        // Concatenate along batch dimension
        let tensor_refs: Vec<&Tensor<T>> = tensors.iter().collect();
        concat(&tensor_refs, 0)
    }

    /// Get pipeline metrics
    pub fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

impl PipelineStage {
    /// Execute forward pass through this stage
    pub fn forward<T>(
        &self,
        input: &Tensor<T>,
        all_layers: &[Box<dyn Layer<T>>],
    ) -> Result<Tensor<T>>
    where
        T: Clone,
    {
        let mut output = input.clone();

        // Execute each layer in the stage using layer indices
        for &layer_idx in &self.layer_indices {
            let layer = &all_layers[layer_idx];
            let layer_output = layer.forward(&output)?;
            output = layer_output;
        }

        Ok(output)
    }

    /// Get memory requirements for this stage
    pub fn memory_requirements(&self) -> usize {
        // Calculate memory requirements based on:
        // 1. Model parameters (weights, biases)
        // 2. Activation memory for intermediate results
        // 3. Gradient storage during backpropagation

        let mut total_memory = 0;

        // Base memory per layer (covers typical dense layer overhead)
        let base_memory_per_layer = 512 * 1024; // 512KB base overhead
        total_memory += self.layer_indices.len() * base_memory_per_layer;

        // Additional memory for pipeline buffer management
        let pipeline_overhead = 256 * 1024; // 256KB for stage coordination
        total_memory += pipeline_overhead;

        // Scale based on device placement (GPU layers need more memory)
        let device_multiplier = get_device_memory_multiplier(&self.device);

        total_memory * device_multiplier
    }
}

fn get_device_memory_multiplier(device: &tenflowers_core::Device) -> usize {
    match device {
        tenflowers_core::Device::Cpu => 1,
        #[cfg(feature = "gpu")]
        _ => 2, // GPU/ROCm tensors need extra memory for transfers
    }
}

impl PipelineScheduler {
    /// Create new pipeline scheduler
    pub fn new(
        num_stages: usize,
        micro_batch_size: usize,
        num_micro_batches: usize,
        metrics: Arc<Mutex<PipelineMetrics>>,
    ) -> Self {
        Self {
            num_stages,
            micro_batch_size,
            num_micro_batches,
            state: SchedulerState {
                current_batch: 0,
                current_micro_batch: 0,
                active_micro_batches: VecDeque::new(),
            },
            metrics,
        }
    }

    /// Schedule next micro-batch for execution
    pub fn schedule_next(&mut self) -> Option<usize> {
        if self.state.active_micro_batches.len() < self.num_micro_batches {
            let micro_batch_id = self.state.current_micro_batch;
            self.state.active_micro_batches.push_back(micro_batch_id);
            self.state.current_micro_batch += 1;
            Some(micro_batch_id)
        } else {
            None
        }
    }

    /// Mark micro-batch as completed
    pub fn complete_micro_batch(&mut self, micro_batch_id: usize) {
        self.state
            .active_micro_batches
            .retain(|&id| id != micro_batch_id);

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.forward_passes += 1;
    }
}

/// Pipeline model builder for easy construction
pub struct PipelineModelBuilder<T> {
    layers: Vec<Box<dyn Layer<T>>>,
    config: Option<PipelineConfig>,
}

impl<T> PipelineModelBuilder<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::One + Send + Sync + 'static,
{
    /// Create new builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            config: None,
        }
    }

    /// Add layer to the pipeline
    pub fn add_layer(mut self, layer: Box<dyn Layer<T>>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Set pipeline configuration
    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the pipeline model
    pub fn build(self) -> Result<PipelineParallelModel<T>>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let config = self.config.ok_or_else(|| TensorError::InvalidArgument {
            operation: "build".to_string(),
            reason: "Pipeline configuration is required".to_string(),
            context: None,
        })?;

        PipelineParallelModel::new(self.layers, config)
    }
}

impl<T> Default for PipelineModelBuilder<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::One + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PipelineMetrics {
    fn clone(&self) -> Self {
        Self {
            forward_passes: self.forward_passes,
            backward_passes: self.backward_passes,
            avg_stage_time: self.avg_stage_time.clone(),
            bubble_time: self.bubble_time,
            throughput: self.throughput,
            memory_usage: self.memory_usage.clone(),
        }
    }
}

/// Utility functions for pipeline parallelism
pub mod pipeline_utils {
    use super::*;

    /// Calculate optimal pipeline configuration
    pub fn calculate_optimal_pipeline_config(
        num_layers: usize,
        available_devices: &[Device],
        batch_size: usize,
    ) -> Result<PipelineConfig> {
        let num_stages = std::cmp::min(available_devices.len(), num_layers);
        let micro_batch_size = std::cmp::max(1, batch_size / 4); // 4 micro-batches by default
        let num_micro_batches = (batch_size + micro_batch_size - 1) / micro_batch_size;

        Ok(PipelineConfig {
            num_stages,
            micro_batch_size,
            num_micro_batches,
            stage_devices: available_devices[..num_stages].to_vec(),
        })
    }

    /// Estimate pipeline efficiency
    pub fn estimate_pipeline_efficiency(
        config: &PipelineConfig,
        layer_compute_times: &[Duration],
    ) -> f64 {
        if layer_compute_times.is_empty() {
            return 0.0;
        }

        // Calculate stage compute times
        let layers_per_stage = layer_compute_times.len() / config.num_stages;
        let mut stage_times = Vec::new();

        for stage in 0..config.num_stages {
            let start_idx = stage * layers_per_stage;
            let end_idx = if stage == config.num_stages - 1 {
                layer_compute_times.len()
            } else {
                (stage + 1) * layers_per_stage
            };

            let stage_time: Duration = layer_compute_times[start_idx..end_idx].iter().sum();
            stage_times.push(stage_time);
        }

        // Find bottleneck stage (slowest)
        let max_stage_time = stage_times.iter().max().unwrap();
        let total_sequential_time: Duration = layer_compute_times.iter().sum();

        // Pipeline efficiency = total work / (bottleneck * num_stages)
        // This represents how much of the pipeline's theoretical capacity is being used
        let pipeline_time = max_stage_time.as_secs_f64() * config.num_stages as f64;
        let sequential_time = total_sequential_time.as_secs_f64();

        if pipeline_time > 0.0 {
            sequential_time / pipeline_time
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tenflowers_core::Device;

    #[test]
    fn test_pipeline_config_creation() {
        let devices = vec![
            #[cfg(feature = "gpu")]
            Device::Gpu(0),
            #[cfg(not(feature = "gpu"))]
            Device::Cpu,
            #[cfg(feature = "gpu")]
            Device::Gpu(1),
            #[cfg(not(feature = "gpu"))]
            Device::Cpu,
        ];
        let config = PipelineConfig {
            num_stages: 2,
            micro_batch_size: 4,
            num_micro_batches: 8,
            stage_devices: devices,
        };

        assert_eq!(config.num_stages, 2);
        assert_eq!(config.micro_batch_size, 4);
    }

    #[test]
    fn test_micro_batch_creation() {
        let data: Tensor<f32> = Tensor::zeros(&[4, 10]); // Batch size 4, feature size 10
        let micro_batch = MicroBatch {
            data,
            micro_batch_id: 0,
            batch_id: 0,
            timestamp: Instant::now(),
        };

        assert_eq!(micro_batch.micro_batch_id, 0);
        assert_eq!(micro_batch.batch_id, 0);
    }

    #[test]
    fn test_pipeline_efficiency_calculation() {
        let config = PipelineConfig {
            num_stages: 2,
            micro_batch_size: 4,
            num_micro_batches: 4,
            stage_devices: vec![
                #[cfg(feature = "gpu")]
                Device::Gpu(0),
                #[cfg(not(feature = "gpu"))]
                Device::Cpu,
                #[cfg(feature = "gpu")]
                Device::Gpu(1),
                #[cfg(not(feature = "gpu"))]
                Device::Cpu,
            ],
        };

        let layer_times = vec![
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(20),
            Duration::from_millis(25),
        ];

        let efficiency = pipeline_utils::estimate_pipeline_efficiency(&config, &layer_times);
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }
}
