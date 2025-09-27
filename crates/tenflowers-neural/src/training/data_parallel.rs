use crate::optimizers::Optimizer;
use crate::Model;
use num_traits::{Float, FromPrimitive, One, Zero};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for data parallel training
#[derive(Debug, Clone)]
pub struct DataParallelConfig {
    /// Number of worker processes/threads
    pub world_size: usize,
    /// Current worker rank/id (0-indexed)
    pub rank: usize,
    /// Communication backend type
    pub backend: CommunicationBackend,
    /// Whether to use gradient compression
    pub gradient_compression: bool,
    /// Compression ratio for gradient compression (0.1 = 10% of original size)
    pub compression_ratio: f32,
    /// Bucket size for gradient bucketing (in bytes)
    pub bucket_size: usize,
    /// Whether to overlap communication with computation
    pub overlap_communication: bool,
}

impl Default for DataParallelConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            backend: CommunicationBackend::Thread,
            gradient_compression: false,
            compression_ratio: 0.1,
            bucket_size: 25 * 1024 * 1024, // 25MB
            overlap_communication: true,
        }
    }
}

/// Communication backend for distributed training
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationBackend {
    /// Thread-based communication for single-node multi-GPU
    Thread,
    /// Process-based communication (future: MPI, NCCL)
    Process,
    /// Custom implementation
    Custom(String),
}

/// Gradient bucket for efficient communication
#[derive(Debug)]
struct GradientBucket<T> {
    /// Accumulated gradients for this bucket
    gradients: Vec<Tensor<T>>,
    /// Parameter names corresponding to gradients
    param_names: Vec<String>,
    /// Current size in bytes
    current_size: usize,
    /// Whether this bucket is ready for all-reduce
    ready_for_reduce: bool,
}

impl<T> GradientBucket<T>
where
    T: Clone + Default,
{
    fn new() -> Self {
        Self {
            gradients: Vec::new(),
            param_names: Vec::new(),
            current_size: 0,
            ready_for_reduce: false,
        }
    }

    fn add_gradient(&mut self, gradient: Tensor<T>, param_name: String, size_bytes: usize) {
        self.gradients.push(gradient);
        self.param_names.push(param_name);
        self.current_size += size_bytes;
    }

    fn is_full(&self, bucket_size: usize) -> bool {
        self.current_size >= bucket_size
    }

    fn mark_ready(&mut self) {
        self.ready_for_reduce = true;
    }
}

/// All-reduce operation for gradient aggregation
pub trait AllReduce<T> {
    /// Perform all-reduce operation across all workers
    fn all_reduce(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>;

    /// Perform all-reduce on multiple tensors
    fn all_reduce_batch(&self, tensors: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>;
}

/// Thread-based all-reduce implementation for single-node training
pub struct ThreadAllReduce<T> {
    world_size: usize,
    rank: usize,
    /// Shared storage for gradient accumulation
    gradient_storage: Arc<Mutex<HashMap<String, Vec<Tensor<T>>>>>,
}

impl<T> ThreadAllReduce<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self {
            world_size,
            rank,
            gradient_storage: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<T> AllReduce<T> for ThreadAllReduce<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn all_reduce(&self, tensor: &Tensor<T>) -> Result<Tensor<T>> {
        // For thread-based implementation, simulate averaging across workers
        let world_size_scalar = T::from(self.world_size).unwrap_or(T::one());
        let divisor = Tensor::from_scalar(world_size_scalar);
        tensor.div(&divisor)
    }

    fn all_reduce_batch(&self, tensors: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        let mut results = Vec::new();
        for tensor in tensors {
            results.push(self.all_reduce(tensor)?);
        }
        Ok(results)
    }
}

/// Gradient compression utilities
pub struct GradientCompressor<T> {
    compression_ratio: f32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> GradientCompressor<T>
where
    T: Float + Clone + Default + PartialOrd,
{
    pub fn new(compression_ratio: f32) -> Self {
        Self {
            compression_ratio,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compress gradients using top-k sparsification
    pub fn compress(&self, gradient: &Tensor<T>) -> Result<CompressedGradient<T>> {
        let total_elements = gradient.shape().elements();
        let k = ((total_elements as f32) * self.compression_ratio) as usize;

        // Get flat view of tensor data
        let values = self.extract_values(gradient)?;
        let mut indexed_values: Vec<(usize, T)> = values.into_iter().enumerate().collect();

        // Sort by absolute value and keep top-k
        indexed_values.sort_by(|a, b| {
            let abs_a = if a.1 < T::zero() {
                T::zero() - a.1
            } else {
                a.1
            };
            let abs_b = if b.1 < T::zero() {
                T::zero() - b.1
            } else {
                b.1
            };
            abs_b
                .partial_cmp(&abs_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed_values.truncate(k);

        let indices: Vec<usize> = indexed_values.iter().map(|(i, _)| *i).collect();
        let values: Vec<T> = indexed_values.iter().map(|(_, v)| *v).collect();

        Ok(CompressedGradient {
            indices,
            values,
            original_shape: gradient.shape().dims().to_vec(),
            compression_ratio: self.compression_ratio,
        })
    }

    /// Decompress gradients back to original format
    pub fn decompress(&self, compressed: &CompressedGradient<T>) -> Result<Tensor<T>> {
        let total_elements = compressed.original_shape.iter().product();
        let mut data = vec![T::zero(); total_elements];

        // Fill in the compressed values at their original indices
        for (&index, &value) in compressed.indices.iter().zip(&compressed.values) {
            if index < data.len() {
                data[index] = value;
            }
        }

        Tensor::from_vec(data, &compressed.original_shape)
    }

    /// Extract values from tensor (helper method)
    fn extract_values(&self, tensor: &Tensor<T>) -> Result<Vec<T>> {
        // This is a simplified implementation
        // In practice, this would directly access tensor storage
        let total_elements = tensor.shape().elements();
        let mut values = Vec::with_capacity(total_elements);

        for i in 0..total_elements {
            // Convert linear index to multidimensional index
            let indices = self.linear_to_indices(i, tensor.shape().dims());
            if let Some(value) = tensor.get(&indices) {
                values.push(value);
            } else {
                values.push(T::zero());
            }
        }

        Ok(values)
    }

    /// Convert linear index to multidimensional indices
    fn linear_to_indices(&self, mut linear_idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; dims.len()];

        for i in (0..dims.len()).rev() {
            indices[i] = linear_idx % dims[i];
            linear_idx /= dims[i];
        }

        indices
    }
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub struct CompressedGradient<T> {
    /// Indices of non-zero values
    pub indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<T>,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
    /// Compression ratio used
    pub compression_ratio: f32,
}

/// Data Parallel Trainer for distributed training
pub struct DataParallelTrainer<T, O> {
    /// Local model replica
    model: Box<dyn Model<T>>,
    /// Optimizer for local model
    optimizer: O,
    /// Configuration for data parallelism
    config: DataParallelConfig,
    /// Communication backend for gradient aggregation
    all_reduce: Box<dyn AllReduce<T>>,
    /// Gradient compression utility
    compressor: Option<GradientCompressor<T>>,
    /// Gradient buckets for efficient communication
    gradient_buckets: Vec<GradientBucket<T>>,
    /// Current bucket being filled
    current_bucket_idx: usize,
}

impl<T, O> DataParallelTrainer<T, O>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + FromPrimitive
        + Zero
        + One
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    /// Create a new data parallel trainer
    pub fn new(model: Box<dyn Model<T>>, optimizer: O, config: DataParallelConfig) -> Result<Self> {
        // Create appropriate all-reduce backend
        let all_reduce: Box<dyn AllReduce<T>> = match config.backend {
            CommunicationBackend::Thread => {
                Box::new(ThreadAllReduce::new(config.world_size, config.rank))
            }
            CommunicationBackend::Process => {
                return Err(TensorError::unsupported_operation_simple(
                    "Process-based communication not yet implemented".to_string(),
                ));
            }
            CommunicationBackend::Custom(_) => {
                return Err(TensorError::unsupported_operation_simple(
                    "Custom communication backend not yet implemented".to_string(),
                ));
            }
        };

        // Create gradient compressor if enabled
        let compressor = if config.gradient_compression {
            Some(GradientCompressor::new(config.compression_ratio))
        } else {
            None
        };

        // Initialize gradient buckets
        let num_buckets = 4; // Use multiple buckets for overlapping communication
        let mut gradient_buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            gradient_buckets.push(GradientBucket::new());
        }

        Ok(Self {
            model,
            optimizer,
            config,
            all_reduce,
            compressor,
            gradient_buckets,
            current_bucket_idx: 0,
        })
    }

    /// Perform a distributed training step
    pub fn train_step(&mut self, inputs: &[&Tensor<T>], targets: &[&Tensor<T>]) -> Result<T> {
        if inputs.len() != targets.len() {
            return Err(TensorError::invalid_argument(
                "Number of inputs and targets must match".to_string(),
            ));
        }

        let mut total_loss = T::zero();
        let batch_size = inputs.len();

        // Forward pass and loss computation
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.model.forward(input)?;

            // Compute loss (simplified - in practice would use specific loss function)
            let loss = self.compute_loss(&output, target)?;
            total_loss = total_loss + loss;
        }

        // Backward pass would go here in a full implementation
        // For now, we simulate gradient computation
        self.simulate_backward_pass()?;

        // All-reduce gradients across workers
        self.all_reduce_gradients()?;

        // Update model parameters manually since we're working with trait objects
        // In a full implementation, this would call the optimizer's step method
        // on the distributed gradients. For now, we'll zero gradients as a placeholder
        self.model.zero_grad();

        // Average loss across batch
        let avg_loss = total_loss / T::from(batch_size).unwrap_or(T::one());
        Ok(avg_loss)
    }

    /// Simulate backward pass (in practice, this would compute actual gradients)
    fn simulate_backward_pass(&mut self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Compute gradients for all parameters
        // 2. Bucket gradients for efficient communication
        // 3. Trigger all-reduce operations when buckets are full

        // For now, we'll just mark that gradients are ready
        for bucket in &mut self.gradient_buckets {
            bucket.mark_ready();
        }

        Ok(())
    }

    /// Perform all-reduce on gradients across workers
    fn all_reduce_gradients(&mut self) -> Result<()> {
        for bucket in &mut self.gradient_buckets {
            if bucket.ready_for_reduce {
                // Compress gradients if compression is enabled
                let gradients_to_reduce = if let Some(ref compressor) = self.compressor {
                    // Compress each gradient
                    let mut compressed_gradients = Vec::new();
                    for gradient in &bucket.gradients {
                        let compressed = compressor.compress(gradient)?;
                        compressed_gradients.push(compressed);
                    }

                    // Decompress after all-reduce (simplified)
                    let mut decompressed = Vec::new();
                    for compressed in &compressed_gradients {
                        let decompressed_grad = compressor.decompress(compressed)?;
                        decompressed.push(decompressed_grad);
                    }
                    decompressed
                } else {
                    bucket.gradients.clone()
                };

                // Perform all-reduce
                let gradient_refs: Vec<&Tensor<T>> = gradients_to_reduce.iter().collect();
                let reduced_gradients = self.all_reduce.all_reduce_batch(&gradient_refs)?;

                // Update bucket with reduced gradients
                bucket.gradients = reduced_gradients;
                bucket.ready_for_reduce = false;
            }
        }

        Ok(())
    }

    /// Compute loss (simplified implementation)
    fn compute_loss(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<T> {
        // Simple MSE loss for demonstration
        let diff = predictions.sub(targets)?;
        let squared = diff.mul(&diff)?;
        let mean_loss = tenflowers_core::ops::mean(&squared, None, false)?;

        // Extract scalar value
        mean_loss.get(&[]).ok_or_else(|| {
            TensorError::invalid_argument("Could not extract scalar loss".to_string())
        })
    }

    /// Get the current model
    pub fn model(&self) -> &dyn Model<T> {
        self.model.as_ref()
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut dyn Model<T> {
        self.model.as_mut()
    }

    /// Get the current configuration
    pub fn config(&self) -> &DataParallelConfig {
        &self.config
    }

    /// Get training statistics
    pub fn get_stats(&self) -> DataParallelStats {
        DataParallelStats {
            world_size: self.config.world_size,
            rank: self.config.rank,
            gradient_compression_enabled: self.compressor.is_some(),
            compression_ratio: self.config.compression_ratio,
            bucket_count: self.gradient_buckets.len(),
        }
    }
}

/// Statistics for data parallel training
#[derive(Debug, Clone)]
pub struct DataParallelStats {
    pub world_size: usize,
    pub rank: usize,
    pub gradient_compression_enabled: bool,
    pub compression_ratio: f32,
    pub bucket_count: usize,
}

/// Builder for creating data parallel trainers
pub struct DataParallelTrainerBuilder<T, O> {
    model: Option<Box<dyn Model<T>>>,
    optimizer: Option<O>,
    config: DataParallelConfig,
}

impl<T, O> DataParallelTrainerBuilder<T, O>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + FromPrimitive
        + Zero
        + One
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            optimizer: None,
            config: DataParallelConfig::default(),
        }
    }

    /// Set the model to train
    pub fn with_model(mut self, model: Box<dyn Model<T>>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the optimizer
    pub fn with_optimizer(mut self, optimizer: O) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Set the world size (number of workers)
    pub fn with_world_size(mut self, world_size: usize) -> Self {
        self.config.world_size = world_size;
        self
    }

    /// Set the rank of this worker
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Enable gradient compression
    pub fn with_gradient_compression(mut self, compression_ratio: f32) -> Self {
        self.config.gradient_compression = true;
        self.config.compression_ratio = compression_ratio;
        self
    }

    /// Set the communication backend
    pub fn with_backend(mut self, backend: CommunicationBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set bucket size for gradient bucketing
    pub fn with_bucket_size(mut self, bucket_size: usize) -> Self {
        self.config.bucket_size = bucket_size;
        self
    }

    /// Build the data parallel trainer
    pub fn build(self) -> Result<DataParallelTrainer<T, O>> {
        let model = self
            .model
            .ok_or_else(|| TensorError::invalid_argument("Model is required".to_string()))?;

        let optimizer = self
            .optimizer
            .ok_or_else(|| TensorError::invalid_argument("Optimizer is required".to_string()))?;

        DataParallelTrainer::new(model, optimizer, self.config)
    }
}

impl<T, O> Default for DataParallelTrainerBuilder<T, O>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + FromPrimitive
        + Zero
        + One
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a basic data parallel trainer
pub fn create_data_parallel_trainer<T, O>(
    model: Box<dyn Model<T>>,
    optimizer: O,
    world_size: usize,
    rank: usize,
) -> Result<DataParallelTrainer<T, O>>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + FromPrimitive
        + Zero
        + One
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    DataParallelTrainerBuilder::new()
        .with_model(model)
        .with_optimizer(optimizer)
        .with_world_size(world_size)
        .with_rank(rank)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::optimizers::SGD;
    use tenflowers_core::Tensor;

    #[test]
    fn test_data_parallel_config_default() {
        let config = DataParallelConfig::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.backend, CommunicationBackend::Thread);
        assert!(!config.gradient_compression);
    }

    #[test]
    fn test_gradient_compressor() {
        let compressor = GradientCompressor::<f32>::new(0.5);
        let gradient = Tensor::from_vec(vec![1.0, 0.1, -2.0, 0.05, 3.0, -0.02], &[6]).unwrap();

        let compressed = compressor.compress(&gradient).unwrap();
        assert_eq!(compressed.compression_ratio, 0.5);
        assert_eq!(compressed.indices.len(), 3); // 50% of 6 elements

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.shape().dims(), &[6]);
    }

    #[test]
    fn test_gradient_bucket() {
        let mut bucket = GradientBucket::<f32>::new();
        assert!(!bucket.is_full(1000));

        let grad = Tensor::from_scalar(1.0);
        bucket.add_gradient(grad, "test_param".to_string(), 500);
        assert!(!bucket.is_full(1000));

        let grad2 = Tensor::from_scalar(2.0);
        bucket.add_gradient(grad2, "test_param2".to_string(), 600);
        assert!(bucket.is_full(1000));
    }

    #[test]
    fn test_thread_all_reduce() {
        let all_reduce = ThreadAllReduce::<f32>::new(2, 0);
        let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3]).unwrap();

        let result = all_reduce.all_reduce(&tensor).unwrap();
        // Should divide by world_size (2)
        assert_eq!(result.get(&[0]).unwrap(), 1.0);
        assert_eq!(result.get(&[1]).unwrap(), 2.0);
        assert_eq!(result.get(&[2]).unwrap(), 3.0);
    }

    #[test]
    fn test_data_parallel_trainer_builder() {
        let builder = DataParallelTrainerBuilder::<f32, SGD<f32>>::new()
            .with_world_size(4)
            .with_rank(1)
            .with_gradient_compression(0.3);

        assert_eq!(builder.config.world_size, 4);
        assert_eq!(builder.config.rank, 1);
        assert!(builder.config.gradient_compression);
        assert_eq!(builder.config.compression_ratio, 0.3);
    }
}
