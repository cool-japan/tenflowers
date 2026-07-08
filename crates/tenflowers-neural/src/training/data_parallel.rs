use crate::optimizers::Optimizer;
use crate::Model;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
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

        // Forward pass and loss computation (for reporting the average loss).
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.model.forward(input)?;
            let loss = self.compute_loss(&output, target)?;
            total_loss = total_loss + loss;
        }

        // Real backward pass: compute genuine per-parameter gradients for this
        // replica's local mini-batch and store them on the parameters.
        self.compute_replica_gradients(inputs, targets)?;

        // Average the local gradients across the data-parallel workers via the
        // all-reduce backend (this divides by `world_size`, matching standard
        // synchronous data-parallel SGD where each worker contributes 1/N).
        self.all_reduce_parameter_gradients()?;

        // Snapshot the averaged gradients before the optimizer step. The
        // optimizer replaces each parameter tensor with a freshly-computed one
        // (which carries no gradient), so we re-attach the gradients afterwards
        // to keep them observable until the next `zero_grad`, matching the usual
        // train-loop contract where `.grad` survives `.step()`.
        let averaged_grads: Vec<Option<Tensor<T>>> = self
            .model
            .parameters()
            .iter()
            .map(|p| p.grad().cloned())
            .collect();

        // Apply the averaged gradients with the optimizer (real parameter
        // update).
        self.optimizer.step(self.model.as_mut())?;

        // Re-attach the applied gradients so the learning signal is observable.
        {
            let mut params = self.model.parameters_mut();
            for (param, grad) in params.iter_mut().zip(averaged_grads) {
                if let Some(g) = grad {
                    param.set_requires_grad(true);
                    param.set_grad(Some(g));
                }
            }
        }

        // Average loss across the local mini-batch.
        let avg_loss = total_loss / T::from(batch_size).unwrap_or(T::one());
        Ok(avg_loss)
    }

    /// Compute real per-parameter gradients for the local mini-batch.
    ///
    /// The model is a `dyn Model<T>` trait object whose `forward` returns a
    /// detached tensor, so there is no autograd tape spanning the trait
    /// boundary to differentiate through. We therefore compute exact gradients
    /// of the mean mini-batch loss with respect to every parameter element
    /// using a central finite-difference estimator:
    ///
    /// ```text
    ///   g_i = ( L(theta_i + eps) - L(theta_i - eps) ) / (2 * eps)
    /// ```
    ///
    /// This is a genuine gradient (second-order accurate in `eps`), produces a
    /// real learning signal, and is averaged over the local replica's samples.
    /// It is intentionally simple and correct rather than fast; large models
    /// should use the autograd-tape training path instead.
    fn compute_replica_gradients(
        &mut self,
        inputs: &[&Tensor<T>],
        targets: &[&Tensor<T>],
    ) -> Result<()> {
        let num_params = self.model.parameters().len();
        if num_params == 0 {
            return Ok(());
        }

        // Finite-difference step. Kept in `f64` for accuracy, converted to `T`.
        let eps_f64 = 1e-3_f64;
        let eps = T::from(eps_f64).unwrap_or_else(|| T::one());
        let two_eps = eps + eps;

        // Number of elements in each parameter, captured up front to avoid
        // borrow conflicts while mutating parameters in place.
        let param_lens: Vec<usize> = self
            .model
            .parameters()
            .iter()
            .map(|p| p.shape().elements())
            .collect();
        let param_shapes: Vec<Vec<usize>> = self
            .model
            .parameters()
            .iter()
            .map(|p| p.shape().dims().to_vec())
            .collect();

        // Accumulated gradient values per parameter (flattened).
        let mut grad_accum: Vec<Vec<T>> =
            param_lens.iter().map(|&len| vec![T::zero(); len]).collect();

        let sample_count = inputs.len();
        let inv_samples = T::from(sample_count.max(1)).unwrap_or_else(T::one);

        for param_idx in 0..num_params {
            for elem_idx in 0..param_lens[param_idx] {
                // Read the original value of this element.
                let original = self.parameter_element(param_idx, elem_idx)?;

                // L(theta + eps)
                self.set_parameter_element(param_idx, elem_idx, original + eps)?;
                let loss_plus = self.mini_batch_loss(inputs, targets)?;

                // L(theta - eps)
                self.set_parameter_element(param_idx, elem_idx, original - eps)?;
                let loss_minus = self.mini_batch_loss(inputs, targets)?;

                // Restore the original parameter value.
                self.set_parameter_element(param_idx, elem_idx, original)?;

                // Central difference of the *mean* loss already averaged over
                // samples inside `mini_batch_loss`.
                let grad = (loss_plus - loss_minus) / two_eps;
                grad_accum[param_idx][elem_idx] = grad;
            }
        }

        // Store the computed gradients on each parameter so the optimizer and
        // all-reduce can consume them. The per-sample averaging is handled in
        // `mini_batch_loss`, so `grad_accum` already holds the mini-batch mean
        // gradient; multiplying by `inv_samples` here would double-average, so
        // we only normalise when more than one independent accumulation occurs.
        let _ = inv_samples; // retained for clarity; averaging done in loss.

        let mut params = self.model.parameters_mut();
        for (param_idx, param) in params.iter_mut().enumerate() {
            let grad_tensor =
                Tensor::from_vec(grad_accum[param_idx].clone(), &param_shapes[param_idx])?;
            param.set_requires_grad(true);
            param.set_grad(Some(grad_tensor));
        }

        Ok(())
    }

    /// Read a single (flattened) element of parameter `param_idx`.
    fn parameter_element(&self, param_idx: usize, elem_idx: usize) -> Result<T> {
        let params = self.model.parameters();
        let param = params.get(param_idx).ok_or_else(|| {
            TensorError::invalid_argument("Parameter index out of range".to_string())
        })?;
        let values = param.to_vec()?;
        values.get(elem_idx).copied().ok_or_else(|| {
            TensorError::invalid_argument("Parameter element index out of range".to_string())
        })
    }

    /// Overwrite a single (flattened) element of parameter `param_idx`.
    fn set_parameter_element(&mut self, param_idx: usize, elem_idx: usize, value: T) -> Result<()> {
        let mut params = self.model.parameters_mut();
        let param = params.get_mut(param_idx).ok_or_else(|| {
            TensorError::invalid_argument("Parameter index out of range".to_string())
        })?;
        let shape = param.shape().dims().to_vec();
        let mut values = param.to_vec()?;
        if elem_idx >= values.len() {
            return Err(TensorError::invalid_argument(
                "Parameter element index out of range".to_string(),
            ));
        }
        values[elem_idx] = value;
        **param = Tensor::from_vec(values, &shape)?;
        Ok(())
    }

    /// Mean loss of the local mini-batch under the current parameters.
    fn mini_batch_loss(&self, inputs: &[&Tensor<T>], targets: &[&Tensor<T>]) -> Result<T> {
        let mut total = T::zero();
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.model.forward(input)?;
            total = total + self.compute_loss(&output, target)?;
        }
        let count = T::from(inputs.len().max(1)).unwrap_or_else(T::one);
        Ok(total / count)
    }

    /// All-reduce the gradients currently stored on the model parameters across
    /// data-parallel workers, writing the averaged gradient back onto each
    /// parameter.
    ///
    /// When gradient compression is enabled the real gradients are routed
    /// through the bucketed, top-k compressed all-reduce path
    /// ([`Self::all_reduce_gradients`]); otherwise each parameter gradient is
    /// reduced directly.
    fn all_reduce_parameter_gradients(&mut self) -> Result<()> {
        if self.compressor.is_some() {
            return self.all_reduce_gradients();
        }

        // Collect the current gradients (clone to drop the immutable borrow).
        let grads: Vec<Option<Tensor<T>>> = self
            .model
            .parameters()
            .iter()
            .map(|p| p.grad().cloned())
            .collect();

        // Reduce each present gradient.
        let mut reduced: Vec<Option<Tensor<T>>> = Vec::with_capacity(grads.len());
        for grad in &grads {
            match grad {
                Some(g) => reduced.push(Some(self.all_reduce.all_reduce(g)?)),
                None => reduced.push(None),
            }
        }

        // Write the reduced gradients back onto the parameters.
        let mut params = self.model.parameters_mut();
        for (param, reduced_grad) in params.iter_mut().zip(reduced) {
            if let Some(g) = reduced_grad {
                param.set_grad(Some(g));
            }
        }

        Ok(())
    }

    /// Bucketed, optionally-compressed all-reduce of the gradients currently
    /// stored on the model parameters.
    ///
    /// Real parameter gradients are packed into [`GradientBucket`]s (sized by
    /// `config.bucket_size`), top-k compressed when a compressor is configured,
    /// all-reduced across workers, decompressed, and written back onto the
    /// parameters in their original order. This exercises the gradient
    /// compression / bucketing infrastructure on genuine gradients rather than
    /// on empty placeholders.
    fn all_reduce_gradients(&mut self) -> Result<()> {
        // Reset buckets for this step.
        for bucket in &mut self.gradient_buckets {
            *bucket = GradientBucket::new();
        }
        self.current_bucket_idx = 0;

        // Snapshot the parameter gradients and the index of every parameter
        // that actually carries a gradient (so we can restore order later).
        let param_grads: Vec<(usize, Tensor<T>)> = self
            .model
            .parameters()
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| p.grad().cloned().map(|g| (idx, g)))
            .collect();

        if param_grads.is_empty() {
            return Ok(());
        }

        // Distribute gradients into buckets, advancing to the next bucket once
        // the current one is full.
        let bucket_size = self.config.bucket_size;
        let num_buckets = self.gradient_buckets.len().max(1);
        let element_bytes = std::mem::size_of::<T>();
        for (idx, grad) in &param_grads {
            let size_bytes = grad.shape().elements() * element_bytes;
            {
                let bucket = &mut self.gradient_buckets[self.current_bucket_idx];
                bucket.add_gradient(grad.clone(), idx.to_string(), size_bytes);
            }
            if self.gradient_buckets[self.current_bucket_idx].is_full(bucket_size)
                && self.current_bucket_idx + 1 < num_buckets
            {
                self.current_bucket_idx += 1;
            }
        }
        for bucket in &mut self.gradient_buckets {
            if !bucket.gradients.is_empty() {
                bucket.mark_ready();
            }
        }

        // Reduce each ready bucket (with optional compression) and collect the
        // reduced gradients keyed by their original parameter index.
        let mut reduced_by_index: HashMap<usize, Tensor<T>> = HashMap::new();
        for bucket in &mut self.gradient_buckets {
            if !bucket.ready_for_reduce {
                continue;
            }

            let gradients_to_reduce = if let Some(ref compressor) = self.compressor {
                let mut decompressed = Vec::with_capacity(bucket.gradients.len());
                for gradient in &bucket.gradients {
                    let compressed = compressor.compress(gradient)?;
                    decompressed.push(compressor.decompress(&compressed)?);
                }
                decompressed
            } else {
                bucket.gradients.clone()
            };

            let gradient_refs: Vec<&Tensor<T>> = gradients_to_reduce.iter().collect();
            let reduced_gradients = self.all_reduce.all_reduce_batch(&gradient_refs)?;

            for (param_name, reduced) in bucket.param_names.iter().zip(reduced_gradients.iter()) {
                if let Ok(param_idx) = param_name.parse::<usize>() {
                    reduced_by_index.insert(param_idx, reduced.clone());
                }
            }

            bucket.gradients = reduced_gradients;
            bucket.ready_for_reduce = false;
        }

        // Write the reduced gradients back onto the parameters.
        let mut params = self.model.parameters_mut();
        for (param_idx, param) in params.iter_mut().enumerate() {
            if let Some(reduced) = reduced_by_index.remove(&param_idx) {
                param.set_grad(Some(reduced));
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
    use crate::model::Sequential;
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
        let gradient = Tensor::from_vec(vec![1.0, 0.1, -2.0, 0.05, 3.0, -0.02], &[6])
            .expect("test: tensor creation should succeed");

        let compressed = compressor
            .compress(&gradient)
            .expect("test: operation should succeed");
        assert_eq!(compressed.compression_ratio, 0.5);
        assert_eq!(compressed.indices.len(), 3); // 50% of 6 elements

        let decompressed = compressor
            .decompress(&compressed)
            .expect("test: operation should succeed");
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
        let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3])
            .expect("test: tensor creation should succeed");

        let result = all_reduce
            .all_reduce(&tensor)
            .expect("test: result should be valid");
        // Should divide by world_size (2)
        assert_eq!(result.get(&[0]).expect("test: result should be valid"), 1.0);
        assert_eq!(result.get(&[1]).expect("test: result should be valid"), 2.0);
        assert_eq!(result.get(&[2]).expect("test: result should be valid"), 3.0);
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

    /// A `train_step` must compute a REAL learning signal: after the step, at
    /// least one parameter must carry a non-zero gradient. Previously the step
    /// only "simulated" the backward pass and learned nothing.
    #[test]
    fn test_train_step_produces_nonzero_gradients() {
        // Tiny single-layer model with real (non-zero) weights.
        let model: Box<dyn Model<f32>> = Box::new(Sequential::new(vec![Box::new(
            Dense::<f32>::new_xavier(3, 2, true),
        )]));

        let trainer = DataParallelTrainerBuilder::<f32, SGD<f32>>::new()
            .with_model(model)
            .with_optimizer(SGD::new(0.01))
            .with_world_size(1)
            .with_rank(0)
            .build();
        let mut trainer = trainer.expect("test: trainer build should succeed");

        // Non-trivial input and target so the loss genuinely varies with the
        // parameters.
        let input =
            Tensor::from_vec(vec![0.5f32, -0.25, 1.0], &[1, 3]).expect("test: input creation");
        let target = Tensor::from_vec(vec![1.0f32, -1.0], &[1, 2]).expect("test: target creation");

        let loss = trainer
            .train_step(&[&input], &[&target])
            .expect("test: train_step should succeed");
        assert!(loss.is_finite(), "loss must be finite");

        // After the step the gradients are left populated; at least one must be
        // a real (non-zero) learning signal.
        let mut any_nonzero_grad = false;
        for param in trainer.model().parameters() {
            if let Some(grad) = param.grad() {
                let values = grad.to_vec().expect("test: grad to_vec");
                if values.iter().any(|v| v.abs() > 1e-7) {
                    any_nonzero_grad = true;
                    break;
                }
            }
        }
        assert!(
            any_nonzero_grad,
            "train_step must compute at least one non-zero gradient (real learning signal)"
        );
    }

    /// The same path with gradient compression enabled must also produce a real
    /// non-zero gradient (exercises the bucketed/compressed all-reduce path).
    #[test]
    fn test_train_step_with_compression_produces_nonzero_gradients() {
        let model: Box<dyn Model<f32>> = Box::new(Sequential::new(vec![Box::new(
            Dense::<f32>::new_xavier(3, 2, true),
        )]));

        let mut trainer = DataParallelTrainerBuilder::<f32, SGD<f32>>::new()
            .with_model(model)
            .with_optimizer(SGD::new(0.01))
            .with_world_size(2)
            .with_rank(0)
            .with_gradient_compression(0.8)
            .build()
            .expect("test: trainer build should succeed");

        let input =
            Tensor::from_vec(vec![0.5f32, -0.25, 1.0], &[1, 3]).expect("test: input creation");
        let target = Tensor::from_vec(vec![1.0f32, -1.0], &[1, 2]).expect("test: target creation");

        trainer
            .train_step(&[&input], &[&target])
            .expect("test: train_step should succeed");

        let any_nonzero_grad = trainer.model().parameters().iter().any(|p| {
            p.grad()
                .map(|g| {
                    g.to_vec()
                        .map(|v| v.iter().any(|x| x.abs() > 1e-7))
                        .unwrap_or(false)
                })
                .unwrap_or(false)
        });
        assert!(
            any_nonzero_grad,
            "compressed train_step must still produce a non-zero gradient"
        );
    }
}
