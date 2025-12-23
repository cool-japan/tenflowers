//! Memory-Efficient Gradient Computation
//!
//! This module provides memory-efficient implementations of gradient computation
//! techniques including gradient checkpointing, memory pooling, and lazy evaluation.

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Memory pool for reusing tensor allocations during gradient computation
pub struct GradientMemoryPool<T> {
    available_tensors: HashMap<Vec<usize>, VecDeque<Tensor<T>>>,
    max_pool_size: usize,
    total_allocated: usize,
}

impl<T> GradientMemoryPool<T>
where
    T: Clone + Default + Send + Sync + 'static + scirs2_core::num_traits::Zero,
{
    /// Create a new gradient memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            available_tensors: HashMap::new(),
            max_pool_size,
            total_allocated: 0,
        }
    }

    /// Get a tensor from the pool or create a new one
    pub fn get_tensor(&mut self, shape: &[usize]) -> Tensor<T> {
        let shape_vec = shape.to_vec();

        if let Some(tensor_queue) = self.available_tensors.get_mut(&shape_vec) {
            if let Some(tensor) = tensor_queue.pop_front() {
                return tensor;
            }
        }

        // Create new tensor if none available in pool
        self.total_allocated += 1;
        Tensor::zeros(shape)
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(&mut self, tensor: Tensor<T>) {
        let shape = tensor.shape().dims().to_vec();

        let tensor_queue = self.available_tensors.entry(shape).or_default();

        if tensor_queue.len() < self.max_pool_size {
            tensor_queue.push_back(tensor);
        }
        // If pool is full, tensor will be dropped
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let total_pooled: usize = self
            .available_tensors
            .values()
            .map(|queue| queue.len())
            .sum();

        MemoryPoolStats {
            total_allocated: self.total_allocated,
            total_pooled,
            pool_hit_ratio: if self.total_allocated > 0 {
                total_pooled as f64 / self.total_allocated as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the memory pool
    pub fn clear(&mut self) {
        self.available_tensors.clear();
        self.total_allocated = 0;
    }
}

/// Statistics for memory pool usage
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated: usize,
    pub total_pooled: usize,
    pub pool_hit_ratio: f64,
}

/// Gradient checkpointing manager for memory-efficient backpropagation
pub struct GradientCheckpointer<T> {
    checkpoints: HashMap<String, CheckpointData<T>>,
    memory_budget: usize,
    current_memory_usage: usize,
}

/// Data stored at a checkpoint
#[derive(Clone)]
struct CheckpointData<T> {
    tensor: Tensor<T>,
    computation_cost: f64,
    memory_size: usize,
    last_accessed: std::time::Instant,
}

impl<T> GradientCheckpointer<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new gradient checkpointer with memory budget
    pub fn new(memory_budget: usize) -> Self {
        Self {
            checkpoints: HashMap::new(),
            memory_budget,
            current_memory_usage: 0,
        }
    }

    /// Store a checkpoint with estimated computation cost
    pub fn store_checkpoint(
        &mut self,
        name: &str,
        tensor: Tensor<T>,
        computation_cost: f64,
    ) -> Result<()> {
        let memory_size = self.estimate_tensor_memory_size(&tensor);

        // Evict checkpoints if needed to fit within budget
        while self.current_memory_usage + memory_size > self.memory_budget
            && !self.checkpoints.is_empty()
        {
            self.evict_least_valuable_checkpoint();
        }

        let checkpoint_data = CheckpointData {
            tensor,
            computation_cost,
            memory_size,
            last_accessed: std::time::Instant::now(),
        };

        if let Some(old_data) = self.checkpoints.insert(name.to_string(), checkpoint_data) {
            self.current_memory_usage -= old_data.memory_size;
        }

        self.current_memory_usage += memory_size;

        Ok(())
    }

    /// Retrieve a checkpoint
    pub fn get_checkpoint(&mut self, name: &str) -> Option<Tensor<T>> {
        if let Some(data) = self.checkpoints.get_mut(name) {
            data.last_accessed = std::time::Instant::now();
            Some(data.tensor.clone())
        } else {
            None
        }
    }

    /// Check if a checkpoint exists
    pub fn has_checkpoint(&self, name: &str) -> bool {
        self.checkpoints.contains_key(name)
    }

    /// Evict the least valuable checkpoint based on computation cost and access time
    fn evict_least_valuable_checkpoint(&mut self) {
        let mut least_valuable_key = None;
        let mut least_value_score = f64::INFINITY;

        let now = std::time::Instant::now();

        for (key, data) in &self.checkpoints {
            let time_since_access = now.duration_since(data.last_accessed).as_secs_f64();
            // Higher computation cost and recent access make checkpoints more valuable
            let value_score = data.computation_cost / (time_since_access + 1.0);

            if value_score < least_value_score {
                least_value_score = value_score;
                least_valuable_key = Some(key.clone());
            }
        }

        if let Some(key) = least_valuable_key {
            if let Some(removed_data) = self.checkpoints.remove(&key) {
                self.current_memory_usage -= removed_data.memory_size;
            }
        }
    }

    /// Estimate memory size of a tensor
    fn estimate_tensor_memory_size(&self, tensor: &Tensor<T>) -> usize {
        let element_count: usize = tensor.shape().dims().iter().product();
        element_count * std::mem::size_of::<T>()
    }

    /// Get checkpointing statistics
    pub fn get_stats(&self) -> CheckpointStats {
        CheckpointStats {
            num_checkpoints: self.checkpoints.len(),
            memory_usage: self.current_memory_usage,
            memory_budget: self.memory_budget,
            memory_utilization: self.current_memory_usage as f64 / self.memory_budget as f64,
        }
    }
}

/// Statistics for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub num_checkpoints: usize,
    pub memory_usage: usize,
    pub memory_budget: usize,
    pub memory_utilization: f64,
}

/// Lazy gradient computation that defers expensive operations
pub struct LazyGradient<T> {
    computation: Box<dyn Fn() -> Result<Tensor<T>> + Send + Sync>,
    cached_result: Arc<Mutex<Option<Tensor<T>>>>,
    is_expensive: bool,
}

impl<T> LazyGradient<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new lazy gradient computation
    pub fn new<F>(computation: F, is_expensive: bool) -> Self
    where
        F: Fn() -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        Self {
            computation: Box::new(computation),
            cached_result: Arc::new(Mutex::new(None)),
            is_expensive,
        }
    }

    /// Get the computed gradient, computing it if necessary
    pub fn get(&self) -> Result<Tensor<T>> {
        let mut cached = self.cached_result.lock().unwrap();

        if let Some(result) = &*cached {
            return Ok(result.clone());
        }

        // Compute the gradient
        let result = (self.computation)()?;
        *cached = Some(result.clone());

        Ok(result)
    }

    /// Check if the gradient has been computed
    pub fn is_computed(&self) -> bool {
        self.cached_result.lock().unwrap().is_some()
    }

    /// Clear cached result to free memory
    pub fn clear_cache(&self) {
        *self.cached_result.lock().unwrap() = None;
    }

    /// Check if this is an expensive computation
    pub fn is_expensive(&self) -> bool {
        self.is_expensive
    }
}

/// Memory-efficient gradient aggregation with streaming computation
pub struct StreamingGradientAggregator<T> {
    accumulated_gradient: Option<Tensor<T>>,
    count: usize,
    memory_threshold: usize,
    temp_gradients: Vec<Tensor<T>>,
}

impl<T> StreamingGradientAggregator<T>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new streaming gradient aggregator
    pub fn new(memory_threshold: usize) -> Self {
        Self {
            accumulated_gradient: None,
            count: 0,
            memory_threshold,
            temp_gradients: Vec::new(),
        }
    }

    /// Add a gradient to the aggregation
    pub fn add_gradient(&mut self, gradient: Tensor<T>) -> Result<()> {
        self.temp_gradients.push(gradient);

        // Check if we should flush to avoid memory buildup
        if self.temp_gradients.len() >= self.memory_threshold {
            self.flush_temp_gradients()?;
        }

        self.count += 1;
        Ok(())
    }

    /// Flush temporary gradients to the main accumulator
    fn flush_temp_gradients(&mut self) -> Result<()> {
        if self.temp_gradients.is_empty() {
            return Ok(());
        }

        // Sum all temporary gradients
        let mut temp_sum = self.temp_gradients[0].clone();
        for grad in &self.temp_gradients[1..] {
            temp_sum = temp_sum.add(grad)?;
        }

        // Add to main accumulator
        self.accumulated_gradient = match &self.accumulated_gradient {
            Some(acc) => Some(acc.add(&temp_sum)?),
            None => Some(temp_sum),
        };

        // Clear temporary storage
        self.temp_gradients.clear();

        Ok(())
    }

    /// Get the final aggregated gradient
    pub fn finalize(&mut self) -> Result<Option<Tensor<T>>> {
        // Flush any remaining temporary gradients
        self.flush_temp_gradients()?;

        if let Some(acc_grad) = &self.accumulated_gradient {
            if self.count > 0 {
                let count_scalar = Tensor::from_scalar(T::from(self.count).unwrap());
                let avg_grad = acc_grad.div(&count_scalar)?;
                Ok(Some(avg_grad))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get current aggregation statistics
    pub fn get_stats(&self) -> AggregationStats {
        AggregationStats {
            total_gradients: self.count,
            temp_gradients_count: self.temp_gradients.len(),
            has_accumulated: self.accumulated_gradient.is_some(),
        }
    }

    /// Reset the aggregator
    pub fn reset(&mut self) {
        self.accumulated_gradient = None;
        self.count = 0;
        self.temp_gradients.clear();
    }
}

/// Statistics for gradient aggregation
#[derive(Debug, Clone)]
pub struct AggregationStats {
    pub total_gradients: usize,
    pub temp_gradients_count: usize,
    pub has_accumulated: bool,
}

/// Global memory manager for gradient computations
pub struct GradientMemoryManager<T> {
    memory_pool: Arc<Mutex<GradientMemoryPool<T>>>,
    checkpointer: Arc<Mutex<GradientCheckpointer<T>>>,
    lazy_computations: Vec<LazyGradient<T>>,
    memory_limit: usize,
}

impl<T> GradientMemoryManager<T>
where
    T: Clone + Default + Send + Sync + 'static + scirs2_core::num_traits::Zero,
{
    /// Create a new gradient memory manager
    pub fn new(memory_limit: usize, pool_size: usize) -> Self {
        Self {
            memory_pool: Arc::new(Mutex::new(GradientMemoryPool::new(pool_size))),
            checkpointer: Arc::new(Mutex::new(GradientCheckpointer::new(memory_limit / 2))),
            lazy_computations: Vec::new(),
            memory_limit,
        }
    }

    /// Get a tensor from the memory pool
    pub fn get_tensor(&self, shape: &[usize]) -> Result<Tensor<T>> {
        let mut pool = self
            .memory_pool
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "get_tensor".to_string(),
                reason: "Failed to acquire memory pool lock".to_string(),
                context: None,
            })?;

        Ok(pool.get_tensor(shape))
    }

    /// Return a tensor to the memory pool
    pub fn return_tensor(&self, tensor: Tensor<T>) -> Result<()> {
        let mut pool = self
            .memory_pool
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "return_tensor".to_string(),
                reason: "Failed to acquire memory pool lock".to_string(),
                context: None,
            })?;

        pool.return_tensor(tensor);
        Ok(())
    }

    /// Store a checkpoint
    pub fn store_checkpoint(&self, name: &str, tensor: Tensor<T>, cost: f64) -> Result<()> {
        let mut checkpointer =
            self.checkpointer
                .lock()
                .map_err(|_| TensorError::InvalidArgument {
                    operation: "store_checkpoint".to_string(),
                    reason: "Failed to acquire checkpointer lock".to_string(),
                    context: None,
                })?;

        checkpointer.store_checkpoint(name, tensor, cost)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> Result<MemoryManagerStats> {
        let pool = self
            .memory_pool
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "get_memory_stats".to_string(),
                reason: "Failed to acquire memory pool lock".to_string(),
                context: None,
            })?;

        let checkpointer = self
            .checkpointer
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "get_memory_stats".to_string(),
                reason: "Failed to acquire checkpointer lock".to_string(),
                context: None,
            })?;

        Ok(MemoryManagerStats {
            pool_stats: pool.get_stats(),
            checkpoint_stats: checkpointer.get_stats(),
            lazy_computations_count: self.lazy_computations.len(),
            memory_limit: self.memory_limit,
        })
    }
}

/// Combined statistics for the memory manager
#[derive(Debug, Clone)]
pub struct MemoryManagerStats {
    pub pool_stats: MemoryPoolStats,
    pub checkpoint_stats: CheckpointStats,
    pub lazy_computations_count: usize,
    pub memory_limit: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = GradientMemoryPool::<f32>::new(10);

        // Test getting and returning tensors
        let tensor1 = pool.get_tensor(&[2, 3]);
        let tensor2 = pool.get_tensor(&[2, 3]);

        pool.return_tensor(tensor1);
        let tensor3 = pool.get_tensor(&[2, 3]); // Should reuse returned tensor

        let stats = pool.get_stats();
        assert!(stats.total_allocated >= 2);
    }

    #[test]
    fn test_checkpointer() {
        let mut checkpointer = GradientCheckpointer::<f32>::new(1024);

        let tensor = Tensor::ones(&[2, 2]);
        checkpointer
            .store_checkpoint("test", tensor.clone(), 10.0)
            .unwrap();

        assert!(checkpointer.has_checkpoint("test"));
        let retrieved = checkpointer.get_checkpoint("test").unwrap();

        // Check shapes match
        assert_eq!(tensor.shape().dims(), retrieved.shape().dims());
    }

    #[test]
    fn test_streaming_aggregator() {
        let mut aggregator = StreamingGradientAggregator::<f32>::new(5);

        // Add some gradients
        for i in 0..10 {
            let grad = Tensor::from_scalar(i as f32).broadcast_to(&[2, 2]).unwrap();
            aggregator.add_gradient(grad).unwrap();
        }

        let final_grad = aggregator.finalize().unwrap();
        assert!(final_grad.is_some());

        let stats = aggregator.get_stats();
        assert_eq!(stats.total_gradients, 10);
    }
}
