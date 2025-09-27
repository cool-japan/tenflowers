use crate::tape::{GradientTape, TensorId, TrackedTensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Checkpointing strategy for memory-computation tradeoff
///
/// Checkpointing is a technique where we save certain intermediate values during
/// the forward pass and recompute others during the backward pass to save memory.
#[derive(Debug, Clone)]
pub enum CheckpointStrategy {
    /// Save all intermediate values (no checkpointing)
    NoCheckpointing,
    /// Save only every N-th layer's output
    EveryNLayers(usize),
    /// Save based on memory usage threshold
    MemoryThreshold(usize), // threshold in bytes
    /// Custom strategy based on operation type
    Custom(fn(&str) -> bool),
}

/// Checkpoint manager that handles saving and restoring intermediate values
#[derive(Debug)]
pub struct CheckpointManager {
    /// Stored checkpoints
    checkpoints: Arc<Mutex<HashMap<TensorId, Box<dyn std::any::Any + Send + Sync>>>>,
    /// Current checkpointing strategy
    strategy: CheckpointStrategy,
    /// Current memory usage estimation
    memory_usage: Arc<Mutex<usize>>,
    /// Whether checkpointing is enabled
    enabled: bool,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(strategy: CheckpointStrategy) -> Self {
        Self {
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            strategy,
            memory_usage: Arc::new(Mutex::new(0)),
            enabled: true,
        }
    }

    /// Enable or disable checkpointing
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if a tensor should be checkpointed based on the strategy
    pub fn should_checkpoint(
        &self,
        _tensor_id: TensorId,
        operation_name: &str,
        layer_index: usize,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        match &self.strategy {
            CheckpointStrategy::NoCheckpointing => false,
            CheckpointStrategy::EveryNLayers(n) => layer_index % n == 0,
            CheckpointStrategy::MemoryThreshold(threshold) => {
                let current_usage = *self.memory_usage.lock().unwrap();
                current_usage < *threshold
            }
            CheckpointStrategy::Custom(predicate) => predicate(operation_name),
        }
    }

    /// Save a tensor to checkpoint
    pub fn save_checkpoint<T>(&self, tensor_id: TensorId, tensor: &Tensor<T>)
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut checkpoints = self.checkpoints.lock().unwrap();

        // Estimate memory usage
        let estimated_size = self.estimate_tensor_size(tensor);
        let mut memory_usage = self.memory_usage.lock().unwrap();
        *memory_usage += estimated_size;

        checkpoints.insert(tensor_id, Box::new(tensor.clone()));
    }

    /// Restore a tensor from checkpoint
    pub fn restore_checkpoint<T>(&self, tensor_id: TensorId) -> Result<Option<Tensor<T>>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let checkpoints = self.checkpoints.lock().unwrap();

        if let Some(checkpoint) = checkpoints.get(&tensor_id) {
            if let Some(tensor) = checkpoint.downcast_ref::<Tensor<T>>() {
                Ok(Some(tensor.clone()))
            } else {
                Err(TensorError::invalid_argument(
                    "Type mismatch in checkpoint".to_string(),
                ))
            }
        } else {
            Ok(None)
        }
    }

    /// Remove a checkpoint (to free memory)
    pub fn remove_checkpoint(&self, tensor_id: TensorId) {
        let mut checkpoints = self.checkpoints.lock().unwrap();
        checkpoints.remove(&tensor_id);
    }

    /// Clear all checkpoints
    pub fn clear_checkpoints(&self) {
        let mut checkpoints = self.checkpoints.lock().unwrap();
        checkpoints.clear();

        let mut memory_usage = self.memory_usage.lock().unwrap();
        *memory_usage = 0;
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        *self.memory_usage.lock().unwrap()
    }

    /// Get number of stored checkpoints
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.lock().unwrap().len()
    }

    /// Estimate the memory size of a tensor
    fn estimate_tensor_size<T>(&self, tensor: &Tensor<T>) -> usize {
        let shape = tensor.shape();
        let elements = shape.dims().iter().product::<usize>();
        elements * std::mem::size_of::<T>()
    }
}

/// Checkpointed function wrapper
///
/// This allows wrapping a function with checkpointing logic
pub struct CheckpointedFunction<F> {
    func: F,
    checkpoint_manager: CheckpointManager,
    operation_name: String,
}

impl<F> CheckpointedFunction<F> {
    /// Create a new checkpointed function
    pub fn new(func: F, checkpoint_manager: CheckpointManager, operation_name: String) -> Self {
        Self {
            func,
            checkpoint_manager,
            operation_name,
        }
    }
}

impl<F> CheckpointedFunction<F> {
    /// Execute the function with checkpointing
    pub fn execute<T>(
        &self,
        input: &TrackedTensor<T>,
        layer_index: usize,
    ) -> Result<TrackedTensor<T>>
    where
        F: Fn(&TrackedTensor<T>) -> Result<TrackedTensor<T>>,
        T: Clone + Send + Sync + 'static,
    {
        // Check if we should checkpoint the input
        if self
            .checkpoint_manager
            .should_checkpoint(input.id, &self.operation_name, layer_index)
        {
            self.checkpoint_manager
                .save_checkpoint(input.id, &input.tensor);
        }

        // Execute the function
        let result = (self.func)(input)?;

        // Check if we should checkpoint the result
        if self
            .checkpoint_manager
            .should_checkpoint(result.id, &self.operation_name, layer_index)
        {
            self.checkpoint_manager
                .save_checkpoint(result.id, &result.tensor);
        }

        Ok(result)
    }
}

/// Type alias for forward computation function
type ForwardFn = Box<dyn Fn(&TrackedTensor<f32>) -> Result<TrackedTensor<f32>> + Send + Sync>;

/// Recomputation context for gradient computation
///
/// This handles recomputing intermediate values during backward pass
/// when they weren't stored in checkpoints
pub struct RecomputationContext {
    /// The original forward computation function
    forward_fn: ForwardFn,
    /// Checkpoint manager for restoring saved values
    checkpoint_manager: CheckpointManager,
}

impl RecomputationContext {
    /// Create a new recomputation context
    pub fn new<F>(forward_fn: F, checkpoint_manager: CheckpointManager) -> Self
    where
        F: Fn(&TrackedTensor<f32>) -> Result<TrackedTensor<f32>> + Send + Sync + 'static,
    {
        Self {
            forward_fn: Box::new(forward_fn),
            checkpoint_manager,
        }
    }

    /// Recompute a tensor if it's not in checkpoint
    pub fn recompute_if_needed(
        &self,
        tensor_id: TensorId,
        input: &TrackedTensor<f32>,
    ) -> Result<TrackedTensor<f32>> {
        // First, try to restore from checkpoint
        if let Some(restored_tensor) = self
            .checkpoint_manager
            .restore_checkpoint::<f32>(tensor_id)?
        {
            // Create a new TrackedTensor with the restored tensor
            // This is a simplified version - in practice we'd need to properly track the computation graph
            let tape = GradientTape::new();
            Ok(tape.watch(restored_tensor))
        } else {
            // Recompute the tensor
            (self.forward_fn)(input)
        }
    }
}

/// Convenience function for creating a memory-efficient computation sequence
///
/// This automatically handles checkpointing for a sequence of operations
pub fn checkpoint_sequence<T, F>(
    operations: Vec<F>,
    strategy: CheckpointStrategy,
    initial_input: TrackedTensor<T>,
) -> Result<TrackedTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    F: Fn(&TrackedTensor<T>) -> Result<TrackedTensor<T>>,
{
    let checkpoint_manager = CheckpointManager::new(strategy);
    let mut current = initial_input;

    for (i, op) in operations.into_iter().enumerate() {
        // Check if we should checkpoint before this operation
        if checkpoint_manager.should_checkpoint(current.id, "sequence_op", i) {
            checkpoint_manager.save_checkpoint(current.id, &current.tensor);
        }

        // Execute the operation
        current = op(&current)?;
    }

    Ok(current)
}

/// Automatic checkpointing for gradient computation
///
/// This extends the GradientTape to automatically manage checkpoints
pub trait CheckpointedGradientTape {
    /// Compute gradients with automatic checkpointing
    fn gradient_with_checkpointing<T>(
        &self,
        target: &TrackedTensor<T>,
        sources: &[&TrackedTensor<T>],
        strategy: CheckpointStrategy,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + num_traits::One
            + num_traits::FromPrimitive
            + num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable;
}

impl CheckpointedGradientTape for GradientTape {
    fn gradient_with_checkpointing<T>(
        &self,
        target: &TrackedTensor<T>,
        sources: &[&TrackedTensor<T>],
        _strategy: CheckpointStrategy,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + num_traits::One
            + num_traits::FromPrimitive
            + num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // For now, this is a simplified implementation that falls back to regular gradient computation
        // In a full implementation, this would:
        // 1. Analyze the computation graph
        // 2. Determine optimal checkpoint placement
        // 3. Recompute intermediate values as needed during backward pass

        // Convert single target to slice
        let targets = std::slice::from_ref(target);

        // Convert slice of references to slice of values
        let source_values: Vec<TrackedTensor<T>> = sources.iter().map(|&s| s.clone()).collect();

        // Get gradients and handle Option return type
        let grad_options = self.gradient(targets, &source_values)?;

        // Convert Vec<Option<Tensor<T>>> to Vec<Tensor<T>>, filtering out None values
        Ok(grad_options.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::Array1;
    use tenflowers_core::Tensor;

    #[test]
    fn test_checkpoint_manager() {
        let manager = CheckpointManager::new(CheckpointStrategy::NoCheckpointing);

        // Test checkpointing decision
        assert!(!manager.should_checkpoint(1, "test", 0));

        // Test with EveryNLayers strategy
        let mut manager = CheckpointManager::new(CheckpointStrategy::EveryNLayers(2));
        assert!(manager.should_checkpoint(1, "test", 0)); // 0 % 2 == 0
        assert!(!manager.should_checkpoint(1, "test", 1)); // 1 % 2 != 0
        assert!(manager.should_checkpoint(1, "test", 2)); // 2 % 2 == 0

        // Test enable/disable
        manager.set_enabled(false);
        assert!(!manager.should_checkpoint(1, "test", 0));
    }

    #[test]
    fn test_checkpoint_save_restore() {
        let manager = CheckpointManager::new(CheckpointStrategy::NoCheckpointing);

        // Create a test tensor
        let data = Array1::from_vec(vec![1.0f32, 2.0f32, 3.0f32]).into_dyn();
        let tensor = Tensor::from_array(data);

        // Save checkpoint
        manager.save_checkpoint(1, &tensor);
        assert_eq!(manager.checkpoint_count(), 1);

        // Restore checkpoint
        let restored: Option<Tensor<f32>> = manager.restore_checkpoint(1).unwrap();
        assert!(restored.is_some());

        let restored_tensor = restored.unwrap();
        if let tenflowers_core::tensor::TensorStorage::Cpu(ref array) = restored_tensor.storage {
            assert_eq!(array[[0]], 1.0);
            assert_eq!(array[[1]], 2.0);
            assert_eq!(array[[2]], 3.0);
        }

        // Test non-existent checkpoint
        let missing: Option<Tensor<f32>> = manager.restore_checkpoint(999).unwrap();
        assert!(missing.is_none());

        // Clear checkpoints
        manager.clear_checkpoints();
        assert_eq!(manager.checkpoint_count(), 0);
    }

    #[test]
    fn test_checkpointed_function() {
        let manager = CheckpointManager::new(CheckpointStrategy::EveryNLayers(1));

        // Create a simple function that doubles the input
        let double_fn = |x: &TrackedTensor<f32>| -> Result<TrackedTensor<f32>> { x.add(x) };

        let checkpointed_fn = CheckpointedFunction::new(double_fn, manager, "double".to_string());

        // Create input tensor
        let tape = GradientTape::new();
        let data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let input = tape.watch(Tensor::from_array(data));

        // Execute checkpointed function
        let result = checkpointed_fn.execute(&input, 0).unwrap();

        // Verify result
        if let tenflowers_core::tensor::TensorStorage::Cpu(ref array) = result.tensor.storage {
            assert_eq!(array[[0]], 2.0);
            assert_eq!(array[[1]], 4.0);
        }
    }

    #[test]
    fn test_checkpoint_sequence() {
        // Create a sequence of operations
        let ops: Vec<Box<dyn Fn(&TrackedTensor<f32>) -> Result<TrackedTensor<f32>>>> = vec![
            Box::new(|x| x.add(x)), // double
            Box::new(|x| x.add(x)), // double again
            Box::new(|x| x.add(x)), // double again
        ];

        let tape = GradientTape::new();
        let data = Array1::from_vec(vec![1.0f32]).into_dyn();
        let input = tape.watch(Tensor::from_array(data));

        // Execute sequence with checkpointing every 2 operations
        let result = checkpoint_sequence(ops, CheckpointStrategy::EveryNLayers(2), input).unwrap();

        // Result should be 1 * 2 * 2 * 2 = 8
        if let tenflowers_core::tensor::TensorStorage::Cpu(ref array) = result.tensor.storage {
            assert_eq!(array[[0]], 8.0);
        }
    }

    #[test]
    fn test_memory_threshold_strategy() {
        let manager = CheckpointManager::new(CheckpointStrategy::MemoryThreshold(1000));

        // Initially under threshold
        assert!(manager.should_checkpoint(1, "test", 0));

        // Save a large tensor to exceed threshold
        let large_data = Array1::from_vec(vec![1.0f32; 1000]).into_dyn();
        let large_tensor = Tensor::from_array(large_data);
        manager.save_checkpoint(1, &large_tensor);

        // Now over threshold
        assert!(!manager.should_checkpoint(2, "test", 0));

        // Clear to reset
        manager.clear_checkpoints();
        assert_eq!(manager.memory_usage(), 0);
    }
}
