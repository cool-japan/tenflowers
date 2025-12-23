use crate::grad_ops;
use crate::tape::{GradientTape, Operation, TensorId, TrackedTensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use tenflowers_core::{Result, Tensor};
// use scirs2_core::num_traits::{Zero, One};

/// In-place operation tracker
///
/// This module provides functionality to detect when operations can be performed
/// in-place to save memory and improve performance, while still maintaining
/// correct gradient computation.
#[derive(Debug, Clone)]
pub struct InPlaceOptimizer {
    /// Track which tensors are safe to modify in-place
    safe_inplace_tensors: Arc<Mutex<HashMap<TensorId, bool>>>,
    /// Track reference counts for tensors
    tensor_ref_counts: Arc<Mutex<HashMap<TensorId, usize>>>,
}

impl InPlaceOptimizer {
    /// Create a new in-place optimizer
    pub fn new() -> Self {
        Self {
            safe_inplace_tensors: Arc::new(Mutex::new(HashMap::new())),
            tensor_ref_counts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if a tensor can be safely modified in-place
    pub fn can_modify_inplace(&self, tensor_id: TensorId) -> bool {
        let ref_counts = self.tensor_ref_counts.lock().unwrap();
        let safe_tensors = self.safe_inplace_tensors.lock().unwrap();

        // A tensor can be modified in-place if:
        // 1. It has only one reference (not shared)
        // 2. It's explicitly marked as safe for in-place operations
        // 3. It's not used in subsequent operations

        let ref_count = ref_counts.get(&tensor_id).cloned().unwrap_or(0);
        let is_safe = safe_tensors.get(&tensor_id).cloned().unwrap_or(false);

        ref_count <= 1 && is_safe
    }

    /// Mark a tensor as safe for in-place operations
    pub fn mark_safe_inplace(&self, tensor_id: TensorId) {
        let mut safe_tensors = self.safe_inplace_tensors.lock().unwrap();
        safe_tensors.insert(tensor_id, true);
    }

    /// Increment reference count for a tensor
    pub fn increment_ref_count(&self, tensor_id: TensorId) {
        let mut ref_counts = self.tensor_ref_counts.lock().unwrap();
        *ref_counts.entry(tensor_id).or_insert(0) += 1;
    }

    /// Decrement reference count for a tensor
    pub fn decrement_ref_count(&self, tensor_id: TensorId) {
        let mut ref_counts = self.tensor_ref_counts.lock().unwrap();
        if let Some(count) = ref_counts.get_mut(&tensor_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Clear all tracking data
    pub fn clear(&self) {
        self.safe_inplace_tensors.lock().unwrap().clear();
        self.tensor_ref_counts.lock().unwrap().clear();
    }
}

impl Default for InPlaceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// In-place operation variants for TrackedTensor
impl<T> TrackedTensor<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// In-place addition: self += other
    /// Only performs in-place if safe to do so, otherwise falls back to regular add
    pub fn add_inplace(&mut self, other: &TrackedTensor<T>) -> Result<()>
    where
        T: std::ops::Add<Output = T>
            + std::ops::AddAssign
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Check if we can perform in-place operation
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);

            // For now, always create new tensor and replace self
            // In a full implementation, we'd check reference counts
            let result = self.tensor.add(&other.tensor)?;

            // Record the operation
            let new_tracked = tape.record_op(
                Operation::Add {
                    lhs: self.id,
                    rhs: other.id,
                },
                result,
            );

            // Replace self with the new tensor
            *self = new_tracked;

            Ok(())
        } else {
            // No tape, just perform the operation directly
            self.tensor = self.tensor.add(&other.tensor)?;
            Ok(())
        }
    }

    /// In-place multiplication: self *= other
    pub fn mul_inplace(&mut self, other: &TrackedTensor<T>) -> Result<()>
    where
        T: std::ops::Mul<Output = T>
            + std::ops::MulAssign
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);

            let result = self.tensor.mul(&other.tensor)?;

            let new_tracked = tape.record_op(
                Operation::Mul {
                    lhs: self.id,
                    rhs: other.id,
                },
                result,
            );

            *self = new_tracked;

            Ok(())
        } else {
            self.tensor = self.tensor.mul(&other.tensor)?;
            Ok(())
        }
    }

    /// In-place subtraction: self -= other
    pub fn sub_inplace(&mut self, other: &TrackedTensor<T>) -> Result<()>
    where
        T: std::ops::Sub<Output = T>
            + std::ops::SubAssign
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);

            let result = self.tensor.sub(&other.tensor)?;

            let new_tracked = tape.record_op(
                Operation::Sub {
                    lhs: self.id,
                    rhs: other.id,
                },
                result,
            );

            *self = new_tracked;

            Ok(())
        } else {
            self.tensor = self.tensor.sub(&other.tensor)?;
            Ok(())
        }
    }

    /// In-place ReLU: self = relu(self)
    pub fn relu_inplace(&mut self) -> Result<()>
    where
        T: PartialOrd
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);

            let result = grad_ops::relu_forward(&self.tensor)?;

            let new_tracked = tape.record_op(Operation::Relu { input: self.id }, result);

            *self = new_tracked;

            Ok(())
        } else {
            self.tensor = grad_ops::relu_forward(&self.tensor)?;
            Ok(())
        }
    }

    /// In-place negation: self = -self
    pub fn neg_inplace(&mut self) -> Result<()>
    where
        T: std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);

            let result = self.tensor.neg()?;

            let new_tracked = tape.record_op(Operation::Neg { input: self.id }, result);

            *self = new_tracked;

            Ok(())
        } else {
            self.tensor = self.tensor.neg()?;
            Ok(())
        }
    }

    /// In-place scalar addition: self += scalar
    pub fn add_scalar_inplace(&mut self, scalar: T) -> Result<()>
    where
        T: std::ops::Add<Output = T>
            + std::ops::AddAssign
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Create a scalar tensor and use add_inplace
        let scalar_tensor = if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);
            tape.watch(Tensor::from_scalar(scalar))
        } else {
            TrackedTensor {
                tensor: Tensor::from_scalar(scalar),
                id: 0,
                tape: Weak::new(),
            }
        };

        self.add_inplace(&scalar_tensor)
    }

    /// In-place scalar multiplication: self *= scalar
    pub fn mul_scalar_inplace(&mut self, scalar: T) -> Result<()>
    where
        T: std::ops::Mul<Output = T>
            + std::ops::MulAssign
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let scalar_tensor = if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);
            tape.watch(Tensor::from_scalar(scalar))
        } else {
            TrackedTensor {
                tensor: Tensor::from_scalar(scalar),
                id: 0,
                tape: Weak::new(),
            }
        };

        self.mul_inplace(&scalar_tensor)
    }

    /// True in-place operation on the underlying tensor data
    /// This is unsafe from a gradient perspective and should only be used
    /// when you're certain no gradients will be computed
    pub fn unsafe_inplace_map<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&T) -> T,
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        self.tensor.map_inplace(f)
    }
}

/// Memory-efficient operation sequence executor
///
/// This helps identify sequences of operations that can be optimized
/// for memory usage through in-place operations
pub struct InPlaceSequenceOptimizer {
    #[allow(dead_code)]
    optimizer: InPlaceOptimizer,
}

impl InPlaceSequenceOptimizer {
    /// Create a new sequence optimizer
    pub fn new() -> Self {
        Self {
            optimizer: InPlaceOptimizer::new(),
        }
    }

    /// Execute a sequence of operations with in-place optimization
    ///
    /// This analyzes the sequence and determines where in-place operations
    /// can be safely used
    pub fn execute_sequence<T, F>(
        &self,
        initial_tensor: TrackedTensor<T>,
        ops: Vec<F>,
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
        F: FnOnce(TrackedTensor<T>) -> Result<TrackedTensor<T>>,
    {
        let mut current = initial_tensor;

        for op in ops {
            current = op(current)?;
        }

        Ok(current)
    }

    /// Optimize a computation graph for in-place operations
    ///
    /// This is a more advanced feature that would analyze the entire
    /// computation graph and determine optimal in-place operation placement
    pub fn optimize_graph(&self, _nodes: &[crate::tape::TapeNode]) -> Result<()> {
        // This would be a complex graph optimization algorithm
        // For now, we just provide the framework
        Ok(())
    }
}

impl Default for InPlaceSequenceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for in-place operation detection
pub mod utils {
    use super::*;

    /// Check if two tensors have the same shape (required for most in-place ops)
    pub fn shapes_compatible_for_inplace<T>(a: &TrackedTensor<T>, b: &TrackedTensor<T>) -> bool {
        a.tensor.shape() == b.tensor.shape()
    }

    /// Estimate memory savings from using in-place operations
    pub fn estimate_memory_savings<T>(tensor: &TrackedTensor<T>) -> usize {
        // Rough estimate: size of tensor in bytes
        let shape = tensor.tensor.shape();
        let elements = shape.dims().iter().product::<usize>();
        elements * std::mem::size_of::<T>()
    }

    /// Check if a tensor is a leaf tensor (no dependencies)
    pub fn is_leaf_tensor<T>(tensor: &TrackedTensor<T>) -> bool {
        // In a full implementation, we'd check the computation graph
        // For now, just check if it has a valid tape
        tensor.tape.upgrade().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use tenflowers_core::Tensor;

    #[test]
    fn test_inplace_add() {
        let tape = GradientTape::new();

        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let y_data = Array1::from_vec(vec![3.0f32, 4.0f32]).into_dyn();

        let mut x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        // Perform in-place addition
        x.add_inplace(&y).unwrap();

        // Check the result
        let tenflowers_core::tensor::TensorStorage::Cpu(ref array) = x.tensor.storage else {
            panic!("Expected CPU storage in test");
        };
        assert!((array[[0]] - 4.0).abs() < 1e-6);
        assert!((array[[1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_inplace_mul() {
        let tape = GradientTape::new();

        let x_data = Array1::from_vec(vec![2.0f32, 3.0f32]).into_dyn();
        let y_data = Array1::from_vec(vec![4.0f32, 5.0f32]).into_dyn();

        let mut x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        // Perform in-place multiplication
        x.mul_inplace(&y).unwrap();

        // Check the result
        let tenflowers_core::tensor::TensorStorage::Cpu(ref array) = x.tensor.storage else {
            panic!("Expected CPU storage in test");
        };
        assert!((array[[0]] - 8.0).abs() < 1e-6);
        assert!((array[[1]] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_inplace_optimizer() {
        let optimizer = InPlaceOptimizer::new();

        // Test reference counting
        let tensor_id = 1;
        optimizer.increment_ref_count(tensor_id);
        assert!(!optimizer.can_modify_inplace(tensor_id));

        // Mark as safe for in-place
        optimizer.mark_safe_inplace(tensor_id);
        optimizer.decrement_ref_count(tensor_id);
        assert!(optimizer.can_modify_inplace(tensor_id));

        // Clear tracking
        optimizer.clear();
        assert!(!optimizer.can_modify_inplace(tensor_id));
    }

    #[test]
    fn test_scalar_inplace_ops() {
        let tape = GradientTape::new();

        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let mut x = tape.watch(Tensor::from_array(x_data));

        // Test scalar addition
        x.add_scalar_inplace(5.0).unwrap();

        let tenflowers_core::tensor::TensorStorage::Cpu(ref array) = x.tensor.storage else {
            panic!("Expected CPU storage in test");
        };
        assert!((array[[0]] - 6.0).abs() < 1e-6);
        assert!((array[[1]] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils() {
        let tape = GradientTape::new();

        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let y_data = Array1::from_vec(vec![3.0f32, 4.0f32]).into_dyn();

        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        // Test shape compatibility
        assert!(utils::shapes_compatible_for_inplace(&x, &y));

        // Test memory savings estimation
        let savings = utils::estimate_memory_savings(&x);
        assert!(savings > 0);

        // Test leaf tensor check
        assert!(utils::is_leaf_tensor(&x));
    }
}
