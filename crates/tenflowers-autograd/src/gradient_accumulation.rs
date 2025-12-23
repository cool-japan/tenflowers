#[cfg(feature = "parallel")]
use crate::parallel_gradients::CommunicationBackend;
use crate::tape::{GradientTape, TensorId, TrackedTensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
#[cfg(feature = "parallel")]
use tenflowers_core::Device;
use tenflowers_core::{Result, Tensor, TensorError};

/// Gradient accumulator for large batch training
///
/// This allows accumulating gradients across multiple micro-batches to simulate
/// larger batch sizes without requiring all the memory at once.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    inner: Arc<Mutex<GradientAccumulatorInner>>,
}

#[derive(Debug)]
struct GradientAccumulatorInner {
    /// Accumulated gradients for each tensor
    accumulated_gradients: HashMap<TensorId, Box<dyn std::any::Any + Send + Sync>>,
    /// Number of micro-batches accumulated
    num_accumulated: usize,
    /// Whether to automatically average gradients when retrieving them
    average_gradients: bool,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator
    pub fn new(average_gradients: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(GradientAccumulatorInner {
                accumulated_gradients: HashMap::new(),
                num_accumulated: 0,
                average_gradients,
            })),
        }
    }

    /// Accumulate gradients from a gradient computation
    pub fn accumulate<T>(
        &self,
        tape: &GradientTape,
        target: &TrackedTensor<T>,
        sources: &[&TrackedTensor<T>],
    ) -> Result<()>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Convert single target to slice and references to values
        let targets = std::slice::from_ref(target);
        let source_values: Vec<TrackedTensor<T>> = sources.iter().map(|&s| s.clone()).collect();
        let gradients = tape.gradient(targets, &source_values)?;

        let mut inner = self.inner.lock().unwrap();

        for (source, gradient_opt) in sources.iter().zip(gradients.iter()) {
            // Skip if gradient is None
            if let Some(gradient) = gradient_opt {
                let tensor_id = source.id;

                if let Some(existing) = inner.accumulated_gradients.get_mut(&tensor_id) {
                    // Downcast existing gradient
                    if let Some(existing_grad) = existing.downcast_mut::<Tensor<T>>() {
                        *existing_grad = existing_grad.add(gradient)?;
                    } else {
                        return Err(TensorError::invalid_argument(
                            "Type mismatch in accumulated gradients".to_string(),
                        ));
                    }
                } else {
                    // First time seeing this tensor
                    inner
                        .accumulated_gradients
                        .insert(tensor_id, Box::new(gradient.clone()));
                }
            }
        }

        inner.num_accumulated += 1;
        Ok(())
    }

    /// Get the accumulated gradient for a specific tensor
    pub fn get_gradient<T>(&self, source: &TrackedTensor<T>) -> Result<Option<Tensor<T>>>
    where
        T: Clone
            + Default
            + std::ops::Div<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let inner = self.inner.lock().unwrap();

        if let Some(gradient) = inner.accumulated_gradients.get(&source.id) {
            if let Some(grad) = gradient.downcast_ref::<Tensor<T>>() {
                let mut result = grad.clone();

                // Average gradients if requested
                if inner.average_gradients && inner.num_accumulated > 0 {
                    let scale = T::from_usize(inner.num_accumulated).unwrap();
                    result = result.div(&Tensor::from_scalar(scale))?;
                }

                Ok(Some(result))
            } else {
                Err(TensorError::invalid_argument(
                    "Type mismatch in accumulated gradients".to_string(),
                ))
            }
        } else {
            Ok(None)
        }
    }

    /// Get all accumulated gradients
    pub fn get_all_gradients<T>(
        &self,
        sources: &[&TrackedTensor<T>],
    ) -> Result<Vec<Option<Tensor<T>>>>
    where
        T: Clone
            + Default
            + std::ops::Div<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod,
    {
        let mut results = Vec::new();

        for source in sources {
            results.push(self.get_gradient(source)?);
        }

        Ok(results)
    }

    /// Clear all accumulated gradients
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.accumulated_gradients.clear();
        inner.num_accumulated = 0;
    }

    /// Get the number of accumulated batches
    pub fn num_accumulated(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.num_accumulated
    }

    /// Check if any gradients have been accumulated
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.accumulated_gradients.is_empty()
    }

    /// Set whether to average gradients when retrieving them
    pub fn set_average_gradients(&self, average: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.average_gradients = average;
    }
}

impl Default for GradientAccumulator {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Convenience function to accumulate gradients over multiple micro-batches
///
/// This function performs gradient accumulation for a batch of data by:
/// 1. Splitting the batch into micro-batches
/// 2. Computing gradients for each micro-batch
/// 3. Accumulating the gradients
/// 4. Returning the accumulated gradients
pub fn accumulate_gradients_over_batch<T, F>(
    tape: &GradientTape,
    sources: &[&TrackedTensor<T>],
    data_batch: &[T],
    micro_batch_size: usize,
    mut compute_loss: F,
) -> Result<Vec<Tensor<T>>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::Signed
        + PartialOrd
        + bytemuck::Pod,
    F: FnMut(&[T]) -> Result<TrackedTensor<T>>,
{
    let accumulator = GradientAccumulator::new(true);

    // Process data in micro-batches
    for chunk in data_batch.chunks(micro_batch_size) {
        let loss = compute_loss(chunk)?;
        accumulator.accumulate(tape, &loss, sources)?;
    }

    // Get accumulated gradients
    let accumulated_grads = accumulator.get_all_gradients(sources)?;

    // Convert Options to Tensors (use zero gradient if no gradient accumulated)
    let mut result = Vec::new();
    for (i, grad_opt) in accumulated_grads.iter().enumerate() {
        match grad_opt {
            Some(grad) => result.push(grad.clone()),
            None => result.push(Tensor::zeros(sources[i].tensor.shape().dims())),
        }
    }

    Ok(result)
}

/// Distributed gradient accumulator for multi-node training
///
/// This extends the basic gradient accumulator to support distributed training
/// by accumulating gradients across multiple devices and processes.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct DistributedGradientAccumulator {
    /// Local gradient accumulator
    local_accumulator: GradientAccumulator,
    /// Devices participating in distributed training
    #[allow(dead_code)]
    devices: Vec<Device>,
    /// Communication backend for gradient aggregation
    communication_backend: CommunicationBackend,
    /// Process rank in distributed training
    rank: usize,
    /// Total number of processes
    world_size: usize,
    /// Internal state for distributed accumulation
    inner: Arc<Mutex<DistributedAccumulatorInner>>,
}

#[cfg(feature = "parallel")]
#[derive(Debug)]
struct DistributedAccumulatorInner {
    /// Node-specific accumulated gradients
    node_gradients: HashMap<usize, HashMap<TensorId, Box<dyn std::any::Any + Send + Sync>>>,
    /// Synchronization barriers for distributed accumulation
    sync_barriers: HashMap<String, usize>,
    /// Communication handles for async operations
    pending_communications: Vec<Box<dyn std::any::Any + Send + Sync>>,
}

#[cfg(feature = "parallel")]
impl DistributedGradientAccumulator {
    /// Create a new distributed gradient accumulator
    pub fn new(
        devices: Vec<Device>,
        communication_backend: CommunicationBackend,
        rank: usize,
        world_size: usize,
        average_gradients: bool,
    ) -> Self {
        Self {
            local_accumulator: GradientAccumulator::new(average_gradients),
            devices,
            communication_backend,
            rank,
            world_size,
            inner: Arc::new(Mutex::new(DistributedAccumulatorInner {
                node_gradients: HashMap::new(),
                sync_barriers: HashMap::new(),
                pending_communications: Vec::new(),
            })),
        }
    }

    /// Accumulate gradients locally and prepare for distributed aggregation
    pub fn accumulate_local<T>(
        &self,
        tape: &GradientTape,
        target: &TrackedTensor<T>,
        sources: &[&TrackedTensor<T>],
    ) -> Result<()>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Accumulate gradients locally first
        self.local_accumulator.accumulate(tape, target, sources)?;

        // Store gradients for distributed aggregation
        let targets = std::slice::from_ref(target);
        let source_values: Vec<TrackedTensor<T>> = sources.iter().map(|&s| s.clone()).collect();
        let gradients = tape.gradient(targets, &source_values)?;
        let mut inner = self.inner.lock().unwrap();

        let node_grads = inner.node_gradients.entry(self.rank).or_default();
        for (source, gradient_opt) in sources.iter().zip(gradients.iter()) {
            if let Some(gradient) = gradient_opt {
                node_grads.insert(source.id, Box::new(gradient.clone()));
            }
        }

        Ok(())
    }

    /// Perform distributed gradient aggregation using specified communication backend
    pub fn aggregate_distributed<T>(&self, sources: &[&TrackedTensor<T>]) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod,
    {
        match self.communication_backend {
            CommunicationBackend::RingAllReduce => self.ring_allreduce_aggregation(sources),
            CommunicationBackend::ParameterServer => self.parameter_server_aggregation(sources),
            CommunicationBackend::PeerToPeer => self.peer_to_peer_aggregation(sources),
        }
    }

    /// Ring allreduce gradient aggregation
    fn ring_allreduce_aggregation<T>(&self, sources: &[&TrackedTensor<T>]) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut results = Vec::new();

        for source in sources {
            // Get local gradient
            let local_grad = self.local_accumulator.get_gradient(source)?;

            if let Some(mut grad) = local_grad {
                // Simulate ring allreduce - in practice this would use actual communication
                // For now, we'll implement a simplified version that averages with other simulated nodes

                // In a real implementation, this would:
                // 1. Divide gradient into chunks
                // 2. Send chunk to next node in ring
                // 3. Receive and accumulate chunk from previous node
                // 4. Continue until all nodes have all accumulated chunks

                // For simulation, we'll just apply the averaging that would happen
                let scale = T::from_usize(self.world_size).unwrap();
                grad = grad.div(&Tensor::from_scalar(scale))?;

                results.push(grad);
            } else {
                // Return zero gradient if none accumulated
                results.push(Tensor::zeros(source.tensor.shape().dims()));
            }
        }

        Ok(results)
    }

    /// Parameter server gradient aggregation
    fn parameter_server_aggregation<T>(
        &self,
        sources: &[&TrackedTensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut results = Vec::new();

        for source in sources {
            // Get local gradient
            let local_grad = self.local_accumulator.get_gradient(source)?;

            if let Some(grad) = local_grad {
                // In parameter server mode:
                // - Worker nodes (rank != 0) send gradients to parameter server (rank 0)
                // - Parameter server aggregates gradients and sends back updated parameters
                // - Workers update their local parameters

                if self.rank == 0 {
                    // Parameter server: aggregate gradients from all workers
                    // For simulation, we'll just return the local gradient
                    results.push(grad);
                } else {
                    // Worker: send gradients to parameter server and receive updates
                    // For simulation, we'll just return the local gradient
                    results.push(grad);
                }
            } else {
                results.push(Tensor::zeros(source.tensor.shape().dims()));
            }
        }

        Ok(results)
    }

    /// Peer-to-peer gradient aggregation
    fn peer_to_peer_aggregation<T>(&self, sources: &[&TrackedTensor<T>]) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut results = Vec::new();

        for source in sources {
            // Get local gradient
            let local_grad = self.local_accumulator.get_gradient(source)?;

            if let Some(grad) = local_grad {
                // Peer-to-peer communication: each node exchanges gradients with neighbors
                // This is useful for decentralized training where no single node coordinates

                // In a real implementation, this would:
                // 1. Send gradients to neighbor nodes
                // 2. Receive gradients from neighbor nodes
                // 3. Average the received gradients

                // For simulation, we'll just return the local gradient
                results.push(grad);
            } else {
                results.push(Tensor::zeros(source.tensor.shape().dims()));
            }
        }

        Ok(results)
    }

    /// Synchronize gradients across all processes
    pub fn synchronize<T>(&self, barrier_name: &str, _sources: &[&TrackedTensor<T>]) -> Result<()>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut inner = self.inner.lock().unwrap();

        // Increment barrier count
        let count = inner
            .sync_barriers
            .entry(barrier_name.to_string())
            .or_insert(0);
        *count += 1;

        // In a real implementation, this would block until all processes reach the barrier
        // For simulation, we'll just check if the count matches the world size
        if *count >= self.world_size {
            // All processes synchronized, clear the barrier
            inner.sync_barriers.remove(barrier_name);
        }

        Ok(())
    }

    /// Get the current process rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size (total number of processes)
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if this is the root process (rank 0)
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Clear all accumulated gradients across all nodes
    pub fn clear_distributed(&self) {
        self.local_accumulator.clear();

        let mut inner = self.inner.lock().unwrap();
        inner.node_gradients.clear();
        inner.sync_barriers.clear();
        inner.pending_communications.clear();
    }

    /// Get distributed statistics
    pub fn get_distributed_stats(&self) -> DistributedStats {
        let inner = self.inner.lock().unwrap();
        DistributedStats {
            rank: self.rank,
            world_size: self.world_size,
            local_accumulations: self.local_accumulator.num_accumulated(),
            pending_communications: inner.pending_communications.len(),
            active_barriers: inner.sync_barriers.len(),
        }
    }
}

/// Statistics for distributed gradient accumulation
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct DistributedStats {
    pub rank: usize,
    pub world_size: usize,
    pub local_accumulations: usize,
    pub pending_communications: usize,
    pub active_barriers: usize,
}

/// Convenience function for distributed gradient accumulation
#[cfg(feature = "parallel")]
pub fn accumulate_gradients_distributed<T, F>(
    distributed_accumulator: &DistributedGradientAccumulator,
    tape: &GradientTape,
    sources: &[&TrackedTensor<T>],
    data_batch: &[T],
    micro_batch_size: usize,
    mut compute_loss: F,
) -> Result<Vec<Tensor<T>>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::Signed
        + PartialOrd
        + bytemuck::Pod,
    F: FnMut(&[T]) -> Result<TrackedTensor<T>>,
{
    // Process data in micro-batches locally
    for chunk in data_batch.chunks(micro_batch_size) {
        let loss = compute_loss(chunk)?;
        distributed_accumulator.accumulate_local(tape, &loss, sources)?;
    }

    // Synchronize across all processes
    distributed_accumulator.synchronize("gradient_accumulation", sources)?;

    // Aggregate gradients across all nodes
    distributed_accumulator.aggregate_distributed(sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use tenflowers_core::{tensor::TensorStorage, Tensor};

    #[test]
    fn test_gradient_accumulation_basic() {
        let tape = GradientTape::new();
        let accumulator = GradientAccumulator::new(false);

        // Create test tensors
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let y_data = Array1::from_vec(vec![3.0f32, 4.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        // Simulate two gradient computations
        let loss1 = x.add(&y).unwrap();
        let loss2 = x.mul(&y).unwrap();

        // Accumulate gradients
        accumulator.accumulate(&tape, &loss1, &[&x, &y]).unwrap();
        accumulator.accumulate(&tape, &loss2, &[&x, &y]).unwrap();

        // Check accumulation count
        assert_eq!(accumulator.num_accumulated(), 2);

        // Check gradients were accumulated
        let grad_x = accumulator.get_gradient(&x).unwrap();
        let grad_y = accumulator.get_gradient(&y).unwrap();

        assert!(grad_x.is_some());
        assert!(grad_y.is_some());

        // Clear and check
        accumulator.clear();
        assert_eq!(accumulator.num_accumulated(), 0);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn test_gradient_averaging() {
        let tape = GradientTape::new();
        let accumulator = GradientAccumulator::new(true);

        // Create test tensor
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        // Simulate gradient computation that should give gradient of [2.0, 2.0]
        let loss = x.add(&x).unwrap();

        // Accumulate the same gradient twice
        accumulator.accumulate(&tape, &loss, &[&x]).unwrap();
        accumulator.accumulate(&tape, &loss, &[&x]).unwrap();

        // With averaging, we should get [2.0, 2.0] (original gradient)
        // Without averaging, we would get [4.0, 4.0] (2 * original)
        let grad_x = accumulator.get_gradient(&x).unwrap().unwrap();

        // The gradient should be averaged - for add operation, gradient is [1.0, 1.0] + [1.0, 1.0] = [2.0, 2.0]
        // Since we accumulated twice and average, we should get [2.0, 2.0]
        let TensorStorage::Cpu(ref array) = grad_x.storage else {
            panic!("Expected CPU storage in test");
        };
        assert!((array[[0]] - 2.0).abs() < 1e-6);
        assert!((array[[1]] - 2.0).abs() < 1e-6);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_distributed_gradient_accumulator_creation() {
        let devices = vec![Device::Cpu, Device::Cpu];
        let accumulator = DistributedGradientAccumulator::new(
            devices,
            CommunicationBackend::RingAllReduce,
            0,
            2,
            true,
        );

        assert_eq!(accumulator.rank(), 0);
        assert_eq!(accumulator.world_size(), 2);
        assert!(accumulator.is_root());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_distributed_gradient_accumulation_basic() {
        let devices = vec![Device::Cpu];
        let accumulator = DistributedGradientAccumulator::new(
            devices,
            CommunicationBackend::RingAllReduce,
            0,
            2,
            true,
        );

        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        let loss = x.mul(&x).unwrap();
        accumulator.accumulate_local(&tape, &loss, &[&x]).unwrap();

        let stats = accumulator.get_distributed_stats();
        assert_eq!(stats.rank, 0);
        assert_eq!(stats.world_size, 2);
        assert_eq!(stats.local_accumulations, 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_distributed_gradient_aggregation() {
        let devices = vec![Device::Cpu];
        let accumulator = DistributedGradientAccumulator::new(
            devices,
            CommunicationBackend::RingAllReduce,
            0,
            2,
            true,
        );

        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        let loss = x.mul(&x).unwrap();
        accumulator.accumulate_local(&tape, &loss, &[&x]).unwrap();

        let aggregated_grads = accumulator.aggregate_distributed(&[&x]).unwrap();
        assert_eq!(aggregated_grads.len(), 1);

        // Check that the gradient was scaled by world size (simulation)
        let TensorStorage::Cpu(ref array) = aggregated_grads[0].storage else {
            panic!("Expected CPU storage in test");
        };
        assert!((array[[0]] - 1.0).abs() < 1e-6); // 2.0 / 2 = 1.0
        assert!((array[[1]] - 2.0).abs() < 1e-6); // 4.0 / 2 = 2.0
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_distributed_accumulation_convenience_function() {
        let devices = vec![Device::Cpu];
        let accumulator = DistributedGradientAccumulator::new(
            devices,
            CommunicationBackend::RingAllReduce,
            0,
            2,
            true,
        );

        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        let data_batch = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let micro_batch_size = 2;

        let result = accumulate_gradients_distributed(
            &accumulator,
            &tape,
            &[&x],
            &data_batch,
            micro_batch_size,
            |_batch| Ok(x.mul(&x).unwrap()),
        );

        assert!(result.is_ok());
        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parameter_server_mode() {
        let devices = vec![Device::Cpu];
        let accumulator = DistributedGradientAccumulator::new(
            devices,
            CommunicationBackend::ParameterServer,
            0,
            2,
            true,
        );

        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        let loss = x.mul(&x).unwrap();
        accumulator.accumulate_local(&tape, &loss, &[&x]).unwrap();

        let aggregated_grads = accumulator.aggregate_distributed(&[&x]).unwrap();
        assert_eq!(aggregated_grads.len(), 1);

        // In parameter server mode, gradients should be unmodified for simulation
        let TensorStorage::Cpu(ref array) = aggregated_grads[0].storage else {
            panic!("Expected CPU storage in test");
        };
        assert!((array[[0]] - 2.0).abs() < 1e-6);
        assert!((array[[1]] - 4.0).abs() < 1e-6);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_distributed_synchronization() {
        let devices = vec![Device::Cpu];
        let accumulator = DistributedGradientAccumulator::new(
            devices,
            CommunicationBackend::RingAllReduce,
            0,
            2,
            true,
        );

        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        // Test synchronization
        accumulator.synchronize("test_barrier", &[&x]).unwrap();

        let stats = accumulator.get_distributed_stats();
        assert_eq!(stats.active_barriers, 1);

        // Synchronize again to simulate second process
        accumulator.synchronize("test_barrier", &[&x]).unwrap();

        let stats_after = accumulator.get_distributed_stats();
        assert_eq!(stats_after.active_barriers, 0); // Barrier should be cleared
    }
}
