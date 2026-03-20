//! Collective operations: AllReduce, AllGather, Broadcast, Scatter, ReduceScatter

use std::time::Instant;
use tenflowers_core::{Result, Tensor, TensorError};

use super::types::{
    BackendConfig, CollectiveOp, CollectiveResult, CommunicationBackend, CommunicationBackendImpl,
    CommunicationGroup, CommunicationMetrics, CommunicationRuntime, ReductionOp,
};
use tenflowers_core::ops::manipulation::slice;

/// Element-wise minimum operation between two tensors
pub(crate) fn element_wise_min<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => tenflowers_core::ops::binary::min(a, b),
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
pub(crate) fn element_wise_max<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => tenflowers_core::ops::binary::max(a, b),
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::invalid_argument_op(
            "element_wise_max",
            "Tensors must be on the same device",
        )),
        #[cfg(not(feature = "gpu"))]
        _ => unreachable!("GPU variant should not exist without gpu feature"),
    }
}

impl Default for CommunicationRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunicationRuntime {
    /// Create new communication runtime
    pub fn new() -> Self {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};
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
        for backend in self.backends.values_mut() {
            backend.initialize(config)?;
        }
        Ok(())
    }

    /// Create a new communication group
    pub fn create_group(&mut self, group: CommunicationGroup) -> Result<()> {
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
                let gathered = backend.all_gather_f32(tensor, group)?;
                let reduced = self.reduce_tensors_f32(&gathered, reduction_op)?;
                let chunk_size = reduced.shape().dims()[0] / group.world_size;
                let start_idx = group.rank * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, reduced.shape().dims()[0]);
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
    pub(super) fn reduce_tensors_f32(
        &self,
        tensors: &[Tensor<f32>],
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
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
            let len_f32 = tensors.len() as f32;
            let divisor = Tensor::from_scalar(len_f32);
            result = result.div(&divisor)?;
        }

        Ok(result)
    }

    /// Update performance metrics for f32 tensors
    pub(super) fn update_metrics(
        &self,
        op: &CollectiveOp,
        tensor: &Tensor<f32>,
        elapsed: std::time::Duration,
    ) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.operation_count += 1;
            metrics.total_time += elapsed;

            let tensor_size = tensor.shape().size() * 4;
            metrics.total_bytes += tensor_size as u64;

            if metrics.total_time.as_secs_f64() > 0.0 {
                metrics.avg_bandwidth =
                    metrics.total_bytes as f64 / metrics.total_time.as_secs_f64();
            }

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
