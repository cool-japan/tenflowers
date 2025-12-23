/// GPU Reduction Kernels for TenfloweRS
///
/// This module implements high-performance GPU reduction operations using
/// tree-based and warp-level primitives for maximum efficiency.
use crate::{Result, Shape, Tensor, TensorError};

/// Reduction operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum reduction
    Sum,
    /// Product reduction
    Prod,
    /// Maximum reduction
    Max,
    /// Minimum reduction
    Min,
    /// Mean (average) reduction
    Mean,
    /// Variance reduction
    Variance,
    /// Standard deviation reduction
    StdDev,
    /// L1 norm reduction
    L1Norm,
    /// L2 norm reduction
    L2Norm,
    /// Any (logical OR) reduction
    Any,
    /// All (logical AND) reduction
    All,
}

impl ReductionOp {
    /// Get the name of this reduction operation
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Max => "max",
            Self::Min => "min",
            Self::Mean => "mean",
            Self::Variance => "var",
            Self::StdDev => "std",
            Self::L1Norm => "l1_norm",
            Self::L2Norm => "l2_norm",
            Self::Any => "any",
            Self::All => "all",
        }
    }

    /// Get the identity value for this reduction
    pub fn identity_f32(&self) -> f32 {
        match self {
            Self::Sum | Self::Mean | Self::L1Norm | Self::L2Norm => 0.0,
            Self::Prod => 1.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
            Self::Any => 0.0,
            Self::All => 1.0,
            Self::Variance | Self::StdDev => 0.0,
        }
    }

    /// Check if this reduction requires two passes
    pub fn requires_two_passes(&self) -> bool {
        matches!(self, Self::Variance | Self::StdDev | Self::Mean)
    }
}

/// GPU reduction configuration
#[derive(Debug, Clone)]
pub struct GpuReductionConfig {
    /// Block size for reduction
    pub block_size: usize,
    /// Whether to use shared memory
    pub use_shared_memory: bool,
    /// Whether to use warp-level primitives
    pub use_warp_primitives: bool,
    /// Maximum workgroup size
    pub max_workgroup_size: usize,
}

impl Default for GpuReductionConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            use_shared_memory: true,
            use_warp_primitives: true,
            max_workgroup_size: 1024,
        }
    }
}

/// Perform GPU reduction along an axis
#[cfg(feature = "gpu")]
pub fn gpu_reduce_axis<T>(
    tensor: &Tensor<T>,
    axis: usize,
    op: ReductionOp,
    keep_dims: bool,
) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    let shape = tensor.shape();
    if axis >= shape.rank() {
        return Err(TensorError::InvalidAxis {
            operation: "reduce".to_string(),
            axis: axis as i32,
            ndim: shape.rank(),
            context: None,
        });
    }

    // For now, fall back to CPU implementation
    // TODO: Implement actual GPU reduction kernels
    super::statistical::reduce_axis_cpu(tensor, axis, op, keep_dims)
}

/// Perform GPU full reduction (reduce all elements)
#[cfg(feature = "gpu")]
pub fn gpu_reduce_all<T>(tensor: &Tensor<T>, op: ReductionOp) -> Result<T>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    // For now, fall back to CPU implementation
    // TODO: Implement actual GPU reduction kernels
    super::statistical::reduce_all_cpu(tensor, op)
}

/// GPU sum reduction along axis
pub fn gpu_sum_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::Sum, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::Sum, keep_dims)
}

/// GPU mean reduction along axis
pub fn gpu_mean_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::Mean, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::Mean, keep_dims)
}

/// GPU max reduction along axis
pub fn gpu_max_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::Max, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::Max, keep_dims)
}

/// GPU min reduction along axis
pub fn gpu_min_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::Min, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::Min, keep_dims)
}

/// GPU variance reduction along axis
pub fn gpu_var_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::Variance, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::Variance, keep_dims)
}

/// GPU standard deviation reduction along axis
pub fn gpu_std_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::StdDev, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::StdDev, keep_dims)
}

/// GPU L1 norm along axis
pub fn gpu_l1_norm_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::L1Norm, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::L1Norm, keep_dims)
}

/// GPU L2 norm along axis
pub fn gpu_l2_norm_axis<T>(tensor: &Tensor<T>, axis: usize, keep_dims: bool) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            return gpu_reduce_axis(tensor, axis, ReductionOp::L2Norm, keep_dims);
        }
    }

    // CPU fallback
    super::statistical::reduce_axis_cpu(tensor, axis, ReductionOp::L2Norm, keep_dims)
}

/// Tree reduction implementation for GPU
/// This uses a logarithmic reduction tree for efficiency
#[cfg(feature = "gpu")]
pub struct TreeReduction {
    config: GpuReductionConfig,
}

#[cfg(feature = "gpu")]
impl TreeReduction {
    /// Create a new tree reduction
    pub fn new(config: GpuReductionConfig) -> Self {
        Self { config }
    }

    /// Compute the number of reduction steps needed
    pub fn num_steps(&self, input_size: usize) -> usize {
        (input_size as f64).log2().ceil() as usize
    }

    /// Compute the workgroup size for reduction
    pub fn workgroup_size(&self, input_size: usize) -> usize {
        self.config
            .block_size
            .min(input_size)
            .min(self.config.max_workgroup_size)
    }
}

/// Warp-level reduction primitives
#[cfg(feature = "gpu")]
pub struct WarpReduction {
    warp_size: usize,
}

#[cfg(feature = "gpu")]
impl WarpReduction {
    /// Create a new warp reduction (typically 32 threads per warp)
    pub fn new() -> Self {
        Self { warp_size: 32 }
    }

    /// Get warp size
    pub fn warp_size(&self) -> usize {
        self.warp_size
    }

    /// Check if warp primitives can be used for this size
    pub fn can_use_warp_primitives(&self, size: usize) -> bool {
        size >= self.warp_size
    }
}

#[cfg(feature = "gpu")]
impl Default for WarpReduction {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-stage reduction for very large tensors
pub struct MultiStageReduction {
    /// Number of elements per stage
    pub stage_size: usize,
    /// Number of stages
    pub num_stages: usize,
}

impl MultiStageReduction {
    /// Create a new multi-stage reduction
    pub fn new(total_elements: usize, max_stage_size: usize) -> Self {
        let num_stages = (total_elements + max_stage_size - 1) / max_stage_size;
        Self {
            stage_size: max_stage_size,
            num_stages,
        }
    }

    /// Get the size of a specific stage
    pub fn stage_size_for(&self, stage: usize) -> usize {
        if stage < self.num_stages - 1 {
            self.stage_size
        } else {
            // Last stage might be smaller
            self.stage_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_op_identity() {
        assert_eq!(ReductionOp::Sum.identity_f32(), 0.0);
        assert_eq!(ReductionOp::Prod.identity_f32(), 1.0);
        assert_eq!(ReductionOp::Max.identity_f32(), f32::NEG_INFINITY);
        assert_eq!(ReductionOp::Min.identity_f32(), f32::INFINITY);
    }

    #[test]
    fn test_reduction_op_two_passes() {
        assert!(ReductionOp::Variance.requires_two_passes());
        assert!(ReductionOp::StdDev.requires_two_passes());
        assert!(!ReductionOp::Sum.requires_two_passes());
    }

    #[test]
    fn test_multi_stage_reduction() {
        let reduction = MultiStageReduction::new(10_000_000, 1_000_000);
        assert_eq!(reduction.num_stages, 10);
        assert_eq!(reduction.stage_size, 1_000_000);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_tree_reduction() {
        let config = GpuReductionConfig::default();
        let tree = TreeReduction::new(config);

        assert_eq!(tree.num_steps(1024), 10); // log2(1024) = 10
        assert_eq!(tree.num_steps(256), 8); // log2(256) = 8
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_warp_reduction() {
        let warp = WarpReduction::new();
        assert_eq!(warp.warp_size(), 32);
        assert!(warp.can_use_warp_primitives(64));
        assert!(warp.can_use_warp_primitives(32));
        assert!(!warp.can_use_warp_primitives(16));
    }
}
