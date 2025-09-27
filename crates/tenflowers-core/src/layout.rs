use crate::device::Device;
/// Data layout optimization for tensors
///
/// This module provides utilities for handling different data layouts
/// to optimize performance on different hardware (CPU vs GPU).
use crate::{Result, Tensor, TensorError};
use std::collections::HashMap;

/// Supported data layouts for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataLayout {
    /// Channels first: [N, C, H, W] - Optimal for GPU/CUDA
    NCHW,
    /// Channels last: [N, H, W, C] - Optimal for CPU/NEON
    NHWC,
    /// Channels first for 3D: [N, C, D, H, W]
    NCDHW,
    /// Channels last for 3D: [N, D, H, W, C]
    NDHWC,
    /// Channels first for 1D: [N, C, L]
    NCL,
    /// Channels last for 1D: [N, L, C]
    NLC,
    /// Auto-detect optimal layout based on device and operation
    Auto,
}

impl DataLayout {
    /// Get the number of dimensions for this layout
    pub fn ndim(&self) -> usize {
        match self {
            DataLayout::NCL | DataLayout::NLC => 3,
            DataLayout::NCHW | DataLayout::NHWC => 4,
            DataLayout::NCDHW | DataLayout::NDHWC => 5,
            DataLayout::Auto => 0, // Variable
        }
    }

    /// Get channel axis for this layout
    pub fn channel_axis(&self) -> usize {
        match self {
            DataLayout::NCHW | DataLayout::NCDHW | DataLayout::NCL => 1,
            DataLayout::NHWC => 3,
            DataLayout::NDHWC | DataLayout::NLC => 4,
            DataLayout::Auto => panic!("Cannot get channel axis for Auto layout"),
        }
    }

    /// Check if this is a channels-first layout
    pub fn is_channels_first(&self) -> bool {
        matches!(self, DataLayout::NCHW | DataLayout::NCDHW | DataLayout::NCL)
    }

    /// Get the permutation indices to convert from this layout to target layout
    pub fn to_permutation(&self, target: DataLayout) -> Option<Vec<usize>> {
        match (self, target) {
            (DataLayout::NCHW, DataLayout::NHWC) => Some(vec![0, 2, 3, 1]), // [N,C,H,W] -> [N,H,W,C]
            (DataLayout::NHWC, DataLayout::NCHW) => Some(vec![0, 3, 1, 2]), // [N,H,W,C] -> [N,C,H,W]
            (DataLayout::NCDHW, DataLayout::NDHWC) => Some(vec![0, 2, 3, 4, 1]), // [N,C,D,H,W] -> [N,D,H,W,C]
            (DataLayout::NDHWC, DataLayout::NCDHW) => Some(vec![0, 4, 1, 2, 3]), // [N,D,H,W,C] -> [N,C,D,H,W]
            (DataLayout::NCL, DataLayout::NLC) => Some(vec![0, 2, 1]), // [N,C,L] -> [N,L,C]
            (DataLayout::NLC, DataLayout::NCL) => Some(vec![0, 2, 1]), // [N,L,C] -> [N,C,L]
            _ if self == &target => None,                              // No conversion needed
            _ => None,                                                 // Unsupported conversion
        }
    }
}

/// Layout optimizer that chooses optimal data layouts based on device and operation type
pub struct LayoutOptimizer {
    /// Preferred layouts for different device types and operations
    layout_preferences: HashMap<(Device, OperationType), DataLayout>,
    /// Performance hints for layout conversions
    conversion_costs: HashMap<(DataLayout, DataLayout), f32>,
}

/// Types of operations that may benefit from specific layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Convolution,
    FullyConnected,
    Pooling,
    Normalization,
    Activation,
    ElementWise,
    Reduction,
}

impl Default for LayoutOptimizer {
    fn default() -> Self {
        let mut layout_preferences = HashMap::new();
        let mut conversion_costs = HashMap::new();

        // GPU preferences (CUDA-style)
        #[cfg(feature = "gpu")]
        {
            layout_preferences.insert(
                (Device::Gpu(0), OperationType::Convolution),
                DataLayout::NCHW,
            );
            layout_preferences.insert(
                (Device::Gpu(0), OperationType::FullyConnected),
                DataLayout::NCHW,
            );
            layout_preferences.insert((Device::Gpu(0), OperationType::Pooling), DataLayout::NCHW);
            layout_preferences.insert(
                (Device::Gpu(0), OperationType::Normalization),
                DataLayout::NCHW,
            );
            layout_preferences.insert(
                (Device::Gpu(0), OperationType::Activation),
                DataLayout::NCHW,
            );
            layout_preferences.insert(
                (Device::Gpu(0), OperationType::ElementWise),
                DataLayout::NCHW,
            );
        }

        // CPU preferences
        layout_preferences.insert((Device::Cpu, OperationType::Convolution), DataLayout::NHWC);
        layout_preferences.insert(
            (Device::Cpu, OperationType::FullyConnected),
            DataLayout::NHWC,
        );
        layout_preferences.insert((Device::Cpu, OperationType::Pooling), DataLayout::NHWC);
        layout_preferences.insert(
            (Device::Cpu, OperationType::Normalization),
            DataLayout::NHWC,
        );
        layout_preferences.insert((Device::Cpu, OperationType::Activation), DataLayout::NHWC);
        layout_preferences.insert((Device::Cpu, OperationType::ElementWise), DataLayout::NHWC);

        // Conversion costs (relative units)
        conversion_costs.insert((DataLayout::NCHW, DataLayout::NHWC), 1.0);
        conversion_costs.insert((DataLayout::NHWC, DataLayout::NCHW), 1.0);
        conversion_costs.insert((DataLayout::NCDHW, DataLayout::NDHWC), 1.5);
        conversion_costs.insert((DataLayout::NDHWC, DataLayout::NCDHW), 1.5);
        conversion_costs.insert((DataLayout::NCL, DataLayout::NLC), 0.5);
        conversion_costs.insert((DataLayout::NLC, DataLayout::NCL), 0.5);

        LayoutOptimizer {
            layout_preferences,
            conversion_costs,
        }
    }
}

impl LayoutOptimizer {
    /// Get the preferred layout for a given device and operation type
    pub fn preferred_layout(&self, device: &Device, op_type: OperationType) -> DataLayout {
        self.layout_preferences
            .get(&(*device, op_type))
            .copied()
            .unwrap_or(DataLayout::NCHW) // Default fallback
    }

    /// Get the cost of converting between two layouts
    pub fn conversion_cost(&self, from: DataLayout, to: DataLayout) -> f32 {
        if from == to {
            return 0.0;
        }
        self.conversion_costs
            .get(&(from, to))
            .copied()
            .unwrap_or(2.0) // Default high cost for unsupported conversions
    }

    /// Determine if a layout conversion is beneficial
    pub fn should_convert(&self, from: DataLayout, to: DataLayout, operation_benefit: f32) -> bool {
        let cost = self.conversion_cost(from, to);
        operation_benefit > cost
    }

    /// Auto-select the best layout for a tensor given the target device and operation
    pub fn auto_layout(
        &self,
        current_layout: DataLayout,
        target_device: &Device,
        op_type: OperationType,
        operation_intensity: f32,
    ) -> DataLayout {
        let preferred = self.preferred_layout(target_device, op_type);

        if self.should_convert(current_layout, preferred, operation_intensity) {
            preferred
        } else {
            current_layout
        }
    }
}

/// Permute tensor dimensions according to given axes order
fn permute_tensor<T>(input: &Tensor<T>, axes: &[usize]) -> Result<Tensor<T>>
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
    use crate::tensor::TensorStorage;

    match &input.storage {
        TensorStorage::Cpu(arr) => {
            let permuted = arr.clone().permuted_axes(axes);

            // Convert the permuted array to a vec and create new tensor
            let new_shape: Vec<usize> = {
                let old_shape = input.shape().dims();
                axes.iter().map(|&i| old_shape[i]).collect()
            };

            let vec_data: Vec<T> = permuted.iter().cloned().collect();
            Tensor::from_vec(vec_data, &new_shape)
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            // GPU tensor permutation using compute shader
            gpu_permute_tensor(gpu_buffer, input.shape().dims(), axes)
        }
    }
}

/// GPU tensor permutation using compute shader
#[cfg(feature = "gpu")]
fn gpu_permute_tensor<T>(
    gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    input_shape: &[usize],
    axes: &[usize],
) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_tensor_permutation;

    // Calculate output shape
    let output_shape: Vec<usize> = axes.iter().map(|&i| input_shape[i]).collect();
    let output_len = output_shape.iter().product();

    // Execute GPU permutation
    let result_buffer = execute_tensor_permutation(gpu_buffer, axes, input_shape, output_len)?;

    Ok(Tensor::from_gpu_buffer(
        result_buffer,
        crate::Shape::new(output_shape),
    ))
}

/// Convert tensor between different data layouts
pub fn convert_layout<T>(
    input: &Tensor<T>,
    from_layout: DataLayout,
    to_layout: DataLayout,
) -> Result<Tensor<T>>
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
    if from_layout == to_layout {
        return Ok(input.clone());
    }

    if let Some(perm) = from_layout.to_permutation(to_layout) {
        permute_tensor(input, &perm)
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "Layout conversion from {from_layout:?} to {to_layout:?} not supported"
        )))
    }
}

/// Infer the likely data layout from tensor shape and context
pub fn infer_layout(shape: &[usize], ndim_hint: Option<usize>) -> DataLayout {
    let ndim = ndim_hint.unwrap_or(shape.len());

    match ndim {
        3 => {
            // For 3D tensors, assume NCL if channel dimension is small
            if shape.len() >= 3 && shape[1] <= 512 && shape[1] < shape[2] {
                DataLayout::NCL
            } else {
                DataLayout::NLC
            }
        }
        4 => {
            // For 4D tensors, assume NCHW if channel dimension is small
            if shape.len() >= 4 && shape[1] <= 2048 && shape[1] < shape[2] && shape[1] < shape[3] {
                DataLayout::NCHW
            } else {
                DataLayout::NHWC
            }
        }
        5 => {
            // For 5D tensors, assume NCDHW if channel dimension is small
            if shape.len() >= 5 && shape[1] <= 2048 && shape[1] < shape[2] {
                DataLayout::NCDHW
            } else {
                DataLayout::NDHWC
            }
        }
        _ => DataLayout::Auto,
    }
}

/// Smart layout converter that minimizes conversions in a computation graph
pub struct LayoutPlan {
    conversions: Vec<(usize, DataLayout, DataLayout)>, // (tensor_id, from, to)
    optimal_layouts: HashMap<usize, DataLayout>,
}

impl LayoutPlan {
    /// Create an optimal layout plan for a sequence of operations
    pub fn optimize(
        tensor_layouts: &[(usize, DataLayout)],
        operations: &[(OperationType, Vec<usize>, Device)], // (op_type, input_tensor_ids, device)
        optimizer: &LayoutOptimizer,
    ) -> Self {
        let mut optimal_layouts = HashMap::new();
        let mut conversions = Vec::new();

        // Initialize with current layouts
        for &(tensor_id, layout) in tensor_layouts {
            optimal_layouts.insert(tensor_id, layout);
        }

        // Process operations and determine optimal layouts
        for (op_type, input_ids, device) in operations {
            for &tensor_id in input_ids {
                if let Some(&current_layout) = optimal_layouts.get(&tensor_id) {
                    let preferred = optimizer.preferred_layout(device, *op_type);

                    // Simple heuristic: convert if operation intensity is high
                    let operation_intensity = match op_type {
                        OperationType::Convolution => 3.0,
                        OperationType::FullyConnected => 2.0,
                        OperationType::Pooling => 1.5,
                        _ => 1.0,
                    };

                    if optimizer.should_convert(current_layout, preferred, operation_intensity) {
                        conversions.push((tensor_id, current_layout, preferred));
                        optimal_layouts.insert(tensor_id, preferred);
                    }
                }
            }
        }

        LayoutPlan {
            conversions,
            optimal_layouts,
        }
    }

    /// Get the planned conversions
    pub fn conversions(&self) -> &[(usize, DataLayout, DataLayout)] {
        &self.conversions
    }

    /// Get the optimal layout for a tensor
    pub fn optimal_layout(&self, tensor_id: usize) -> Option<DataLayout> {
        self.optimal_layouts.get(&tensor_id).copied()
    }
}

/// Automatic layout optimization pass for computation graphs
pub struct AutoLayoutOptimizer {
    optimizer: LayoutOptimizer,
    /// Track tensor layouts throughout the computation
    tensor_layouts: HashMap<usize, DataLayout>,
    /// Track conversion costs
    total_conversion_cost: f32,
}

impl AutoLayoutOptimizer {
    /// Create a new automatic layout optimizer
    pub fn new() -> Self {
        Self {
            optimizer: LayoutOptimizer::default(),
            tensor_layouts: HashMap::new(),
            total_conversion_cost: 0.0,
        }
    }

    /// Register a tensor with its initial layout
    pub fn register_tensor(&mut self, tensor_id: usize, layout: DataLayout) {
        self.tensor_layouts.insert(tensor_id, layout);
    }

    /// Optimize layout for a specific operation
    pub fn optimize_for_operation<T>(
        &mut self,
        tensors: &mut [&mut Tensor<T>],
        tensor_ids: &[usize],
        op_type: OperationType,
        device: &Device,
    ) -> Result<()>
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
        let preferred_layout = self.optimizer.preferred_layout(device, op_type);

        // Determine operation intensity for cost-benefit analysis
        let operation_intensity = match op_type {
            OperationType::Convolution => 3.0,
            OperationType::FullyConnected => 2.0,
            OperationType::Pooling => 1.5,
            OperationType::Normalization => 1.2,
            OperationType::Activation => 0.8,
            OperationType::ElementWise => 0.5,
            OperationType::Reduction => 1.0,
        };

        // Check if conversion is beneficial for each tensor
        for (tensor, &tensor_id) in tensors.iter_mut().zip(tensor_ids.iter()) {
            if let Some(&current_layout) = self.tensor_layouts.get(&tensor_id) {
                if current_layout != preferred_layout {
                    let conversion_cost = self
                        .optimizer
                        .conversion_cost(current_layout, preferred_layout);

                    if operation_intensity > conversion_cost {
                        // Convert the tensor
                        let converted = convert_layout(tensor, current_layout, preferred_layout)?;
                        **tensor = converted;

                        // Update tracking
                        self.tensor_layouts.insert(tensor_id, preferred_layout);
                        self.total_conversion_cost += conversion_cost;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the current layout of a tensor
    pub fn get_layout(&self, tensor_id: usize) -> Option<DataLayout> {
        self.tensor_layouts.get(&tensor_id).copied()
    }

    /// Get the total conversion cost incurred
    pub fn total_cost(&self) -> f32 {
        self.total_conversion_cost
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.tensor_layouts.clear();
        self.total_conversion_cost = 0.0;
    }
}

impl Default for AutoLayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Layout optimization hint for specific operations
#[derive(Debug, Clone)]
pub struct LayoutHint {
    pub operation: OperationType,
    pub preferred_layout: DataLayout,
    pub priority: f32,
}

impl LayoutHint {
    /// Create a new layout hint
    pub fn new(operation: OperationType, preferred_layout: DataLayout, priority: f32) -> Self {
        Self {
            operation,
            preferred_layout,
            priority,
        }
    }

    /// High priority hint for convolution operations
    pub fn convolution_hint(layout: DataLayout) -> Self {
        Self::new(OperationType::Convolution, layout, 3.0)
    }

    /// Medium priority hint for fully connected operations
    pub fn dense_hint(layout: DataLayout) -> Self {
        Self::new(OperationType::FullyConnected, layout, 2.0)
    }

    /// Low priority hint for element-wise operations
    pub fn elementwise_hint(layout: DataLayout) -> Self {
        Self::new(OperationType::ElementWise, layout, 0.5)
    }
}

/// Global layout optimization context
pub struct LayoutContext {
    optimizer: AutoLayoutOptimizer,
    /// Hints for upcoming operations
    hints: Vec<LayoutHint>,
    /// Enable/disable automatic optimization
    auto_optimize: bool,
}

impl LayoutContext {
    /// Create a new layout context
    pub fn new() -> Self {
        Self {
            optimizer: AutoLayoutOptimizer::new(),
            hints: Vec::new(),
            auto_optimize: true,
        }
    }

    /// Add a layout hint for future operations
    pub fn add_hint(&mut self, hint: LayoutHint) {
        self.hints.push(hint);
    }

    /// Enable or disable automatic layout optimization
    pub fn set_auto_optimize(&mut self, enable: bool) {
        self.auto_optimize = enable;
    }

    /// Get the best layout for a tensor considering all hints
    pub fn best_layout(
        &self,
        tensor_id: usize,
        op_type: OperationType,
        device: &Device,
    ) -> DataLayout {
        if !self.auto_optimize {
            return self
                .optimizer
                .get_layout(tensor_id)
                .unwrap_or(DataLayout::Auto);
        }

        // Consider hints first
        let mut best_layout = self.optimizer.optimizer.preferred_layout(device, op_type);
        let mut best_priority = 1.0;

        for hint in &self.hints {
            if hint.operation == op_type && hint.priority > best_priority {
                best_layout = hint.preferred_layout;
                best_priority = hint.priority;
            }
        }

        best_layout
    }

    /// Clear all hints
    pub fn clear_hints(&mut self) {
        self.hints.clear();
    }
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_permutations() {
        assert_eq!(
            DataLayout::NCHW.to_permutation(DataLayout::NHWC),
            Some(vec![0, 2, 3, 1])
        );
        assert_eq!(
            DataLayout::NHWC.to_permutation(DataLayout::NCHW),
            Some(vec![0, 3, 1, 2])
        );
    }

    #[test]
    fn test_layout_inference() {
        // Typical image tensor: small channel dim
        assert_eq!(infer_layout(&[32, 3, 224, 224], None), DataLayout::NCHW);

        // Typical feature map: large channel dim
        assert_eq!(infer_layout(&[32, 224, 224, 256], None), DataLayout::NHWC);
    }

    #[test]
    fn test_layout_optimizer() {
        let optimizer = LayoutOptimizer::default();

        // GPU should prefer NCHW for convolution (if GPU feature is enabled)
        #[cfg(feature = "gpu")]
        assert_eq!(
            optimizer.preferred_layout(&Device::Gpu(0), OperationType::Convolution),
            DataLayout::NCHW
        );

        // CPU should prefer NHWC for convolution
        assert_eq!(
            optimizer.preferred_layout(&Device::Cpu, OperationType::Convolution),
            DataLayout::NHWC
        );
    }

    #[test]
    fn test_conversion_costs() {
        let optimizer = LayoutOptimizer::default();

        assert_eq!(
            optimizer.conversion_cost(DataLayout::NCHW, DataLayout::NCHW),
            0.0
        );
        assert!(optimizer.conversion_cost(DataLayout::NCHW, DataLayout::NHWC) > 0.0);
    }

    #[test]
    fn test_auto_layout_optimizer() {
        let mut auto_optimizer = AutoLayoutOptimizer::new();

        // Register a tensor with NCHW layout
        auto_optimizer.register_tensor(0, DataLayout::NCHW);

        // Check initial layout
        assert_eq!(auto_optimizer.get_layout(0), Some(DataLayout::NCHW));

        // Check that total cost starts at 0
        assert_eq!(auto_optimizer.total_cost(), 0.0);
    }

    #[test]
    fn test_layout_hints() {
        let hint = LayoutHint::convolution_hint(DataLayout::NCHW);
        assert_eq!(hint.operation, OperationType::Convolution);
        assert_eq!(hint.preferred_layout, DataLayout::NCHW);
        assert_eq!(hint.priority, 3.0);

        let hint = LayoutHint::dense_hint(DataLayout::NHWC);
        assert_eq!(hint.operation, OperationType::FullyConnected);
        assert_eq!(hint.preferred_layout, DataLayout::NHWC);
        assert_eq!(hint.priority, 2.0);
    }

    #[test]
    fn test_layout_context() {
        let mut context = LayoutContext::new();

        // Add a convolution hint
        context.add_hint(LayoutHint::convolution_hint(DataLayout::NCHW));

        // Check that it returns the hinted layout
        let best_layout = context.best_layout(0, OperationType::Convolution, &Device::Cpu);
        assert_eq!(best_layout, DataLayout::NCHW);

        // Clear hints
        context.clear_hints();

        // Should now return device-preferred layout
        let best_layout = context.best_layout(0, OperationType::Convolution, &Device::Cpu);
        assert_eq!(best_layout, DataLayout::NHWC); // CPU prefers NHWC
    }

    #[test]
    fn test_layout_context_auto_optimize() {
        let mut context = LayoutContext::new();

        // Disable auto optimization
        context.set_auto_optimize(false);

        // Should return Auto layout when disabled
        let best_layout = context.best_layout(0, OperationType::Convolution, &Device::Cpu);
        assert_eq!(best_layout, DataLayout::Auto);

        // Re-enable auto optimization
        context.set_auto_optimize(true);

        // Should return device-preferred layout when enabled
        let best_layout = context.best_layout(0, OperationType::Convolution, &Device::Cpu);
        assert_eq!(best_layout, DataLayout::NHWC);
    }
}
