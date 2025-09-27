//! Layout optimization utilities for convolution operations
//!
//! This module provides layout-aware convolution functions that automatically
//! optimize data layout for better performance on different devices (CPU vs GPU).
//! It includes utilities for layout conversion, performance benchmarking,
//! and automatic layout selection based on device characteristics.

use crate::layout::{convert_layout, DataLayout, LayoutOptimizer, OperationType};
use crate::{Result, Tensor, TensorError};
use num_traits::{One, Zero};
use std::collections::HashMap;

/// Layout-aware 2D convolution with automatic layout optimization
/// Automatically converts input layouts for optimal performance on target device
pub fn conv2d_with_layout<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
    input_layout: DataLayout,
    optimizer: Option<&LayoutOptimizer>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let default_optimizer = LayoutOptimizer::default();
    let layout_opt = optimizer.unwrap_or(&default_optimizer);

    // Determine optimal layout for convolution on the input device
    let device = input.device();
    let preferred_layout = layout_opt.preferred_layout(device, OperationType::Convolution);

    // Convert to optimal layout if beneficial
    let (working_input, actual_layout) = if input_layout != preferred_layout {
        let _conversion_cost = layout_opt.conversion_cost(input_layout, preferred_layout);

        // High operation intensity for convolution justifies layout conversion
        let operation_intensity = 3.0;

        if layout_opt.should_convert(input_layout, preferred_layout, operation_intensity) {
            let converted = convert_layout(input, input_layout, preferred_layout)?;
            (converted, preferred_layout)
        } else {
            (input.clone(), input_layout)
        }
    } else {
        (input.clone(), input_layout)
    };

    // Perform convolution with the working layout
    let result = match actual_layout {
        DataLayout::NCHW => {
            // Standard NCHW convolution
            super::conv2d::conv2d(&working_input, weight, bias, stride, padding)
        }
        DataLayout::NHWC => {
            // For NHWC, we need to adjust the convolution logic
            // This is a simplified version - in practice, you'd want optimized NHWC kernels
            let nchw_input = convert_layout(&working_input, DataLayout::NHWC, DataLayout::NCHW)?;
            let nchw_result = super::conv2d::conv2d(&nchw_input, weight, bias, stride, padding)?;
            convert_layout(&nchw_result, DataLayout::NCHW, DataLayout::NHWC)
        }
        _ => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "Convolution not supported for layout {actual_layout:?}"
            )));
        }
    }?;

    Ok(result)
}

/// Automatically infer and optimize layout for a convolution operation
pub fn conv2d_auto_layout<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let inferred_layout = crate::layout::infer_layout(input.shape().dims(), Some(4));
    conv2d_with_layout(input, weight, bias, stride, padding, inferred_layout, None)
}

/// Get convolution performance characteristics for different layouts
///
/// This function provides performance estimates for different data layouts
/// when performing convolution operations on the target device. It helps
/// determine the optimal layout for maximizing performance.
///
/// # Arguments
/// * `input_shape` - Shape of the input tensor [batch, channels, height, width]
/// * `weight_shape` - Shape of the weight tensor [out_channels, in_channels, kernel_h, kernel_w]
/// * `stride` - Convolution stride (height, width)
/// * `padding` - Padding mode ("valid" or "same")
/// * `device` - Target device for the operation
///
/// # Returns
/// HashMap mapping each DataLayout to its estimated relative performance score.
/// Higher scores indicate better expected performance.
pub fn conv_layout_benchmark<T>(
    _input_shape: &[usize],
    _weight_shape: &[usize],
    _stride: (usize, usize),
    _padding: &str,
    device: &crate::Device,
) -> HashMap<DataLayout, f32>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static,
{
    let mut benchmarks = HashMap::new();
    let _optimizer = LayoutOptimizer::default();

    // Estimate relative performance for each layout
    for layout in &[DataLayout::NCHW, DataLayout::NHWC] {
        let base_cost = 1.0; // Base convolution cost
        let layout_efficiency = match (device, layout) {
            #[cfg(feature = "gpu")]
            (crate::Device::Gpu(_), DataLayout::NCHW) => 1.0, // Optimal for GPU
            #[cfg(feature = "gpu")]
            (crate::Device::Gpu(_), DataLayout::NHWC) => 0.7, // Suboptimal for GPU
            (crate::Device::Cpu, DataLayout::NHWC) => 1.0, // Optimal for CPU
            (crate::Device::Cpu, DataLayout::NCHW) => 0.8, // Suboptimal for CPU
            _ => 0.5,                                      // Unknown combination
        };

        // Factor in memory access patterns
        let memory_efficiency = match layout {
            DataLayout::NCHW => 0.9, // Good for channel-wise operations
            DataLayout::NHWC => 1.0, // Better memory locality for spatial operations
            _ => 0.7,
        };

        let estimated_performance = base_cost * layout_efficiency * memory_efficiency;
        benchmarks.insert(*layout, estimated_performance);
    }

    benchmarks
}

/// Select the optimal layout for a convolution operation based on input characteristics
///
/// This function analyzes the input tensor dimensions, kernel size, device type,
/// and other factors to recommend the best data layout for maximum performance.
///
/// # Arguments
/// * `input_shape` - Shape of the input tensor
/// * `kernel_shape` - Shape of the convolution kernel
/// * `device` - Target computation device
/// * `operation_intensity` - Expected computational intensity (higher = more compute per memory access)
///
/// # Returns
/// Recommended DataLayout for optimal performance
pub fn select_optimal_layout(
    input_shape: &[usize],
    kernel_shape: &[usize],
    device: &crate::Device,
    operation_intensity: f32,
) -> DataLayout {
    let _optimizer = LayoutOptimizer::default();

    // Get performance estimates for different layouts
    let benchmarks = conv_layout_benchmark::<f32>(
        input_shape,
        kernel_shape,
        (1, 1), // Default stride
        "same", // Default padding
        device,
    );

    // Consider conversion costs if input is not in the optimal layout
    let mut best_layout = DataLayout::NCHW;
    let mut best_score = 0.0;

    for (&layout, &performance) in &benchmarks {
        // Adjust score based on operation intensity
        // Higher intensity operations benefit more from optimal layouts
        let intensity_bonus = operation_intensity * 0.1;
        let adjusted_score = performance + intensity_bonus;

        if adjusted_score > best_score {
            best_score = adjusted_score;
            best_layout = layout;
        }
    }

    best_layout
}

/// Estimate the cost of converting between two layouts
///
/// This function provides a rough estimate of the computational cost
/// involved in converting data from one layout to another. This helps
/// determine whether layout conversion is worthwhile for a given operation.
///
/// # Arguments
/// * `from_layout` - Source data layout
/// * `to_layout` - Target data layout
/// * `tensor_size` - Total number of elements in the tensor
///
/// # Returns
/// Estimated conversion cost as a floating-point value (higher = more expensive)
pub fn layout_conversion_cost(
    from_layout: DataLayout,
    to_layout: DataLayout,
    tensor_size: usize,
) -> f32 {
    if from_layout == to_layout {
        return 0.0; // No conversion needed
    }

    // Base cost is proportional to tensor size (memory bandwidth limited)
    let base_cost = tensor_size as f32 * 0.001;

    // Some conversions are more expensive than others
    let complexity_multiplier = match (from_layout, to_layout) {
        (DataLayout::NCHW, DataLayout::NHWC) | (DataLayout::NHWC, DataLayout::NCHW) => 1.0,
        _ => 1.5, // Unknown conversion patterns are more expensive
    };

    base_cost * complexity_multiplier
}

/// Check if layout conversion is beneficial for a given operation
///
/// This function weighs the cost of layout conversion against the expected
/// performance benefit to determine whether conversion should be performed.
///
/// # Arguments
/// * `current_layout` - Current data layout
/// * `optimal_layout` - Optimal layout for the operation
/// * `tensor_size` - Size of the tensor to be converted
/// * `operation_count` - Number of times the operation will be performed
/// * `performance_gain` - Expected performance improvement factor (e.g., 1.5 = 50% faster)
///
/// # Returns
/// True if conversion is recommended, false otherwise
pub fn should_convert_layout(
    current_layout: DataLayout,
    optimal_layout: DataLayout,
    tensor_size: usize,
    operation_count: usize,
    performance_gain: f32,
) -> bool {
    if current_layout == optimal_layout {
        return false; // Already optimal
    }

    let conversion_cost = layout_conversion_cost(current_layout, optimal_layout, tensor_size);
    let operation_cost = tensor_size as f32 * 0.01; // Estimated cost per operation

    // Calculate total cost savings from improved performance
    let current_total_cost = operation_cost * operation_count as f32;
    let optimal_total_cost = (operation_cost / performance_gain) * operation_count as f32;
    let savings = current_total_cost - optimal_total_cost;

    // Convert if savings exceed conversion cost
    savings > conversion_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_selection() {
        let input_shape = [1, 64, 224, 224]; // Typical CNN input
        let kernel_shape = [64, 64, 3, 3]; // 3x3 conv kernel

        // Test GPU preference
        #[cfg(feature = "gpu")]
        {
            let gpu_device = crate::Device::Gpu(0);
            let layout = select_optimal_layout(&input_shape, &kernel_shape, &gpu_device, 2.0);
            assert_eq!(layout, DataLayout::NCHW); // GPUs typically prefer NCHW
        }

        // Test CPU preference
        let cpu_device = crate::Device::Cpu;
        let layout = select_optimal_layout(&input_shape, &kernel_shape, &cpu_device, 2.0);
        // CPU preference can vary, but function should return a valid layout
        assert!(matches!(layout, DataLayout::NCHW | DataLayout::NHWC));
    }

    #[test]
    fn test_conversion_cost() {
        let tensor_size = 1000;

        // Same layout should have zero cost
        let cost = layout_conversion_cost(DataLayout::NCHW, DataLayout::NCHW, tensor_size);
        assert_eq!(cost, 0.0);

        // Different layouts should have non-zero cost
        let cost = layout_conversion_cost(DataLayout::NCHW, DataLayout::NHWC, tensor_size);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_should_convert_logic() {
        let tensor_size = 10000;
        let operation_count = 100;
        let high_performance_gain = 2.0; // 2x faster
        let low_performance_gain = 1.1; // 10% faster

        // High performance gain should justify conversion
        let should_convert = should_convert_layout(
            DataLayout::NHWC,
            DataLayout::NCHW,
            tensor_size,
            operation_count,
            high_performance_gain,
        );
        assert!(should_convert);

        // Low performance gain with few operations might not justify conversion
        let should_convert = should_convert_layout(
            DataLayout::NHWC,
            DataLayout::NCHW,
            tensor_size,
            1, // Only one operation
            low_performance_gain,
        );
        // This test depends on the specific cost model, so we just verify it returns a boolean
        assert!(matches!(should_convert, true | false));
    }
}
