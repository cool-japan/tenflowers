use super::conv2d::conv2d_backward;
use super::types::Conv2dBackwardResult;
use super::utils::slice_tensor_channels;
use num_traits::{One, Zero};
use tenflowers_core::ops::concat;
use tenflowers_core::{Shape, Tensor, TensorError};

/// Backward pass for Depthwise Convolution 2D
/// Depthwise convolution applies a separate filter to each input channel
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
    groups: usize,
) -> Conv2dBackwardResult<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Depthwise convolution is equivalent to grouped convolution with groups = in_channels
    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(TensorError::invalid_operation_simple(
            "Depthwise convolution requires 4D input".to_string(),
        ));
    }

    // Validate that groups equals input channels for depthwise convolution
    let expected_groups = input_shape[1]; // in_channels
    if groups != expected_groups {
        return Err(TensorError::InvalidArgument {
            operation: "depthwise_conv2d_backward".to_string(),
            reason: format!("Depthwise convolution requires groups ({groups}) to equal input channels ({expected_groups})"),
            context: None,
        });
    }

    grouped_conv2d_backward(grad_output, input, weight, bias, stride, padding, groups)
}

/// Backward pass for Grouped Convolution 2D
/// Grouped convolution divides input channels into groups and applies separate convolutions
#[allow(clippy::too_many_arguments)]
pub fn grouped_conv2d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
    groups: usize,
) -> Conv2dBackwardResult<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    if input_shape.len() != 4 || weight_shape.len() != 4 || grad_output_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Grouped Conv2D backward requires 4D tensors".to_string(),
        ));
    }

    if groups == 0 {
        return Err(TensorError::invalid_operation_simple(
            "Groups must be greater than 0".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    // Validate group parameters
    if in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(TensorError::invalid_operation_simple(format!(
            "Input channels ({in_channels}) and output channels ({out_channels}) must be divisible by groups ({groups})"
        )));
    }

    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    // Initialize result tensors
    let mut grad_input_groups = Vec::new();
    let mut grad_weight_groups = Vec::new();
    let mut grad_bias_groups = Vec::new();

    // Process each group separately
    for group in 0..groups {
        // Slice input for this group
        let input_start = group * in_channels_per_group;
        let input_end = (group + 1) * in_channels_per_group;
        let _input_group_shape =
            Shape::new(vec![batch_size, in_channels_per_group, in_height, in_width]);

        // Slice grad_output for this group
        let output_start = group * out_channels_per_group;
        let output_end = (group + 1) * out_channels_per_group;
        let _grad_output_group_shape = Shape::new(vec![
            batch_size,
            out_channels_per_group,
            grad_output_shape[2],
            grad_output_shape[3],
        ]);

        // Slice weight for this group
        let _weight_group_shape = Shape::new(vec![
            out_channels_per_group,
            in_channels_per_group,
            kernel_height,
            kernel_width,
        ]);

        // Create sliced tensors for this group
        let input_group = slice_tensor_channels(input, input_start, input_end)?;
        let grad_output_group = slice_tensor_channels(grad_output, output_start, output_end)?;
        let weight_group = slice_tensor_channels(weight, output_start, output_end)?;

        // Process bias for this group if it exists
        let bias_group = if let Some(bias_tensor) = bias {
            Some(slice_tensor_channels(
                bias_tensor,
                output_start,
                output_end,
            )?)
        } else {
            None
        };

        // Apply regular conv2d backward to this group
        let (grad_input_group, grad_weight_group, grad_bias_group) = conv2d_backward(
            &grad_output_group,
            &input_group,
            &weight_group,
            bias_group.as_ref(),
            stride,
            padding,
        )?;

        grad_input_groups.push(grad_input_group);
        grad_weight_groups.push(grad_weight_group);
        if let Some(bias_grad) = grad_bias_group {
            grad_bias_groups.push(bias_grad);
        }
    }

    // Concatenate results from all groups
    let grad_input_refs: Vec<&Tensor<T>> = grad_input_groups.iter().collect();
    let grad_weight_refs: Vec<&Tensor<T>> = grad_weight_groups.iter().collect();
    let grad_input = concat(&grad_input_refs, 1)?; // Concatenate along channel axis
    let grad_weight = concat(&grad_weight_refs, 0)?; // Concatenate along output channel axis

    let grad_bias = if !grad_bias_groups.is_empty() {
        let grad_bias_refs: Vec<&Tensor<T>> = grad_bias_groups.iter().collect();
        Some(concat(&grad_bias_refs, 0)?) // Concatenate along channel axis
    } else {
        None
    };

    Ok((grad_input, grad_weight, grad_bias))
}
