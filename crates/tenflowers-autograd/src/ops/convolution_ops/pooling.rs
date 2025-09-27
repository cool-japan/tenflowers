use num_traits::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Backward pass for 2D Max Pooling
/// For maxpool, gradients flow only to the input positions that were selected as maximum
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    pool_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
    dilation: (usize, usize),
) -> Result<Tensor<T>>
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
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_shape = input.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    // Expected shapes:
    // input: [batch_size, channels, in_height, in_width]
    // grad_output: [batch_size, channels, out_height, out_width]

    if input_shape.len() != 4 || grad_output_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "MaxPool2D backward requires 4D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_height = grad_output_shape[2];
    let out_width = grad_output_shape[3];

    // Create zero tensor for input gradients
    let mut grad_input_data = vec![T::zero(); input.shape().size()];

    // Compute padding values
    let (pad_h, pad_w) = match padding {
        "valid" => (0, 0),
        "same" => {
            let pad_h = ((out_height - 1) * stride.0 + dilation.0 * (pool_size.0 - 1) + 1)
                .saturating_sub(in_height)
                / 2;
            let pad_w = ((out_width - 1) * stride.1 + dilation.1 * (pool_size.1 - 1) + 1)
                .saturating_sub(in_width)
                / 2;
            (pad_h, pad_w)
        }
        _ => {
            return Err(TensorError::invalid_shape_simple(format!(
                "Unsupported padding mode: {padding}",
            )));
        }
    };

    // Extract raw data for input and grad_output
    let input_data = input.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!("Failed to get input data: {e}"))
    })?;
    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!("Failed to get grad_output data: {e}"))
    })?;

    // Iterate through each position in the output
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    // Calculate the receptive field in the input
                    let ih_start = (oh * stride.0).saturating_sub(pad_h);
                    let iw_start = (ow * stride.1).saturating_sub(pad_w);

                    let ih_end =
                        std::cmp::min(ih_start + dilation.0 * (pool_size.0 - 1) + 1, in_height);
                    let iw_end =
                        std::cmp::min(iw_start + dilation.1 * (pool_size.1 - 1) + 1, in_width);

                    // Find the position of the maximum value in this receptive field
                    let mut max_val = T::zero();
                    let mut max_ih = ih_start;
                    let mut max_iw = iw_start;
                    let mut first = true;

                    for ih in (ih_start..ih_end).step_by(dilation.0) {
                        for iw in (iw_start..iw_end).step_by(dilation.1) {
                            let input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                            let val = input_data[input_idx];

                            if first || val > max_val {
                                max_val = val;
                                max_ih = ih;
                                max_iw = iw;
                                first = false;
                            }
                        }
                    }

                    // Route gradient to the position with maximum value
                    let grad_output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    let grad_input_idx =
                        ((b * channels + c) * in_height + max_ih) * in_width + max_iw;

                    grad_input_data[grad_input_idx] =
                        grad_input_data[grad_input_idx] + grad_output_data[grad_output_idx];
                }
            }
        }
    }

    Tensor::from_vec(grad_input_data, input.shape().dims())
}

/// Backward pass for 2D Average Pooling
/// For avgpool, gradients are distributed uniformly across all input positions in each pooling window
#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    pool_size: (usize, usize),
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
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_shape = input.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    // Expected shapes:
    // input: [batch_size, channels, in_height, in_width]
    // grad_output: [batch_size, channels, out_height, out_width]

    if input_shape.len() != 4 || grad_output_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "AvgPool2D backward requires 4D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_height = grad_output_shape[2];
    let out_width = grad_output_shape[3];

    // Create zero tensor for input gradients
    let mut grad_input_data = vec![T::zero(); input.shape().size()];

    // Compute padding values
    let (pad_h, pad_w) = match padding {
        "valid" => (0, 0),
        "same" => {
            let pad_h = ((out_height - 1) * stride.0 + pool_size.0).saturating_sub(in_height) / 2;
            let pad_w = ((out_width - 1) * stride.1 + pool_size.1).saturating_sub(in_width) / 2;
            (pad_h, pad_w)
        }
        _ => {
            return Err(TensorError::invalid_shape_simple(format!(
                "Unsupported padding mode: {padding}",
            )));
        }
    };

    // Extract raw data for grad_output
    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!("Failed to get grad_output data: {e}"))
    })?;

    // Pool size as floating point for division
    let pool_area = T::from_usize(pool_size.0 * pool_size.1).unwrap_or_else(|| T::one());

    // Iterate through each position in the output
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    // Calculate the receptive field in the input
                    let ih_start = (oh * stride.0).saturating_sub(pad_h);
                    let iw_start = (ow * stride.1).saturating_sub(pad_w);

                    let ih_end = std::cmp::min(ih_start + pool_size.0, in_height);
                    let iw_end = std::cmp::min(iw_start + pool_size.1, in_width);

                    // Get the gradient for this output position
                    let grad_output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    let grad_val = grad_output_data[grad_output_idx];

                    // Count the number of valid positions in the pooling window
                    let valid_count = (ih_end - ih_start) * (iw_end - iw_start);
                    let valid_count_t = T::from_usize(valid_count).unwrap_or(pool_area);

                    // Distribute gradient evenly across all positions in the pooling window
                    let grad_per_position = grad_val / valid_count_t;

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            // Distribute gradient evenly across all positions in the pooling window
                            let grad_input_idx =
                                ((b * channels + c) * in_height + ih) * in_width + iw;
                            grad_input_data[grad_input_idx] =
                                grad_input_data[grad_input_idx] + grad_per_position;
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(grad_input_data, input.shape().dims())
}
