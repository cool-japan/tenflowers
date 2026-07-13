use super::conv1d_utils::{compute_conv1d_input_gradient, compute_conv1d_weight_gradient};
use super::types::Conv1dBackwardResult;
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Tensor, TensorError};

/// Backward pass for 1D Convolution
/// Computes gradients for input, weight, and bias
#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: &str,
) -> Conv1dBackwardResult<T>
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
        + scirs2_core::num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Conv1D backward pass computation (1D specialization of conv2d_backward):
    // For y = conv1d(x, w, stride, padding), given grad_y, we need:
    // - grad_x = transposed_conv1d(grad_y, w_flipped, stride, padding)
    // - grad_w = conv1d(x, grad_y, stride=1, padding='valid') with proper axis arrangements
    // - grad_bias = sum(grad_y) over batch and length dimensions

    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    // Expected shapes:
    // input: [batch_size, in_channels, in_length]
    // weight: [out_channels, in_channels, kernel_length]
    // grad_output: [batch_size, out_channels, out_length]

    if input_shape.len() != 3 || weight_shape.len() != 3 || grad_output_shape.len() != 3 {
        return Err(TensorError::invalid_shape_simple(
            "Conv1D backward requires 3D tensors".to_string(),
        ));
    }

    let _batch_size = input_shape[0];
    let _in_channels = input_shape[1];
    let _in_length = input_shape[2];

    let _out_channels = weight_shape[0];
    let _kernel_length = weight_shape[2];

    // Compute gradient w.r.t. bias (if bias exists)
    let grad_bias = if bias.is_some() {
        // Sum grad_output over batch and length dimensions, keeping only channels
        let axes = vec![0i32, 2i32]; // Sum over batch, length
        Some(grad_output.sum(Some(&axes), false)?)
    } else {
        None
    };

    // Compute gradient w.r.t. input using transposed convolution
    let grad_input =
        compute_conv1d_input_gradient(grad_output, weight, input_shape, stride, padding)?;

    // Compute gradient w.r.t. weight
    let grad_weight =
        compute_conv1d_weight_gradient(input, grad_output, weight_shape, stride, padding)?;

    Ok((grad_input, grad_weight, grad_bias))
}
