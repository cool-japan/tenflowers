use super::types::ConvTranspose2dBackwardResult;
use super::utils::{
    compute_conv_transpose2d_input_gradient, compute_conv_transpose2d_weight_gradient,
};
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Tensor, TensorError};

/// Backward pass for 2D Transposed Convolution (Deconvolution)
/// Computes gradients for input, weight, and bias
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
    output_padding: (usize, usize),
) -> ConvTranspose2dBackwardResult<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    // Expected shapes for transposed conv:
    // input: [batch_size, in_channels, in_height, in_width]
    // weight: [in_channels, out_channels, kernel_height, kernel_width] (NOTE: different from regular conv!)
    // grad_output: [batch_size, out_channels, out_height, out_width]

    if input_shape.len() != 4 || weight_shape.len() != 4 || grad_output_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "ConvTranspose2D backward requires 4D tensors".to_string(),
        ));
    }

    let _batch_size = input_shape[0];
    let _in_channels = input_shape[1];
    let _in_height = input_shape[2];
    let _in_width = input_shape[3];

    let _out_channels = weight_shape[1];
    let _kernel_height = weight_shape[2];
    let _kernel_width = weight_shape[3];

    // Compute gradient w.r.t. bias (if bias exists)
    let grad_bias = if bias.is_some() {
        // Sum grad_output over batch, height, width dimensions, keeping only channels
        let axes = vec![0i32, 2i32, 3i32]; // Sum over batch, height, width
        Some(grad_output.sum(Some(&axes), false)?)
    } else {
        None
    };

    // For transposed convolution, the gradients are computed differently:
    // - grad_x = regular_conv2d(grad_y, w_transposed, stride, padding)
    // - grad_w = compute gradient w.r.t. transposed conv weight

    let grad_input = compute_conv_transpose2d_input_gradient(
        grad_output,
        weight,
        input_shape,
        stride,
        padding,
        output_padding,
    )?;

    let grad_weight = compute_conv_transpose2d_weight_gradient(
        input,
        grad_output,
        weight_shape,
        stride,
        padding,
        output_padding,
    )?;

    Ok((grad_input, grad_weight, grad_bias))
}
