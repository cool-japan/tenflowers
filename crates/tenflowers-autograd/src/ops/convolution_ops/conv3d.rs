use super::types::Conv3dBackwardResult;
use super::utils::{compute_conv3d_input_gradient, compute_conv3d_weight_gradient};
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Tensor, TensorError};

/// Backward pass for 3D Convolution
/// Computes gradients for input, weight, and bias
#[allow(clippy::too_many_arguments)]
pub fn conv3d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
    padding: &str,
) -> Conv3dBackwardResult<T>
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
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    // Expected shapes:
    // input: [batch_size, in_channels, in_depth, in_height, in_width]
    // weight: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    // grad_output: [batch_size, out_channels, out_depth, out_height, out_width]

    if input_shape.len() != 5 || weight_shape.len() != 5 || grad_output_shape.len() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Conv3D backward requires 5D tensors".to_string(),
        ));
    }

    let _batch_size = input_shape[0];
    let _in_channels = input_shape[1];
    let _in_depth = input_shape[2];
    let _in_height = input_shape[3];
    let _in_width = input_shape[4];

    let _out_channels = weight_shape[0];
    let _kernel_depth = weight_shape[2];
    let _kernel_height = weight_shape[3];
    let _kernel_width = weight_shape[4];

    // Compute gradient w.r.t. bias (if bias exists)
    let grad_bias = if bias.is_some() {
        // Sum grad_output over batch, depth, height, width dimensions, keeping only channels
        let axes = vec![0i32, 2i32, 3i32, 4i32]; // Sum over batch, depth, height, width
        Some(grad_output.sum(Some(&axes), false)?)
    } else {
        None
    };

    // Compute gradient w.r.t. input using transposed 3D convolution
    let grad_input =
        compute_conv3d_input_gradient(grad_output, weight, input_shape, stride, padding)?;

    // Compute gradient w.r.t. weight
    let grad_weight =
        compute_conv3d_weight_gradient(input, grad_output, weight_shape, stride, padding)?;

    Ok((grad_input, grad_weight, grad_bias))
}
