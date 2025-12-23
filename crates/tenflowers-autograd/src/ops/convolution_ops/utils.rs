use super::types::get_tensor_element_4d;
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Helper function to compute Conv2D input gradient using transposed convolution
pub(crate) fn compute_conv2d_input_gradient<T>(
    _grad_output: &Tensor<T>,
    _weight: &Tensor<T>,
    input_shape: &[usize],
    _stride: (usize, usize),
    _padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + scirs2_core::num_traits::Float,
{
    // This is a placeholder implementation for the input gradient computation
    // In a full implementation, this would compute the transposed convolution
    // grad_x = conv_transpose2d(grad_y, w_flipped, stride, padding)

    // For now, return a zero tensor with the same shape as input
    let zero_data = vec![T::zero(); input_shape.iter().product()];
    Ok(Tensor::from_vec(zero_data, input_shape).unwrap())
}

/// Helper function to compute Conv2D weight gradient
pub(crate) fn compute_conv2d_weight_gradient<T>(
    _input: &Tensor<T>,
    _grad_output: &Tensor<T>,
    weight_shape: &[usize],
    _stride: (usize, usize),
    _padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + scirs2_core::num_traits::Float,
{
    // This is a placeholder implementation for the weight gradient computation
    // In a full implementation, this would compute:
    // grad_w = conv2d(x, grad_y, stride=1, padding='valid') with proper axis arrangements

    // For now, return a zero tensor with the same shape as weight
    let zero_data = vec![T::zero(); weight_shape.iter().product()];
    Ok(Tensor::from_vec(zero_data, weight_shape).unwrap())
}

/// Helper function to compute Conv3D input gradient using transposed convolution
pub(crate) fn compute_conv3d_input_gradient<T>(
    _grad_output: &Tensor<T>,
    _weight: &Tensor<T>,
    input_shape: &[usize],
    _stride: (usize, usize, usize),
    _padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + scirs2_core::num_traits::Float,
{
    // This is a placeholder implementation for the 3D input gradient computation
    // For now, return a zero tensor with the same shape as input
    let zero_data = vec![T::zero(); input_shape.iter().product()];
    Ok(Tensor::from_vec(zero_data, input_shape).unwrap())
}

/// Helper function to compute Conv3D weight gradient
pub(crate) fn compute_conv3d_weight_gradient<T>(
    _input: &Tensor<T>,
    _grad_output: &Tensor<T>,
    weight_shape: &[usize],
    _stride: (usize, usize, usize),
    _padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + scirs2_core::num_traits::Float,
{
    // This is a placeholder implementation for the 3D weight gradient computation
    // For now, return a zero tensor with the same shape as weight
    let zero_data = vec![T::zero(); weight_shape.iter().product()];
    Ok(Tensor::from_vec(zero_data, weight_shape).unwrap())
}

/// Helper function to compute ConvTranspose2D input gradient
pub(crate) fn compute_conv_transpose2d_input_gradient<T>(
    grad_output: &Tensor<T>,
    weight: &Tensor<T>,
    input_shape: &[usize],
    stride: (usize, usize),
    padding: &str,
    _output_padding: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For transposed convolution, the gradient w.r.t. input is computed by:
    // 1. Flipping the weights (rotating 180 degrees)
    // 2. Performing regular convolution of grad_output with flipped weights

    // For now, implement a simplified version that works for stride=(1,1), padding='same'
    if stride != (1, 1) || padding != "same" {
        // For unsupported stride/padding combinations, return zeros for now
        return Ok(Tensor::zeros(input_shape));
    }

    // The gradient w.r.t. input for transposed conv is a regular convolution
    // with the weight tensor transposed in the channel dimensions
    correlate_transpose_conv_input(grad_output, weight, input_shape)
}

/// Helper function to compute ConvTranspose2D weight gradient
pub(crate) fn compute_conv_transpose2d_weight_gradient<T>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    weight_shape: &[usize],
    stride: (usize, usize),
    padding: &str,
    _output_padding: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For transposed convolution, the gradient w.r.t. weight is computed by:
    // correlating the input with the grad_output, but with proper indexing for transposed conv

    if stride != (1, 1) || padding != "same" {
        // For unsupported stride/padding combinations, return zeros for now
        return Ok(Tensor::zeros(weight_shape));
    }

    // The gradient w.r.t. weight for transposed conv
    correlate_transpose_conv_weight(input, grad_output, weight_shape)
}

/// Helper function to slice tensor along channel dimension
pub(crate) fn slice_tensor_channels<T>(
    tensor: &Tensor<T>,
    start: usize,
    end: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // This is a placeholder implementation for tensor channel slicing
    // In a full implementation, this would slice the tensor along the specified channel range
    let shape = tensor.shape().dims();
    let mut new_shape = shape.to_vec();
    new_shape[1] = end - start; // Update channel dimension

    let zero_data = vec![T::zero(); new_shape.iter().product()];
    Ok(Tensor::from_vec(zero_data, &new_shape).unwrap())
}

/// Helper function to concatenate tensors along specified axis
#[allow(dead_code)]
pub(crate) fn concatenate_tensors<T>(tensors: &[Tensor<T>], _axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // This is a placeholder implementation for tensor concatenation
    // In a full implementation, this would concatenate tensors along the specified axis
    if tensors.is_empty() {
        return Err(TensorError::invalid_operation_simple(
            "Cannot concatenate empty tensor list".to_string(),
        ));
    }

    // For now, return the first tensor as a placeholder
    Ok(tensors[0].clone())
}

/// Correlate for transposed convolution input gradient
pub(crate) fn correlate_transpose_conv_input<T>(
    grad_output: &Tensor<T>,
    weight: &Tensor<T>,
    input_shape: &[usize],
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
        + scirs2_core::num_traits::Float
        + bytemuck::Pod,
{
    // Extract shapes
    let grad_output_shape = grad_output.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    // Weight shape for transposed conv: [in_channels, out_channels, kernel_h, kernel_w]
    let out_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    let grad_out_height = grad_output_shape[2];
    let grad_out_width = grad_output_shape[3];

    // Initialize output data
    let total_elements = batch_size * in_channels * in_height * in_width;
    let mut grad_input_data = vec![T::zero(); total_elements];

    // Helper to calculate flat index for input
    let input_index = |b: usize, ic: usize, h: usize, w: usize| -> usize {
        b * in_channels * in_height * in_width + ic * in_height * in_width + h * in_width + w
    };

    // For each batch
    for b in 0..batch_size {
        // For each input channel
        for ic in 0..in_channels {
            // For each output channel
            for oc in 0..out_channels {
                // For each position in the gradient output
                for gy in 0..grad_out_height {
                    for gx in 0..grad_out_width {
                        // Get the gradient value at this position
                        if let Some(grad_val) = get_tensor_element_4d(grad_output, b, oc, gy, gx) {
                            // For each kernel position
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    // Calculate input position
                                    let input_y = gy + ky;
                                    let input_x = gx + kx;

                                    // Check bounds
                                    if input_y < in_height && input_x < in_width {
                                        // Get weight value (note: transposed conv weight indexing)
                                        if let Some(weight_val) =
                                            get_tensor_element_4d(weight, ic, oc, ky, kx)
                                        {
                                            // Accumulate gradient
                                            let contribution = grad_val * weight_val;
                                            let idx = input_index(b, ic, input_y, input_x);
                                            grad_input_data[idx] =
                                                grad_input_data[idx] + contribution;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(grad_input_data, input_shape)
}

/// Correlate for transposed convolution weight gradient
pub(crate) fn correlate_transpose_conv_weight<T>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    weight_shape: &[usize],
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
        + scirs2_core::num_traits::Float
        + bytemuck::Pod,
{
    // Extract shapes
    let input_shape = input.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    // Weight shape for transposed conv: [in_channels, out_channels, kernel_h, kernel_w]
    let out_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    let grad_out_height = grad_output_shape[2];
    let grad_out_width = grad_output_shape[3];

    // Initialize weight gradient data
    let total_weight_elements = in_channels * out_channels * kernel_height * kernel_width;
    let mut grad_weight_data = vec![T::zero(); total_weight_elements];

    // Helper to calculate flat index for weight
    let weight_index = |ic: usize, oc: usize, ky: usize, kx: usize| -> usize {
        ic * out_channels * kernel_height * kernel_width
            + oc * kernel_height * kernel_width
            + ky * kernel_width
            + kx
    };

    // For each input channel
    for ic in 0..in_channels {
        // For each output channel
        for oc in 0..out_channels {
            // For each kernel position
            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let mut weight_grad_sum = T::zero();

                    // Sum over all batches and spatial positions
                    for b in 0..batch_size {
                        for gy in 0..grad_out_height {
                            for gx in 0..grad_out_width {
                                // Calculate corresponding input position
                                let input_y = gy + ky;
                                let input_x = gx + kx;

                                // Check bounds
                                if input_y < in_height && input_x < in_width {
                                    // Get input and grad_output values
                                    if let (Some(input_val), Some(grad_val)) = (
                                        get_tensor_element_4d(input, b, ic, input_y, input_x),
                                        get_tensor_element_4d(grad_output, b, oc, gy, gx),
                                    ) {
                                        // Accumulate gradient
                                        weight_grad_sum = weight_grad_sum + (input_val * grad_val);
                                    }
                                }
                            }
                        }
                    }

                    // Set the accumulated gradient
                    let idx = weight_index(ic, oc, ky, kx);
                    grad_weight_data[idx] = weight_grad_sum;
                }
            }
        }
    }

    Tensor::from_vec(grad_weight_data, weight_shape)
}

/// Helper function to slice a 4D tensor
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn slice_tensor_4d<T>(
    tensor: &Tensor<T>,
    n_start: usize,
    n_end: usize,
    c_start: usize,
    c_end: usize,
    h_start: usize,
    h_end: usize,
    w_start: usize,
    w_end: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = tensor.shape().dims();
    let [n, c, h, w] = [shape[0], shape[1], shape[2], shape[3]];

    // Validate slice bounds
    if n_end > n || c_end > c || h_end > h || w_end > w {
        return Err(TensorError::InvalidArgument {
            operation: "tensor_setitem_backward".to_string(),
            reason: format!("Index out of bounds: [{n_start}:{n_end}, {c_start}:{c_end}, {h_start}:{h_end}, {w_start}:{w_end}] for shape {shape:?}"),
            context: None,
        });
    }

    // Use the proper tensor slicing operation
    let ranges = [
        n_start..n_end,
        c_start..c_end,
        h_start..h_end,
        w_start..w_end,
    ];

    tensor.slice(&ranges)
}

#[cfg(test)]
mod tests {
    use super::super::{avg_pool2d_backward, conv2d_backward, max_pool2d_backward};
    use tenflowers_core::Tensor;

    #[test]
    fn test_conv2d_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[1, 3, 32, 32]);
        let weight = Tensor::<f32>::zeros(&[64, 3, 3, 3]);
        let grad_output = Tensor::<f32>::zeros(&[1, 64, 30, 30]);

        let result = conv2d_backward(&grad_output, &input, &weight, None, (1, 1), "valid");
        assert!(result.is_ok());

        let (grad_input, grad_weight, grad_bias) = result.unwrap();
        assert_eq!(grad_input.shape().dims(), &[1, 3, 32, 32]);
        assert_eq!(grad_weight.shape().dims(), &[64, 3, 3, 3]);
        assert!(grad_bias.is_none());
    }

    #[test]
    fn test_max_pool2d_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[1, 3, 32, 32]);
        let grad_output = Tensor::<f32>::zeros(&[1, 3, 16, 16]);

        let result = max_pool2d_backward(&grad_output, &input, (2, 2), (2, 2), "valid", (1, 1));
        assert!(result.is_ok());

        let grad_input = result.unwrap();
        assert_eq!(grad_input.shape().dims(), &[1, 3, 32, 32]);
    }

    #[test]
    fn test_avg_pool2d_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[1, 3, 32, 32]);
        let grad_output = Tensor::<f32>::zeros(&[1, 3, 16, 16]);

        let result = avg_pool2d_backward(&grad_output, &input, (2, 2), (2, 2), "valid");
        assert!(result.is_ok());

        let grad_input = result.unwrap();
        assert_eq!(grad_input.shape().dims(), &[1, 3, 32, 32]);
    }
}
