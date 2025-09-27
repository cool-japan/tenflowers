use num_traits::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Backward pass for Global Average Pooling 2D
/// For y = global_avg_pool2d(x), where x has shape [N, C, H, W] and y has shape [N, C, 1, 1]
/// The gradient is uniformly distributed across the spatial dimensions
pub fn global_avg_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if input_shape.len() != 4 {
        return Err(TensorError::shape_mismatch(
            "global_avg_pool2d_backward",
            "4D tensor [N, C, H, W]",
            &format!("{input_shape:?}"),
        ));
    }

    let [n, c, h, w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let spatial_size = h * w;

    // Check that grad_output has the correct shape [N, C, 1, 1]
    let expected_output_shape = [n, c, 1, 1];
    if grad_output.shape().dims() != expected_output_shape {
        return Err(TensorError::shape_mismatch(
            "global_avg_pool2d_backward",
            &format!("{expected_output_shape:?}"),
            &format!("{:?}", grad_output.shape().dims()),
        ));
    }

    // The gradient from global average pooling is the output gradient divided by spatial size
    // and broadcast to the full input shape
    let scale_factor = T::from_usize(spatial_size).ok_or_else(|| {
        TensorError::invalid_argument("Could not convert spatial size to tensor type".to_string())
    })?;

    // Create scale tensor
    let scale_tensor = Tensor::from_scalar(scale_factor);

    // Scale the gradient
    let scaled_grad = grad_output.div(&scale_tensor)?;

    // Broadcast to input shape
    broadcast_to_shape(&scaled_grad, input_shape)
}

/// Backward pass for Global Max Pooling 2D
/// For y = global_max_pool2d(x), gradients flow only to the maximum element positions
pub fn global_max_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    max_indices: &Tensor<usize>, // Pre-computed indices of max elements
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(TensorError::shape_mismatch(
            "global_max_pool2d_backward",
            "4D tensor [N, C, H, W]",
            &format!("{input_shape:?}"),
        ));
    }

    let [n, c, h, w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    // Will create grad_input later

    // Get data arrays
    let grad_output_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access grad_output data".to_string())
    })?;

    let max_indices_data = max_indices.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access max_indices data".to_string())
    })?;

    // Note: as_mut_slice doesn't exist, we'll create new data vector

    // Route gradients to max positions
    let mut grad_input_data = vec![T::zero(); n * c * h * w];
    for batch in 0..n {
        for channel in 0..c {
            let output_idx = batch * c + channel;
            let max_spatial_idx = max_indices_data[output_idx];

            // Convert spatial index to full tensor index
            let spatial_h = max_spatial_idx / w;
            let spatial_w = max_spatial_idx % w;
            let input_idx = ((batch * c + channel) * h + spatial_h) * w + spatial_w;

            if input_idx < grad_input_data.len() && output_idx < grad_output_data.len() {
                grad_input_data[input_idx] = grad_output_data[output_idx].clone();
            }
        }
    }

    let grad_input = Tensor::from_vec(grad_input_data, input_shape)?;

    Ok(grad_input)
}

/// Backward pass for Adaptive Average Pooling 2D
/// Maps from output size back to input size with appropriate scaling
pub fn adaptive_avg_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    output_size: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive,
{
    if input_shape.len() != 4 {
        return Err(TensorError::shape_mismatch(
            "adaptive_avg_pool2d_backward",
            "4D tensor [N, C, H, W]",
            &format!("{input_shape:?}"),
        ));
    }

    let [n, c, input_h, input_w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let (output_h, output_w) = output_size;

    // Check grad_output shape
    let expected_output_shape = [n, c, output_h, output_w];
    if grad_output.shape().dims() != expected_output_shape {
        return Err(TensorError::shape_mismatch(
            "adaptive_avg_pool2d_backward",
            &format!("{expected_output_shape:?}"),
            &format!("{:?}", grad_output.shape().dims()),
        ));
    }

    // Calculate adaptive pooling parameters
    let stride_h = input_h as f64 / output_h as f64;
    let stride_w = input_w as f64 / output_w as f64;

    // Initialize gradient input
    // Will create grad_input later

    let grad_output_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access grad_output data".to_string())
    })?;

    // Distribute gradients from output to input
    let mut grad_input_data = vec![T::zero(); n * c * input_h * input_w];
    for batch in 0..n {
        for channel in 0..c {
            for out_h in 0..output_h {
                for out_w in 0..output_w {
                    // Calculate input region for this output position
                    let start_h = (out_h as f64 * stride_h).floor() as usize;
                    let end_h = ((out_h + 1) as f64 * stride_h).ceil() as usize;
                    let start_w = (out_w as f64 * stride_w).floor() as usize;
                    let end_w = ((out_w + 1) as f64 * stride_w).ceil() as usize;

                    let end_h = end_h.min(input_h);
                    let end_w = end_w.min(input_w);

                    // Calculate the area and scale factor
                    let area = (end_h - start_h) * (end_w - start_w);
                    let scale = T::from_usize(area).unwrap_or_else(|| T::one());

                    // Get output gradient
                    let output_idx = ((batch * c + channel) * output_h + out_h) * output_w + out_w;
                    let output_grad = &grad_output_data[output_idx];

                    // Distribute to input region
                    for in_h in start_h..end_h {
                        for in_w in start_w..end_w {
                            let input_idx =
                                ((batch * c + channel) * input_h + in_h) * input_w + in_w;
                            if input_idx < grad_input_data.len() {
                                grad_input_data[input_idx] = grad_input_data[input_idx].clone()
                                    + output_grad.clone().div(scale.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    let grad_input = Tensor::from_vec(grad_input_data, input_shape)?;

    Ok(grad_input)
}

/// Backward pass for Fractional Adaptive Average Pooling 2D
/// Supports sub-pixel precision pooling with fractional overlap regions
pub fn fractional_adaptive_avg_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    output_size: (usize, usize),
    alpha: T, // Fractional overlap parameter (0.0 to 1.0)
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + num_traits::Float,
{
    if input_shape.len() != 4 {
        return Err(TensorError::shape_mismatch(
            "fractional_adaptive_avg_pool2d_backward",
            "4D tensor [N, C, H, W]",
            &format!("{input_shape:?}"),
        ));
    }

    let [n, c, input_h, input_w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let (output_h, output_w) = output_size;

    // Check grad_output shape
    let expected_output_shape = [n, c, output_h, output_w];
    if grad_output.shape().dims() != expected_output_shape {
        return Err(TensorError::shape_mismatch(
            "fractional_adaptive_avg_pool2d_backward",
            &format!("{expected_output_shape:?}"),
            &format!("{:?}", grad_output.shape().dims()),
        ));
    }

    // Calculate fractional adaptive pooling parameters
    let stride_h = T::from_usize(input_h).unwrap() / T::from_usize(output_h).unwrap();
    let stride_w = T::from_usize(input_w).unwrap() / T::from_usize(output_w).unwrap();

    let grad_output_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access grad_output data".to_string())
    })?;

    // Initialize gradient input with fractional precision
    let mut grad_input_data = vec![T::zero(); n * c * input_h * input_w];

    for batch in 0..n {
        for channel in 0..c {
            for out_h in 0..output_h {
                for out_w in 0..output_w {
                    // Calculate fractional input region for this output position
                    let center_h =
                        (T::from_usize(out_h).unwrap() + T::from(0.5).unwrap()) * stride_h;
                    let center_w =
                        (T::from_usize(out_w).unwrap() + T::from(0.5).unwrap()) * stride_w;

                    // Calculate fractional pooling region size
                    let region_h = stride_h * (T::one() + alpha);
                    let region_w = stride_w * (T::one() + alpha);

                    let start_h = center_h - region_h / T::from(2.0).unwrap();
                    let end_h = center_h + region_h / T::from(2.0).unwrap();
                    let start_w = center_w - region_w / T::from(2.0).unwrap();
                    let end_w = center_w + region_w / T::from(2.0).unwrap();

                    // Convert to integer bounds with fractional weighting
                    let start_h_int = start_h
                        .floor()
                        .max(T::zero())
                        .min(T::from_usize(input_h - 1).unwrap())
                        .to_usize()
                        .unwrap_or(0);
                    let end_h_int = end_h
                        .ceil()
                        .max(T::one())
                        .min(T::from_usize(input_h).unwrap())
                        .to_usize()
                        .unwrap_or(input_h);
                    let start_w_int = start_w
                        .floor()
                        .max(T::zero())
                        .min(T::from_usize(input_w - 1).unwrap())
                        .to_usize()
                        .unwrap_or(0);
                    let end_w_int = end_w
                        .ceil()
                        .max(T::one())
                        .min(T::from_usize(input_w).unwrap())
                        .to_usize()
                        .unwrap_or(input_w);

                    // Calculate total weight for normalization
                    let mut total_weight = T::zero();

                    // First pass: calculate total weight
                    for in_h in start_h_int..end_h_int {
                        for in_w in start_w_int..end_w_int {
                            let h_pos = T::from_usize(in_h).unwrap() + T::from(0.5).unwrap();
                            let w_pos = T::from_usize(in_w).unwrap() + T::from(0.5).unwrap();

                            // Calculate fractional overlap weight
                            let h_overlap = T::one()
                                - (h_pos - center_h).abs() / (region_h / T::from(2.0).unwrap());
                            let w_overlap = T::one()
                                - (w_pos - center_w).abs() / (region_w / T::from(2.0).unwrap());

                            let weight = h_overlap.max(T::zero()) * w_overlap.max(T::zero());
                            total_weight = total_weight + weight;
                        }
                    }

                    // Get output gradient
                    let output_idx = ((batch * c + channel) * output_h + out_h) * output_w + out_w;
                    let output_grad = &grad_output_data[output_idx];

                    // Second pass: distribute weighted gradients
                    if total_weight > T::zero() {
                        for in_h in start_h_int..end_h_int {
                            for in_w in start_w_int..end_w_int {
                                let h_pos = T::from_usize(in_h).unwrap() + T::from(0.5).unwrap();
                                let w_pos = T::from_usize(in_w).unwrap() + T::from(0.5).unwrap();

                                // Calculate fractional overlap weight
                                let h_overlap = T::one()
                                    - (h_pos - center_h).abs() / (region_h / T::from(2.0).unwrap());
                                let w_overlap = T::one()
                                    - (w_pos - center_w).abs() / (region_w / T::from(2.0).unwrap());

                                let weight = h_overlap.max(T::zero()) * w_overlap.max(T::zero());
                                let normalized_weight = weight / total_weight;

                                let input_idx =
                                    ((batch * c + channel) * input_h + in_h) * input_w + in_w;
                                if input_idx < grad_input_data.len()
                                    && normalized_weight > T::zero()
                                {
                                    grad_input_data[input_idx] = grad_input_data[input_idx]
                                        + *output_grad * normalized_weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let grad_input = Tensor::from_vec(grad_input_data, input_shape)?;

    Ok(grad_input)
}

/// Backward pass for Adaptive Max Pooling 2D
pub fn adaptive_max_pool2d_backward<T>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    max_indices: &Tensor<(usize, usize)>, // (h, w) indices for each output position
    output_size: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    if input_shape.len() != 4 {
        return Err(TensorError::shape_mismatch(
            "adaptive_max_pool2d_backward",
            "4D tensor [N, C, H, W]",
            &format!("{input_shape:?}"),
        ));
    }

    let [n, c, input_h, input_w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let (output_h, output_w) = output_size;

    // Initialize gradient input
    // Will create grad_input later

    let grad_output_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access grad_output data".to_string())
    })?;

    let max_indices_data = max_indices.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access max_indices data".to_string())
    })?;

    // Note: as_mut_slice doesn't exist, we'll create new data vector

    // Route gradients to max positions
    let mut grad_input_data = vec![T::zero(); n * c * input_h * input_w];
    for batch in 0..n {
        for channel in 0..c {
            for out_h in 0..output_h {
                for out_w in 0..output_w {
                    let output_idx = ((batch * c + channel) * output_h + out_h) * output_w + out_w;
                    let (max_h, max_w) = max_indices_data[output_idx];

                    let input_idx = ((batch * c + channel) * input_h + max_h) * input_w + max_w;

                    if input_idx < grad_input_data.len() && output_idx < grad_output_data.len() {
                        grad_input_data[input_idx] = grad_input_data[input_idx].clone()
                            + grad_output_data[output_idx].clone();
                    }
                }
            }
        }
    }

    let grad_input = Tensor::from_vec(grad_input_data, input_shape)?;

    Ok(grad_input)
}

/// Helper function to broadcast a tensor to a target shape
fn broadcast_to_shape<T>(tensor: &Tensor<T>, target_shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    // For now, implement a simple broadcast for the global pooling case
    // In a full implementation, this would be a more general broadcast operation

    let current_shape = tensor.shape().dims();

    // Handle the case where we're broadcasting [N, C, 1, 1] to [N, C, H, W]
    if current_shape.len() == 4 && target_shape.len() == 4 {
        let [n, c, curr_h, curr_w] = [
            current_shape[0],
            current_shape[1],
            current_shape[2],
            current_shape[3],
        ];
        let [target_n, target_c, target_h, target_w] = [
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
        ];

        if n == target_n && c == target_c && curr_h == 1 && curr_w == 1 {
            // Simple repeat operation for [N, C, 1, 1] -> [N, C, H, W]
            if let Some(data) = tensor.as_slice() {
                let mut broadcasted_data =
                    Vec::with_capacity(target_n * target_c * target_h * target_w);

                for batch in 0..target_n {
                    for channel in 0..target_c {
                        let value_idx = batch * target_c + channel;
                        let value = &data[value_idx];

                        // Repeat this value target_h * target_w times
                        for _ in 0..(target_h * target_w) {
                            broadcasted_data.push(value.clone());
                        }
                    }
                }

                return Tensor::from_vec(broadcasted_data, target_shape);
            }
        }
    }

    // Fallback for other cases
    Err(TensorError::unsupported_operation_simple(format!(
        "Broadcasting from {current_shape:?} to {target_shape:?} not implemented"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_global_avg_pool2d_backward() {
        // Input shape [1, 2, 3, 3] -> 1 batch, 2 channels, 3x3 spatial
        let input_shape = [1, 2, 3, 3];

        // Output gradient shape [1, 2, 1, 1]
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2, 1, 1]).unwrap();

        // Compute backward pass
        let grad_input = global_avg_pool2d_backward(&grad_output, &input_shape).unwrap();

        // Check that gradients are properly scaled and broadcasted
        assert_eq!(grad_input.shape().dims(), input_shape);

        if let Some(data) = grad_input.as_slice() {
            // Each spatial position should have gradient / spatial_size
            // Channel 0: 1.0 / 9.0 ≈ 0.111, Channel 1: 2.0 / 9.0 ≈ 0.222
            for i in 0..9 {
                assert!((data[i] - 1.0 / 9.0).abs() < 1e-6);
            }
            for i in 9..18 {
                assert!((data[i] - 2.0 / 9.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_fractional_adaptive_avg_pool2d_backward() {
        let input_shape = [1, 1, 4, 4];
        let output_size = (2, 2);
        let alpha = 0.5f32; // 50% fractional overlap

        // Create a simple grad_output
        let grad_output_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let grad_output = Tensor::from_vec(grad_output_data, &[1, 1, 2, 2]).unwrap();

        let grad_input =
            fractional_adaptive_avg_pool2d_backward(&grad_output, &input_shape, output_size, alpha)
                .unwrap();

        // Check that the output has the right shape
        assert_eq!(grad_input.shape().dims(), input_shape);

        // Check that gradients are distributed with fractional weights
        let grad_input_data = grad_input.as_slice().unwrap();

        // The sum of gradients should be preserved (energy conservation)
        let total_input_grad: f32 = grad_input_data.iter().sum();
        let total_output_grad: f32 = 1.0 + 2.0 + 3.0 + 4.0;

        assert!((total_input_grad - total_output_grad).abs() < 1e-5);

        // All gradients should be non-negative due to the fractional weighting
        for &val in grad_input_data {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_adaptive_avg_pool2d_backward() {
        // Input shape [1, 1, 4, 4] -> output size (2, 2)
        let input_shape = [1, 1, 4, 4];
        let output_size = (2, 2);

        // Output gradient shape [1, 1, 2, 2]
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();

        // Compute backward pass
        let grad_input =
            adaptive_avg_pool2d_backward(&grad_output, &input_shape, output_size).unwrap();

        // Check shape
        assert_eq!(grad_input.shape().dims(), input_shape);

        // The gradient should be distributed across input regions
        if let Some(data) = grad_input.as_slice() {
            // Each 2x2 region in input should get 1/4 of the corresponding output gradient
            assert!((data[0] - 1.0 / 4.0).abs() < 1e-6); // First region gets grad 1.0
        }
    }
}
