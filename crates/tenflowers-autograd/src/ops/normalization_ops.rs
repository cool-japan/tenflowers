//! Normalization Operations Module
//!
//! This module contains gradient operations for normalization functions including:
//! - Batch Normalization (BatchNorm)
//! - Layer Normalization (LayerNorm)
//! - Group Normalization (GroupNorm)
//! - Instance Normalization (InstanceNorm)

use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Helper function to calculate batch size for normalization operations
fn calculate_batch_size(shape: &[usize], axes: &[i32]) -> usize {
    axes.iter()
        .map(|&axis| {
            let idx = if axis < 0 {
                (shape.len() as i32 + axis) as usize
            } else {
                axis as usize
            };
            shape[idx]
        })
        .product()
}

/// Backward pass for batch normalization with running statistics updates
/// Computes gradients for input, gamma, and beta, and optionally updates running statistics
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    _running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    training: bool,
    epsilon: T,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
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
    let ndim = input_shape.len();

    // For batch norm, assume NCHW format (batch, channels, height, width)
    // We need to reduce over all dimensions except channel dimension (dimension 1)
    let axes: Vec<i32> = if ndim == 4 {
        // NCHW format: reduce over batch, height, width (dimensions 0, 2, 3)
        vec![0, 2, 3]
    } else if ndim == 2 {
        // NC format: reduce over batch (dimension 0)
        vec![0]
    } else {
        // Default: assume channel-last format for other cases
        (0..ndim - 1).map(|i| i as i32).collect()
    };

    if training {
        // Training mode: use batch statistics
        let batch_mean = input.mean(Some(&axes), true)?;
        let centered = input.sub(&batch_mean)?;
        let squared = centered.mul(&centered)?;
        let batch_var = squared.mean(Some(&axes), true)?;

        // Compute normalized input
        let eps_tensor = Tensor::from_scalar(epsilon);
        let std = batch_var.add(&eps_tensor)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Gradient w.r.t. gamma: sum(grad_output * normalized)
        let grad_gamma_full = grad_output.mul(&normalized)?;
        let grad_gamma = grad_gamma_full.sum(Some(&axes), false)?;

        // Gradient w.r.t. beta: sum(grad_output)
        let grad_beta = grad_output.sum(Some(&axes), false)?;

        // Gradient w.r.t. input (proper BatchNorm backward)
        // The full BatchNorm backward computation is:
        // grad_x = (1/m) * gamma / std * [m * grad_out - sum(grad_out) - x_normalized * sum(grad_out * x_normalized)]
        // where m is the batch size and x_normalized is the normalized input

        let batch_size_f =
            T::from_usize(calculate_batch_size(input_shape, &axes)).unwrap_or(T::one());

        // sum(grad_output) along batch dimensions
        let grad_sum = grad_output.sum(Some(&axes), true)?;

        // sum(grad_output * normalized) along batch dimensions
        let grad_norm_sum = grad_output.mul(&normalized)?.sum(Some(&axes), true)?;

        // normalized * sum(grad_output * normalized)
        let norm_grad_norm_sum = normalized.mul(&grad_norm_sum)?;

        // m * grad_output - sum(grad_output) - normalized * sum(grad_output * normalized)
        let batch_size_tensor = Tensor::from_scalar(batch_size_f);
        let m_grad_out = grad_output.mul(&batch_size_tensor)?;
        let diff1 = m_grad_out.sub(&grad_sum)?;
        let diff2 = diff1.sub(&norm_grad_norm_sum)?;

        // (1/m) * gamma / std * [...]
        let one_over_m = Tensor::from_scalar(T::one() / batch_size_f);

        // Reshape gamma to match std's shape for proper broadcasting
        let gamma_reshaped = if ndim == 4 {
            let channels = input_shape[1];
            gamma.reshape(&[1, channels, 1, 1])?
        } else {
            gamma.clone()
        };

        let gamma_over_std = gamma_reshaped.div(&std)?;
        let grad_input = diff2.mul(&one_over_m)?.mul(&gamma_over_std)?;

        Ok((grad_input, grad_gamma, grad_beta))
    } else {
        // Inference mode: use running statistics
        let eps_tensor = Tensor::from_scalar(epsilon);
        let std = running_var.add(&eps_tensor)?.sqrt()?;

        // Reshape gamma and std for proper broadcasting
        let (gamma_reshaped, std_reshaped) = if ndim == 4 {
            let channels = input_shape[1];
            let gamma_reshaped = gamma.reshape(&[1, channels, 1, 1])?;
            let std_reshaped = std.reshape(&[1, channels, 1, 1])?;
            (gamma_reshaped, std_reshaped)
        } else {
            (gamma.clone(), std)
        };

        // Gradient w.r.t. input
        let grad_input = grad_output.mul(&gamma_reshaped)?.div(&std_reshaped)?;

        // Gradients w.r.t. gamma and beta are zero in inference mode
        let grad_gamma = Tensor::zeros(gamma.shape().dims());
        let grad_beta = Tensor::zeros(beta.shape().dims());

        Ok((grad_input, grad_gamma, grad_beta))
    }
}

/// Batch normalization forward pass with running statistics updates
/// Returns the normalized output and optionally updates running statistics
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward_with_stats<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &mut Tensor<T>,
    running_var: &mut Tensor<T>,
    training: bool,
    momentum: T,
    epsilon: T,
) -> Result<Tensor<T>>
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
    let ndim = input_shape.len();

    // For batch norm, assume channel-last format
    let axes: Vec<i32> = (0..ndim - 1).map(|i| i as i32).collect();

    if training {
        // Training mode: compute batch statistics and update running statistics
        let batch_mean = input.mean(Some(&axes), true)?;
        let centered = input.sub(&batch_mean)?;
        let squared = centered.mul(&centered)?;
        let batch_var = squared.mean(Some(&axes), true)?;

        // Update running statistics with momentum
        // running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        let one_minus_momentum = T::one() - momentum;
        let momentum_tensor = Tensor::from_scalar(momentum);
        let one_minus_momentum_tensor = Tensor::from_scalar(one_minus_momentum);

        let updated_running_mean = running_mean
            .mul(&momentum_tensor)?
            .add(&batch_mean.mul(&one_minus_momentum_tensor)?)?;

        // For variance, use unbiased estimate: batch_var * N / (N - 1)
        let batch_size =
            T::from_usize(calculate_batch_size(input_shape, &axes)).unwrap_or(T::one());
        let bias_correction = batch_size / (batch_size - T::one());
        let unbiased_var = batch_var.mul(&Tensor::from_scalar(bias_correction))?;

        let updated_running_var = running_var
            .mul(&momentum_tensor)?
            .add(&unbiased_var.mul(&one_minus_momentum_tensor)?)?;

        // Update the running statistics tensors
        *running_mean = updated_running_mean;
        *running_var = updated_running_var;

        // Compute normalized output using batch statistics
        let eps_tensor = Tensor::from_scalar(epsilon);
        let std = batch_var.add(&eps_tensor)?.sqrt()?;
        let normalized = centered.div(&std)?;
        let output = normalized.mul(gamma)?.add(beta)?;

        Ok(output)
    } else {
        // Inference mode: use running statistics
        let eps_tensor = Tensor::from_scalar(epsilon);
        let std = running_var.add(&eps_tensor)?.sqrt()?;

        // Normalize using running statistics
        let centered = input.sub(running_mean)?;
        let normalized = centered.div(&std)?;
        let output = normalized.mul(gamma)?.add(beta)?;

        Ok(output)
    }
}

/// Enhanced batch normalization backward pass that can handle updated statistics
/// This version is aware that running statistics may have been updated during forward pass
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward_with_stats<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    training: bool,
    epsilon: T,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
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
    // This is the same as the original batch_norm_backward but with awareness of updated statistics
    // The backward pass computation doesn't change, but this function is explicit about handling
    // cases where running statistics have been updated during the forward pass
    batch_norm_backward(
        grad_output,
        input,
        gamma,
        beta,
        running_mean,
        running_var,
        training,
        epsilon,
    )
}

/// Backward pass for layer normalization
/// Computes gradients for input, gamma, and beta
/// LayerNorm normalizes across the last dimension(s) unlike BatchNorm which normalizes across batch
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    _beta: &Tensor<T>,
    normalized_shape: &[usize],
    epsilon: T,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
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
    let ndim = input_shape.len();

    // LayerNorm normalizes over the last normalized_shape.len() dimensions
    let norm_dims = normalized_shape.len();
    let reduce_axes: Vec<i32> = ((ndim - norm_dims)..ndim).map(|i| i as i32).collect();

    // Compute statistics over the normalization dimensions
    let mean = input.mean(Some(&reduce_axes), true)?;
    let centered = input.sub(&mean)?;
    let variance = centered.mul(&centered)?.mean(Some(&reduce_axes), true)?;

    // Compute normalized input
    let eps_tensor = Tensor::from_scalar(epsilon);
    let std = variance.add(&eps_tensor)?.sqrt()?;
    let normalized = centered.div(&std)?;

    // Number of elements being normalized over
    let norm_size: usize = normalized_shape.iter().product();
    let norm_size_f = T::from_usize(norm_size).unwrap_or(T::one());

    // Gradient w.r.t. gamma: sum(grad_output * normalized) over non-normalized dimensions
    let grad_gamma_full = grad_output.mul(&normalized)?;
    let non_norm_axes: Vec<i32> = (0..(ndim - norm_dims)).map(|i| i as i32).collect();
    let grad_gamma = if non_norm_axes.is_empty() {
        grad_gamma_full
    } else {
        grad_gamma_full.sum(Some(&non_norm_axes), false)?
    };

    // Gradient w.r.t. beta: sum(grad_output) over non-normalized dimensions
    let grad_beta = if non_norm_axes.is_empty() {
        grad_output.clone()
    } else {
        grad_output.sum(Some(&non_norm_axes), false)?
    };

    // Gradient w.r.t. input (complex LayerNorm backward computation)
    // The formula is similar to BatchNorm but applied to different dimensions:
    // grad_x = (1/N) * gamma / std * [N * grad_out - sum(grad_out) - normalized * sum(grad_out * normalized)]

    // sum(grad_output) over normalization dimensions
    let grad_sum = grad_output.sum(Some(&reduce_axes), true)?;

    // sum(grad_output * normalized) over normalization dimensions
    let grad_norm_sum = grad_output
        .mul(&normalized)?
        .sum(Some(&reduce_axes), true)?;

    // normalized * sum(grad_output * normalized)
    let norm_grad_norm_sum = normalized.mul(&grad_norm_sum)?;

    // N * grad_output - sum(grad_output) - normalized * sum(grad_output * normalized)
    let norm_size_tensor = Tensor::from_scalar(norm_size_f);
    let n_grad_out = grad_output.mul(&norm_size_tensor)?;
    let diff1 = n_grad_out.sub(&grad_sum)?;
    let diff2 = diff1.sub(&norm_grad_norm_sum)?;

    // (1/N) * gamma / std * [...]
    let one_over_n = Tensor::from_scalar(T::one() / norm_size_f);
    let gamma_over_std = gamma.div(&std)?;
    let grad_input = diff2.mul(&one_over_n)?.mul(&gamma_over_std)?;

    Ok((grad_input, grad_gamma, grad_beta))
}

/// Backward pass for group normalization
/// Computes gradients for input, gamma, and beta
/// GroupNorm divides channels into groups and normalizes within each group
#[allow(clippy::too_many_arguments)]
pub fn group_norm_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    _beta: &Tensor<T>,
    num_groups: usize,
    epsilon: T,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
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

    // Assume NCHW format: [batch_size, channels, height, width]
    if input_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "GroupNorm requires 4D input (NCHW format)".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    if channels % num_groups != 0 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Channels {channels} must be divisible by num_groups {num_groups}"
        )));
    }

    let channels_per_group = channels / num_groups;

    // Reshape input to [batch_size, num_groups, channels_per_group, height, width]
    let reshaped_input =
        input.reshape(&[batch_size, num_groups, channels_per_group, height, width])?;
    let reshaped_grad_output =
        grad_output.reshape(&[batch_size, num_groups, channels_per_group, height, width])?;

    // Normalize over [channels_per_group, height, width] dimensions (axes 2, 3, 4)
    let reduce_axes = vec![2i32, 3i32, 4i32];

    // Compute statistics
    let mean = reshaped_input.mean(Some(&reduce_axes), true)?;
    let centered = reshaped_input.sub(&mean)?;
    let variance = centered.mul(&centered)?.mean(Some(&reduce_axes), true)?;

    // Compute normalized input
    let eps_tensor = Tensor::from_scalar(epsilon);
    let std = variance.add(&eps_tensor)?.sqrt()?;
    let normalized = centered.div(&std)?;

    // Number of elements per group
    let group_size = channels_per_group * height * width;
    let group_size_f = T::from_usize(group_size).unwrap_or(T::one());

    // Gradient computation similar to LayerNorm but applied per group
    let grad_sum = reshaped_grad_output.sum(Some(&reduce_axes), true)?;
    let grad_norm_sum = reshaped_grad_output
        .mul(&normalized)?
        .sum(Some(&reduce_axes), true)?;
    let norm_grad_norm_sum = normalized.mul(&grad_norm_sum)?;

    let group_size_tensor = Tensor::from_scalar(group_size_f);
    let n_grad_out = reshaped_grad_output.mul(&group_size_tensor)?;
    let diff1 = n_grad_out.sub(&grad_sum)?;
    let diff2 = diff1.sub(&norm_grad_norm_sum)?;

    let one_over_n = Tensor::from_scalar(T::one() / group_size_f);

    // Reshape gamma and beta to broadcast correctly
    let gamma_reshaped =
        gamma
            .reshape(&[1, channels, 1, 1])?
            .reshape(&[1, num_groups, channels_per_group, 1, 1])?;
    let gamma_over_std = gamma_reshaped.div(&std)?;
    let grad_input_reshaped = diff2.mul(&one_over_n)?.mul(&gamma_over_std)?;

    // Reshape back to original shape
    let grad_input = grad_input_reshaped.reshape(input_shape)?;

    // Gradients for gamma and beta
    let normalized_original = normalized.reshape(input_shape)?;
    let grad_gamma_full = grad_output.mul(&normalized_original)?;
    // Sum over batch, height, width dimensions
    let grad_gamma = grad_gamma_full.sum(Some(&[0i32, 2i32, 3i32]), false)?;

    let grad_beta = grad_output.sum(Some(&[0i32, 2i32, 3i32]), false)?;

    Ok((grad_input, grad_gamma, grad_beta))
}

/// Backward pass for instance normalization
/// Computes gradients for input, gamma, and beta
/// InstanceNorm normalizes each sample and channel independently
#[allow(clippy::too_many_arguments)]
pub fn instance_norm_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    _beta: &Tensor<T>,
    epsilon: T,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
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

    // Assume NCHW format: [batch_size, channels, height, width]
    if input_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "InstanceNorm requires 4D input (NCHW format)".to_string(),
        ));
    }

    let _batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    // Normalize over spatial dimensions [height, width] for each (batch, channel) pair
    let reduce_axes = vec![2i32, 3i32];

    // Compute statistics
    let mean = input.mean(Some(&reduce_axes), true)?;
    let centered = input.sub(&mean)?;
    let variance = centered.mul(&centered)?.mean(Some(&reduce_axes), true)?;

    // Compute normalized input
    let eps_tensor = Tensor::from_scalar(epsilon);
    let std = variance.add(&eps_tensor)?.sqrt()?;
    let normalized = centered.div(&std)?;

    // Number of spatial elements
    let spatial_size = height * width;
    let spatial_size_f = T::from_usize(spatial_size).unwrap_or(T::one());

    // Gradient computation
    let grad_sum = grad_output.sum(Some(&reduce_axes), true)?;
    let grad_norm_sum = grad_output
        .mul(&normalized)?
        .sum(Some(&reduce_axes), true)?;
    let norm_grad_norm_sum = normalized.mul(&grad_norm_sum)?;

    let spatial_size_tensor = Tensor::from_scalar(spatial_size_f);
    let n_grad_out = grad_output.mul(&spatial_size_tensor)?;
    let diff1 = n_grad_out.sub(&grad_sum)?;
    let diff2 = diff1.sub(&norm_grad_norm_sum)?;

    let one_over_n = Tensor::from_scalar(T::one() / spatial_size_f);

    // Reshape gamma to broadcast correctly: [1, channels, 1, 1]
    let gamma_reshaped = gamma.reshape(&[1, channels, 1, 1])?;
    let gamma_over_std = gamma_reshaped.div(&std)?;
    let grad_input = diff2.mul(&one_over_n)?.mul(&gamma_over_std)?;

    // Gradients for gamma and beta
    let grad_gamma_full = grad_output.mul(&normalized)?;
    // Sum over batch and spatial dimensions
    let grad_gamma = grad_gamma_full.sum(Some(&[0i32, 2i32, 3i32]), false)?;

    let grad_beta = grad_output.sum(Some(&[0i32, 2i32, 3i32]), false)?;

    Ok((grad_input, grad_gamma, grad_beta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Shape;

    #[test]
    fn test_batch_norm_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[2, 4, 8, 8]);
        let gamma = Tensor::<f32>::ones(&[4]);
        let beta = Tensor::<f32>::zeros(&[4]);
        let running_mean = Tensor::<f32>::zeros(&[4]);
        let running_var = Tensor::<f32>::ones(&[4]);
        let grad_output = Tensor::<f32>::ones(&[2, 4, 8, 8]);

        let result = batch_norm_backward(
            &grad_output,
            &input,
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            true,
            1e-5_f32,
        );
        assert!(result.is_ok());

        if let Ok((grad_input, grad_gamma, grad_beta)) = result {
            assert_eq!(grad_input.shape().dims(), &[2, 4, 8, 8]);
            assert_eq!(grad_gamma.shape().dims(), &[4]);
            assert_eq!(grad_beta.shape().dims(), &[4]);
        }
    }

    #[test]
    fn test_layer_norm_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[2, 8]);
        let gamma = Tensor::<f32>::ones(&[8]);
        let beta = Tensor::<f32>::zeros(&[8]);
        let grad_output = Tensor::<f32>::ones(&[2, 8]);
        let normalized_shape = vec![8];

        let result = layer_norm_backward(
            &grad_output,
            &input,
            &gamma,
            &beta,
            &normalized_shape,
            1e-5_f32,
        );
        assert!(result.is_ok());

        if let Ok((grad_input, grad_gamma, grad_beta)) = result {
            assert_eq!(grad_input.shape().dims(), &[2, 8]);
            assert_eq!(grad_gamma.shape().dims(), &[8]);
            assert_eq!(grad_beta.shape().dims(), &[8]);
        }
    }

    #[test]
    fn test_group_norm_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[1, 6, 4, 4]);
        let gamma = Tensor::<f32>::ones(&[6]);
        let beta = Tensor::<f32>::zeros(&[6]);
        let grad_output = Tensor::<f32>::ones(&[1, 6, 4, 4]);

        let result = group_norm_backward(&grad_output, &input, &gamma, &beta, 2, 1e-5_f32);
        assert!(result.is_ok());

        if let Ok((grad_input, grad_gamma, grad_beta)) = result {
            assert_eq!(grad_input.shape().dims(), &[1, 6, 4, 4]);
            assert_eq!(grad_gamma.shape().dims(), &[6]);
            assert_eq!(grad_beta.shape().dims(), &[6]);
        }
    }

    #[test]
    fn test_instance_norm_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[1, 3, 4, 4]);
        let gamma = Tensor::<f32>::ones(&[3]);
        let beta = Tensor::<f32>::zeros(&[3]);
        let grad_output = Tensor::<f32>::ones(&[1, 3, 4, 4]);

        let result = instance_norm_backward(&grad_output, &input, &gamma, &beta, 1e-5_f32);
        assert!(result.is_ok());

        if let Ok((grad_input, grad_gamma, grad_beta)) = result {
            assert_eq!(grad_input.shape().dims(), &[1, 3, 4, 4]);
            assert_eq!(grad_gamma.shape().dims(), &[3]);
            assert_eq!(grad_beta.shape().dims(), &[3]);
        }
    }
}
