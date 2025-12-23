//! Reduction Operations Module
//!
//! This module contains gradient implementations for reduction operations including:
//! - sum_backward: Sum reduction gradients
//! - mean_backward: Mean reduction gradients  
//! - weighted_mean_backward: Weighted mean reduction gradients
//! - max_backward: Maximum reduction gradients
//! - min_backward: Minimum reduction gradients
//! - var_backward: Variance computation gradients
//! - std_backward: Standard deviation computation gradients
//!
//! These operations handle the backward pass for tensor reduction operations,
//! properly managing gradient flow through dimension reductions.

#[allow(unused_imports)]
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use tenflowers_core::ops::broadcast_to;
use tenflowers_core::{Result, Tensor};

/// Backward pass for sum reduction
/// For y = sum(x), grad_x = grad_y broadcast to shape of x
pub fn sum_backward<T>(grad_output: &Tensor<T>, input_shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For sum, the gradient is just the output gradient broadcasted to input shape
    broadcast_to(grad_output, input_shape)
}

/// Backward pass for mean reduction
/// For y = mean(x), grad_x = grad_y / numel(x) broadcast to shape of x
pub fn mean_backward<T>(grad_output: &Tensor<T>, input_shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + std::ops::Div<Output = T>
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For mean, the gradient is the output gradient divided by number of elements
    let numel: usize = input_shape.iter().product();
    let numel_scalar =
        T::from_usize(numel).unwrap_or_else(|| panic!("Cannot convert {numel} to type T"));

    // Create a scalar tensor and divide
    let numel_tensor = Tensor::<T>::from_scalar(numel_scalar);
    let grad_scaled = grad_output.div(&numel_tensor)?;

    broadcast_to(&grad_scaled, input_shape)
}

/// Backward pass for weighted mean reduction
/// For y = weighted_mean(x, w) = sum(x * w) / sum(w),
/// grad_x = grad_y * w / sum(w), grad_w = grad_y * x / sum(w)
pub fn weighted_mean_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weights: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute sum of weights
    let weight_sum = weights.sum(axes, keepdims)?;

    // Gradient w.r.t. input: grad_y * w / sum(w)
    let grad_input_intermediate = grad_output.mul(weights)?;
    let grad_input = grad_input_intermediate.div(&weight_sum)?;

    // Gradient w.r.t. weights: grad_y * x / sum(w)
    let grad_weights_intermediate = grad_output.mul(input)?;
    let grad_weights = grad_weights_intermediate.div(&weight_sum)?;

    Ok((grad_input, grad_weights))
}

/// Backward pass for max operation
/// For z = max(a, axes), the gradient only flows to the maximum elements
pub fn max_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute the max values to find which elements are maximum
    let max_values = input.max(axes, keepdims)?;

    // Create a mask identifying maximum elements
    let max_mask = create_max_mask(input, &max_values, axes, keepdims)?;

    // Broadcast grad_output to input shape if needed
    let broadcasted_grad = if axes.is_some() && !keepdims {
        // grad_output needs to be expanded to match input dimensions
        let expanded_grad = expand_for_reduction(grad_output, input.shape().dims(), axes)?;
        expanded_grad
    } else {
        grad_output.clone()
    };

    // Apply the mask to route gradients only to maximum elements
    broadcasted_grad.mul(&max_mask)
}

/// Backward pass for min operation  
/// For z = min(a, axes), the gradient only flows to the minimum elements
pub fn min_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute the min values to find which elements are minimum
    let min_values = input.min(axes, keepdims)?;

    // Create a mask identifying minimum elements
    let min_mask = create_min_mask(input, &min_values, axes, keepdims)?;

    // Broadcast grad_output to input shape if needed
    let broadcasted_grad = if axes.is_some() && !keepdims {
        // grad_output needs to be expanded to match input dimensions
        let expanded_grad = expand_for_reduction(grad_output, input.shape().dims(), axes)?;
        expanded_grad
    } else {
        grad_output.clone()
    };

    // Apply the mask to route gradients only to minimum elements
    broadcasted_grad.mul(&min_mask)
}

/// Backward pass for variance computation
/// For y = var(x) = mean((x - mean(x))²)
/// grad_x = grad_y * 2 * (x - mean(x)) / n
pub fn var_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
    correction: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute the mean
    let mean_val = input.mean(axes, keepdims)?;

    // Compute (x - mean)
    let centered = input.sub(&mean_val)?;

    // Compute the number of elements being reduced over
    let input_shape = input.shape().dims();
    let numel = if let Some(axes) = axes {
        axes.iter()
            .map(|&axis| {
                let actual_axis = if axis < 0 {
                    input_shape.len() as i32 + axis
                } else {
                    axis
                } as usize;
                input_shape[actual_axis]
            })
            .product::<usize>()
    } else {
        input_shape.iter().product()
    };

    // Apply Bessel's correction if specified
    let denom = if correction > 0 {
        numel - correction
    } else {
        numel
    };
    let denom_scalar =
        T::from_usize(denom).unwrap_or_else(|| panic!("Cannot convert {denom} to type T"));

    // grad_x = grad_y * 2 * (x - mean) / n
    let two = T::from_u8(2).unwrap();
    let two_tensor = Tensor::from_scalar(two);
    let denom_tensor = Tensor::from_scalar(denom_scalar);

    let grad_intermediate = grad_output
        .mul(&two_tensor)?
        .mul(&centered)?
        .div(&denom_tensor)?;

    // Broadcast back to input shape if needed
    broadcast_to(&grad_intermediate, input_shape)
}

/// Backward pass for standard deviation computation
/// For y = std(x) = sqrt(var(x))
/// grad_x = grad_y * (1 / (2 * sqrt(var(x)))) * grad_var
pub fn std_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
    correction: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute variance using our own implementation since var method may not exist
    let mean_val = input.mean(axes, keepdims)?;
    let centered = input.sub(&mean_val)?;
    let squared = centered.mul(&centered)?;
    let var_val = squared.mean(axes, keepdims)?;

    // Compute std = sqrt(var)
    let std_val = var_val.sqrt()?;

    // Compute 1 / (2 * std)
    let two = T::from_u8(2).unwrap();
    let two_tensor = Tensor::from_scalar(two);
    let two_std = two_tensor.mul(&std_val)?;
    let grad_scale = grad_output.div(&two_std)?;

    // Compute variance gradient and apply the scaling
    let var_grad = var_backward(&grad_scale, input, axes, keepdims, correction)?;

    Ok(var_grad)
}

/// Helper function to create a mask for maximum elements
fn create_max_mask<T>(
    input: &Tensor<T>,
    max_values: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Broadcast max_values to input shape for comparison
    let broadcasted_max = if axes.is_some() && !keepdims {
        let expanded_max = expand_for_reduction(max_values, input.shape().dims(), axes)?;
        expanded_max
    } else {
        broadcast_to(max_values, input.shape().dims())?
    };

    // Create mask where input equals max values and convert bool to T
    let bool_mask = input.eq(&broadcasted_max)?;
    // Convert bool tensor to T tensor (1 for true, 0 for false)
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let one_tensor = Tensor::ones(input.shape().dims());
    tenflowers_core::ops::where_op(&bool_mask, &one_tensor, &zero_tensor)
}

/// Helper function to create a mask for minimum elements
fn create_min_mask<T>(
    input: &Tensor<T>,
    min_values: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + PartialOrd + Send + Sync + 'static + bytemuck::Pod,
{
    // Broadcast min_values to input shape for comparison
    let broadcasted_min = if axes.is_some() && !keepdims {
        let expanded_min = expand_for_reduction(min_values, input.shape().dims(), axes)?;
        expanded_min
    } else {
        broadcast_to(min_values, input.shape().dims())?
    };

    // Create mask where input equals min values and convert bool to T
    let bool_mask = input.eq(&broadcasted_min)?;
    // Convert bool tensor to T tensor (1 for true, 0 for false)
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let one_tensor = Tensor::ones(input.shape().dims());
    tenflowers_core::ops::where_op(&bool_mask, &one_tensor, &zero_tensor)
}

/// Helper function to expand tensors for reduction operations
fn expand_for_reduction<T>(
    tensor: &Tensor<T>,
    target_shape: &[usize],
    axes: Option<&[i32]>,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if let Some(axes) = axes {
        // Add dimensions back that were reduced
        let mut expanded_shape = vec![1; target_shape.len()];
        let mut _tensor_dim_idx = 0;

        for (i, &dim_size) in target_shape.iter().enumerate() {
            let axis_reduced = axes.iter().any(|&axis| {
                let actual_axis = if axis < 0 {
                    target_shape.len() as i32 + axis
                } else {
                    axis
                } as usize;
                actual_axis == i
            });

            if !axis_reduced {
                expanded_shape[i] = dim_size;
                _tensor_dim_idx += 1;
            }
        }

        // Reshape and then broadcast
        let reshaped = tensor.reshape(&expanded_shape)?;
        broadcast_to(&reshaped, target_shape)
    } else {
        Ok(tensor.clone())
    }
}
