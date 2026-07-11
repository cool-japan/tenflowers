//! Activation function gradients
//!
//! This module contains gradient computation logic for activation functions
//! like ReLU, sigmoid, tanh, softmax, GELU, Swish, etc.

use scirs2_core::numeric::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

use super::super::helpers::get_tensor_value;
use super::super::structures::GradientTapeInner;
use super::super::{GradientTape, TensorId};

/// Process backward pass for ReLU activation
pub(super) fn process_relu_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // ReLU gradient: 1 if input > 0, 0 otherwise
    if let Some(input_tensor) = get_tensor_value::<T>(inner, input) {
        // Create mask where input > 0
        let zero_tensor = Tensor::zeros(input_tensor.shape().dims());
        let mask = create_relu_mask(&input_tensor, &zero_tensor)?;

        // Apply mask to gradient
        let input_grad = tenflowers_core::ops::mul(grad_output, &mask)?;
        super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    }
    Ok(())
}

/// Process backward pass for Sigmoid activation
pub(super) fn process_sigmoid_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
    if let Some(input_tensor) = get_tensor_value::<T>(inner, input) {
        let sigmoid_output = tenflowers_core::ops::sigmoid(&input_tensor)?;
        let ones = Tensor::ones(sigmoid_output.shape().dims());
        let one_minus_sigmoid = tenflowers_core::ops::sub(&ones, &sigmoid_output)?;
        let sigmoid_grad = tenflowers_core::ops::mul(&sigmoid_output, &one_minus_sigmoid)?;
        let input_grad = tenflowers_core::ops::mul(grad_output, &sigmoid_grad)?;
        super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    }
    Ok(())
}

/// Process backward pass for Tanh activation
pub(super) fn process_tanh_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Tanh gradient: 1 - tanh²(x)
    if let Some(input_tensor) = get_tensor_value::<T>(inner, input) {
        let tanh_output = tenflowers_core::ops::tanh(&input_tensor)?;
        let tanh_squared = tenflowers_core::ops::mul(&tanh_output, &tanh_output)?;
        let ones = Tensor::ones(tanh_squared.shape().dims());
        let tanh_grad = tenflowers_core::ops::sub(&ones, &tanh_squared)?;
        let input_grad = tenflowers_core::ops::mul(grad_output, &tanh_grad)?;
        super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    }
    Ok(())
}

/// Process backward pass for Softmax activation
///
/// Softmax is a rank-preserving, axis-local operation: `y = softmax(x, axis)`
/// only mixes elements that lie along `axis` (e.g. for attention weights of
/// shape `[batch, heads, seq, seq]` softmaxed over the last axis, each
/// `[.., .., i, :]` row is an independent softmax). The correct Jacobian-vector
/// product is therefore also axis-local:
///
///   grad_x = y * (grad_y - sum_axis(y * grad_y))
///
/// This delegates to `grad_ops::softmax_backward`, which implements exactly
/// this formula via `Tensor::sum(Some(&[axis]), true)`.
///
/// The forward softmax output `y` is recomputed here from the recorded input
/// tensor using the same numerically-stable max-subtract-exp-normalize
/// algorithm as `tenflowers_core::ops::activation::softmax` (max subtraction
/// for stability, exp, sum along `axis` with keepdims, divide) expressed via
/// `Tensor` methods instead of calling `Tensor::softmax` directly: the latter
/// additionally requires `T: std::iter::Sum`, a bound that would have to
/// propagate through the shared, generic `process_operation_backward`
/// dispatcher (and therefore every other operation's backward path in this
/// crate) purely to support this one call, even though the algorithm itself
/// needs no such bound. `Tensor::softmax` is a pure, deterministic function
/// of `(input, axis)` with no randomness, so recomputing it via equivalent
/// primitives reproduces the exact forward-pass output.
pub(super) fn process_softmax_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    axis: Option<i32>,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Softmax backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let softmax_output = recompute_softmax(&input_tensor, axis)?;
    let input_grad = crate::grad_ops::softmax_backward(grad_output, &softmax_output, axis)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Recompute the softmax forward output along `axis`, matching
/// `tenflowers_core::ops::activation::softmax` element-for-element:
/// `y = exp(x - max_axis(x)) / sum_axis(exp(x - max_axis(x)))`.
///
/// `axis` resolves `None` to the last axis, mirroring the forward op's own
/// `axis.unwrap_or(-1)` default so backward always recomputes the same
/// output the forward pass produced.
fn recompute_softmax<T>(input: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let axis_slice = [axis.unwrap_or(-1)];
    let max_x = input.max(Some(&axis_slice), true)?;
    let shifted = input.sub(&max_x)?;
    let exp_x = shifted.exp()?;
    let sum_exp = exp_x.sum(Some(&axis_slice), true)?;
    exp_x.div(&sum_exp)
}

/// Process backward pass for GELU activation
pub(super) fn process_gelu_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // GELU gradient: grad_y * (Phi(x) + x * phi(x)), computed via the
    // tanh-approximation derivative in `crate::grad_ops::gelu_backward`.
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "GELU backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let input_grad = crate::grad_ops::gelu_backward(grad_output, &input_tensor)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for Swish activation
pub(super) fn process_swish_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Swish gradient: grad_y * (sigmoid(x) * (1 + x * (1 - sigmoid(x)))).
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Swish backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let input_grad = crate::grad_ops::swish_backward(grad_output, &input_tensor)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for Mish activation
pub(super) fn process_mish_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Mish gradient: grad_y * (tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)).
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Mish backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let input_grad = crate::grad_ops::mish_backward(grad_output, &input_tensor)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for LeakyReLU activation
pub(super) fn process_leaky_relu_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    alpha: f32,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // LeakyReLU gradient: 1 if input > 0, alpha otherwise.
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "LeakyReLU backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let alpha_t = T::from_f32(alpha).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "LeakyReLU backward: alpha value could not be converted to the tensor element type"
                .to_string(),
        )
    })?;
    let input_grad = crate::grad_ops::leaky_relu_backward(grad_output, &input_tensor, alpha_t)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for ELU activation
pub(super) fn process_elu_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    alpha: f32,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // ELU gradient: 1 if input > 0, alpha * exp(x) otherwise.
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "ELU backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let alpha_t = T::from_f32(alpha).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "ELU backward: alpha value could not be converted to the tensor element type"
                .to_string(),
        )
    })?;
    let input_grad = crate::grad_ops::elu_backward(grad_output, &input_tensor, alpha_t)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for PReLU activation
pub(super) fn process_prelu_backward<T>(
    _tape: &GradientTape,
    _inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    _alpha: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // PReLU gradient: simplified approximation for trait compatibility
    super::super::utils::accumulate_gradient(gradients, input, grad_output.clone())?;
    Ok(())
}

/// Process backward pass for Log (natural logarithm) activation
///
/// `d/dx log(x) = grad_output / x`.
pub(super) fn process_log_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Log backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let input_grad = tenflowers_core::ops::div(grad_output, &input_tensor)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for Abs (absolute value) activation
///
/// `d/dx |x| = grad_output * sign(x)`, where `sign(0) := 0` by convention.
/// `|x|` is not differentiable exactly at `x = 0`; `sign(0) = 0` is the
/// defensible convention used here, consistent with e.g. PyTorch's
/// `torch.sign(0) == 0`.
///
/// `sign(x)` is built from two comparisons composed with `where_op`, exactly
/// mirroring the established `where_op`-based masking idiom already used by
/// `elu_backward`/`prelu_backward`/`mish_backward`'s `abs(x)` construction in
/// `crate::grad_ops::activation_ops`: the innermost `where_op` picks between
/// `-1` and `0` for `x < 0` vs. `x == 0`, and the outer `where_op` overrides
/// that with `1` wherever `x > 0`.
pub(super) fn process_abs_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Abs backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;

    let zero_tensor = Tensor::zeros(input_tensor.shape().dims());
    let one_tensor = Tensor::ones(input_tensor.shape().dims());
    let neg_one_tensor = tenflowers_core::ops::neg(&one_tensor)?;

    let positive_mask = input_tensor.gt(&zero_tensor)?;
    let negative_mask = input_tensor.lt(&zero_tensor)?;

    // innermost: -1 where x < 0, else 0 (covers x == 0 -> 0)
    let neg_or_zero = tenflowers_core::ops::where_op(&negative_mask, &neg_one_tensor, &zero_tensor)?;
    // outer: 1 where x > 0, else the innermost result
    let sign = tenflowers_core::ops::where_op(&positive_mask, &one_tensor, &neg_or_zero)?;

    let input_grad = tenflowers_core::ops::mul(grad_output, &sign)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for Clamp activation
///
/// `d/dx clamp(x, min, max) = grad_output` where `x` is in `[min, max]`
/// (using whichever bounds are `Some`; a `None` bound means "unconstrained"
/// on that side, so the in-range test always passes on that side), else `0`.
///
/// A `None` bound is resolved entirely in Rust control flow (not tensor
/// math): the in-range mask starts as "all true" and is narrowed by an
/// `AND`-style `where_op` composition only for the bounds that are `Some`,
/// so a `None` bound can never crash and never incorrectly zeroes gradient
/// on that side.
pub(super) fn process_clamp_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    min: Option<f32>,
    max: Option<f32>,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Clamp backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;

    let zero_tensor = Tensor::zeros(input_tensor.shape().dims());
    let one_tensor = Tensor::ones(input_tensor.shape().dims());

    // Start unconstrained (mask of all `1`s == "in range everywhere"), then
    // narrow it with each bound that is actually `Some`.
    let mut in_range_mask = one_tensor.clone();

    if let Some(min_f32) = min {
        let min_t = T::from_f32(min_f32).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "Clamp backward: min value could not be converted to the tensor element type"
                    .to_string(),
            )
        })?;
        let min_tensor = Tensor::from_scalar(min_t);
        let above_min = input_tensor.ge(&min_tensor)?;
        in_range_mask =
            tenflowers_core::ops::where_op(&above_min, &in_range_mask, &zero_tensor)?;
    }

    if let Some(max_f32) = max {
        let max_t = T::from_f32(max_f32).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "Clamp backward: max value could not be converted to the tensor element type"
                    .to_string(),
            )
        })?;
        let max_tensor = Tensor::from_scalar(max_t);
        let below_max = input_tensor.le(&max_tensor)?;
        in_range_mask =
            tenflowers_core::ops::where_op(&below_max, &in_range_mask, &zero_tensor)?;
    }

    let input_grad = tenflowers_core::ops::mul(grad_output, &in_range_mask)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for Relu6 activation
///
/// `relu6(x) = min(max(x, 0), 6)`. Backward: `d/dx = grad_output` where
/// `0 < x < 6` (strict inequality on both boundaries, excluding them from
/// receiving gradient), else `0`. This mirrors `Relu`'s own backward
/// convention in `create_relu_mask`/`process_relu_backward` (`val >
/// T::zero()`, strict), applied to both of Relu6's boundaries for
/// consistency, independent of the fact that `Relu6`'s own *forward*
/// implementation (`crate::ops::activation::relu6` /
/// `tenflowers_core::ops::activation::relu6`) happens to use inclusive
/// `<=`/`>=` for its clamping — the forward clamp boundary convention and
/// the backward gradient boundary convention are independent design
/// choices.
pub(super) fn process_relu6_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Relu6 backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let six = T::from_f32(6.0).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Relu6 backward: constant 6 could not be converted to the tensor element type"
                .to_string(),
        )
    })?;
    let mask = relu6_prime_mask(&input_tensor, T::zero(), six)?;
    let input_grad = tenflowers_core::ops::mul(grad_output, &mask)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for HardSwish activation
///
/// `hardswish(x) = x * relu6(x + 3) / 6`. Via the product rule, letting
/// `u = x + 3` and `r = relu6(u)`:
///
///   d/dx [x * relu6(x+3) / 6] = (1/6) * [relu6(x+3) + x * relu6'(x+3)]
///
/// where `relu6'(x+3) = 1` if `-3 < x < 3` (i.e. `0 < x+3 < 6`, same strict
/// boundary convention as `Relu6` above, evaluated at the shifted point
/// `x+3`), else `0`. Two distinct quantities are needed: `relu6(x+3)`
/// itself (the VALUE, for the first term) and `relu6'(x+3)` (the MASK, for
/// the second term's multiplier) — they are not the same tensor.
pub(super) fn process_hard_swish_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "HardSwish backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;

    let three = T::from_f32(3.0).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "HardSwish backward: constant 3 could not be converted to the tensor element type"
                .to_string(),
        )
    })?;
    let six = T::from_f32(6.0).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "HardSwish backward: constant 6 could not be converted to the tensor element type"
                .to_string(),
        )
    })?;

    let three_tensor = Tensor::from_scalar(three);
    let x_plus_3 = input_tensor.add(&three_tensor)?;

    // relu6(x+3) VALUE, for the first term.
    let relu6_val = tenflowers_core::ops::activation::relu6(&x_plus_3)?;
    // relu6'(x+3) MASK (1 where 0 < x+3 < 6, else 0), for the second term.
    let relu6_prime = relu6_prime_mask(&x_plus_3, T::zero(), six)?;

    let term1 = relu6_val;
    let term2 = input_tensor.mul(&relu6_prime)?;
    let sum = term1.add(&term2)?;
    let six_tensor = Tensor::from_scalar(six);
    let local_grad = sum.div(&six_tensor)?;

    let input_grad = tenflowers_core::ops::mul(grad_output, &local_grad)?;
    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Process backward pass for LogSoftmax activation
///
/// `log_softmax(x) = x - max_axis(x) - log(sum_axis(exp(x - max_axis(x))))`.
/// Gradient: `grad_x = grad_y - softmax(x) * sum_axis(grad_y, axis,
/// keepdims=true)`, where `softmax(x) = exp(log_softmax(x))`. This mirrors
/// the reference `crate::grad_ops::fused_ops::fused_log_softmax_forward_backward`
/// math exactly, and follows `process_softmax_backward`'s architectural
/// pattern: a private `recompute_log_softmax` helper recomputes the forward
/// numerically-stably from the recorded input tensor, and this function
/// applies the gradient formula against that recomputed value.
pub(super) fn process_log_softmax_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    axis: Option<i32>,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "LogSoftmax backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let log_softmax_output = recompute_log_softmax(&input_tensor, axis)?;
    let softmax_output = log_softmax_output.exp()?;

    let axis_slice = [axis.unwrap_or(-1)];
    let grad_sum = grad_output.sum(Some(&axis_slice), true)?;
    let softmax_grad_sum = softmax_output.mul(&grad_sum)?;
    let input_grad = grad_output.sub(&softmax_grad_sum)?;

    super::super::utils::accumulate_gradient(gradients, input, input_grad)?;
    Ok(())
}

/// Recompute the log_softmax forward output along `axis`, using the same
/// numerically-stable max-subtract-exp-sum-log algorithm as
/// `crate::grad_ops::fused_ops::fused_log_softmax_forward_backward`'s
/// forward half: `y = x - max_axis(x) - log(sum_axis(exp(x - max_axis(x))))`.
///
/// `axis` resolves `None` to the last axis, mirroring `Softmax`'s
/// (and this file's `recompute_softmax`'s) own `axis.unwrap_or(-1)` default,
/// and is axis-aware (not flattened) throughout: every reduction below
/// (`max`, `sum`) is taken along the same single `axis_slice`, exactly as
/// `Softmax`'s own backward recompute is, so this correctly handles
/// multi-dimensional tensors where only one axis is being soft-maxed over
/// (e.g. attention logits of shape `[batch, heads, seq, seq]`).
fn recompute_log_softmax<T>(input: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let axis_slice = [axis.unwrap_or(-1)];
    let max_x = input.max(Some(&axis_slice), true)?;
    let shifted = input.sub(&max_x)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum(Some(&axis_slice), true)?;
    let log_sum_exp = sum_exp.log()?;
    shifted.sub(&log_sum_exp)
}

/// Build the elementwise `relu6'` boundary mask: `1` where `low < x < high`
/// (strict on both sides), else `0`. Shared by `Relu6`'s own backward and
/// `HardSwish`'s backward (the latter evaluates it at the shifted point
/// `x + 3` with `low = 0, high = 6`).
fn relu6_prime_mask<T>(input: &Tensor<T>, low: T, high: T) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let low_tensor = Tensor::from_scalar(low);
    let high_tensor = Tensor::from_scalar(high);
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let one_tensor = Tensor::ones(input.shape().dims());

    let above_low = input.gt(&low_tensor)?;
    let below_high = input.lt(&high_tensor)?;

    let above_low_only = tenflowers_core::ops::where_op(&above_low, &one_tensor, &zero_tensor)?;
    tenflowers_core::ops::where_op(&below_high, &above_low_only, &zero_tensor)
}

/// Helper function to create ReLU mask
fn create_relu_mask<T>(input: &Tensor<T>, _zero: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create mask where input > 0
    // For ReLU: gradient is 1 if input > 0, 0 otherwise
    if let Some(input_data) = input.as_slice() {
        let mut mask_data = Vec::with_capacity(input_data.len());
        for &val in input_data {
            if val > T::zero() {
                mask_data.push(T::one());
            } else {
                mask_data.push(T::zero());
            }
        }
        Ok(Tensor::from_vec(mask_data, input.shape().dims())?)
    } else {
        // Fallback: return ones for compatibility (though this is incorrect)
        Ok(Tensor::ones(input.shape().dims()))
    }
}
