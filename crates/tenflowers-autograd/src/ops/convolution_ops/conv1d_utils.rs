//! Gradient math helpers for 1D convolution backward.
//!
//! This is the 1D specialization of the Conv2D gradient math in [`super::utils`]
//! (which is itself mirrored by the Conv3D helpers in the same file) — same
//! scatter/gather convention, one fewer spatial dimension. Kept in its own file
//! so `utils.rs` (already the largest file in this module) does not grow further.

use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Compute the padding offset `pad_left` matching the forward Conv1D
/// cross-correlation convention used in `tenflowers_core::ops::conv1d`
/// (`conv1d_cpu`).
///
/// Forward: `out[b,oc,ol] = sum input[b,ic, ol*stride + k - pad_left] * w[oc,ic,k]`
///
/// Mirrors `conv2d_pad_offsets` in `super::utils`, minus the width dimension.
/// Note: the forward kernel computes `pad_total` with `std::cmp::max(0, ...)`
/// on `usize` arithmetic (a no-op that can underflow-panic for pathological
/// stride/kernel/length combinations); here we use `saturating_sub`, which is
/// safe and produces an identical result in the well-formed (non-panicking)
/// case that the forward pass actually exercises — the same relationship
/// `conv2d_pad_offsets`/`conv3d_pad_offsets` already have with their forward
/// counterparts.
fn conv1d_pad_offset(
    in_length: usize,
    kernel_length: usize,
    stride: usize,
    padding: &str,
) -> Result<usize> {
    match padding {
        "valid" => Ok(0),
        "same" => {
            let out_len = (in_length + stride - 1) / stride;
            let pad_total =
                ((out_len.saturating_sub(1)) * stride + kernel_length).saturating_sub(in_length);
            Ok(pad_total / 2)
        }
        _ => Err(TensorError::invalid_operation_simple(format!(
            "Unknown padding mode for Conv1D backward: {padding}"
        ))),
    }
}

/// Compute Conv1D gradient w.r.t. the input.
///
/// Exact transpose of the forward cross-correlation. For every output element
/// that read `input[b, ic, il]` we route the corresponding
/// `grad_output * weight` contribution back onto that input position:
///
/// `grad_input[b,ic,il] = sum_{oc,k : ol*stride+k-pad_left == il}
///                            grad_output[b,oc,ol] * weight[oc,ic,k]`
pub(crate) fn compute_conv1d_input_gradient<T>(
    grad_output: &Tensor<T>,
    weight: &Tensor<T>,
    input_shape: &[usize],
    stride: usize,
    padding: &str,
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let weight_shape = weight.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    if input_shape.len() != 3 || weight_shape.len() != 3 || grad_output_shape.len() != 3 {
        return Err(TensorError::invalid_shape_simple(
            "Conv1D input-gradient requires 3D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_length = weight_shape[2];

    if weight_in_channels != in_channels {
        return Err(TensorError::invalid_shape_simple(format!(
            "Conv1D input-gradient channel mismatch: input has {in_channels} channels but weight expects {weight_in_channels}"
        )));
    }

    let out_length = grad_output_shape[2];

    let pad_left = conv1d_pad_offset(in_length, kernel_length, stride, padding)?;

    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read grad_output for Conv1D input gradient: {e}"
        ))
    })?;
    let weight_data = weight.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read weight for Conv1D input gradient: {e}"
        ))
    })?;

    let mut grad_input_data = vec![T::zero(); batch_size * in_channels * in_length];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                let grad_idx = (b * out_channels + oc) * out_length + ol;
                let grad_val = grad_output_data[grad_idx];
                if grad_val == T::zero() {
                    continue;
                }

                for ic in 0..in_channels {
                    for k in 0..kernel_length {
                        // Position in the (virtually) padded input.
                        let padded_il = ol * stride + k;
                        if padded_il < pad_left {
                            continue;
                        }
                        let il = padded_il - pad_left;
                        if il >= in_length {
                            continue;
                        }

                        let weight_idx = (oc * in_channels + ic) * kernel_length + k;
                        let contribution = grad_val * weight_data[weight_idx];

                        let input_idx = (b * in_channels + ic) * in_length + il;
                        grad_input_data[input_idx] = grad_input_data[input_idx] + contribution;
                    }
                }
            }
        }
    }

    Tensor::from_vec(grad_input_data, input_shape)
}

/// Compute Conv1D gradient w.r.t. the weight.
///
/// `grad_weight[oc,ic,k] = sum_{b,ol}
///        input[b,ic, ol*stride+k-pad_left] * grad_output[b,oc,ol]`
pub(crate) fn compute_conv1d_weight_gradient<T>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    weight_shape: &[usize],
    stride: usize,
    padding: &str,
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_shape = input.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    if input_shape.len() != 3 || weight_shape.len() != 3 || grad_output_shape.len() != 3 {
        return Err(TensorError::invalid_shape_simple(
            "Conv1D weight-gradient requires 3D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_length = weight_shape[2];

    if weight_in_channels != in_channels {
        return Err(TensorError::invalid_shape_simple(format!(
            "Conv1D weight-gradient channel mismatch: input has {in_channels} channels but weight expects {weight_in_channels}"
        )));
    }

    let out_length = grad_output_shape[2];

    let pad_left = conv1d_pad_offset(in_length, kernel_length, stride, padding)?;

    let input_data = input.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read input for Conv1D weight gradient: {e}"
        ))
    })?;
    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read grad_output for Conv1D weight gradient: {e}"
        ))
    })?;

    let mut grad_weight_data = vec![T::zero(); out_channels * in_channels * kernel_length];

    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for k in 0..kernel_length {
                let mut acc = T::zero();

                for b in 0..batch_size {
                    for ol in 0..out_length {
                        let padded_il = ol * stride + k;
                        if padded_il < pad_left {
                            continue;
                        }
                        let il = padded_il - pad_left;
                        if il >= in_length {
                            continue;
                        }

                        let input_idx = (b * in_channels + ic) * in_length + il;
                        let grad_idx = (b * out_channels + oc) * out_length + ol;
                        acc = acc + input_data[input_idx] * grad_output_data[grad_idx];
                    }
                }

                let weight_idx = (oc * in_channels + ic) * kernel_length + k;
                grad_weight_data[weight_idx] = acc;
            }
        }
    }

    Tensor::from_vec(grad_weight_data, weight_shape)
}
