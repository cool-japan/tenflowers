use super::types::get_tensor_element_4d;
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Compute padding offsets `(pad_top, pad_left)` matching the forward Conv2D
/// cross-correlation convention used in `tenflowers_core::ops::conv2d`.
///
/// Forward: `out[b,oc,oh,ow] = sum input[b,ic, oh*s_h + kh - pad_top,
///                                          ow*s_w + kw - pad_left] * w[oc,ic,kh,kw]`
fn conv2d_pad_offsets(
    in_height: usize,
    in_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: (usize, usize),
    padding: &str,
) -> Result<(usize, usize)> {
    match padding {
        "valid" => Ok((0, 0)),
        "same" => {
            let out_h = (in_height + stride.0 - 1) / stride.0;
            let out_w = (in_width + stride.1 - 1) / stride.1;
            let pad_h = ((out_h - 1) * stride.0 + kernel_height).saturating_sub(in_height);
            let pad_w = ((out_w - 1) * stride.1 + kernel_width).saturating_sub(in_width);
            Ok((pad_h / 2, pad_w / 2))
        }
        _ => Err(TensorError::invalid_operation_simple(format!(
            "Unknown padding mode for Conv2D backward: {padding}"
        ))),
    }
}

/// Compute Conv2D gradient w.r.t. the input.
///
/// This is the exact transpose of the forward cross-correlation. For every
/// output element that read `input[b, ic, ih, iw]` we route the corresponding
/// `grad_output * weight` contribution back onto that input position:
///
/// `grad_input[b,ic,ih,iw] = sum_{oc,kh,kw : oh*s_h+kh-pad_top == ih,
///                                            ow*s_w+kw-pad_left == iw}
///                               grad_output[b,oc,oh,ow] * weight[oc,ic,kh,kw]`
pub(crate) fn compute_conv2d_input_gradient<T>(
    grad_output: &Tensor<T>,
    weight: &Tensor<T>,
    input_shape: &[usize],
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
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let weight_shape = weight.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    if input_shape.len() != 4 || weight_shape.len() != 4 || grad_output_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Conv2D input-gradient requires 4D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    if weight_in_channels != in_channels {
        return Err(TensorError::invalid_shape_simple(format!(
            "Conv2D input-gradient channel mismatch: input has {in_channels} channels but weight expects {weight_in_channels}"
        )));
    }

    let out_height = grad_output_shape[2];
    let out_width = grad_output_shape[3];

    let (pad_top, pad_left) = conv2d_pad_offsets(
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride,
        padding,
    )?;

    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read grad_output for Conv2D input gradient: {e}"
        ))
    })?;
    let weight_data = weight.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read weight for Conv2D input gradient: {e}"
        ))
    })?;

    let mut grad_input_data = vec![T::zero(); batch_size * in_channels * in_height * in_width];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let grad_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    let grad_val = grad_output_data[grad_idx];
                    if grad_val == T::zero() {
                        continue;
                    }

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            // Position in the (virtually) padded input.
                            let padded_ih = oh * stride.0 + kh;
                            if padded_ih < pad_top {
                                continue;
                            }
                            let ih = padded_ih - pad_top;
                            if ih >= in_height {
                                continue;
                            }

                            for kw in 0..kernel_width {
                                let padded_iw = ow * stride.1 + kw;
                                if padded_iw < pad_left {
                                    continue;
                                }
                                let iw = padded_iw - pad_left;
                                if iw >= in_width {
                                    continue;
                                }

                                let weight_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                    * kernel_width
                                    + kw;
                                let contribution = grad_val * weight_data[weight_idx];

                                let input_idx =
                                    ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                grad_input_data[input_idx] =
                                    grad_input_data[input_idx] + contribution;
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(grad_input_data, input_shape)
}

/// Compute Conv2D gradient w.r.t. the weight.
///
/// `grad_weight[oc,ic,kh,kw] = sum_{b,oh,ow}
///        input[b,ic, oh*s_h+kh-pad_top, ow*s_w+kw-pad_left]
///        * grad_output[b,oc,oh,ow]`
pub(crate) fn compute_conv2d_weight_gradient<T>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    weight_shape: &[usize],
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
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_shape = input.shape().dims();
    let grad_output_shape = grad_output.shape().dims();

    if input_shape.len() != 4 || weight_shape.len() != 4 || grad_output_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Conv2D weight-gradient requires 4D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    if weight_in_channels != in_channels {
        return Err(TensorError::invalid_shape_simple(format!(
            "Conv2D weight-gradient channel mismatch: input has {in_channels} channels but weight expects {weight_in_channels}"
        )));
    }

    let out_height = grad_output_shape[2];
    let out_width = grad_output_shape[3];

    let (pad_top, pad_left) = conv2d_pad_offsets(
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride,
        padding,
    )?;

    let input_data = input.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read input for Conv2D weight gradient: {e}"
        ))
    })?;
    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read grad_output for Conv2D weight gradient: {e}"
        ))
    })?;

    let mut grad_weight_data =
        vec![T::zero(); out_channels * in_channels * kernel_height * kernel_width];

    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for kh in 0..kernel_height {
                for kw in 0..kernel_width {
                    let mut acc = T::zero();

                    for b in 0..batch_size {
                        for oh in 0..out_height {
                            let padded_ih = oh * stride.0 + kh;
                            if padded_ih < pad_top {
                                continue;
                            }
                            let ih = padded_ih - pad_top;
                            if ih >= in_height {
                                continue;
                            }

                            for ow in 0..out_width {
                                let padded_iw = ow * stride.1 + kw;
                                if padded_iw < pad_left {
                                    continue;
                                }
                                let iw = padded_iw - pad_left;
                                if iw >= in_width {
                                    continue;
                                }

                                let input_idx =
                                    ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                let grad_idx =
                                    ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                                acc = acc + input_data[input_idx] * grad_output_data[grad_idx];
                            }
                        }
                    }

                    let weight_idx =
                        ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                    grad_weight_data[weight_idx] = acc;
                }
            }
        }
    }

    Tensor::from_vec(grad_weight_data, weight_shape)
}

/// Compute padding offsets `(pad_front, pad_top, pad_left)` matching the forward
/// Conv3D cross-correlation convention (NCDHW). This is the 3D analogue of
/// [`conv2d_pad_offsets`].
///
/// Forward: `out[b,oc,od,oh,ow] = sum input[b,ic, od*s_d+kd-pad_front,
///                                          oh*s_h+kh-pad_top,
///                                          ow*s_w+kw-pad_left] * w[oc,ic,kd,kh,kw]`
fn conv3d_pad_offsets(
    in_depth: usize,
    in_height: usize,
    in_width: usize,
    kernel_depth: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<(usize, usize, usize)> {
    match padding {
        "valid" => Ok((0, 0, 0)),
        "same" => {
            let out_d = (in_depth + stride.0 - 1) / stride.0;
            let out_h = (in_height + stride.1 - 1) / stride.1;
            let out_w = (in_width + stride.2 - 1) / stride.2;
            let pad_d = ((out_d - 1) * stride.0 + kernel_depth).saturating_sub(in_depth);
            let pad_h = ((out_h - 1) * stride.1 + kernel_height).saturating_sub(in_height);
            let pad_w = ((out_w - 1) * stride.2 + kernel_width).saturating_sub(in_width);
            Ok((pad_d / 2, pad_h / 2, pad_w / 2))
        }
        _ => Err(TensorError::invalid_operation_simple(format!(
            "Unknown padding mode for Conv3D backward: {padding}"
        ))),
    }
}

/// Compute Conv3D gradient w.r.t. the input (NCDHW).
///
/// Exact transpose of the forward 3D cross-correlation. For every output element
/// that read `input[b, ic, id, ih, iw]` the corresponding `grad_output * weight`
/// contribution is scattered back onto that input position:
///
/// `grad_input[b,ic,id,ih,iw] = sum_{oc,kd,kh,kw} grad_output[b,oc,od,oh,ow]
///                                                 * weight[oc,ic,kd,kh,kw]`
/// where `id = od*s_d+kd-pad_front`, `ih = oh*s_h+kh-pad_top`,
/// `iw = ow*s_w+kw-pad_left` (bounds-checked against the input volume).
pub(crate) fn compute_conv3d_input_gradient<T>(
    grad_output: &Tensor<T>,
    weight: &Tensor<T>,
    input_shape: &[usize],
    stride: (usize, usize, usize),
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

    if input_shape.len() != 5 || weight_shape.len() != 5 || grad_output_shape.len() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Conv3D input-gradient requires 5D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_depth = input_shape[2];
    let in_height = input_shape[3];
    let in_width = input_shape[4];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    if weight_in_channels != in_channels {
        return Err(TensorError::invalid_shape_simple(format!(
            "Conv3D input-gradient channel mismatch: input has {in_channels} channels but weight expects {weight_in_channels}"
        )));
    }

    let out_depth = grad_output_shape[2];
    let out_height = grad_output_shape[3];
    let out_width = grad_output_shape[4];

    let (pad_front, pad_top, pad_left) = conv3d_pad_offsets(
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride,
        padding,
    )?;

    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read grad_output for Conv3D input gradient: {e}"
        ))
    })?;
    let weight_data = weight.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read weight for Conv3D input gradient: {e}"
        ))
    })?;

    let mut grad_input_data =
        vec![T::zero(); batch_size * in_channels * in_depth * in_height * in_width];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let grad_idx = (((b * out_channels + oc) * out_depth + od) * out_height
                            + oh)
                            * out_width
                            + ow;
                        let grad_val = grad_output_data[grad_idx];
                        if grad_val == T::zero() {
                            continue;
                        }

                        for ic in 0..in_channels {
                            for kd in 0..kernel_depth {
                                let padded_id = od * stride.0 + kd;
                                if padded_id < pad_front {
                                    continue;
                                }
                                let id = padded_id - pad_front;
                                if id >= in_depth {
                                    continue;
                                }

                                for kh in 0..kernel_height {
                                    let padded_ih = oh * stride.1 + kh;
                                    if padded_ih < pad_top {
                                        continue;
                                    }
                                    let ih = padded_ih - pad_top;
                                    if ih >= in_height {
                                        continue;
                                    }

                                    for kw in 0..kernel_width {
                                        let padded_iw = ow * stride.2 + kw;
                                        if padded_iw < pad_left {
                                            continue;
                                        }
                                        let iw = padded_iw - pad_left;
                                        if iw >= in_width {
                                            continue;
                                        }

                                        let weight_idx =
                                            ((((oc * in_channels + ic) * kernel_depth + kd)
                                                * kernel_height
                                                + kh)
                                                * kernel_width)
                                                + kw;
                                        let contribution = grad_val * weight_data[weight_idx];

                                        let input_idx = (((b * in_channels + ic) * in_depth + id)
                                            * in_height
                                            + ih)
                                            * in_width
                                            + iw;
                                        grad_input_data[input_idx] =
                                            grad_input_data[input_idx] + contribution;
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

/// Compute Conv3D gradient w.r.t. the weight (NCDHW).
///
/// `grad_weight[oc,ic,kd,kh,kw] = sum_{b,od,oh,ow}
///        input[b,ic, od*s_d+kd-pad_front, oh*s_h+kh-pad_top, ow*s_w+kw-pad_left]
///        * grad_output[b,oc,od,oh,ow]`
pub(crate) fn compute_conv3d_weight_gradient<T>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    weight_shape: &[usize],
    stride: (usize, usize, usize),
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

    if input_shape.len() != 5 || weight_shape.len() != 5 || grad_output_shape.len() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Conv3D weight-gradient requires 5D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_depth = input_shape[2];
    let in_height = input_shape[3];
    let in_width = input_shape[4];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    if weight_in_channels != in_channels {
        return Err(TensorError::invalid_shape_simple(format!(
            "Conv3D weight-gradient channel mismatch: input has {in_channels} channels but weight expects {weight_in_channels}"
        )));
    }

    let out_depth = grad_output_shape[2];
    let out_height = grad_output_shape[3];
    let out_width = grad_output_shape[4];

    let (pad_front, pad_top, pad_left) = conv3d_pad_offsets(
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride,
        padding,
    )?;

    let input_data = input.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read input for Conv3D weight gradient: {e}"
        ))
    })?;
    let grad_output_data = grad_output.to_vec().map_err(|e| {
        TensorError::invalid_operation_simple(format!(
            "Failed to read grad_output for Conv3D weight gradient: {e}"
        ))
    })?;

    let mut grad_weight_data =
        vec![T::zero(); out_channels * in_channels * kernel_depth * kernel_height * kernel_width];

    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for kd in 0..kernel_depth {
                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        let mut acc = T::zero();

                        for b in 0..batch_size {
                            for od in 0..out_depth {
                                let padded_id = od * stride.0 + kd;
                                if padded_id < pad_front {
                                    continue;
                                }
                                let id = padded_id - pad_front;
                                if id >= in_depth {
                                    continue;
                                }

                                for oh in 0..out_height {
                                    let padded_ih = oh * stride.1 + kh;
                                    if padded_ih < pad_top {
                                        continue;
                                    }
                                    let ih = padded_ih - pad_top;
                                    if ih >= in_height {
                                        continue;
                                    }

                                    for ow in 0..out_width {
                                        let padded_iw = ow * stride.2 + kw;
                                        if padded_iw < pad_left {
                                            continue;
                                        }
                                        let iw = padded_iw - pad_left;
                                        if iw >= in_width {
                                            continue;
                                        }

                                        let input_idx = (((b * in_channels + ic) * in_depth + id)
                                            * in_height
                                            + ih)
                                            * in_width
                                            + iw;
                                        let grad_idx = (((b * out_channels + oc) * out_depth + od)
                                            * out_height
                                            + oh)
                                            * out_width
                                            + ow;
                                        acc = acc
                                            + input_data[input_idx] * grad_output_data[grad_idx];
                                    }
                                }
                            }
                        }

                        let weight_idx =
                            ((((oc * in_channels + ic) * kernel_depth + kd) * kernel_height + kh)
                                * kernel_width)
                                + kw;
                        grad_weight_data[weight_idx] = acc;
                    }
                }
            }
        }
    }

    Tensor::from_vec(grad_weight_data, weight_shape)
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

/// Slice a tensor along its channel dimension, returning the contiguous range
/// `[start, end)` of channels.
///
/// The channel axis is dimension 1 for the usual NCHW / NCDHW layouts
/// (`[B, C, ...]`); for a rank-1 tensor (e.g. a per-channel bias vector
/// `[C]`) the channel axis is dimension 0. The data for the selected channels
/// is copied into a freshly allocated output tensor of the appropriate shape.
pub(crate) fn slice_tensor_channels<T>(
    tensor: &Tensor<T>,
    start: usize,
    end: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let shape = tensor.shape().dims();
    if shape.is_empty() {
        return Err(TensorError::invalid_shape_simple(
            "slice_tensor_channels requires a tensor with at least one dimension".to_string(),
        ));
    }

    // Channel axis: dim 1 for rank >= 2, dim 0 for a 1D (bias) tensor.
    let channel_axis = if shape.len() >= 2 { 1 } else { 0 };
    let channels = shape[channel_axis];

    if start > end || end > channels {
        return Err(TensorError::invalid_operation_simple(format!(
            "slice_tensor_channels: invalid channel range [{start}, {end}) for {channels} channels"
        )));
    }

    // Build the per-dimension ranges: full extent on every axis except the
    // channel axis, which is restricted to [start, end).
    let ranges: Vec<std::ops::Range<usize>> = shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if axis == channel_axis {
                start..end
            } else {
                0..dim
            }
        })
        .collect();

    tensor.slice(&ranges)
}

/// Concatenate a list of tensors along the given `axis`.
///
/// All tensors must share the same rank and the same extent on every axis
/// except `axis`; the output extent along `axis` is the sum of the inputs'
/// extents there. Data is copied row-by-row (over the contiguous trailing
/// block) into the correct offset of the freshly allocated output buffer.
pub(crate) fn concatenate_tensors<T>(tensors: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if tensors.is_empty() {
        return Err(TensorError::invalid_operation_simple(
            "Cannot concatenate empty tensor list".to_string(),
        ));
    }

    let reference_shape = tensors[0].shape().dims().to_vec();
    let rank = reference_shape.len();
    if axis >= rank {
        return Err(TensorError::invalid_operation_simple(format!(
            "concatenate_tensors: axis {axis} out of bounds for rank {rank}"
        )));
    }

    // Validate that every tensor matches on all non-concatenation axes and
    // accumulate the total size along the concatenation axis.
    let mut concat_dim_total = 0usize;
    for (tensor_index, tensor) in tensors.iter().enumerate() {
        let shape = tensor.shape().dims();
        if shape.len() != rank {
            return Err(TensorError::invalid_shape_simple(format!(
                "concatenate_tensors: tensor {tensor_index} has rank {} but expected {rank}",
                shape.len()
            )));
        }
        for (current_axis, (&dim, &reference_dim)) in
            shape.iter().zip(reference_shape.iter()).enumerate()
        {
            if current_axis != axis && dim != reference_dim {
                return Err(TensorError::invalid_shape_simple(format!(
                    "concatenate_tensors: tensor {tensor_index} dimension {current_axis} is {dim} but expected {reference_dim}"
                )));
            }
        }
        concat_dim_total += shape[axis];
    }

    // Output shape: identical to the reference except along `axis`.
    let mut output_shape = reference_shape.clone();
    output_shape[axis] = concat_dim_total;

    // `outer` = product of dims before `axis`; `inner` = product of dims after
    // `axis`. Each tensor contributes a block of `axis_len * inner` per outer
    // index, written into the output at the running channel offset.
    let outer: usize = reference_shape[..axis].iter().product();
    let inner: usize = reference_shape[axis + 1..].iter().product();
    let output_axis_len = concat_dim_total;

    let mut output_data = vec![T::zero(); output_shape.iter().product()];

    let mut axis_offset = 0usize;
    for tensor in tensors {
        let shape = tensor.shape().dims();
        let axis_len = shape[axis];
        let tensor_data = tensor.to_vec().map_err(|e| {
            TensorError::invalid_operation_simple(format!(
                "Failed to read tensor data for concatenation: {e}"
            ))
        })?;

        for outer_index in 0..outer {
            let src_base = outer_index * axis_len * inner;
            let dst_base = (outer_index * output_axis_len + axis_offset) * inner;
            let block_len = axis_len * inner;
            output_data[dst_base..dst_base + block_len]
                .clone_from_slice(&tensor_data[src_base..src_base + block_len]);
        }

        axis_offset += axis_len;
    }

    Tensor::from_vec(output_data, &output_shape)
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

        let (grad_input, grad_weight, grad_bias) =
            result.expect("test: gradient computation should succeed");
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

        let grad_input = result.expect("test: gradient computation should succeed");
        assert_eq!(grad_input.shape().dims(), &[1, 3, 32, 32]);
    }

    #[test]
    fn test_avg_pool2d_backward_shapes() {
        let input = Tensor::<f32>::zeros(&[1, 3, 32, 32]);
        let grad_output = Tensor::<f32>::zeros(&[1, 3, 16, 16]);

        let result = avg_pool2d_backward(&grad_output, &input, (2, 2), (2, 2), "valid");
        assert!(result.is_ok());

        let grad_input = result.expect("test: gradient computation should succeed");
        assert_eq!(grad_input.shape().dims(), &[1, 3, 32, 32]);
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::{
        compute_conv2d_input_gradient, compute_conv2d_weight_gradient, concatenate_tensors,
        conv2d_pad_offsets, slice_tensor_channels,
    };
    use tenflowers_core::Tensor;

    /// Deterministic pseudo-random value in roughly `[-1, 1)`, used so the
    /// finite-difference comparisons exercise non-trivial (non-zero) gradients
    /// without pulling in an RNG dependency.
    fn pseudo_random(seed: usize) -> f64 {
        let x = (seed as f64 * 12.9898 + 78.233).sin() * 43758.5453;
        (x - x.floor()) * 2.0 - 1.0
    }

    fn filled_vec(len: usize, offset: usize) -> Vec<f64> {
        (0..len).map(|i| pseudo_random(i + offset)).collect()
    }

    /// Reference forward 2D convolution mirroring `tenflowers_core::ops::conv2d`
    /// (`conv2d_cpu`): NCHW input, weight `[out_c, in_c, kh, kw]`, cross-correlation.
    /// Returns the flat NCHW output buffer together with `(out_h, out_w)`.
    #[allow(clippy::too_many_arguments)]
    fn forward_conv2d_reference(
        input: &[f64],
        input_shape: [usize; 4],
        weight: &[f64],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: &str,
    ) -> (Vec<f64>, usize, usize) {
        let [batch_size, in_channels, in_height, in_width] = input_shape;
        let [out_channels, _w_in, kernel_height, kernel_width] = weight_shape;

        let (pad_top, pad_left) = conv2d_pad_offsets(
            in_height,
            in_width,
            kernel_height,
            kernel_width,
            stride,
            padding,
        )
        .expect("test: padding offsets should be computable");

        let (out_height, out_width) = match padding {
            "valid" => (
                (in_height - kernel_height) / stride.0 + 1,
                (in_width - kernel_width) / stride.1 + 1,
            ),
            "same" => (
                (in_height + stride.0 - 1) / stride.0,
                (in_width + stride.1 - 1) / stride.1,
            ),
            other => panic!("test: unsupported padding {other}"),
        };

        let mut output = vec![0.0_f64; batch_size * out_channels * out_height * out_width];

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0_f64;
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                let padded_ih = oh * stride.0 + kh;
                                if padded_ih < pad_top {
                                    continue;
                                }
                                let ih = padded_ih - pad_top;
                                if ih >= in_height {
                                    continue;
                                }
                                for kw in 0..kernel_width {
                                    let padded_iw = ow * stride.1 + kw;
                                    if padded_iw < pad_left {
                                        continue;
                                    }
                                    let iw = padded_iw - pad_left;
                                    if iw >= in_width {
                                        continue;
                                    }
                                    let input_idx =
                                        ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    let weight_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                        * kernel_width
                                        + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                        let out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        (output, out_height, out_width)
    }

    /// Scalar loss `L = <grad_seed, forward(input, weight)>` — the upstream
    /// gradient flowing into the conv is exactly `grad_seed`, so the analytical
    /// gradients returned by the kernels must match `dL/dinput` / `dL/dweight`.
    fn conv2d_loss(
        input: &[f64],
        input_shape: [usize; 4],
        weight: &[f64],
        weight_shape: [usize; 4],
        grad_seed: &[f64],
        stride: (usize, usize),
        padding: &str,
    ) -> f64 {
        let (output, _, _) =
            forward_conv2d_reference(input, input_shape, weight, weight_shape, stride, padding);
        output
            .iter()
            .zip(grad_seed.iter())
            .map(|(o, g)| o * g)
            .sum()
    }

    #[test]
    fn test_conv2d_input_gradient_finite_difference() {
        let input_shape = [1usize, 2, 5, 5];
        let weight_shape = [3usize, 2, 3, 3];
        let stride = (1usize, 1usize);
        let padding = "valid";

        let input = filled_vec(input_shape.iter().product(), 1);
        let weight = filled_vec(weight_shape.iter().product(), 1000);

        // Output dimensions for valid padding, stride 1: (5-3)+1 = 3 -> [1,3,3,3].
        let grad_seed = filled_vec(3 * 3 * 3, 7000);

        // Analytical gradient from the kernel under test.
        let grad_output_tensor = Tensor::<f64>::from_vec(grad_seed.clone(), &[1, 3, 3, 3])
            .expect("test: grad_output tensor");
        let weight_tensor =
            Tensor::<f64>::from_vec(weight.clone(), &weight_shape).expect("test: weight tensor");
        let analytical = compute_conv2d_input_gradient(
            &grad_output_tensor,
            &weight_tensor,
            &input_shape,
            stride,
            padding,
        )
        .expect("test: input gradient should compute")
        .to_vec()
        .expect("test: analytical grad to_vec");

        // Numerical gradient via central finite differences.
        let eps = 1e-4_f64;
        for idx in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let l_plus = conv2d_loss(
                &plus,
                input_shape,
                &weight,
                weight_shape,
                &grad_seed,
                stride,
                padding,
            );
            let l_minus = conv2d_loss(
                &minus,
                input_shape,
                &weight,
                weight_shape,
                &grad_seed,
                stride,
                padding,
            );
            let numerical = (l_plus - l_minus) / (2.0 * eps);

            let denom = numerical.abs().max(1e-6);
            let rel_err = (analytical[idx] - numerical).abs() / denom;
            assert!(
                rel_err < 1e-3,
                "input grad mismatch at {idx}: analytical={}, numerical={}, rel_err={}",
                analytical[idx],
                numerical,
                rel_err
            );
        }
    }

    #[test]
    fn test_conv2d_weight_gradient_finite_difference() {
        let input_shape = [1usize, 2, 5, 5];
        let weight_shape = [3usize, 2, 3, 3];
        let stride = (1usize, 1usize);
        let padding = "valid";

        let input = filled_vec(input_shape.iter().product(), 1);
        let weight = filled_vec(weight_shape.iter().product(), 1000);
        let grad_seed = filled_vec(3 * 3 * 3, 7000);

        let input_tensor =
            Tensor::<f64>::from_vec(input.clone(), &input_shape).expect("test: input tensor");
        let grad_output_tensor = Tensor::<f64>::from_vec(grad_seed.clone(), &[1, 3, 3, 3])
            .expect("test: grad_output tensor");
        let analytical = compute_conv2d_weight_gradient(
            &input_tensor,
            &grad_output_tensor,
            &weight_shape,
            stride,
            padding,
        )
        .expect("test: weight gradient should compute")
        .to_vec()
        .expect("test: analytical grad to_vec");

        let eps = 1e-4_f64;
        for idx in 0..weight.len() {
            let mut plus = weight.clone();
            let mut minus = weight.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let l_plus = conv2d_loss(
                &input,
                input_shape,
                &plus,
                weight_shape,
                &grad_seed,
                stride,
                padding,
            );
            let l_minus = conv2d_loss(
                &input,
                input_shape,
                &minus,
                weight_shape,
                &grad_seed,
                stride,
                padding,
            );
            let numerical = (l_plus - l_minus) / (2.0 * eps);

            let denom = numerical.abs().max(1e-6);
            let rel_err = (analytical[idx] - numerical).abs() / denom;
            assert!(
                rel_err < 1e-3,
                "weight grad mismatch at {idx}: analytical={}, numerical={}, rel_err={}",
                analytical[idx],
                numerical,
                rel_err
            );
        }
    }

    #[test]
    fn test_slice_tensor_channels_nchw() {
        // [1, 4, 2, 2]: channel c filled with the constant value `c`.
        let mut data = vec![0.0_f32; 4 * 2 * 2];
        for c in 0..4 {
            for hw in 0..4 {
                data[c * 4 + hw] = c as f32;
            }
        }
        let tensor = Tensor::<f32>::from_vec(data, &[1, 4, 2, 2]).expect("test: tensor");

        let sliced = slice_tensor_channels(&tensor, 1, 3).expect("test: slice channels");
        assert_eq!(sliced.shape().dims(), &[1, 2, 2, 2]);

        let sliced_data = sliced.to_vec().expect("test: sliced to_vec");
        // Channels 1 and 2 -> first 4 entries are 1.0, next 4 are 2.0.
        for hw in 0..4 {
            assert_eq!(sliced_data[hw], 1.0);
            assert_eq!(sliced_data[4 + hw], 2.0);
        }
    }

    #[test]
    fn test_slice_tensor_channels_bias_1d() {
        // A rank-1 bias vector slices along axis 0.
        let tensor =
            Tensor::<f32>::from_vec(vec![10.0, 11.0, 12.0, 13.0], &[4]).expect("test: bias");
        let sliced = slice_tensor_channels(&tensor, 2, 4).expect("test: slice bias");
        assert_eq!(sliced.shape().dims(), &[2]);
        assert_eq!(sliced.to_vec().expect("test: to_vec"), vec![12.0, 13.0]);
    }

    #[test]
    fn test_slice_tensor_channels_invalid_range() {
        let tensor = Tensor::<f32>::zeros(&[1, 4, 2, 2]);
        assert!(slice_tensor_channels(&tensor, 2, 1).is_err());
        assert!(slice_tensor_channels(&tensor, 0, 5).is_err());
    }

    #[test]
    fn test_concatenate_tensors_channel_axis() {
        // Two [1, 2, 2, 2] tensors concatenated along the channel axis -> [1, 4, 2, 2].
        let a =
            Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 2, 2, 2])
                .expect("test: tensor a");
        let b = Tensor::<f32>::from_vec(
            vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            &[1, 2, 2, 2],
        )
        .expect("test: tensor b");

        let result = concatenate_tensors(&[&a, &b], 1).expect("test: concat");
        assert_eq!(result.shape().dims(), &[1, 4, 2, 2]);

        let expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
        assert_eq!(result.to_vec().expect("test: to_vec"), expected);
    }

    #[test]
    fn test_concatenate_tensors_outer_axis() {
        // Concatenation along axis 0 (batch / out-channel style).
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("test: tensor a");
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0], &[1, 2]).expect("test: tensor b");

        let result = concatenate_tensors(&[&a, &b], 0).expect("test: concat axis 0");
        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(
            result.to_vec().expect("test: to_vec"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_concatenate_tensors_shape_mismatch() {
        let a = Tensor::<f32>::zeros(&[1, 2, 2, 2]);
        let b = Tensor::<f32>::zeros(&[1, 2, 3, 2]); // mismatched non-concat axis
        assert!(concatenate_tensors(&[&a, &b], 1).is_err());
        assert!(concatenate_tensors::<f32>(&[], 0).is_err());
    }
}
