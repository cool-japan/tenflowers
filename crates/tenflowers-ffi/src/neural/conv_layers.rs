//! Convolutional layers module for TenfloweRS FFI
//!
//! This module provides convolutional layer implementations including Conv2D, Conv1D,
//! MaxPool2D, AvgPool2D and related operations for computer vision and sequence modeling.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;
use tenflowers_neural::layers::Layer;

/// Pooling reduction kind dispatched to the matching core op.
#[derive(Clone, Copy)]
enum PoolKind {
    Max,
    Avg,
}

/// Run a 2D pooling op on an `NCHW` tensor.
///
/// The CPU pooling kernels in `tenflowers-core` operate on the `NHWC` layout, so
/// the `NCHW` input is permuted to `NHWC`, pooled, then permuted back. The result
/// is materialised contiguously so downstream consumers observe logical `NCHW`
/// order. This wires the FFI pooling layers to the real `max_pool2d`/`avg_pool2d`
/// implementations instead of returning a placeholder.
fn pool2d_nchw(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    kind: PoolKind,
) -> PyResult<PyTensor> {
    let nhwc = tenflowers_core::ops::manipulation::transpose_axes(
        input.tensor.as_ref(),
        Some(&[0, 2, 3, 1]),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("pooling layout transpose failed: {e}")))?;

    let pooled = match kind {
        PoolKind::Max => tenflowers_core::ops::max_pool2d(&nhwc, kernel_size, stride, "valid"),
        PoolKind::Avg => tenflowers_core::ops::avg_pool2d(&nhwc, kernel_size, stride, "valid"),
    }
    .map_err(|e| PyRuntimeError::new_err(format!("pooling forward failed: {e}")))?;

    let nchw_view =
        tenflowers_core::ops::manipulation::transpose_axes(&pooled, Some(&[0, 3, 1, 2])).map_err(
            |e| PyRuntimeError::new_err(format!("pooling layout transpose failed: {e}")),
        )?;
    let dims = nchw_view.shape().dims().to_vec();
    let data = nchw_view
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("pooling output read failed: {e}")))?;
    let output = Tensor::from_vec(data, &dims)
        .map_err(|e| PyRuntimeError::new_err(format!("pooling output build failed: {e}")))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Compute a single spatial output length for pooling, replicating PyTorch's
/// floor/ceil-mode rules exactly.
///
/// Returns `(output_len, extra_end_pad)` where `extra_end_pad` is the number of
/// additional padding elements that must be appended to the END (bottom/right)
/// side so a plain "valid" pooling sweep over the padded input reproduces the
/// ceil-mode window count. In floor mode `extra_end_pad` is always `0`.
///
/// `dilation` is always `1` for average pooling; max pooling may pass a real
/// dilation. This is the single source of truth for pooling output shapes and is
/// shared by both `MaxPool2D` and `AvgPool2D` (including the dilated max path).
fn pooling_output_len(
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> (usize, usize) {
    let effective_kernel = dilation * (kernel_len - 1) + 1;
    let padded_len = input_len + 2 * padding;

    let floor_len = if padded_len < effective_kernel {
        0
    } else {
        (padded_len - effective_kernel) / stride + 1
    };

    if !ceil_mode {
        return (floor_len, 0);
    }

    // Ceil-division of `padded_len - effective_kernel` by `stride`, using the
    // integer `(n + s - 1) / s` idiom already used for "same"-padding formulas
    // elsewhere in this codebase (no float / `div_ceil` calls).
    let numerator = padded_len.saturating_sub(effective_kernel);
    let mut ceil_len = (numerator + stride - 1) / stride + 1;

    // PyTorch drops a pooling window that would start entirely inside the right
    // padding region.
    if (ceil_len - 1) * stride >= input_len + padding {
        ceil_len -= 1;
    }

    if ceil_len <= floor_len {
        // ceil_mode is a no-op for this configuration.
        return (floor_len, 0);
    }

    let needed_padded_len = (ceil_len - 1) * stride + effective_kernel;
    let extra_end_pad = needed_padded_len - padded_len;
    (ceil_len, extra_end_pad)
}

/// Correct an `AvgPool2D` result for the ceil-mode overhang region.
///
/// The core "valid" average pool divides every window by the full kernel area,
/// which is correct for interior windows and for windows touched only by the
/// explicit `padding` (PyTorch's `count_include_pad=True` default). PyTorch,
/// however, always excludes the ceil-mode overhang from the divisor. For every
/// output position the true divisor is the window area clipped to the
/// padding-only extent (`in + 2*padding`); where that differs from the full
/// kernel area the value is rescaled accordingly.
fn avg_pool2d_boundary_correct(
    pooled: &PyTensor,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
) -> PyResult<PyTensor> {
    let dims = pooled.tensor.shape().dims().to_vec();
    let batch = dims[0];
    let channels = dims[1];
    let out_h = dims[2];
    let out_w = dims[3];

    let mut data = pooled
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("AvgPool2D output read failed: {e}")))?;

    let full = (kh * kw) as f32;
    let padded_h = in_h + 2 * padding.0;
    let padded_w = in_w + 2 * padding.1;

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                let valid_h = std::cmp::min(padded_h.saturating_sub(oh * stride.0), kh);
                for ow in 0..out_w {
                    let valid_w = std::cmp::min(padded_w.saturating_sub(ow * stride.1), kw);
                    let valid = valid_h * valid_w;
                    if valid > 0 && valid != kh * kw {
                        let idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        data[idx] *= full / valid as f32;
                    }
                }
            }
        }
    }

    let output = Tensor::from_vec(data, &dims)
        .map_err(|e| PyRuntimeError::new_err(format!("AvgPool2D output build failed: {e}")))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: pooled.requires_grad,
        is_pinned: pooled.is_pinned,
    })
}

/// Dilated 2D max pooling over an `NCHW` `f32` tensor.
///
/// The core "valid" max-pool op samples contiguous kernel taps only, so genuine
/// dilation needs an explicit loop. The input is first padded with
/// `f32::NEG_INFINITY` (which never wins a max) by the explicit `padding` on both
/// sides plus any ceil-mode overhang (`extra`) on the end, then each output
/// position samples taps at `start + k * dilation`, mirroring the index-transform
/// used by this workspace's depthwise convolution.
fn max_pool2d_dilated(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    out_hw: (usize, usize),
    extra_hw: (usize, usize),
) -> PyResult<PyTensor> {
    let (kh, kw) = kernel_size;
    let (out_h, out_w) = out_hw;
    let (extra_h, extra_w) = extra_hw;

    let in_shape = input.tensor.shape().dims().to_vec();
    let batch = in_shape[0];
    let channels = in_shape[1];

    let pad_spec = [
        (0, 0),
        (0, 0),
        (padding.0, padding.0 + extra_h),
        (padding.1, padding.1 + extra_w),
    ];
    let padded = tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, f32::NEG_INFINITY)
        .map_err(|e| PyRuntimeError::new_err(format!("MaxPool2D padding failed: {e}")))?;
    let padded_dims = padded.shape().dims().to_vec();
    let padded_h = padded_dims[2];
    let padded_w = padded_dims[3];
    let padded_data = padded
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("MaxPool2D input read failed: {e}")))?;

    let mut output_data = vec![0.0f32; batch * channels * out_h * out_w];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = oh * stride.0;
                for ow in 0..out_w {
                    let w_start = ow * stride.1;
                    let mut max_val = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        let ih = h_start + ki * dilation.0;
                        if ih >= padded_h {
                            continue;
                        }
                        for kj in 0..kw {
                            let iw = w_start + kj * dilation.1;
                            if iw >= padded_w {
                                continue;
                            }
                            let idx = ((b * channels + c) * padded_h + ih) * padded_w + iw;
                            let val = padded_data[idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    output_data[out_idx] = max_val;
                }
            }
        }
    }

    let output = Tensor::from_vec(output_data, &[batch, channels, out_h, out_w])
        .map_err(|e| PyRuntimeError::new_err(format!("MaxPool2D output build failed: {e}")))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// General 2D convolution over an `NCHW` `f32` tensor supporting dilation and
/// grouped convolution.
///
/// The core `conv2d` op implements standard convolution only, so this port of
/// the same nested-loop pattern used by this workspace's `conv1d`/`conv3d`
/// forwards handles `dilation != (1, 1)` and `groups != 1`. Each output channel
/// `oc` only sees the input channels owned by its group
/// (`group = oc / (out_channels / groups)`), and kernel tap `k` reads
/// `start * stride + k * dilation`. Explicit integer padding is composed by the
/// caller (pad-then-valid), matching the standard path.
fn conv2d_general(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    stride: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> tenflowers_core::Result<Tensor<f32>> {
    let in_shape = input.shape().dims().to_vec();
    let w_shape = weight.shape().dims().to_vec();

    let batch = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let out_channels = w_shape[0];
    let in_channels_per_group = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];

    if in_channels != in_channels_per_group * groups {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
            "Conv2D: input channels ({in_channels}) must equal in_channels_per_group ({in_channels_per_group}) * groups ({groups})"
        )));
    }

    let out_h = if in_h < (kh - 1) * dilation.0 + 1 {
        0
    } else {
        (in_h - (kh - 1) * dilation.0 - 1) / stride.0 + 1
    };
    let out_w = if in_w < (kw - 1) * dilation.1 + 1 {
        0
    } else {
        (in_w - (kw - 1) * dilation.1 - 1) / stride.1 + 1
    };

    if out_h == 0 || out_w == 0 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Conv2D: output size would be zero with the given parameters".to_string(),
        ));
    }

    let input_data = input.to_vec()?;
    let weight_data = weight.to_vec()?;
    let bias_data = bias.map(|b| b.to_vec()).transpose()?;

    let out_per_group = out_channels / groups;
    let mut output_data = vec![0.0f32; batch * out_channels * out_h * out_w];

    for b in 0..batch {
        for oc in 0..out_channels {
            let group = oc / out_per_group;
            let group_start_ic = group * in_channels_per_group;
            for oh in 0..out_h {
                let h_start = oh * stride.0;
                for ow in 0..out_w {
                    let w_start = ow * stride.1;
                    let mut sum = 0.0f32;
                    for ic in 0..in_channels_per_group {
                        let real_ic = group_start_ic + ic;
                        for ki in 0..kh {
                            let ih = h_start + ki * dilation.0;
                            if ih >= in_h {
                                continue;
                            }
                            for kj in 0..kw {
                                let iw = w_start + kj * dilation.1;
                                if iw >= in_w {
                                    continue;
                                }
                                let input_idx =
                                    ((b * in_channels + real_ic) * in_h + ih) * in_w + iw;
                                let weight_idx =
                                    ((oc * in_channels_per_group + ic) * kh + ki) * kw + kj;
                                sum += input_data[input_idx] * weight_data[weight_idx];
                            }
                        }
                    }
                    if let Some(ref bd) = bias_data {
                        sum += bd[oc];
                    }
                    let out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                    output_data[out_idx] = sum;
                }
            }
        }
    }

    Tensor::from_vec(output_data, &[batch, out_channels, out_h, out_w])
}

/// 2D Convolutional Layer
///
/// Applies a 2D convolution over an input signal composed of several input planes.
/// Commonly used in computer vision applications.
#[pyclass(name = "Conv2D")]
#[derive(Debug, Clone)]
pub struct PyConv2D {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels (filters)
    pub out_channels: usize,
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width)
    pub dilation: (usize, usize),
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Whether to include bias
    pub use_bias: bool,
    /// Convolution weights (out_channels, in_channels/groups, kernel_h, kernel_w)
    pub weight: Option<Tensor<f32>>,
    /// Bias terms (out_channels,)
    pub bias: Option<Tensor<f32>>,
}

#[pymethods]
impl PyConv2D {
    /// Create a new Conv2D layer
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of channels in the input image
    /// * `out_channels` - Number of channels produced by the convolution
    /// * `kernel_size` - Size of the convolving kernel (single int or tuple)
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Zero-padding added to both sides of the input (default: 0)
    /// * `dilation` - Spacing between kernel elements (default: 1)
    /// * `groups` - Number of blocked connections from input to output channels (default: 1)
    /// * `bias` - If True, adds a learnable bias to the output (default: True)
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None, bias=None))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        if in_channels == 0 {
            return Err(PyValueError::new_err("in_channels must be positive"));
        }
        if out_channels == 0 {
            return Err(PyValueError::new_err("out_channels must be positive"));
        }
        if groups == 0 {
            return Err(PyValueError::new_err("groups must be positive"));
        }
        if in_channels % groups != 0 {
            return Err(PyValueError::new_err(format!(
                "in_channels ({}) must be divisible by groups ({})",
                in_channels, groups
            )));
        }
        if out_channels % groups != 0 {
            return Err(PyValueError::new_err(format!(
                "out_channels ({}) must be divisible by groups ({})",
                out_channels, groups
            )));
        }

        // Initialize weights using Xavier/Glorot initialization
        let weight_shape = vec![
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weight = Tensor::zeros(&weight_shape);

        let bias_tensor = if use_bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Ok(PyConv2D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            weight: Some(weight),
            bias: bias_tensor,
        })
    }

    /// Forward pass through the convolutional layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (N, C_in, H_in, W_in)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (N, C_out, H_out, W_out)
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "Expected 4D input (N, C, H, W), got {}D",
                input_shape.len()
            )));
        }

        let in_c = input_shape[1];
        if in_c != self.in_channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} input channels, got {}",
                self.in_channels, in_c
            )));
        }

        // A real weight is required - never fabricate a zero-filled output.
        let weight = self.weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Conv2D: weight not initialized; call reset_parameters first")
        })?;

        let bias_ref = self.bias.as_ref();

        // Dilation and grouped convolution cannot be expressed through the
        // standard conv2d op, so route those to the general NCHW loop. The
        // standard op is retained for the common (dilation == (1, 1),
        // groups == 1) case.
        let use_general = self.dilation != (1, 1) || self.groups != 1;

        // conv2d understands only "valid"/"same" string padding, so for explicit
        // integer padding we zero-pad the spatial dims and run a valid convolution.
        let result = if self.padding == (0, 0) {
            if use_general {
                conv2d_general(
                    input.tensor.as_ref(),
                    weight,
                    bias_ref,
                    self.stride,
                    self.dilation,
                    self.groups,
                )
            } else {
                tenflowers_core::ops::conv2d(
                    input.tensor.as_ref(),
                    weight,
                    bias_ref,
                    self.stride,
                    "valid",
                )
            }
        } else {
            let pad_spec = [
                (0, 0),
                (0, 0),
                (self.padding.0, self.padding.0),
                (self.padding.1, self.padding.1),
            ];
            match tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, 0.0) {
                Ok(padded) => {
                    if use_general {
                        conv2d_general(
                            &padded,
                            weight,
                            bias_ref,
                            self.stride,
                            self.dilation,
                            self.groups,
                        )
                    } else {
                        tenflowers_core::ops::conv2d(
                            &padded,
                            weight,
                            bias_ref,
                            self.stride,
                            "valid",
                        )
                    }
                }
                Err(e) => Err(e),
            }
        };

        match result {
            Ok(output) => Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Conv2D forward failed: {e}"
            ))),
        }
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        let weight_shape = vec![
            self.out_channels,
            self.in_channels / self.groups,
            self.kernel_size.0,
            self.kernel_size.1,
        ];
        self.weight = Some(Tensor::zeros(&weight_shape));

        if self.use_bias {
            self.bias = Some(Tensor::zeros(&[self.out_channels]));
        }

        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.weight {
            let weight_data: Vec<f32> = weight
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert weight: {}", e)))?;
            dict.set_item("weight", weight_data)?;
        }

        if let Some(ref bias) = self.bias {
            let bias_data: Vec<f32> = bias
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert bias: {}", e)))?;
            dict.set_item("bias", bias_data)?;
        }

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            let weight_shape = vec![
                self.out_channels,
                self.in_channels / self.groups,
                self.kernel_size.0,
                self.kernel_size.1,
            ];
            self.weight =
                Some(Tensor::from_vec(weight_vec, &weight_shape).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load weight: {}", e))
                })?);
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            self.bias = Some(
                Tensor::from_vec(bias_vec, &[self.out_channels])
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load bias: {}", e)))?,
            );
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Conv2D(in_channels={}, out_channels={}, kernel_size={:?}, stride={:?}, padding={:?}, dilation={:?}, groups={}, bias={})",
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups, self.use_bias
        )
    }
}

/// 2D Max Pooling Layer
///
/// Applies a 2D max pooling over an input signal composed of several input planes.
#[pyclass(name = "MaxPool2D")]
#[derive(Debug, Clone)]
pub struct PyMaxPool2D {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width)
    pub dilation: (usize, usize),
    /// Whether to return indices for unpooling
    pub return_indices: bool,
    /// Whether to use ceil instead of floor for output shape
    pub ceil_mode: bool,
}

#[pymethods]
impl PyMaxPool2D {
    /// Create a new MaxPool2D layer
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling window (default: kernel_size)
    /// * `padding` - Zero-padding added to both sides (default: 0)
    /// * `dilation` - Spacing between kernel elements (default: 1)
    /// * `return_indices` - If True, return the max indices along with the outputs (default: False)
    /// * `ceil_mode` - When True, use ceil instead of floor to compute output shape (default: False)
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dilation=None, return_indices=None, ceil_mode=None))]
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        return_indices: Option<bool>,
        ceil_mode: Option<bool>,
    ) -> PyResult<Self> {
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let return_indices = return_indices.unwrap_or(false);
        let ceil_mode = ceil_mode.unwrap_or(false);

        Ok(PyMaxPool2D {
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode,
        })
    }

    /// Forward pass through the max pooling layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "Expected 4D input (N, C, H, W), got {}D",
                input_shape.len()
            )));
        }

        // return_indices (the argmax mask consumed by MaxUnpool) is out of scope
        // for this op.
        if self.return_indices {
            return Err(PyValueError::new_err(
                "MaxPool2D: return_indices=True is not supported",
            ));
        }

        let stride = self.stride.unwrap_or(self.kernel_size);
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let (kh, kw) = self.kernel_size;

        let (out_h, extra_h) = pooling_output_len(
            in_h,
            kh,
            stride.0,
            self.padding.0,
            self.dilation.0,
            self.ceil_mode,
        );
        let (out_w, extra_w) = pooling_output_len(
            in_w,
            kw,
            stride.1,
            self.padding.1,
            self.dilation.1,
            self.ceil_mode,
        );

        if out_h == 0 || out_w == 0 {
            return Err(PyValueError::new_err(
                "MaxPool2D: computed output size is zero for the given parameters",
            ));
        }

        // Dilation cannot be expressed via the core "valid" op (it samples
        // contiguous kernel taps only), so route it to an explicit dilated loop.
        if self.dilation != (1, 1) {
            return max_pool2d_dilated(
                input,
                self.kernel_size,
                stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                (extra_h, extra_w),
            );
        }

        // Non-dilated path: compose explicit padding (plus any ceil-mode overhang
        // on the end) with the core "valid" max pool. `-inf` fill never wins a
        // max, so the padded windows reproduce PyTorch's semantics exactly.
        if self.padding == (0, 0) && extra_h == 0 && extra_w == 0 {
            return pool2d_nchw(input, self.kernel_size, stride, PoolKind::Max);
        }

        let pad_spec = [
            (0, 0),
            (0, 0),
            (self.padding.0, self.padding.0 + extra_h),
            (self.padding.1, self.padding.1 + extra_w),
        ];
        let padded = tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, f32::NEG_INFINITY)
            .map_err(|e| PyRuntimeError::new_err(format!("MaxPool2D padding failed: {e}")))?;
        let padded_py = PyTensor {
            tensor: Arc::new(padded),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        };
        pool2d_nchw(&padded_py, self.kernel_size, stride, PoolKind::Max)
    }

    fn __repr__(&self) -> String {
        format!(
            "MaxPool2D(kernel_size={:?}, stride={:?}, padding={:?}, dilation={:?})",
            self.kernel_size, self.stride, self.padding, self.dilation
        )
    }
}

/// 2D Average Pooling Layer
///
/// Applies a 2D average pooling over an input signal composed of several input planes.
#[pyclass(name = "AvgPool2D")]
#[derive(Debug, Clone)]
pub struct PyAvgPool2D {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Whether to use ceil instead of floor for output shape
    pub ceil_mode: bool,
    /// Whether to include padding in average calculation
    pub count_include_pad: bool,
    /// If specified, divide by divisor instead of pool size
    pub divisor_override: Option<usize>,
}

#[pymethods]
impl PyAvgPool2D {
    /// Create a new AvgPool2D layer
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling window (default: kernel_size)
    /// * `padding` - Zero-padding added to both sides (default: 0)
    /// * `ceil_mode` - When True, use ceil instead of floor to compute output shape (default: False)
    /// * `count_include_pad` - When True, include zero-padding in the averaging calculation (default: True)
    /// * `divisor_override` - If specified, used as the divisor for every window instead of the pool size (default: None)
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, ceil_mode=None, count_include_pad=None, divisor_override=None))]
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        ceil_mode: Option<bool>,
        count_include_pad: Option<bool>,
        divisor_override: Option<usize>,
    ) -> PyResult<Self> {
        let padding = padding.unwrap_or((0, 0));
        let ceil_mode = ceil_mode.unwrap_or(false);
        let count_include_pad = count_include_pad.unwrap_or(true);

        // Fail fast on an invalid divisor rather than deferring to forward().
        if divisor_override == Some(0) {
            return Err(PyValueError::new_err(
                "AvgPool2D: divisor_override must be a positive (non-zero) integer",
            ));
        }

        Ok(PyAvgPool2D {
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        })
    }

    /// Forward pass through the average pooling layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "Expected 4D input (N, C, H, W), got {}D",
                input_shape.len()
            )));
        }

        let stride = self.stride.unwrap_or(self.kernel_size);
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let (kh, kw) = self.kernel_size;

        // Average pooling has no dilation; compute the output length and any
        // ceil-mode overhang per spatial axis.
        let (out_h, extra_h) =
            pooling_output_len(in_h, kh, stride.0, self.padding.0, 1, self.ceil_mode);
        let (out_w, extra_w) =
            pooling_output_len(in_w, kw, stride.1, self.padding.1, 1, self.ceil_mode);

        if out_h == 0 || out_w == 0 {
            return Err(PyValueError::new_err(
                "AvgPool2D: computed output size is zero for the given parameters",
            ));
        }

        // Compose explicit padding (plus any ceil-mode overhang on the end) with
        // the core "valid" average pool. The core op divides every window by the
        // full kernel area, which matches count_include_pad=True over the padded
        // region.
        let pooled = if self.padding == (0, 0) && extra_h == 0 && extra_w == 0 {
            pool2d_nchw(input, self.kernel_size, stride, PoolKind::Avg)?
        } else {
            let pad_spec = [
                (0, 0),
                (0, 0),
                (self.padding.0, self.padding.0 + extra_h),
                (self.padding.1, self.padding.1 + extra_w),
            ];
            let padded = tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, 0.0)
                .map_err(|e| PyRuntimeError::new_err(format!("AvgPool2D padding failed: {e}")))?;
            let padded_py = PyTensor {
                tensor: Arc::new(padded),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            pool2d_nchw(&padded_py, self.kernel_size, stride, PoolKind::Avg)?
        };

        // divisor_override supersedes both count_include_pad and the ceil-mode
        // overhang exclusion: every window is divided uniformly by the override.
        // The pooled result currently divides uniformly by the kernel area, so a
        // single rescale converts that divisor and the boundary correction is
        // skipped entirely.
        if let Some(d) = self.divisor_override {
            let scale = (kh * kw) as f32 / d as f32;
            let scaled = pooled.tensor.multiply_scalar(scale).map_err(|e| {
                PyRuntimeError::new_err(format!("AvgPool2D divisor_override scaling failed: {e}"))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(scaled),
                requires_grad: pooled.requires_grad,
                is_pinned: pooled.is_pinned,
            });
        }

        // Only the ceil-mode overhang region needs its divisor corrected away from
        // the full kernel area (explicit padding alone already matches PyTorch's
        // count_include_pad=True default).
        if extra_h > 0 || extra_w > 0 {
            return avg_pool2d_boundary_correct(&pooled, in_h, in_w, kh, kw, stride, self.padding);
        }

        Ok(pooled)
    }

    fn __repr__(&self) -> String {
        format!(
            "AvgPool2D(kernel_size={:?}, stride={:?}, padding={:?})",
            self.kernel_size, self.stride, self.padding
        )
    }
}

/// 1D Convolutional Layer
///
/// Applies a 1D convolution over an input signal composed of several input planes.
/// Commonly used for sequence modeling and time series.
#[pyclass(name = "Conv1D")]
#[derive(Debug, Clone)]
pub struct PyConv1D {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels (filters)
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Dilation
    pub dilation: usize,
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Whether to include bias
    pub use_bias: bool,
    /// Convolution weights (out_channels, in_channels/groups, kernel_size)
    pub weight: Option<Tensor<f32>>,
    /// Bias terms (out_channels,)
    pub bias: Option<Tensor<f32>>,
}

#[pymethods]
impl PyConv1D {
    /// Create a new Conv1D layer
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None, bias=None))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        if in_channels == 0 {
            return Err(PyValueError::new_err("in_channels must be positive"));
        }
        if out_channels == 0 {
            return Err(PyValueError::new_err("out_channels must be positive"));
        }
        if groups == 0 {
            return Err(PyValueError::new_err("groups must be positive"));
        }
        if in_channels % groups != 0 {
            return Err(PyValueError::new_err(format!(
                "in_channels ({}) must be divisible by groups ({})",
                in_channels, groups
            )));
        }

        let weight_shape = vec![out_channels, in_channels / groups, kernel_size];
        let weight = Tensor::zeros(&weight_shape);

        let bias_tensor = if use_bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Ok(PyConv1D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            weight: Some(weight),
            bias: bias_tensor,
        })
    }

    /// Forward pass through the 1D convolutional layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (N, C_in, L_in)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (N, C_out, L_out)
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input (N, C, L), got {}D",
                input_shape.len()
            )));
        }

        let in_c = input_shape[1];
        if in_c != self.in_channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} input channels, got {}",
                self.in_channels, in_c
            )));
        }

        // A real weight is required - never fabricate a zero-filled output.
        let weight = self.weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Conv1D: weight not initialized; call reset_parameters first")
        })?;

        // The core conv1d op implements neither dilation nor grouped convolution,
        // so delegate to the neural Conv1D layer whose forward supports both. The
        // real weight/bias are attached explicitly (so `use_bias=false` avoids a
        // throwaway zero bias allocation). Explicit integer padding is still
        // composed here exactly as before: zero-pad the length dim and run a
        // "valid" convolution.
        let mut layer = tenflowers_neural::layers::conv::Conv1D::<f32>::new(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            Some(self.stride),
            Some("valid".to_string()),
            Some(self.dilation),
            Some(self.groups),
            false,
        );
        layer.set_weight(weight.clone());
        layer.set_bias(self.bias.clone());

        let result = if self.padding == 0 {
            Layer::forward(&layer, input.tensor.as_ref())
        } else {
            let pad_spec = [(0, 0), (0, 0), (self.padding, self.padding)];
            match tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, 0.0) {
                Ok(padded) => Layer::forward(&layer, &padded),
                Err(e) => Err(e),
            }
        };

        match result {
            Ok(output) => Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Conv1D forward failed: {e}"
            ))),
        }
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        let weight_shape = vec![
            self.out_channels,
            self.in_channels / self.groups,
            self.kernel_size,
        ];
        self.weight = Some(Tensor::zeros(&weight_shape));

        if self.use_bias {
            self.bias = Some(Tensor::zeros(&[self.out_channels]));
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Conv1D(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={})",
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups, self.use_bias
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    #[test]
    fn conv2d_forward_matches_core_op() {
        // in=2, out=3, kernel 2x2, valid padding, no bias.
        let mut conv = PyConv2D::new(2, 3, (2, 2), None, None, None, None, Some(false))
            .expect("conv2d construction");
        // Distinct non-zero weights [out=3, in=2, 2, 2] = 24 elements.
        let weight_data: Vec<f32> = (1..=24).map(|v| v as f32 * 0.1).collect();
        let weight = Tensor::from_vec(weight_data, &[3, 2, 2, 2]).expect("weight");
        conv.weight = Some(weight.clone());

        // Input [1, 2, 4, 4] = 32 elements.
        let input_data: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 2, 4, 4]);

        let out = conv.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 3, 3, 3]);

        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "output must not be all zeros"
        );

        // Cross-check against the core op invoked directly.
        let direct =
            tenflowers_core::ops::conv2d(input.tensor.as_ref(), &weight, None, (1, 1), "valid")
                .expect("direct conv2d");
        let direct_vec = direct.to_vec().expect("direct vec");
        assert_eq!(out_vec.len(), direct_vec.len());
        for (a, b) in out_vec.iter().zip(direct_vec.iter()) {
            assert!((a - b).abs() < 1e-6, "ffi conv2d must equal core conv2d");
        }
    }

    #[test]
    fn conv2d_padding_changes_output_shape() {
        let mut conv = PyConv2D::new(1, 1, (3, 3), None, Some((1, 1)), None, None, Some(false))
            .expect("conv2d construction");
        let weight = Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]).expect("weight");
        conv.weight = Some(weight);
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = conv.forward(&input).expect("forward");
        // "same"-style padding 1 with 3x3 kernel keeps spatial dims.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 4, 4]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn conv2d_uninitialized_weight_errors() {
        let mut conv = PyConv2D::new(1, 1, (2, 2), None, None, None, None, Some(false))
            .expect("conv2d construction");
        conv.weight = None;
        let input = make_tensor(vec![1.0; 16], &[1, 1, 4, 4]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn conv2d_groups_real_values() {
        // groups=2: output channel `oc` must only see its own group's input
        // channel. With in=2/out=2/groups=2 and 1x1 kernels, oc0 reads input
        // channel 0 and oc1 reads input channel 1 - no cross-group contamination.
        let mut conv = PyConv2D::new(2, 2, (1, 1), None, None, None, Some(2), Some(false))
            .expect("conv2d construction");
        // Weight [out=2, in/groups=1, 1, 1]: oc0 scales by 10, oc1 scales by 100.
        let weight = Tensor::from_vec(vec![10.0, 100.0], &[2, 1, 1, 1]).expect("weight");
        conv.weight = Some(weight);
        // Input [1, 2, 2, 2]: channel 0 = 2.0, channel 1 = 3.0.
        let input = make_tensor(vec![2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0], &[1, 2, 2, 2]);
        let out = conv.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // oc0 = 2*10 = 20 (would be 3*10 = 30 if groups were ignored); oc1 = 3*100 = 300.
        assert_eq!(
            out_vec,
            vec![20.0, 20.0, 20.0, 20.0, 300.0, 300.0, 300.0, 300.0]
        );
    }

    #[test]
    fn conv2d_dilation_real_values() {
        // 1 in / 1 out, kernel 2x2, dilation 2, valid, all-ones weight.
        let mut conv = PyConv2D::new(1, 1, (2, 2), None, None, Some((2, 2)), None, Some(false))
            .expect("conv2d construction");
        let weight = Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).expect("weight");
        conv.weight = Some(weight);
        // 4x4 input with values 1..16.
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = conv.forward(&input).expect("forward");
        // effective kernel = 3, out = (4-3)/1 + 1 = 2.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // (0,0)=1+3+9+11=24, (0,1)=2+4+10+12=28, (1,0)=5+7+13+15=40, (1,1)=6+8+14+16=44.
        assert_eq!(out_vec, vec![24.0, 28.0, 40.0, 44.0]);
    }

    #[test]
    fn conv2d_dilation_groups_padding_combined() {
        // Combined case: dilation + groups + explicit padding all at once.
        let mut conv = PyConv2D::new(
            2,
            2,
            (2, 2),
            None,
            Some((1, 1)),
            Some((2, 2)),
            Some(2),
            Some(false),
        )
        .expect("conv2d construction");
        // Weight [out=2, in/groups=1, 2, 2].
        let weight = Tensor::from_vec(vec![1.0; 8], &[2, 1, 2, 2]).expect("weight");
        conv.weight = Some(weight);
        // Input [1, 2, 4, 4].
        let input_data: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 2, 4, 4]);
        let out = conv.forward(&input).expect("forward");
        // padded 6x6, effective kernel 3, out = (6-3)/1 + 1 = 4.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4, 4]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn max_pool2d_forward_real_values() {
        let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), None, None, None, None)
            .expect("maxpool construction");
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(out_vec, vec![5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn max_pool2d_explicit_padding_negative_input() {
        // Negative input proves the -inf fill (a 0.0 fill would wrongly win the max).
        let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), Some((1, 1)), None, None, None)
            .expect("maxpool construction");
        let input = make_tensor(vec![-5.0; 4], &[1, 1, 2, 2]);
        let out = pool.forward(&input).expect("forward");
        // padded 4x4, out = (4-2)/2 + 1 = 2.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // Each window sees exactly one real -5.0 (rest is -inf), so max = -5.0.
        assert_eq!(out_vec, vec![-5.0, -5.0, -5.0, -5.0]);
    }

    #[test]
    fn max_pool2d_ceil_mode_real_values() {
        // 3x3 input, kernel 2, stride 2, ceil_mode=True -> 2x2 output.
        let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), None, None, None, Some(true))
            .expect("maxpool construction");
        let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 3, 3]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // Boundary windows overhang the input; -inf fill means each takes the max
        // of its real elements only: [max(1,2,4,5), max(3,6), max(7,8), max(9)].
        assert_eq!(out_vec, vec![5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn max_pool2d_dilation_real_values() {
        // kernel 2x2, dilation 2, stride 1, valid -> 2x2 output over a 4x4 input.
        let pool = PyMaxPool2D::new((2, 2), Some((1, 1)), None, Some((2, 2)), None, None)
            .expect("maxpool construction");
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // (0,0)=max(1,3,9,11)=11, (0,1)=max(2,4,10,12)=12,
        // (1,0)=max(5,7,13,15)=15, (1,1)=max(6,8,14,16)=16.
        assert_eq!(out_vec, vec![11.0, 12.0, 15.0, 16.0]);
    }

    #[test]
    fn avg_pool2d_forward_real_values() {
        let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, None, None, None)
            .expect("avgpool construction");
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(out_vec, vec![2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn avg_pool2d_explicit_padding_real_values() {
        // 2x2 input [[1,2],[3,4]], kernel 2, stride 1, padding 1. The core divides
        // every window by the full kernel area (count_include_pad=True default).
        let pool = PyAvgPool2D::new((2, 2), Some((1, 1)), Some((1, 1)), None, None, None)
            .expect("avgpool construction");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = pool.forward(&input).expect("forward");
        // padded 4x4, out = (4-2)/1 + 1 = 3.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(
            out_vec,
            vec![0.25, 0.75, 0.5, 1.0, 2.5, 1.5, 0.75, 1.75, 1.0]
        );
    }

    #[test]
    fn avg_pool2d_divisor_override_real_values() {
        // 4x4 input 1..16, kernel 2, stride 2, divisor_override=2.
        let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, None, None, Some(2))
            .expect("avgpool construction");
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // sum(window)/2: (1+2+5+6)/2=7, (3+4+7+8)/2=11, (9+10+13+14)/2=23, (11+12+15+16)/2=27.
        assert_eq!(out_vec, vec![7.0, 11.0, 23.0, 27.0]);
    }

    #[test]
    fn avg_pool2d_divisor_override_with_padding() {
        // Interaction case: divisor_override together with explicit padding. Every
        // window (including padding-touched boundary windows) is divided by 2.
        let pool = PyAvgPool2D::new((2, 2), Some((1, 1)), Some((1, 1)), None, None, Some(2))
            .expect("avgpool construction");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // window sums /2: 1/2, 3/2, 2/2, 4/2, 10/2, 6/2, 3/2, 7/2, 4/2.
        assert_eq!(out_vec, vec![0.5, 1.5, 1.0, 2.0, 5.0, 3.0, 1.5, 3.5, 2.0]);
    }

    #[test]
    fn avg_pool2d_ceil_mode_all_ones_oracle() {
        // All-ones oracle: the average of ones is always exactly 1.0, no matter how
        // many real elements are averaged. A 5x5 all-ones input with kernel 2,
        // stride 2, ceil_mode=True yields a 3x3 output whose last row/col are
        // boundary/overhang windows. Every element MUST be exactly 1.0, proving the
        // ceil-overhang divisor exclusion.
        let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, Some(true), None, None)
            .expect("avgpool construction");
        let input = make_tensor(vec![1.0; 25], &[1, 1, 5, 5]);
        let out = pool.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        for v in out_vec {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "every all-ones average must equal 1.0, got {v}"
            );
        }
    }

    #[test]
    fn conv1d_forward_real_values() {
        let mut conv = PyConv1D::new(1, 1, 2, None, None, None, None, Some(false))
            .expect("conv1d construction");
        let weight = Tensor::from_vec(vec![1.0, 1.0], &[1, 1, 2]).expect("weight");
        conv.weight = Some(weight);
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let out = conv.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(out_vec, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn conv1d_dilation_real_values() {
        // in=1, out=1, kernel 2, dilation 2, valid, all-ones weight.
        let mut conv = PyConv1D::new(1, 1, 2, None, None, Some(2), None, Some(false))
            .expect("conv1d construction");
        let weight = Tensor::from_vec(vec![1.0, 1.0], &[1, 1, 2]).expect("weight");
        conv.weight = Some(weight);
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let out = conv.forward(&input).expect("forward");
        // output_length = (5 - (2-1)*2 - 1)/1 + 1 = 3.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // out[0]=in[0]+in[2]=1+3=4, out[1]=in[1]+in[3]=2+4=6, out[2]=in[2]+in[4]=3+5=8.
        assert_eq!(out_vec, vec![4.0, 6.0, 8.0]);
    }

    #[test]
    fn conv1d_groups_real_values() {
        // groups=2: output channel `oc` only sees its own group's input channel.
        let mut conv = PyConv1D::new(2, 2, 1, None, None, None, Some(2), Some(false))
            .expect("conv1d construction");
        // Weight [out=2, in/groups=1, 1]: oc0 scales by 10, oc1 scales by 100.
        let weight = Tensor::from_vec(vec![10.0, 100.0], &[2, 1, 1]).expect("weight");
        conv.weight = Some(weight);
        // Input [1, 2, 3]: channel 0 = 2.0, channel 1 = 3.0.
        let input = make_tensor(vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0], &[1, 2, 3]);
        let out = conv.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // oc0 = 2*10 = 20 (would be 3*10 = 30 if groups were ignored); oc1 = 3*100 = 300.
        assert_eq!(out_vec, vec![20.0, 20.0, 20.0, 300.0, 300.0, 300.0]);
    }
}
