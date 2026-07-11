//! Convolutional layers module for TenfloweRS FFI
//!
//! This module provides convolutional layer implementations including Conv1D,
//! Conv2D, Conv3D, MaxPool2D, AvgPool2D and related operations for computer
//! vision and sequence modeling.
//!
//! # Autograd
//!
//! `Conv1D`/`Conv2D`/`Conv3D` follow the exact same implicit-autograd template
//! as [`super::layers::PyDense`] (see that struct's own "Autograd" doc for the
//! full design this mirrors): the weight and (optional) bias are each held as
//! a single `Py<`[`super::layers::PyParameter`]`>` — a stable-identity,
//! mutable-in-place cell (see that type's own doc for why a plain
//! `Arc<Tensor<f32>>` cannot support this) — constructed once from the
//! underlying `tenflowers_neural::layers::conv::*` layer's freshly-initialized
//! weights. `forward()` borrows that same `Py<PyParameter>` (via
//! [`Py::borrow`]), snapshots its current value via `to_tensor()`, marks the
//! snapshot as a tape leaf via [`crate::implicit_autograd::mark_leaf_param`]
//! keyed by the parameter's own `id()`, computes the result through real
//! convolution math, and — for the subset of configurations
//! [`crate::implicit_autograd::TernaryOpKind::Conv1D`]/`Conv2D`/`Conv3D`
//! actually support (unit dilation, `groups == 1`, no explicit integer padding
//! — see `forward()`'s own doc for exactly why this boundary is where it is)
//! — additionally records the operation onto the implicit tape via
//! [`crate::implicit_autograd::record_and_link_ternary`] so gradients flow
//! correctly on a later `.backward()` call. `parameters()` returns additional
//! owning references to the exact same `Py<PyParameter>` objects (via
//! [`Py::clone_ref`], never [`super::layers::PyParameter::clone_param`], which
//! would allocate a *different* identity — see [`super::layers::PyDense::parameters`]'s
//! doc for why that distinction matters), so `.grad()`/`.set_data()` on a
//! handle returned by `parameters()` observably affects (and is affected by)
//! this exact layer.
//!
//! `MaxPool2D`/`AvgPool2D` have no trainable parameters, but their forward
//! pass still participates in the tape (so gradients flow *through* them to
//! whatever precedes them) via the dedicated
//! [`crate::implicit_autograd::record_and_link_pooling`] hook, added
//! alongside this module specifically because pooling has no `TrackedTensor`
//! method to delegate to the way every `BinaryOpKind`/`UnaryOpKind`/
//! `TernaryOpKind` variant does (see that hook's own doc for the full
//! reasoning).

use crate::implicit_autograd::{self, PoolingOpKind, TernaryOpKind};
use crate::neural::layers::PyParameter;
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
///
/// This path is never recorded onto the implicit autograd tape (see
/// [`PyConv2D::forward`]'s doc for why) — it exists purely to preserve this
/// layer's full forward-math feature set.
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

/// General 3D convolution over an `NCDHW` `f32` tensor supporting dilation
/// and grouped convolution.
///
/// The core `conv3d` op implements standard convolution only (see
/// `tenflowers_core::ops::conv3d`'s own CPU kernel: it requires `weight`'s
/// `in_channels` to equal `input`'s exactly, with no groups splitting and no
/// dilation stepping at all), and — unlike `Conv1D`, whose
/// `tenflowers_neural::layers::conv::Conv1D::set_weight`/`set_bias`
/// genuinely override the `Layer` trait's default (error-returning) stubs —
/// `tenflowers_neural::layers::conv::Conv3D` does **not** override either
/// method (confirmed by reading that type's `impl Layer<T> for Conv3D<T>`
/// block in full), so routing this layer's general path through
/// `Layer::set_weight`/`Layer::forward` the way `PyConv1D` does is not
/// available here: every call would fail with `"This layer type does not
/// support weight setting"`. This function is therefore this workspace's own
/// explicit 5D port of exactly the same nested-loop pattern
/// [`conv2d_general`] already uses for the analogous Conv2D gap, extended one
/// more dimension (depth): each output channel `oc` only sees the input
/// channels owned by its group, and kernel tap `k` reads
/// `start * stride + k * dilation` along every one of the three spatial axes.
/// Explicit integer padding is composed by the caller (pad-then-valid),
/// matching the standard path.
///
/// This path is never recorded onto the implicit autograd tape (see
/// [`PyConv3D::forward`]'s doc for why) — it exists purely to preserve this
/// layer's full forward-math feature set.
fn conv3d_general(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> tenflowers_core::Result<Tensor<f32>> {
    let in_shape = input.shape().dims().to_vec();
    let w_shape = weight.shape().dims().to_vec();

    let batch = in_shape[0];
    let in_channels = in_shape[1];
    let in_d = in_shape[2];
    let in_h = in_shape[3];
    let in_w = in_shape[4];

    let out_channels = w_shape[0];
    let in_channels_per_group = w_shape[1];
    let kd = w_shape[2];
    let kh = w_shape[3];
    let kw = w_shape[4];

    if in_channels != in_channels_per_group * groups {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
            "Conv3D: input channels ({in_channels}) must equal in_channels_per_group ({in_channels_per_group}) * groups ({groups})"
        )));
    }

    let out_d = if in_d < (kd - 1) * dilation.0 + 1 {
        0
    } else {
        (in_d - (kd - 1) * dilation.0 - 1) / stride.0 + 1
    };
    let out_h = if in_h < (kh - 1) * dilation.1 + 1 {
        0
    } else {
        (in_h - (kh - 1) * dilation.1 - 1) / stride.1 + 1
    };
    let out_w = if in_w < (kw - 1) * dilation.2 + 1 {
        0
    } else {
        (in_w - (kw - 1) * dilation.2 - 1) / stride.2 + 1
    };

    if out_d == 0 || out_h == 0 || out_w == 0 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Conv3D: output size would be zero with the given parameters".to_string(),
        ));
    }

    let input_data = input.to_vec()?;
    let weight_data = weight.to_vec()?;
    let bias_data = bias.map(|b| b.to_vec()).transpose()?;

    let out_per_group = out_channels / groups;
    let mut output_data = vec![0.0f32; batch * out_channels * out_d * out_h * out_w];

    for b in 0..batch {
        for oc in 0..out_channels {
            let group = oc / out_per_group;
            let group_start_ic = group * in_channels_per_group;
            for od in 0..out_d {
                let d_start = od * stride.0;
                for oh in 0..out_h {
                    let h_start = oh * stride.1;
                    for ow in 0..out_w {
                        let w_start = ow * stride.2;
                        let mut sum = 0.0f32;
                        for ic in 0..in_channels_per_group {
                            let real_ic = group_start_ic + ic;
                            for kdi in 0..kd {
                                let id = d_start + kdi * dilation.0;
                                if id >= in_d {
                                    continue;
                                }
                                for ki in 0..kh {
                                    let ih = h_start + ki * dilation.1;
                                    if ih >= in_h {
                                        continue;
                                    }
                                    for kj in 0..kw {
                                        let iw = w_start + kj * dilation.2;
                                        if iw >= in_w {
                                            continue;
                                        }
                                        let input_idx =
                                            (((b * in_channels + real_ic) * in_d + id) * in_h + ih)
                                                * in_w
                                                + iw;
                                        let weight_idx =
                                            (((oc * in_channels_per_group + ic) * kd + kdi) * kh
                                                + ki)
                                                * kw
                                                + kj;
                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }
                        if let Some(ref bd) = bias_data {
                            sum += bd[oc];
                        }
                        let out_idx =
                            (((b * out_channels + oc) * out_d + od) * out_h + oh) * out_w + ow;
                        output_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    Tensor::from_vec(output_data, &[batch, out_channels, out_d, out_h, out_w])
}

/// Construct a fresh, zero-initialized `Py<`[`PyParameter`]`>` of the given
/// `shape`, matching the zero-initialization every
/// `tenflowers_neural::layers::conv::*` layer already performs (none of
/// `Conv1D`/`Conv2D`/`Conv3D` in that crate do Xavier/Kaiming initialization
/// the way `Dense::new_xavier` does — see that crate's `Conv{1,2,3}D::new`).
/// `requires_grad=true` always, matching every trainable parameter elsewhere
/// in this crate (e.g. `PyDense::new`'s own weight/bias construction, which
/// this mirrors).
fn zero_parameter_handle(py: Python<'_>, shape: &[usize]) -> PyResult<Py<PyParameter>> {
    let tensor = PyTensor {
        tensor: Arc::new(Tensor::zeros(shape)),
        requires_grad: true,
        is_pinned: false,
    };
    Py::new(py, PyParameter::new(tensor, Some(true)))
}

/// Borrow `param` (a `Py<`[`PyParameter`]`>` field on one of this module's
/// layers), snapshot its current value into a fresh [`PyTensor`], and mark
/// that snapshot as an implicit-autograd leaf under the parameter's own
/// stable `id()`.
///
/// Returns `(id, snapshot)`: every caller needs the `id` again immediately
/// afterward (to key [`crate::implicit_autograd::get_grad_by_id`] once
/// `.backward()` has run, and to build [`TernaryOpKind`]'s operands), so it is
/// handed back rather than forcing the caller to re-borrow `param` a second
/// time under the GIL. Mirrors [`super::layers::PyDense::forward`]'s own
/// inline `Python::attach(|py| { let weight_param = self.weight_param.borrow(py); ... })`
/// idiom, factored out here since every one of `PyConv1D`/`PyConv2D`/`PyConv3D`'s
/// `forward()` needs it twice (weight, and optionally bias).
fn snapshot_and_mark_leaf(py: Python<'_>, param: &Py<PyParameter>) -> PyResult<(usize, PyTensor)> {
    let param_ref = param.borrow(py);
    let id = param_ref.id();
    let snapshot = param_ref.to_tensor()?;
    implicit_autograd::mark_leaf_param(&snapshot, id);
    Ok((id, snapshot))
}

/// 2D Convolutional Layer
///
/// Applies a 2D convolution over an input signal composed of several input planes.
/// Commonly used in computer vision applications.
///
/// # Autograd
///
/// The weight and (optional) bias are each held as a single
/// `Py<`[`PyParameter`]`>` with a stable identity across every
/// `forward()`/`parameters()` call — see the module-level "Autograd" doc for
/// the full design, which is identical to [`super::layers::PyDense`]'s.
///
/// `forward()` always computes a real, correct convolution for the layer's
/// full configured feature set (stride, explicit integer padding, dilation,
/// groups — unchanged from this layer's pre-autograd behavior). It
/// *additionally* records the operation onto the implicit tape — via
/// [`crate::implicit_autograd::record_and_link_ternary`] with
/// [`TernaryOpKind::Conv2D`] — only for the exact subset of configurations
/// that op actually has real backward math for: unit dilation, `groups == 1`,
/// and no explicit integer padding (`self.padding == (0, 0)`), which is
/// precisely the `(stride, padding: &str)` shape
/// `tenflowers_autograd::Operation::Conv2D`'s backward
/// (`tenflowers_autograd::ops::convolution_ops::conv2d_backward`) supports.
///
/// This split is deliberate, not a limitation this file introduces: dilation
/// and grouped convolution are computed via [`conv2d_general`] (a plain NCHW
/// loop with no tape awareness at all), and explicit integer padding is
/// composed by zero-padding the input *before* convolving — in both cases
/// there is no matching `Operation` variant (or, for padding, no way to
/// attribute the recorded op's gradient back to the original, unpadded
/// `input` operand rather than a throwaway padded intermediate) for
/// `record_and_link_ternary` to link onto the tape. A configuration outside
/// this subset still produces a fully correct forward value; it simply does
/// not (yet) support `.backward()` through this particular layer instance —
/// exactly the same "real math now, tape support only where a real backward
/// implementation exists" boundary [`UnaryOpKind`](crate::implicit_autograd::UnaryOpKind)'s
/// own doc documents for `Gelu`/`Swish`/`Mish`/etc.
#[pyclass(name = "Conv2D")]
#[derive(Debug)]
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
    /// Stable parameter identity for the weight, shared between every
    /// `forward()` snapshot and every `parameters()` call — see the
    /// struct-level "Autograd" doc.
    weight_param: Py<PyParameter>,
    /// Stable parameter identity for the bias (`None` when `use_bias=False`).
    bias_param: Option<Py<PyParameter>>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
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

        // Zero-initialize weight/bias, matching
        // `tenflowers_neural::layers::conv::Conv2D::new`'s own initialization
        // (see `zero_parameter_handle`'s doc).
        let weight_shape = vec![
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weight_param = zero_parameter_handle(py, &weight_shape)?;
        let bias_param = if use_bias {
            Some(zero_parameter_handle(py, &[out_channels])?)
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
            weight_param,
            bias_param,
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
    ///
    /// See the struct-level "Autograd" doc for exactly which configurations
    /// additionally get tape-recorded (and therefore support `.backward()`)
    /// versus which only get a correct forward value.
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
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

        let (_, weight) = snapshot_and_mark_leaf(py, &self.weight_param)?;
        let bias = self
            .bias_param
            .as_ref()
            .map(|b| snapshot_and_mark_leaf(py, b))
            .transpose()?
            .map(|(_, snapshot)| snapshot);

        // Dilation and grouped convolution cannot be expressed through the
        // standard conv2d op, so route those to the general NCHW loop. The
        // standard op is retained for the common (dilation == (1, 1),
        // groups == 1) case, which is also the only case tape-recordable
        // below (see the struct-level "Autograd" doc).
        let use_general = self.dilation != (1, 1) || self.groups != 1;
        // The plain, tape-recordable path additionally requires no explicit
        // integer padding — see the struct-level "Autograd" doc for why
        // pre-padding the input would misattribute the recorded gradient to
        // a throwaway padded intermediate instead of the real `input`.
        let tape_recordable = !use_general && self.padding == (0, 0);

        let result = if self.padding == (0, 0) {
            if use_general {
                conv2d_general(
                    input.tensor.as_ref(),
                    &weight.tensor,
                    bias.as_ref().map(|b| b.tensor.as_ref()),
                    self.stride,
                    self.dilation,
                    self.groups,
                )
            } else {
                tenflowers_core::ops::conv2d(
                    input.tensor.as_ref(),
                    &weight.tensor,
                    bias.as_ref().map(|b| b.tensor.as_ref()),
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
                            &weight.tensor,
                            bias.as_ref().map(|b| b.tensor.as_ref()),
                            self.stride,
                            self.dilation,
                            self.groups,
                        )
                    } else {
                        tenflowers_core::ops::conv2d(
                            &padded,
                            &weight.tensor,
                            bias.as_ref().map(|b| b.tensor.as_ref()),
                            self.stride,
                            "valid",
                        )
                    }
                }
                Err(e) => Err(e),
            }
        };

        let output = match result {
            Ok(output) => PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad || weight.requires_grad,
                is_pinned: input.is_pinned,
            },
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Conv2D forward failed: {e}"
                )))
            }
        };

        if tape_recordable {
            implicit_autograd::record_and_link_ternary(
                TernaryOpKind::Conv2D {
                    stride: self.stride,
                    padding: "valid".to_string(),
                },
                input,
                &weight,
                bias.as_ref(),
                None,
                &output,
            )?;
        }

        Ok(output)
    }

    /// Reset layer parameters to zero, preserving each parameter's stable
    /// identity (via [`PyParameter::set_data`]) rather than replacing the
    /// `Py<PyParameter>` objects themselves — replacing them would silently
    /// detach `.parameters()`'s previously-returned handles from any future
    /// `forward()` call, breaking the exact identity-sharing this layer's
    /// autograd design depends on.
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        let weight_shape = vec![
            self.out_channels,
            self.in_channels / self.groups,
            self.kernel_size.0,
            self.kernel_size.1,
        ];
        self.weight_param
            .borrow(py)
            .set_data(Tensor::zeros(&weight_shape))?;

        if let Some(bias_param) = &self.bias_param {
            bias_param
                .borrow(py)
                .set_data(Tensor::zeros(&[self.out_channels]))?;
        }

        Ok(())
    }

    /// Get layer parameters as [`PyParameter`] handles.
    ///
    /// Returns additional owning references to the exact same
    /// `Py<PyParameter>` objects this layer itself holds (via
    /// [`Py::clone_ref`]), so `.grad()` on the parameters returned here is
    /// populated after a `.backward()` call that passes through this layer's
    /// `forward()` in its tape-recordable configuration, and `.set_data()` on
    /// a returned handle (e.g. from an optimizer) is visible to this layer's
    /// own next `forward()` call. See the module-level "Autograd" doc and
    /// [`super::layers::PyDense::parameters`]'s doc for the full reasoning.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = vec![self.weight_param.clone_ref(py)];
        if let Some(bias_param) = &self.bias_param {
            params.push(bias_param.clone_ref(py));
        }
        params
    }

    /// Get number of parameters
    pub fn num_parameters(&self, py: Python<'_>) -> usize {
        let mut count = self.weight_param.borrow(py).size();
        if let Some(bias_param) = &self.bias_param {
            count += bias_param.borrow(py).size();
        }
        count
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let weight_data: Vec<f32> = self
            .weight_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert weight: {}", e)))?;
        dict.set_item("weight", weight_data)?;

        if let Some(bias_param) = &self.bias_param {
            let bias_data: Vec<f32> = bias_param
                .borrow(py)
                .to_tensor()?
                .tensor
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert bias: {}", e)))?;
            dict.set_item("bias", bias_data)?;
        }

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            let weight_shape = vec![
                self.out_channels,
                self.in_channels / self.groups,
                self.kernel_size.0,
                self.kernel_size.1,
            ];
            let tensor = Tensor::from_vec(weight_vec, &weight_shape)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load weight: {}", e)))?;
            self.weight_param.borrow(py).set_data(tensor)?;
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            let tensor = Tensor::from_vec(bias_vec, &[self.out_channels])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load bias: {}", e)))?;
            match &self.bias_param {
                Some(bias_param) => bias_param.borrow(py).set_data(tensor)?,
                None => {
                    return Err(PyRuntimeError::new_err(
                        "Cannot load bias into a Conv2D layer constructed with bias=False",
                    ))
                }
            }
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
///
/// # Autograd
///
/// `MaxPool2D` has no trainable parameters, but its forward pass still
/// participates in the implicit autograd tape via
/// [`crate::implicit_autograd::record_and_link_pooling`] with
/// [`PoolingOpKind::MaxPool2D`] — exactly like `Conv2D`, this is only wired
/// for the subset of configurations that op's real backward
/// (`tenflowers_autograd::ops::convolution_ops::max_pool2d_backward`) actually
/// supports: no explicit padding, no ceil-mode overhang, and unit dilation
/// (dilated max pooling always uses [`max_pool2d_dilated`]'s explicit loop,
/// which has no matching `Operation` variant). A configuration outside that
/// subset still produces a fully correct forward value; it simply does not
/// (yet) support `.backward()` through this particular call.
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
        // This path has no matching `Operation` variant, so it is never
        // tape-recorded — see the struct-level "Autograd" doc.
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
        // Only the padding==0, no-ceil-overhang case is tape-recordable (see the
        // struct-level "Autograd" doc): every other case pre-pads the input
        // (a throwaway intermediate the tape cannot attribute a gradient back
        // through to the real `input`), exactly mirroring why `Conv2D::forward`
        // only tape-records its own padding==0 case.
        if self.padding == (0, 0) && extra_h == 0 && extra_w == 0 {
            let output = pool2d_nchw(input, self.kernel_size, stride, PoolKind::Max)?;
            implicit_autograd::record_and_link_pooling(
                PoolingOpKind::MaxPool2D {
                    kernel_size: self.kernel_size,
                    stride,
                    padding: "valid".to_string(),
                },
                input,
                &output,
            )?;
            return Ok(output);
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
///
/// # Autograd
///
/// `AvgPool2D` has no trainable parameters, but its forward pass still
/// participates in the implicit autograd tape via
/// [`crate::implicit_autograd::record_and_link_pooling`] with
/// [`PoolingOpKind::AvgPool2D`] for the subset of configurations that op's
/// real backward (`tenflowers_autograd::ops::convolution_ops::avg_pool2d_backward`)
/// supports: no explicit padding, no ceil-mode overhang, and no
/// `divisor_override` (the backward always divides by the true kernel area).
/// A configuration outside that subset still produces a fully correct forward
/// value; it simply does not (yet) support `.backward()` through this
/// particular call — see [`PyMaxPool2D`]'s struct-level doc for why this
/// split exists.
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

        // Whether this call's configuration is within the real backward's
        // supported subset — see the struct-level "Autograd" doc. Computed
        // up front since several early-return branches below (divisor
        // override, boundary correction) fall outside it.
        let tape_recordable = self.padding == (0, 0)
            && extra_h == 0
            && extra_w == 0
            && self.divisor_override.is_none();

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

        if tape_recordable {
            implicit_autograd::record_and_link_pooling(
                PoolingOpKind::AvgPool2D {
                    kernel_size: self.kernel_size,
                    stride,
                    padding: "valid".to_string(),
                },
                input,
                &pooled,
            )?;
        }

        // divisor_override supersedes both count_include_pad and the ceil-mode
        // overhang exclusion: every window is divided uniformly by the override.
        // The pooled result currently divides uniformly by the kernel area, so a
        // single rescale converts that divisor and the boundary correction is
        // skipped entirely. This path is never tape-recordable (see
        // `tape_recordable` above), so no further recording is needed for the
        // rescaled result.
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
        // count_include_pad=True default). This path is never tape-recordable
        // either (see `tape_recordable` above).
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
///
/// # Autograd
///
/// Follows the exact same design as [`PyConv2D`] (see that struct's own
/// "Autograd" doc) at the 1D level: weight/bias are each a single
/// `Py<`[`PyParameter`]`>`, `forward()` is tape-recordable via
/// [`crate::implicit_autograd::record_and_link_ternary`] with
/// [`TernaryOpKind::Conv1D`] only when `dilation == 1`, `groups == 1`, and
/// `padding == 0` — the exact subset
/// `tenflowers_autograd::Operation::Conv1D`'s backward
/// (`tenflowers_autograd::ops::convolution_ops::conv1d_backward`) supports.
#[pyclass(name = "Conv1D")]
#[derive(Debug)]
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
    /// Stable parameter identity for the weight — see [`PyConv2D`]'s
    /// struct-level "Autograd" doc.
    weight_param: Py<PyParameter>,
    /// Stable parameter identity for the bias (`None` when `use_bias=False`).
    bias_param: Option<Py<PyParameter>>,
}

#[pymethods]
impl PyConv1D {
    /// Create a new Conv1D layer
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None, bias=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
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
        let weight_param = zero_parameter_handle(py, &weight_shape)?;
        let bias_param = if use_bias {
            Some(zero_parameter_handle(py, &[out_channels])?)
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
            weight_param,
            bias_param,
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
    ///
    /// See the struct-level "Autograd" doc for exactly which configurations
    /// additionally get tape-recorded.
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
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

        let (_, weight) = snapshot_and_mark_leaf(py, &self.weight_param)?;
        let bias = self
            .bias_param
            .as_ref()
            .map(|b| snapshot_and_mark_leaf(py, b))
            .transpose()?
            .map(|(_, snapshot)| snapshot);

        let tape_recordable = self.dilation == 1 && self.groups == 1 && self.padding == 0;

        // The plain (dilation == 1, groups == 1, padding == 0) case is
        // computed directly via the core op — matching exactly what
        // `record_and_link_ternary`'s `TernaryOpKind::Conv1D` records so the
        // two never disagree. Every other configuration (the core op
        // implements neither dilation nor grouped convolution) delegates to
        // the neural Conv1D layer's `Layer::forward`, which supports both;
        // explicit integer padding is composed by zero-padding the length
        // dim first, exactly as before.
        let result = if tape_recordable {
            tenflowers_core::ops::conv1d(
                input.tensor.as_ref(),
                &weight.tensor,
                bias.as_ref().map(|b| b.tensor.as_ref()),
                self.stride,
                "valid",
            )
        } else {
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
            let set_params: tenflowers_core::Result<()> = (|| {
                Layer::set_weight(&mut layer, (*weight.tensor).clone())?;
                Layer::set_bias(&mut layer, bias.as_ref().map(|b| (*b.tensor).clone()))?;
                Ok(())
            })();

            match set_params {
                Ok(()) => {
                    if self.padding == 0 {
                        Layer::forward(&layer, input.tensor.as_ref())
                    } else {
                        let pad_spec = [(0, 0), (0, 0), (self.padding, self.padding)];
                        match tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, 0.0) {
                            Ok(padded) => Layer::forward(&layer, &padded),
                            Err(e) => Err(e),
                        }
                    }
                }
                Err(e) => Err(e),
            }
        };

        let output = match result {
            Ok(output) => PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad || weight.requires_grad,
                is_pinned: input.is_pinned,
            },
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Conv1D forward failed: {e}"
                )))
            }
        };

        if tape_recordable {
            implicit_autograd::record_and_link_ternary(
                TernaryOpKind::Conv1D {
                    stride: self.stride,
                    padding: "valid".to_string(),
                },
                input,
                &weight,
                bias.as_ref(),
                None,
                &output,
            )?;
        }

        Ok(output)
    }

    /// Reset layer parameters to zero, preserving each parameter's stable
    /// identity — see [`PyConv2D::reset_parameters`]'s doc for why this uses
    /// `set_data` rather than replacing the `Py<PyParameter>` objects.
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        let weight_shape = vec![
            self.out_channels,
            self.in_channels / self.groups,
            self.kernel_size,
        ];
        self.weight_param
            .borrow(py)
            .set_data(Tensor::zeros(&weight_shape))?;

        if let Some(bias_param) = &self.bias_param {
            bias_param
                .borrow(py)
                .set_data(Tensor::zeros(&[self.out_channels]))?;
        }

        Ok(())
    }

    /// Get layer parameters as [`PyParameter`] handles. See
    /// [`PyConv2D::parameters`]'s doc.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = vec![self.weight_param.clone_ref(py)];
        if let Some(bias_param) = &self.bias_param {
            params.push(bias_param.clone_ref(py));
        }
        params
    }

    /// Get number of parameters
    pub fn num_parameters(&self, py: Python<'_>) -> usize {
        let mut count = self.weight_param.borrow(py).size();
        if let Some(bias_param) = &self.bias_param {
            count += bias_param.borrow(py).size();
        }
        count
    }

    fn __repr__(&self) -> String {
        format!(
            "Conv1D(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={})",
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups, self.use_bias
        )
    }
}

/// 3D Convolutional Layer
///
/// Applies a 3D convolution over an input signal composed of several input
/// planes. Commonly used for video and volumetric (e.g. medical imaging)
/// data.
///
/// # Autograd
///
/// Follows the exact same design as [`PyConv2D`] (see that struct's own
/// "Autograd" doc) at the 3D-tuple level: weight/bias are each a single
/// `Py<`[`PyParameter`]`>`, `forward()` is tape-recordable via
/// [`crate::implicit_autograd::record_and_link_ternary`] with
/// [`TernaryOpKind::Conv3D`] only when `dilation == (1, 1, 1)`,
/// `groups == 1`, and `padding == (0, 0, 0)` — the exact subset
/// `tenflowers_autograd::Operation::Conv3D`'s backward
/// (`tenflowers_autograd::ops::convolution_ops::conv3d_backward`) supports.
#[pyclass(name = "Conv3D")]
#[derive(Debug)]
pub struct PyConv3D {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels (filters)
    pub out_channels: usize,
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
    /// Dilation (depth, height, width)
    pub dilation: (usize, usize, usize),
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Whether to include bias
    pub use_bias: bool,
    /// Stable parameter identity for the weight — see [`PyConv2D`]'s
    /// struct-level "Autograd" doc.
    weight_param: Py<PyParameter>,
    /// Stable parameter identity for the bias (`None` when `use_bias=False`).
    bias_param: Option<Py<PyParameter>>,
}

#[pymethods]
impl PyConv3D {
    /// Create a new Conv3D layer
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of channels in the input volume
    /// * `out_channels` - Number of channels produced by the convolution
    /// * `kernel_size` - Size of the convolving kernel (depth, height, width)
    /// * `stride` - Stride of the convolution (default: (1, 1, 1))
    /// * `padding` - Zero-padding added to all sides of the input (default: (0, 0, 0))
    /// * `dilation` - Spacing between kernel elements (default: (1, 1, 1))
    /// * `groups` - Number of blocked connections from input to output channels (default: 1)
    /// * `bias` - If True, adds a learnable bias to the output (default: True)
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None, bias=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: Option<(usize, usize, usize)>,
        dilation: Option<(usize, usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or((1, 1, 1));
        let padding = padding.unwrap_or((0, 0, 0));
        let dilation = dilation.unwrap_or((1, 1, 1));
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

        let weight_shape = vec![
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ];
        let weight_param = zero_parameter_handle(py, &weight_shape)?;
        let bias_param = if use_bias {
            Some(zero_parameter_handle(py, &[out_channels])?)
        } else {
            None
        };

        Ok(PyConv3D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            weight_param,
            bias_param,
        })
    }

    /// Forward pass through the 3D convolutional layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (N, C_in, D_in, H_in, W_in)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (N, C_out, D_out, H_out, W_out)
    ///
    /// See the struct-level "Autograd" doc for exactly which configurations
    /// additionally get tape-recorded.
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 5 {
            return Err(PyValueError::new_err(format!(
                "Expected 5D input (N, C, D, H, W), got {}D",
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

        let (_, weight) = snapshot_and_mark_leaf(py, &self.weight_param)?;
        let bias = self
            .bias_param
            .as_ref()
            .map(|b| snapshot_and_mark_leaf(py, b))
            .transpose()?
            .map(|(_, snapshot)| snapshot);

        let tape_recordable =
            self.dilation == (1, 1, 1) && self.groups == 1 && self.padding == (0, 0, 0);

        // Same split as Conv2D: the plain case goes through the core op
        // directly (matching exactly what `TernaryOpKind::Conv3D` records);
        // everything else (dilation, grouped convolution) is computed via
        // [`conv3d_general`] — this workspace's own explicit-loop
        // implementation, NOT a delegation to
        // `tenflowers_neural::layers::conv::Conv3D::forward` the way Conv1D
        // uses `Layer::forward` — see that function's own doc for why: unlike
        // `Conv1D`, `Conv3D` in that crate does not override
        // `Layer::set_weight`/`set_bias`, so there is no way to hand this
        // layer's actual weight/bias to it at all. Explicit integer padding
        // is composed by zero-padding the input first, exactly like Conv2D.
        let use_general = self.dilation != (1, 1, 1) || self.groups != 1;
        let result = if self.padding == (0, 0, 0) {
            if use_general {
                conv3d_general(
                    input.tensor.as_ref(),
                    &weight.tensor,
                    bias.as_ref().map(|b| b.tensor.as_ref()),
                    self.stride,
                    self.dilation,
                    self.groups,
                )
            } else {
                tenflowers_core::ops::conv3d(
                    input.tensor.as_ref(),
                    &weight.tensor,
                    bias.as_ref().map(|b| b.tensor.as_ref()),
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
                (self.padding.2, self.padding.2),
            ];
            match tenflowers_core::ops::pad(input.tensor.as_ref(), &pad_spec, 0.0) {
                Ok(padded) => {
                    if use_general {
                        conv3d_general(
                            &padded,
                            &weight.tensor,
                            bias.as_ref().map(|b| b.tensor.as_ref()),
                            self.stride,
                            self.dilation,
                            self.groups,
                        )
                    } else {
                        tenflowers_core::ops::conv3d(
                            &padded,
                            &weight.tensor,
                            bias.as_ref().map(|b| b.tensor.as_ref()),
                            self.stride,
                            "valid",
                        )
                    }
                }
                Err(e) => Err(e),
            }
        };

        let output = match result {
            Ok(output) => PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad || weight.requires_grad,
                is_pinned: input.is_pinned,
            },
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Conv3D forward failed: {e}"
                )))
            }
        };

        if tape_recordable {
            implicit_autograd::record_and_link_ternary(
                TernaryOpKind::Conv3D {
                    stride: self.stride,
                    padding: "valid".to_string(),
                },
                input,
                &weight,
                bias.as_ref(),
                None,
                &output,
            )?;
        }

        Ok(output)
    }

    /// Reset layer parameters to zero, preserving each parameter's stable
    /// identity — see [`PyConv2D::reset_parameters`]'s doc for why this uses
    /// `set_data` rather than replacing the `Py<PyParameter>` objects.
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        let weight_shape = vec![
            self.out_channels,
            self.in_channels / self.groups,
            self.kernel_size.0,
            self.kernel_size.1,
            self.kernel_size.2,
        ];
        self.weight_param
            .borrow(py)
            .set_data(Tensor::zeros(&weight_shape))?;

        if let Some(bias_param) = &self.bias_param {
            bias_param
                .borrow(py)
                .set_data(Tensor::zeros(&[self.out_channels]))?;
        }

        Ok(())
    }

    /// Get layer parameters as [`PyParameter`] handles. See
    /// [`PyConv2D::parameters`]'s doc.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = vec![self.weight_param.clone_ref(py)];
        if let Some(bias_param) = &self.bias_param {
            params.push(bias_param.clone_ref(py));
        }
        params
    }

    /// Get number of parameters
    pub fn num_parameters(&self, py: Python<'_>) -> usize {
        let mut count = self.weight_param.borrow(py).size();
        if let Some(bias_param) = &self.bias_param {
            count += bias_param.borrow(py).size();
        }
        count
    }

    fn __repr__(&self) -> String {
        format!(
            "Conv3D(in_channels={}, out_channels={}, kernel_size={:?}, stride={:?}, padding={:?}, dilation={:?}, groups={}, bias={})",
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups, self.use_bias
        )
    }
}

#[cfg(test)]
mod tests;
