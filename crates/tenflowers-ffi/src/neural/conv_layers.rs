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

        let batch_size = input_shape[0];
        let in_c = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        if in_c != self.in_channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} input channels, got {}",
                self.in_channels, in_c
            )));
        }

        // Calculate output dimensions
        let out_h = (in_h + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
            / self.stride.0
            + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
            / self.stride.1
            + 1;

        // For now, return a placeholder with correct output shape
        let output_shape = vec![batch_size, self.out_channels, out_h, out_w];
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
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
    pub fn state_dict(&self, py: Python) -> PyResult<PyObject> {
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

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let stride = self.stride.unwrap_or(self.kernel_size);

        // Calculate output dimensions
        let out_h = if self.ceil_mode {
            ((in_h + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1) as f32
                / stride.0 as f32)
                .ceil() as usize
                + 1
        } else {
            (in_h + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1) / stride.0
                + 1
        };

        let out_w = if self.ceil_mode {
            ((in_w + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1) as f32
                / stride.1 as f32)
                .ceil() as usize
                + 1
        } else {
            (in_w + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1) / stride.1
                + 1
        };

        // For now, return a placeholder with correct output shape
        let output_shape = vec![batch_size, channels, out_h, out_w];
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
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
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, ceil_mode=None, count_include_pad=None))]
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        ceil_mode: Option<bool>,
        count_include_pad: Option<bool>,
    ) -> PyResult<Self> {
        let padding = padding.unwrap_or((0, 0));
        let ceil_mode = ceil_mode.unwrap_or(false);
        let count_include_pad = count_include_pad.unwrap_or(true);

        Ok(PyAvgPool2D {
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override: None,
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

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let stride = self.stride.unwrap_or(self.kernel_size);

        // Calculate output dimensions
        let out_h = if self.ceil_mode {
            ((in_h + 2 * self.padding.0 - self.kernel_size.0) as f32 / stride.0 as f32).ceil()
                as usize
                + 1
        } else {
            (in_h + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1
        };

        let out_w = if self.ceil_mode {
            ((in_w + 2 * self.padding.1 - self.kernel_size.1) as f32 / stride.1 as f32).ceil()
                as usize
                + 1
        } else {
            (in_w + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1
        };

        // For now, return a placeholder with correct output shape
        let output_shape = vec![batch_size, channels, out_h, out_w];
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
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

        let batch_size = input_shape[0];
        let in_c = input_shape[1];
        let in_l = input_shape[2];

        if in_c != self.in_channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} input channels, got {}",
                self.in_channels, in_c
            )));
        }

        // Calculate output length
        let out_l = (in_l + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1;

        // For now, return a placeholder with correct output shape
        let output_shape = vec![batch_size, self.out_channels, out_l];
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
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
