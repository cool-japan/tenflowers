//! 3D Convolution Layer Implementation
//!
//! This module contains the Conv3D layer implementation for 3D convolution operations,
//! typically used for video/volumetric data processing in neural networks.

use crate::layers::Layer;
use num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// 3D Convolutional layer for video/volumetric data processing
pub struct Conv3D<T> {
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    stride: (usize, usize, usize),
    padding: String,
    dilation: (usize, usize, usize),
    groups: usize,
    training: bool,
}

impl<T> Conv3D<T>
where
    T: Clone + Default + Zero,
{
    /// Create a new Conv3D layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel (depth, height, width)
    /// * `stride` - Stride of the convolution (default: (1, 1, 1))
    /// * `padding` - Padding mode: "valid" or "same"
    /// * `dilation` - Dilation rate (default: (1, 1, 1))
    /// * `groups` - Number of groups for grouped convolution (default: 1)
    /// * `use_bias` - Whether to include bias term
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: Option<String>,
        dilation: Option<(usize, usize, usize)>,
        groups: Option<usize>,
        use_bias: bool,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1, 1));
        let padding = padding.unwrap_or_else(|| "valid".to_string());
        let dilation = dilation.unwrap_or((1, 1, 1));
        let groups = groups.unwrap_or(1);

        // Weight shape: [out_channels, in_channels / groups, kernel_depth, kernel_height, kernel_width]
        let weight = Tensor::zeros(&[
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            training: false,
        }
    }

    /// Simplified constructor for common use cases
    pub fn simple(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            None,
            None,
            None,
            None,
            true,
        )
    }

    /// Create a deep copy of this Conv3D layer
    pub fn clone_layer(&self) -> Self
    where
        T: Clone,
    {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            stride: self.stride,
            padding: self.padding.clone(),
            dilation: self.dilation,
            groups: self.groups,
            training: self.training,
        }
    }
}

impl<T> Clone for Conv3D<T>
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
        + Float,
{
    fn clone(&self) -> Self {
        self.clone_layer()
    }
}

impl<T> Layer<T> for Conv3D<T>
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
        + num_traits::Float,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Input shape: [batch_size, in_channels, depth, height, width]
        // Weight shape: [out_channels, in_channels / groups, kernel_depth, kernel_height, kernel_width]
        // Output shape: [batch_size, out_channels, output_depth, output_height, output_width]

        conv3d_forward(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            &self.padding,
            self.dilation,
            self.groups,
        )
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new((*self).clone())
    }

    fn layer_type(&self) -> crate::layers::LayerType {
        crate::layers::LayerType::Conv3D
    }
}

/// 3D convolution forward pass implementation
fn conv3d_forward<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
    padding: &str,
    dilation: (usize, usize, usize),
    groups: usize,
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
        + num_traits::Float,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 5 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
            "Conv3D expects 5D input [batch, channels, depth, height, width], got {}D",
            input_shape.len()
        )));
    }

    if weight_shape.len() != 5 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            format!("Conv3D weight expects 5D [out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width], got {}D", weight_shape.len())
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    let out_channels = weight_shape[0];
    let in_channels_per_group = weight_shape[1];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    // Validate channel dimensions
    if in_channels != in_channels_per_group * groups {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            format!("Input channels ({in_channels}) must equal in_channels_per_group ({in_channels_per_group}) * groups ({groups})")
        ));
    }

    // Calculate output dimensions based on padding
    let (output_depth, output_height, output_width) = match padding {
        "valid" => {
            let d = if input_depth < (kernel_depth - 1) * dilation.0 + 1 {
                0
            } else {
                (input_depth - (kernel_depth - 1) * dilation.0 - 1) / stride.0 + 1
            };
            let h = if input_height < (kernel_height - 1) * dilation.1 + 1 {
                0
            } else {
                (input_height - (kernel_height - 1) * dilation.1 - 1) / stride.1 + 1
            };
            let w = if input_width < (kernel_width - 1) * dilation.2 + 1 {
                0
            } else {
                (input_width - (kernel_width - 1) * dilation.2 - 1) / stride.2 + 1
            };
            (d, h, w)
        }
        "same" => {
            let d = (input_depth + stride.0 - 1) / stride.0;
            let h = (input_height + stride.1 - 1) / stride.1;
            let w = (input_width + stride.2 - 1) / stride.2;
            (d, h, w)
        }
        _ => {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "Unsupported padding mode: {padding}"
            )));
        }
    };

    if output_depth == 0 || output_height == 0 || output_width == 0 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Output dimensions would be zero with current parameters".to_string(),
        ));
    }

    // Initialize output tensor
    let total_output_elements =
        batch_size * out_channels * output_depth * output_height * output_width;
    let mut output_data = vec![T::zero(); total_output_elements];

    // Get input data
    let input_data = input.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "Cannot access input tensor data".to_string(),
        )
    })?;

    let weight_data = weight.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "Cannot access weight tensor data".to_string(),
        )
    })?;

    // Perform convolution for each batch and output position
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for out_d in 0..output_depth {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let mut sum = T::zero();

                        // Calculate input position based on output position and padding
                        let (input_start_d, input_start_h, input_start_w) = match padding {
                            "valid" => (out_d * stride.0, out_h * stride.1, out_w * stride.2),
                            "same" => {
                                let pad_d = (kernel_depth - 1) * dilation.0 / 2;
                                let pad_h = (kernel_height - 1) * dilation.1 / 2;
                                let pad_w = (kernel_width - 1) * dilation.2 / 2;
                                (
                                    (out_d * stride.0).saturating_sub(pad_d),
                                    (out_h * stride.1).saturating_sub(pad_h),
                                    (out_w * stride.2).saturating_sub(pad_w),
                                )
                            }
                            _ => unreachable!(),
                        };

                        // Determine which group this output channel belongs to
                        let group = oc / (out_channels / groups);
                        let group_start_ic = group * in_channels_per_group;
                        let group_end_ic = group_start_ic + in_channels_per_group;

                        // Convolve with kernel
                        for ic in group_start_ic..group_end_ic {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let input_d = input_start_d + kd * dilation.0;
                                        let input_h = input_start_h + kh * dilation.1;
                                        let input_w = input_start_w + kw * dilation.2;

                                        // Check bounds
                                        if input_d < input_depth
                                            && input_h < input_height
                                            && input_w < input_width
                                        {
                                            let input_idx = b
                                                * in_channels
                                                * input_depth
                                                * input_height
                                                * input_width
                                                + ic * input_depth * input_height * input_width
                                                + input_d * input_height * input_width
                                                + input_h * input_width
                                                + input_w;

                                            let weight_idx = oc
                                                * in_channels_per_group
                                                * kernel_depth
                                                * kernel_height
                                                * kernel_width
                                                + (ic - group_start_ic)
                                                    * kernel_depth
                                                    * kernel_height
                                                    * kernel_width
                                                + kd * kernel_height * kernel_width
                                                + kh * kernel_width
                                                + kw;

                                            sum = sum
                                                + input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias_tensor) = bias {
                            if let Some(bias_data) = bias_tensor.as_slice() {
                                sum = sum + bias_data[oc];
                            }
                        }

                        // Store result
                        let output_idx =
                            b * out_channels * output_depth * output_height * output_width
                                + oc * output_depth * output_height * output_width
                                + out_d * output_height * output_width
                                + out_h * output_width
                                + out_w;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[
            batch_size,
            out_channels,
            output_depth,
            output_height,
            output_width,
        ],
    )
}
