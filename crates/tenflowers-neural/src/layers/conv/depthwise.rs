//! Depthwise convolution layer implementations
//!
//! This module contains the DepthwiseConv2D layer which implements mobile-optimized
//! depthwise separable convolutions. These layers apply a single filter per input channel,
//! followed by a pointwise convolution, making them highly efficient for mobile and embedded
//! applications.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Depthwise 2D Convolutional layer for mobile-optimized convolutions
/// Applies a single filter per input channel, followed by a pointwise convolution
pub struct DepthwiseConv2D<T> {
    depthwise_weight: Tensor<T>,
    pointwise_weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    stride: (usize, usize),
    padding: String,
    dilation: (usize, usize),
    depth_multiplier: usize,
    training: bool,
}

impl<T> DepthwiseConv2D<T>
where
    T: Clone + Default + Zero,
{
    /// Create a new DepthwiseConv2D layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `kernel_size` - Size of the depthwise convolution kernel (height, width)
    /// * `stride` - Stride of the convolution (default: (1, 1))
    /// * `padding` - Padding mode: "valid" or "same"
    /// * `dilation` - Dilation rate (default: (1, 1))
    /// * `depth_multiplier` - Multiplier for the number of depthwise channels (default: 1)
    /// * `use_bias` - Whether to include bias term
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<String>,
        dilation: Option<(usize, usize)>,
        depth_multiplier: Option<usize>,
        use_bias: bool,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or_else(|| "valid".to_string());
        let dilation = dilation.unwrap_or((1, 1));
        let depth_multiplier = depth_multiplier.unwrap_or(1);

        // Depthwise weight shape: [in_channels * depth_multiplier, 1, kernel_height, kernel_width]
        let depthwise_weight = Tensor::zeros(&[
            in_channels * depth_multiplier,
            1,
            kernel_size.0,
            kernel_size.1,
        ]);

        // Pointwise weight shape: [out_channels, in_channels * depth_multiplier, 1, 1]
        // For simplicity, we'll set out_channels = in_channels * depth_multiplier initially
        let out_channels = in_channels * depth_multiplier;
        let pointwise_weight = Tensor::zeros(&[out_channels, in_channels * depth_multiplier, 1, 1]);

        let bias = if use_bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Self {
            depthwise_weight,
            pointwise_weight,
            bias,
            stride,
            padding,
            dilation,
            depth_multiplier,
            training: false,
        }
    }

    /// Simplified constructor for common use cases
    pub fn simple(in_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self::new(in_channels, kernel_size, None, None, None, None, true)
    }

    /// Create a deep copy of this DepthwiseConv2D layer
    pub fn clone_layer(&self) -> Self
    where
        T: Clone,
    {
        Self {
            depthwise_weight: self.depthwise_weight.clone(),
            pointwise_weight: self.pointwise_weight.clone(),
            bias: self.bias.clone(),
            stride: self.stride,
            padding: self.padding.clone(),
            dilation: self.dilation,
            depth_multiplier: self.depth_multiplier,
            training: self.training,
        }
    }
}

impl<T> Clone for DepthwiseConv2D<T>
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

impl<T> Layer<T> for DepthwiseConv2D<T>
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
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // First apply depthwise convolution
        let depthwise_output = depthwise_conv2d_forward(
            input,
            &self.depthwise_weight,
            None, // No bias in depthwise step
            self.stride,
            &self.padding,
            self.dilation,
            self.depth_multiplier,
        )?;

        // Then apply pointwise convolution (1x1 conv)
        tenflowers_core::ops::conv2d(
            &depthwise_output,
            &self.pointwise_weight,
            self.bias.as_ref(),
            (1, 1),  // Always stride 1 for pointwise
            "valid", // Always valid padding for pointwise
        )
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.depthwise_weight, &self.pointwise_weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.depthwise_weight, &mut self.pointwise_weight];
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
}

/// Depthwise 2D convolution forward pass implementation
fn depthwise_conv2d_forward<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
    dilation: (usize, usize),
    depth_multiplier: usize,
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
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 4 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
            "DepthwiseConv2D expects 4D input [batch, channels, height, width], got {}D",
            input_shape.len()
        )));
    }

    if weight_shape.len() != 4 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            format!("DepthwiseConv2D weight expects 4D [out_channels, 1, kernel_height, kernel_width], got {}D", weight_shape.len())
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let expected_out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    // Validate that output channels = input channels * depth_multiplier
    if expected_out_channels != in_channels * depth_multiplier {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            format!("Expected {expected_out_channels} output channels, but got in_channels ({in_channels}) * depth_multiplier ({depth_multiplier}) = {}", in_channels * depth_multiplier)
        ));
    }

    // Calculate output dimensions
    let (output_height, output_width) = match padding {
        "valid" => {
            let h = if input_height < (kernel_height - 1) * dilation.0 + 1 {
                0
            } else {
                (input_height - (kernel_height - 1) * dilation.0 - 1) / stride.0 + 1
            };
            let w = if input_width < (kernel_width - 1) * dilation.1 + 1 {
                0
            } else {
                (input_width - (kernel_width - 1) * dilation.1 - 1) / stride.1 + 1
            };
            (h, w)
        }
        "same" => {
            let h = (input_height + stride.0 - 1) / stride.0;
            let w = (input_width + stride.1 - 1) / stride.1;
            (h, w)
        }
        _ => {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "Unsupported padding mode: {padding}"
            )));
        }
    };

    if output_height == 0 || output_width == 0 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Output dimensions would be zero with current parameters".to_string(),
        ));
    }

    let out_channels = in_channels * depth_multiplier;

    // Initialize output tensor
    let total_output_elements = batch_size * out_channels * output_height * output_width;
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

    // Perform depthwise convolution
    for b in 0..batch_size {
        for in_c in 0..in_channels {
            for depth_idx in 0..depth_multiplier {
                let out_c = in_c * depth_multiplier + depth_idx;

                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let mut sum = T::zero();

                        // Calculate input position based on output position and padding
                        let (input_start_h, input_start_w) = match padding {
                            "valid" => (out_h * stride.0, out_w * stride.1),
                            "same" => {
                                let pad_h = (kernel_height - 1) * dilation.0 / 2;
                                let pad_w = (kernel_width - 1) * dilation.1 / 2;
                                (
                                    (out_h * stride.0).saturating_sub(pad_h),
                                    (out_w * stride.1).saturating_sub(pad_w),
                                )
                            }
                            _ => unreachable!(),
                        };

                        // Convolve with kernel for this specific input channel
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let input_h = input_start_h + kh * dilation.0;
                                let input_w = input_start_w + kw * dilation.1;

                                // Check bounds
                                if input_h < input_height && input_w < input_width {
                                    let input_idx = b * in_channels * input_height * input_width
                                        + in_c * input_height * input_width
                                        + input_h * input_width
                                        + input_w;

                                    let weight_idx = out_c * kernel_height * kernel_width
                                        + kh * kernel_width
                                        + kw;

                                    sum = sum + input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias_tensor) = bias {
                            if let Some(bias_data) = bias_tensor.as_slice() {
                                sum = sum + bias_data[out_c];
                            }
                        }

                        // Store result
                        let output_idx = b * out_channels * output_height * output_width
                            + out_c * output_height * output_width
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
        &[batch_size, out_channels, output_height, output_width],
    )
}
