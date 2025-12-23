//! 1D Convolutional Layer Implementation
//!
//! This module contains the implementation of the Conv1D layer for sequence processing.
//! The Conv1D layer performs 1D convolution operations on input sequences, commonly used
//! for time series data, text processing, and other sequential data applications.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// 1D Convolutional layer for sequence processing
pub struct Conv1D<T> {
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    stride: usize,
    padding: String,
    dilation: usize,
    groups: usize,
    training: bool,
}

impl<T> Conv1D<T>
where
    T: Clone + Default + Zero,
{
    /// Create a new Conv1D layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Padding mode: "valid", "same", or "causal"
    /// * `dilation` - Dilation rate (default: 1)
    /// * `groups` - Number of groups for grouped convolution (default: 1)
    /// * `use_bias` - Whether to include bias term
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<String>,
        dilation: Option<usize>,
        groups: Option<usize>,
        use_bias: bool,
    ) -> Self {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or_else(|| "valid".to_string());
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);

        // Weight shape: [out_channels, in_channels / groups, kernel_size]
        let weight = Tensor::zeros(&[out_channels, in_channels / groups, kernel_size]);
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
    pub fn simple(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
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

    /// Get a reference to the weight tensor
    pub fn weight(&self) -> &Tensor<T> {
        &self.weight
    }

    /// Get a reference to the bias tensor (if any)
    pub fn bias(&self) -> Option<&Tensor<T>> {
        self.bias.as_ref()
    }

    /// Set the weight tensor
    pub fn set_weight(&mut self, weight: Tensor<T>) {
        self.weight = weight;
    }

    /// Set the bias tensor
    pub fn set_bias(&mut self, bias: Option<Tensor<T>>) {
        self.bias = bias;
    }

    /// Get the stride
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get the padding
    pub fn padding(&self) -> &str {
        &self.padding
    }

    /// Get the dilation
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// Get the groups
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Check if the layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Create a deep copy of this Conv1D layer
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

impl<T> Layer<T> for Conv1D<T>
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
        + scirs2_core::num_traits::Float,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // For now, implement a simplified 1D convolution
        // Input shape: [batch_size, in_channels, sequence_length]
        // Weight shape: [out_channels, in_channels / groups, kernel_size]
        // Output shape: [batch_size, out_channels, output_length]

        conv1d_forward(
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
        crate::layers::LayerType::Conv1D
    }

    fn set_weight(&mut self, weight: Tensor<T>) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    fn set_bias(&mut self, bias: Option<Tensor<T>>) -> Result<()> {
        self.bias = bias;
        Ok(())
    }
}

impl<T> Clone for Conv1D<T>
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

/// 1D convolution forward pass implementation
fn conv1d_forward<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: &str,
    dilation: usize,
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
        + scirs2_core::num_traits::Float,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if input_shape.len() != 3 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
            "Conv1D expects 3D input [batch, channels, length], got {}D",
            input_shape.len()
        )));
    }

    if weight_shape.len() != 3 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
            "Conv1D weight expects 3D [out_channels, in_channels/groups, kernel_size], got {}D",
            weight_shape.len()
        )));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let out_channels = weight_shape[0];
    let in_channels_per_group = weight_shape[1];
    let kernel_size = weight_shape[2];

    // Validate channel dimensions
    if in_channels != in_channels_per_group * groups {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            format!("Input channels ({in_channels}) must equal in_channels_per_group ({in_channels_per_group}) * groups ({groups})")
        ));
    }

    // Calculate output length based on padding
    let output_length = match padding {
        "valid" => {
            if input_length < (kernel_size - 1) * dilation + 1 {
                0
            } else {
                (input_length - (kernel_size - 1) * dilation - 1) / stride + 1
            }
        }
        "same" => (input_length + stride - 1) / stride,
        "causal" => input_length,
        _ => {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "Unsupported padding mode: {padding}"
            )));
        }
    };

    if output_length == 0 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Output length would be zero with current parameters".to_string(),
        ));
    }

    // Initialize output tensor
    let total_output_elements = batch_size * out_channels * output_length;
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
            for out_pos in 0..output_length {
                let mut sum = T::zero();

                // Calculate input position based on output position and padding
                let input_start = match padding {
                    "valid" => out_pos * stride,
                    "same" => {
                        let pad_total = (kernel_size - 1) * dilation;
                        let pad_left = pad_total / 2;
                        (out_pos * stride).saturating_sub(pad_left)
                    }
                    "causal" => {
                        let pad_left = (kernel_size - 1) * dilation;
                        (out_pos * stride).saturating_sub(pad_left)
                    }
                    _ => unreachable!(),
                };

                // Determine which group this output channel belongs to
                let group = oc / (out_channels / groups);
                let group_start_ic = group * in_channels_per_group;
                let group_end_ic = group_start_ic + in_channels_per_group;

                // Convolve with kernel
                for ic in group_start_ic..group_end_ic {
                    for k in 0..kernel_size {
                        let input_pos = input_start + k * dilation;

                        // Check bounds
                        if input_pos < input_length {
                            let input_idx =
                                b * in_channels * input_length + ic * input_length + input_pos;
                            let weight_idx = oc * in_channels_per_group * kernel_size
                                + (ic - group_start_ic) * kernel_size
                                + k;

                            sum = sum + input_data[input_idx] * weight_data[weight_idx];
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
                let output_idx = b * out_channels * output_length + oc * output_length + out_pos;
                output_data[output_idx] = sum;
            }
        }
    }

    Tensor::from_vec(output_data, &[batch_size, out_channels, output_length])
}
