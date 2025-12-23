use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

#[derive(Clone)]
pub struct MaxPool2D {
    #[allow(dead_code)]
    kernel_size: (usize, usize),
    #[allow(dead_code)]
    stride: (usize, usize),
    #[allow(dead_code)]
    padding: String,
}

impl MaxPool2D {
    pub fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: "valid".to_string(),
        }
    }
}

impl<T> Layer<T> for MaxPool2D
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        tenflowers_core::ops::max_pool2d(input, self.kernel_size, self.stride, &self.padding)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // Pooling layers don't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Adaptive max pooling 2D forward implementation
fn adaptive_max_pool2d_forward<T>(
    input: &Tensor<T>,
    output_size: (usize, usize),
    input_shape: (usize, usize, usize, usize), // (batch, channels, height, width)
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + PartialOrd + Send + Sync + 'static,
{
    let (batch_size, channels, input_height, input_width) = input_shape;
    let (output_height, output_width) = output_size;

    // Get input data
    let input_data = input.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "Cannot access input tensor data".to_string(),
        )
    })?;

    // Initialize output tensor
    let total_output_elements = batch_size * channels * output_height * output_width;
    let mut output_data = vec![T::zero(); total_output_elements];

    // Calculate adaptive pooling regions
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Calculate input region for this output pixel
                    let ih_start = (oh * input_height) / output_height;
                    let ih_end = ((oh + 1) * input_height + output_height - 1) / output_height;
                    let iw_start = (ow * input_width) / output_width;
                    let iw_end = ((ow + 1) * input_width + output_width - 1) / output_width;

                    // Find maximum in this region
                    let mut max_val = T::zero();
                    let mut first = true;

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            if ih < input_height && iw < input_width {
                                let input_idx = b * channels * input_height * input_width
                                    + c * input_height * input_width
                                    + ih * input_width
                                    + iw;

                                if first || input_data[input_idx] > max_val {
                                    max_val = input_data[input_idx].clone();
                                    first = false;
                                }
                            }
                        }
                    }

                    // Store result
                    let output_idx = b * channels * output_height * output_width
                        + c * output_height * output_width
                        + oh * output_width
                        + ow;
                    output_data[output_idx] = max_val;
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        &[batch_size, channels, output_height, output_width],
    )
}

/// Adaptive average pooling 2D forward implementation
fn adaptive_avg_pool2d_forward<T>(
    input: &Tensor<T>,
    output_size: (usize, usize),
    input_shape: (usize, usize, usize, usize), // (batch, channels, height, width)
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + FromPrimitive + Send + Sync + 'static,
{
    let (batch_size, channels, input_height, input_width) = input_shape;
    let (output_height, output_width) = output_size;

    // Get input data
    let input_data = input.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "Cannot access input tensor data".to_string(),
        )
    })?;

    // Initialize output tensor
    let total_output_elements = batch_size * channels * output_height * output_width;
    let mut output_data = vec![T::zero(); total_output_elements];

    // Calculate adaptive pooling regions
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Calculate input region for this output pixel
                    let ih_start = (oh * input_height) / output_height;
                    let ih_end = ((oh + 1) * input_height + output_height - 1) / output_height;
                    let iw_start = (ow * input_width) / output_width;
                    let iw_end = ((ow + 1) * input_width + output_width - 1) / output_width;

                    // Calculate average in this region
                    let mut sum = T::zero();
                    let mut count = 0;

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            if ih < input_height && iw < input_width {
                                let input_idx = b * channels * input_height * input_width
                                    + c * input_height * input_width
                                    + ih * input_width
                                    + iw;

                                sum = sum + input_data[input_idx];
                                count += 1;
                            }
                        }
                    }

                    // Calculate average
                    let avg = if count > 0 {
                        sum / T::from_usize(count).unwrap_or(T::one())
                    } else {
                        T::zero()
                    };

                    // Store result
                    let output_idx = b * channels * output_height * output_width
                        + c * output_height * output_width
                        + oh * output_width
                        + ow;
                    output_data[output_idx] = avg;
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        &[batch_size, channels, output_height, output_width],
    )
}

#[derive(Clone)]
pub struct AvgPool2D {
    #[allow(dead_code)]
    kernel_size: (usize, usize),
    #[allow(dead_code)]
    stride: (usize, usize),
    #[allow(dead_code)]
    padding: String,
}

impl AvgPool2D {
    pub fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: "valid".to_string(),
        }
    }
}

impl<T> Layer<T> for AvgPool2D
where
    T: Clone
        + Default
        + Zero
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        tenflowers_core::ops::avg_pool2d(input, self.kernel_size, self.stride, &self.padding)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // Pooling layers don't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct AdaptiveMaxPool2D {
    output_size: (usize, usize),
}

impl AdaptiveMaxPool2D {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T> Layer<T> for AdaptiveMaxPool2D
where
    T: Clone + Default + Zero + One + PartialOrd + Send + Sync + 'static,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let input_shape = input.shape().dims();
        if input_shape.len() != 4 {
            return Err(tenflowers_core::TensorError::invalid_shape(
                "AdaptiveMaxPool2D",
                "4D tensor [batch, channels, height, width]",
                &format!("{}D tensor", input_shape.len()),
            ));
        }

        let shape_tuple = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        adaptive_max_pool2d_forward(input, self.output_size, shape_tuple)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // Adaptive pooling layers don't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct AdaptiveAvgPool2D {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2D {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T> Layer<T> for AdaptiveAvgPool2D
where
    T: Clone + Default + Zero + One + Float + FromPrimitive + Send + Sync + 'static,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let input_shape = input.shape().dims();
        if input_shape.len() != 4 {
            return Err(tenflowers_core::TensorError::invalid_shape(
                "AdaptiveAvgPool2D",
                "4D tensor [batch, channels, height, width]",
                &format!("{}D tensor", input_shape.len()),
            ));
        }

        let shape_tuple = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        adaptive_avg_pool2d_forward(input, self.output_size, shape_tuple)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // Adaptive pooling layers don't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}
