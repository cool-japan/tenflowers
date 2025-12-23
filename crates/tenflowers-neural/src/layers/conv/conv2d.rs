//! 2D Convolution Layer Implementation
//!
//! This module contains the Conv2D layer implementation for 2D convolution operations,
//! commonly used in computer vision and image processing tasks.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// 2D Convolutional layer for image and spatial data processing
///
/// This layer performs 2D convolution operations on input tensors, applying learnable
/// filters across spatial dimensions. It supports configurable stride, padding, and
/// optional bias terms.
pub struct Conv2D<T>
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
        + Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    stride: (usize, usize),
    padding: String,
    training: bool,
}

impl<T> Conv2D<T>
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
        + Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Creates a new Conv2D layer with the specified parameters
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels (filters)
    /// * `kernel_size` - Size of the convolution kernel (height, width)
    /// * `stride` - Stride for the convolution (height, width)
    /// * `padding` - Padding strategy ("valid" or "same")
    /// * `use_bias` - Whether to include bias terms
    ///
    /// # Returns
    ///
    /// A new Conv2D layer instance
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: String,
        use_bias: bool,
    ) -> Self {
        let weight = Tensor::zeros(&[out_channels, in_channels, kernel_size.0, kernel_size.1]);
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
            training: false,
        }
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
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Get the padding
    pub fn padding(&self) -> &str {
        &self.padding
    }

    /// Check if the layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Create a deep copy of this Conv2D layer
    pub fn clone_layer(&self) -> Self
    where
        T: Clone,
    {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            stride: self.stride,
            padding: self.padding.clone(),
            training: self.training,
        }
    }
}

impl<T> Clone for Conv2D<T>
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
        + Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn clone(&self) -> Self {
        self.clone_layer()
    }
}

impl<T> Layer<T> for Conv2D<T>
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
        + Float
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        tenflowers_core::ops::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            &self.padding,
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
        crate::layers::LayerType::Conv2D
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
