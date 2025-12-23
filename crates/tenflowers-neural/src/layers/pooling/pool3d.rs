use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, Zero};
use tenflowers_core::{Result, Tensor};

/// 3D Max Pooling Layer
#[derive(Clone)]
pub struct MaxPool3D {
    #[allow(dead_code)]
    kernel_size: (usize, usize, usize),
    #[allow(dead_code)]
    stride: (usize, usize, usize),
    #[allow(dead_code)]
    padding: String,
}

impl MaxPool3D {
    pub fn new(kernel_size: (usize, usize, usize), stride: Option<(usize, usize, usize)>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: "valid".to_string(),
        }
    }

    pub fn with_padding(mut self, padding: &str) -> Self {
        self.padding = padding.to_string();
        self
    }
}

impl<T> Layer<T> for MaxPool3D
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
        tenflowers_core::ops::max_pool3d(input, self.kernel_size, self.stride, &self.padding)
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

/// 3D Average Pooling Layer
#[derive(Clone)]
pub struct AvgPool3D {
    #[allow(dead_code)]
    kernel_size: (usize, usize, usize),
    #[allow(dead_code)]
    stride: (usize, usize, usize),
    #[allow(dead_code)]
    padding: String,
}

impl AvgPool3D {
    pub fn new(kernel_size: (usize, usize, usize), stride: Option<(usize, usize, usize)>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: "valid".to_string(),
        }
    }

    pub fn with_padding(mut self, padding: &str) -> Self {
        self.padding = padding.to_string();
        self
    }
}

impl<T> Layer<T> for AvgPool3D
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
        tenflowers_core::ops::avg_pool3d(input, self.kernel_size, self.stride, &self.padding)
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
