use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, Zero};
use tenflowers_core::{Result, Tensor};

/// Fractional Max Pooling 2D - stochastic or deterministic fractional pooling
#[derive(Clone)]
pub struct FractionalMaxPool2D {
    pooling_ratio: (f32, f32),
    deterministic: bool,
    random_samples: Option<Tensor<f32>>,
}

impl FractionalMaxPool2D {
    pub fn new(pooling_ratio: (f32, f32)) -> Self {
        Self {
            pooling_ratio,
            deterministic: false,
            random_samples: None,
        }
    }

    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    pub fn with_random_samples(mut self, samples: Tensor<f32>) -> Self {
        self.random_samples = Some(samples);
        self.deterministic = true;
        self
    }
}

impl<T> Layer<T> for FractionalMaxPool2D
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let samples = if self.deterministic {
            self.random_samples.as_ref().map(|s| {
                // Convert f32 samples to T
                let f32_data = s.as_slice().unwrap();
                let converted_data: Vec<T> = f32_data
                    .iter()
                    .map(|&x| T::from_f32(x).unwrap_or(T::zero()))
                    .collect();
                Tensor::from_vec(converted_data, s.shape().dims()).unwrap()
            })
        } else {
            None
        };

        tenflowers_core::ops::fractional_max_pool2d(input, self.pooling_ratio, samples.as_ref())
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

/// Fractional Average Pooling 2D - stochastic or deterministic fractional pooling
#[derive(Clone)]
pub struct FractionalAvgPool2D {
    pooling_ratio: (f32, f32),
    deterministic: bool,
    random_samples: Option<Tensor<f32>>,
}

impl FractionalAvgPool2D {
    pub fn new(pooling_ratio: (f32, f32)) -> Self {
        Self {
            pooling_ratio,
            deterministic: false,
            random_samples: None,
        }
    }

    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    pub fn with_random_samples(mut self, samples: Tensor<f32>) -> Self {
        self.random_samples = Some(samples);
        self.deterministic = true;
        self
    }
}

impl<T> Layer<T> for FractionalAvgPool2D
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
        let samples = if self.deterministic {
            self.random_samples.as_ref().map(|s| {
                // Convert f32 samples to T
                let f32_data = s.as_slice().unwrap();
                let converted_data: Vec<T> = f32_data
                    .iter()
                    .map(|&x| T::from_f32(x).unwrap_or(T::zero()))
                    .collect();
                Tensor::from_vec(converted_data, s.shape().dims()).unwrap()
            })
        } else {
            None
        };

        tenflowers_core::ops::fractional_avg_pool2d(input, self.pooling_ratio, samples.as_ref())
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
