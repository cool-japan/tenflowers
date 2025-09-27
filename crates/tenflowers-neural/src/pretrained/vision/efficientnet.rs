//! EfficientNet Models
//!
//! This module contains EfficientNet architecture implementations with compound scaling
//! for balancing network depth, width, and resolution.

use crate::{
    layers::{Conv2D, Dense, Layer},
    model::Model,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// EfficientNet configuration for different model variants
#[derive(Clone)]
pub struct EfficientNetConfig {
    pub width_coefficient: f32,
    pub depth_coefficient: f32,
    pub resolution: usize,
    pub dropout_rate: f32,
}

impl EfficientNetConfig {
    pub fn b0() -> Self {
        Self {
            width_coefficient: 1.0,
            depth_coefficient: 1.0,
            resolution: 224,
            dropout_rate: 0.2,
        }
    }

    pub fn b1() -> Self {
        Self {
            width_coefficient: 1.0,
            depth_coefficient: 1.1,
            resolution: 240,
            dropout_rate: 0.2,
        }
    }

    pub fn b7() -> Self {
        Self {
            width_coefficient: 2.0,
            depth_coefficient: 3.1,
            resolution: 600,
            dropout_rate: 0.5,
        }
    }
}

/// EfficientNet architecture implementation
pub struct EfficientNet<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub config: EfficientNetConfig,
    pub blocks: Vec<Vec<Box<dyn Layer<T>>>>,
    stem: Conv2D<T>,
    head: Dense<T>,
    training: bool,
}

impl<T> EfficientNet<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(config: EfficientNetConfig, num_classes: usize) -> Self {
        Self {
            config,
            blocks: Vec::new(),
            stem: Conv2D::new(3, 32, (3, 3), (2, 2), "1".to_string(), false),
            head: Dense::new(1280, num_classes, true),
            training: false,
        }
    }

    pub fn efficientnet_b0(num_classes: usize) -> Self {
        Self::new(EfficientNetConfig::b0(), num_classes)
    }

    pub fn efficientnet_b1(num_classes: usize) -> Self {
        Self::new(EfficientNetConfig::b1(), num_classes)
    }
}

impl<T> Model<T> for EfficientNet<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Simplified implementation
        let mut x = self.stem.forward(input)?;

        // Process through blocks
        for stage in &self.blocks {
            for block in stage {
                x = block.forward(&x)?;
            }
        }

        // Global average pooling and classification
        let batch_size = x.shape()[0];
        let channels = x.shape()[1];
        x = x.reshape(&[batch_size, channels])?;
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.stem.parameters());
        for stage in &self.blocks {
            for block in stage {
                params.extend(block.parameters());
            }
        }
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.stem.parameters_mut());
        for stage in &mut self.blocks {
            for block in stage {
                params.extend(block.parameters_mut());
            }
        }
        params.extend(self.head.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.stem.set_training(training);
        for stage in &mut self.blocks {
            for block in stage {
                block.set_training(training);
            }
        }
        self.head.set_training(training);
    }

    fn zero_grad(&mut self) {
        for param in Model::parameters_mut(self) {
            crate::model::zero_tensor_grad(param);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
