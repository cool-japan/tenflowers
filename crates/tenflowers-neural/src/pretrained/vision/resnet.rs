//! ResNet (Residual Network) Models
//!
//! This module contains ResNet architecture implementations including ResNet-18, ResNet-34,
//! ResNet-50, ResNet-101, and ResNet-152 variants using both Basic and Bottleneck blocks.

use crate::{
    layers::{
        pooling::{AdaptiveAvgPool2D, MaxPool2D},
        BatchNorm, Conv2D, Dense, Layer,
    },
    model::{Model, Sequential},
    pretrained::common::{BasicBlock, BottleneckBlock, ReLU},
};
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// ResNet block type configuration
#[derive(Copy, Clone)]
pub enum ResNetBlockType {
    Basic,
    Bottleneck,
}

/// ResNet architecture implementation
pub struct ResNet<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Initial conv layer
    conv1: Conv2D<T>,
    bn1: BatchNorm<T>,
    relu: ReLU<T>,
    maxpool: MaxPool2D,

    // ResNet layers
    layer1: Vec<Box<dyn Layer<T>>>,
    layer2: Vec<Box<dyn Layer<T>>>,
    layer3: Vec<Box<dyn Layer<T>>>,
    layer4: Vec<Box<dyn Layer<T>>>,

    // Final layers
    avgpool: AdaptiveAvgPool2D,
    fc: Dense<T>,

    training: bool,
}

impl<T> ResNet<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new ResNet with specified configuration
    pub fn new(block_type: ResNetBlockType, layers: [usize; 4], num_classes: usize) -> Self {
        let mut resnet = Self {
            // Initial 7x7 conv with stride 2
            conv1: Conv2D::new(3, 64, (7, 7), (2, 2), "3".to_string(), false),
            bn1: BatchNorm::new(64),
            relu: ReLU::new(),
            maxpool: MaxPool2D::new((3, 3), Some((2, 2))),

            layer1: Vec::new(),
            layer2: Vec::new(),
            layer3: Vec::new(),
            layer4: Vec::new(),

            avgpool: AdaptiveAvgPool2D::new((1, 1)),
            fc: Dense::new(
                match block_type {
                    ResNetBlockType::Basic => 512,
                    ResNetBlockType::Bottleneck => 2048,
                },
                num_classes,
                true, // use_bias
            ),
            training: false,
        };

        // Build ResNet layers
        resnet.layer1 = Self::make_layer(block_type, 64, 64, layers[0], 1);
        resnet.layer2 = Self::make_layer(block_type, 64, 128, layers[1], 2);
        resnet.layer3 = Self::make_layer(block_type, 128, 256, layers[2], 2);
        resnet.layer4 = Self::make_layer(block_type, 256, 512, layers[3], 2);

        resnet
    }

    /// Create a ResNet layer with specified blocks
    fn make_layer(
        block_type: ResNetBlockType,
        in_channels: usize,
        out_channels: usize,
        blocks: usize,
        stride: usize,
    ) -> Vec<Box<dyn Layer<T>>> {
        let mut layers = Vec::new();

        // Determine if we need downsampling for the first block
        let downsample = if stride != 1 || in_channels != out_channels {
            let mut downsample_layers = Vec::new();
            downsample_layers.push(Box::new(Conv2D::new(
                in_channels,
                out_channels,
                (1, 1),
                (stride, stride),
                "0".to_string(),
                false,
            )) as Box<dyn Layer<T>>);
            downsample_layers.push(Box::new(BatchNorm::new(out_channels)) as Box<dyn Layer<T>>);
            Some(Sequential::new(downsample_layers))
        } else {
            None
        };

        // First block with potential downsampling
        match block_type {
            ResNetBlockType::Basic => {
                layers.push(Box::new(BasicBlock::new(
                    in_channels,
                    out_channels,
                    stride,
                    downsample,
                )) as Box<dyn Layer<T>>);
            }
            ResNetBlockType::Bottleneck => {
                layers.push(Box::new(BottleneckBlock::new(
                    in_channels,
                    out_channels * 4, // Bottleneck expansion
                    stride,
                    downsample,
                )) as Box<dyn Layer<T>>);
            }
        }

        // Remaining blocks
        for _ in 1..blocks {
            match block_type {
                ResNetBlockType::Basic => {
                    layers.push(
                        Box::new(BasicBlock::new(out_channels, out_channels, 1, None))
                            as Box<dyn Layer<T>>,
                    );
                }
                ResNetBlockType::Bottleneck => {
                    layers.push(Box::new(BottleneckBlock::new(
                        out_channels * 4,
                        out_channels * 4,
                        1,
                        None,
                    )) as Box<dyn Layer<T>>);
                }
            }
        }

        layers
    }

    /// ResNet-18
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(ResNetBlockType::Basic, [2, 2, 2, 2], num_classes)
    }

    /// ResNet-34
    pub fn resnet34(num_classes: usize) -> Self {
        Self::new(ResNetBlockType::Basic, [3, 4, 6, 3], num_classes)
    }

    /// ResNet-50
    pub fn resnet50(num_classes: usize) -> Self {
        Self::new(ResNetBlockType::Bottleneck, [3, 4, 6, 3], num_classes)
    }

    /// ResNet-101
    pub fn resnet101(num_classes: usize) -> Self {
        Self::new(ResNetBlockType::Bottleneck, [3, 4, 23, 3], num_classes)
    }

    /// ResNet-152
    pub fn resnet152(num_classes: usize) -> Self {
        Self::new(ResNetBlockType::Bottleneck, [3, 8, 36, 3], num_classes)
    }
}

impl<T> Model<T> for ResNet<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + std::cmp::PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Initial conv block
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = self.relu.forward(&x)?;
        x = self.maxpool.forward(&x)?;

        // ResNet layers
        for layer in &self.layer1 {
            x = layer.forward(&x)?;
        }
        for layer in &self.layer2 {
            x = layer.forward(&x)?;
        }
        for layer in &self.layer3 {
            x = layer.forward(&x)?;
        }
        for layer in &self.layer4 {
            x = layer.forward(&x)?;
        }

        // Final layers
        x = self.avgpool.forward(&x)?;

        // Flatten for fully connected layer
        let batch_size = x.shape()[0];
        let flattened_size = x.shape().dims().iter().skip(1).product();
        x = x.reshape(&[batch_size, flattened_size])?;

        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();

        // Initial layers
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());

        // ResNet layers
        for layer in &self.layer1 {
            params.extend(layer.parameters());
        }
        for layer in &self.layer2 {
            params.extend(layer.parameters());
        }
        for layer in &self.layer3 {
            params.extend(layer.parameters());
        }
        for layer in &self.layer4 {
            params.extend(layer.parameters());
        }

        // Final layer
        params.extend(self.fc.parameters());

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();

        // Initial layers
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());

        // ResNet layers
        for layer in &mut self.layer1 {
            params.extend(layer.parameters_mut());
        }
        for layer in &mut self.layer2 {
            params.extend(layer.parameters_mut());
        }
        for layer in &mut self.layer3 {
            params.extend(layer.parameters_mut());
        }
        for layer in &mut self.layer4 {
            params.extend(layer.parameters_mut());
        }

        // Final layer
        params.extend(self.fc.parameters_mut());

        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;

        // Initial layers
        self.conv1.set_training(training);
        self.bn1.set_training(training);

        // ResNet layers
        for layer in &mut self.layer1 {
            layer.set_training(training);
        }
        for layer in &mut self.layer2 {
            layer.set_training(training);
        }
        for layer in &mut self.layer3 {
            layer.set_training(training);
        }
        for layer in &mut self.layer4 {
            layer.set_training(training);
        }

        // Final layer
        self.fc.set_training(training);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;

    #[test]
    fn test_resnet18_creation() {
        let resnet = ResNet::<f32>::resnet18(1000);

        // Test that the model was created successfully
        assert_eq!(resnet.layer1.len(), 2);
        assert_eq!(resnet.layer2.len(), 2);
        assert_eq!(resnet.layer3.len(), 2);
        assert_eq!(resnet.layer4.len(), 2);
    }

    #[test]
    fn test_resnet50_creation() {
        let resnet = ResNet::<f32>::resnet50(1000);

        // Test that the model was created successfully
        assert_eq!(resnet.layer1.len(), 3);
        assert_eq!(resnet.layer2.len(), 4);
        assert_eq!(resnet.layer3.len(), 6);
        assert_eq!(resnet.layer4.len(), 3);
    }
}
