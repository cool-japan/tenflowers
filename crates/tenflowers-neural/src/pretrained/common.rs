//! Common Building Blocks for Pretrained Models
//!
//! This module contains shared components and building blocks that are used
//! across multiple pretrained model architectures.

use crate::{
    layers::{pooling::GlobalAvgPool2D, BatchNorm, Conv2D, Dense, DepthwiseConv2D, Layer},
    model::{Model, Sequential},
};
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{
    ops::{activation::swish, relu, sigmoid},
    Result, Tensor,
};

/// ReLU activation function
#[derive(Clone)]
pub struct ReLU<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ReLU<T>
where
    T: scirs2_core::num_traits::FromPrimitive + bytemuck::Pod + bytemuck::Zeroable,
{
    pub fn new() -> Self {
        Self {
            training: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Default for ReLU<T>
where
    T: scirs2_core::num_traits::FromPrimitive + bytemuck::Pod + bytemuck::Zeroable,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Layer<T> for ReLU<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::cmp::PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        relu(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// ResNet Basic Block used in ResNet18 and ResNet34
pub struct BasicBlock<T>
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
    conv1: Conv2D<T>,
    bn1: BatchNorm<T>,
    relu: ReLU<T>,
    conv2: Conv2D<T>,
    bn2: BatchNorm<T>,
    downsample: Option<Sequential<T>>,
    stride: usize,
    training: bool,
}

impl<
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    > Clone for BasicBlock<T>
{
    fn clone(&self) -> Self {
        Self {
            conv1: self.conv1.clone(),
            bn1: self.bn1.clone(),
            relu: self.relu.clone(),
            conv2: self.conv2.clone(),
            bn2: self.bn2.clone(),
            downsample: self.downsample.clone(),
            stride: self.stride,
            training: self.training,
        }
    }
}

impl<T> BasicBlock<T>
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
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: Option<Sequential<T>>,
    ) -> Self {
        Self {
            conv1: Conv2D::new(
                in_channels,
                out_channels,
                (3, 3),
                (stride, stride),
                "1".to_string(),
                false,
            ),
            bn1: BatchNorm::new(out_channels),
            relu: ReLU::new(),
            conv2: Conv2D::new(
                out_channels,
                out_channels,
                (3, 3),
                (1, 1),
                "1".to_string(),
                false,
            ),
            bn2: BatchNorm::new(out_channels),
            downsample,
            stride,
            training: false,
        }
    }

    pub fn expansion() -> usize {
        1
    }
}

impl<T> Layer<T> for BasicBlock<T>
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
        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(input)?
        } else {
            input.clone()
        };

        out = out.add(&identity)?;
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        if let Some(ref mut downsample) = self.downsample {
            params.extend(downsample.parameters_mut());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.conv1.set_training(training);
        self.bn1.set_training(training);
        self.relu.set_training(training);
        self.conv2.set_training(training);
        self.bn2.set_training(training);
        if let Some(ref mut downsample) = self.downsample {
            downsample.set_training(training);
        }
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// ResNet Bottleneck Block used in ResNet50, ResNet101, and ResNet152
pub struct BottleneckBlock<T>
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
    conv1: Conv2D<T>,
    bn1: BatchNorm<T>,
    conv2: Conv2D<T>,
    bn2: BatchNorm<T>,
    conv3: Conv2D<T>,
    bn3: BatchNorm<T>,
    relu: ReLU<T>,
    downsample: Option<Sequential<T>>,
    stride: usize,
    training: bool,
}

impl<
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    > Clone for BottleneckBlock<T>
{
    fn clone(&self) -> Self {
        Self {
            conv1: self.conv1.clone(),
            bn1: self.bn1.clone(),
            conv2: self.conv2.clone(),
            bn2: self.bn2.clone(),
            conv3: self.conv3.clone(),
            bn3: self.bn3.clone(),
            relu: self.relu.clone(),
            downsample: self.downsample.clone(),
            stride: self.stride,
            training: self.training,
        }
    }
}

impl<T> BottleneckBlock<T>
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
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: Option<Sequential<T>>,
    ) -> Self {
        let bottleneck_channels = out_channels / 4;

        Self {
            conv1: Conv2D::new(
                in_channels,
                bottleneck_channels,
                (1, 1),
                (1, 1),
                "0".to_string(),
                false,
            ),
            bn1: BatchNorm::new(bottleneck_channels),
            conv2: Conv2D::new(
                bottleneck_channels,
                bottleneck_channels,
                (3, 3),
                (stride, stride),
                "1".to_string(),
                false,
            ),
            bn2: BatchNorm::new(bottleneck_channels),
            conv3: Conv2D::new(
                bottleneck_channels,
                out_channels,
                (1, 1),
                (1, 1),
                "0".to_string(),
                false,
            ),
            bn3: BatchNorm::new(out_channels),
            relu: ReLU::new(),
            downsample,
            stride,
            training: false,
        }
    }

    pub fn expansion() -> usize {
        4
    }
}

impl<T> Layer<T> for BottleneckBlock<T>
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
        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv3.forward(&out)?;
        out = self.bn3.forward(&out)?;

        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(input)?
        } else {
            input.clone()
        };

        out = out.add(&identity)?;
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        params.extend(self.conv3.parameters_mut());
        params.extend(self.bn3.parameters_mut());
        if let Some(ref mut downsample) = self.downsample {
            params.extend(downsample.parameters_mut());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.conv1.set_training(training);
        self.bn1.set_training(training);
        self.conv2.set_training(training);
        self.bn2.set_training(training);
        self.conv3.set_training(training);
        self.bn3.set_training(training);
        if let Some(ref mut downsample) = self.downsample {
            downsample.set_training(training);
        }
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Squeeze-and-Excitation Block used in EfficientNet and other architectures
pub struct SEBlock<T>
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
    avg_pool: GlobalAvgPool2D<T>,
    squeeze: Dense<T>,
    relu: ReLU<T>,
    excite: Dense<T>,
    sigmoid: std::marker::PhantomData<T>,
    training: bool,
}

impl<
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
    > Clone for SEBlock<T>
{
    fn clone(&self) -> Self {
        Self {
            avg_pool: self.avg_pool.clone(),
            squeeze: self.squeeze.clone(),
            relu: self.relu.clone(),
            excite: self.excite.clone(),
            sigmoid: std::marker::PhantomData,
            training: self.training,
        }
    }
}

impl<T> SEBlock<T>
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
    pub fn new(channels: usize, reduction_ratio: usize) -> Self {
        let reduced_channels = (channels / reduction_ratio).max(1);

        Self {
            avg_pool: GlobalAvgPool2D::new(),
            squeeze: Dense::new(channels, reduced_channels, true),
            relu: ReLU::new(),
            excite: Dense::new(reduced_channels, channels, true),
            sigmoid: std::marker::PhantomData,
            training: false,
        }
    }
}

impl<T> Layer<T> for SEBlock<T>
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
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];

        // Global average pooling
        let mut se = self.avg_pool.forward(input)?;

        // Reshape to (batch_size, channels)
        se = se.reshape(&[batch_size, channels])?;

        // Squeeze
        se = self.squeeze.forward(&se)?;
        se = self.relu.forward(&se)?;

        // Excite
        se = self.excite.forward(&se)?;
        se = sigmoid(&se)?;

        // Reshape back to (batch_size, channels, 1, 1)
        se = se.reshape(&[batch_size, channels, 1, 1])?;

        // Scale the input
        let scaled = input.mul(&se)?;
        Ok(scaled)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.squeeze.parameters());
        params.extend(self.excite.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.squeeze.parameters_mut());
        params.extend(self.excite.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.squeeze.set_training(training);
        self.relu.set_training(training);
        self.excite.set_training(training);
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Mobile Inverted Bottleneck Convolution Block used in EfficientNet
pub struct MBConvBlock<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    expand_conv: Option<Conv2D<T>>,
    expand_bn: Option<BatchNorm<T>>,
    depthwise_conv: DepthwiseConv2D<T>,
    depthwise_bn: BatchNorm<T>,
    se_block: Option<SEBlock<T>>,
    project_conv: Conv2D<T>,
    project_bn: BatchNorm<T>,
    use_residual: bool,
    expand_ratio: usize,
    training: bool,
}

impl<
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + scirs2_core::num_traits::FromPrimitive,
    > Clone for MBConvBlock<T>
{
    fn clone(&self) -> Self {
        Self {
            expand_conv: self.expand_conv.clone(),
            expand_bn: self.expand_bn.clone(),
            depthwise_conv: self.depthwise_conv.clone(),
            depthwise_bn: self.depthwise_bn.clone(),
            se_block: self.se_block.clone(),
            project_conv: self.project_conv.clone(),
            project_bn: self.project_bn.clone(),
            use_residual: self.use_residual,
            expand_ratio: self.expand_ratio,
            training: self.training,
        }
    }
}

impl<T> MBConvBlock<T>
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
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: usize,
        se_ratio: Option<f32>,
    ) -> Self {
        let expanded_channels = in_channels * expand_ratio;

        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            (
                Some(Conv2D::new(
                    in_channels,
                    expanded_channels,
                    (1, 1),
                    (1, 1),
                    "0".to_string(),
                    false,
                )),
                Some(BatchNorm::new(expanded_channels)),
            )
        } else {
            (None, None)
        };

        let se_block = se_ratio.map(|ratio| {
            let se_channels = (in_channels as f32 * ratio).max(1.0) as usize;
            SEBlock::new(expanded_channels, expanded_channels / se_channels)
        });

        let padding = (kernel_size - 1) / 2;

        Self {
            expand_conv,
            expand_bn,
            depthwise_conv: DepthwiseConv2D::new(
                expanded_channels,
                (kernel_size, kernel_size),
                Some((stride, stride)),
                Some(padding.to_string()),
                None, // dilation
                None, // depth_multiplier
                false,
            ),
            depthwise_bn: BatchNorm::new(expanded_channels),
            se_block,
            project_conv: Conv2D::new(
                expanded_channels,
                out_channels,
                (1, 1),
                (1, 1),
                "0".to_string(),
                false,
            ),
            project_bn: BatchNorm::new(out_channels),
            use_residual: stride == 1 && in_channels == out_channels,
            expand_ratio,
            training: false,
        }
    }
}

impl<T> Layer<T> for MBConvBlock<T>
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
        let mut x = input.clone();

        // Expansion phase
        if let (Some(ref expand_conv), Some(ref expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            x = expand_conv.forward(&x)?;
            x = expand_bn.forward(&x)?;
            x = swish(&x)?;
        }

        // Depthwise convolution
        x = self.depthwise_conv.forward(&x)?;
        x = self.depthwise_bn.forward(&x)?;
        x = swish(&x)?;

        // Squeeze-and-excitation
        if let Some(ref se) = self.se_block {
            x = se.forward(&x)?;
        }

        // Projection phase
        x = self.project_conv.forward(&x)?;
        x = self.project_bn.forward(&x)?;

        // Residual connection
        if self.use_residual {
            x = x.add(input)?;
        }

        Ok(x)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();

        if let Some(ref expand_conv) = self.expand_conv {
            params.extend(expand_conv.parameters());
        }
        if let Some(ref expand_bn) = self.expand_bn {
            params.extend(expand_bn.parameters());
        }

        params.extend(self.depthwise_conv.parameters());
        params.extend(self.depthwise_bn.parameters());

        if let Some(ref se) = self.se_block {
            params.extend(se.parameters());
        }

        params.extend(self.project_conv.parameters());
        params.extend(self.project_bn.parameters());

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();

        if let Some(ref mut expand_conv) = self.expand_conv {
            params.extend(expand_conv.parameters_mut());
        }
        if let Some(ref mut expand_bn) = self.expand_bn {
            params.extend(expand_bn.parameters_mut());
        }

        params.extend(self.depthwise_conv.parameters_mut());
        params.extend(self.depthwise_bn.parameters_mut());

        if let Some(ref mut se) = self.se_block {
            params.extend(se.parameters_mut());
        }

        params.extend(self.project_conv.parameters_mut());
        params.extend(self.project_bn.parameters_mut());

        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;

        if let Some(ref mut expand_conv) = self.expand_conv {
            expand_conv.set_training(training);
        }
        if let Some(ref mut expand_bn) = self.expand_bn {
            expand_bn.set_training(training);
        }

        self.depthwise_conv.set_training(training);
        self.depthwise_bn.set_training(training);

        if let Some(ref mut se) = self.se_block {
            se.set_training(training);
        }

        self.project_conv.set_training(training);
        self.project_bn.set_training(training);
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}
