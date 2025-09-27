//! Vision Transformer (ViT) Models
//!
//! This module contains Vision Transformer implementations with patch-based image processing
//! and transformer architectures for computer vision tasks.

use crate::{
    layers::{Conv2D, Dense, Layer},
    model::Model,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_autograd::ndarray::Array3;
use scirs2_core::random::rng;
use tenflowers_core::tensor::TensorStorage;
use tenflowers_core::{Result, Tensor};

/// Patch embedding layer for Vision Transformers
pub struct PatchEmbedding<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    patch_size: usize,
    embed_dim: usize,
    num_patches: usize,
    proj: Conv2D<T>,
}

impl<T> PatchEmbedding<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(img_size: usize, patch_size: usize, embed_dim: usize, in_channels: usize) -> Self {
        assert_eq!(
            img_size % patch_size,
            0,
            "Image size must be divisible by patch size"
        );

        let num_patches = (img_size / patch_size).pow(2);

        Self {
            patch_size,
            embed_dim,
            num_patches,
            proj: Conv2D::new(
                in_channels,
                embed_dim,
                (patch_size, patch_size),
                (patch_size, patch_size),
                "0".to_string(),
                true,
            ),
        }
    }

    pub fn num_patches(&self) -> usize {
        self.num_patches
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

impl<T> Layer<T> for PatchEmbedding<T>
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
        // Project patches
        let mut x = self.proj.forward(input)?;

        // Flatten spatial dimensions and reshape to (batch_size, num_patches, embed_dim)
        let batch_size = x.shape()[0];
        x.reshape(&[batch_size, self.num_patches, self.embed_dim])
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        self.proj.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.proj.parameters_mut()
    }

    fn set_training(&mut self, training: bool) {
        self.proj.set_training(training);
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

impl<
        T: Clone
            + Default
            + Zero
            + One
            + Float
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    > Clone for PatchEmbedding<T>
{
    fn clone(&self) -> Self {
        Self {
            patch_size: self.patch_size,
            embed_dim: self.embed_dim,
            num_patches: self.num_patches,
            proj: self.proj.clone(),
        }
    }
}

/// Vision Transformer (ViT) model
pub struct VisionTransformer<T>
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
    pub img_size: usize,
    pub patch_size: usize,
    pub num_classes: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub patch_embed: PatchEmbedding<T>,
    pub transformer_layers: Vec<Box<dyn Layer<T>>>,
    pub head: Dense<T>,

    // Vision Transformer specific components
    pub cls_token: Tensor<T>,
    pub pos_embedding: Tensor<T>,

    training: bool,
}

impl<T> VisionTransformer<T>
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        num_classes: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        mlp_ratio: f32,
        in_channels: usize,
    ) -> Self {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, embed_dim, in_channels);
        let num_patches = patch_embed.num_patches();

        // Initialize CLS token with random values
        let mut rng = rng();
        let cls_data = Array3::from_shape_fn((1, 1, embed_dim), |_| {
            let range = T::from(0.04).unwrap(); // Total range from -0.02 to +0.02
            let random_val: f64 = rng.gen_range(0.0..1.0);
            T::from(random_val).unwrap() * range - T::from(0.02).unwrap()
        });
        let cls_token = Tensor::from_array(cls_data.into_dyn());

        // Initialize positional embeddings (num_patches + 1 for CLS token)
        let pos_data = Array3::from_shape_fn((1, num_patches + 1, embed_dim), |_| {
            let range = T::from(0.04).unwrap(); // Total range from -0.02 to +0.02
            let random_val: f64 = rng.gen_range(0.0..1.0);
            T::from(random_val).unwrap() * range - T::from(0.02).unwrap()
        });
        let pos_embedding = Tensor::from_array(pos_data.into_dyn());

        Self {
            img_size,
            patch_size,
            num_classes,
            embed_dim,
            num_layers,
            num_heads,
            patch_embed,
            transformer_layers: Vec::new(),
            head: Dense::new(embed_dim, num_classes, true),
            cls_token,
            pos_embedding,
            training: false,
        }
    }

    /// ViT-Tiny (5.5M parameters)
    pub fn vit_tiny_patch16_224(num_classes: usize) -> Self {
        Self::new(224, 16, num_classes, 192, 12, 3, 4.0, 3)
    }

    /// ViT-Small (22M parameters)
    pub fn vit_small_patch16_224(num_classes: usize) -> Self {
        Self::new(224, 16, num_classes, 384, 12, 6, 4.0, 3)
    }

    /// ViT-Base (86M parameters)
    pub fn vit_base_patch16_224(num_classes: usize) -> Self {
        Self::new(224, 16, num_classes, 768, 12, 12, 4.0, 3)
    }

    /// ViT-Base with 32x32 patches
    pub fn vit_base_patch32_224(num_classes: usize) -> Self {
        Self::new(224, 32, num_classes, 768, 12, 12, 4.0, 3)
    }

    /// ViT-Large (307M parameters)
    pub fn vit_large_patch16_224(num_classes: usize) -> Self {
        Self::new(224, 16, num_classes, 1024, 24, 16, 4.0, 3)
    }
}

impl<T> Model<T> for VisionTransformer<T>
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

        // Patch embedding
        let mut x = self.patch_embed.forward(input)?; // Shape: [batch_size, num_patches, embed_dim]

        // Expand CLS token for batch size
        let cls_tokens = {
            let cls_shape = self.cls_token.shape();
            let mut expanded_cls = Array3::zeros((batch_size, 1, self.embed_dim));

            for b in 0..batch_size {
                for e in 0..self.embed_dim {
                    expanded_cls[[b, 0, e]] = self.cls_token.as_slice().ok_or_else(|| {
                        tenflowers_core::TensorError::invalid_operation_simple(
                            "Cannot access CLS token data".to_string(),
                        )
                    })?[e];
                }
            }

            Tensor::from_storage(
                TensorStorage::Cpu(expanded_cls.into_dyn()),
                input.device().clone(),
            )
        };

        // Concatenate CLS token with patch embeddings
        x = tenflowers_core::ops::manipulation::concatenation::concat(&[&cls_tokens, &x], 1)?; // Shape: [batch_size, num_patches + 1, embed_dim]

        // Expand positional embeddings for batch size and add them
        let pos_embeddings = {
            let pos_shape = self.pos_embedding.shape();
            let seq_length = pos_shape[1]; // num_patches + 1
            let mut expanded_pos = Array3::zeros((batch_size, seq_length, self.embed_dim));

            let pos_data = self.pos_embedding.as_slice().ok_or_else(|| {
                tenflowers_core::TensorError::invalid_operation_simple(
                    "Cannot access positional embedding data".to_string(),
                )
            })?;

            for b in 0..batch_size {
                for s in 0..seq_length {
                    for e in 0..self.embed_dim {
                        expanded_pos[[b, s, e]] = pos_data[s * self.embed_dim + e];
                    }
                }
            }

            Tensor::from_storage(
                TensorStorage::Cpu(expanded_pos.into_dyn()),
                input.device().clone(),
            )
        };

        // Add positional embeddings
        x = x.add(&pos_embeddings)?;

        // Process through transformer layers
        for layer in &self.transformer_layers {
            x = layer.forward(&x)?;
        }

        // Classification head (use [CLS] token - first token)
        let cls_output = x.slice(&[0..batch_size, 0..1, 0..self.embed_dim])?;
        let cls_output = cls_output.reshape(&[batch_size, self.embed_dim])?;

        self.head.forward(&cls_output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        params.push(&self.cls_token);
        params.push(&self.pos_embedding);
        for layer in &self.transformer_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters_mut());
        params.push(&mut self.cls_token);
        params.push(&mut self.pos_embedding);
        for layer in &mut self.transformer_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.head.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.patch_embed.set_training(training);
        for layer in &mut self.transformer_layers {
            layer.set_training(training);
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

impl<T> Clone for VisionTransformer<T>
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
    fn clone(&self) -> Self {
        Self {
            img_size: self.img_size,
            patch_size: self.patch_size,
            num_classes: self.num_classes,
            embed_dim: self.embed_dim,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            patch_embed: self.patch_embed.clone(),
            transformer_layers: self
                .transformer_layers
                .iter()
                .map(|layer| layer.clone_box())
                .collect(),
            head: self.head.clone(),
            cls_token: self.cls_token.clone(),
            pos_embedding: self.pos_embedding.clone(),
            training: self.training,
        }
    }
}
