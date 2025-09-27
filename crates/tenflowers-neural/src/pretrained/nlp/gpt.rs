//! GPT (Generative Pre-trained Transformer) Models
//!
//! This module contains GPT architecture implementations for autoregressive
//! language generation tasks.

use crate::{
    layers::{Dense, Layer},
    model::Model,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_autograd::ndarray::Array2;
use tenflowers_core::{Result, Tensor};

/// GPT (Generative Pre-trained Transformer) Model
pub struct GPT<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    vocab_size: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_positions: usize,
    dropout: f32,

    // Embeddings
    token_embedding: Dense<T>,
    position_embedding: Dense<T>,

    // Transformer layers (simplified)
    decoder_layers: Vec<Box<dyn Layer<T>>>,

    // Language modeling head
    lm_head: Dense<T>,

    training: bool,
}

impl<T> Clone for GPT<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            vocab_size: self.vocab_size,
            n_embd: self.n_embd,
            n_layer: self.n_layer,
            n_head: self.n_head,
            n_positions: self.n_positions,
            dropout: self.dropout,
            token_embedding: self.token_embedding.clone(),
            position_embedding: self.position_embedding.clone(),
            decoder_layers: self
                .decoder_layers
                .iter()
                .map(|layer| layer.clone_box())
                .collect(),
            lm_head: self.lm_head.clone(),
            training: self.training,
        }
    }
}

impl<T> GPT<T>
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
        + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_layer: usize,
        n_head: usize,
        n_positions: usize,
        dropout: f32,
        attn_dropout: f32,
        resid_dropout: f32,
    ) -> Self {
        Self {
            vocab_size,
            n_embd,
            n_layer,
            n_head,
            n_positions,
            dropout,
            token_embedding: Dense::new(vocab_size, n_embd, true),
            position_embedding: Dense::new(n_positions, n_embd, true),
            decoder_layers: Vec::new(),
            lm_head: Dense::new(n_embd, vocab_size, false),
            training: false,
        }
    }

    /// GPT-2 Small (117M parameters)
    pub fn gpt2_small(vocab_size: usize) -> Self {
        Self::new(
            vocab_size, 768,  // n_embd
            12,   // n_layer
            12,   // n_head
            1024, // n_positions
            0.1,  // dropout
            0.1,  // attn_dropout
            0.1,  // resid_dropout
        )
    }

    /// GPT-2 Medium (345M parameters)
    pub fn gpt2_medium(vocab_size: usize) -> Self {
        Self::new(
            vocab_size, 1024, // n_embd
            24,   // n_layer
            16,   // n_head
            1024, // n_positions
            0.1,  // dropout
            0.1,  // attn_dropout
            0.1,  // resid_dropout
        )
    }

    /// GPT-2 Large (762M parameters)
    pub fn gpt2_large(vocab_size: usize) -> Self {
        Self::new(
            vocab_size, 1280, // n_embd
            36,   // n_layer
            20,   // n_head
            1024, // n_positions
            0.1,  // dropout
            0.1,  // attn_dropout
            0.1,  // resid_dropout
        )
    }

    /// GPT-2 XL (1.5B parameters)
    pub fn gpt2_xl(vocab_size: usize) -> Self {
        Self::new(
            vocab_size, 1600, // n_embd
            48,   // n_layer
            25,   // n_head
            1024, // n_positions
            0.1,  // dropout
            0.1,  // attn_dropout
            0.1,  // resid_dropout
        )
    }
}

impl<T> Model<T> for GPT<T>
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
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_length = input_shape[1];

        // Token embeddings
        let mut embeddings = self.token_embedding.forward(input)?;

        // Generate position IDs (0, 1, 2, ..., seq_length-1)
        let position_ids = {
            let positions: Vec<T> = (0..seq_length)
                .map(|i| T::from_usize(i).unwrap_or_else(T::zero))
                .collect();
            let position_array =
                Array2::from_shape_vec((1, seq_length), positions).map_err(|e| {
                    tenflowers_core::TensorError::shape_mismatch(
                        "Position array creation",
                        "valid shape for positions",
                        &format!("invalid shape: {}", e),
                    )
                })?;

            // Broadcast to batch size
            let mut batch_positions = Array2::zeros((batch_size, seq_length));
            for b in 0..batch_size {
                for s in 0..seq_length {
                    batch_positions[[b, s]] = position_array[[0, s]];
                }
            }

            Tensor::from_array(batch_positions.into_dyn())
        };

        // Add positional embeddings
        let pos_embeddings = self.position_embedding.forward(&position_ids)?;
        embeddings = embeddings.add(&pos_embeddings)?;

        // Process through decoder layers
        for layer in &self.decoder_layers {
            embeddings = layer.forward(&embeddings)?;
        }

        // Language modeling head
        self.lm_head.forward(&embeddings)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.extend(self.position_embedding.parameters());
        for layer in &self.decoder_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.lm_head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters_mut());
        params.extend(self.position_embedding.parameters_mut());
        for layer in &mut self.decoder_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.lm_head.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.token_embedding.set_training(training);
        self.position_embedding.set_training(training);
        for layer in &mut self.decoder_layers {
            layer.set_training(training);
        }
        self.lm_head.set_training(training);
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
