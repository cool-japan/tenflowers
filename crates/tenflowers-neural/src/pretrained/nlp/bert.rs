//! BERT (Bidirectional Encoder Representations from Transformers) Models
//!
//! This module contains BERT architecture implementations for bidirectional
//! language understanding tasks.

use crate::{
    layers::{Dense, Layer},
    model::Model,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_autograd::ndarray::Array2;
use tenflowers_core::tensor::TensorStorage;
use tenflowers_core::{Result, Tensor};

/// BERT (Bidirectional Encoder Representations from Transformers) Model
pub struct BERT<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    dropout_prob: f32,

    // Embeddings
    word_embeddings: Dense<T>,
    position_embeddings: Dense<T>,
    token_type_embeddings: Dense<T>,

    // Transformer layers (simplified)
    encoder_layers: Vec<Box<dyn Layer<T>>>,

    // Output layers
    pooler: Dense<T>,

    training: bool,
}

impl<T> Clone for BERT<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            dropout_prob: self.dropout_prob,
            word_embeddings: self.word_embeddings.clone(),
            position_embeddings: self.position_embeddings.clone(),
            token_type_embeddings: self.token_type_embeddings.clone(),
            encoder_layers: self
                .encoder_layers
                .iter()
                .map(|layer| layer.clone_box())
                .collect(),
            pooler: self.pooler.clone(),
            training: self.training,
        }
    }
}

impl<T> BERT<T>
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
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        dropout_prob: f32,
        type_vocab_size: usize,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            max_position_embeddings,
            dropout_prob,
            word_embeddings: Dense::new(vocab_size, hidden_size, true),
            position_embeddings: Dense::new(max_position_embeddings, hidden_size, true),
            token_type_embeddings: Dense::new(type_vocab_size, hidden_size, true),
            encoder_layers: Vec::new(),
            pooler: Dense::new(hidden_size, hidden_size, true),
            training: false,
        }
    }

    /// BERT-Base configuration
    pub fn bert_base(vocab_size: usize) -> Self {
        Self::new(
            vocab_size, 768,  // hidden_size
            12,   // num_layers
            12,   // num_heads
            3072, // intermediate_size
            512,  // max_position_embeddings
            0.1,  // dropout_prob
            2,    // type_vocab_size
        )
    }

    /// BERT-Large configuration
    pub fn bert_large(vocab_size: usize) -> Self {
        Self::new(
            vocab_size, 1024, // hidden_size
            24,   // num_layers
            16,   // num_heads
            4096, // intermediate_size
            512,  // max_position_embeddings
            0.1,  // dropout_prob
            2,    // type_vocab_size
        )
    }
}

impl<T> Model<T> for BERT<T>
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

        // Word embeddings
        let mut embeddings = self.word_embeddings.forward(input)?;

        // Generate position IDs (0, 1, 2, ..., seq_length-1)
        let position_ids = {
            let positions: Vec<T> = (0..seq_length)
                .map(|i| T::from_usize(i).unwrap_or_else(T::zero))
                .collect();
            let position_array =
                Array2::from_shape_vec((1, seq_length), positions).map_err(|e| {
                    tenflowers_core::TensorError::invalid_shape_simple(format!(
                        "Position array creation failed: {}",
                        e
                    ))
                })?;

            // Broadcast to batch size
            let mut batch_positions = Array2::zeros((batch_size, seq_length));
            for b in 0..batch_size {
                for s in 0..seq_length {
                    batch_positions[[b, s]] = position_array[[0, s]];
                }
            }

            Tensor::from_storage(
                TensorStorage::Cpu(batch_positions.into_dyn()),
                input.device().clone(),
            )
        };

        // Add positional embeddings
        let pos_embeddings = self.position_embeddings.forward(&position_ids)?;
        embeddings = embeddings.add(&pos_embeddings)?;

        // Generate token type IDs (assuming all tokens are from first segment for simplicity)
        let token_type_ids = {
            let token_types = Array2::zeros((batch_size, seq_length));
            Tensor::from_array(token_types.into_dyn())
        };

        // Add token type embeddings
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        embeddings = embeddings.add(&token_type_embeddings)?;

        // Process through encoder layers
        for layer in &self.encoder_layers {
            embeddings = layer.forward(&embeddings)?;
        }

        // Pooling (simplified - just use first token)
        let pooled = embeddings.slice(&[0..batch_size, 0..1, 0..self.hidden_size])?;
        let pooled = pooled.reshape(&[batch_size, self.hidden_size])?;

        self.pooler.forward(&pooled)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.word_embeddings.parameters());
        params.extend(self.position_embeddings.parameters());
        params.extend(self.token_type_embeddings.parameters());
        for layer in &self.encoder_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.pooler.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        params.extend(self.word_embeddings.parameters_mut());
        params.extend(self.position_embeddings.parameters_mut());
        params.extend(self.token_type_embeddings.parameters_mut());
        for layer in &mut self.encoder_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.pooler.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.word_embeddings.set_training(training);
        self.position_embeddings.set_training(training);
        self.token_type_embeddings.set_training(training);
        for layer in &mut self.encoder_layers {
            layer.set_training(training);
        }
        self.pooler.set_training(training);
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
