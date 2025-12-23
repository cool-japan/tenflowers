//! Attention mechanisms and transformer components
//!
//! This module provides comprehensive attention implementations including:
//! - Multi-head attention mechanisms with KV caching
//! - Multi-query attention for efficient inference
//! - Complete transformer encoder and decoder blocks
//! - Various feed-forward network variants (FFN, SwiGLU, GeGLU)
//! - Mixture of Experts (MoE) implementation

use std::collections::HashMap;
#[cfg(feature = "gpu")]
use tenflowers_core::{Device, Result, Tensor, TensorError};
#[cfg(not(feature = "gpu"))]
use tenflowers_core::{Result, Tensor};

#[cfg(feature = "gpu")]
use std::any::TypeId;
#[cfg(feature = "gpu")]
use tenflowers_core::gpu::attention_ops::GpuAttentionOps;

// Re-export specialized modules
pub mod feed_forward;
pub mod mixture_of_experts;
pub mod multi_head;
pub mod multi_query;
pub mod transformer;
pub mod utils;

// Re-export commonly used types
pub use feed_forward::{FeedForwardNetwork, GeGLU, SwiGLU};
pub use mixture_of_experts::MixtureOfExperts;
pub use multi_head::{FlashAttentionConfig, MultiHeadAttention};
pub use multi_query::MultiQueryAttention;
pub use transformer::{TransformerDecoder, TransformerEncoder};
pub use utils::{
    analyze_attention_patterns, apply_attention_mask, apply_rotary_position_embedding,
    create_causal_mask, create_padding_mask, scaled_dot_product_attention,
    sinusoidal_positional_encoding, AttentionStats,
};

/// Key-Value cache for efficient autoregressive generation
#[derive(Debug, Clone)]
pub struct KVCache<T> {
    /// Cached keys for each layer, indexed by layer_id
    pub keys: HashMap<String, Tensor<T>>,
    /// Cached values for each layer, indexed by layer_id
    pub values: HashMap<String, Tensor<T>>,
    /// Maximum sequence length supported by the cache
    pub max_seq_len: usize,
    /// Current sequence position
    pub current_pos: usize,
}

impl<T> KVCache<T>
where
    T: Clone + Default,
{
    /// Create a new KV cache
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
            max_seq_len,
            current_pos: 0,
        }
    }

    /// Clear the cache (reset for new sequence)
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.current_pos = 0;
    }

    /// Get cached key and value for a layer
    pub fn get(&self, layer_id: &str) -> Option<(&Tensor<T>, &Tensor<T>)> {
        if let (Some(key), Some(value)) = (self.keys.get(layer_id), self.values.get(layer_id)) {
            Some((key, value))
        } else {
            None
        }
    }

    /// Update cache with new key and value tensors
    pub fn update(&mut self, layer_id: String, key: Tensor<T>, value: Tensor<T>) -> Result<()> {
        self.keys.insert(layer_id.clone(), key);
        self.values.insert(layer_id, value);
        Ok(())
    }

    /// Advance the current position (for next token generation)
    pub fn advance(&mut self) {
        self.current_pos += 1;
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.current_pos
    }

    /// Check if cache is at capacity
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }
}
