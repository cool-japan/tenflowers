//! Activation cache for storing intermediate activations during a forward pass.

use super::helpers::vec_sub;
use std::collections::HashMap;

/// Stores all intermediate activations captured during a forward pass.
///
/// Keys follow the convention:
/// `"layer_{i}_attn_output"`, `"layer_{i}_mlp"`, `"residual_{i}"`, etc.
#[derive(Debug, Clone)]
pub struct ActivationCache {
    cache: HashMap<String, Vec<f64>>,
    /// Number of transformer layers the model has.
    pub layer_count: usize,
    /// Hidden dimensionality of the model.
    pub d_model: usize,
}

impl ActivationCache {
    /// Create an empty cache for a model with `layer_count` layers of width `d_model`.
    pub fn new(layer_count: usize, d_model: usize) -> Self {
        Self {
            cache: HashMap::new(),
            layer_count,
            d_model,
        }
    }

    /// Store a named activation vector.
    pub fn store(&mut self, key: &str, values: Vec<f64>) {
        self.cache.insert(key.to_string(), values);
    }

    /// Retrieve a named activation vector, if present.
    pub fn get(&self, key: &str) -> Option<&Vec<f64>> {
        self.cache.get(key)
    }

    /// Return all stored key names, sorted for determinism.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.cache.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Element-wise difference `self[key] - other[key]`.
    ///
    /// Returns `None` if `key` is absent from either cache or if the lengths
    /// differ.
    pub fn diff(&self, other: &ActivationCache, key: &str) -> Option<Vec<f64>> {
        let a = self.cache.get(key)?;
        let b = other.cache.get(key)?;
        if a.len() != b.len() {
            return None;
        }
        Some(vec_sub(a, b))
    }
}
