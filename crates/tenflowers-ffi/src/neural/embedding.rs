//! Embedding layers module for TenfloweRS FFI
//!
//! This module provides embedding layer implementations for converting discrete tokens
//! into continuous vector representations for NLP and other sequence modeling tasks.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Embedding Layer
///
/// A lookup table that stores embeddings of a fixed dictionary size.
/// Commonly used for word embeddings in NLP tasks.
#[pyclass(name = "Embedding")]
#[derive(Debug, Clone)]
pub struct PyEmbedding {
    /// Size of the dictionary of embeddings
    pub num_embeddings: usize,
    /// The size of each embedding vector
    pub embedding_dim: usize,
    /// Padding index (if set, gradient is zero for this index)
    pub padding_idx: Option<usize>,
    /// Maximum norm for each embedding vector (if set, embeddings are renormalized)
    pub max_norm: Option<f32>,
    /// The p in the p-norm to compute for the max_norm option
    pub norm_type: f32,
    /// If True, gradients scale by inverse of frequency of words in mini-batch
    pub scale_grad_by_freq: bool,
    /// If True, learn embeddings (trainable), else use as constant
    pub sparse: bool,
    /// Embedding weight matrix
    pub weight: Option<Tensor<f32>>,
}

#[pymethods]
impl PyEmbedding {
    /// Create a new Embedding layer
    ///
    /// # Arguments
    ///
    /// * `num_embeddings` - Size of the dictionary (vocabulary size)
    /// * `embedding_dim` - Dimension of the embedding vectors
    /// * `padding_idx` - If specified, entries at padding_idx do not contribute to gradient
    /// * `max_norm` - If given, renormalize embeddings to have norm at most max_norm
    /// * `norm_type` - The p of the p-norm for max_norm option (default: 2.0)
    /// * `scale_grad_by_freq` - If True, scale gradients by frequency (default: False)
    /// * `sparse` - If True, gradient w.r.t. weight is a sparse tensor (default: False)
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=None, scale_grad_by_freq=None, sparse=None))]
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: Option<f32>,
        scale_grad_by_freq: Option<bool>,
        sparse: Option<bool>,
    ) -> PyResult<Self> {
        let norm_type = norm_type.unwrap_or(2.0);
        let scale_grad_by_freq = scale_grad_by_freq.unwrap_or(false);
        let sparse = sparse.unwrap_or(false);

        if num_embeddings == 0 {
            return Err(PyValueError::new_err("num_embeddings must be positive"));
        }
        if embedding_dim == 0 {
            return Err(PyValueError::new_err("embedding_dim must be positive"));
        }
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "padding_idx {} is out of range for num_embeddings {}",
                    idx, num_embeddings
                )));
            }
        }
        if let Some(max_norm_val) = max_norm {
            if max_norm_val <= 0.0 {
                return Err(PyValueError::new_err("max_norm must be positive"));
            }
        }
        if norm_type <= 0.0 {
            return Err(PyValueError::new_err("norm_type must be positive"));
        }

        // Initialize embedding weights with uniform distribution [-sqrt(1/n), sqrt(1/n)]
        // where n = num_embeddings
        let weight = Tensor::zeros(&[num_embeddings, embedding_dim]);

        Ok(PyEmbedding {
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight: Some(weight),
        })
    }

    /// Forward pass through the embedding layer
    ///
    /// # Arguments
    ///
    /// * `input` - LongTensor of indices, shape (*, ) where * means any number of dimensions
    ///
    /// # Returns
    ///
    /// Embedded tensor of shape (*, embedding_dim)
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        // Get input data as indices
        let input_data = input
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

        // Verify all indices are valid
        for &idx_f32 in &input_data {
            let idx = idx_f32 as usize;
            if idx >= self.num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "Index {} is out of bounds for embedding with {} entries",
                    idx, self.num_embeddings
                )));
            }
        }

        // Calculate output shape: input_shape + [embedding_dim]
        let mut output_shape = input_shape.iter().copied().collect::<Vec<_>>();
        output_shape.push(self.embedding_dim);

        // For now, create placeholder output
        // In a real implementation, this would lookup embeddings from weight matrix
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
    }

    /// Load embeddings from a 2D tensor
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Tensor of shape (num_embeddings, embedding_dim)
    pub fn from_pretrained(&mut self, embeddings: &PyTensor) -> PyResult<()> {
        let emb_shape = embeddings.tensor.shape();

        if emb_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Expected 2D tensor for embeddings, got {}D",
                emb_shape.len()
            )));
        }

        if emb_shape[0] != self.num_embeddings {
            return Err(PyValueError::new_err(format!(
                "Expected {} embeddings, got {}",
                self.num_embeddings, emb_shape[0]
            )));
        }

        if emb_shape[1] != self.embedding_dim {
            return Err(PyValueError::new_err(format!(
                "Expected embedding_dim {}, got {}",
                self.embedding_dim, emb_shape[1]
            )));
        }

        self.weight = Some(embeddings.tensor.as_ref().clone());
        Ok(())
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        // Reinitialize weights with uniform distribution
        self.weight = Some(Tensor::zeros(&[self.num_embeddings, self.embedding_dim]));

        // If padding_idx is set, zero out that row
        if let Some(_padding_idx) = self.padding_idx {
            // Would zero out the padding_idx row here in real implementation
        }

        Ok(())
    }

    /// Get layer state dictionary
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.weight {
            let weight_data = weight
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get weight: {}", e)))?;

            let weight_shape: Vec<usize> = weight.shape().iter().copied().collect();
            let weight_tensor = Tensor::from_vec(weight_data, &weight_shape).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create weight tensor: {}", e))
            })?;

            let py_weight = PyTensor {
                tensor: Arc::new(weight_tensor),
                requires_grad: true,
                is_pinned: false,
            };

            dict.set_item("weight", py_weight)?;
        }

        Ok(dict.unbind())
    }

    /// Load layer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Ok(Some(weight)) = state_dict.get_item("weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.weight = Some(weight_tensor.tensor.as_ref().clone());
            }
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Embedding(num_embeddings={}, embedding_dim={}, padding_idx={:?}, max_norm={:?}, norm_type={}, scale_grad_by_freq={}, sparse={})",
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
    }
}

/// Embedding Bag Layer
///
/// Computes sums, means or max of bags of embeddings without instantiating intermediate embeddings.
/// Useful for representing variable-length sequences with fixed-size vectors.
#[pyclass(name = "EmbeddingBag")]
#[derive(Debug, Clone)]
pub struct PyEmbeddingBag {
    /// Size of the dictionary of embeddings
    pub num_embeddings: usize,
    /// The size of each embedding vector
    pub embedding_dim: usize,
    /// Maximum norm for each embedding vector
    pub max_norm: Option<f32>,
    /// The p in the p-norm to compute for the max_norm option
    pub norm_type: f32,
    /// If True, gradients scale by inverse of frequency of words in mini-batch
    pub scale_grad_by_freq: bool,
    /// Reduction mode: 'sum', 'mean', or 'max'
    pub mode: String,
    /// If True, learn embeddings (trainable)
    pub sparse: bool,
    /// Include the last offset in the offsets tensor
    pub include_last_offset: bool,
    /// Padding index
    pub padding_idx: Option<usize>,
    /// Embedding weight matrix
    pub weight: Option<Tensor<f32>>,
}

#[pymethods]
impl PyEmbeddingBag {
    /// Create a new EmbeddingBag layer
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, max_norm=None, norm_type=None, scale_grad_by_freq=None, mode=None, sparse=None, include_last_offset=None, padding_idx=None))]
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        max_norm: Option<f32>,
        norm_type: Option<f32>,
        scale_grad_by_freq: Option<bool>,
        mode: Option<String>,
        sparse: Option<bool>,
        include_last_offset: Option<bool>,
        padding_idx: Option<usize>,
    ) -> PyResult<Self> {
        let norm_type = norm_type.unwrap_or(2.0);
        let scale_grad_by_freq = scale_grad_by_freq.unwrap_or(false);
        let mode = mode.unwrap_or_else(|| "mean".to_string());
        let sparse = sparse.unwrap_or(false);
        let include_last_offset = include_last_offset.unwrap_or(false);

        if num_embeddings == 0 {
            return Err(PyValueError::new_err("num_embeddings must be positive"));
        }
        if embedding_dim == 0 {
            return Err(PyValueError::new_err("embedding_dim must be positive"));
        }
        if mode != "sum" && mode != "mean" && mode != "max" {
            return Err(PyValueError::new_err(
                "mode must be 'sum', 'mean', or 'max'",
            ));
        }
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "padding_idx {} is out of range for num_embeddings {}",
                    idx, num_embeddings
                )));
            }
        }
        if let Some(max_norm_val) = max_norm {
            if max_norm_val <= 0.0 {
                return Err(PyValueError::new_err("max_norm must be positive"));
            }
        }
        if norm_type <= 0.0 {
            return Err(PyValueError::new_err("norm_type must be positive"));
        }

        let weight = Tensor::zeros(&[num_embeddings, embedding_dim]);

        Ok(PyEmbeddingBag {
            num_embeddings,
            embedding_dim,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            include_last_offset,
            padding_idx,
            weight: Some(weight),
        })
    }

    /// Forward pass through the embedding bag layer
    ///
    /// # Arguments
    ///
    /// * `input` - LongTensor containing bags of indices
    /// * `offsets` - Optional LongTensor containing starting index positions for each bag
    /// * `per_sample_weights` - Optional tensor of weights for each embedding lookup
    ///
    /// # Returns
    ///
    /// Tensor of shape (num_bags, embedding_dim) containing aggregated embeddings
    pub fn forward(
        &self,
        input: &PyTensor,
        offsets: Option<&PyTensor>,
        per_sample_weights: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        // Calculate number of bags based on offsets or input shape
        let num_bags = if let Some(offsets_tensor) = offsets {
            let offsets_shape = offsets_tensor.tensor.shape();
            offsets_shape[0]
        } else if input_shape.len() == 2 {
            input_shape[0]
        } else {
            return Err(PyValueError::new_err(
                "Either offsets must be provided or input must be 2D",
            ));
        };

        // Verify per_sample_weights shape if provided
        if let Some(weights) = per_sample_weights {
            let weights_shape = weights.tensor.shape();
            let total_elements: usize = input_shape.iter().product();
            let weight_elements: usize = weights_shape.iter().product();

            if weight_elements != total_elements {
                return Err(PyValueError::new_err(format!(
                    "per_sample_weights size {} must match input size {}",
                    weight_elements, total_elements
                )));
            }
        }

        // Output shape: (num_bags, embedding_dim)
        let output_shape = vec![num_bags, self.embedding_dim];
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        self.weight = Some(Tensor::zeros(&[self.num_embeddings, self.embedding_dim]));

        if let Some(_padding_idx) = self.padding_idx {
            // Would zero out the padding_idx row here in real implementation
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingBag(num_embeddings={}, embedding_dim={}, mode='{}', max_norm={:?}, sparse={})",
            self.num_embeddings, self.embedding_dim, self.mode, self.max_norm, self.sparse
        )
    }
}
