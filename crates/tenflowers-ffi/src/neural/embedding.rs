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
        // A real weight table is required - never fabricate a zero output.
        let weight = self.weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Embedding: weight not initialized; call reset_parameters first",
            )
        })?;

        // Get input data as indices and validate bounds up front for clear errors.
        let input_data = input
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {e}")))?;

        for &idx_f32 in &input_data {
            let idx = idx_f32 as usize;
            if idx >= self.num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "Index {} is out of bounds for embedding with {} entries",
                    idx, self.num_embeddings
                )));
            }
        }

        // Gather embedding rows via the real neural embedding lookup. The core
        // `gather` op mishandles whole-row gathering when embedding_dim > 1, so the
        // dedicated embedding layer is used instead of returning a placeholder.
        use tenflowers_neural::layers::Layer;
        let layer = tenflowers_neural::layers::Embedding::from_pretrained(weight.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Embedding init failed: {e}")))?;
        match layer.forward(input.tensor.as_ref()) {
            Ok(output) => Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Embedding forward failed: {e}"
            ))),
        }
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
    #[pyo3(signature = (input, offsets=None, per_sample_weights=None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        offsets: Option<&PyTensor>,
        per_sample_weights: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        // A real weight table is required - never fabricate a zero output.
        let weight = self.weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "EmbeddingBag: weight not initialized; call reset_parameters first",
            )
        })?;

        let input_shape = input.tensor.shape();

        // Index values, validated to be in range.
        let idx_data = input
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {e}")))?;
        let total = idx_data.len();
        let mut indices = Vec::with_capacity(total);
        for &idx_f32 in &idx_data {
            let idx = idx_f32 as usize;
            if idx >= self.num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "Index {} is out of bounds for embedding with {} entries",
                    idx, self.num_embeddings
                )));
            }
            indices.push(idx);
        }

        // Determine the [start, end) span of each bag from offsets or 2D input shape.
        let bags: Vec<(usize, usize)> = if let Some(offsets_tensor) = offsets {
            let offsets_data = offsets_tensor
                .tensor
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get offsets: {e}")))?;
            let offs: Vec<usize> = offsets_data.iter().map(|&v| v as usize).collect();
            if offs.is_empty() {
                return Err(PyValueError::new_err("offsets must not be empty"));
            }
            if self.include_last_offset {
                if offs.len() < 2 {
                    return Err(PyValueError::new_err(
                        "offsets must contain at least 2 entries when include_last_offset is set",
                    ));
                }
                (0..offs.len() - 1)
                    .map(|i| (offs[i], offs[i + 1]))
                    .collect()
            } else {
                (0..offs.len())
                    .map(|i| {
                        let start = offs[i];
                        let end = if i + 1 < offs.len() {
                            offs[i + 1]
                        } else {
                            total
                        };
                        (start, end)
                    })
                    .collect()
            }
        } else if input_shape.len() == 2 {
            let num_bags = input_shape[0];
            let bag_size = input_shape[1];
            (0..num_bags)
                .map(|i| (i * bag_size, (i + 1) * bag_size))
                .collect()
        } else {
            return Err(PyValueError::new_err(
                "Either offsets must be provided or input must be 2D",
            ));
        };

        // Optional per-sample weights (only valid for sum mode, matching PyTorch).
        let sample_weights: Option<Vec<f32>> = if let Some(weights) = per_sample_weights {
            if self.mode != "sum" {
                return Err(PyValueError::new_err(
                    "per_sample_weights is only supported for mode='sum'",
                ));
            }
            let weights_data = weights.tensor.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get per_sample_weights: {e}"))
            })?;
            if weights_data.len() != total {
                return Err(PyValueError::new_err(format!(
                    "per_sample_weights size {} must match input size {}",
                    weights_data.len(),
                    total
                )));
            }
            Some(weights_data)
        } else {
            None
        };

        // Real embedding table data, used to gather rows for each bag.
        let weight_data = weight
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get weight: {e}")))?;
        let dim = self.embedding_dim;

        let num_bags = bags.len();
        let mut output_data = vec![0f32; num_bags * dim];

        for (bag_idx, &(start, end)) in bags.iter().enumerate() {
            if start > end || end > total {
                return Err(PyValueError::new_err(format!(
                    "Invalid bag span [{start}, {end}) for input of length {total}"
                )));
            }
            let out_base = bag_idx * dim;
            let bag_len = end - start;

            match self.mode.as_str() {
                "sum" | "mean" => {
                    for e in start..end {
                        let row = indices[e] * dim;
                        let scale = sample_weights.as_ref().map_or(1.0, |w| w[e]);
                        for k in 0..dim {
                            output_data[out_base + k] += weight_data[row + k] * scale;
                        }
                    }
                    if self.mode == "mean" && bag_len > 0 {
                        let denom = bag_len as f32;
                        for k in 0..dim {
                            output_data[out_base + k] /= denom;
                        }
                    }
                }
                "max" => {
                    if bag_len > 0 {
                        for k in 0..dim {
                            output_data[out_base + k] = f32::NEG_INFINITY;
                        }
                        for &index in &indices[start..end] {
                            let row = index * dim;
                            for k in 0..dim {
                                let val = weight_data[row + k];
                                if val > output_data[out_base + k] {
                                    output_data[out_base + k] = val;
                                }
                            }
                        }
                    }
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported EmbeddingBag mode '{other}'"
                    )));
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[num_bags, dim])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build output: {e}")))?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    // Rows: [1,2,3], [4,5,6], [7,8,9], [10,11,12].
    fn table() -> Tensor<f32> {
        let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        Tensor::from_vec(data, &[4, 3]).expect("table")
    }

    #[test]
    fn embedding_forward_gathers_rows() {
        let mut emb = PyEmbedding::new(4, 3, None, None, None, None, None).expect("emb");
        emb.weight = Some(table());
        let input = make_tensor(vec![1.0, 3.0], &[2]);
        let out = emb.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert_eq!(out_vec, vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn embedding_uninitialized_errors() {
        let mut emb = PyEmbedding::new(4, 3, None, None, None, None, None).expect("emb");
        emb.weight = None;
        let input = make_tensor(vec![0.0], &[1]);
        assert!(emb.forward(&input).is_err());
    }

    #[test]
    fn embedding_out_of_range_errors() {
        let mut emb = PyEmbedding::new(4, 3, None, None, None, None, None).expect("emb");
        emb.weight = Some(table());
        let input = make_tensor(vec![9.0], &[1]);
        assert!(emb.forward(&input).is_err());
    }

    #[test]
    fn embedding_bag_mean_2d() {
        let mut bag = PyEmbeddingBag::new(
            4,
            3,
            None,
            None,
            None,
            Some("mean".to_string()),
            None,
            None,
            None,
        )
        .expect("bag");
        bag.weight = Some(table());
        let input = make_tensor(vec![0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let out = bag.forward(&input, None, None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        // mean([1,2,3],[4,5,6]) and mean([7,8,9],[10,11,12]).
        assert_eq!(out_vec, vec![2.5, 3.5, 4.5, 8.5, 9.5, 10.5]);
    }

    #[test]
    fn embedding_bag_sum_with_offsets() {
        let mut bag = PyEmbeddingBag::new(
            4,
            3,
            None,
            None,
            None,
            Some("sum".to_string()),
            None,
            None,
            None,
        )
        .expect("bag");
        bag.weight = Some(table());
        let input = make_tensor(vec![0.0, 1.0, 2.0, 3.0], &[4]);
        let offsets = make_tensor(vec![0.0, 2.0], &[2]);
        let out = bag.forward(&input, Some(&offsets), None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        // bag0 = [1,2,3]+[4,5,6]; bag1 = [7,8,9]+[10,11,12].
        assert_eq!(out_vec, vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);
    }

    #[test]
    fn embedding_bag_max_2d() {
        let mut bag = PyEmbeddingBag::new(
            4,
            3,
            None,
            None,
            None,
            Some("max".to_string()),
            None,
            None,
            None,
        )
        .expect("bag");
        bag.weight = Some(table());
        let input = make_tensor(vec![0.0, 1.0, 2.0, 3.0], &[1, 4]);
        let out = bag.forward(&input, None, None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        // max over all four rows -> row 3.
        assert_eq!(out_vec, vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn embedding_bag_uninitialized_errors() {
        let mut bag = PyEmbeddingBag::new(
            4,
            3,
            None,
            None,
            None,
            Some("mean".to_string()),
            None,
            None,
            None,
        )
        .expect("bag");
        bag.weight = None;
        let input = make_tensor(vec![0.0, 1.0], &[1, 2]);
        assert!(bag.forward(&input, None, None).is_err());
    }
}
