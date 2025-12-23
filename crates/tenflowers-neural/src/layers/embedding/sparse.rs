//! Sparse Embedding Layers
//!
//! This module contains sparse embedding implementations optimized for large vocabularies
//! where only a small fraction of embeddings are accessed in each batch:
//!
//! - **SparseEmbedding**: Embedding layer with sparse gradient tracking and optimization
//! - **SparseEmbeddingGrad**: Sparse gradient tracking structure for memory efficiency
//! - **SparseGradientStats**: Statistics and monitoring for sparse gradient patterns
//!
//! Sparse embeddings are particularly useful for:
//! - Large vocabulary language models
//! - Recommendation systems with millions of items/users
//! - Any scenario where the embedding matrix is very large but sparsely accessed

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use std::collections::{HashMap, HashSet};
use tenflowers_core::{Result, Tensor, TensorError};

/// Statistics for sparse gradient analysis and monitoring
///
/// Provides insights into gradient sparsity patterns, magnitudes, and optimization behavior
/// for sparse embedding layers.
#[derive(Debug, Clone)]
pub struct SparseGradientStats<T> {
    /// Number of embeddings that received gradient updates
    pub num_accessed_embeddings: usize,
    /// Total number of embeddings in the vocabulary
    pub total_embeddings: usize,
    /// Sparsity ratio (1.0 - accessed/total)
    pub sparsity_ratio: f64,
    /// Average magnitude of gradients across all accessed embeddings
    pub avg_gradient_magnitude: T,
    /// Maximum gradient magnitude observed
    pub max_gradient_magnitude: T,
    /// Minimum gradient magnitude observed
    pub min_gradient_magnitude: T,
}

/// Sparse gradient information for embeddings
///
/// Tracks which embeddings were accessed and stores gradients only for those embeddings.
/// This is much more memory efficient for large vocabularies where only a small fraction
/// of embeddings are used in each batch.
#[derive(Debug, Clone)]
pub struct SparseEmbeddingGrad<T> {
    /// Map from embedding index to gradient vector
    gradients: HashMap<usize, Vec<T>>,
    /// Set of indices that were accessed in the forward pass
    accessed_indices: HashSet<usize>,
    /// Embedding dimension for validation
    embedding_dim: usize,
}

impl<T> SparseEmbeddingGrad<T>
where
    T: Clone + Default + Zero + One,
{
    /// Create a new sparse gradient structure
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            gradients: HashMap::new(),
            accessed_indices: HashSet::new(),
            embedding_dim,
        }
    }

    /// Record that an embedding index was accessed
    pub fn record_access(&mut self, index: usize) {
        self.accessed_indices.insert(index);
    }

    /// Add gradient for a specific embedding index
    pub fn add_gradient(&mut self, index: usize, grad: Vec<T>) -> Result<()> {
        if grad.len() != self.embedding_dim {
            return Err(TensorError::invalid_shape_simple(format!(
                "Gradient dimension {} doesn't match embedding dimension {}",
                grad.len(),
                self.embedding_dim
            )));
        }

        // Add to existing gradient or create new one
        if let Some(existing_grad) = self.gradients.get_mut(&index) {
            for (i, g) in grad.into_iter().enumerate() {
                existing_grad[i] = existing_grad[i].clone() + g;
            }
        } else {
            self.gradients.insert(index, grad);
        }
        Ok(())
    }

    /// Get gradient for a specific index
    pub fn get_gradient(&self, index: usize) -> Option<&Vec<T>> {
        self.gradients.get(&index)
    }

    /// Get all accessed indices
    pub fn accessed_indices(&self) -> &HashSet<usize> {
        &self.accessed_indices
    }

    /// Get the number of accessed embeddings
    pub fn num_accessed(&self) -> usize {
        self.accessed_indices.len()
    }

    /// Check if gradients exist for all accessed indices
    pub fn is_complete(&self) -> bool {
        self.accessed_indices
            .iter()
            .all(|&idx| self.gradients.contains_key(&idx))
    }

    /// Clear all gradients and accessed indices
    pub fn clear(&mut self) {
        self.gradients.clear();
        self.accessed_indices.clear();
    }
}

/// Embedding layer with sparse gradient support
///
/// This version of the embedding layer tracks which embeddings are accessed during
/// the forward pass and only computes gradients for those embeddings. This is much
/// more efficient for large vocabularies.
pub struct SparseEmbedding<T> {
    /// Number of unique items in the vocabulary (input dimension)
    num_embeddings: usize,
    /// Size of each embedding vector (output dimension)
    embedding_dim: usize,
    /// Weight matrix storing the embedding vectors
    /// Shape: [num_embeddings, embedding_dim]
    weight: Tensor<T>,
    /// Sparse gradient information
    sparse_grad: Option<SparseEmbeddingGrad<T>>,
    /// Whether layer is in training mode
    training: bool,
    /// Whether to track sparse gradients
    sparse_gradients: bool,
}

impl<T> SparseEmbedding<T>
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
    /// Create a new sparse embedding layer
    ///
    /// # Arguments
    /// * `num_embeddings` - Size of the vocabulary (number of unique items)
    /// * `embedding_dim` - Size of each embedding vector
    /// * `sparse_gradients` - Whether to use sparse gradients (recommended for large vocabularies)
    pub fn new(num_embeddings: usize, embedding_dim: usize, sparse_gradients: bool) -> Self {
        // Initialize embeddings with small random values
        // For now, using zeros - in practice you'd want proper initialization
        let weight = Tensor::zeros(&[num_embeddings, embedding_dim]);

        Self {
            num_embeddings,
            embedding_dim,
            weight,
            sparse_grad: if sparse_gradients {
                Some(SparseEmbeddingGrad::new(embedding_dim))
            } else {
                None
            },
            training: true,
            sparse_gradients,
        }
    }

    /// Create sparse embedding layer with pre-trained weights
    ///
    /// # Arguments
    /// * `weights` - Pre-trained embedding matrix [num_embeddings, embedding_dim]
    /// * `sparse_gradients` - Whether to use sparse gradients
    pub fn from_pretrained(weights: Tensor<T>, sparse_gradients: bool) -> Result<Self> {
        let shape = weights.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Embedding weights must be 2D".to_string(),
            ));
        }

        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Ok(Self {
            num_embeddings,
            embedding_dim,
            weight: weights,
            sparse_grad: if sparse_gradients {
                Some(SparseEmbeddingGrad::new(embedding_dim))
            } else {
                None
            },
            training: true,
            sparse_gradients,
        })
    }

    /// Get the vocabulary size
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Check if sparse gradients are enabled
    pub fn using_sparse_gradients(&self) -> bool {
        self.sparse_gradients
    }

    /// Get sparse gradient information (if available)
    pub fn sparse_grad(&self) -> Option<&SparseEmbeddingGrad<T>> {
        self.sparse_grad.as_ref()
    }

    /// Get mutable sparse gradient information (if available)
    pub fn sparse_grad_mut(&mut self) -> Option<&mut SparseEmbeddingGrad<T>> {
        self.sparse_grad.as_mut()
    }

    /// Clear sparse gradients
    pub fn clear_sparse_gradients(&mut self) {
        if let Some(ref mut sparse_grad) = self.sparse_grad {
            sparse_grad.clear();
        }
    }

    /// Get the number of accessed embeddings in the last forward pass
    pub fn num_accessed_embeddings(&self) -> usize {
        self.sparse_grad.as_ref().map_or(0, |sg| sg.num_accessed())
    }

    /// Get the sparsity ratio (accessed / total embeddings)
    pub fn sparsity_ratio(&self) -> f32 {
        if self.num_embeddings == 0 {
            return 0.0;
        }
        self.num_accessed_embeddings() as f32 / self.num_embeddings as f32
    }

    /// Perform embedding lookup for given indices with sparse gradient tracking
    ///
    /// # Arguments
    /// * `indices` - Tensor of indices to look up. Can be any shape.
    ///
    /// # Returns
    /// Tensor with shape [...indices.shape, embedding_dim]
    fn lookup_embeddings_sparse(&mut self, indices: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::ToPrimitive + scirs2_core::num_traits::FromPrimitive,
    {
        // Clear previous sparse gradient tracking
        if let Some(ref mut sparse_grad) = self.sparse_grad {
            sparse_grad.clear();
        }

        // Get indices data
        let indices_data = indices.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access indices tensor data".to_string())
        })?;

        let weight_data = self.weight.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access weight tensor data".to_string())
        })?;

        let indices_shape = indices.shape().dims();
        let total_indices = indices_shape.iter().product::<usize>();

        // Build output shape: [...indices_shape, embedding_dim]
        let mut output_shape = indices_shape.to_vec();
        output_shape.push(self.embedding_dim);

        let mut output_data = Vec::with_capacity(total_indices * self.embedding_dim);

        for &index_val in indices_data {
            // Convert index to usize
            let index = index_val.to_usize().ok_or_else(|| {
                TensorError::invalid_argument("Invalid index in embedding lookup".to_string())
            })?;

            if index >= self.num_embeddings {
                return Err(TensorError::invalid_argument(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    index, self.num_embeddings
                )));
            }

            // Record access for sparse gradients
            if let Some(ref mut sparse_grad) = self.sparse_grad {
                sparse_grad.record_access(index);
            }

            // Extract embedding vector for this index
            let start_idx = index * self.embedding_dim;
            let end_idx = start_idx + self.embedding_dim;
            output_data.extend_from_slice(&weight_data[start_idx..end_idx]);
        }

        Tensor::from_vec(output_data, &output_shape)
    }

    /// Apply sparse gradients to the embedding weights
    ///
    /// This method applies only the gradients for embeddings that were accessed
    /// during the forward pass, making it much more efficient than dense updates.
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Performance
    /// Only updates embeddings that were accessed, providing significant memory
    /// and computational savings for large vocabularies with sparse access patterns.
    pub fn apply_sparse_gradients(&mut self, learning_rate: T) -> Result<()> {
        if !self.sparse_gradients {
            return Ok(()); // No-op if sparse gradients are disabled
        }

        let sparse_grad = self.sparse_grad.as_ref().ok_or_else(|| {
            TensorError::invalid_argument("Sparse gradients not initialized".to_string())
        })?;

        if !sparse_grad.is_complete() {
            return Err(TensorError::invalid_argument(
                "Sparse gradients incomplete - not all accessed embeddings have gradients"
                    .to_string(),
            ));
        }

        // Validate all gradients before applying
        for (&embedding_idx, gradient) in &sparse_grad.gradients {
            if embedding_idx >= self.num_embeddings {
                return Err(TensorError::invalid_argument(format!(
                    "Gradient index {} exceeds vocabulary size {}",
                    embedding_idx, self.num_embeddings
                )));
            }

            if gradient.len() != self.embedding_dim {
                return Err(TensorError::invalid_argument(format!(
                    "Gradient dimension {} does not match embedding dimension {}",
                    gradient.len(),
                    self.embedding_dim
                )));
            }
        }

        // Clone the gradients map and learning rate for use in closure
        let gradients = sparse_grad.gradients.clone();
        let lr = learning_rate.clone();
        let embedding_dim = self.embedding_dim;

        // Use Cell for position tracking inside closure
        use std::cell::Cell;
        let position = Cell::new(0usize);

        self.weight.map_inplace(|weight_val| {
            let pos = position.get();
            let embedding_idx = pos / embedding_dim;
            let dim_idx = pos % embedding_dim;

            // Check if this embedding has a gradient update
            let result = if let Some(gradient) = gradients.get(&embedding_idx) {
                if dim_idx < gradient.len() {
                    // Apply gradient descent: weight = weight - learning_rate * gradient
                    weight_val.clone() - lr.clone() * gradient[dim_idx]
                } else {
                    weight_val.clone()
                }
            } else {
                weight_val.clone()
            };

            position.set(pos + 1);
            result
        })?;

        Ok(())
    }

    /// Apply sparse gradients with momentum
    ///
    /// Enhanced sparse gradient application with momentum support for better convergence.
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `momentum` - Momentum coefficient (typically 0.9)
    /// * `momentum_buffer` - Optional momentum buffer for velocity tracking
    pub fn apply_sparse_gradients_with_momentum(
        &mut self,
        learning_rate: T,
        momentum: T,
        momentum_buffer: &mut Option<std::collections::HashMap<usize, Vec<T>>>,
    ) -> Result<()> {
        if !self.sparse_gradients {
            return Ok(()); // No-op if sparse gradients are disabled
        }

        let sparse_grad = self.sparse_grad.as_ref().ok_or_else(|| {
            TensorError::invalid_argument("Sparse gradients not initialized".to_string())
        })?;

        if !sparse_grad.is_complete() {
            return Err(TensorError::invalid_argument(
                "Sparse gradients incomplete - not all accessed embeddings have gradients"
                    .to_string(),
            ));
        }

        // Initialize momentum buffer if needed
        if momentum_buffer.is_none() {
            *momentum_buffer = Some(std::collections::HashMap::new());
        }

        let momentum_map = momentum_buffer.as_mut().unwrap();

        // Validate all gradients and initialize momentum buffers
        for (&embedding_idx, gradient) in &sparse_grad.gradients {
            if embedding_idx >= self.num_embeddings {
                return Err(TensorError::invalid_argument(format!(
                    "Gradient index {} exceeds vocabulary size {}",
                    embedding_idx, self.num_embeddings
                )));
            }

            if gradient.len() != self.embedding_dim {
                return Err(TensorError::invalid_argument(format!(
                    "Gradient dimension {} does not match embedding dimension {}",
                    gradient.len(),
                    self.embedding_dim
                )));
            }

            // Initialize momentum for this embedding if not present
            momentum_map
                .entry(embedding_idx)
                .or_insert_with(|| vec![T::zero(); self.embedding_dim]);
        }

        // Update momentum buffers first
        for (&embedding_idx, gradient) in &sparse_grad.gradients {
            let velocity = momentum_map.get_mut(&embedding_idx).unwrap();

            // Apply momentum update: v = momentum * v + learning_rate * gradient
            for (i, &grad_val) in gradient.iter().enumerate() {
                if i < velocity.len() {
                    velocity[i] =
                        momentum.clone() * velocity[i].clone() + learning_rate.clone() * grad_val;
                }
            }
        }

        // Clone data for closure
        let gradients = sparse_grad.gradients.clone();
        let momentum_velocities = momentum_map.clone();
        let embedding_dim = self.embedding_dim;

        // Apply momentum-updated gradients using map_inplace
        use std::cell::Cell;
        let position = Cell::new(0usize);

        self.weight.map_inplace(|weight_val| {
            let pos = position.get();
            let embedding_idx = pos / embedding_dim;
            let dim_idx = pos % embedding_dim;

            // Check if this embedding has a momentum update
            let result = if gradients.contains_key(&embedding_idx) {
                if let Some(velocity) = momentum_velocities.get(&embedding_idx) {
                    if dim_idx < velocity.len() {
                        // Apply velocity to weight: weight = weight - velocity
                        weight_val.clone() - velocity[dim_idx].clone()
                    } else {
                        weight_val.clone()
                    }
                } else {
                    weight_val.clone()
                }
            } else {
                weight_val.clone()
            };

            position.set(pos + 1);
            result
        })?;

        Ok(())
    }

    /// Get sparse gradient statistics for monitoring and optimization
    ///
    /// Returns information about gradient sparsity, magnitude, and access patterns.
    pub fn sparse_gradient_stats(&self) -> Result<SparseGradientStats<T>> {
        let sparse_grad = self.sparse_grad.as_ref().ok_or_else(|| {
            TensorError::invalid_argument("Sparse gradients not initialized".to_string())
        })?;

        let num_accessed = sparse_grad.gradients.len();
        let total_embeddings = self.num_embeddings;
        let sparsity = 1.0 - (num_accessed as f64 / total_embeddings as f64);

        // Calculate gradient magnitude statistics
        let mut total_magnitude = T::zero();
        let mut max_magnitude = T::zero();
        let mut min_magnitude = T::from(f64::INFINITY).unwrap_or(T::one());

        for gradient in sparse_grad.gradients.values() {
            for &grad_val in gradient {
                let magnitude = grad_val.abs();
                total_magnitude = total_magnitude + magnitude.clone();

                if magnitude > max_magnitude {
                    max_magnitude = magnitude.clone();
                }
                if magnitude < min_magnitude {
                    min_magnitude = magnitude;
                }
            }
        }

        let avg_magnitude = if num_accessed > 0 {
            total_magnitude / T::from(num_accessed * self.embedding_dim).unwrap_or(T::one())
        } else {
            T::zero()
        };

        Ok(SparseGradientStats {
            num_accessed_embeddings: num_accessed,
            total_embeddings,
            sparsity_ratio: sparsity,
            avg_gradient_magnitude: avg_magnitude,
            max_gradient_magnitude: max_magnitude,
            min_gradient_magnitude: if min_magnitude == T::from(f64::INFINITY).unwrap_or(T::one()) {
                T::zero()
            } else {
                min_magnitude
            },
        })
    }
}

impl<T> Layer<T> for SparseEmbedding<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::ToPrimitive
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Note: This creates a mutable copy to track sparse gradients
        // In a real implementation, you'd want to handle this differently
        // to avoid the clone. For now, this demonstrates the concept.
        let mut self_mut = self.clone();
        self_mut.lookup_embeddings_sparse(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

// Implement Clone for SparseEmbedding to support the forward method
impl<T> Clone for SparseEmbedding<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
            weight: self.weight.clone(),
            sparse_grad: self.sparse_grad.clone(),
            training: self.training,
            sparse_gradients: self.sparse_gradients,
        }
    }
}
