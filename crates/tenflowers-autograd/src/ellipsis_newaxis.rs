use num_traits::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Index specification for advanced indexing with ellipsis and newaxis support
#[derive(Debug, Clone)]
pub enum IndexSpec {
    /// Slice with start:stop:step (None means default)
    Slice {
        start: Option<i32>,
        stop: Option<i32>,
        step: Option<i32>,
    },
    /// Single index
    Index(i32),
    /// Ellipsis - fill remaining dimensions with full slices
    Ellipsis,
    /// NewAxis - insert a new dimension of size 1
    NewAxis,
    /// Boolean mask for this dimension
    BoolMask(Tensor<bool>),
    /// Integer array indexing
    IntArray(Vec<i32>),
}

/// Advanced indexing operation supporting ellipsis and newaxis
pub struct AdvancedIndexer {
    /// Original tensor shape
    original_shape: Vec<usize>,
    /// Index specifications
    indices: Vec<IndexSpec>,
    /// Resolved indices after processing ellipsis
    resolved_indices: Vec<ResolvedIndex>,
    /// Output shape after indexing
    output_shape: Vec<usize>,
}

/// Resolved index after processing ellipsis and newaxis
#[derive(Debug, Clone)]
enum ResolvedIndex {
    Slice {
        start: usize,
        stop: usize,
        step: usize,
    },
    Index(usize),
    NewAxis,
    BoolMask(Tensor<bool>),
    IntArray(Vec<usize>),
}

impl AdvancedIndexer {
    /// Create a new advanced indexer
    pub fn new(original_shape: Vec<usize>, indices: Vec<IndexSpec>) -> Result<Self> {
        let mut indexer = Self {
            original_shape: original_shape.clone(),
            indices,
            resolved_indices: Vec::new(),
            output_shape: Vec::new(),
        };

        indexer.resolve_indices()?;
        indexer.compute_output_shape()?;

        Ok(indexer)
    }

    /// Resolve ellipsis and convert to concrete indices
    fn resolve_indices(&mut self) -> Result<()> {
        let original_ndim = self.original_shape.len();

        // Find ellipsis position if any
        let ellipsis_pos = self
            .indices
            .iter()
            .position(|idx| matches!(idx, IndexSpec::Ellipsis));

        if let Some(ellipsis_idx) = ellipsis_pos {
            // Check for multiple ellipsis (invalid)
            if self
                .indices
                .iter()
                .skip(ellipsis_idx + 1)
                .any(|idx| matches!(idx, IndexSpec::Ellipsis))
            {
                return Err(TensorError::invalid_argument(
                    "Only one ellipsis (...) is allowed per indexing operation".to_string(),
                ));
            }

            // Count non-ellipsis indices that consume dimensions
            let non_ellipsis_dims = self
                .indices
                .iter()
                .filter(|idx| !matches!(idx, IndexSpec::Ellipsis | IndexSpec::NewAxis))
                .count();

            if non_ellipsis_dims > original_ndim {
                return Err(TensorError::invalid_argument(
                    format!("Too many indices for array: array is {original_ndim}-dimensional, but {non_ellipsis_dims} were indexed")
                ));
            }

            // Calculate how many dimensions the ellipsis should represent
            let ellipsis_dims = original_ndim - non_ellipsis_dims;

            // Build resolved indices
            let mut resolved = Vec::new();

            // Add indices before ellipsis
            for idx in &self.indices[..ellipsis_idx] {
                resolved.push(self.convert_index_spec(idx.clone())?);
            }

            // Add full slices for ellipsis dimensions
            for _ in 0..ellipsis_dims {
                resolved.push(ResolvedIndex::Slice {
                    start: 0,
                    stop: usize::MAX,
                    step: 1,
                });
            }

            // Add indices after ellipsis
            for idx in &self.indices[ellipsis_idx + 1..] {
                resolved.push(self.convert_index_spec(idx.clone())?);
            }

            self.resolved_indices = resolved;
        } else {
            // No ellipsis, just convert all indices
            self.resolved_indices = self
                .indices
                .iter()
                .map(|idx| self.convert_index_spec(idx.clone()))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(())
    }

    /// Convert IndexSpec to ResolvedIndex
    fn convert_index_spec(&self, spec: IndexSpec) -> Result<ResolvedIndex> {
        match spec {
            IndexSpec::Slice { start, stop, step } => {
                let step = step.unwrap_or(1);
                if step == 0 {
                    return Err(TensorError::invalid_argument(
                        "slice step cannot be zero".to_string(),
                    ));
                }

                let start = start.unwrap_or(0);
                let stop = stop.unwrap_or(i32::MAX);

                Ok(ResolvedIndex::Slice {
                    start: start.max(0) as usize,
                    stop: if stop == i32::MAX {
                        usize::MAX
                    } else {
                        stop.max(0) as usize
                    },
                    step: step.unsigned_abs() as usize,
                })
            }
            IndexSpec::Index(idx) => {
                let resolved_idx = if idx < 0 {
                    // Negative indexing is not yet supported in this basic implementation
                    return Err(TensorError::invalid_argument(
                        "Negative indexing not yet supported".to_string(),
                    ));
                } else {
                    idx as usize
                };
                Ok(ResolvedIndex::Index(resolved_idx))
            }
            IndexSpec::Ellipsis => {
                unreachable!("Ellipsis should be handled in resolve_indices")
            }
            IndexSpec::NewAxis => Ok(ResolvedIndex::NewAxis),
            IndexSpec::BoolMask(mask) => Ok(ResolvedIndex::BoolMask(mask)),
            IndexSpec::IntArray(indices) => {
                let resolved: Vec<usize> = indices
                    .into_iter()
                    .map(|i| {
                        if i < 0 {
                            // For now, don't support negative indexing
                            Err(TensorError::invalid_argument(
                                "Negative indexing not yet supported".to_string(),
                            ))
                        } else {
                            Ok(i as usize)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ResolvedIndex::IntArray(resolved))
            }
        }
    }

    /// Compute the output shape after indexing
    fn compute_output_shape(&mut self) -> Result<()> {
        let mut shape = Vec::new();
        let mut dim_idx = 0;

        for resolved_idx in &self.resolved_indices {
            match resolved_idx {
                ResolvedIndex::Slice { start, stop, step } => {
                    if dim_idx >= self.original_shape.len() {
                        return Err(TensorError::invalid_argument(
                            "Too many indices".to_string(),
                        ));
                    }

                    let dim_size = self.original_shape[dim_idx];
                    let actual_stop = (*stop).min(dim_size);
                    let actual_start = (*start).min(dim_size);

                    if actual_start < actual_stop {
                        let slice_size = ((actual_stop - actual_start) + step - 1) / step;
                        shape.push(slice_size);
                    } else {
                        shape.push(0);
                    }
                    dim_idx += 1;
                }
                ResolvedIndex::Index(idx) => {
                    if dim_idx >= self.original_shape.len() {
                        return Err(TensorError::invalid_argument(
                            "Too many indices".to_string(),
                        ));
                    }

                    if *idx >= self.original_shape[dim_idx] {
                        return Err(TensorError::invalid_argument(format!(
                            "Index {idx} is out of bounds for dimension {dim_idx} with size {}",
                            self.original_shape[dim_idx]
                        )));
                    }
                    // Single index reduces dimensionality
                    dim_idx += 1;
                }
                ResolvedIndex::NewAxis => {
                    // NewAxis adds a dimension of size 1
                    shape.push(1);
                    // NewAxis doesn't consume a dimension from the original tensor
                }
                ResolvedIndex::BoolMask(mask) => {
                    if dim_idx >= self.original_shape.len() {
                        return Err(TensorError::invalid_argument(
                            "Too many indices".to_string(),
                        ));
                    }

                    // Boolean mask result size is the number of True values
                    let true_count = mask
                        .as_slice()
                        .ok_or_else(|| {
                            TensorError::invalid_argument("Cannot access mask data".to_string())
                        })?
                        .iter()
                        .filter(|&&val| val)
                        .count();
                    shape.push(true_count);
                    dim_idx += 1;
                }
                ResolvedIndex::IntArray(indices) => {
                    if dim_idx >= self.original_shape.len() {
                        return Err(TensorError::invalid_argument(
                            "Too many indices".to_string(),
                        ));
                    }

                    // Check bounds
                    let dim_size = self.original_shape[dim_idx];
                    for &idx in indices {
                        if idx >= dim_size {
                            return Err(TensorError::invalid_argument(
                                format!("Index {idx} is out of bounds for dimension {dim_idx} with size {dim_size}")
                            ));
                        }
                    }

                    shape.push(indices.len());
                    dim_idx += 1;
                }
            }
        }

        // Add remaining dimensions that weren't indexed
        while dim_idx < self.original_shape.len() {
            shape.push(self.original_shape[dim_idx]);
            dim_idx += 1;
        }

        self.output_shape = shape;
        Ok(())
    }

    /// Get the output shape
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Perform the indexing operation
    pub fn index<T>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // For now, this is a placeholder implementation
        // In a real implementation, this would efficiently perform the advanced indexing
        // using the resolved indices

        // Simple case: if no indexing operations, return clone
        if self.resolved_indices.is_empty() {
            return Ok(tensor.clone());
        }

        // For now, return an error indicating this needs full implementation
        Err(TensorError::not_implemented_simple(
            "Advanced indexing with ellipsis/newaxis not fully implemented yet".to_string(),
        ))
    }
}

/// Backward pass for advanced indexing with ellipsis and newaxis
pub fn ellipsis_newaxis_backward<T>(
    _grad_output: &Tensor<T>,
    original_shape: &[usize],
    indices: &[IndexSpec],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    let _indexer = AdvancedIndexer::new(original_shape.to_vec(), indices.to_vec())?;

    // Create gradient tensor with original shape, initialized to zero
    let grad_input_data = vec![T::zero(); original_shape.iter().product()];

    // For now, this is a placeholder for the backward pass
    // In a full implementation, this would scatter gradients back according to the indexing operation

    Tensor::from_vec(grad_input_data, original_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_indexer_creation() {
        let shape = vec![3, 4, 5];
        let indices = vec![
            IndexSpec::Slice {
                start: Some(0),
                stop: Some(2),
                step: None,
            },
            IndexSpec::Index(1),
        ];

        let indexer = AdvancedIndexer::new(shape, indices).unwrap();
        assert_eq!(indexer.output_shape(), &[2, 5]);
    }

    #[test]
    fn test_ellipsis_resolution() {
        let shape = vec![2, 3, 4, 5];
        let indices = vec![
            IndexSpec::Index(0),
            IndexSpec::Ellipsis,
            IndexSpec::Index(2),
        ];

        let indexer = AdvancedIndexer::new(shape, indices).unwrap();
        // Should be: index(0), slice(:), slice(:), index(2)
        // Output shape: [3, 4] (first and last dims removed by indexing)
        assert_eq!(indexer.output_shape(), &[3, 4]);
    }

    #[test]
    fn test_newaxis_insertion() {
        let shape = vec![3, 4];
        let indices = vec![
            IndexSpec::NewAxis,
            IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            },
            IndexSpec::NewAxis,
        ];

        let indexer = AdvancedIndexer::new(shape, indices).unwrap();
        // Should insert newaxis dims: [1, 3, 1, 4]
        assert_eq!(indexer.output_shape(), &[1, 3, 1, 4]);
    }

    #[test]
    fn test_multiple_ellipsis_error() {
        let shape = vec![3, 4, 5];
        let indices = vec![
            IndexSpec::Ellipsis,
            IndexSpec::Index(1),
            IndexSpec::Ellipsis,
        ];

        let result = AdvancedIndexer::new(shape, indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_bounds_index() {
        let shape = vec![3, 4];
        let indices = vec![IndexSpec::Index(5)]; // Out of bounds for first dim (size 3)

        let result = AdvancedIndexer::new(shape, indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_integer_array_indexing() {
        let shape = vec![5, 3];
        let indices = vec![IndexSpec::IntArray(vec![0, 2, 4])];

        let indexer = AdvancedIndexer::new(shape, indices).unwrap();
        assert_eq!(indexer.output_shape(), &[3, 3]); // Selected 3 elements from first dim
    }
}
