use crate::ops::utils::normalize_index;
use scirs2_core::numeric::{One, Zero};
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

            // Build resolved indices, tracking a running dimension-index counter so
            // Index/IntArray conversion can look up the correct original dimension
            // size for negative-index normalization. NewAxis entries insert a new
            // dimension and therefore do not consume (advance) dim_idx.
            let mut resolved = Vec::new();
            let mut dim_idx = 0usize;

            // Add indices before ellipsis
            for idx in &self.indices[..ellipsis_idx] {
                let consumes_dim = !matches!(idx, IndexSpec::NewAxis);
                resolved.push(self.convert_index_spec(idx.clone(), dim_idx)?);
                if consumes_dim {
                    dim_idx += 1;
                }
            }

            // Add full slices for ellipsis dimensions
            for _ in 0..ellipsis_dims {
                resolved.push(ResolvedIndex::Slice {
                    start: 0,
                    stop: usize::MAX,
                    step: 1,
                });
                dim_idx += 1;
            }

            // Add indices after ellipsis
            for idx in &self.indices[ellipsis_idx + 1..] {
                let consumes_dim = !matches!(idx, IndexSpec::NewAxis);
                resolved.push(self.convert_index_spec(idx.clone(), dim_idx)?);
                if consumes_dim {
                    dim_idx += 1;
                }
            }

            self.resolved_indices = resolved;
        } else {
            // No ellipsis: convert all indices in order, tracking dim_idx the same
            // way as above so negative-index normalization sees the correct
            // original dimension size.
            let mut resolved = Vec::with_capacity(self.indices.len());
            let mut dim_idx = 0usize;
            for idx in &self.indices {
                let consumes_dim = !matches!(idx, IndexSpec::NewAxis);
                resolved.push(self.convert_index_spec(idx.clone(), dim_idx)?);
                if consumes_dim {
                    dim_idx += 1;
                }
            }
            self.resolved_indices = resolved;
        }

        Ok(())
    }

    /// Convert IndexSpec to ResolvedIndex
    ///
    /// `dim_idx` is the index (into `self.original_shape`) of the original
    /// dimension this spec applies to. It is used to resolve negative indices
    /// (Python-style: `-1` means the last element along that dimension) via
    /// `normalize_index`. `NewAxis` specs do not consume an original dimension,
    /// so callers must not advance `dim_idx` for them (see `resolve_indices`).
    fn convert_index_spec(&self, spec: IndexSpec, dim_idx: usize) -> Result<ResolvedIndex> {
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
                let dim_size = *self.original_shape.get(dim_idx).ok_or_else(|| {
                    TensorError::invalid_argument(format!(
                        "Too many indices for array: array is {}-dimensional, but an index was given for dimension {dim_idx}",
                        self.original_shape.len()
                    ))
                })?;
                let resolved_idx = normalize_index(idx as isize, dim_size)?;
                Ok(ResolvedIndex::Index(resolved_idx))
            }
            IndexSpec::Ellipsis => {
                // An ellipsis spans zero-or-more dimensions and can only be
                // expanded with knowledge of the full index list and the source
                // rank (done in `resolve_indices`). `convert_index_spec` handles a
                // single index in isolation and has no such context, so it cannot
                // turn an ellipsis into concrete `ResolvedIndex` values. Reaching
                // here means an un-expanded ellipsis was passed in directly;
                // `IndexSpec::Ellipsis` is publicly constructible, so we return an
                // honest, recoverable error instead of panicking via `unreachable!`.
                Err(TensorError::invalid_argument(
                    "ellipsis (...) must be expanded against the tensor rank before \
                     conversion; pass index specifications through AdvancedIndexer so the \
                     ellipsis is resolved to concrete slices first"
                        .to_string(),
                ))
            }
            IndexSpec::NewAxis => Ok(ResolvedIndex::NewAxis),
            IndexSpec::BoolMask(mask) => Ok(ResolvedIndex::BoolMask(mask)),
            IndexSpec::IntArray(indices) => {
                let dim_size = *self.original_shape.get(dim_idx).ok_or_else(|| {
                    TensorError::invalid_argument(format!(
                        "Too many indices for array: array is {}-dimensional, but an index array was given for dimension {dim_idx}",
                        self.original_shape.len()
                    ))
                })?;
                let resolved: Vec<usize> = indices
                    .into_iter()
                    .map(|i| normalize_index(i as isize, dim_size))
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

        let indexer =
            AdvancedIndexer::new(shape, indices).expect("test: construction should succeed");
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

        let indexer =
            AdvancedIndexer::new(shape, indices).expect("test: construction should succeed");
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

        let indexer =
            AdvancedIndexer::new(shape, indices).expect("test: construction should succeed");
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

        let indexer =
            AdvancedIndexer::new(shape, indices).expect("test: construction should succeed");
        assert_eq!(indexer.output_shape(), &[3, 3]); // Selected 3 elements from first dim
    }

    #[test]
    fn test_convert_index_spec_ellipsis_returns_error_not_panic() {
        // A stray, un-expanded ellipsis passed directly to convert_index_spec must
        // produce an honest, recoverable error rather than panicking via
        // unreachable!(). IndexSpec::Ellipsis is publicly constructible.
        let indexer = AdvancedIndexer::new(
            vec![3, 4],
            vec![IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            }],
        )
        .expect("test: construction should succeed");

        let result = indexer.convert_index_spec(IndexSpec::Ellipsis, 0);
        assert!(
            result.is_err(),
            "an un-expanded ellipsis must yield an honest Err, not a panic"
        );
    }

    #[test]
    fn test_ellipsis_resolution_still_succeeds_through_public_api() {
        // The normal path (ellipsis expanded inside resolve_indices) must remain
        // unaffected by the convert_index_spec hardening.
        let indexer = AdvancedIndexer::new(
            vec![2, 3, 4, 5],
            vec![
                IndexSpec::Index(0),
                IndexSpec::Ellipsis,
                IndexSpec::Index(2),
            ],
        )
        .expect("test: ellipsis should resolve through resolve_indices");
        assert_eq!(indexer.output_shape(), &[3, 4]);
    }

    #[test]
    fn test_convert_index_spec_negative_index_boundary_last_element() {
        // shape [5]; -1 must resolve to the last valid index, 4.
        let indexer = AdvancedIndexer::new(
            vec![5],
            vec![IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            }],
        )
        .expect("test: construction should succeed");

        let resolved = indexer
            .convert_index_spec(IndexSpec::Index(-1), 0)
            .expect("test: -1 should normalize to the last element");
        match resolved {
            ResolvedIndex::Index(v) => assert_eq!(v, 4),
            other => panic!("expected ResolvedIndex::Index(4), got {other:?}"),
        }
    }

    #[test]
    fn test_convert_index_spec_negative_index_wraps_to_first_element() {
        // shape [5]; -5 must resolve to index 0.
        let indexer = AdvancedIndexer::new(
            vec![5],
            vec![IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            }],
        )
        .expect("test: construction should succeed");

        let resolved = indexer
            .convert_index_spec(IndexSpec::Index(-5), 0)
            .expect("test: -5 on a size-5 dim should normalize to 0");
        match resolved {
            ResolvedIndex::Index(v) => assert_eq!(v, 0),
            other => panic!("expected ResolvedIndex::Index(0), got {other:?}"),
        }
    }

    #[test]
    fn test_convert_index_spec_negative_index_out_of_range_errors() {
        // shape [5]; -6 has no valid positive mapping (5 + (-6) = -1 < 0) -> Err, not a panic.
        let indexer = AdvancedIndexer::new(
            vec![5],
            vec![IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            }],
        )
        .expect("test: construction should succeed");

        let result = indexer.convert_index_spec(IndexSpec::Index(-6), 0);
        assert!(
            result.is_err(),
            "-6 on a size-5 dimension must be an honest Err"
        );
    }

    #[test]
    fn test_negative_index_through_public_api_various_dim_sizes() {
        // shape [7, 5]; dim 0 (size 7) fully sliced, dim 1 (size 5) indexed with -1 -> last element.
        let indexer = AdvancedIndexer::new(
            vec![7, 5],
            vec![
                IndexSpec::Slice {
                    start: None,
                    stop: None,
                    step: None,
                },
                IndexSpec::Index(-1),
            ],
        )
        .expect("test: negative index should resolve through the public constructor");
        assert_eq!(indexer.output_shape(), &[7]);
    }

    #[test]
    fn test_negative_index_array_various_dim_sizes() {
        // shape [5]; IntArray([-1, -2, 0]) -> [4, 3, 0].
        let indexer = AdvancedIndexer::new(
            vec![5],
            vec![IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            }],
        )
        .expect("test: construction should succeed");

        let resolved = indexer
            .convert_index_spec(IndexSpec::IntArray(vec![-1, -2, 0]), 0)
            .expect("test: negative indices in IntArray should normalize");
        match resolved {
            ResolvedIndex::IntArray(values) => assert_eq!(values, vec![4, 3, 0]),
            other => panic!("expected ResolvedIndex::IntArray([4,3,0]), got {other:?}"),
        }
    }

    #[test]
    fn test_negative_index_array_preserves_per_element_errors() {
        // shape [5]; IntArray([-1, -10]): -1 is valid (-> 4), but -10 has no valid mapping.
        let indexer = AdvancedIndexer::new(
            vec![5],
            vec![IndexSpec::Slice {
                start: None,
                stop: None,
                step: None,
            }],
        )
        .expect("test: construction should succeed");

        let result = indexer.convert_index_spec(IndexSpec::IntArray(vec![-1, -10]), 0);
        assert!(
            result.is_err(),
            "an out-of-range element anywhere in the array must fail the whole conversion"
        );
    }

    #[test]
    fn test_negative_index_array_through_public_api() {
        // shape [6]; select indices [-1, -3, 2] -> output shape [3].
        let indexer = AdvancedIndexer::new(vec![6], vec![IndexSpec::IntArray(vec![-1, -3, 2])])
            .expect(
                "test: negative IntArray indices should resolve through the public constructor",
            );
        assert_eq!(indexer.output_shape(), &[3]);
    }
}
