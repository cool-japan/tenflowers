use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Backward pass for boolean mask indexing
/// For y = x[mask], where mask is a boolean tensor, gradient flows only to positions where mask is true
pub fn boolean_mask_backward<T>(
    grad_output: &Tensor<T>,
    mask: &Tensor<bool>,
    input_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    // Will create grad_input later

    // Get the mask data and gradient data
    let mask_data = mask
        .as_slice()
        .ok_or_else(|| TensorError::invalid_argument("Could not access mask data".to_string()))?;

    let grad_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access gradient data".to_string())
    })?;

    // Check that mask and input have compatible shapes
    if mask.shape().dims() != input_shape {
        return Err(TensorError::shape_mismatch(
            "boolean_indexing",
            &format!("{input_shape:?}"),
            &format!("{:?}", mask.shape().dims()),
        ));
    }

    // Count true values in mask to verify gradient size
    let true_count = mask_data.iter().filter(|&&val| val).count();
    if grad_data.len() != true_count {
        return Err(TensorError::shape_mismatch(
            "boolean_indexing",
            &format!("gradient size {true_count}"),
            &format!("gradient size {}", grad_data.len()),
        ));
    }

    // Scatter gradients back to positions where mask is true
    // Create the data vector manually since we don't have mutable access
    let mut result_data = vec![T::zero(); input_shape.iter().product()];
    let mut grad_idx = 0;
    for (i, &mask_val) in mask_data.iter().enumerate() {
        if mask_val && grad_idx < grad_data.len() {
            result_data[i] = grad_data[grad_idx].clone();
            grad_idx += 1;
        }
    }

    let grad_input = Tensor::from_vec(result_data, input_shape)?;

    Ok(grad_input)
}

/// Backward pass for advanced boolean indexing with multiple dimensions
pub fn boolean_mask_nd_backward<T>(
    grad_output: &Tensor<T>,
    mask: &Tensor<bool>,
    input_shape: &[usize],
    mask_dims: &[usize], // Which dimensions the mask applies to
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    // For advanced boolean indexing in multiple dimensions
    // This is more complex and would require careful index calculation

    // For now, implement the simple case where mask applies to all dimensions
    if mask_dims.len() == input_shape.len() {
        return boolean_mask_backward(grad_output, mask, input_shape);
    }

    // Implement N-dimensional boolean indexing
    ndimensional_boolean_indexing_backward(grad_output, mask, input_shape)
}

/// N-dimensional boolean indexing backward pass
/// Handles complex multi-dimensional boolean masks with advanced indexing patterns
fn ndimensional_boolean_indexing_backward<T>(
    grad_output: &Tensor<T>,
    mask: &Tensor<bool>,
    input_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    let mask_dims = mask.shape().dims();
    let grad_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access gradient data".to_string())
    })?;

    // Initialize result tensor with zeros
    let mut result_data = vec![T::zero(); input_shape.iter().product()];

    // Handle different dimensionality cases
    if mask_dims.len() <= input_shape.len() {
        // Case 1: Mask dimensions are less than or equal to input dimensions
        // Apply mask to the leading dimensions
        let mask_data = mask.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Could not access mask data".to_string())
        })?;

        let _mask_size = mask_dims.iter().product::<usize>();
        let trailing_size = input_shape[mask_dims.len()..].iter().product::<usize>();

        let mut grad_idx = 0;
        for (mask_idx, &mask_val) in mask_data.iter().enumerate() {
            if mask_val {
                // Calculate the starting index in the result tensor
                let start_idx = mask_idx * trailing_size;

                // Copy gradient values for all trailing dimensions
                for i in 0..trailing_size {
                    if grad_idx < grad_data.len() {
                        result_data[start_idx + i] = grad_data[grad_idx].clone();
                        grad_idx += 1;
                    }
                }
            }
        }
    } else {
        // Case 2: Mask has more dimensions than input
        // This is a more complex case where we need to handle broadcasting
        return advanced_boolean_indexing_backward(grad_output, mask, input_shape);
    }

    Tensor::from_vec(result_data, input_shape)
}

/// Advanced boolean indexing for cases where mask has more dimensions than input
fn advanced_boolean_indexing_backward<T>(
    grad_output: &Tensor<T>,
    mask: &Tensor<bool>,
    input_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    let mask_dims = mask.shape().dims();
    let grad_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access gradient data".to_string())
    })?;

    let mask_data = mask
        .as_slice()
        .ok_or_else(|| TensorError::invalid_argument("Could not access mask data".to_string()))?;

    // Initialize result tensor with zeros
    let mut result_data = vec![T::zero(); input_shape.iter().product()];

    // Calculate strides for efficient indexing
    let input_strides = calculate_strides(input_shape);
    let mask_strides = calculate_strides(mask_dims);

    let mut grad_idx = 0;

    // Iterate through mask indices
    for (flat_mask_idx, &mask_val) in mask_data.iter().enumerate() {
        if mask_val {
            // Convert flat mask index to multi-dimensional indices
            let mask_indices = flat_to_multi_index(flat_mask_idx, &mask_strides);

            // Map mask indices to input indices (handle broadcasting)
            let input_indices = broadcast_indices(&mask_indices, mask_dims, input_shape)?;

            // Convert multi-dimensional input indices to flat index
            let flat_input_idx = multi_to_flat_index(&input_indices, &input_strides);

            // Set gradient value
            if grad_idx < grad_data.len() && flat_input_idx < result_data.len() {
                result_data[flat_input_idx] = grad_data[grad_idx].clone();
                grad_idx += 1;
            }
        }
    }

    Tensor::from_vec(result_data, input_shape)
}

/// Calculate strides for multi-dimensional indexing
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert flat index to multi-dimensional indices
fn flat_to_multi_index(flat_idx: usize, strides: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; strides.len()];
    let mut remaining = flat_idx;

    for (i, &stride) in strides.iter().enumerate() {
        indices[i] = remaining / stride;
        remaining %= stride;
    }

    indices
}

/// Convert multi-dimensional indices to flat index
fn multi_to_flat_index(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(idx, stride)| idx * stride)
        .sum()
}

/// Handle broadcasting of indices from mask dimensions to input dimensions
fn broadcast_indices(
    mask_indices: &[usize],
    mask_dims: &[usize],
    input_shape: &[usize],
) -> Result<Vec<usize>> {
    let mut input_indices = vec![0; input_shape.len()];

    // Handle different broadcasting cases
    if mask_dims.len() <= input_shape.len() {
        // Mask dimensions fit within input dimensions
        for (i, &mask_idx) in mask_indices.iter().enumerate() {
            if i < input_shape.len() {
                input_indices[i] = mask_idx;
            }
        }
    } else {
        // Mask has more dimensions - use last dimensions of input
        let offset = mask_dims.len() - input_shape.len();
        for (i, &mask_idx) in mask_indices.iter().enumerate() {
            if i >= offset {
                input_indices[i - offset] = mask_idx;
            }
        }
    }

    // Validate indices are within bounds
    for (i, (&idx, &dim)) in input_indices.iter().zip(input_shape.iter()).enumerate() {
        if idx >= dim {
            return Err(TensorError::invalid_argument(format!(
                "Index {idx} out of bounds for dimension {i} with size {dim}"
            )));
        }
    }

    Ok(input_indices)
}

/// Multi-dimensional integer array indexing backward pass
/// Handles complex integer array indexing patterns with multiple dimensions
fn multidimensional_integer_indexing_backward<T>(
    grad_output: &Tensor<T>,
    indices: &Tensor<i64>,
    input_shape: &[usize],
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let indices_data = indices.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access indices data".to_string())
    })?;

    let grad_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access gradient data".to_string())
    })?;

    if axis >= input_shape.len() {
        return Err(TensorError::invalid_argument(format!(
            "Axis {} out of range for input with {} dimensions",
            axis,
            input_shape.len()
        )));
    }

    // Initialize result tensor with zeros
    let mut result_data = vec![T::zero(); input_shape.iter().product()];

    // Handle different indexing patterns
    let indices_shape = indices.shape().dims();

    // Calculate strides for efficient indexing
    let input_strides = calculate_strides(input_shape);
    let indices_strides = calculate_strides(indices_shape);

    // Process each index
    for (flat_idx, &index) in indices_data.iter().enumerate() {
        if index < 0 || index as usize >= input_shape[axis] {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for axis {} with size {}",
                index, axis, input_shape[axis]
            )));
        }

        // Convert flat index to multi-dimensional indices for the indices tensor
        let multi_indices = flat_to_multi_index(flat_idx, &indices_strides);

        // Calculate the corresponding position in the input tensor
        let mut input_position = vec![0; input_shape.len()];

        // Handle different indexing modes
        if indices_shape.len() == 1 {
            // Simple 1D indexing
            input_position[axis] = index as usize;

            // For other dimensions, use the same indexing pattern
            for i in 0..input_shape.len() {
                if i != axis {
                    input_position[i] = if i < multi_indices.len() {
                        multi_indices[i]
                    } else {
                        0
                    };
                }
            }
        } else {
            // Multi-dimensional indexing
            input_position[axis] = index as usize;

            // Map remaining dimensions
            let mut pos_idx = 0;
            for (i, position) in input_position
                .iter_mut()
                .enumerate()
                .take(input_shape.len())
            {
                if i != axis && pos_idx < multi_indices.len() {
                    *position = multi_indices[pos_idx];
                    pos_idx += 1;
                }
            }
        }

        // Validate all indices are within bounds
        for (i, (&pos, &dim)) in input_position.iter().zip(input_shape.iter()).enumerate() {
            if pos >= dim {
                return Err(TensorError::invalid_argument(format!(
                    "Computed index {pos} out of bounds for dimension {i} with size {dim}"
                )));
            }
        }

        // Convert multi-dimensional position to flat index
        let flat_input_idx = multi_to_flat_index(&input_position, &input_strides);

        // Accumulate gradient (in case of duplicate indices)
        if flat_idx < grad_data.len() && flat_input_idx < result_data.len() {
            result_data[flat_input_idx] =
                result_data[flat_input_idx].clone() + grad_data[flat_idx].clone();
        }
    }

    Tensor::from_vec(result_data, input_shape)
}

/// Backward pass for where operation (torch.where equivalent)
/// For y = where(condition, x, z), gradients flow to x where condition is true, to z where false
pub fn where_backward<T>(
    grad_output: &Tensor<T>,
    condition: &Tensor<bool>,
    x_shape: &[usize],
    z_shape: &[usize],
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let condition_data = condition.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access condition data".to_string())
    })?;

    let grad_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access gradient data".to_string())
    })?;

    // Will initialize gradients later

    // Handle broadcasting - for simplicity, assume same shape for now
    if condition.shape().dims() == x_shape && x_shape == z_shape {
        let mut grad_x_data = vec![T::zero(); x_shape.iter().product()];
        let mut grad_z_data = vec![T::zero(); z_shape.iter().product()];

        for (i, (&cond, grad)) in condition_data.iter().zip(grad_data.iter()).enumerate() {
            if cond {
                grad_x_data[i] = grad.clone();
            } else {
                grad_z_data[i] = grad.clone();
            }
        }

        let grad_x = Tensor::from_vec(grad_x_data, x_shape)?;
        let grad_z = Tensor::from_vec(grad_z_data, z_shape)?;

        Ok((grad_x, grad_z))
    } else {
        // Handle broadcasting case - would need to implement proper broadcasting logic
        Err(TensorError::unsupported_operation_simple(
            "Broadcasting in where operation not yet implemented".to_string(),
        ))
    }
}

/// Backward pass for integer array indexing
/// For y = x[indices], gradient flows back to the indexed positions
pub fn integer_array_indexing_backward<T>(
    grad_output: &Tensor<T>,
    indices: &Tensor<i64>,
    input_shape: &[usize],
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let indices_data = indices.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access indices data".to_string())
    })?;

    let grad_data = grad_output.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Could not access gradient data".to_string())
    })?;

    // Will create grad_input later

    // For simplicity, handle 1D case first
    if input_shape.len() == 1 && axis == 0 {
        let mut input_grad_data = vec![T::zero(); input_shape[0]];
        for (grad_idx, &index) in indices_data.iter().enumerate() {
            if grad_idx < grad_data.len() {
                let idx = index as usize;
                if idx < input_shape[0] {
                    // Add gradient (handle duplicate indices)
                    input_grad_data[idx] =
                        input_grad_data[idx].clone() + grad_data[grad_idx].clone();
                }
            }
        }
        let grad_input = Tensor::from_vec(input_grad_data, input_shape)?;
        Ok(grad_input)
    } else {
        // Implement multi-dimensional integer array indexing
        multidimensional_integer_indexing_backward(grad_output, indices, input_shape, axis)
    }
}

/// Advanced indexing with mixed boolean and integer indices
pub fn mixed_indexing_backward<T>(
    grad_output: &Tensor<T>,
    boolean_mask: Option<&Tensor<bool>>,
    integer_indices: Option<&Tensor<i64>>,
    input_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    match (boolean_mask, integer_indices) {
        (Some(mask), None) => boolean_mask_backward(grad_output, mask, input_shape),
        (None, Some(indices)) => {
            integer_array_indexing_backward(grad_output, indices, input_shape, 0)
        }
        (Some(mask), Some(indices)) => {
            // Implement combination of boolean and integer indexing
            mixed_boolean_integer_indexing_backward(grad_output, mask, indices, input_shape)
        }
        (None, None) => {
            // No indexing, return identity
            Ok(grad_output.clone())
        }
    }
}

/// Mixed boolean and integer indexing backward pass
/// Handles combination of boolean mask and integer indices
fn mixed_boolean_integer_indexing_backward<T>(
    grad_output: &Tensor<T>,
    boolean_mask: &Tensor<bool>,
    integer_indices: &Tensor<i64>,
    input_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    // This is a simplified implementation
    // In practice, we would need to carefully handle the interaction between boolean and integer indexing

    // For now, apply boolean indexing first, then integer indexing
    let intermediate_result = boolean_mask_backward(grad_output, boolean_mask, input_shape)?;

    // Then apply integer indexing (simplified approach)
    // In a full implementation, we would need to consider the exact semantics of mixed indexing
    integer_array_indexing_backward(&intermediate_result, integer_indices, input_shape, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_boolean_mask_backward() {
        // Create input shape [5]
        let input_shape = [5];

        // Create mask [true, false, true, false, true]
        let mask = Tensor::from_vec(vec![true, false, true, false, true], &[5]).unwrap();

        // Create gradient output for 3 selected elements
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();

        // Compute backward pass
        let grad_input = boolean_mask_backward(&grad_output, &mask, &input_shape).unwrap();

        // Check result
        if let Some(data) = grad_input.as_slice() {
            assert_eq!(data, &[1.0, 0.0, 2.0, 0.0, 3.0]);
        } else {
            panic!("Could not access gradient data");
        }
    }

    #[test]
    fn test_where_backward() {
        // Create condition [true, false, true]
        let condition = Tensor::from_vec(vec![true, false, true], &[3]).unwrap();

        // Create gradient output
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();

        // Shapes for x and z
        let x_shape = [3];
        let z_shape = [3];

        // Compute backward pass
        let (grad_x, grad_z) =
            where_backward(&grad_output, &condition, &x_shape, &z_shape).unwrap();

        // Check results
        if let (Some(x_data), Some(z_data)) = (grad_x.as_slice(), grad_z.as_slice()) {
            assert_eq!(x_data, &[1.0, 0.0, 3.0]); // Gradient flows to x where condition is true
            assert_eq!(z_data, &[0.0, 2.0, 0.0]); // Gradient flows to z where condition is false
        } else {
            panic!("Could not access gradient data");
        }
    }

    #[test]
    fn test_integer_array_indexing_backward() {
        // Create indices [0, 2, 1, 2] - note duplicate index 2
        let indices = Tensor::from_vec(vec![0i64, 2, 1, 2], &[4]).unwrap();

        // Create gradient output
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Input shape [3]
        let input_shape = [3];

        // Compute backward pass
        let grad_input =
            integer_array_indexing_backward(&grad_output, &indices, &input_shape, 0).unwrap();

        // Check result - index 2 should accumulate gradients 2.0 + 4.0 = 6.0
        if let Some(data) = grad_input.as_slice() {
            assert_eq!(data, &[1.0, 3.0, 6.0]);
        } else {
            panic!("Could not access gradient data");
        }
    }
}
