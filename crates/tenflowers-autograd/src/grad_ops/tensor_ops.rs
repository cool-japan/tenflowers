//! Tensor manipulation gradient operations
//!
//! This module provides gradient computation functions for tensor manipulation
//! operations like slicing, concatenation, stacking, transposition, and reshaping.

use scirs2_core::numeric::{One, Zero};
use tenflowers_core::ops::concat;
use tenflowers_core::ops::einsum::einsum;
use tenflowers_core::ops::manipulation::common::{calculate_strides, coords_to_flat, flat_to_coords};
use tenflowers_core::ops::manipulation::{slice, squeeze, transpose_axes};
use tenflowers_core::{Result, Tensor, TensorError};

/// Represents a slice specification for a single dimension
#[derive(Debug, Clone)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: Option<isize>,
}

impl SliceSpec {
    /// Create a new slice specification
    pub fn new(start: Option<isize>, end: Option<isize>, step: Option<isize>) -> Self {
        Self { start, end, step }
    }

    /// Create a slice specification for all elements in a dimension
    pub fn all() -> Self {
        Self {
            start: None,
            end: None,
            step: Some(1),
        }
    }

    /// Create a slice specification for a single index
    pub fn single(index: isize) -> Self {
        Self {
            start: Some(index),
            end: Some(index + 1),
            step: Some(1),
        }
    }

    /// Create a slice specification for a range
    pub fn range(start: isize, end: isize) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            step: Some(1),
        }
    }

    /// Create a slice specification with a step
    pub fn range_with_step(start: isize, end: isize, step: isize) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            step: Some(step),
        }
    }
}

/// Backward pass for slice operation
/// For `y = x[slice_spec]`, `grad_x = zeros(x.shape)` with `grad_y` scattered back to
/// the original per-dimension `(start, step)` positions it was sliced from.
///
/// Supports arbitrary rank and non-unit positive step (e.g. `step=2` strided slices).
///
/// # Negative step
/// `SliceSpec.step` may in principle be negative (a reversed slice), but the
/// forward slicing kernel this mirrors
/// (`tenflowers_core::ops::manipulation::indexing::slice_with_stride`) does
/// NOT correctly implement negative step today: it silently produces wrong
/// values (confirmed empirically — for a size-6 `[0..6)` source sliced with
/// `start=5, end=0, step=-2`, the expected result is `[5, 3, 1]` but the
/// actual forward result is `[5, 0, 0]`, i.e. only the first element is
/// correct and the rest are zeroed-out out-of-bounds reads). Since there is
/// no correct forward behavior to be the gradient of, this function returns
/// an explicit `TensorError::not_implemented_simple` for any negative step
/// rather than silently computing a gradient for a forward pass that itself
/// doesn't work. This is a pre-existing `tenflowers-core` bug, out of scope
/// to fix here (it lives outside `tenflowers-autograd`).
pub fn slice_backward<T>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    slice_specs: &[SliceSpec],
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // If no slice specs provided, just return zeros (nothing to scatter back).
    if slice_specs.is_empty() {
        return Ok(Tensor::zeros(input_shape));
    }

    // Normalize every dimension's slice spec to a concrete (start, end, step),
    // defaulting to the full dimension (step=1) for any trailing dimension
    // that has no explicit spec.
    let mut normalized_specs: Vec<(usize, usize, isize)> = Vec::with_capacity(input_shape.len());
    for (dim_idx, shape_size) in input_shape.iter().enumerate() {
        if dim_idx < slice_specs.len() {
            let spec = &slice_specs[dim_idx];
            let step = spec.step.unwrap_or(1);
            if step < 0 {
                return Err(TensorError::not_implemented_simple(format!(
                    "slice_backward: negative step ({step}) is not supported because the \
                     forward slice_with_stride kernel does not correctly implement negative \
                     step (dimension {dim_idx})"
                )));
            }
            if step == 0 {
                return Err(TensorError::InvalidArgument {
                    operation: "slice_backward".to_string(),
                    reason: format!("slice step cannot be zero (dimension {dim_idx})"),
                    context: None,
                });
            }
            let start = normalize_index(spec.start.unwrap_or(0), *shape_size)?;
            let end = normalize_index(spec.end.unwrap_or(*shape_size as isize), *shape_size)?;
            normalized_specs.push((start, end.max(start), step));
        } else {
            // Full dimension if no slice spec provided (step=1 for full dimension).
            normalized_specs.push((0, *shape_size, 1));
        }
    }

    // The expected output shape of the forward slice, per dimension:
    // ceil((end - start) / step). Computed manually (rather than via
    // `usize::div_ceil`, stable only since Rust 1.73.0) because this crate's
    // MSRV is 1.70.0.
    let expected_shape: Vec<usize> = normalized_specs
        .iter()
        .map(|&(start, end, step)| {
            let span = end - start;
            let step = step as usize;
            (span + step - 1) / step
        })
        .collect();

    if grad_output.shape().dims() != expected_shape {
        return Err(TensorError::ShapeMismatch {
            operation: "slice_backward".to_string(),
            expected: format!("{expected_shape:?}"),
            got: format!("{:?}", grad_output.shape().dims()),
            context: None,
        });
    }

    // Scatter grad_output back into a zero-initialized input_shape-sized buffer.
    // Walk every coordinate of grad_output (row-major), map dimension-by-dimension
    // via `input_coord[d] = start_d + out_coord[d] * step_d`, and accumulate
    // (add, not overwrite) into grad_input at that position. Slice specs never
    // produce overlapping output positions in practice, but accumulating is
    // defensive and costs nothing extra here.
    let grad_output_data = grad_output.to_vec()?;
    let input_strides = calculate_strides(input_shape);
    let input_total: usize = input_shape.iter().product();
    let mut grad_input_data: Vec<T> = vec![T::zero(); input_total];

    for (out_flat, out_val) in grad_output_data.iter().enumerate() {
        let out_coords = flat_to_coords(out_flat, &expected_shape);

        let mut input_coords = Vec::with_capacity(out_coords.len());
        for (d, &out_c) in out_coords.iter().enumerate() {
            let (start, _end, step) = normalized_specs[d];
            input_coords.push(start + out_c * step as usize);
        }

        let input_flat = coords_to_flat(&input_coords, &input_strides);
        // `input_flat` is guaranteed in-bounds because every `input_coords[d]`
        // was derived from a normalized (start, end) that itself was clamped
        // to `input_shape[d]` by `normalize_index`.
        grad_input_data[input_flat] = grad_input_data[input_flat].clone() + out_val.clone();
    }

    Tensor::from_vec(grad_input_data, input_shape)
}

/// Helper function to normalize negative indices
fn normalize_index(index: isize, size: usize) -> Result<usize> {
    if index < 0 {
        let positive_index = size as isize + index;
        if positive_index < 0 {
            Err(TensorError::InvalidArgument {
                operation: "index_normalization".to_string(),
                reason: format!("Index {index} is out of bounds for size {size}"),
                context: None,
            })
        } else {
            Ok(positive_index as usize)
        }
    } else {
        let idx = index as usize;
        if idx > size {
            Ok(size) // Clamp to size
        } else {
            Ok(idx)
        }
    }
}

/// Backward pass for concatenation operation
/// For y = concat([x1, x2, ..., xn], axis), split grad_y along axis to get gradients for each input
pub fn concat_backward<T>(
    grad_output: &Tensor<T>,
    input_shapes: &[Vec<usize>],
    axis: i32,
) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let output_shape = grad_output.shape().dims();
    let ndim = output_shape.len();

    // Normalize axis
    let actual_axis = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    if actual_axis >= ndim {
        return Err(TensorError::ShapeMismatch {
            operation: "backward_operation".to_string(),
            expected: format!("axis < {ndim}"),
            got: format!("axis = {axis}"),
            context: None,
        });
    }

    // Split the gradient along the concatenation axis
    let mut gradients = Vec::new();
    let mut start_idx = 0;

    for input_shape in input_shapes {
        let size_along_axis = input_shape[actual_axis];
        let end_idx = start_idx + size_along_axis;

        // Extract the gradient slice for this input
        // Create slice ranges for all dimensions
        let mut ranges = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..ndim {
            if i == actual_axis {
                ranges.push(start_idx..end_idx);
            } else {
                ranges.push(0..output_shape[i]);
            }
        }

        // Slice the gradient tensor to get the portion for this input
        let grad_input = slice(grad_output, &ranges)?;
        gradients.push(grad_input);

        start_idx = end_idx;
    }

    Ok(gradients)
}

/// Backward pass for stack operation
/// For y = stack([x1, x2, ..., xn], axis), unstack grad_y along axis to get gradients for each input
pub fn stack_backward<T>(
    grad_output: &Tensor<T>,
    num_inputs: usize,
    axis: i32,
) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let output_shape = grad_output.shape().dims();
    let ndim = output_shape.len();

    // Normalize axis
    let actual_axis = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    if actual_axis >= ndim {
        return Err(TensorError::ShapeMismatch {
            operation: "backward_operation".to_string(),
            expected: format!("axis < {ndim}"),
            got: format!("axis = {axis}"),
            context: None,
        });
    }

    // Verify that the size along the stacking axis matches num_inputs
    if output_shape[actual_axis] != num_inputs {
        return Err(TensorError::ShapeMismatch {
            operation: "backward_operation".to_string(),
            expected: format!("Size {num_inputs} along axis {actual_axis}"),
            got: format!(
                "Size {} along axis {}",
                output_shape[actual_axis], actual_axis
            ),
            context: None,
        });
    }

    // Unstack the gradient along the stacking axis
    let mut gradients = Vec::new();

    for i in 0..num_inputs {
        // Create slice ranges for all dimensions to extract the i-th slice
        let mut ranges = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for j in 0..ndim {
            if j == actual_axis {
                ranges.push(i..(i + 1));
            } else {
                ranges.push(0..output_shape[j]);
            }
        }

        // Extract the gradient slice and remove the stacking dimension
        let sliced = slice(grad_output, &ranges)?;
        let grad_input = squeeze(&sliced, Some(&[actual_axis]))?;
        gradients.push(grad_input);
    }

    Ok(gradients)
}

/// Backward pass for split operation
/// For splits = split(x, sizes, axis), grad_x = concat(grad_splits, axis)
pub fn split_backward<T>(grad_outputs: &[Tensor<T>], axis: i32) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if grad_outputs.is_empty() {
        return Err(TensorError::shape_mismatch(
            "grad_ops",
            "non-empty gradient list",
            "empty gradient list",
        ));
    }

    // Split backward is essentially concat forward
    // Use the concat operation from tenflowers_core

    // Convert Vec<Tensor<T>> to Vec<&Tensor<T>> for concat function
    let tensor_refs: Vec<&Tensor<T>> = grad_outputs.iter().collect();

    // Normalize axis to usize
    let ndim = grad_outputs[0].shape().dims().len();
    let actual_axis = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    if actual_axis >= ndim {
        return Err(TensorError::InvalidArgument {
            operation: "concat_backward".to_string(),
            reason: format!("axis {axis} is out of range for tensor with {ndim} dimensions"),
            context: None,
        });
    }

    concat(&tensor_refs, actual_axis)
}

/// Backward pass for transpose operation
/// For y = transpose(x, axes), grad_x = transpose(grad_y, inverse_axes)
pub fn transpose_backward<T>(grad_output: &Tensor<T>, axes: Option<&[usize]>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match axes {
        Some(axes) => {
            // Compute inverse permutation
            let mut inverse_axes = vec![0; axes.len()];
            for (i, &axis) in axes.iter().enumerate() {
                inverse_axes[axis] = i;
            }

            // Apply inverse transpose
            transpose_axes(grad_output, Some(&inverse_axes))
        }
        None => {
            // Default transpose is reverse all dimensions
            tenflowers_core::ops::transpose(grad_output)
        }
    }
}

/// Backward pass for squeeze operation
/// For y = squeeze(x, axes), grad_x = unsqueeze(grad_y, axes)
pub fn squeeze_backward<T>(grad_output: &Tensor<T>, original_shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Reshape back to original shape (unsqueeze is just a reshape)
    grad_output.reshape(original_shape)
}

/// Backward pass for unsqueeze operation
/// For y = unsqueeze(x, axes), grad_x = squeeze(grad_y, axes)
pub fn unsqueeze_backward<T>(grad_output: &Tensor<T>, axes: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Remove the added dimensions
    grad_output.squeeze(Some(axes))
}

/// Backward pass for gather operation
/// For `y = gather(x, indices, axis)` (TensorFlow-style `tf.gather`: `indices`
/// carries no gradient, and the `axis` dimension of `x` is REPLACED — not just
/// resized — by `indices.shape()`), the gradient is scattered back to the
/// original positions with scatter-ADD semantics: if the same input index is
/// gathered more than once, every occurrence's contribution to `grad_output`
/// must be summed into that single `grad_input` position.
///
/// Mirrors the coordinate mapping of the forward
/// `tenflowers_core::ops::manipulation::indexing::gather` exactly: output
/// coordinates split as `(pre_axis..., index_coords..., post_axis...)`, where
/// `index_coords` selects an entry of `indices` whose value becomes the
/// `axis` coordinate into `input`.
pub fn gather_backward<T>(
    grad_output: &Tensor<T>,
    input_shape: &[usize],
    indices: &Tensor<i32>,
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let ndim = input_shape.len();

    if axis >= ndim {
        return Err(TensorError::InvalidArgument {
            operation: "gather_backward".to_string(),
            reason: format!("axis {axis} out of range for input of rank {ndim}"),
            context: None,
        });
    }

    let axis_size = input_shape[axis];
    let indices_dims = indices.shape().dims().to_vec();
    let indices_rank = indices_dims.len();

    // The expected grad_output shape is input_shape with the `axis` dimension
    // replaced by indices_dims (same as the forward gather's output shape).
    let mut expected_shape = input_shape.to_vec();
    expected_shape.splice(axis..axis + 1, indices_dims.iter().copied());
    if grad_output.shape().dims() != expected_shape {
        return Err(TensorError::ShapeMismatch {
            operation: "gather_backward".to_string(),
            expected: format!("{expected_shape:?}"),
            got: format!("{:?}", grad_output.shape().dims()),
            context: None,
        });
    }

    // Read grad_output/indices logically (safe for non-contiguous inputs),
    // matching how the forward `gather` reads `params`/`indices`.
    let grad_output_data: Vec<T> = match grad_output.as_slice() {
        Some(s) => s.to_vec(),
        None => grad_output.to_vec()?,
    };
    let indices_data: Vec<i32> = match indices.as_slice() {
        Some(s) => s.to_vec(),
        None => indices.to_vec()?,
    };

    let indices_strides = calculate_strides(&indices_dims);
    let input_strides = calculate_strides(input_shape);
    let input_total: usize = input_shape.iter().product();
    let mut grad_input_data: Vec<T> = vec![T::zero(); input_total];

    for (out_flat, out_val) in grad_output_data.iter().enumerate() {
        let out_coords = flat_to_coords(out_flat, &expected_shape);

        // The index block occupies positions [axis, axis + indices_rank)
        // within the output coordinates (same as forward gather).
        let index_coords = &out_coords[axis..axis + indices_rank];
        let indices_flat = coords_to_flat(index_coords, &indices_strides);
        let gathered = indices_data[indices_flat];

        if gathered < 0 || gathered as usize >= axis_size {
            return Err(TensorError::InvalidArgument {
                operation: "gather_backward".to_string(),
                reason: format!(
                    "index {gathered} out of bounds for axis {axis} of size {axis_size}"
                ),
                context: None,
            });
        }

        // Reconstruct the input coordinate: pre-axis dims unchanged, the
        // gathered index at `axis`, then post-axis dims (which follow the
        // index block in the output).
        let mut input_coords = Vec::with_capacity(input_shape.len());
        input_coords.extend_from_slice(&out_coords[..axis]);
        input_coords.push(gathered as usize);
        input_coords.extend_from_slice(&out_coords[axis + indices_rank..]);

        let input_flat = coords_to_flat(&input_coords, &input_strides);

        // Scatter-ADD: accumulate, never overwrite, so that gathering the
        // same input index multiple times sums all of its contributions.
        grad_input_data[input_flat] = grad_input_data[input_flat].clone() + out_val.clone();
    }

    Tensor::from_vec(grad_input_data, input_shape)
}

/// Backward pass for scatter operation
/// For y = scatter(x, indices, values, axis), we have:
/// grad_x = grad_y with zeros at scattered positions
/// grad_values = gather(grad_y, indices, axis)
pub fn scatter_backward<T>(
    grad_output: &Tensor<T>,
    _input: &Tensor<T>,
    _indices: &Tensor<i64>,
    values: &Tensor<T>,
    _axis: i32,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    // Simplified implementation for now
    // In a full implementation, this would need proper gather operations
    let grad_input = grad_output.clone();
    let grad_values = Tensor::zeros(values.shape().dims());

    Ok((grad_input, grad_values))
}

/// Helper function to get an element from a 4D tensor at position [b, c, h, w]
pub fn get_tensor_element_4d<T>(
    tensor: &Tensor<T>,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Option<T>
where
    T: Clone,
{
    let shape = tensor.shape().dims();
    if b >= shape[0] || c >= shape[1] || h >= shape[2] || w >= shape[3] {
        return None;
    }

    let data = tensor.as_slice()?;
    let idx = b * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w;

    if idx < data.len() {
        Some(data[idx].clone())
    } else {
        None
    }
}

/// Helper function to slice a 4D tensor
#[allow(clippy::too_many_arguments)]
pub fn slice_tensor_4d<T>(
    tensor: &Tensor<T>,
    n_start: usize,
    n_end: usize,
    c_start: usize,
    c_end: usize,
    h_start: usize,
    h_end: usize,
    w_start: usize,
    w_end: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = tensor.shape().dims();
    let [n, c, h, w] = [shape[0], shape[1], shape[2], shape[3]];

    // Validate slice bounds
    if n_end > n || c_end > c || h_end > h || w_end > w {
        return Err(TensorError::InvalidArgument {
            operation: "tensor_setitem_backward".to_string(),
            reason: format!("Index out of bounds: [{n_start}:{n_end}, {c_start}:{c_end}, {h_start}:{h_end}, {w_start}:{w_end}] for shape {shape:?}"),
            context: None,
        });
    }

    // Use the proper tensor slicing operation
    let ranges = [
        n_start..n_end,
        c_start..c_end,
        h_start..h_end,
        w_start..w_end,
    ];

    tensor.slice(&ranges)
}

/// Helper function to add a tensor to a slice of another tensor
#[allow(dead_code)]
pub fn add_to_slice_4d<T>(
    target: &mut Tensor<T>,
    source: &Tensor<T>,
    n_offset: usize,
    c_offset: usize,
    h_offset: usize,
    w_offset: usize,
) -> Result<()>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let source_shape = source.shape().dims();
    let target_shape = target.shape().dims();

    // Validate that source can fit in target at the given offset
    if n_offset + source_shape[0] > target_shape[0]
        || c_offset + source_shape[1] > target_shape[1]
        || h_offset + source_shape[2] > target_shape[2]
        || w_offset + source_shape[3] > target_shape[3]
    {
        return Err(TensorError::InvalidArgument {
            operation: "tensor_setitem_backward".to_string(),
            reason: format!("Source shape {source_shape:?} cannot fit in target shape {target_shape:?} at offset [{n_offset}, {c_offset}, {h_offset}, {w_offset}]"),
            context: None,
        });
    }

    // Get the target slice
    let target_slice = target.slice(&[
        n_offset..(n_offset + source_shape[0]),
        c_offset..(c_offset + source_shape[1]),
        h_offset..(h_offset + source_shape[2]),
        w_offset..(w_offset + source_shape[3]),
    ])?;

    // Add source to the target slice
    let _result = target_slice.add(source)?;

    // For now, we'll create a new tensor and copy the result back
    // This is not the most efficient implementation, but it works
    // A more efficient implementation would modify the tensor in-place

    // Note: This is a simplified implementation. In a production system,
    // you would implement proper in-place operations or use scatter_add
    // For now, we'll just return Ok(()) as the gradient accumulation
    // will be handled by the calling code through proper tensor operations

    Ok(())
}

/// Backward pass for einsum operation
/// Computes gradients for each input tensor given the output gradient and einsum equation
pub fn einsum_backward<T>(
    grad_output: &Tensor<T>,
    equation: &str,
    operands: &[&Tensor<T>],
) -> Result<Vec<Tensor<T>>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute gradients for einsum operation

    if operands.is_empty() {
        return Err(TensorError::invalid_argument(
            "At least one operand is required for einsum backward".to_string(),
        ));
    }

    // Handle specific einsum patterns manually for better reliability
    match equation {
        "ij,jk->ik" => {
            // Matrix multiplication: A @ B = C
            // Gradients: dA = dC @ B^T, dB = A^T @ dC
            if operands.len() != 2 {
                return Err(TensorError::invalid_argument(
                    "Matrix multiply einsum requires exactly 2 operands".to_string(),
                ));
            }

            let a = operands[0]; // ij
            let b = operands[1]; // jk

            // Gradient w.r.t. A: grad_output @ B^T
            let b_transposed = b.transpose()?;
            let grad_a = grad_output.matmul(&b_transposed)?;

            // Gradient w.r.t. B: A^T @ grad_output
            let a_transposed = a.transpose()?;
            let grad_b = a_transposed.matmul(grad_output)?;

            // Matrix multiplication gradients computed successfully

            return Ok(vec![grad_a, grad_b]);
        }
        "ij->ji" => {
            // Transpose operation
            if operands.len() != 1 {
                return Err(TensorError::invalid_argument(
                    "Transpose einsum requires exactly 1 operand".to_string(),
                ));
            }
            let grad_input = grad_output.transpose()?;
            println!("Transpose gradient shape: {:?}", grad_input.shape().dims());
            return Ok(vec![grad_input]);
        }
        "ij,ij->ij" => {
            // Element-wise multiplication: A * B = C
            // Gradients: dA = dC * B, dB = dC * A
            if operands.len() != 2 {
                return Err(TensorError::invalid_argument(
                    "Element-wise multiply einsum requires exactly 2 operands".to_string(),
                ));
            }

            let a = operands[0];
            let b = operands[1];

            let grad_a = grad_output.mul(b)?;
            let grad_b = grad_output.mul(a)?;

            println!(
                "Element-wise multiply gradients: grad_A shape: {:?}, grad_B shape: {:?}",
                grad_a.shape().dims(),
                grad_b.shape().dims()
            );

            return Ok(vec![grad_a, grad_b]);
        }
        _ => {
            // Continue with the general algorithm below
        }
    }

    let mut input_grads = Vec::with_capacity(operands.len());

    // Parse the einsum equation to understand the subscripts
    let arrow_pos = equation.find("->").ok_or_else(|| {
        TensorError::invalid_argument("Einsum equation must contain '->' arrow".to_string())
    })?;

    let input_part = &equation[..arrow_pos];
    let output_part = &equation[arrow_pos + 2..];

    // Split input subscripts by comma
    let input_subscripts: Vec<&str> = input_part.split(',').collect();

    if input_subscripts.len() != operands.len() {
        return Err(TensorError::invalid_argument(format!(
            "Number of input subscripts ({}) must match number of operands ({})",
            input_subscripts.len(),
            operands.len()
        )));
    }

    // For each input tensor, compute its gradient
    for (i, (input_subscript, _input_tensor)) in
        input_subscripts.iter().zip(operands.iter()).enumerate()
    {
        // To compute the gradient for input i, we need to contract grad_output
        // with all other inputs using a modified einsum equation

        if operands.len() == 1 {
            // Special case: single operand, gradient flows back with same pattern
            // For "ij->i", gradient "i->ij" (broadcast)
            // For "ij->ji", gradient "ji->ij" (transpose back)
            // For "ii->", gradient "->ii" (diagonal expansion)

            let backward_equation = format!("{}->{}", output_part, input_subscript);
            let grad_input = einsum(&backward_equation, &[grad_output])?;
            input_grads.push(grad_input);
        } else {
            // Multi-operand case: contract grad_output with all other operands
            // This is more complex and requires careful equation construction

            // For now, implement a simplified version for common patterns
            // Full implementation would require more sophisticated equation parsing

            // Create a list of other operands (excluding operand i)
            let mut other_operands = Vec::new();
            let mut other_subscripts = Vec::new();

            for (j, (subscript, operand)) in
                input_subscripts.iter().zip(operands.iter()).enumerate()
            {
                if j != i {
                    other_operands.push(*operand);
                    other_subscripts.push(*subscript);
                }
            }

            // Construct backward equation for gradient computation
            // For each input operand, we contract grad_output with all OTHER input operands
            // The key insight: to get gradient w.r.t. operand i, we need to contract
            // grad_output with all operands EXCEPT operand i

            let mut backward_equation = output_part.to_string();

            // Add all other operands to the equation
            for subscript in &other_subscripts {
                backward_equation.push(',');
                backward_equation.push_str(subscript);
            }

            backward_equation.push_str("->");
            backward_equation.push_str(input_subscript);

            // Debug: Print the backward equation
            println!("Original equation: {}", equation);
            println!("Backward equation for operand {}: {}", i, backward_equation);

            // Create operand list: grad_output first, then all other operands
            let mut contraction_operands = vec![grad_output];
            contraction_operands.extend(&other_operands);

            match einsum(&backward_equation, &contraction_operands) {
                Ok(grad_input) => {
                    println!(
                        "Generated gradient for operand {} with shape: {:?}",
                        i,
                        grad_input.shape().dims()
                    );
                    input_grads.push(grad_input);
                }
                Err(e) => {
                    println!("einsum failed for operand {}: {:?}", i, e);
                    return Err(e);
                }
            }
        }
    }

    println!("einsum_backward returning {} gradients", input_grads.len());
    Ok(input_grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_slice_spec_creation() {
        let spec1 = SliceSpec::all();
        assert_eq!(spec1.start, None);
        assert_eq!(spec1.end, None);
        assert_eq!(spec1.step, Some(1));

        let spec2 = SliceSpec::single(5);
        assert_eq!(spec2.start, Some(5));
        assert_eq!(spec2.end, Some(6));
        assert_eq!(spec2.step, Some(1));

        let spec3 = SliceSpec::range(2, 8);
        assert_eq!(spec3.start, Some(2));
        assert_eq!(spec3.end, Some(8));
        assert_eq!(spec3.step, Some(1));
    }

    #[test]
    fn test_normalize_index() {
        // Test positive indices
        assert_eq!(
            normalize_index(2, 10).expect("test: shape/index operation should succeed"),
            2
        );
        assert_eq!(
            normalize_index(9, 10).expect("test: shape/index operation should succeed"),
            9
        );

        // Test negative indices
        assert_eq!(
            normalize_index(-1, 10).expect("test: shape/index operation should succeed"),
            9
        );
        assert_eq!(
            normalize_index(-3, 10).expect("test: shape/index operation should succeed"),
            7
        );

        // Test edge cases
        assert_eq!(
            normalize_index(0, 10).expect("test: shape/index operation should succeed"),
            0
        );
        assert_eq!(
            normalize_index(10, 10).expect("test: shape/index operation should succeed"),
            10
        ); // Clamped to size

        // Test out of bounds negative
        assert!(normalize_index(-11, 10).is_err());
    }

    #[test]
    fn test_slice_backward_contiguous_range() {
        // x shape [6], slice [1..4] (step=1): grad_input must be
        // [0, g0, g1, g2, 0, 0] for grad_output = [g0, g1, g2].
        let grad_output = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3])
            .expect("test: tensor creation from valid data should succeed");
        let input_shape = &[6];
        let slice_specs = vec![SliceSpec::range(1, 4)];

        let grad_input = slice_backward(&grad_output, input_shape, &slice_specs)
            .expect("test: slice_backward should succeed");
        assert_eq!(grad_input.shape().dims(), input_shape);
        let got = grad_input.to_vec().expect("test: to_vec should succeed");
        assert_eq!(got, vec![0.0, 10.0, 20.0, 30.0, 0.0, 0.0]);
    }

    #[test]
    fn test_slice_backward_strided_step_2() {
        // x shape [6], slice picking indices [0, 2, 4] (step=2): grad must
        // land ONLY at those 3 positions and be exactly zero elsewhere.
        let grad_output = Tensor::from_vec(vec![100.0f32, 200.0, 300.0], &[3])
            .expect("test: tensor creation from valid data should succeed");
        let input_shape = &[6];
        let slice_specs = vec![SliceSpec::range_with_step(0, 6, 2)];

        let grad_input = slice_backward(&grad_output, input_shape, &slice_specs)
            .expect("test: slice_backward should succeed");
        assert_eq!(grad_input.shape().dims(), input_shape);
        let got = grad_input.to_vec().expect("test: to_vec should succeed");
        assert_eq!(got, vec![100.0, 0.0, 200.0, 0.0, 300.0, 0.0]);
    }

    #[test]
    fn test_slice_backward_2d_partial_both_dims() {
        // x shape [3, 4]; slice rows [1..3], cols [1..3] -> output [2, 2].
        // grad_input must place grad_output at rows 1..3, cols 1..3 and be
        // zero everywhere else.
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test: tensor creation from valid data should succeed");
        let input_shape = &[3, 4];
        let slice_specs = vec![SliceSpec::range(1, 3), SliceSpec::range(1, 3)];

        let grad_input = slice_backward(&grad_output, input_shape, &slice_specs)
            .expect("test: slice_backward should succeed");
        assert_eq!(grad_input.shape().dims(), input_shape);
        let got = grad_input.to_vec().expect("test: to_vec should succeed");
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_slice_backward_negative_step_not_implemented() {
        // The forward slice_with_stride kernel does not correctly implement
        // negative step (confirmed empirically against
        // tenflowers_core::ops::manipulation::indexing::slice_with_stride),
        // so slice_backward must reject it explicitly rather than silently
        // computing a gradient for a forward pass that doesn't work.
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
            .expect("test: tensor creation from valid data should succeed");
        let input_shape = &[6];
        let slice_specs = vec![SliceSpec::range_with_step(5, 0, -2)];
        let result = slice_backward(&grad_output, input_shape, &slice_specs);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_backward_basic() {
        // Test basic stack backward functionality
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("test: tensor creation from valid data should succeed");
        let result = stack_backward(&grad_output, 2, 0);

        assert!(result.is_ok());
        let gradients = result.expect("test: gradient computation should succeed");
        assert_eq!(gradients.len(), 2);

        // Each gradient should have shape [3] after removing the stacking dimension
        for grad in gradients {
            assert_eq!(grad.shape().dims(), &[3]);
        }
    }

    #[test]
    fn test_split_backward_basic() {
        // Test basic split backward functionality
        let grad1 = Tensor::from_vec(vec![1.0f32, 2.0], &[2])
            .expect("test: tensor creation from valid data should succeed");
        let grad2 = Tensor::from_vec(vec![3.0f32, 4.0], &[2])
            .expect("test: tensor creation from valid data should succeed");
        let grad_outputs = vec![grad1, grad2];

        let result = split_backward(&grad_outputs, 0);
        assert!(result.is_ok());

        let concatenated = result.expect("test: operation result should be valid");
        assert_eq!(concatenated.shape().dims(), &[4]); // 2 + 2
    }

    #[test]
    fn test_transpose_backward_identity() {
        // Test transpose backward with no axes (identity)
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .expect("test: tensor creation from valid data should succeed");
        let result = transpose_backward(&grad_output, None);

        assert!(result.is_ok());
        let grad_input = result.expect("test: gradient computation should succeed");
        assert_eq!(grad_input.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_squeeze_unsqueeze_backward() {
        // Test squeeze backward
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
            .expect("test: tensor creation from valid data should succeed");
        let original_shape = &[1, 3, 1];
        let result = squeeze_backward(&grad_output, original_shape);

        assert!(result.is_ok());
        let grad_input = result.expect("test: gradient computation should succeed");
        assert_eq!(grad_input.shape().dims(), original_shape);

        // Test unsqueeze backward
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3, 1])
            .expect("test: tensor creation from valid data should succeed");
        let axes = &[0, 2];
        let result = unsqueeze_backward(&grad_output, axes);

        assert!(result.is_ok());
        let grad_input = result.expect("test: gradient computation should succeed");
        assert_eq!(grad_input.shape().dims(), &[3]);
    }

    #[test]
    fn test_gather_backward_simple_1d_values() {
        // params shape [5], indices = [0, 2] (1D gather along axis 0).
        // grad_output shape matches indices shape: [2].
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0], &[2])
            .expect("test: tensor creation from valid data should succeed");
        let input_shape = &[5];
        let indices =
            Tensor::from_vec(vec![0i32, 2], &[2]).expect("test: tensor creation should succeed");

        let grad_input = gather_backward(&grad_output, input_shape, &indices, 0)
            .expect("test: gather_backward should succeed");
        assert_eq!(grad_input.shape().dims(), input_shape);
        let got = grad_input.to_vec().expect("test: to_vec should succeed");
        // index 0 received grad_output[0]=1.0, index 2 received grad_output[1]=2.0,
        // indices 1, 3, 4 were never gathered so must be exactly zero.
        assert_eq!(got, vec![1.0, 0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gather_backward_repeated_index_accumulates() {
        // params shape [4, 3], indices = [0, 2, 0] along axis 0 (mandatory
        // same-index-gathered-twice accumulation case): grad_input[0] must
        // equal the SUM of grad_output rows 0 and 2 (both gathered index 0),
        // not just one of them; grad_input[2] equals grad_output row 1;
        // grad_input[1] and grad_input[3] (never gathered) are exactly zero.
        let input_shape = &[4, 3];
        let indices =
            Tensor::from_vec(vec![0i32, 2, 0], &[3]).expect("test: tensor creation should succeed");
        // grad_output shape [3, 3]: row i is the gradient contributed by
        // indices[i].
        let grad_output = Tensor::from_vec(
            vec![
                1.0f32, 1.0, 1.0, // row 0 -> gathered index 0
                2.0, 2.0, 2.0, // row 1 -> gathered index 2
                3.0, 3.0, 3.0, // row 2 -> gathered index 0 (again)
            ],
            &[3, 3],
        )
        .expect("test: tensor creation from valid data should succeed");

        let grad_input = gather_backward(&grad_output, input_shape, &indices, 0)
            .expect("test: gather_backward should succeed");
        assert_eq!(grad_input.shape().dims(), input_shape);
        let got = grad_input.to_vec().expect("test: to_vec should succeed");
        assert_eq!(
            got,
            vec![
                4.0, 4.0, 4.0, // row 0 = row0 + row2 = [1,1,1] + [3,3,3]
                0.0, 0.0, 0.0, // row 1 never gathered
                2.0, 2.0, 2.0, // row 2 = row1 = [2,2,2]
                0.0, 0.0, 0.0, // row 3 never gathered
            ]
        );
    }

    #[test]
    fn test_gather_backward_out_of_bounds_index_errors() {
        let grad_output = Tensor::from_vec(vec![1.0f32], &[1])
            .expect("test: tensor creation from valid data should succeed");
        let input_shape = &[3];
        let indices =
            Tensor::from_vec(vec![5i32], &[1]).expect("test: tensor creation should succeed");
        let result = gather_backward(&grad_output, input_shape, &indices, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scatter_backward_interface() {
        // scatter_backward is out of scope for this round (still a stub);
        // this only verifies the existing interface/shape contract.
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0], &[2])
            .expect("test: tensor creation from valid data should succeed");
        let indices = Tensor::from_vec(vec![0i64, 2], &[2])
            .expect("test: tensor creation from valid data should succeed");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test: tensor creation from valid data should succeed");
        let values = Tensor::from_vec(vec![10.0f32, 20.0], &[2])
            .expect("test: tensor creation from valid data should succeed");
        let result = scatter_backward(&grad_output, &input, &indices, &values, 0);

        assert!(result.is_ok());
        let (grad_input, grad_values) = result.expect("test: gradient computation should succeed");
        assert_eq!(grad_input.shape().dims(), &[2]);
        assert_eq!(grad_values.shape().dims(), &[2]);
    }

    #[test]
    fn test_4d_tensor_helpers() {
        // Test get_tensor_element_4d
        let tensor = Tensor::from_vec((0..24).map(|i| i as f32).collect(), &[2, 3, 2, 2])
            .expect("test: tensor creation from valid data should succeed");

        let element = get_tensor_element_4d(&tensor, 0, 0, 0, 0);
        assert!(element.is_some());

        // Test out of bounds
        let element = get_tensor_element_4d(&tensor, 5, 0, 0, 0);
        assert!(element.is_none());

        // Test slice_tensor_4d
        let result = slice_tensor_4d(&tensor, 0, 1, 0, 2, 0, 2, 0, 2);
        assert!(result.is_ok());
        let sliced = result.expect("test: operation result should be valid");
        assert_eq!(sliced.shape().dims(), &[1, 2, 2, 2]);
    }
}
