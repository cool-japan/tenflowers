//! Utility tensor manipulation operations
//!
//! This module contains utility operations for tensor manipulation including:
//! - Identity operation (tensor copying)
//! - Type casting between tensor types
//! - Padding tensors with constant values
//! - One-hot encoding for categorical data
//!
//! All operations support both CPU and GPU execution where applicable.

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{One, Zero};

/// Identity operation - returns a copy of the input tensor
///
/// This operation simply clones the tensor, which can be useful in computational graphs
/// where you need to explicitly copy data or create a new tensor node.
///
/// # Arguments
/// * `tensor` - The input tensor to copy
///
/// # Returns
/// A new tensor that is an exact copy of the input
///
/// # Example
/// ```rust,ignore
/// use tenflowers_core::ops::manipulation::utilities::identity;
/// let tensor = Tensor::zeros(&[2, 3]);
/// let copied = identity(&tensor)?;
/// assert_eq!(tensor.shape(), copied.shape());
/// ```
pub fn identity<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    // Simply clone the tensor
    Ok(tensor.clone())
}

/// Cast operation - convert tensor from one type to another
///
/// This operation converts a tensor from type T to type U using the Into trait.
/// The conversion is performed element-wise across the entire tensor.
///
/// Note: GPU casting is currently limited to specific type combinations.
/// For unsupported GPU types, the operation will fall back to CPU or return an error.
///
/// # Arguments
/// * `tensor` - The input tensor to cast
///
/// # Type Parameters
/// * `T` - Source type (must implement `Into<U>`)
/// * `U` - Target type
///
/// # Returns
/// A new tensor of type U with the same shape as the input
///
/// # Example
/// ```rust,ignore
/// use tenflowers_core::ops::manipulation::utilities::cast;
/// let tensor_f32 = Tensor::from_vec(vec![1.0f32, 2.0f32], &[2]);
/// let tensor_f64: Tensor<f64> = cast(&tensor_f32)?;
/// ```
pub fn cast<T, U>(tensor: &Tensor<T>) -> Result<Tensor<U>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + Into<U>,
    U: Clone + Default + Zero + Send + Sync + 'static,
{
    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // Map each element to the new type
            let new_array = array.mapv(|x| x.into());
            Ok(Tensor::from_array(new_array))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            // GPU operations require T,U: Pod + Zeroable which isn't guaranteed for generic types
            Err(TensorError::unsupported_operation_simple(
                "GPU cast not implemented for these types. Only f32 is currently supported."
                    .to_string(),
            ))
        }
    }
}

/// Pad operation - add padding to tensor with constant values
///
/// Pads the input tensor along each dimension with the specified padding amounts.
/// The padding is filled with a constant value.
///
/// # Arguments
/// * `tensor` - The input tensor to pad
/// * `padding` - Array of (before, after) padding amounts for each dimension
/// * `constant_value` - Value to use for padding
///
/// # Returns
/// A new tensor with padding applied
///
/// # Errors
/// Returns an error if the padding array length doesn't match the tensor rank
///
/// # Example
/// ```rust,ignore
/// use tenflowers_core::ops::manipulation::utilities::pad;
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
/// let padded = pad(&tensor, &[(1, 1), (1, 1)], 0.0)?;
/// // Result shape: [4, 4] with 0.0 padding around the original [2, 2] data
/// ```
pub fn pad<T>(
    tensor: &Tensor<T>,
    padding: &[(usize, usize)],
    constant_value: T,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    if padding.len() != tensor.shape().rank() {
        return Err(TensorError::invalid_argument(format!(
            "Padding length {} does not match tensor rank {}",
            padding.len(),
            tensor.shape().rank()
        )));
    }

    // Calculate output shape
    let mut out_shape = Vec::new();
    for (i, &dim) in tensor.shape().dims().iter().enumerate() {
        out_shape.push(dim + padding[i].0 + padding[i].1);
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // Create output tensor filled with constant value
            let mut result = ArrayD::from_elem(IxDyn(&out_shape), constant_value);

            // Use manual indexing for all dimensions to avoid the slice macro complexity
            let mut indices = vec![0; tensor.shape().rank()];
            let tensor_shape = tensor.shape().dims();

            loop {
                // Calculate padded indices
                let mut padded_indices = indices.clone();
                for i in 0..indices.len() {
                    padded_indices[i] += padding[i].0;
                }

                // Copy value
                result[IxDyn(&padded_indices)] = array[IxDyn(&indices)].clone();

                // Increment indices
                let mut carry = true;
                for i in (0..tensor_shape.len()).rev() {
                    if carry {
                        indices[i] += 1;
                        if indices[i] < tensor_shape[i] {
                            carry = false;
                        } else {
                            indices[i] = 0;
                        }
                    }
                }
                if carry {
                    break;
                }
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => gpu_pad_dispatch(tensor, padding, constant_value),
    }
}

/// One-hot encoding operation
///
/// Converts a tensor of indices to a one-hot encoded tensor. Each index in the input
/// tensor is converted to a vector where all elements are `off_value` except for the
/// element at the index position, which is set to `on_value`.
///
/// # Arguments
/// * `indices` - Tensor of integer indices to convert
/// * `depth` - Size of the one-hot dimension (number of classes)
/// * `on_value` - Value to use for the "on" position
/// * `off_value` - Value to use for all other positions
///
/// # Returns
/// A new tensor with an additional dimension of size `depth` containing one-hot vectors
///
/// # Errors
/// Returns an error if any index is out of range [0, depth)
///
/// # Example
/// ```rust,ignore
/// use tenflowers_core::ops::manipulation::utilities::one_hot;
/// let indices = Tensor::from_vec(vec![0i32, 2i32, 1i32], &[3]);
/// let encoded = one_hot(&indices, 3, 1.0f32, 0.0f32)?;
/// // Result shape: [3, 3] with one-hot encoding
/// ```
pub fn one_hot<T>(
    indices: &Tensor<i32>,
    depth: usize,
    on_value: T,
    off_value: T,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let indices_shape = indices.shape();
    let mut out_shape = indices_shape.dims().to_vec();
    out_shape.push(depth);

    match &indices.storage {
        TensorStorage::Cpu(indices_arr) => {
            let mut result = ArrayD::from_elem(IxDyn(&out_shape), off_value.clone());

            // Iterate through all indices
            let indices_flat = indices_arr
                .view()
                .into_shape_with_order((indices_arr.len(),))
                .map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to flatten indices: {e}"))
                })?;

            for (i, &idx) in indices_flat.iter().enumerate() {
                if idx < 0 || idx as usize >= depth {
                    return Err(TensorError::invalid_argument(format!(
                        "Index {idx} out of range for depth {depth}"
                    )));
                }

                // Calculate the position in the output array
                let mut out_position = Vec::with_capacity(out_shape.len());
                let mut remaining = i;
                for &dim in indices_shape.dims().iter().rev() {
                    out_position.push(remaining % dim);
                    remaining /= dim;
                }
                out_position.reverse();
                out_position.push(idx as usize);

                // Set the on_value
                result[IxDyn(&out_position)] = on_value.clone();
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => gpu_one_hot_dispatch(indices, depth, on_value, off_value),
    }
}

/// GPU dispatch function for pad operation
///
/// Handles GPU execution of padding operations. Currently only supports f32 tensors.
/// Uses unsafe transmutation to work with the GPU buffer system.
#[cfg(feature = "gpu")]
fn gpu_pad_dispatch<T>(
    tensor: &Tensor<T>,
    padding: &[(usize, usize)],
    constant_value: T,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        // Cast to f32 for the actual GPU operation
        let gpu_buffer = match &tensor.storage {
            TensorStorage::Gpu(buf) => unsafe {
                std::mem::transmute::<
                    &crate::gpu::buffer::GpuBuffer<T>,
                    &crate::gpu::buffer::GpuBuffer<f32>,
                >(buf)
            },
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let constant_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&constant_value) };

        // Calculate output shape first
        let mut out_shape = Vec::new();
        for (i, &dim) in tensor.shape().dims().iter().enumerate() {
            out_shape.push(dim + padding[i].0 + padding[i].1);
        }
        let output_len: usize = out_shape.iter().product();

        let result_buffer = crate::gpu::ops::execute_pad(
            gpu_buffer,
            padding,
            constant_f32,
            tensor.shape().dims(),
            output_len,
        )?;

        // Cast result back to T
        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(
            result_buffer_t,
            crate::Shape::from_slice(&out_shape),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU pad only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

/// GPU dispatch function for one-hot operation
///
/// Handles GPU execution of one-hot encoding. Currently only supports f32 output tensors.
/// Uses unsafe transmutation to work with the GPU buffer system.
#[cfg(feature = "gpu")]
fn gpu_one_hot_dispatch<T>(
    indices: &Tensor<i32>,
    depth: usize,
    on_value: T,
    off_value: T,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        let indices_gpu_buffer = match &indices.storage {
            TensorStorage::Gpu(buf) => buf,
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let on_value_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&on_value) };
        let off_value_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&off_value) };

        // Calculate output shape - one-hot adds depth dimension at the end
        let mut out_shape = indices.shape().dims().to_vec();
        out_shape.push(depth);
        let output_len: usize = out_shape.iter().product();

        // Cast indices buffer to u32 if needed
        let indices_gpu_buffer_u32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<i32>,
                &crate::gpu::buffer::GpuBuffer<u32>,
            >(indices_gpu_buffer)
        };

        let result_buffer = crate::gpu::ops::execute_one_hot(
            indices_gpu_buffer_u32,
            depth,
            on_value_f32,
            off_value_f32,
            -1i32, // axis = -1 (append dimension at end)
            indices.shape().dims(),
            output_len,
        )?;

        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(
            result_buffer_t,
            crate::Shape::from_slice(&out_shape),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU one_hot only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}
