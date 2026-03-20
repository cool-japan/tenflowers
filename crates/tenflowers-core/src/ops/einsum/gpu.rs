//! GPU-Accelerated Operations for Einstein Summation
//!
//! This module contains GPU-optimized implementations for einsum operations
//! using compute shaders and GPU kernels when the gpu feature is enabled.

#[allow(unused_imports)]
use crate::TensorError;
use crate::{Result, Tensor};
use scirs2_core::numeric::{One, Zero};

// GPU einsum implementations
#[cfg(feature = "gpu")]
pub fn gpu_einsum_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_matmul;
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            // Ensure matrices are 2D for standard matmul
            if a.shape().dims().len() != 2 || b.shape().dims().len() != 2 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum matrix multiplication requires 2D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            // Check dimensions are compatible for matmul
            if a_shape[1] != b_shape[0] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_matmul_gpu".to_string(),
                    expected: format!("({}, K) and (K, {})", a_shape[0], b_shape[1]),
                    got: format!(
                        "({}, {}) and ({}, {})",
                        a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                    ),
                    context: None,
                });
            }

            let output_shape = crate::Shape::new(vec![a_shape[0], b_shape[1]]);

            // Execute GPU einsum matrix multiplication
            let result_buffer = execute_einsum_matmul(
                gpu_a,
                gpu_b,
                &a_shape,
                &b_shape,
                output_shape.dims().iter().product::<usize>(),
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            crate::ops::matmul(a, b)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_batched_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_batched_matmul;
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            // Ensure matrices are 3D for batched matmul (batch, height, width)
            if a.shape().dims().len() != 3 || b.shape().dims().len() != 3 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum batched matrix multiplication requires 3D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            // Check dimensions are compatible for batched matmul
            if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_batch_matmul_gpu".to_string(),
                    expected: "(B, M, K) and (B, K, N)".to_string(),
                    got: format!(
                        "({}, {}, {}) and ({}, {}, {})",
                        a_shape[0], a_shape[1], a_shape[2], b_shape[0], b_shape[1], b_shape[2]
                    ),
                    context: None,
                });
            }

            let output_shape = crate::Shape::new(vec![a_shape[0], a_shape[1], b_shape[2]]);

            // Execute GPU einsum batched matrix multiplication
            let result_buffer = execute_einsum_batched_matmul(
                gpu_a,
                gpu_b,
                &a_shape,
                &b_shape,
                output_shape.dims().iter().product::<usize>(),
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            crate::ops::matmul(a, b)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_transpose<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_transpose;
    use crate::tensor::TensorStorage;

    match &tensor.storage {
        TensorStorage::Gpu(gpu_tensor) => {
            // Ensure tensor is 2D for transpose
            if tensor.shape().dims().len() != 2 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum transpose requires 2D tensor".to_string(),
                ));
            }

            let input_shape = tensor.shape().dims();
            let output_shape = crate::Shape::new(vec![input_shape[1], input_shape[0]]);

            // Execute GPU einsum transpose
            let result_buffer = execute_einsum_transpose(gpu_tensor, &[1, 0])?; // Basic 2D transpose

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            tensor.transpose()
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_diagonal<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_diagonal;
    use crate::tensor::TensorStorage;

    match &tensor.storage {
        TensorStorage::Gpu(gpu_tensor) => {
            // Ensure tensor is 2D for diagonal extraction
            if tensor.shape().dims().len() != 2 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum diagonal requires 2D tensor".to_string(),
                ));
            }

            let input_shape = tensor.shape().dims();
            let min_dim = input_shape[0].min(input_shape[1]);
            let output_shape = crate::Shape::new(vec![min_dim]);

            // Execute GPU einsum diagonal
            let output_len = output_shape.dims().iter().product();
            let result_buffer =
                execute_einsum_diagonal(gpu_tensor, &input_shape, &[0, 1], output_len)?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            Err(TensorError::unsupported_operation_simple(
                "GPU diagonal extraction fallback not implemented yet".to_string(),
            ))
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_outer_product<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_outer_product;
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            // Ensure tensors are 1D for outer product
            if a.shape().dims().len() != 1 || b.shape().dims().len() != 1 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum outer product requires 1D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();
            let output_shape = crate::Shape::new(vec![a_shape[0], b_shape[0]]);

            // Execute GPU einsum outer product
            let result_buffer = execute_einsum_outer_product(
                gpu_a,
                gpu_b,
                &a_shape,
                &b_shape,
                output_shape.dims().iter().product::<usize>(),
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            Err(TensorError::unsupported_operation_simple(
                "GPU outer product fallback not implemented yet".to_string(),
            ))
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_vector_dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_vector_dot;
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            // Ensure tensors are 1D for vector dot product
            if a.shape().dims().len() != 1 || b.shape().dims().len() != 1 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum vector dot product requires 1D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            // Check dimensions are compatible for dot product
            if a_shape[0] != b_shape[0] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_dot_gpu".to_string(),
                    expected: format!("({},) and ({},)", a_shape[0], a_shape[0]),
                    got: format!("({},) and ({},)", a_shape[0], b_shape[0]),
                    context: None,
                });
            }

            let output_shape = crate::Shape::new(vec![]); // Scalar result

            // Execute GPU einsum vector dot product
            let result_buffer = execute_einsum_vector_dot(
                gpu_a,
                gpu_b,
                &a_shape,
                &b_shape,
                output_shape.dims().iter().product::<usize>(),
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            crate::ops::dot(a, b)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_trace<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
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
    use crate::gpu::ops::execute_einsum_trace;
    use crate::tensor::TensorStorage;

    match &tensor.storage {
        TensorStorage::Gpu(gpu_tensor) => {
            // Ensure tensor is 2D for trace calculation
            if tensor.shape().dims().len() != 2 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum trace requires 2D tensor".to_string(),
                ));
            }

            let input_shape = tensor.shape().dims();

            // Check if it's a square matrix
            if input_shape[0] != input_shape[1] {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum trace requires square matrix".to_string(),
                ));
            }

            let output_shape = crate::Shape::new(vec![]); // Scalar result

            // Execute GPU einsum trace
            let result_buffer = execute_einsum_trace(gpu_tensor, &[0, 1])?; // Trace over diagonal

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            Err(TensorError::unsupported_operation_simple(
                "GPU trace extraction fallback not implemented yet".to_string(),
            ))
        }
    }
}

// Fallback implementations when GPU is not available
#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_matmul<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_batched_matmul<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_transpose<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_diagonal<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_outer_product<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_vector_dot<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_trace<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("GPU functions should not be called when GPU is not available")
}
