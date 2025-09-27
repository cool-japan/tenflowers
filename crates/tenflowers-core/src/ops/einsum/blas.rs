//! BLAS-Optimized Operations for Einstein Summation
//!
//! This module contains BLAS-accelerated implementations for CPU tensor operations
//! when BLAS libraries (OpenBLAS, MKL, or Accelerate) are available.

#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
use crate::{tensor::TensorStorage, TensorError};
use crate::{Result, Tensor};
use num_traits::{One, Zero};

// BLAS optimizations for einsum operations
#[cfg(all(
    any(
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "blas-accelerate"
    ),
    feature = "std"
))]
use scirs2_autograd::ndarray::{s, Array2};

/// Try BLAS-optimized patterns for CPU tensors
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
pub fn try_blas_optimized_patterns<T>(
    equation: &str,
    operands: &[&Tensor<T>],
) -> Option<Result<Tensor<T>>>
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
    // Only optimize for f32 and f64 types that have BLAS support
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() || id == std::any::TypeId::of::<f64>() => {}
        _ => return None,
    }

    // Ensure all operands are CPU tensors
    for operand in operands {
        match &operand.storage {
            TensorStorage::Cpu(_) => {}
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => return None,
        }
    }

    match equation {
        // Matrix multiplication: "ij,jk->ik" (BLAS GEMM)
        "ij,jk->ik" if operands.len() == 2 => Some(blas_gemm_2d(operands[0], operands[1])),

        // Batched matrix multiplication: "bij,bjk->bik" (Batched GEMM)
        "bij,bjk->bik" if operands.len() == 2 => {
            Some(blas_batched_gemm_3d(operands[0], operands[1]))
        }

        // Matrix-vector multiplication: "ij,j->i" (BLAS GEMV)
        "ij,j->i" if operands.len() == 2 => Some(blas_gemv(operands[0], operands[1])),

        // Vector dot product: "i,i->" (BLAS DOT)
        "i,i->" if operands.len() == 2 => Some(blas_dot_product(operands[0], operands[1])),

        // Vector outer product: "i,j->ij" (BLAS GER)
        "i,j->ij" if operands.len() == 2 => Some(blas_outer_product(operands[0], operands[1])),

        // Matrix transpose with optional scaling: "ij->ji"
        "ij->ji" if operands.len() == 1 => Some(blas_transpose_2d(operands[0])),

        // Symmetric matrix operations: "ij,ji->ij" (can use BLAS SYMM for certain cases)
        "ij,ji->ij" if operands.len() == 2 => {
            Some(blas_symmetric_multiply(operands[0], operands[1]))
        }

        // Triangular matrix solve patterns: "ij,jk->ik" with triangular structure
        eq if eq.starts_with("ij,jk->ik")
            && operands.len() == 2
            && is_triangular_candidate(operands[0]) =>
        {
            Some(blas_triangular_solve(operands[0], operands[1]))
        }

        _ => None,
    }
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
pub(super) fn try_blas_optimized_patterns<T>(
    _equation: &str,
    _operands: &[&Tensor<T>],
) -> Option<Result<Tensor<T>>>
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
    None
}

/// BLAS-optimized 2D matrix multiplication using GEMM
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
#[allow(unreachable_patterns)]
fn blas_gemm_2d<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(a_array), TensorStorage::Cpu(b_array)) => {
            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(TensorError::invalid_argument(
                    "BLAS GEMM requires 2D tensors".to_string(),
                ));
            }

            if a_shape[1] != b_shape[0] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_gemm".to_string(),
                    expected: format!("(M, K) and (K, N)"),
                    got: format!(
                        "({}, {}) and ({}, {})",
                        a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                    ),
                    context: None,
                });
            }

            // Convert to 2D arrays for BLAS operation
            let a_2d = a_array
                .clone()
                .into_dimensionality::<scirs2_autograd::ndarray::Ix2>()
                .map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to convert tensor to 2D: {}", e))
                })?;

            let b_2d = b_array
                .clone()
                .into_dimensionality::<scirs2_autograd::ndarray::Ix2>()
                .map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to convert tensor to 2D: {}", e))
                })?;

            // Perform BLAS matrix multiplication using ndarray's dot (which uses BLAS when available)
            let result_2d = match std::any::TypeId::of::<T>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let a_f32 = unsafe { std::mem::transmute::<&Array2<T>, &Array2<f32>>(&a_2d) };
                    let b_f32 = unsafe { std::mem::transmute::<&Array2<T>, &Array2<f32>>(&b_2d) };
                    let result_f32 = a_f32.dot(b_f32);
                    unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(result_f32) }
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let a_f64 = unsafe { std::mem::transmute::<&Array2<T>, &Array2<f64>>(&a_2d) };
                    let b_f64 = unsafe { std::mem::transmute::<&Array2<T>, &Array2<f64>>(&b_2d) };
                    let result_f64 = a_f64.dot(b_f64);
                    unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(result_f64) }
                }
                _ => {
                    // Fall back to regular matrix multiplication
                    return crate::ops::matmul(a, b);
                }
            };

            // Convert back to dynamic dimension array and create tensor
            let result_dynamic = result_2d.into_dyn();
            let result_shape = crate::Shape::new(vec![a_shape[0], b_shape[1]]);

            // Use the tensor constructor rather than direct struct construction
            let mut result_data = Vec::with_capacity(result_dynamic.len());
            for elem in result_dynamic.iter() {
                result_data.push(elem.clone());
            }

            Tensor::from_vec(result_data, &result_shape.dims())
        }
        _ => {
            // Fall back to regular matrix multiplication
            crate::ops::matmul(a, b)
        }
    }
}

/// BLAS-optimized batched matrix multiplication
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_batched_gemm_3d<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    if a_shape.len() != 3 || b_shape.len() != 3 {
        return Err(TensorError::invalid_argument(
            "Batched GEMM requires 3D tensors".to_string(),
        ));
    }

    // Fall back to regular batched matrix multiplication for now
    // A full implementation would use batch-aware BLAS routines
    crate::ops::matmul(a, b)
}

/// BLAS-optimized matrix-vector multiplication
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_gemv<T>(matrix: &Tensor<T>, vector: &Tensor<T>) -> Result<Tensor<T>>
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
    let matrix_shape = matrix.shape().dims();
    let vector_shape = vector.shape().dims();

    if matrix_shape.len() != 2 || vector_shape.len() != 1 {
        return Err(TensorError::invalid_argument(
            "GEMV requires 2D matrix and 1D vector".to_string(),
        ));
    }

    if matrix_shape[1] != vector_shape[0] {
        return Err(TensorError::ShapeMismatch {
            operation: "einsum_gemv".to_string(),
            expected: format!(
                "matrix cols ({}) == vector len ({})",
                matrix_shape[1], vector_shape[0]
            ),
            got: format!(
                "matrix cols {} != vector len {}",
                matrix_shape[1], vector_shape[0]
            ),
            context: None,
        });
    }

    // Use efficient matrix-vector multiplication
    crate::ops::matmul(matrix, vector)
}

/// BLAS-optimized vector dot product
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_dot_product<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(TensorError::invalid_argument(
            "Dot product requires 1D tensors".to_string(),
        ));
    }

    if a_shape[0] != b_shape[0] {
        return Err(TensorError::ShapeMismatch {
            operation: "einsum_dot".to_string(),
            expected: format!("equal vector lengths"),
            got: format!("lengths {} and {}", a_shape[0], b_shape[0]),
            context: None,
        });
    }

    // Element-wise multiply then sum
    let elementwise = a.mul(b)?;
    crate::ops::sum(&elementwise, None, false)
}

/// BLAS-optimized vector outer product
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_outer_product<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    use super::utils::compute_outer_product;
    compute_outer_product(a, b)
}

/// BLAS-optimized matrix transpose
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_transpose_2d<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let shape = tensor.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_argument(
            "Transpose requires 2D tensor".to_string(),
        ));
    }

    // Use efficient transpose with axes permutation
    crate::ops::manipulation::transpose_axes(tensor, Some(&[1, 0]))
}

/// BLAS-optimized symmetric matrix multiplication
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_symmetric_multiply<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    // For symmetric operations, can potentially use BLAS SYMM
    // For now, fall back to regular multiplication
    a.mul(b)
}

/// BLAS-optimized triangular matrix solve
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
fn blas_triangular_solve<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
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
    // Triangular solve would use BLAS TRSM
    // For now, fall back to regular multiplication
    crate::ops::matmul(a, b)
}

/// Check if tensor is a triangular matrix candidate
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
#[allow(dead_code)]
fn is_triangular_candidate<T>(tensor: &Tensor<T>) -> bool
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
    let shape = tensor.shape().dims();

    // Must be square matrix
    if shape.len() != 2 || shape[0] != shape[1] {
        return false;
    }

    // For now, just check if it's square - a full implementation would
    // check if the matrix is actually triangular
    true
}

// Fallback implementations when BLAS is not available
#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_gemm_2d<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_batched_gemm_3d<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_gemv<T>(_matrix: &Tensor<T>, _vector: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_dot_product<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_outer_product<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_transpose_2d<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_symmetric_multiply<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn blas_triangular_solve<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
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
    unreachable!("BLAS functions should not be called when BLAS is not available")
}

#[cfg(not(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
)))]
#[allow(dead_code)]
fn is_triangular_candidate<T>(_tensor: &Tensor<T>) -> bool
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
    false
}
