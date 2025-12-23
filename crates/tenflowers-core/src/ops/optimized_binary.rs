//! Optimized CPU binary operations for maximum performance
//!
//! This module provides highly optimized CPU implementations of binary operations
//! that aim to match NumPy's performance through:
//! - SIMD-friendly memory access patterns
//! - Vectorized operations using rayon
//! - Zero-copy optimizations where possible
//! - Specialized fast paths for common cases

use crate::shape_error_taxonomy::ShapeErrorUtils;
use crate::tensor::TensorStorage;
use crate::{Result, Shape, Tensor, TensorError};
use rayon::prelude::*;
use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};
use scirs2_core::numeric::Zero;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

/// Threshold for switching to parallel processing (number of elements)
const PARALLEL_THRESHOLD: usize = 10000;

/// SIMD-friendly chunk size (should be multiple of cache line size)
const SIMD_CHUNK_SIZE: usize = 64;

/// Optimized binary operation trait with vectorized implementations
pub trait OptimizedBinaryOp<T> {
    fn apply(&self, a: T, b: T) -> T;
    fn name(&self) -> &str;

    /// Vectorized operation on array chunks - default implementation falls back to element-wise
    fn apply_chunk(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T])
    where
        T: Clone,
    {
        for ((a_val, b_val), out) in a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
        {
            *out = self.apply(a_val.clone(), b_val.clone());
        }
    }

    /// Optimized chunk operation for Copy types (avoids cloning)
    fn apply_chunk_copy(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T])
    where
        T: Copy,
    {
        for ((a_val, b_val), out) in a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
        {
            *out = self.apply(*a_val, *b_val);
        }
    }

    /// Check if this operation supports vectorized chunks
    fn supports_vectorization(&self) -> bool {
        false
    }
}

/// Optimized addition operation
#[derive(Clone)]
pub struct OptimizedAddOp;

impl<T: StdAdd<Output = T> + Clone> OptimizedBinaryOp<T> for OptimizedAddOp {
    fn apply(&self, a: T, b: T) -> T {
        a + b
    }
    fn name(&self) -> &str {
        "Add"
    }

    fn supports_vectorization(&self) -> bool {
        true
    }

    fn apply_chunk(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T]) {
        // Use iterator for better compiler optimization
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = a.clone() + b.clone();
            });
    }

    fn apply_chunk_copy(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T])
    where
        T: Copy,
    {
        // Optimized for Copy types - no cloning needed
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = *a + *b;
            });
    }
}

/// Optimized multiplication operation
#[derive(Clone)]
pub struct OptimizedMulOp;

impl<T: StdMul<Output = T> + Clone> OptimizedBinaryOp<T> for OptimizedMulOp {
    fn apply(&self, a: T, b: T) -> T {
        a * b
    }
    fn name(&self) -> &str {
        "Mul"
    }

    fn supports_vectorization(&self) -> bool {
        true
    }

    fn apply_chunk(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T]) {
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = a.clone() * b.clone();
            });
    }

    fn apply_chunk_copy(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T])
    where
        T: Copy,
    {
        // Optimized for Copy types - no cloning needed
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = *a * *b;
            });
    }
}

/// Optimized subtraction operation
#[derive(Clone)]
pub struct OptimizedSubOp;

impl<T: StdSub<Output = T> + Clone> OptimizedBinaryOp<T> for OptimizedSubOp {
    fn apply(&self, a: T, b: T) -> T {
        a - b
    }
    fn name(&self) -> &str {
        "Sub"
    }

    fn supports_vectorization(&self) -> bool {
        true
    }

    fn apply_chunk(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T]) {
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = a.clone() - b.clone();
            });
    }

    fn apply_chunk_copy(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T])
    where
        T: Copy,
    {
        // Optimized for Copy types - no cloning needed
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = *a - *b;
            });
    }
}

/// Optimized division operation
#[derive(Clone)]
pub struct OptimizedDivOp;

impl<T: StdDiv<Output = T> + Clone> OptimizedBinaryOp<T> for OptimizedDivOp {
    fn apply(&self, a: T, b: T) -> T {
        a / b
    }
    fn name(&self) -> &str {
        "Div"
    }

    fn supports_vectorization(&self) -> bool {
        true
    }

    fn apply_chunk(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T]) {
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = a.clone() / b.clone();
            });
    }

    fn apply_chunk_copy(&self, a_chunk: &[T], b_chunk: &[T], output_chunk: &mut [T])
    where
        T: Copy,
    {
        // Optimized for Copy types - no cloning needed
        a_chunk
            .iter()
            .zip(b_chunk.iter())
            .zip(output_chunk.iter_mut())
            .for_each(|((a, b), out)| {
                *out = *a / *b;
            });
    }
}

/// Fast path for contiguous same-shape tensors (most common case)
fn fast_binary_op_contiguous<T, Op>(a_data: &[T], b_data: &[T], output_data: &mut [T], op: &Op)
where
    T: Clone + Send + Sync,
    Op: OptimizedBinaryOp<T> + Sync,
{
    let len = a_data.len();

    // Use parallel processing for large arrays
    if len > PARALLEL_THRESHOLD && op.supports_vectorization() {
        // Process in parallel chunks
        output_data
            .par_chunks_mut(SIMD_CHUNK_SIZE)
            .zip(a_data.par_chunks(SIMD_CHUNK_SIZE))
            .zip(b_data.par_chunks(SIMD_CHUNK_SIZE))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                op.apply_chunk(a_chunk, b_chunk, out_chunk);
            });
    } else if op.supports_vectorization() {
        // Sequential vectorized processing for smaller arrays
        for ((out_chunk, a_chunk), b_chunk) in output_data
            .chunks_mut(SIMD_CHUNK_SIZE)
            .zip(a_data.chunks(SIMD_CHUNK_SIZE))
            .zip(b_data.chunks(SIMD_CHUNK_SIZE))
        {
            op.apply_chunk(a_chunk, b_chunk, out_chunk);
        }
    } else {
        // Fallback to element-wise
        for i in 0..len {
            output_data[i] = op.apply(a_data[i].clone(), b_data[i].clone());
        }
    }
}

/// Optimized fast path for Copy types (avoids cloning)
fn fast_binary_op_contiguous_copy<T, Op>(a_data: &[T], b_data: &[T], output_data: &mut [T], op: &Op)
where
    T: Copy + Send + Sync,
    Op: OptimizedBinaryOp<T> + Sync,
{
    let len = a_data.len();

    // Use parallel processing for large arrays
    if len > PARALLEL_THRESHOLD && op.supports_vectorization() {
        // Process in parallel chunks using optimized copy path
        output_data
            .par_chunks_mut(SIMD_CHUNK_SIZE)
            .zip(a_data.par_chunks(SIMD_CHUNK_SIZE))
            .zip(b_data.par_chunks(SIMD_CHUNK_SIZE))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                op.apply_chunk_copy(a_chunk, b_chunk, out_chunk);
            });
    } else if op.supports_vectorization() {
        // Sequential vectorized processing for smaller arrays
        for ((out_chunk, a_chunk), b_chunk) in output_data
            .chunks_mut(SIMD_CHUNK_SIZE)
            .zip(a_data.chunks(SIMD_CHUNK_SIZE))
            .zip(b_data.chunks(SIMD_CHUNK_SIZE))
        {
            op.apply_chunk_copy(a_chunk, b_chunk, out_chunk);
        }
    } else {
        // Fallback to element-wise - no cloning needed for Copy types
        for i in 0..len {
            output_data[i] = op.apply(a_data[i], b_data[i]);
        }
    }
}

/// Optimized binary operation with fast paths for common cases
pub fn optimized_binary_op<T, Op>(a: &Tensor<T>, b: &Tensor<T>, op: Op) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: OptimizedBinaryOp<T> + Sync,
{
    // Check device compatibility
    if a.device() != b.device() {
        return Err(TensorError::device_mismatch(
            "optimized_binary_op",
            &a.device().to_string(),
            &b.device().to_string(),
        ));
    }

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
            // Fast path 1: Same shape, contiguous arrays
            if a.shape() == b.shape() && arr_a.is_standard_layout() && arr_b.is_standard_layout() {
                let mut result = ArrayD::zeros(arr_a.raw_dim());

                // Use raw data slices for maximum performance
                if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
                    (arr_a.as_slice(), arr_b.as_slice(), result.as_slice_mut())
                {
                    fast_binary_op_contiguous(a_slice, b_slice, out_slice, &op);
                    return Ok(Tensor::from_array(result));
                }
            }

            // Fast path 2: Scalar operations (broadcasting with scalar)
            if a.shape().size() == 1 || b.shape().size() == 1 {
                return scalar_broadcast_op(arr_a, arr_b, a.shape(), b.shape(), op);
            }

            // General case: broadcasting required
            let broadcast_shape = a.shape().broadcast_shape(b.shape()).ok_or_else(|| {
                ShapeErrorUtils::broadcast_incompatible("optimized_binary_op", a.shape(), b.shape())
            })?;

            // Broadcast arrays to common shape
            let a_broadcast = broadcast_array(arr_a, &broadcast_shape)?;
            let b_broadcast = broadcast_array(arr_b, &broadcast_shape)?;

            // Apply operation with optimized loop
            let mut result = ArrayD::zeros(a_broadcast.raw_dim());

            // Check if we can use the fast contiguous path after broadcasting
            if a_broadcast.is_standard_layout() && b_broadcast.is_standard_layout() {
                if let (Some(a_slice), Some(b_slice), Some(out_slice)) = (
                    a_broadcast.as_slice(),
                    b_broadcast.as_slice(),
                    result.as_slice_mut(),
                ) {
                    fast_binary_op_contiguous(a_slice, b_slice, out_slice, &op);
                    return Ok(Tensor::from_array(result));
                }
            }

            // Fallback to ndarray's Zip for complex broadcasting cases
            Zip::from(&mut result)
                .and(&a_broadcast)
                .and(&b_broadcast)
                .for_each(|r, a_val, b_val| {
                    *r = op.apply(*a_val, *b_val);
                });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // For GPU tensors, fall back to the existing GPU implementation
            super::binary::binary_op(a, b, GPUOpWrapper { op })
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!("Device mismatch should have been caught earlier"),
    }
}

/// Optimized binary operation with fast paths for Copy types (avoids cloning)
pub fn optimized_binary_op_copy<T, Op>(a: &Tensor<T>, b: &Tensor<T>, op: Op) -> Result<Tensor<T>>
where
    T: Copy + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: OptimizedBinaryOp<T> + Sync,
{
    // Check device compatibility
    if a.device() != b.device() {
        return Err(TensorError::device_mismatch(
            "optimized_binary_op_copy",
            &a.device().to_string(),
            &b.device().to_string(),
        ));
    }

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
            // Fast path 1: Same shape, contiguous arrays - use Copy optimization
            if a.shape() == b.shape() && arr_a.is_standard_layout() && arr_b.is_standard_layout() {
                let mut result = ArrayD::zeros(arr_a.raw_dim());

                // Use raw data slices for maximum performance with Copy optimization
                if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
                    (arr_a.as_slice(), arr_b.as_slice(), result.as_slice_mut())
                {
                    fast_binary_op_contiguous_copy(a_slice, b_slice, out_slice, &op);
                    return Ok(Tensor::from_array(result));
                }
            }

            // Fast path 2: Scalar operations (broadcasting with scalar)
            if a.shape().size() == 1 || b.shape().size() == 1 {
                return scalar_broadcast_op_copy(arr_a, arr_b, a.shape(), b.shape(), op);
            }

            // General case: broadcasting required
            let broadcast_shape = a.shape().broadcast_shape(b.shape()).ok_or_else(|| {
                ShapeErrorUtils::broadcast_incompatible(
                    "optimized_binary_op_copy",
                    a.shape(),
                    b.shape(),
                )
            })?;

            // Broadcast arrays to common shape
            let a_broadcast = broadcast_array(arr_a, &broadcast_shape)?;
            let b_broadcast = broadcast_array(arr_b, &broadcast_shape)?;

            // Apply operation with optimized loop
            let mut result = ArrayD::zeros(a_broadcast.raw_dim());

            // Check if we can use the fast contiguous path after broadcasting
            if a_broadcast.is_standard_layout() && b_broadcast.is_standard_layout() {
                if let (Some(a_slice), Some(b_slice), Some(out_slice)) = (
                    a_broadcast.as_slice(),
                    b_broadcast.as_slice(),
                    result.as_slice_mut(),
                ) {
                    fast_binary_op_contiguous_copy(a_slice, b_slice, out_slice, &op);
                    return Ok(Tensor::from_array(result));
                }
            }

            // Fallback to ndarray's Zip for complex broadcasting cases - no cloning for Copy
            Zip::from(&mut result)
                .and(&a_broadcast)
                .and(&b_broadcast)
                .for_each(|r, a_val, b_val| {
                    *r = op.apply(*a_val, *b_val);
                });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // For GPU tensors, fall back to the existing GPU implementation
            super::binary::binary_op(a, b, GPUOpWrapper { op })
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!("Device mismatch should have been caught earlier"),
    }
}

/// Optimized scalar broadcasting operation
fn scalar_broadcast_op<T, Op>(
    arr_a: &ArrayD<T>,
    arr_b: &ArrayD<T>,
    shape_a: &Shape,
    shape_b: &Shape,
    op: Op,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
    Op: OptimizedBinaryOp<T> + Sync,
{
    if shape_a.size() == 1 && shape_b.size() > 1 {
        // a is scalar, b is array
        let scalar_val = &arr_a.iter().next().unwrap().clone();
        let mut result = ArrayD::zeros(arr_b.raw_dim());

        if let (Some(b_slice), Some(out_slice)) = (arr_b.as_slice(), result.as_slice_mut()) {
            if b_slice.len() > PARALLEL_THRESHOLD {
                out_slice
                    .par_iter_mut()
                    .zip(b_slice.par_iter())
                    .for_each(|(out, b_val)| {
                        *out = op.apply(scalar_val.clone(), b_val.clone());
                    });
            } else {
                for (out, b_val) in out_slice.iter_mut().zip(b_slice.iter()) {
                    *out = op.apply(scalar_val.clone(), b_val.clone());
                }
            }
        } else {
            // Fallback for non-contiguous arrays
            Zip::from(&mut result).and(arr_b).for_each(|r, b_val| {
                *r = op.apply(scalar_val.clone(), b_val.clone());
            });
        }

        Ok(Tensor::from_array(result))
    } else if shape_b.size() == 1 && shape_a.size() > 1 {
        // b is scalar, a is array
        let scalar_val = &arr_b.iter().next().unwrap().clone();
        let mut result = ArrayD::zeros(arr_a.raw_dim());

        if let (Some(a_slice), Some(out_slice)) = (arr_a.as_slice(), result.as_slice_mut()) {
            if a_slice.len() > PARALLEL_THRESHOLD {
                out_slice
                    .par_iter_mut()
                    .zip(a_slice.par_iter())
                    .for_each(|(out, a_val)| {
                        *out = op.apply(a_val.clone(), scalar_val.clone());
                    });
            } else {
                for (out, a_val) in out_slice.iter_mut().zip(a_slice.iter()) {
                    *out = op.apply(a_val.clone(), scalar_val.clone());
                }
            }
        } else {
            // Fallback for non-contiguous arrays
            Zip::from(&mut result).and(arr_a).for_each(|r, a_val| {
                *r = op.apply(a_val.clone(), scalar_val.clone());
            });
        }

        Ok(Tensor::from_array(result))
    } else {
        // Both are scalars
        let a_val = arr_a.iter().next().unwrap().clone();
        let b_val = arr_b.iter().next().unwrap().clone();
        let result_val = op.apply(a_val, b_val);
        Ok(Tensor::from_array(ArrayD::from_elem(
            IxDyn(&[1]),
            result_val,
        )))
    }
}

/// Optimized scalar broadcasting operation for Copy types (avoids cloning)
fn scalar_broadcast_op_copy<T, Op>(
    arr_a: &ArrayD<T>,
    arr_b: &ArrayD<T>,
    shape_a: &Shape,
    shape_b: &Shape,
    op: Op,
) -> Result<Tensor<T>>
where
    T: Copy + Default + Zero + Send + Sync + 'static,
    Op: OptimizedBinaryOp<T> + Sync,
{
    if shape_a.size() == 1 && shape_b.size() > 1 {
        // a is scalar, b is array
        let scalar_val = *arr_a.iter().next().unwrap();
        let mut result = ArrayD::zeros(arr_b.raw_dim());

        if let (Some(b_slice), Some(out_slice)) = (arr_b.as_slice(), result.as_slice_mut()) {
            if b_slice.len() > PARALLEL_THRESHOLD {
                out_slice
                    .par_iter_mut()
                    .zip(b_slice.par_iter())
                    .for_each(|(out, b_val)| {
                        *out = op.apply(scalar_val, *b_val);
                    });
            } else {
                for (out, b_val) in out_slice.iter_mut().zip(b_slice.iter()) {
                    *out = op.apply(scalar_val, *b_val);
                }
            }
        } else {
            // Fallback for non-contiguous arrays
            Zip::from(&mut result).and(arr_b).for_each(|r, b_val| {
                *r = op.apply(scalar_val, *b_val);
            });
        }

        Ok(Tensor::from_array(result))
    } else if shape_b.size() == 1 && shape_a.size() > 1 {
        // b is scalar, a is array
        let scalar_val = *arr_b.iter().next().unwrap();
        let mut result = ArrayD::zeros(arr_a.raw_dim());

        if let (Some(a_slice), Some(out_slice)) = (arr_a.as_slice(), result.as_slice_mut()) {
            if a_slice.len() > PARALLEL_THRESHOLD {
                out_slice
                    .par_iter_mut()
                    .zip(a_slice.par_iter())
                    .for_each(|(out, a_val)| {
                        *out = op.apply(*a_val, scalar_val);
                    });
            } else {
                for (out, a_val) in out_slice.iter_mut().zip(a_slice.iter()) {
                    *out = op.apply(*a_val, scalar_val);
                }
            }
        } else {
            // Fallback for non-contiguous arrays
            Zip::from(&mut result).and(arr_a).for_each(|r, a_val| {
                *r = op.apply(*a_val, scalar_val);
            });
        }

        Ok(Tensor::from_array(result))
    } else {
        // Both are scalars
        let a_val = *arr_a.iter().next().unwrap();
        let b_val = *arr_b.iter().next().unwrap();
        let result_val = op.apply(a_val, b_val);
        Ok(Tensor::from_array(ArrayD::from_elem(
            IxDyn(&[1]),
            result_val,
        )))
    }
}

/// Broadcast an array to a target shape (unchanged from original)
fn broadcast_array<T: Clone>(array: &ArrayD<T>, target_shape: &Shape) -> Result<ArrayD<T>> {
    let target_dims = IxDyn(target_shape.dims());

    // If shapes match, just clone
    if array.shape() == target_shape.dims() {
        return Ok(array.clone());
    }

    // Convert array shape to Shape object for standardized error messages
    let array_shape = Shape::from_slice(array.shape());

    // Use ndarray's broadcast functionality
    array
        .broadcast(target_dims)
        .ok_or_else(|| {
            // Use standardized broadcast error message
            ShapeErrorUtils::broadcast_incompatible("broadcast_cpu", &array_shape, target_shape)
        })
        .map(|view| view.to_owned())
}

/// Wrapper to bridge OptimizedBinaryOp to BinaryOp for GPU fallback
#[cfg(feature = "gpu")]
struct GPUOpWrapper<Op> {
    op: Op,
}

#[cfg(feature = "gpu")]
impl<T, Op> super::binary::BinaryOp<T> for GPUOpWrapper<Op>
where
    Op: OptimizedBinaryOp<T>,
    T: Clone,
{
    fn apply(&self, a: T, b: T) -> T {
        self.op.apply(a, b)
    }

    fn name(&self) -> &str {
        self.op.name()
    }
}

// Optimized public functions
pub fn optimized_add<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + StdAdd<Output = T> + Send + Sync + 'static + bytemuck::Pod,
{
    optimized_binary_op(a, b, OptimizedAddOp)
}

pub fn optimized_sub<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + StdSub<Output = T> + Send + Sync + 'static + bytemuck::Pod,
{
    optimized_binary_op(a, b, OptimizedSubOp)
}

pub fn optimized_mul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + StdMul<Output = T> + Send + Sync + 'static + bytemuck::Pod,
{
    optimized_binary_op(a, b, OptimizedMulOp)
}

pub fn optimized_div<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + StdDiv<Output = T> + Send + Sync + 'static + bytemuck::Pod,
{
    optimized_binary_op(a, b, OptimizedDivOp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::binary;
    use crate::Tensor;

    #[test]
    fn test_optimized_add_contiguous() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[4]).unwrap();

        let result = optimized_add(&a, &b).unwrap();
        let expected = vec![6.0f32, 8.0, 10.0, 12.0];

        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_optimized_mul_large_array() {
        let size = PARALLEL_THRESHOLD + 100;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

        let a = Tensor::from_vec(a_data.clone(), &[size]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), &[size]).unwrap();

        let result = optimized_mul(&a, &b).unwrap();
        let result_data = result.to_vec().unwrap();

        // Check a few elements
        assert_eq!(result_data[0], 0.0); // 0 * 0
        assert_eq!(result_data[1], 2.0); // 1 * 2
        assert_eq!(result_data[10], 200.0); // 10 * 20
    }

    #[test]
    fn test_scalar_broadcast() {
        let a = Tensor::from_vec(vec![2.0f32], &[1]).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();

        let result = optimized_add(&a, &b).unwrap();
        let expected = vec![3.0f32, 4.0, 5.0];

        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_performance_benchmark() {
        use std::time::Instant;

        let size = 1_000_000;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

        let a = Tensor::from_vec(a_data, &[size]).unwrap();
        let b = Tensor::from_vec(b_data, &[size]).unwrap();

        // Time optimized version
        let start = Instant::now();
        let _result = optimized_add(&a, &b).unwrap();
        let optimized_time = start.elapsed();

        // Time original version
        let start = Instant::now();
        let _result = binary::add(&a, &b).unwrap();
        let original_time = start.elapsed();

        println!("Original time: {:?}", original_time);
        println!("Optimized time: {:?}", optimized_time);
        println!(
            "Speedup: {:.2}x",
            original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64
        );

        // Optimized version should be faster (though this is not guaranteed in all test environments)
        // This test serves as a performance indicator
    }
}
