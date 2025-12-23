//! Generic Binary Operation Implementation
//!
//! This module provides the core implementation of binary operations with
//! comprehensive broadcasting support and device management.

use super::core::BinaryOp;
use crate::shape_error_taxonomy::ShapeErrorUtils;
use crate::{Device, Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::Zero;

/// Generic binary operation implementation with broadcasting
#[allow(unused_variables)]
pub fn binary_op<T, Op>(a: &Tensor<T>, b: &Tensor<T>, op: Op) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: BinaryOp<T>,
{
    // Simple profiling: Track binary operation
    // TODO: Implement profiler integration
    // #[cfg(feature = "autograd")]
    // if let Ok(mut p) = get_profiler().lock() {
    //     p.record_operation(&format!("binary_{}", op.name()));
    // }

    // Check device compatibility
    if a.device() != b.device() {
        return Err(TensorError::device_mismatch(
            "binary_op",
            &a.device().to_string(),
            &b.device().to_string(),
        ));
    }

    // Compute broadcast shape
    let broadcast_shape = a.shape().broadcast_shape(b.shape()).ok_or_else(|| {
        ShapeErrorUtils::broadcast_incompatible("binary_op", a.shape(), b.shape())
    })?;

    // Handle different device types
    match (a.device(), b.device()) {
        (Device::Cpu, Device::Cpu) => {
            // CPU implementation
            cpu_binary_op(a, b, op, &broadcast_shape)
        }
        #[cfg(feature = "gpu")]
        (Device::Gpu(_), Device::Gpu(_)) => {
            // GPU implementation
            gpu_binary_op(a, b, op, &broadcast_shape)
        }
        #[cfg(feature = "rocm")]
        (Device::Rocm(_), Device::Rocm(_)) => {
            // ROCm implementation would go here
            todo!("ROCm binary operations not yet implemented")
        }
        #[cfg(any(feature = "gpu", feature = "rocm"))]
        _ => {
            // Device mismatch between different device types
            Err(TensorError::device_mismatch(
                "binary_op",
                &a.device().to_string(),
                &b.device().to_string(),
            ))
        }
    }
}

/// CPU binary operation implementation with broadcasting
fn cpu_binary_op<T, Op>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op: Op,
    broadcast_shape: &crate::Shape,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: BinaryOp<T>,
{
    use crate::tensor::TensorStorage;

    // Get CPU arrays from tensors
    let (a_array, b_array) = match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(ref a_arr), TensorStorage::Cpu(ref b_arr)) => (a_arr, b_arr),
        #[cfg(feature = "gpu")]
        _ => {
            return Err(TensorError::device_mismatch(
                "cpu_binary_op",
                "cpu",
                "non-cpu",
            ))
        }
    };

    // Create output array with broadcast shape
    let output_dims = IxDyn(broadcast_shape.dims());
    let mut output = ArrayD::zeros(output_dims);

    // Handle broadcasting with full NumPy-style compatibility
    if a.shape() == b.shape() && a.shape().dims() == broadcast_shape.dims() {
        // Simple case: same shapes - direct element-wise operation
        for ((a_val, b_val), out_val) in a_array.iter().zip(b_array.iter()).zip(output.iter_mut()) {
            *out_val = op.apply(*a_val, *b_val);
        }
    } else {
        // Advanced broadcasting case - handle different but compatible shapes
        broadcast_operation(
            a_array,
            b_array,
            &mut output,
            &op,
            a.shape(),
            b.shape(),
            broadcast_shape,
        )?;
    }

    // Create result tensor using from_array
    Ok(Tensor::from_array(output))
}

/// Advanced broadcasting operation for tensors with different but compatible shapes
fn broadcast_operation<T, Op>(
    a_array: &ArrayD<T>,
    b_array: &ArrayD<T>,
    output: &mut ArrayD<T>,
    op: &Op,
    a_shape: &crate::Shape,
    b_shape: &crate::Shape,
    broadcast_shape: &crate::Shape,
) -> Result<()>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: BinaryOp<T>,
{
    // Handle simple scalar cases first for performance
    if a_shape.size() == 1 {
        // a is scalar, broadcast to b
        if let Some(a_scalar) = a_array.iter().next() {
            for (b_val, out_val) in b_array.iter().zip(output.iter_mut()) {
                *out_val = op.apply(*a_scalar, *b_val);
            }
        }
        return Ok(());
    }

    if b_shape.size() == 1 {
        // b is scalar, broadcast to a
        if let Some(b_scalar) = b_array.iter().next() {
            for (a_val, out_val) in a_array.iter().zip(output.iter_mut()) {
                *out_val = op.apply(*a_val, *b_scalar);
            }
        }
        return Ok(());
    }

    // Full broadcasting for complex shapes
    // Use coordinate iteration to handle broadcasting properly
    let output_shape = broadcast_shape.dims();
    let a_dims = a_shape.dims();
    let b_dims = b_shape.dims();

    // Calculate strides for each dimension to enable broadcasting
    let mut a_strides = vec![0; output_shape.len()];
    let mut b_strides = vec![0; output_shape.len()];

    // Align dimensions from the right (trailing dimensions)
    let a_offset = output_shape.len() - a_dims.len();
    let b_offset = output_shape.len() - b_dims.len();

    // Calculate actual strides for a
    let mut a_stride_acc = 1;
    for i in (0..a_dims.len()).rev() {
        let out_idx = a_offset + i;
        if a_dims[i] == 1 {
            a_strides[out_idx] = 0; // Broadcasting: stride is 0 for singleton dimensions
        } else {
            a_strides[out_idx] = a_stride_acc;
        }
        a_stride_acc *= a_dims[i];
    }

    // Calculate actual strides for b
    let mut b_stride_acc = 1;
    for i in (0..b_dims.len()).rev() {
        let out_idx = b_offset + i;
        if b_dims[i] == 1 {
            b_strides[out_idx] = 0; // Broadcasting: stride is 0 for singleton dimensions
        } else {
            b_strides[out_idx] = b_stride_acc;
        }
        b_stride_acc *= b_dims[i];
    }

    // Iterate through all output positions and compute indices for a and b
    let total_elements: usize = output_shape.iter().product();
    for linear_idx in 0..total_elements {
        // Convert linear index to multi-dimensional coordinates
        let mut coords = vec![0; output_shape.len()];
        let mut remaining = linear_idx;
        for i in (0..output_shape.len()).rev() {
            coords[i] = remaining % output_shape[i];
            remaining /= output_shape[i];
        }

        // Calculate corresponding indices in a and b using strides
        let mut a_idx = 0;
        let mut b_idx = 0;
        for i in 0..output_shape.len() {
            a_idx += coords[i] * a_strides[i];
            b_idx += coords[i] * b_strides[i];
        }

        // Get values from a and b arrays using linear indexing
        let a_val = a_array
            .as_slice()
            .unwrap_or_else(|| panic!("Failed to get slice from a_array"))[a_idx];
        let b_val = b_array
            .as_slice()
            .unwrap_or_else(|| panic!("Failed to get slice from b_array"))[b_idx];

        // Apply operation and store result
        let result_val = op.apply(a_val, b_val);
        output
            .as_slice_mut()
            .unwrap_or_else(|| panic!("Failed to get mutable slice from output"))[linear_idx] =
            result_val;
    }

    Ok(())
}

/// GPU binary operation implementation with broadcasting
#[cfg(feature = "gpu")]
fn gpu_binary_op<T, Op>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    op: Op,
    broadcast_shape: &crate::Shape,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: BinaryOp<T>,
{
    use crate::gpu::binary_ops;
    use crate::tensor::TensorStorage;

    // Get GPU buffers from tensors
    let (a_buffer, b_buffer) = match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(ref a_buf), TensorStorage::Gpu(ref b_buf)) => (a_buf, b_buf),
        _ => {
            return Err(TensorError::device_mismatch(
                "gpu_binary_op",
                "gpu",
                "non-gpu",
            ))
        }
    };

    // Map operation name to GPU operation type
    let gpu_op = match op.name() {
        "Add" => binary_ops::BinaryOp::Add,
        "Sub" => binary_ops::BinaryOp::Sub,
        "Mul" => binary_ops::BinaryOp::Mul,
        "Div" => binary_ops::BinaryOp::Div,
        "Pow" => binary_ops::BinaryOp::Pow,
        "Min" => binary_ops::BinaryOp::Min,
        "Max" => binary_ops::BinaryOp::Max,
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unsupported GPU binary operation: {}",
                op.name()
            )))
        }
    };

    // Calculate output size
    let output_len = broadcast_shape.size();

    // Execute GPU operation with broadcasting if needed
    let result_buffer = if a.shape() == b.shape() && a.shape().dims() == broadcast_shape.dims() {
        // Simple case: same shapes - no broadcasting needed
        binary_ops::execute_binary_op(a_buffer, b_buffer, gpu_op, output_len)?
    } else {
        // Broadcasting case
        binary_ops::execute_binary_op_with_broadcasting(
            a_buffer,
            b_buffer,
            gpu_op,
            a.shape().dims(),
            b.shape().dims(),
            broadcast_shape.dims(),
            output_len,
        )?
    };

    // Create result tensor from GPU buffer
    Ok(Tensor::from_gpu_buffer(
        result_buffer,
        broadcast_shape.clone(),
    ))
}
