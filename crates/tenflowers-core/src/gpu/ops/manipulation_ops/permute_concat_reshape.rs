//! GPU Tensor Permutation, Concatenation, and Reshape Operations

use super::super::super::*;
use crate::Result;

/// Execute tensor permutation on GPU
pub fn execute_tensor_permutation<T>(
    input: &GpuBuffer<T>,
    permutation: &[usize],
    input_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Tensor permutation is the same as transpose with axes = permutation
    super::transpose_slice::execute_transpose(input, permutation, input_shape, output_len)
}

/// Execute concatenate operation on GPU
pub fn execute_concatenate<T>(
    inputs: &[&GpuBuffer<T>],
    axis: usize,
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use wgpu::util::DeviceExt as _;

    if inputs.is_empty() {
        return Err(crate::TensorError::InvalidShape {
            operation: "concatenate".to_string(),
            reason: "Cannot concatenate empty list of tensors".to_string(),
            shape: None,
            context: None,
        });
    }

    // Get GPU context
    let context = crate::gpu::GpuContext::global()?;
    let device = &context.device;
    let queue = &context.queue;

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("concatenate_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // For now, use simple copy operations for concatenation
    // More advanced GPU kernels could be implemented later for complex cases
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("concatenate_encoder"),
    });

    let mut offset = 0u64;
    for input_buffer in inputs {
        let buffer_size = input_buffer.len() * std::mem::size_of::<T>();
        encoder.copy_buffer_to_buffer(
            input_buffer.buffer(),
            0,
            &output_buffer,
            offset,
            buffer_size as u64,
        );
        offset += buffer_size as u64;
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Extract device_id from first input buffer
    let device_id = match inputs[0].device_enum() {
        Device::Gpu(id) => id,
        _ => 0, // Default for CPU
    };
    // Create GpuBuffer from the result
    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        context.device.clone(),
        context.queue.clone(),
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Execute reshape operation on GPU
pub fn execute_reshape<T>(input: &GpuBuffer<T>, new_shape: &[usize]) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // For reshape, we just need to return the same buffer with new metadata
    // since reshape doesn't change the actual data layout in memory
    let output_len = new_shape.iter().product();

    // Verify that the total number of elements matches
    if output_len != input.len() {
        return Err(crate::TensorError::InvalidShape {
            operation: "reshape".to_string(),
            reason: format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                input.len(),
                new_shape,
                output_len
            ),
            shape: Some(new_shape.to_vec()),
            context: None,
        });
    }

    // For GPU reshape, we can just copy the buffer since the data layout is the same
    let context = crate::gpu::GpuContext::global()?;
    let device = &context.device;

    // Create new buffer and copy data
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reshape_output"),
        size: input.buffer().size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy data directly
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reshape_copy_encoder"),
    });

    encoder.copy_buffer_to_buffer(input.buffer(), 0, &output_buffer, 0, input.buffer().size());

    context.queue.submit(std::iter::once(encoder.finish()));

    // Extract device_id from input buffer
    let device_id = match input.device_enum() {
        Device::Gpu(id) => id,
        _ => 0, // Default for CPU
    };
    // Create GpuBuffer from the result
    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        context.device.clone(),
        context.queue.clone(),
        Device::Gpu(device_id),
        output_len,
    ))
}
