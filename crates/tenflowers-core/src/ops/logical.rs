use crate::tensor::TensorStorage;
use crate::{Result, Shape, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, Zip};

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;

/// Logical operation trait for boolean tensors (represented as u8)
pub trait LogicalOp {
    fn apply(&self, a: bool, b: bool) -> bool;
    fn name(&self) -> &str;
}

/// Unary logical operation trait
pub trait UnaryLogicalOp {
    fn apply(&self, a: bool) -> bool;
    fn name(&self) -> &str;
}

/// Logical AND operation
pub struct AndOp;
impl LogicalOp for AndOp {
    fn apply(&self, a: bool, b: bool) -> bool {
        a && b
    }
    fn name(&self) -> &str {
        "And"
    }
}

/// Logical OR operation
pub struct OrOp;
impl LogicalOp for OrOp {
    fn apply(&self, a: bool, b: bool) -> bool {
        a || b
    }
    fn name(&self) -> &str {
        "Or"
    }
}

/// Logical XOR operation
pub struct XorOp;
impl LogicalOp for XorOp {
    fn apply(&self, a: bool, b: bool) -> bool {
        a ^ b
    }
    fn name(&self) -> &str {
        "Xor"
    }
}

/// Logical NOT operation
pub struct NotOp;
impl UnaryLogicalOp for NotOp {
    fn apply(&self, a: bool) -> bool {
        !a
    }
    fn name(&self) -> &str {
        "Not"
    }
}

/// Generic binary logical operation implementation with broadcasting
/// Operates on boolean tensors (represented as u8: 0=false, 1=true)
pub fn logical_binary_op<Op>(a: &Tensor<u8>, b: &Tensor<u8>, op: Op) -> Result<Tensor<u8>>
where
    Op: LogicalOp,
{
    // Check device compatibility
    if a.device() != b.device() {
        return Err(TensorError::device_mismatch(
            "logical_op",
            &a.device().to_string(),
            &b.device().to_string(),
        ));
    }

    // Compute broadcast shape
    let broadcast_shape =
        a.shape()
            .broadcast_shape(b.shape())
            .ok_or_else(|| TensorError::ShapeMismatch {
                operation: "logical_op".to_string(),
                expected: a.shape().to_string(),
                got: b.shape().to_string(),
                context: None,
            })?;

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
            // Broadcast arrays to common shape
            let a_broadcast = broadcast_array(arr_a, &broadcast_shape)?;
            let b_broadcast = broadcast_array(arr_b, &broadcast_shape)?;

            // Apply logical operation element-wise
            let mut result = ArrayD::zeros(a_broadcast.raw_dim());
            Zip::from(&mut result)
                .and(&a_broadcast)
                .and(&b_broadcast)
                .for_each(|r, a_val, b_val| {
                    let a_bool = *a_val != 0;
                    let b_bool = *b_val != 0;
                    *r = if op.apply(a_bool, b_bool) { 1u8 } else { 0u8 };
                });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            gpu_logical_op_dispatch(gpu_a, gpu_b, &op, a.shape(), b.shape(), &broadcast_shape)
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!("Device mismatch should have been caught earlier"),
    }
}

/// Generic unary logical operation implementation
pub fn logical_unary_op<Op>(a: &Tensor<u8>, op: Op) -> Result<Tensor<u8>>
where
    Op: UnaryLogicalOp,
{
    match &a.storage {
        TensorStorage::Cpu(arr_a) => {
            // Apply logical operation element-wise
            let mut result = ArrayD::zeros(arr_a.raw_dim());
            Zip::from(&mut result).and(arr_a).for_each(|r, a_val| {
                let a_bool = *a_val != 0;
                *r = if op.apply(a_bool) { 1u8 } else { 0u8 };
            });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_a) => gpu_logical_unary_op_dispatch(gpu_a, &op, a.shape()),
    }
}

/// Broadcast an array to a target shape
fn broadcast_array<T: Clone>(array: &ArrayD<T>, target_shape: &Shape) -> Result<ArrayD<T>> {
    let target_dims = scirs2_core::ndarray::IxDyn(target_shape.dims());

    // If shapes match, just clone
    if array.shape() == target_shape.dims() {
        return Ok(array.clone());
    }

    // Use ndarray's broadcast functionality
    array
        .broadcast(target_dims)
        .ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Cannot broadcast from {:?} to {:?}",
                target_shape,
                array.shape()
            ))
        })
        .map(|view| view.to_owned())
}

// Concrete implementations using the generic logical operations

pub fn logical_and(a: &Tensor<u8>, b: &Tensor<u8>) -> Result<Tensor<u8>> {
    logical_binary_op(a, b, AndOp)
}

pub fn logical_or(a: &Tensor<u8>, b: &Tensor<u8>) -> Result<Tensor<u8>> {
    logical_binary_op(a, b, OrOp)
}

pub fn logical_xor(a: &Tensor<u8>, b: &Tensor<u8>) -> Result<Tensor<u8>> {
    logical_binary_op(a, b, XorOp)
}

pub fn logical_not(a: &Tensor<u8>) -> Result<Tensor<u8>> {
    logical_unary_op(a, NotOp)
}

/// GPU logical operation dispatch
#[cfg(feature = "gpu")]
fn gpu_logical_op_dispatch(
    gpu_a: &crate::gpu::buffer::GpuBuffer<u8>,
    gpu_b: &crate::gpu::buffer::GpuBuffer<u8>,
    op: &dyn LogicalOp,
    shape_a: &Shape,
    shape_b: &Shape,
    broadcast_shape: &Shape,
) -> Result<Tensor<u8>> {
    let gpu_op = match op.name() {
        "And" => crate::gpu::ops::LogicalOp::And,
        "Or" => crate::gpu::ops::LogicalOp::Or,
        "Xor" => crate::gpu::ops::LogicalOp::Xor,
        _ => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "GPU logical operation {} not implemented",
                op.name()
            )))
        }
    };

    let output_len: usize = broadcast_shape.dims().iter().product();

    // Convert u8 inputs to u32 for GPU operations (shader expects u32)
    let gpu_a_u32 = convert_u8_to_u32_gpu_buffer(gpu_a)?;
    let gpu_b_u32 = convert_u8_to_u32_gpu_buffer(gpu_b)?;

    // Map to GPU logical operation type
    let logical_gpu_op = match gpu_op {
        crate::gpu::ops::LogicalOp::And => crate::gpu::logical_ops::LogicalOp::And,
        crate::gpu::ops::LogicalOp::Or => crate::gpu::logical_ops::LogicalOp::Or,
        crate::gpu::ops::LogicalOp::Xor => crate::gpu::logical_ops::LogicalOp::Xor,
        _ => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "GPU logical operation {:?} not supported",
                gpu_op
            )))
        }
    };

    // Execute GPU logical operation
    let result_buffer_u32 = crate::gpu::logical_ops::execute_logical_op(
        &gpu_a_u32,
        &gpu_b_u32,
        logical_gpu_op,
        output_len,
    )?;

    // Convert u32 result back to u8
    let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

    Ok(Tensor::from_gpu_buffer(
        result_buffer,
        broadcast_shape.clone(),
    ))
}

/// GPU unary logical operation dispatch
#[cfg(feature = "gpu")]
fn gpu_logical_unary_op_dispatch(
    gpu_a: &crate::gpu::buffer::GpuBuffer<u8>,
    op: &dyn UnaryLogicalOp,
    shape: &Shape,
) -> Result<Tensor<u8>> {
    let gpu_op = match op.name() {
        "Not" => crate::gpu::unary_ops::UnaryLogicalOp::Not,
        _ => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "GPU unary logical operation {} not implemented",
                op.name()
            )))
        }
    };

    // Convert u8 input to u32 for GPU operations (shader expects u32)
    let gpu_a_u32 = convert_u8_to_u32_gpu_buffer(gpu_a)?;

    // Map to GPU unary logical operation type
    let unary_logical_gpu_op = match gpu_op {
        crate::gpu::unary_ops::UnaryLogicalOp::Not => crate::gpu::logical_ops::UnaryLogicalOp::Not,
    };

    // Execute GPU unary logical operation
    let output_len = shape.size();
    let result_buffer_u32 = crate::gpu::logical_ops::execute_unary_logical_op(
        &gpu_a_u32,
        unary_logical_gpu_op,
        output_len,
    )?;

    // Convert u32 result back to u8
    let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

    Ok(Tensor::from_gpu_buffer(result_buffer, shape.clone()))
}

#[cfg(test)]
#[allow(irrefutable_let_patterns)] // Pattern matching on TensorStorage is irrefutable when GPU feature is disabled
mod tests {
    use super::*;

    #[test]
    fn test_logical_and_same_shape() {
        let a = Tensor::<u8>::from_vec(vec![1u8, 0u8, 1u8], &[3]).unwrap();
        let b = Tensor::<u8>::from_vec(vec![1u8, 1u8, 0u8], &[3]).unwrap();

        let c = logical_and(&a, &b).unwrap();
        let expected = vec![1u8, 0u8, 0u8];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_logical_or_broadcast() {
        let a = Tensor::<u8>::from_vec(vec![1u8, 0u8], &[2, 1]).unwrap();
        let b = Tensor::<u8>::from_vec(vec![0u8, 1u8], &[1, 2]).unwrap();

        let c = logical_or(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);

        // Expected: [[1, 1], [0, 1]]
        let expected = vec![1u8, 1u8, 0u8, 1u8];
        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_logical_xor() {
        let a = Tensor::<u8>::from_vec(vec![1u8, 0u8, 1u8, 0u8], &[4]).unwrap();
        let b = Tensor::<u8>::from_vec(vec![1u8, 1u8, 0u8, 0u8], &[4]).unwrap();

        let c = logical_xor(&a, &b).unwrap();
        let expected = vec![0u8, 1u8, 1u8, 0u8];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_logical_not() {
        let a = Tensor::<u8>::from_vec(vec![1u8, 0u8, 1u8, 0u8], &[4]).unwrap();

        let c = logical_not(&a).unwrap();
        let expected = vec![0u8, 1u8, 0u8, 1u8];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_logical_and_scalar_broadcast() {
        let a = Tensor::<u8>::from_vec(vec![1u8, 0u8, 1u8], &[3]).unwrap();
        let scalar = Tensor::<u8>::from_vec(vec![1u8], &[1]).unwrap();

        let c = logical_and(&a, &scalar).unwrap();
        let expected = vec![1u8, 0u8, 1u8]; // AND with 1 preserves the original

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }
}

/// Convert a GPU buffer of u8 values to u32 values for shader compatibility
#[cfg(feature = "gpu")]
fn convert_u8_to_u32_gpu_buffer(
    input: &crate::gpu::buffer::GpuBuffer<u8>,
) -> Result<crate::gpu::buffer::GpuBuffer<u32>> {
    let device = &input.device;
    let queue = &input.queue;
    let output_len = input.len();

    // Create u32 output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u8_to_u32_output"),
        size: (output_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create conversion shader
    let convert_shader_source = "
        @group(0) @binding(0) var<storage, read> input_u8: array<u32>; // u8 values packed in u32
        @group(0) @binding(1) var<storage, read_write> output_u32: array<u32>;
        
        @compute @workgroup_size(64)
        fn convert_u8_to_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&output_u32)) {
                return;
            }
            
            // For simplicity, treat input as already u32 (since GPU buffers are usually aligned)
            // In practice, u8 tensors are often stored as u32 on GPU anyway
            let packed_index = index / 4u;
            let element_offset = index % 4u;
            
            if (packed_index < arrayLength(&input_u8)) {
                let packed_value = input_u8[packed_index];
                let u8_value = (packed_value >> (element_offset * 8u)) & 0xFFu;
                output_u32[index] = u8_value;
            } else {
                output_u32[index] = 0u;
            }
        }
    ";

    let convert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("u8_to_u32_convert"),
        source: wgpu::ShaderSource::Wgsl(convert_shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("u8_to_u32_convert_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("u8_to_u32_convert_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("u8_to_u32_convert_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let convert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("u8_to_u32_convert_pipeline"),
        layout: Some(&pipeline_layout),
        module: &convert_shader,
        entry_point: Some("convert_u8_to_u32"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Execute conversion
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("u8_to_u32_convert_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("u8_to_u32_convert_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&convert_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (output_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Create result GpuBuffer
    let device_id = match input.device_enum() {
        crate::Device::Gpu(id) => id,
        _ => {
            return Err(crate::TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    Ok(crate::gpu::buffer::GpuBuffer::from_raw_buffer(
        output_buffer,
        device.clone(),
        queue.clone(),
        crate::Device::Gpu(device_id),
        output_len,
    ))
}

/// Convert a GPU buffer of u32 values (0 or 1) to u8 values (duplicate from comparison.rs)
#[cfg(feature = "gpu")]
fn convert_u32_to_u8_gpu_buffer(
    input: crate::gpu::buffer::GpuBuffer<u32>,
) -> Result<crate::gpu::buffer::GpuBuffer<u8>> {
    use crate::TensorError;

    let device = &input.device;
    let queue = &input.queue;
    let output_len = input.len();

    // Create u8 output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u32_to_u8_output"),
        size: (output_len * std::mem::size_of::<u8>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create conversion shader
    let convert_shader_source = "
        @group(0) @binding(0) var<storage, read> input_u32: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output_u8: array<u32>; // u8 values packed into u32
        
        @compute @workgroup_size(64)
        fn convert_u32_to_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let base_index = global_id.x * 4u;
            if (base_index >= arrayLength(&input_u32)) {
                return;
            }
            
            var packed_result = 0u;
            for (var i = 0u; i < 4u; i++) {
                let input_index = base_index + i;
                if (input_index < arrayLength(&input_u32)) {
                    let u8_value = min(input_u32[input_index], 1u);
                    packed_result |= (u8_value << (i * 8u));
                }
            }
            
            output_u8[global_id.x] = packed_result;
        }
    ";

    let convert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("u32_to_u8_convert"),
        source: wgpu::ShaderSource::Wgsl(convert_shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("convert_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("convert_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("convert_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let convert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("convert_pipeline"),
        layout: Some(&pipeline_layout),
        module: &convert_shader,
        entry_point: Some("convert_u32_to_u8"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Execute conversion
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("convert_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("convert_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&convert_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (output_len + 4 * workgroup_size - 1) / (4 * workgroup_size); // Process 4 elements per workgroup
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Create result GpuBuffer
    let device_id = match input.device_enum() {
        crate::Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    Ok(crate::gpu::buffer::GpuBuffer::from_raw_buffer(
        output_buffer,
        device.clone(),
        queue.clone(),
        crate::Device::Gpu(device_id),
        output_len,
    ))
}
