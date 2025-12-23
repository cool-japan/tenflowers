use crate::tensor::TensorStorage;
use crate::{Result, Shape, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, Zip};
use scirs2_core::numeric::Zero;

#[cfg(feature = "gpu")]
use crate::gpu::gpu_comparison_op_dispatch;

/// Comparison operation trait
pub trait ComparisonOp<T> {
    fn apply(&self, a: T, b: T) -> bool;
    fn name(&self) -> &str;
}

/// Equal operation
pub struct EqOp;
impl<T: PartialEq> ComparisonOp<T> for EqOp {
    fn apply(&self, a: T, b: T) -> bool {
        a == b
    }
    fn name(&self) -> &str {
        "Eq"
    }
}

/// Not equal operation
pub struct NeOp;
impl<T: PartialEq> ComparisonOp<T> for NeOp {
    fn apply(&self, a: T, b: T) -> bool {
        a != b
    }
    fn name(&self) -> &str {
        "Ne"
    }
}

/// Less than operation
pub struct LtOp;
impl<T: PartialOrd> ComparisonOp<T> for LtOp {
    fn apply(&self, a: T, b: T) -> bool {
        a < b
    }
    fn name(&self) -> &str {
        "Lt"
    }
}

/// Less than or equal operation
pub struct LeOp;
impl<T: PartialOrd> ComparisonOp<T> for LeOp {
    fn apply(&self, a: T, b: T) -> bool {
        a <= b
    }
    fn name(&self) -> &str {
        "Le"
    }
}

/// Greater than operation
pub struct GtOp;
impl<T: PartialOrd> ComparisonOp<T> for GtOp {
    fn apply(&self, a: T, b: T) -> bool {
        a > b
    }
    fn name(&self) -> &str {
        "Gt"
    }
}

/// Greater than or equal operation
pub struct GeOp;
impl<T: PartialOrd> ComparisonOp<T> for GeOp {
    fn apply(&self, a: T, b: T) -> bool {
        a >= b
    }
    fn name(&self) -> &str {
        "Ge"
    }
}

/// Generic comparison operation implementation with broadcasting
/// Returns a boolean tensor (represented as u8: 0=false, 1=true)
pub fn comparison_op<T, Op>(a: &Tensor<T>, b: &Tensor<T>, op: Op) -> Result<Tensor<u8>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    Op: ComparisonOp<T>,
{
    // Check device compatibility
    if a.device() != b.device() {
        return Err(TensorError::device_mismatch(
            "comparison_op",
            &a.device().to_string(),
            &b.device().to_string(),
        ));
    }

    // Compute broadcast shape
    let broadcast_shape =
        a.shape()
            .broadcast_shape(b.shape())
            .ok_or_else(|| TensorError::ShapeMismatch {
                operation: "comparison_op".to_string(),
                expected: a.shape().to_string(),
                got: b.shape().to_string(),
                context: None,
            })?;

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(arr_a), TensorStorage::Cpu(arr_b)) => {
            // Broadcast arrays to common shape
            let a_broadcast = broadcast_array(arr_a, &broadcast_shape)?;
            let b_broadcast = broadcast_array(arr_b, &broadcast_shape)?;

            // Apply comparison element-wise
            let mut result = ArrayD::zeros(a_broadcast.raw_dim());
            Zip::from(&mut result)
                .and(&a_broadcast)
                .and(&b_broadcast)
                .for_each(|r, a_val, b_val| {
                    *r = if op.apply(*a_val, *b_val) { 1u8 } else { 0u8 };
                });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            gpu_comparison_op_impl(gpu_a, gpu_b, &op, a.shape(), b.shape(), &broadcast_shape)
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!("Device mismatch should have been caught earlier"),
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

// Concrete implementations using the generic comparison_op

pub fn eq<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<u8>>
where
    T: Clone
        + Default
        + Zero
        + PartialEq
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    comparison_op(a, b, EqOp)
}

pub fn ne<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<u8>>
where
    T: Clone
        + Default
        + Zero
        + PartialEq
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    comparison_op(a, b, NeOp)
}

pub fn lt<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<u8>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    comparison_op(a, b, LtOp)
}

pub fn le<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<u8>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    comparison_op(a, b, LeOp)
}

pub fn gt<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<u8>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    comparison_op(a, b, GtOp)
}

pub fn ge<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<u8>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    comparison_op(a, b, GeOp)
}

/// GPU comparison operation implementation for supported types
#[cfg(feature = "gpu")]
fn gpu_comparison_op_impl<T>(
    gpu_a: &crate::gpu::buffer::GpuBuffer<T>,
    gpu_b: &crate::gpu::buffer::GpuBuffer<T>,
    op: &dyn ComparisonOp<T>,
    shape_a: &Shape,
    shape_b: &Shape,
    broadcast_shape: &Shape,
) -> Result<Tensor<u8>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Currently, we support f32 and i32 for GPU comparison operations
    let type_name = std::any::type_name::<T>();

    let gpu_op = match op.name() {
        "Eq" => crate::gpu::ops::ComparisonOp::Eq,
        "Ne" => crate::gpu::ops::ComparisonOp::Ne,
        "Lt" => crate::gpu::ops::ComparisonOp::Lt,
        "Le" => crate::gpu::ops::ComparisonOp::Le,
        "Gt" => crate::gpu::ops::ComparisonOp::Gt,
        "Ge" => crate::gpu::ops::ComparisonOp::Ge,
        _ => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "GPU comparison operation {} not implemented",
                op.name()
            )))
        }
    };

    if type_name == "f32" {
        // Cast to f32 buffers for the actual GPU operation
        let gpu_a_f32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<f32>,
            >(gpu_a)
        };
        let gpu_b_f32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<f32>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();

        let result_buffer_u32 = if gpu_a.len() == gpu_b.len() && gpu_a.len() == output_len {
            crate::gpu::ops::execute_comparison_op(gpu_a_f32, gpu_b_f32, gpu_op, output_len)?
        } else {
            // Use broadcasting version
            crate::gpu::ops::execute_comparison_op_with_broadcasting(
                gpu_a_f32,
                gpu_b_f32,
                gpu_op,
                shape_a.dims(),
                shape_b.dims(),
                broadcast_shape.dims(),
                output_len,
            )?
        };

        // Convert u32 result to u8
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "i32" {
        // Cast to i32 buffers for the actual GPU operation
        let gpu_a_i32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i32>,
            >(gpu_a)
        };
        let gpu_b_i32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i32>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();

        let result_buffer_u32 = if gpu_a.len() == gpu_b.len() && gpu_a.len() == output_len {
            crate::gpu::ops::execute_comparison_op(gpu_a_i32, gpu_b_i32, gpu_op, output_len)?
        } else {
            // Use broadcasting version
            crate::gpu::ops::execute_comparison_op_with_broadcasting(
                gpu_a_i32,
                gpu_b_i32,
                gpu_op,
                shape_a.dims(),
                shape_b.dims(),
                broadcast_shape.dims(),
                output_len,
            )?
        };

        // Convert u32 result to u8
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "i64" {
        let gpu_a_i64 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i64>,
            >(gpu_a)
        };
        let gpu_b_i64 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i64>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();

        let result_buffer_u32 = if gpu_a.len() == gpu_b.len() && gpu_a.len() == output_len {
            crate::gpu::ops::execute_comparison_op(gpu_a_i64, gpu_b_i64, gpu_op, output_len)?
        } else {
            // Broadcasting not yet implemented for GPU comparison operations
            // For now, use the regular comparison op
            crate::gpu::ops::execute_comparison_op(gpu_a_i64, gpu_b_i64, gpu_op, output_len)?
        };

        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "f64" {
        let gpu_a_f64 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<f64>,
            >(gpu_a)
        };
        let gpu_b_f64 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<f64>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();

        let result_buffer_u32 = if gpu_a.len() == gpu_b.len() && gpu_a.len() == output_len {
            crate::gpu::ops::execute_comparison_op(gpu_a_f64, gpu_b_f64, gpu_op, output_len)?
        } else {
            // Use broadcasting version
            crate::gpu::ops::execute_comparison_op_with_broadcasting(
                gpu_a_f64,
                gpu_b_f64,
                gpu_op,
                shape_a.dims(),
                shape_b.dims(),
                broadcast_shape.dims(),
                output_len,
            )?
        };

        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "u32" {
        let gpu_a_u32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u32>,
            >(gpu_a)
        };
        let gpu_b_u32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u32>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();
        let result_buffer_u32 =
            crate::gpu::ops::execute_comparison_op(gpu_a_u32, gpu_b_u32, gpu_op, output_len)?;
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "u64" {
        let gpu_a_u64 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u64>,
            >(gpu_a)
        };
        let gpu_b_u64 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u64>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();
        let result_buffer_u32 =
            crate::gpu::ops::execute_comparison_op(gpu_a_u64, gpu_b_u64, gpu_op, output_len)?;
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "i16" {
        let gpu_a_i16 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i16>,
            >(gpu_a)
        };
        let gpu_b_i16 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i16>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();
        let result_buffer_u32 =
            crate::gpu::ops::execute_comparison_op(gpu_a_i16, gpu_b_i16, gpu_op, output_len)?;
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "u16" {
        let gpu_a_u16 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u16>,
            >(gpu_a)
        };
        let gpu_b_u16 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u16>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();
        let result_buffer_u32 =
            crate::gpu::ops::execute_comparison_op(gpu_a_u16, gpu_b_u16, gpu_op, output_len)?;
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "i8" {
        let gpu_a_i8 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i8>,
            >(gpu_a)
        };
        let gpu_b_i8 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<i8>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();
        let result_buffer_u32 =
            crate::gpu::ops::execute_comparison_op(gpu_a_i8, gpu_b_i8, gpu_op, output_len)?;
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else if type_name == "u8" {
        let gpu_a_u8 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u8>,
            >(gpu_a)
        };
        let gpu_b_u8 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<u8>,
            >(gpu_b)
        };

        let output_len: usize = broadcast_shape.dims().iter().product();
        let result_buffer_u32 =
            crate::gpu::ops::execute_comparison_op(gpu_a_u8, gpu_b_u8, gpu_op, output_len)?;
        let result_buffer = convert_u32_to_u8_gpu_buffer(result_buffer_u32)?;

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            broadcast_shape.clone(),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(
            format!("GPU comparison operations only support f32, i32, i64, f64, u32, u64, i16, u16, i8, u8, got {}", std::any::type_name::<T>())
        ))
    }
}

#[cfg(test)]
#[allow(irrefutable_let_patterns)] // Pattern matching on TensorStorage is irrefutable when GPU feature is disabled
mod tests {
    use super::*;

    #[test]
    fn test_eq_same_shape() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 4.0], &[3]).unwrap();

        let c = eq(&a, &b).unwrap();
        let expected = vec![1u8, 1u8, 0u8];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_lt_broadcast() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![2.0, 1.0], &[1, 2]).unwrap();

        let c = lt(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[3, 2]);

        // Expected: [[1, 0], [0, 0], [0, 0]] (1.0 < 2.0, but not < 1.0, etc.)
        let expected = vec![1u8, 0u8, 0u8, 0u8, 0u8, 0u8];
        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_ne_integers() {
        let a = Tensor::<i32>::from_vec(vec![1, 2, 3], &[3]).unwrap();
        let b = Tensor::<i32>::from_vec(vec![1, 3, 3], &[3]).unwrap();

        let c = ne(&a, &b).unwrap();
        let expected = vec![0u8, 1u8, 0u8];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_ge_scalar_broadcast() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let scalar = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();

        let c = ge(&a, &scalar).unwrap();
        let expected = vec![0u8, 1u8, 1u8];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }
}

/// Convert a GPU buffer of u32 values (0 or 1) to u8 values
#[cfg(feature = "gpu")]
fn convert_u32_to_u8_gpu_buffer(
    input: crate::gpu::buffer::GpuBuffer<u32>,
) -> Result<crate::gpu::buffer::GpuBuffer<u8>> {
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
    let device_id = match &input.device_enum() {
        crate::Device::Gpu(id) => *id,
        _ => {
            return Err(crate::TensorError::DeviceError {
                operation: "u32_to_u8".to_string(),
                details: "Expected GPU device".to_string(),
                device: "Unknown".to_string(),
                context: None,
            })
        }
    };

    Ok(crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
        output_buffer,
        device.clone(),
        queue.clone(),
        crate::Device::Gpu(device_id),
        output_len,
    ))
}
