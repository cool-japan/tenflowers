/*!
 * Memory coalescing optimization for GPU operations
 *
 * This module provides optimized memory access patterns for GPU operations
 * to improve performance through better memory coalescing.
 */

use crate::gpu::{buffer::GpuBuffer, ops::BinaryOp};
use crate::{Result, TensorError};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Memory coalescing strategies for GPU operations
#[derive(Debug, Clone, Copy)]
pub enum CoalescingStrategy {
    /// Linear 1D access pattern (default)
    Linear,
    /// 2D tiled access pattern with shared memory
    Tiled2D,
    /// Vectorized access pattern for large tensors
    Vectorized,
    /// Bank-conflict free shared memory access
    BankConflictFree,
    /// Adaptive strategy based on tensor characteristics
    Adaptive,
}

/// Tensor characteristics for coalescing strategy selection
#[derive(Debug)]
pub struct TensorCharacteristics {
    pub total_elements: usize,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub is_contiguous: bool,
    pub dtype_size: usize,
}

impl TensorCharacteristics {
    /// Create tensor characteristics from shape
    pub fn from_shape(shape: &[usize], dtype_size: usize) -> Self {
        let total_elements = shape.iter().product();
        let (width, height, depth) = match shape.len() {
            1 => (shape[0], 1, 1),
            2 => (shape[1], shape[0], 1),
            3 => (shape[2], shape[1], shape[0]),
            _ => {
                let width = shape[shape.len() - 1];
                let height = shape[shape.len() - 2];
                let depth = shape[0..shape.len() - 2].iter().product();
                (width, height, depth)
            }
        };

        Self {
            total_elements,
            width,
            height,
            depth,
            is_contiguous: true, // Assume contiguous for now
            dtype_size,
        }
    }

    /// Select optimal coalescing strategy based on tensor characteristics
    pub fn select_optimal_strategy(&self) -> CoalescingStrategy {
        // Large tensors benefit from vectorized access
        if self.total_elements > 1_000_000 {
            return CoalescingStrategy::Vectorized;
        }

        // 2D/3D tensors benefit from tiled access
        if self.width > 16 && self.height > 16 {
            return CoalescingStrategy::Tiled2D;
        }

        // Small tensors or 1D tensors use linear access
        if self.total_elements < 1024 || self.height == 1 {
            return CoalescingStrategy::Linear;
        }

        // Default to adaptive for medium-sized tensors
        CoalescingStrategy::Adaptive
    }
}

/// Optimized binary operation with memory coalescing
pub fn execute_binary_op_coalesced<T>(
    input_a: &GpuBuffer<T>,
    input_b: &GpuBuffer<T>,
    operation: BinaryOp,
    output_len: usize,
    shape: &[usize],
    strategy: CoalescingStrategy,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device = &input_a.device;
    let queue = &input_a.queue;

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Binary Op Coalesced Output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create shape metadata buffer
    let characteristics = TensorCharacteristics::from_shape(shape, std::mem::size_of::<T>());
    let shape_metadata = [
        characteristics.width as u32,
        characteristics.height as u32,
        characteristics.depth as u32,
        1u32, // batch_size
    ];

    let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Shape Metadata"),
        contents: bytemuck::cast_slice(&shape_metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Select shader and dispatch configuration based on strategy
    let (shader_source, dispatch_config) =
        select_shader_and_config::<T>(strategy, &characteristics, operation);

    // Create compute pipeline
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Binary Op Coalesced Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Binary Op Coalesced Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Binary Op Coalesced Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Binary Op Coalesced Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: Some(dispatch_config.entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Binary Op Coalesced Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shape_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Binary Op Coalesced Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Binary Op Coalesced Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            dispatch_config.workgroups_x,
            dispatch_config.workgroups_y,
            dispatch_config.workgroups_z,
        );
    }

    queue.submit(Some(encoder.finish()));

    // Create result buffer
    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input_a.device),
        Arc::clone(&input_a.queue),
        input_a.device_enum().clone(),
        output_len,
    ))
}

/// Dispatch configuration for different coalescing strategies
#[derive(Debug)]
struct DispatchConfig {
    entry_point: &'static str,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
}

/// Select shader source and dispatch configuration based on strategy and data type
fn select_shader_and_config<T>(
    strategy: CoalescingStrategy,
    characteristics: &TensorCharacteristics,
    operation: BinaryOp,
) -> (&'static str, DispatchConfig) {
    let type_name = std::any::type_name::<T>();

    let entry_point = match operation {
        BinaryOp::Add => match strategy {
            CoalescingStrategy::Linear => "add_op",
            CoalescingStrategy::Tiled2D => match type_name {
                "f32" => "add_op_coalesced",
                "f64" => "add_op_coalesced_f64",
                "i32" => "add_op_coalesced_i32",
                "i64" => "add_op_coalesced_i64",
                "u32" => "add_op_coalesced_u32",
                "u64" => "add_op_coalesced_u64",
                _ => "add_op_coalesced", // fallback to f32
            },
            CoalescingStrategy::Vectorized => match type_name {
                "f32" => "add_op_vectorized",
                "f64" => "add_op_vectorized_f64",
                "i32" => "add_op_vectorized_i32",
                "i64" => "add_op_vectorized_i64",
                "u32" => "add_op_vectorized_u32",
                "u64" => "add_op_vectorized_u64",
                _ => "add_op_vectorized", // fallback to f32
            },
            CoalescingStrategy::BankConflictFree => "add_op_bank_conflict_free",
            CoalescingStrategy::Adaptive => "add_op_adaptive",
        },
        BinaryOp::Sub => match strategy {
            CoalescingStrategy::Tiled2D => match type_name {
                "f32" => "sub_op_coalesced",
                "f64" => "sub_op_coalesced_f64",
                "i32" => "sub_op_coalesced_i32",
                "i64" => "sub_op_coalesced_i64",
                "u32" => "sub_op_coalesced_u32",
                "u64" => "sub_op_coalesced_u64",
                _ => "sub_op_coalesced", // fallback to f32
            },
            _ => "sub_op", // Fallback for other strategies
        },
        BinaryOp::Mul => match strategy {
            CoalescingStrategy::Tiled2D => match type_name {
                "f32" => "mul_op_coalesced",
                "f64" => "mul_op_coalesced_f64",
                "i32" => "mul_op_coalesced_i32",
                "i64" => "mul_op_coalesced_i64",
                "u32" => "mul_op_coalesced_u32",
                "u64" => "mul_op_coalesced_u64",
                _ => "mul_op_coalesced", // fallback to f32
            },
            _ => "mul_op", // Fallback for other strategies
        },
        BinaryOp::Div => match strategy {
            CoalescingStrategy::Tiled2D => match type_name {
                "f32" => "div_op_coalesced",
                "f64" => "div_op_coalesced_f64",
                "i32" => "div_op_coalesced_i32",
                "i64" => "div_op_coalesced_i64",
                "u32" => "div_op_coalesced_u32",
                "u64" => "div_op_coalesced_u64",
                _ => "div_op_coalesced", // fallback to f32
            },
            _ => "div_op", // Fallback for other strategies
        },
        BinaryOp::Pow => match strategy {
            CoalescingStrategy::Tiled2D => match type_name {
                "f32" => "pow_op_coalesced",
                "f64" => "pow_op_coalesced_f64",
                "i32" => "pow_op_coalesced_i32",
                "i64" => "pow_op_coalesced_i64",
                "u32" => "pow_op_coalesced_u32",
                "u64" => "pow_op_coalesced_u64",
                _ => "pow_op_coalesced", // fallback to f32
            },
            _ => "pow_op", // Fallback for other strategies
        },
        BinaryOp::PReLU => match strategy {
            CoalescingStrategy::Tiled2D => match type_name {
                "f32" => "prelu_op_coalesced",
                "f64" => "prelu_op_coalesced_f64",
                "i32" => "prelu_op_coalesced_i32",
                "i64" => "prelu_op_coalesced_i64",
                "u32" => "prelu_op_coalesced_u32",
                "u64" => "prelu_op_coalesced_u64",
                _ => "prelu_op_coalesced", // fallback to f32
            },
            _ => "prelu_op", // Fallback for other strategies
        },
        _ => "add_op", // Default fallback
    };

    let dispatch_config = match strategy {
        CoalescingStrategy::Linear => DispatchConfig {
            entry_point,
            workgroups_x: ((characteristics.total_elements + 63) / 64) as u32,
            workgroups_y: 1,
            workgroups_z: 1,
        },
        CoalescingStrategy::Tiled2D => {
            // Adjust workgroup size based on data type
            let (tile_x, tile_y) = match type_name {
                "f64" | "i64" | "u64" => (16, 8), // Smaller tiles for 64-bit types
                _ => (16, 16),                    // Default tiles for 32-bit types
            };
            DispatchConfig {
                entry_point,
                workgroups_x: ((characteristics.width + tile_x - 1) / tile_x) as u32,
                workgroups_y: ((characteristics.height + tile_y - 1) / tile_y) as u32,
                workgroups_z: characteristics.depth.max(1) as u32,
            }
        }
        CoalescingStrategy::Vectorized => {
            // Adjust vectorization based on data type
            let vector_size = match type_name {
                "f64" | "i64" | "u64" => 2, // Process 2 elements for 64-bit types
                _ => 4,                     // Process 4 elements for 32-bit types
            };
            DispatchConfig {
                entry_point,
                workgroups_x: ((characteristics.total_elements + (64 * vector_size - 1))
                    / (64 * vector_size)) as u32,
                workgroups_y: 1,
                workgroups_z: 1,
            }
        }
        CoalescingStrategy::BankConflictFree => DispatchConfig {
            entry_point,
            workgroups_x: ((characteristics.width + 31) / 32) as u32,
            workgroups_y: ((characteristics.height + 7) / 8) as u32,
            workgroups_z: 1,
        },
        CoalescingStrategy::Adaptive => DispatchConfig {
            entry_point,
            workgroups_x: ((characteristics.total_elements + 255) / 256) as u32,
            workgroups_y: 1,
            workgroups_z: 1,
        },
    };

    let shader_source = match type_name {
        "f32" => include_str!("shaders/binary_ops_coalesced.wgsl"),
        "f64" => include_str!("shaders/binary_ops_coalesced_f64.wgsl"),
        "i32" => include_str!("shaders/binary_ops_coalesced_i32.wgsl"),
        "i64" => include_str!("shaders/binary_ops_coalesced_i64.wgsl"),
        "u32" => include_str!("shaders/binary_ops_coalesced_u32.wgsl"),
        "u64" => include_str!("shaders/binary_ops_coalesced_u64.wgsl"),
        _ => include_str!("shaders/binary_ops_coalesced.wgsl"), // fallback to f32
    };

    (shader_source, dispatch_config)
}

/// Benchmark different coalescing strategies to find the best one
pub fn benchmark_coalescing_strategies<T>(
    input_a: &GpuBuffer<T>,
    input_b: &GpuBuffer<T>,
    operation: BinaryOp,
    output_len: usize,
    shape: &[usize],
) -> Result<CoalescingStrategy>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let characteristics = TensorCharacteristics::from_shape(shape, std::mem::size_of::<T>());
    let strategies = [
        CoalescingStrategy::Linear,
        CoalescingStrategy::Tiled2D,
        CoalescingStrategy::Vectorized,
        CoalescingStrategy::BankConflictFree,
        CoalescingStrategy::Adaptive,
    ];

    let mut best_strategy = CoalescingStrategy::Linear;
    let mut best_time = std::time::Duration::MAX;

    for strategy in strategies.iter() {
        let start = std::time::Instant::now();

        // Run the operation multiple times for better timing
        for _ in 0..10 {
            let _result = execute_binary_op_coalesced(
                input_a, input_b, operation, output_len, shape, *strategy,
            )?;
        }

        let elapsed = start.elapsed();
        if elapsed < best_time {
            best_time = elapsed;
            best_strategy = *strategy;
        }
    }

    Ok(best_strategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_characteristics_strategy_selection() {
        // Small tensor should use linear
        let small_chars = TensorCharacteristics::from_shape(&[100], 4);
        assert!(matches!(
            small_chars.select_optimal_strategy(),
            CoalescingStrategy::Linear
        ));

        // Large tensor should use vectorized
        let large_chars = TensorCharacteristics::from_shape(&[2000, 2000], 4);
        assert!(matches!(
            large_chars.select_optimal_strategy(),
            CoalescingStrategy::Vectorized
        ));

        // 2D tensor should use tiled
        let tiled_chars = TensorCharacteristics::from_shape(&[100, 100], 4);
        assert!(matches!(
            tiled_chars.select_optimal_strategy(),
            CoalescingStrategy::Tiled2D
        ));
    }

    #[test]
    fn test_dispatch_config_calculation() {
        let chars = TensorCharacteristics::from_shape(&[64, 64], 4);
        let (_, config) =
            select_shader_and_config::<f32>(CoalescingStrategy::Tiled2D, &chars, BinaryOp::Add);

        assert_eq!(config.workgroups_x, 4); // (64 + 15) / 16
        assert_eq!(config.workgroups_y, 4); // (64 + 15) / 16
        assert_eq!(config.workgroups_z, 1);
    }
}
