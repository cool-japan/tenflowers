//! GPU-accelerated random horizontal flip transform
//!
//! This module provides GPU-accelerated random horizontal flipping using WGPU compute shaders
//! for efficient data augmentation.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use scirs2_core::random::Random;

#[cfg(feature = "gpu")]
use scirs2_core::Rng;

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, MapMode, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource,
};

#[cfg(feature = "gpu")]
use super::context::GpuContext;

/// GPU-accelerated random horizontal flip
#[cfg(feature = "gpu")]
pub struct GpuRandomHorizontalFlip {
    probability: f32,
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuRandomHorizontalFlip {
    /// Create a new GPU random horizontal flip transform
    pub fn new(probability: f32, context: Arc<GpuContext>) -> Result<Self> {
        let shader_source = r#"
@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> uniforms: FlipUniforms;

struct FlipUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let c = global_id.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    // Horizontal flip: flip x coordinate
    let flipped_x = uniforms.width - 1u - x;

    let input_idx = c * uniforms.width * uniforms.height + y * uniforms.width + x;
    let output_idx = c * uniforms.width * uniforms.height + y * uniforms.width + flipped_x;

    output_data[output_idx] = input_data[input_idx];
}
"#;

        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("flip_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("flip_bind_group_layout"),
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = context
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("flip_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("flip_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            probability,
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Apply horizontal flip to image tensor using GPU
    pub async fn flip_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "Expected 3D tensor (C×H×W)".to_string(),
            ));
        }

        // Check if we should flip
        let mut rng = scirs2_core::random::rng();
        let should_flip = rng.gen_range(0.0..1.0) < self.probability;

        if !should_flip {
            return Ok(input.clone());
        }

        let shape = input.shape().dims();
        let (channels, height, width) = (shape[0], shape[1], shape[2]);

        // Get input data
        let input_data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access tensor data".to_string())
        })?;

        // Create buffers
        let input_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("input_buffer"),
                    contents: bytemuck::cast_slice(input_data),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

        let output_size = std::mem::size_of_val(input_data);
        let output_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("output_buffer"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dimensions
        let uniforms = [
            width as u32,
            height as u32,
            channels as u32,
            0, // padding
        ];
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uniform_buffer"),
                    contents: bytemuck::cast_slice(&uniforms),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        // Create bind group
        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("flip_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch compute shader
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("flip_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("flip_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup size 8x8
            let workgroup_size = 8u32;
            let dispatch_x = (width as u32 + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (height as u32 + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, channels as u32);
        }

        // Create staging buffer and copy result
        let staging_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("staging_buffer"),
            size: output_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);

        // Submit and wait
        self.context.queue.submit(std::iter::once(encoder.finish()));

        // Map and read result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |v| {
            if sender.send(v).is_err() {
                eprintln!("Warning: Failed to send GPU buffer read result");
            }
        });
        self.context.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);
            let output_tensor = Tensor::from_vec(result.to_vec(), &[channels, height, width])?;
            Ok(output_tensor)
        } else {
            Err(TensorError::device_error_simple(
                "Failed to read GPU buffer".to_string(),
            ))
        }
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuRandomHorizontalFlip {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;

        let flipped_tensor = pollster::block_on(self.flip_tensor(&image_tensor))?;

        Ok((flipped_tensor, label_tensor))
    }
}

/// CPU fallback flip transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuRandomHorizontalFlip;

#[cfg(not(feature = "gpu"))]
impl GpuRandomHorizontalFlip {
    /// Create a new flip transform (fallback to CPU)
    pub fn new(_probability: f32, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GPU transforms require 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_flip_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(context) => {
                let context = Arc::new(context);
                let flip = GpuRandomHorizontalFlip::new(0.5, context);
                assert!(flip.is_ok());
            }
            Err(_) => {
                // GPU not available in test environment
                println!("GPU not available for flip test");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_flip_fallback() {
        let result = GpuRandomHorizontalFlip::new(0.5, ());
        assert!(result.is_err());
    }
}
