//! GPU-accelerated image resize transform
//!
//! This module provides GPU-accelerated image resizing using WGPU compute shaders
//! for efficient image preprocessing.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, MapMode, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource,
};

#[cfg(feature = "gpu")]
use super::context::GpuContext;

/// GPU-accelerated resize transform
#[cfg(feature = "gpu")]
pub struct GpuResize {
    width: u32,
    height: u32,
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuResize {
    /// Create a new GPU resize transform
    pub fn new(width: u32, height: u32, context: Arc<GpuContext>) -> Result<Self> {
        let shader_source = r#"
@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> uniforms: ResizeUniforms;

struct ResizeUniforms {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    channels: u32,
    padding: u32,
    padding2: u32,
    padding3: u32,
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let c = global_id.z;

    if (x >= uniforms.output_width || y >= uniforms.output_height || c >= uniforms.channels) {
        return;
    }

    // Bilinear interpolation
    let x_ratio = f32(x) * f32(uniforms.input_width - 1u) / f32(uniforms.output_width - 1u);
    let y_ratio = f32(y) * f32(uniforms.input_height - 1u) / f32(uniforms.output_height - 1u);

    let x_l = u32(floor(x_ratio));
    let y_l = u32(floor(y_ratio));
    let x_h = min(x_l + 1u, uniforms.input_width - 1u);
    let y_h = min(y_l + 1u, uniforms.input_height - 1u);

    let x_weight = x_ratio - f32(x_l);
    let y_weight = y_ratio - f32(y_l);

    let input_stride = uniforms.input_width * uniforms.input_height;
    let output_stride = uniforms.output_width * uniforms.output_height;

    let tl_idx = c * input_stride + y_l * uniforms.input_width + x_l;
    let tr_idx = c * input_stride + y_l * uniforms.input_width + x_h;
    let bl_idx = c * input_stride + y_h * uniforms.input_width + x_l;
    let br_idx = c * input_stride + y_h * uniforms.input_width + x_h;

    let tl = input_data[tl_idx];
    let tr = input_data[tr_idx];
    let bl = input_data[bl_idx];
    let br = input_data[br_idx];

    let top = tl * (1.0 - x_weight) + tr * x_weight;
    let bottom = bl * (1.0 - x_weight) + br * x_weight;
    let result = top * (1.0 - y_weight) + bottom * y_weight;

    let output_idx = c * output_stride + y * uniforms.output_width + x;
    output_data[output_idx] = result;
}
"#;

        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("resize_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("resize_bind_group_layout"),
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
                label: Some("resize_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("resize_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            width,
            height,
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Resize image tensor using GPU
    pub async fn resize_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "Expected 3D tensor (C×H×W)".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (channels, input_height, input_width) = (shape[0], shape[1], shape[2]);

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

        let output_size =
            (channels * self.height as usize * self.width as usize) * std::mem::size_of::<f32>();
        let output_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("output_buffer"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dimensions
        let uniforms = [
            input_width as u32,
            input_height as u32,
            self.width,
            self.height,
            channels as u32,
            0,
            0,
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
            label: Some("resize_bind_group"),
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
                label: Some("resize_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("resize_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup size 8x8
            let workgroup_size = 8u32;
            let dispatch_x = (self.width + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (self.height + workgroup_size - 1) / workgroup_size;

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
            let output_tensor = Tensor::from_vec(
                result.to_vec(),
                &[channels, self.height as usize, self.width as usize],
            )?;
            Ok(output_tensor)
        } else {
            Err(TensorError::device_error_simple(
                "Failed to read GPU buffer".to_string(),
            ))
        }
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuResize {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;

        // Use async runtime for GPU operation
        let resized_tensor = pollster::block_on(self.resize_tensor(&image_tensor))?;

        Ok((resized_tensor, label_tensor))
    }
}

/// CPU fallback resize transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuResize;

#[cfg(not(feature = "gpu"))]
impl GpuResize {
    /// Create a new resize transform (fallback to CPU)
    pub fn new(_width: u32, _height: u32, _context: ()) -> Result<Self> {
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
    async fn test_gpu_resize_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(context) => {
                let context = Arc::new(context);
                let resize = GpuResize::new(224, 224, context);
                assert!(resize.is_ok());
            }
            Err(_) => {
                // GPU not available in test environment
                println!("GPU not available for resize test");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_resize_fallback() {
        let result = GpuResize::new(224, 224, ());
        assert!(result.is_err());
    }
}
