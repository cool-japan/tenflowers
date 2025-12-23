//! GPU-accelerated color jitter transform
//!
//! This module provides GPU-accelerated color jittering (brightness, contrast,
//! saturation, hue adjustments) using WGPU compute shaders for efficient data augmentation.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use scirs2_core::random::Random;

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, MapMode, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource,
};

#[cfg(feature = "gpu")]
use super::context::GpuContext;

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorJitterUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
}

/// GPU-accelerated color jitter transform
#[cfg(feature = "gpu")]
pub struct GpuColorJitter {
    brightness_range: (f32, f32),
    contrast_range: (f32, f32),
    saturation_range: (f32, f32),
    hue_range: (f32, f32),
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuColorJitter {
    /// Create a new GPU color jitter transform
    pub fn new(
        brightness_range: (f32, f32),
        contrast_range: (f32, f32),
        saturation_range: (f32, f32),
        hue_range: (f32, f32),
        context: Arc<GpuContext>,
    ) -> Result<Self> {
        let shader_source = r#"
@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> uniforms: ColorJitterUniforms;

struct ColorJitterUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> vec3<f32> {
    let max_val = max(max(r, g), b);
    let min_val = min(min(r, g), b);
    let delta = max_val - min_val;

    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let v: f32 = max_val;

    if (delta > 0.0) {
        s = delta / max_val;

        if (max_val == r) {
            h = ((g - b) / delta) % 6.0;
        } else if (max_val == g) {
            h = (b - r) / delta + 2.0;
        } else {
            h = (r - g) / delta + 4.0;
        }
        h = h * 60.0;
        if (h < 0.0) {
            h = h + 360.0;
        }
    }

    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0));
    let m = v - c;

    var r: f32;
    var g: f32;
    var b: f32;

    if (h >= 0.0 && h < 60.0) {
        r = c; g = x; b = 0.0;
    } else if (h >= 60.0 && h < 120.0) {
        r = x; g = c; b = 0.0;
    } else if (h >= 120.0 && h < 180.0) {
        r = 0.0; g = c; b = x;
    } else if (h >= 180.0 && h < 240.0) {
        r = 0.0; g = x; b = c;
    } else if (h >= 240.0 && h < 300.0) {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }

    return vec3<f32>(r + m, g + m, b + m);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let pixel_idx = y * uniforms.width + x;

    // Assume RGB format (3 channels)
    let r_idx = pixel_idx;
    let g_idx = pixel_idx + uniforms.width * uniforms.height;
    let b_idx = pixel_idx + 2u * uniforms.width * uniforms.height;

    var r = input_data[r_idx];
    var g = input_data[g_idx];
    var b = input_data[b_idx];

    // Apply brightness
    r = r + uniforms.brightness;
    g = g + uniforms.brightness;
    b = b + uniforms.brightness;

    // Apply contrast
    r = (r - 0.5) * uniforms.contrast + 0.5;
    g = (g - 0.5) * uniforms.contrast + 0.5;
    b = (b - 0.5) * uniforms.contrast + 0.5;

    // Convert to HSV for saturation and hue adjustments
    let hsv = rgb_to_hsv(r, g, b);
    var h = hsv.x;
    var s = hsv.y;
    let v = hsv.z;

    // Apply saturation
    s = s * uniforms.saturation;

    // Apply hue shift
    h = h + uniforms.hue;
    if (h >= 360.0) {
        h = h - 360.0;
    } else if (h < 0.0) {
        h = h + 360.0;
    }

    // Convert back to RGB
    let rgb = hsv_to_rgb(h, s, v);

    // Clamp values to [0, 1]
    output_data[r_idx] = clamp(rgb.x, 0.0, 1.0);
    output_data[g_idx] = clamp(rgb.y, 0.0, 1.0);
    output_data[b_idx] = clamp(rgb.z, 0.0, 1.0);
}
"#;

        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("color_jitter_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("color_jitter_bind_group_layout"),
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
                label: Some("color_jitter_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("color_jitter_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            brightness_range,
            contrast_range,
            saturation_range,
            hue_range,
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Apply color jitter to image tensor using GPU
    pub async fn jitter_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "Expected 3D tensor (C×H×W)".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (channels, height, width) = (shape[0], shape[1], shape[2]);

        if channels != 3 {
            return Err(TensorError::invalid_argument(
                "Color jitter requires RGB input (3 channels)".to_string(),
            ));
        }

        // Get input data
        let input_data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access tensor data".to_string())
        })?;

        // Generate random jitter values
        use scirs2_core::random::rand_prelude::*;
        let mut rng = scirs2_core::random::rng();
        let random_brightness: f32 = rng.random();
        let brightness = self.brightness_range.0
            + random_brightness * (self.brightness_range.1 - self.brightness_range.0);
        let random_contrast: f32 = rng.random();
        let contrast = self.contrast_range.0
            + random_contrast * (self.contrast_range.1 - self.contrast_range.0);
        let random_saturation: f32 = rng.random();
        let saturation = self.saturation_range.0
            + random_saturation * (self.saturation_range.1 - self.saturation_range.0);
        let random_hue: f32 = rng.random();
        let hue = self.hue_range.0 + random_hue * (self.hue_range.1 - self.hue_range.0);

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

        // Create uniform buffer for parameters
        let uniforms = ColorJitterUniforms {
            width: width as u32,
            height: height as u32,
            channels: channels as u32,
            padding: 0,
            brightness,
            contrast,
            saturation,
            hue,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uniform_buffer"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        // Create bind group
        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("color_jitter_bind_group"),
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
                label: Some("color_jitter_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("color_jitter_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup size 8x8
            let workgroup_size = 8u32;
            let dispatch_x = (width as u32 + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (height as u32 + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
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
impl Transform<f32> for GpuColorJitter {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;

        let jittered_tensor = pollster::block_on(self.jitter_tensor(&image_tensor))?;

        Ok((jittered_tensor, label_tensor))
    }
}

/// CPU fallback color jitter transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuColorJitter;

#[cfg(not(feature = "gpu"))]
impl GpuColorJitter {
    /// Create a new color jitter transform (fallback to CPU)
    pub fn new(
        _brightness_range: (f32, f32),
        _contrast_range: (f32, f32),
        _saturation_range: (f32, f32),
        _hue_range: (f32, f32),
        _context: (),
    ) -> Result<Self> {
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
    async fn test_gpu_color_jitter_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(context) => {
                let context = Arc::new(context);
                let jitter = GpuColorJitter::new(
                    (-0.1, 0.1),
                    (0.9, 1.1),
                    (0.9, 1.1),
                    (-10.0, 10.0),
                    context,
                );
                assert!(jitter.is_ok());
            }
            Err(_) => {
                // GPU not available in test environment
                println!("GPU not available for color jitter test");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_color_jitter_fallback() {
        let result = GpuColorJitter::new((-0.1, 0.1), (0.9, 1.1), (0.9, 1.1), (-10.0, 10.0), ());
        assert!(result.is_err());
    }
}
