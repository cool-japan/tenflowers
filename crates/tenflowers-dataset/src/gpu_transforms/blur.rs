//! GPU-accelerated Gaussian blur transform
//!
//! This module provides GPU-accelerated Gaussian blur filtering using WGPU compute shaders
//! for efficient image preprocessing and data augmentation.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "gpu")]
#[allow(unused_imports)]
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
struct GaussianBlurUniforms {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f32,
    padding: f32,
    padding2: f32,
    padding3: f32,
}

/// GPU-accelerated Gaussian blur transform
#[cfg(feature = "gpu")]
pub struct GpuGaussianBlur {
    kernel_size: u32,
    sigma: f32,
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuGaussianBlur {
    /// Create a new GPU Gaussian blur transform
    pub fn new(kernel_size: u32, sigma: f32, context: Arc<GpuContext>) -> Result<Self> {
        let shader_source = r#"
@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> uniforms: GaussianBlurUniforms;

struct GaussianBlurUniforms {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f32,
    padding: f32,
    padding2: f32,
    padding3: f32,
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let c = global_id.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    let half_kernel = i32(uniforms.kernel_size) / 2;
    var sum = 0.0;
    var weight_sum = 0.0;

    for (var ky = -half_kernel; ky <= half_kernel; ky++) {
        for (var kx = -half_kernel; kx <= half_kernel; kx++) {
            let sample_x = i32(x) + kx;
            let sample_y = i32(y) + ky;

            if (sample_x >= 0 && sample_x < i32(uniforms.width) &&
                sample_y >= 0 && sample_y < i32(uniforms.height)) {

                let distance = sqrt(f32(kx * kx + ky * ky));
                let weight = exp(-(distance * distance) / (2.0 * uniforms.sigma * uniforms.sigma));

                let sample_idx = c * uniforms.width * uniforms.height +
                                u32(sample_y) * uniforms.width + u32(sample_x);

                sum += input_data[sample_idx] * weight;
                weight_sum += weight;
            }
        }
    }

    let output_idx = c * uniforms.width * uniforms.height + y * uniforms.width + x;
    output_data[output_idx] = sum / weight_sum;
}
"#;

        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("gaussian_blur_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gaussian_blur_bind_group_layout"),
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
                label: Some("gaussian_blur_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("gaussian_blur_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            kernel_size,
            sigma,
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Apply separable Gaussian blur to a C×H×W image tensor.
    ///
    /// A 1-D Gaussian kernel of length `kernel_size` with standard deviation `sigma`
    /// is applied first horizontally, then vertically (separable convolution), giving
    /// O(k·N) complexity instead of O(k²·N) for a full 2-D kernel.  Pixels outside
    /// the image boundary are treated as zero (zero-padding).
    ///
    /// A native WGPU compute-shader path is deferred; this CPU path is mathematically
    /// correct and is used as the fallback.
    pub async fn blur_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "GpuGaussianBlur: expected 3D tensor (C×H×W)".to_string(),
            ));
        }

        let shape = input.shape().dims();
        let (channels, height, width) = (shape[0], shape[1], shape[2]);
        let k = self.kernel_size as usize;

        if k == 0 {
            return Err(TensorError::invalid_argument(
                "GpuGaussianBlur: kernel_size must be ≥ 1".to_string(),
            ));
        }

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("GpuGaussianBlur: cannot access tensor data".to_string())
        })?;

        // Build normalised 1-D Gaussian kernel.
        let half = (k / 2) as isize;
        let sigma = self.sigma;
        let mut kernel: Vec<f32> = (0..k)
            .map(|i| {
                let x = i as f32 - half as f32;
                (-0.5 * (x / sigma) * (x / sigma)).exp()
            })
            .collect();
        let kernel_sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|v| *v /= kernel_sum);

        // Horizontal pass: convolve along width axis.
        let mut h_pass = vec![0.0f32; channels * height * width];
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let mut acc = 0.0f32;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sx = x as isize + ki as isize - half;
                        if sx >= 0 && sx < width as isize {
                            let idx = c * height * width + y * width + sx as usize;
                            acc += data[idx] * kv;
                        }
                    }
                    h_pass[c * height * width + y * width + x] = acc;
                }
            }
        }

        // Vertical pass: convolve along height axis.
        let mut out = vec![0.0f32; channels * height * width];
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let mut acc = 0.0f32;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sy = y as isize + ki as isize - half;
                        if sy >= 0 && sy < height as isize {
                            let idx = c * height * width + sy as usize * width + x;
                            acc += h_pass[idx] * kv;
                        }
                    }
                    out[c * height * width + y * width + x] = acc;
                }
            }
        }

        Tensor::from_vec(out, &[channels, height, width])
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuGaussianBlur {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;
        let blurred_tensor = pollster::block_on(self.blur_tensor(&image_tensor))?;
        Ok((blurred_tensor, label_tensor))
    }
}

/// CPU fallback blur transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuGaussianBlur;

#[cfg(not(feature = "gpu"))]
impl GpuGaussianBlur {
    /// Create a new blur transform (fallback to CPU)
    pub fn new(_kernel_size: u32, _sigma: f32, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GPU transforms require 'gpu' feature to be enabled".to_string(),
        ))
    }
}
