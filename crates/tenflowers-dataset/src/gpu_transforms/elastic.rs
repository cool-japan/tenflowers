//! GPU-accelerated elastic distortion transform
//!
//! Implements SimoNikolenko-style elastic deformation for data augmentation,
//! as described in:
//!
//! > Simard, Steinkraus, Platt — "Best Practices for Convolutional Neural
//! > Networks Applied to Visual Document Analysis", ICDAR 2003.
//!
//! The procedure is:
//! 1. Generate random displacement fields `dx[y, x]` and `dy[y, x]` sampled
//!    from `Uniform(-1, 1)`.
//! 2. Apply a Gaussian low-pass filter (kernel size `k`, standard deviation
//!    `sigma`) to both fields on the CPU to produce smooth displacements.
//! 3. Scale by `alpha` (controls distortion magnitude).
//! 4. Upload the smoothed fields alongside the image to the GPU.
//! 5. Each GPU thread at `(x, y)` bilinearly samples from
//!    `(x + dx[y, x], y + dy[y, x])`.
//!
//! # Notes
//!
//! The GPU shader receives a combined buffer:
//! ```text
//! [image_data (C * H * W), dx_field (H * W), dy_field (H * W)]
//! ```
//! The `alpha` parameter is already baked into the displacement fields by the
//! CPU pre-processing step, so the shader just reads them directly.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

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
use scirs2_core::random::rand_prelude::*;

#[cfg(feature = "gpu")]
use super::context::GpuContext;

// ─────────────────────────────────────────────────────────────────────────────
// CPU helpers: Gaussian kernel + separable convolution
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 1-D Gaussian kernel of length `2*radius + 1`.
fn gaussian_kernel_1d(sigma: f32, radius: usize) -> Vec<f32> {
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let v = (-(x * x) / two_sigma_sq).exp();
        kernel.push(v);
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Separable 2-D Gaussian convolution on a single-channel f32 field (row-major,
/// height × width).  Out-of-bounds accesses are handled by clamping (reflect).
fn gaussian_smooth(field: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel = gaussian_kernel_1d(sigma, radius);

    // --- horizontal pass ---
    let mut h_pass = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (k, &kernel_weight) in kernel.iter().enumerate() {
                let sx = x as isize + k as isize - radius as isize;
                let sx_clamped = sx.clamp(0, width as isize - 1) as usize;
                acc += field[y * width + sx_clamped] * kernel_weight;
            }
            h_pass[y * width + x] = acc;
        }
    }

    // --- vertical pass ---
    let mut v_pass = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (k, &kernel_weight) in kernel.iter().enumerate() {
                let sy = y as isize + k as isize - radius as isize;
                let sy_clamped = sy.clamp(0, height as isize - 1) as usize;
                acc += h_pass[sy_clamped * width + x] * kernel_weight;
            }
            v_pass[y * width + x] = acc;
        }
    }
    v_pass
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ElasticUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    alpha: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU transform
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated elastic distortion transform.
///
/// # Parameters
///
/// * `alpha`  — displacement magnitude (larger → stronger distortion, typical: 30–100)
/// * `sigma`  — Gaussian smoothing standard deviation (larger → smoother, typical: 4–10)
/// * `seed`   — optional RNG seed; `None` = non-deterministic
#[cfg(feature = "gpu")]
pub struct GpuElasticDistortion {
    alpha: f32,
    sigma: f32,
    seed: Option<u64>,
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuElasticDistortion {
    const SHADER_SRC: &'static str = include_str!("../shaders/elastic_distortion.wgsl");

    /// Create a new elastic distortion pipeline.
    pub fn new(
        alpha: f32,
        sigma: f32,
        seed: Option<u64>,
        context: Arc<GpuContext>,
    ) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(TensorError::invalid_argument(
                "GpuElasticDistortion: sigma must be positive".to_string(),
            ));
        }

        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("elastic_shader"),
            source: ShaderSource::Wgsl(Self::SHADER_SRC.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("elastic_bgl"),
                    entries: &[
                        // Binding 0: combined input (image + displacement fields)
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
                        // Binding 1: output image
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
                        // Binding 2: uniforms
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
                label: Some("elastic_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("elastic_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            alpha,
            sigma,
            seed,
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Generate smoothed displacement fields (CPU) and return `(dx, dy)`.
    ///
    /// Both vectors have `height * width` elements and are already scaled by
    /// `alpha`.
    fn generate_displacement_fields(&self, width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let n = width * height;
        // Use a seeded `Random<StdRng>` so both branches share one concrete type.
        // When no seed is supplied, draw a non-deterministic seed from the
        // thread-local RNG to preserve randomness across invocations.
        let seed = self.seed.unwrap_or_else(scirs2_core::random::random::<u64>);
        let mut rng = scirs2_core::random::seeded_rng(seed);

        // Raw uniform(-1, 1) noise
        let dx_raw: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let dy_raw: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        // Gaussian smooth
        let dx_smooth = gaussian_smooth(&dx_raw, width, height, self.sigma);
        let dy_smooth = gaussian_smooth(&dy_raw, width, height, self.sigma);

        // Scale by alpha
        let dx: Vec<f32> = dx_smooth.iter().map(|v| v * self.alpha).collect();
        let dy: Vec<f32> = dy_smooth.iter().map(|v| v * self.alpha).collect();
        (dx, dy)
    }

    /// Apply elastic distortion to a flat f32 CHW image buffer.
    ///
    /// `input_f32` must have exactly `channels * height * width` elements.
    pub async fn apply(
        &self,
        input_f32: &[f32],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Vec<f32>> {
        let img_n = (channels * height * width) as usize;
        if input_f32.len() != img_n {
            return Err(TensorError::invalid_argument(format!(
                "GpuElasticDistortion: expected {} elements, got {}",
                img_n,
                input_f32.len()
            )));
        }

        let (dx, dy) = self.generate_displacement_fields(width as usize, height as usize);

        // Build combined input buffer: [image | dx | dy]
        let mut combined: Vec<f32> = Vec::with_capacity(img_n + dx.len() + dy.len());
        combined.extend_from_slice(input_f32);
        combined.extend_from_slice(&dx);
        combined.extend_from_slice(&dy);

        let combined_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("elastic_input_combined"),
                    contents: bytemuck::cast_slice(&combined),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

        let output_bytes = (img_n * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("elastic_output"),
            size: output_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // alpha is already baked in; shader multiplies by it again only if alpha != 1.
        // We pass alpha=1.0 here since scaling is done in generate_displacement_fields.
        let uniforms = ElasticUniforms {
            width,
            height,
            channels,
            padding: 0,
            alpha: 1.0,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("elastic_uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("elastic_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: combined_buffer.as_entire_binding(),
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

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("elastic_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("elastic_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wx = (width + 15) / 16;
            let wy = (height + 15) / 16;
            pass.dispatch_workgroups(wx, wy, channels);
        }

        let staging = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("elastic_staging"),
            size: output_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_bytes);
        self.context.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(MapMode::Read, move |v| {
            if tx.send(v).is_err() {
                eprintln!("Warning: failed to send elastic GPU result");
            }
        });
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();

        match rx.await {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range().map_err(|_| {
                    TensorError::device_error_simple(
                        "GpuElasticDistortion: failed to read GPU output buffer".to_string(),
                    )
                })?;
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                Ok(result)
            }
            _ => Err(TensorError::device_error_simple(
                "GpuElasticDistortion: failed to read GPU output buffer".to_string(),
            )),
        }
    }

    /// Apply elastic distortion to a `Tensor<f32>` (CHW layout, rank 3).
    pub async fn distort_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "GpuElasticDistortion: expected rank-3 CHW tensor".to_string(),
            ));
        }
        let dims = input.shape().dims();
        let (channels, height, width) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
        let data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access tensor data".to_string())
        })?;
        let out = self.apply(data, width, height, channels).await?;
        Tensor::from_vec(out, &[channels as usize, height as usize, width as usize])
            .map_err(|e| TensorError::invalid_argument(e.to_string()))
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuElasticDistortion {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (img, lbl) = sample;
        let distorted = pollster::block_on(self.distort_tensor(&img))?;
        Ok((distorted, lbl))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-only stub — created when the `gpu` feature is disabled.
#[cfg(not(feature = "gpu"))]
pub struct GpuElasticDistortion;

#[cfg(not(feature = "gpu"))]
impl GpuElasticDistortion {
    /// Always returns an error without the `gpu` feature.
    pub fn new(_alpha: f32, _sigma: f32, _seed: Option<u64>, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GpuElasticDistortion requires the 'gpu' feature".to_string(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_1d_sums_to_one() {
        let kernel = gaussian_kernel_1d(2.0, 6);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0_f32).abs() < 1e-5, "kernel sum = {}", sum);
    }

    #[test]
    fn test_gaussian_kernel_1d_symmetric() {
        let k = gaussian_kernel_1d(1.5, 4);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(
                (k[i] - k[n - 1 - i]).abs() < 1e-6,
                "kernel not symmetric at {}",
                i
            );
        }
    }

    #[test]
    fn test_gaussian_smooth_constant_field_unchanged() {
        // A constant field convolved with any kernel remains constant.
        let (w, h) = (8usize, 8usize);
        let field = vec![0.5f32; w * h];
        let smoothed = gaussian_smooth(&field, w, h, 2.0);
        for &v in &smoothed {
            assert!((v - 0.5_f32).abs() < 1e-5, "constant field changed: {}", v);
        }
    }

    #[test]
    fn test_gaussian_smooth_reduces_variance() {
        // After smoothing, the variance of a random field should decrease.
        let (w, h) = (32usize, 32usize);
        // Alternating ±1 field — maximum variance
        let field: Vec<f32> = (0..w * h)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let smoothed = gaussian_smooth(&field, w, h, 3.0);
        let mean = smoothed.iter().sum::<f32>() / smoothed.len() as f32;
        let var: f32 =
            smoothed.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / smoothed.len() as f32;
        assert!(var < 1.0, "smoothing did not reduce variance: {}", var);
    }

    #[test]
    fn test_displacement_field_output_sizes() {
        // Build a temporary ElasticDistortion struct just to call the CPU helper.
        // (We can't do it without a real GPU context, so we test the pure CPU path.)
        let (w, h) = (16usize, 16usize);
        let sigma = 4.0f32;
        let alpha = 34.0f32;

        // Manually reproduce what generate_displacement_fields does.
        let n = w * h;
        let raw: Vec<f32> = (0..n).map(|i| (i as f32 % 2.0) * 2.0 - 1.0).collect();
        let smooth = gaussian_smooth(&raw, w, h, sigma);
        let scaled: Vec<f32> = smooth.iter().map(|v| v * alpha).collect();
        assert_eq!(scaled.len(), n);
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_elastic_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let result = GpuElasticDistortion::new(34.0, 4.0, Some(42), ctx);
                assert!(
                    result.is_ok(),
                    "pipeline creation failed: {:?}",
                    result.err()
                );
            }
            Err(_) => println!("GPU not available; skipping elastic creation test"),
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_elastic_zero_alpha_passthrough() {
        // With alpha = 0 the displacement fields are all zeros → identity warp.
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let transform = GpuElasticDistortion::new(0.0, 4.0, Some(0), Arc::clone(&ctx))
                    .expect("pipeline creation");

                let (w, h, c) = (8u32, 8u32, 1u32);
                let input: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
                let output = transform.apply(&input, w, h, c).await.expect("apply");

                for (a, b) in input.iter().zip(output.iter()) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "zero-alpha elastic changed pixel: {} vs {}",
                        a,
                        b
                    );
                }
            }
            Err(_) => println!("GPU not available; skipping elastic zero-alpha test"),
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_elastic_no_gpu_feature() {
        let result = GpuElasticDistortion::new(34.0, 4.0, None, ());
        assert!(result.is_err());
    }
}
