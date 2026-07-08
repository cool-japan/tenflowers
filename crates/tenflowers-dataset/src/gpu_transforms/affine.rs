//! GPU-accelerated 2-D affine transform
//!
//! Applies an arbitrary 2×3 affine warp matrix to image data.  The matrix
//! encodes any combination of translation, scale, shear, and rotation.
//!
//! The transform is defined in *destination-to-source* (inverse-warp) form:
//! for each output pixel `(x, y)` the source coordinates are
//!
//! ```text
//! src_x = m[0][0]*x + m[0][1]*y + m[0][2]
//! src_y = m[1][0]*x + m[1][1]*y + m[1][2]
//! ```
//!
//! Out-of-bounds source positions are filled with zero.

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
use super::context::GpuContext;

/// A 2×3 affine transform matrix stored in row-major order.
///
/// Layout: `[[m00, m01, m02], [m10, m11, m12]]`
pub type AffineMatrix = [[f32; 3]; 2];

/// Returns the identity 2×3 affine matrix.
pub fn affine_identity() -> AffineMatrix {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
}

/// Build a pure-translation matrix.
pub fn affine_translation(tx: f32, ty: f32) -> AffineMatrix {
    [[1.0, 0.0, tx], [0.0, 1.0, ty]]
}

/// Build a uniform-scale matrix (scale from origin).
pub fn affine_scale(sx: f32, sy: f32) -> AffineMatrix {
    [[sx, 0.0, 0.0], [0.0, sy, 0.0]]
}

/// Build a counter-clockwise rotation matrix (radians) about the origin.
pub fn affine_rotation(angle_rad: f32) -> AffineMatrix {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    [[c, -s, 0.0], [s, c, 0.0]]
}

/// Build a shear matrix.
pub fn affine_shear(shear_x: f32, shear_y: f32) -> AffineMatrix {
    [[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0]]
}

/// Compose two 2×3 affine matrices: `a_then_b(p) = b(a(p))`.
pub fn affine_compose(a: &AffineMatrix, b: &AffineMatrix) -> AffineMatrix {
    // Treat each 2×3 as a 3×3 with last row [0,0,1], then multiply.
    let mul = |r: usize, c: usize| -> f32 {
        let a_row = b[r];
        // b[r][0]*a_col(c,0) + b[r][1]*a_col(c,1) + b[r][2]*a_col(c,2 or 1)
        if c < 2 {
            // standard column
            a_row[0] * a[0][c] + a_row[1] * a[1][c] + if c == 2 { a_row[2] } else { 0.0 }
        } else {
            // translation column
            a_row[0] * a[0][2] + a_row[1] * a[1][2] + a_row[2]
        }
    };
    [
        [mul(0, 0), mul(0, 1), mul(0, 2)],
        [mul(1, 0), mul(1, 1), mul(1, 2)],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AffineUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    pad0: f32,
    pad1: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU transform struct
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated 2-D affine warp transform.
///
/// Provide the inverse transform (output → source mapping).  Pixels whose
/// back-projected source falls outside the image boundary are set to zero.
#[cfg(feature = "gpu")]
pub struct GpuAffineTransform {
    /// Inverse 2×3 affine matrix (destination → source).
    matrix: AffineMatrix,
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuAffineTransform {
    const SHADER_SRC: &'static str = include_str!("../shaders/affine.wgsl");

    /// Create a new affine-transform pipeline.
    ///
    /// `matrix` is the **inverse** 2×3 matrix: for each destination pixel
    /// `(x, y)` the source is `matrix * [x, y, 1]^T`.
    pub fn new(matrix: AffineMatrix, context: Arc<GpuContext>) -> Result<Self> {
        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("affine_shader"),
            source: ShaderSource::Wgsl(Self::SHADER_SRC.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("affine_bgl"),
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
                label: Some("affine_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("affine_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            matrix,
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Apply the affine warp to a flat f32 CHW image buffer.
    ///
    /// `input_f32` must have exactly `channels * height * width` elements.
    /// Returns the warped image in the same CHW layout.
    pub async fn apply(
        &self,
        input_f32: &[f32],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Vec<f32>> {
        let expected = (channels * height * width) as usize;
        if input_f32.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "AffineTransform: expected {} elements, got {}",
                expected,
                input_f32.len()
            )));
        }

        let input_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("affine_input"),
                    contents: bytemuck::cast_slice(input_f32),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

        let output_bytes = (expected * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("affine_output"),
            size: output_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let [[m00, m01, m02], [m10, m11, m12]] = self.matrix;
        let uniforms = AffineUniforms {
            width,
            height,
            channels,
            padding: 0,
            m00,
            m01,
            m02,
            m10,
            m11,
            m12,
            pad0: 0.0,
            pad1: 0.0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("affine_uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("affine_bg"),
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

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("affine_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("affine_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wx = (width + 15) / 16;
            let wy = (height + 15) / 16;
            pass.dispatch_workgroups(wx, wy, channels);
        }

        let staging = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("affine_staging"),
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
                eprintln!("Warning: failed to send affine GPU result");
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
                        "GpuAffineTransform: failed to read GPU output buffer".to_string(),
                    )
                })?;
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                Ok(result)
            }
            _ => Err(TensorError::device_error_simple(
                "GpuAffineTransform: failed to read GPU output buffer".to_string(),
            )),
        }
    }

    /// Apply the affine warp to a `Tensor<f32>` (CHW layout, rank 3).
    pub async fn warp_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "GpuAffineTransform: expected rank-3 tensor in CHW layout".to_string(),
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
impl Transform<f32> for GpuAffineTransform {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (img, lbl) = sample;
        let warped = pollster::block_on(self.warp_tensor(&img))?;
        Ok((warped, lbl))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-only stub — created when the `gpu` feature is disabled.
#[cfg(not(feature = "gpu"))]
pub struct GpuAffineTransform;

#[cfg(not(feature = "gpu"))]
impl GpuAffineTransform {
    /// Always returns an error without the `gpu` feature.
    pub fn new(_matrix: AffineMatrix, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GpuAffineTransform requires the 'gpu' feature".to_string(),
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
    fn test_affine_identity_matrix() {
        let id = affine_identity();
        assert_eq!(id, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    #[test]
    fn test_affine_translation_matrix() {
        let t = affine_translation(3.0, -5.0);
        assert_eq!(t[0][2], 3.0);
        assert_eq!(t[1][2], -5.0);
    }

    #[test]
    fn test_affine_scale_matrix() {
        let s = affine_scale(2.0, 3.0);
        assert_eq!(s[0][0], 2.0);
        assert_eq!(s[1][1], 3.0);
        assert_eq!(s[0][1], 0.0);
    }

    #[test]
    fn test_affine_rotation_identity() {
        let r = affine_rotation(0.0);
        assert!((r[0][0] - 1.0_f32).abs() < 1e-6);
        assert!((r[0][1]).abs() < 1e-6);
        assert!((r[1][0]).abs() < 1e-6);
        assert!((r[1][1] - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_affine_rotation_90_degrees() {
        let r = affine_rotation(std::f32::consts::FRAC_PI_2);
        // cos(90°) ≈ 0, sin(90°) ≈ 1
        assert!((r[0][0]).abs() < 1e-5);
        assert!((r[0][1] + 1.0).abs() < 1e-5);
        assert!((r[1][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_affine_shear_matrix() {
        let sh = affine_shear(0.5, 0.0);
        assert_eq!(sh[0][1], 0.5);
        assert_eq!(sh[1][0], 0.0);
    }

    #[test]
    fn test_affine_compose_identity() {
        let id = affine_identity();
        let t = affine_translation(10.0, 20.0);
        let ct = affine_compose(&id, &t);
        // compose identity then t should equal t
        assert!((ct[0][2] - 10.0_f32).abs() < 1e-6);
        assert!((ct[1][2] - 20.0_f32).abs() < 1e-6);
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_affine_transform_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let mat = affine_identity();
                let result = GpuAffineTransform::new(mat, ctx);
                assert!(
                    result.is_ok(),
                    "Pipeline creation failed: {:?}",
                    result.err()
                );
            }
            Err(_) => println!("GPU not available; skipping affine GPU creation test"),
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_affine_identity_passthrough() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let transform = GpuAffineTransform::new(affine_identity(), Arc::clone(&ctx))
                    .expect("pipeline creation");

                // 1-channel 4×4 image filled with a ramp
                let (w, h, c) = (4u32, 4u32, 1u32);
                let input: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
                let output = transform.apply(&input, w, h, c).await.expect("apply");

                for (a, b) in input.iter().zip(output.iter()) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "identity warp changed pixel: {} vs {}",
                        a,
                        b
                    );
                }
            }
            Err(_) => println!("GPU not available; skipping affine identity test"),
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_affine_no_gpu_feature() {
        let result = GpuAffineTransform::new(affine_identity(), ());
        assert!(result.is_err());
    }
}
