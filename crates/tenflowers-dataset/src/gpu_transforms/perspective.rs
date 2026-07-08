//! GPU-accelerated perspective (homographic) transform
//!
//! Applies a 3×3 homography matrix to image data using inverse-warp with
//! bilinear interpolation.  The transform maps each destination pixel back to
//! its source via the full projective equation:
//!
//! ```text
//! w' = H[2][0]*x + H[2][1]*y + H[2][2]
//! src_x = (H[0][0]*x + H[0][1]*y + H[0][2]) / w'
//! src_y = (H[1][0]*x + H[1][1]*y + H[1][2]) / w'
//! ```
//!
//! Pixels whose source falls outside the image are filled with zero.

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

/// A 3×3 homography matrix stored in row-major order.
pub type HomographyMatrix = [[f32; 3]; 3];

/// Returns the 3×3 identity homography.
pub fn homography_identity() -> HomographyMatrix {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Build a homography that maps a quadrilateral to the unit square.
///
/// `src_pts` is a 4-element array of `(x, y)` corresponding to
/// `[(0,0), (1,0), (1,1), (0,1)]` in the destination.
/// The resulting matrix is the **inverse** warp (dest → src).
pub fn homography_from_quad(src_pts: &[(f32, f32); 4]) -> HomographyMatrix {
    // Direct linear transform for a general 4-point correspondence to unit square.
    // dst points: (0,0),(1,0),(1,1),(0,1)
    // Adapted from OpenCV's getPerspectiveTransform algorithm.

    let [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = *src_pts;

    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let dx3 = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let dy3 = y0 - y1 + y2 - y3;

    let denom = dx1 * dy2 - dx2 * dy1;
    // Guard against degenerate configurations
    if denom.abs() < 1e-10 {
        return homography_identity();
    }
    let g = (dx3 * dy2 - dx2 * dy3) / denom;
    let h = (dx1 * dy3 - dx3 * dy1) / denom;

    let a = x1 - x0 + g * x1;
    let b = x3 - x0 + h * x3;
    let c = x0;
    let d = y1 - y0 + g * y1;
    let e = y3 - y0 + h * y3;
    let f = y0;

    [[a, b, c], [d, e, f], [g, h, 1.0]]
}

/// Compose two homographies: `b(a(p))`.
pub fn homography_compose(a: &HomographyMatrix, b: &HomographyMatrix) -> HomographyMatrix {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 {
                out[r][c] += b[r][k] * a[k][c];
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform (must be 16-byte aligned)
// Layout: 4 u32s + 9 f32s + 3 padding f32s = 4*4 + 12*4 = 16+48 = 64 bytes
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PerspectiveUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
    h00: f32,
    h01: f32,
    h02: f32,
    h10: f32,
    h11: f32,
    h12: f32,
    h20: f32,
    h21: f32,
    h22: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU transform
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated perspective (homographic) warp.
///
/// Supply the **inverse** 3×3 homography matrix (destination → source).
#[cfg(feature = "gpu")]
pub struct GpuPerspectiveTransform {
    matrix: HomographyMatrix,
    context: Arc<GpuContext>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuPerspectiveTransform {
    const SHADER_SRC: &'static str = include_str!("../shaders/perspective.wgsl");

    /// Create a new perspective-transform pipeline.
    ///
    /// `matrix` is the **inverse** 3×3 homography: for each destination pixel
    /// `(x, y)` the source is obtained by projecting `matrix * [x, y, 1]^T`.
    pub fn new(matrix: HomographyMatrix, context: Arc<GpuContext>) -> Result<Self> {
        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("perspective_shader"),
            source: ShaderSource::Wgsl(Self::SHADER_SRC.into()),
        });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("perspective_bgl"),
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
                label: Some("perspective_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("perspective_pipeline"),
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

    /// Apply the perspective warp to a flat f32 CHW image buffer.
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
                "GpuPerspectiveTransform: expected {} elements, got {}",
                expected,
                input_f32.len()
            )));
        }

        let input_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("perspective_input"),
                    contents: bytemuck::cast_slice(input_f32),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

        let output_bytes = (expected * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("perspective_output"),
            size: output_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]] = self.matrix;
        let uniforms = PerspectiveUniforms {
            width,
            height,
            channels,
            padding: 0,
            h00,
            h01,
            h02,
            h10,
            h11,
            h12,
            h20,
            h21,
            h22,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("perspective_uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("perspective_bg"),
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
                label: Some("perspective_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("perspective_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wx = (width + 15) / 16;
            let wy = (height + 15) / 16;
            pass.dispatch_workgroups(wx, wy, channels);
        }

        let staging = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("perspective_staging"),
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
                eprintln!("Warning: failed to send perspective GPU result");
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
                        "GpuPerspectiveTransform: failed to read GPU output buffer".to_string(),
                    )
                })?;
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                Ok(result)
            }
            _ => Err(TensorError::device_error_simple(
                "GpuPerspectiveTransform: failed to read GPU output buffer".to_string(),
            )),
        }
    }

    /// Apply the homographic warp to a `Tensor<f32>` (CHW layout, rank 3).
    pub async fn warp_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "GpuPerspectiveTransform: expected rank-3 CHW tensor".to_string(),
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
impl Transform<f32> for GpuPerspectiveTransform {
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
pub struct GpuPerspectiveTransform;

#[cfg(not(feature = "gpu"))]
impl GpuPerspectiveTransform {
    /// Always returns an error without the `gpu` feature.
    pub fn new(_matrix: HomographyMatrix, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GpuPerspectiveTransform requires the 'gpu' feature".to_string(),
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
    fn test_homography_identity_structure() {
        let id = homography_identity();
        assert_eq!(id[0], [1.0, 0.0, 0.0]);
        assert_eq!(id[1], [0.0, 1.0, 0.0]);
        assert_eq!(id[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_homography_compose_identity() {
        let id = homography_identity();
        let t: HomographyMatrix = [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]];
        let composed = homography_compose(&id, &t);
        assert!((composed[0][2] - 5.0_f32).abs() < 1e-5);
        assert!((composed[1][2] - 3.0_f32).abs() < 1e-5);
        assert!((composed[2][2] - 1.0_f32).abs() < 1e-5);
    }

    #[test]
    fn test_homography_from_quad_unit_square() {
        // Source quad = canonical unit square corners → identity warp
        let src = [(0.0f32, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let h = homography_from_quad(&src);
        // Should be close to identity (up to scale)
        let scale = h[2][2];
        assert!((h[0][0] / scale - 1.0_f32).abs() < 1e-4);
        assert!((h[1][1] / scale - 1.0_f32).abs() < 1e-4);
    }

    #[test]
    fn test_uniform_size_alignment() {
        #[cfg(feature = "gpu")]
        {
            // 4 u32 (16 bytes) + 12 f32 (48 bytes) = 64 bytes
            assert_eq!(std::mem::size_of::<PerspectiveUniforms>(), 64);
        }
        let _ = 0_u8; // no-op when gpu not enabled
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_perspective_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let result = GpuPerspectiveTransform::new(homography_identity(), ctx);
                assert!(result.is_ok());
            }
            Err(_) => println!("GPU not available; skipping perspective creation test"),
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_perspective_identity_passthrough() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let transform =
                    GpuPerspectiveTransform::new(homography_identity(), Arc::clone(&ctx))
                        .expect("pipeline creation");

                let (w, h, c) = (4u32, 4u32, 1u32);
                let input: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
                let output = transform.apply(&input, w, h, c).await.expect("apply");

                for (a, b) in input.iter().zip(output.iter()) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "identity perspective warp changed pixel: {} vs {}",
                        a,
                        b
                    );
                }
            }
            Err(_) => println!("GPU not available; skipping perspective identity test"),
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_perspective_no_gpu_feature() {
        let result = GpuPerspectiveTransform::new(homography_identity(), ());
        assert!(result.is_err());
    }
}
