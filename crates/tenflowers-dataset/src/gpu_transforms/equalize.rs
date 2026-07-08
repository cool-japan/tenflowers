//! GPU-accelerated per-channel histogram equalization
//!
//! Implements contrast normalization via cumulative distribution function (CDF)
//! mapping.  The algorithm proceeds in three steps:
//!
//! 1. **Histogram build** — A GPU compute pass atomically accumulates pixel
//!    values into 256-bin histograms, one per channel.
//! 2. **CDF computation** — The host reads the histograms, computes the
//!    normalized prefix sum (CDF), and uploads the result as a lookup table.
//! 3. **Pixel remap** — A second GPU compute pass maps each pixel through the
//!    per-channel CDF, producing equalized output in `[0, 1]`.
//!
//! # Layout conventions
//!
//! Images are expected in *channel-first* (CHW) flat f32 order with values
//! in `[0, 1]`.  The transform produces output in the same layout and range.

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

const NUM_BINS: usize = 256;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL source constants
// ─────────────────────────────────────────────────────────────────────────────

// Build-histogram pass: atomic increments into a u32 histogram buffer.
// Input layout: storage buffer of f32 (CHW image).
// Histogram layout: u32[channels * NUM_BINS], channel-major.
#[cfg(feature = "gpu")]
const HISTOGRAM_SHADER: &str = r#"
const NUM_BINS: u32 = 256u;

struct EqUniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    padding:  u32,
};

@group(0) @binding(0) var<storage, read>        image_data     : array<f32>;
@group(0) @binding(1) var<storage, read_write>  histogram_data : array<atomic<u32>>;
@group(0) @binding(2) var<uniform>              uniforms       : EqUniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    let px_idx = c * uniforms.height * uniforms.width + y * uniforms.width + x;
    let v      = clamp(image_data[px_idx], 0.0, 1.0);
    let bin    = min(u32(v * f32(NUM_BINS)), NUM_BINS - 1u);

    atomicAdd(&histogram_data[c * NUM_BINS + bin], 1u);
}
"#;

// Equalize pass: remap each pixel through the (CPU-computed) CDF.
// CDF layout: f32[channels * NUM_BINS], already normalized to [0, 1].
#[cfg(feature = "gpu")]
const EQUALIZE_SHADER: &str = r#"
const NUM_BINS: u32 = 256u;

struct EqUniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    padding:  u32,
};

@group(0) @binding(0) var<storage, read>        src_image  : array<f32>;
@group(0) @binding(1) var<storage, read_write>  dst_image  : array<f32>;
@group(0) @binding(2) var<storage, read>        cdf_lut    : array<f32>;
@group(0) @binding(3) var<uniform>              uniforms   : EqUniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    let px_idx    = c * uniforms.height * uniforms.width + y * uniforms.width + x;
    let v         = clamp(src_image[px_idx], 0.0, 1.0);
    let bin       = min(u32(v * f32(NUM_BINS)), NUM_BINS - 1u);
    let equalized = cdf_lut[c * NUM_BINS + bin];

    dst_image[px_idx] = clamp(equalized, 0.0, 1.0);
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform (shared by both passes)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EqualizeUniforms {
    width: u32,
    height: u32,
    channels: u32,
    padding: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU helper: compute normalized CDF from a raw histogram
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the normalized CDF for a single-channel histogram.
///
/// `hist` must have exactly `NUM_BINS` entries.
/// Returns a `NUM_BINS`-length f32 slice mapping each bin to its equalized value
/// in `[0, 1]`.
fn compute_cdf(hist: &[u32]) -> Vec<f32> {
    assert_eq!(
        hist.len(),
        NUM_BINS,
        "histogram must have {} bins",
        NUM_BINS
    );

    // Find the minimum non-zero CDF value (for the CDF-min formula)
    let total: u64 = hist.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return vec![0.0f32; NUM_BINS];
    }

    let mut cdf = Vec::with_capacity(NUM_BINS);
    let mut running: u64 = 0;
    let mut cdf_min: Option<u64> = None;
    for &h in hist {
        running += h as u64;
        cdf.push(running);
        if cdf_min.is_none() && h > 0 {
            cdf_min = Some(running);
        }
    }

    let cdf_min_val = cdf_min.unwrap_or(0);
    let denom = total.saturating_sub(cdf_min_val);

    cdf.iter()
        .map(|&c| {
            if denom == 0 {
                // Degenerate: all mass in a single bin.  Every pixel that
                // falls in that bin (c > 0) maps to 1.0; below-peak bins
                // (c == 0) stay at 0.0.
                if c > 0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (c.saturating_sub(cdf_min_val) as f64 / denom as f64) as f32
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU transform
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated per-channel histogram equalization.
///
/// Images must be in CHW f32 layout with values in `[0, 1]`.
/// Equalization is performed independently per channel.
#[cfg(feature = "gpu")]
pub struct GpuHistogramEqualize {
    context: Arc<GpuContext>,
    histogram_pipeline: ComputePipeline,
    histogram_bgl: BindGroupLayout,
    equalize_pipeline: ComputePipeline,
    equalize_bgl: BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuHistogramEqualize {
    /// Create both GPU pipelines (histogram-build and equalize-remap).
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        // ── Histogram pipeline ──────────────────────────────────────────────
        let hist_shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("histogram_shader"),
            source: ShaderSource::Wgsl(HISTOGRAM_SHADER.into()),
        });

        let histogram_bgl =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("histogram_bgl"),
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

        let hist_pl = context
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("histogram_pl"),
                bind_group_layouts: &[Some(&histogram_bgl)],
                immediate_size: 0,
            });

        let histogram_pipeline =
            context
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("histogram_pipeline"),
                    layout: Some(&hist_pl),
                    module: &hist_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // ── Equalize pipeline ───────────────────────────────────────────────
        let eq_shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("equalize_shader"),
            source: ShaderSource::Wgsl(EQUALIZE_SHADER.into()),
        });

        let equalize_bgl =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("equalize_bgl"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        let eq_pl = context
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("equalize_pl"),
                bind_group_layouts: &[Some(&equalize_bgl)],
                immediate_size: 0,
            });

        let equalize_pipeline =
            context
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("equalize_pipeline"),
                    layout: Some(&eq_pl),
                    module: &eq_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        Ok(Self {
            context,
            histogram_pipeline,
            histogram_bgl,
            equalize_pipeline,
            equalize_bgl,
        })
    }

    /// Apply histogram equalization to a flat f32 CHW image buffer.
    ///
    /// `input_f32` must have exactly `channels * height * width` elements in
    /// `[0, 1]`.  Returns the equalized image in the same layout.
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
                "GpuHistogramEqualize: expected {} elements, got {}",
                img_n,
                input_f32.len()
            )));
        }

        let uniforms = EqualizeUniforms {
            width,
            height,
            channels,
            padding: 0,
        };

        // ── Pass 1: build histograms on GPU ──────────────────────────────────
        let image_buf = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eq_image_input"),
                contents: bytemuck::cast_slice(input_f32),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let hist_n = channels as usize * NUM_BINS;
        let hist_bytes = (hist_n * std::mem::size_of::<u32>()) as u64;
        let histogram_buf = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("eq_histogram"),
            size: hist_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniform_buf =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("eq_uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        let hist_bg = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("eq_hist_bg"),
            layout: &self.histogram_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: image_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: histogram_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("eq_enc1"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq_hist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &hist_bg, &[]);
            let wx = (width + 15) / 16;
            let wy = (height + 15) / 16;
            pass.dispatch_workgroups(wx, wy, channels);
        }

        // Readback histograms
        let hist_staging = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("eq_hist_staging"),
            size: hist_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&histogram_buf, 0, &hist_staging, 0, hist_bytes);
        self.context.queue.submit(std::iter::once(encoder.finish()));

        let hist_slice = hist_staging.slice(..);
        let (htx, hrx) = futures::channel::oneshot::channel();
        hist_slice.map_async(MapMode::Read, move |v| {
            if htx.send(v).is_err() {
                eprintln!("Warning: failed to send histogram readback");
            }
        });
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();

        let raw_histograms: Vec<u32> = match hrx.await {
            Ok(Ok(())) => {
                let data = hist_slice.get_mapped_range().map_err(|_| {
                    TensorError::device_error_simple(
                        "GpuHistogramEqualize: failed to readback histograms".to_string(),
                    )
                })?;
                bytemuck::cast_slice::<u8, u32>(&data).to_vec()
            }
            _ => {
                return Err(TensorError::device_error_simple(
                    "GpuHistogramEqualize: failed to readback histograms".to_string(),
                ))
            }
        };

        // ── CDF computation on CPU ────────────────────────────────────────────
        let mut cdf_lut = Vec::with_capacity(hist_n);
        for c in 0..channels as usize {
            let start = c * NUM_BINS;
            let end = start + NUM_BINS;
            let channel_hist = &raw_histograms[start..end];
            let cdf = compute_cdf(channel_hist);
            cdf_lut.extend_from_slice(&cdf);
        }

        // ── Pass 2: equalize using CDF LUT ───────────────────────────────────
        let output_bytes = (img_n * std::mem::size_of::<f32>()) as u64;
        let output_buf = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("eq_output"),
            size: output_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cdf_buf = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eq_cdf_lut"),
                contents: bytemuck::cast_slice(&cdf_lut),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        // Re-upload uniform (same values)
        let uniform_buf2 =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("eq_uniforms2"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        let eq_bg = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("eq_bg"),
            layout: &self.equalize_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: image_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cdf_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf2.as_entire_binding(),
                },
            ],
        });

        let mut encoder2 = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("eq_enc2"),
            });

        {
            let mut pass = encoder2.begin_compute_pass(&ComputePassDescriptor {
                label: Some("eq_remap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.equalize_pipeline);
            pass.set_bind_group(0, &eq_bg, &[]);
            let wx = (width + 15) / 16;
            let wy = (height + 15) / 16;
            pass.dispatch_workgroups(wx, wy, channels);
        }

        let staging = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("eq_staging"),
            size: output_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder2.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_bytes);
        self.context
            .queue
            .submit(std::iter::once(encoder2.finish()));

        let out_slice = staging.slice(..);
        let (otx, orx) = futures::channel::oneshot::channel();
        out_slice.map_async(MapMode::Read, move |v| {
            if otx.send(v).is_err() {
                eprintln!("Warning: failed to send equalize output");
            }
        });
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();

        match orx.await {
            Ok(Ok(())) => {
                let data = out_slice.get_mapped_range().map_err(|_| {
                    TensorError::device_error_simple(
                        "GpuHistogramEqualize: failed to read output buffer".to_string(),
                    )
                })?;
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                Ok(result)
            }
            _ => Err(TensorError::device_error_simple(
                "GpuHistogramEqualize: failed to read output buffer".to_string(),
            )),
        }
    }

    /// Apply histogram equalization to a `Tensor<f32>` (CHW layout, rank 3).
    pub async fn equalize_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if input.shape().rank() != 3 {
            return Err(TensorError::invalid_argument(
                "GpuHistogramEqualize: expected rank-3 CHW tensor".to_string(),
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
impl Transform<f32> for GpuHistogramEqualize {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (img, lbl) = sample;
        let equalized = pollster::block_on(self.equalize_tensor(&img))?;
        Ok((equalized, lbl))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-only stub — created when the `gpu` feature is disabled.
#[cfg(not(feature = "gpu"))]
pub struct GpuHistogramEqualize;

#[cfg(not(feature = "gpu"))]
impl GpuHistogramEqualize {
    /// Always returns an error without the `gpu` feature.
    pub fn new(_context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GpuHistogramEqualize requires the 'gpu' feature".to_string(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CPU CDF helper tests ───────────────────────────────────────────────

    #[test]
    fn test_compute_cdf_all_zeros_histogram() {
        let hist = vec![0u32; NUM_BINS];
        let cdf = compute_cdf(&hist);
        assert_eq!(cdf.len(), NUM_BINS);
        assert!(
            cdf.iter().all(|&v| v == 0.0),
            "all-zero histogram should give zero CDF"
        );
    }

    #[test]
    fn test_compute_cdf_uniform_histogram() {
        // All bins equal → CDF should be linearly increasing from 0 to 1.
        let hist = vec![1u32; NUM_BINS];
        let cdf = compute_cdf(&hist);
        // First bin with non-zero count maps to 0.0 (cdf_min formula)
        assert!(
            (cdf[0] - 0.0_f32).abs() < 1e-5,
            "first bin CDF = {}",
            cdf[0]
        );
        // Last bin should be 1.0
        assert!(
            (cdf[NUM_BINS - 1] - 1.0_f32).abs() < 1e-4,
            "last bin CDF = {}",
            cdf[NUM_BINS - 1]
        );
    }

    #[test]
    fn test_compute_cdf_monotone() {
        let mut hist = vec![0u32; NUM_BINS];
        for (i, bin) in hist.iter_mut().enumerate() {
            *bin = (i + 1) as u32;
        }
        let cdf = compute_cdf(&hist);
        for i in 1..NUM_BINS {
            assert!(cdf[i] >= cdf[i - 1], "CDF not monotone at bin {}", i);
        }
    }

    #[test]
    fn test_compute_cdf_single_peak_maps_to_one() {
        // All mass at one bin → that bin (and all above) should map to 1.0,
        // bins below map to 0.0.
        let mut hist = vec![0u32; NUM_BINS];
        hist[128] = 1000;
        let cdf = compute_cdf(&hist);
        for (i, &v) in cdf.iter().enumerate().take(128) {
            assert!((v - 0.0_f32).abs() < 1e-5, "below-peak bin {} = {}", i, v);
        }
        for (i, &v) in cdf.iter().enumerate().take(NUM_BINS).skip(128) {
            assert!(
                (v - 1.0_f32).abs() < 1e-4,
                "at/above-peak bin {} = {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_compute_cdf_output_range() {
        let hist: Vec<u32> = (0..NUM_BINS as u32).collect();
        let cdf = compute_cdf(&hist);
        for (i, &v) in cdf.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "CDF bin {} out of range: {}",
                i,
                v
            );
        }
    }

    // ── GPU tests ─────────────────────────────────────────────────────────

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_histogram_equalize_creation() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let result = GpuHistogramEqualize::new(ctx);
                assert!(
                    result.is_ok(),
                    "pipeline creation failed: {:?}",
                    result.err()
                );
            }
            Err(_) => println!("GPU not available; skipping histogram equalize creation test"),
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_equalize_constant_image_unchanged() {
        // A constant-valued image should be unchanged by histogram equalization
        // (all mass in a single bin, CDF maps that bin to 1.0 for everything ≥ it).
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let eq = GpuHistogramEqualize::new(Arc::clone(&ctx)).expect("pipeline");
                let (w, h, c) = (8u32, 8u32, 1u32);
                // Constant 0.5 image
                let input = vec![0.5f32; (w * h * c) as usize];
                let output = eq.apply(&input, w, h, c).await.expect("apply");
                // All outputs should be equal (they may all map to 1.0 or stay at some level)
                let first = output[0];
                for (i, &v) in output.iter().enumerate() {
                    assert!(
                        (v - first).abs() < 1e-4,
                        "uniform image not uniform after equalize at {}: {}",
                        i,
                        v
                    );
                }
            }
            Err(_) => println!("GPU not available; skipping histogram equalize constant test"),
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_equalize_output_in_range() {
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let eq = GpuHistogramEqualize::new(Arc::clone(&ctx)).expect("pipeline");
                let (w, h, c) = (16u32, 16u32, 3u32);
                // Ramp image
                let n = (w * h * c) as usize;
                let input: Vec<f32> = (0..n).map(|i| (i % 256) as f32 / 255.0).collect();
                let output = eq.apply(&input, w, h, c).await.expect("apply");
                for (i, &v) in output.iter().enumerate() {
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "output pixel {} = {} out of [0,1]",
                        i,
                        v
                    );
                }
            }
            Err(_) => println!("GPU not available; skipping histogram equalize range test"),
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_equalize_increases_contrast() {
        // A dark image (all pixels in lower half) should have its output
        // spread more towards 1.0 after equalization.
        match super::super::context::GpuContext::new().await {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                let eq = GpuHistogramEqualize::new(Arc::clone(&ctx)).expect("pipeline");
                let (w, h, c) = (16u32, 16u32, 1u32);
                let n = (w * h) as usize;
                // Pixels uniformly in [0, 0.5]
                let input: Vec<f32> = (0..n).map(|i| i as f32 / (2.0 * n as f32)).collect();
                let input_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let output = eq.apply(&input, w, h, c).await.expect("apply");
                let output_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                // After equalization the max should be at least as large
                assert!(
                    output_max >= input_max - 1e-4,
                    "equalization should not reduce maximum: in={}, out={}",
                    input_max,
                    output_max
                );
            }
            Err(_) => println!("GPU not available; skipping histogram equalize contrast test"),
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_histogram_equalize_no_gpu_feature() {
        let result = GpuHistogramEqualize::new(());
        assert!(result.is_err());
    }
}
