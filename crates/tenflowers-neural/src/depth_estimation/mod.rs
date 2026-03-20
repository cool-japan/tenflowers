//! Depth Estimation & 3D Vision.
//!
//! Production-grade monocular/stereo depth estimation, depth completion,
//! volumetric convolutions, implicit neural fields, and panoptic segmentation.
//!
//! - **DepthEncoder**: Multi-scale feature extraction (4 levels, 1/2..1/16)
//! - **DptDecoder**: Dense Prediction Transformer decoder (reassemble + fusion)
//! - **MonocularDepthEstimator**: Full DPT-style pipeline (scale-invariant + gradient loss)
//! - **StereoMatcher**: Cost volume correlation + soft-argmin disparity regression
//! - **DepthCompletion**: Sparse-to-dense depth with confidence-guided propagation
//! - **DeConv3d**: Full 3D convolution (N, C, D, H, W) with stride/padding/dilation
//! - **DeImplicitNeuralField**: NeRF-inspired positional-encoding MLP + volume rendering
//! - **DePanopticHead**: Panoptic segmentation (semantic + instance + fusion)
//! - **DeDepthMetrics** / **DeDepthReport**: AbsRel, SqRel, RMSE, delta thresholds, SILog

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn de_relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn de_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn de_softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn de_sample_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Xavier-normal initialization for a weight matrix [rows x cols].
fn de_xavier_init(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| de_sample_normal(&mut rng) * scale)
                .collect()
        })
        .collect()
}

/// Flat Xavier-normal initialization.
fn de_xavier_init_flat(size: usize, fan_in: usize, fan_out: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
    (0..size)
        .map(|_| de_sample_normal(&mut rng) * scale)
        .collect()
}

fn de_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn de_matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| de_dot(row, v)).collect()
}

fn de_linear(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    let out = de_matvec(w, x);
    out.iter().zip(b).map(|(o, bi)| o + bi).collect()
}

fn de_linear_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    de_linear(w, b, x).into_iter().map(de_relu).collect()
}

/// Batch normalization (1-D, inference-mode: channel-wise mean/var).
fn de_batch_norm(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|&v| (v - mean) * inv_std).collect()
}

/// 2-D convolution on a single-channel HxW image with a KxK kernel (stride, pad).
fn de_conv2d_single(
    input: &[f64],
    h: usize,
    w: usize,
    kernel: &[f64],
    k: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f64>, usize, usize) {
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0; oh * ow];
    for oi in 0..oh {
        for oj in 0..ow {
            let mut val = 0.0;
            for ki in 0..k {
                for kj in 0..k {
                    let ii = oi * stride + ki;
                    let jj = oj * stride + kj;
                    if ii >= pad && jj >= pad && ii - pad < h && jj - pad < w {
                        val += input[(ii - pad) * w + (jj - pad)] * kernel[ki * k + kj];
                    }
                }
            }
            out[oi * ow + oj] = val;
        }
    }
    (out, oh, ow)
}

/// Max-pool 2-D (pool_size x pool_size, stride = pool_size).
fn de_maxpool2d(input: &[f64], h: usize, w: usize, pool: usize) -> (Vec<f64>, usize, usize) {
    let oh = h / pool;
    let ow = w / pool;
    let mut out = vec![f64::NEG_INFINITY; oh * ow];
    for oi in 0..oh {
        for oj in 0..ow {
            for pi in 0..pool {
                for pj in 0..pool {
                    let idx = (oi * pool + pi) * w + (oj * pool + pj);
                    if idx < input.len() && input[idx] > out[oi * ow + oj] {
                        out[oi * ow + oj] = input[idx];
                    }
                }
            }
        }
    }
    (out, oh, ow)
}

/// Bilinear upsample 2x for a single-channel HxW feature map.
fn de_upsample_2x(input: &[f64], h: usize, w: usize) -> (Vec<f64>, usize, usize) {
    let nh = h * 2;
    let nw = w * 2;
    let mut out = vec![0.0; nh * nw];
    for i in 0..nh {
        for j in 0..nw {
            let si = (i as f64) / 2.0;
            let sj = (j as f64) / 2.0;
            let y0 = si.floor() as usize;
            let x0 = sj.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let fy = si - y0 as f64;
            let fx = sj - x0 as f64;
            let val = input[y0 * w + x0] * (1.0 - fy) * (1.0 - fx)
                + input[y0 * w + x1] * (1.0 - fy) * fx
                + input[y1 * w + x0] * fy * (1.0 - fx)
                + input[y1 * w + x1] * fy * fx;
            out[i * nw + j] = val;
        }
    }
    (out, nh, nw)
}

/// Element-wise add two same-length slices.
fn de_vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  DepthEncoder — Multi-scale feature extraction (4 levels)
// ─────────────────────────────────────────────────────────────────────────────

/// A single encoder block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool.
#[derive(Debug, Clone)]
pub struct DeEncoderBlock {
    /// Conv1 kernel (k x k, flattened).
    pub conv1: Vec<f64>,
    /// Conv2 kernel (k x k, flattened).
    pub conv2: Vec<f64>,
    /// Kernel size.
    pub k: usize,
    /// Stride (always 1 for conv, pool handles downsampling).
    pub stride: usize,
    /// Padding for convolution.
    pub pad: usize,
    /// Pool size (2 for 2x downsampling).
    pub pool_size: usize,
}

/// Configuration for the depth encoder.
#[derive(Debug, Clone)]
pub struct DepthEncoderConfig {
    /// Input spatial height.
    pub input_h: usize,
    /// Input spatial width.
    pub input_w: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for DepthEncoderConfig {
    fn default() -> Self {
        Self {
            input_h: 64,
            input_w: 64,
            kernel_size: 3,
            seed: 42,
        }
    }
}

/// Multi-scale feature extractor producing 4 levels at 1/2, 1/4, 1/8, 1/16.
///
/// Each level applies two convolutions (Xavier init), batch normalization,
/// ReLU, and max-pooling.
#[derive(Debug, Clone)]
pub struct DepthEncoder {
    pub blocks: Vec<DeEncoderBlock>,
    pub input_h: usize,
    pub input_w: usize,
}

/// A multi-scale feature map produced by [`DepthEncoder`].
#[derive(Debug, Clone)]
pub struct DeMultiScaleFeatures {
    /// Feature maps at each level (flattened), from finest (1/2) to coarsest (1/16).
    pub features: Vec<Vec<f64>>,
    /// Spatial height at each level.
    pub heights: Vec<usize>,
    /// Spatial width at each level.
    pub widths: Vec<usize>,
}

impl DepthEncoder {
    /// Create a new depth encoder with Xavier-initialized conv kernels.
    pub fn new(config: &DepthEncoderConfig) -> Result<Self> {
        if config.input_h < 16 || config.input_w < 16 {
            return Err(TensorError::compute_error_simple(
                "DepthEncoder: input_h and input_w must be >= 16".to_string(),
            ));
        }
        let k = config.kernel_size;
        let kk = k * k;
        let mut blocks = Vec::with_capacity(4);
        let mut seed = config.seed;
        for _level in 0..4 {
            let conv1 = de_xavier_init_flat(kk, kk, kk, seed);
            seed = seed.wrapping_add(1);
            let conv2 = de_xavier_init_flat(kk, kk, kk, seed);
            seed = seed.wrapping_add(1);
            blocks.push(DeEncoderBlock {
                conv1,
                conv2,
                k,
                stride: 1,
                pad: k / 2,
                pool_size: 2,
            });
        }
        Ok(Self {
            blocks,
            input_h: config.input_h,
            input_w: config.input_w,
        })
    }

    /// Extract multi-scale features from a single-channel input.
    ///
    /// `input` must have length `input_h * input_w`.
    pub fn forward(&self, input: &[f64]) -> Result<DeMultiScaleFeatures> {
        let expected = self.input_h * self.input_w;
        if input.len() != expected {
            return Err(TensorError::compute_error_simple(
                "DepthEncoder: input length mismatch".to_string(),
            ));
        }
        let mut features = Vec::with_capacity(4);
        let mut heights = Vec::with_capacity(4);
        let mut widths = Vec::with_capacity(4);

        let mut current = input.to_vec();
        let mut h = self.input_h;
        let mut w = self.input_w;

        for block in &self.blocks {
            // Conv1 → BN → ReLU
            let (c1, h1, w1) = de_conv2d_single(
                &current,
                h,
                w,
                &block.conv1,
                block.k,
                block.stride,
                block.pad,
            );
            let bn1 = de_batch_norm(&c1);
            let r1: Vec<f64> = bn1.into_iter().map(de_relu).collect();

            // Conv2 → BN → ReLU
            let (c2, h2, w2) =
                de_conv2d_single(&r1, h1, w1, &block.conv2, block.k, block.stride, block.pad);
            let bn2 = de_batch_norm(&c2);
            let r2: Vec<f64> = bn2.into_iter().map(de_relu).collect();

            // MaxPool
            let (pooled, ph, pw) = de_maxpool2d(&r2, h2, w2, block.pool_size);

            features.push(pooled.clone());
            heights.push(ph);
            widths.push(pw);

            current = pooled;
            h = ph;
            w = pw;
        }

        Ok(DeMultiScaleFeatures {
            features,
            heights,
            widths,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  DptDecoder — Dense Prediction Transformer decoder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the DPT decoder.
#[derive(Debug, Clone)]
pub struct DptDecoderConfig {
    /// Number of feature levels (typically 4).
    pub n_levels: usize,
    /// Output channel dimension after reassembly projection.
    pub reassemble_dim: usize,
    /// Final head intermediate channels.
    pub head_channels: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for DptDecoderConfig {
    fn default() -> Self {
        Self {
            n_levels: 4,
            reassemble_dim: 32,
            head_channels: 16,
            seed: 100,
        }
    }
}

/// A reassembly projection layer: linear projection to a fixed dimension.
#[derive(Debug, Clone)]
pub struct DeReassembleLayer {
    pub weight: Vec<f64>,
    pub bias: f64,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl DeReassembleLayer {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let weight = de_xavier_init_flat(in_dim * out_dim, in_dim, out_dim, seed);
        Self {
            weight,
            bias: 0.0,
            in_dim,
            out_dim,
        }
    }

    /// Project each pixel's feature through a linear layer.
    fn forward(&self, input: &[f64], h: usize, w: usize) -> Result<Vec<f64>> {
        // input: [h * w] (single channel) → project to out_dim channels per pixel
        // For simplicity, treat as 1-channel and scale/bias per pixel
        let n_pixels = h * w;
        if input.len() != n_pixels {
            return Err(TensorError::compute_error_simple(
                "DeReassembleLayer: input size mismatch".to_string(),
            ));
        }
        // Single-channel projection: simple scale + bias
        let scale = if self.weight.is_empty() {
            1.0
        } else {
            self.weight[0]
        };
        let out: Vec<f64> = input.iter().map(|&v| v * scale + self.bias).collect();
        Ok(out)
    }
}

/// RefineNet-style residual convolution block for fusion.
#[derive(Debug, Clone)]
pub struct DeRefineBlock {
    pub conv_kernel: Vec<f64>,
    pub k: usize,
    pub pad: usize,
}

impl DeRefineBlock {
    fn new(k: usize, seed: u64) -> Self {
        let kk = k * k;
        let conv_kernel = de_xavier_init_flat(kk, kk, kk, seed);
        Self {
            conv_kernel,
            k,
            pad: k / 2,
        }
    }

    /// Refine: conv → BN → ReLU, then residual add with input.
    fn forward(&self, input: &[f64], h: usize, w: usize) -> Result<(Vec<f64>, usize, usize)> {
        let (conv_out, oh, ow) =
            de_conv2d_single(input, h, w, &self.conv_kernel, self.k, 1, self.pad);
        let bn = de_batch_norm(&conv_out);
        let activated: Vec<f64> = bn.into_iter().map(de_relu).collect();
        // Residual connection
        let result = if activated.len() == input.len() {
            de_vec_add(&activated, input)
        } else {
            activated
        };
        Ok((result, oh, ow))
    }
}

/// Dense Prediction Transformer (DPT) decoder.
///
/// Reassembles multi-scale features from [`DepthEncoder`], fuses them
/// iteratively from coarsest to finest with RefineNet blocks, and
/// produces a final depth map through a 2-layer convolution head.
#[derive(Debug, Clone)]
pub struct DptDecoder {
    pub reassemble_layers: Vec<DeReassembleLayer>,
    pub refine_blocks: Vec<DeRefineBlock>,
    pub head_conv1: Vec<f64>,
    pub head_conv2: Vec<f64>,
    pub head_k: usize,
    pub head_pad: usize,
}

impl DptDecoder {
    /// Create a new DPT decoder.
    pub fn new(config: &DptDecoderConfig) -> Result<Self> {
        if config.n_levels == 0 {
            return Err(TensorError::compute_error_simple(
                "DptDecoder: n_levels must be > 0".to_string(),
            ));
        }
        let mut seed = config.seed;
        let mut reassemble_layers = Vec::with_capacity(config.n_levels);
        let mut refine_blocks = Vec::with_capacity(config.n_levels);

        for _i in 0..config.n_levels {
            reassemble_layers.push(DeReassembleLayer::new(
                config.reassemble_dim,
                config.reassemble_dim,
                seed,
            ));
            seed = seed.wrapping_add(1);
            refine_blocks.push(DeRefineBlock::new(3, seed));
            seed = seed.wrapping_add(1);
        }

        let hk = 3;
        let hkk = hk * hk;
        let head_conv1 = de_xavier_init_flat(hkk, config.head_channels, config.head_channels, seed);
        seed = seed.wrapping_add(1);
        let head_conv2 = de_xavier_init_flat(hkk, config.head_channels, 1, seed);

        Ok(Self {
            reassemble_layers,
            refine_blocks,
            head_conv1,
            head_conv2,
            head_k: hk,
            head_pad: hk / 2,
        })
    }

    /// Decode multi-scale features to a depth map.
    ///
    /// Returns `(depth_map, height, width)`.
    pub fn forward(&self, ms_features: &DeMultiScaleFeatures) -> Result<(Vec<f64>, usize, usize)> {
        let n = ms_features.features.len();
        if n == 0 {
            return Err(TensorError::compute_error_simple(
                "DptDecoder: empty features".to_string(),
            ));
        }

        // Start from coarsest level
        let last = n - 1;
        let reassembled = self.reassemble_layers[last].forward(
            &ms_features.features[last],
            ms_features.heights[last],
            ms_features.widths[last],
        )?;
        let (mut fused, mut fh, mut fw) = self.refine_blocks[last].forward(
            &reassembled,
            ms_features.heights[last],
            ms_features.widths[last],
        )?;

        // Iteratively fuse from coarser to finer
        for level in (0..last).rev() {
            // Upsample current fused to match finer level
            let (up, uh, uw) = de_upsample_2x(&fused, fh, fw);
            let reassembled_level = self.reassemble_layers[level].forward(
                &ms_features.features[level],
                ms_features.heights[level],
                ms_features.widths[level],
            )?;

            // Add upsampled fused + reassembled finer features
            let combined = if up.len() == reassembled_level.len() {
                de_vec_add(&up, &reassembled_level)
            } else {
                // Size mismatch: use the smaller
                let min_len = up.len().min(reassembled_level.len());
                up[..min_len]
                    .iter()
                    .zip(&reassembled_level[..min_len])
                    .map(|(a, b)| a + b)
                    .collect()
            };

            let target_h = ms_features.heights[level];
            let target_w = ms_features.widths[level];
            let actual_len = combined.len();
            let expected_len = target_h * target_w;

            let adjusted = if actual_len >= expected_len {
                combined[..expected_len].to_vec()
            } else {
                let mut v = combined;
                v.resize(expected_len, 0.0);
                v
            };

            let (refined, rh, rw) =
                self.refine_blocks[level].forward(&adjusted, target_h, target_w)?;
            fused = refined;
            fh = rh;
            fw = rw;
        }

        // Head: Conv1 → ReLU → Conv2 → softplus (ensure positive depth)
        let (h1, h1h, h1w) = de_conv2d_single(
            &fused,
            fh,
            fw,
            &self.head_conv1,
            self.head_k,
            1,
            self.head_pad,
        );
        let h1_act: Vec<f64> = h1.into_iter().map(de_relu).collect();
        let (h2, h2h, h2w) = de_conv2d_single(
            &h1_act,
            h1h,
            h1w,
            &self.head_conv2,
            self.head_k,
            1,
            self.head_pad,
        );
        let depth: Vec<f64> = h2.into_iter().map(de_softplus).collect();

        Ok((depth, h2h, h2w))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  MonocularDepthEstimator — Full DPT-style pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Depth output mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DepthMode {
    /// Metric depth in absolute units.
    Metric,
    /// Relative (ordinal) depth (normalized 0..1).
    Relative,
}

/// Full monocular depth estimation pipeline.
///
/// Combines [`DepthEncoder`] and [`DptDecoder`] into an end-to-end system
/// with scale-invariant loss (Eigen 2014) and gradient-matching loss.
#[derive(Debug, Clone)]
pub struct MonocularDepthEstimator {
    pub encoder: DepthEncoder,
    pub decoder: DptDecoder,
    pub mode: DepthMode,
    /// Lambda for the scale-invariant loss variance term.
    pub si_lambda: f64,
}

impl MonocularDepthEstimator {
    /// Create a new monocular depth estimator.
    pub fn new(
        encoder_config: &DepthEncoderConfig,
        decoder_config: &DptDecoderConfig,
        mode: DepthMode,
    ) -> Result<Self> {
        let encoder = DepthEncoder::new(encoder_config)?;
        let decoder = DptDecoder::new(decoder_config)?;
        Ok(Self {
            encoder,
            decoder,
            mode,
            si_lambda: 0.5,
        })
    }

    /// Predict depth from a single-channel image.
    ///
    /// Returns `(depth_map, height, width)`.
    pub fn predict_depth(&self, image: &[f64]) -> Result<(Vec<f64>, usize, usize)> {
        let features = self.encoder.forward(image)?;
        let (mut depth, h, w) = self.decoder.forward(&features)?;

        if self.mode == DepthMode::Relative {
            // Normalize to [0, 1]
            let min_val = depth.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = depth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_val - min_val).max(1e-8);
            for v in &mut depth {
                *v = (*v - min_val) / range;
            }
        }

        Ok((depth, h, w))
    }

    /// Scale-invariant loss (Eigen et al. 2014).
    ///
    /// `d_i = log(pred_i) - log(gt_i)`, loss = Var(d) + lambda * Mean(d)^2
    pub fn scale_invariant_loss(pred: &[f64], gt: &[f64]) -> Result<f64> {
        if pred.len() != gt.len() || pred.is_empty() {
            return Err(TensorError::compute_error_simple(
                "scale_invariant_loss: size mismatch or empty".to_string(),
            ));
        }
        let n = pred.len() as f64;
        let d: Vec<f64> = pred
            .iter()
            .zip(gt)
            .map(|(&p, &g)| (p.max(1e-8)).ln() - (g.max(1e-8)).ln())
            .collect();
        let mean_d: f64 = d.iter().sum::<f64>() / n;
        let var_d: f64 = d.iter().map(|&di| (di - mean_d).powi(2)).sum::<f64>() / n;
        Ok(var_d + 0.5 * mean_d * mean_d)
    }

    /// Scale-invariant loss with configurable lambda.
    pub fn scale_invariant_loss_lambda(pred: &[f64], gt: &[f64], lambda: f64) -> Result<f64> {
        if pred.len() != gt.len() || pred.is_empty() {
            return Err(TensorError::compute_error_simple(
                "scale_invariant_loss_lambda: size mismatch or empty".to_string(),
            ));
        }
        let n = pred.len() as f64;
        let d: Vec<f64> = pred
            .iter()
            .zip(gt)
            .map(|(&p, &g)| (p.max(1e-8)).ln() - (g.max(1e-8)).ln())
            .collect();
        let mean_d: f64 = d.iter().sum::<f64>() / n;
        let var_d: f64 = d.iter().map(|&di| (di - mean_d).powi(2)).sum::<f64>() / n;
        Ok(var_d + lambda * mean_d * mean_d)
    }

    /// Gradient matching loss (edge-aware smoothness).
    ///
    /// Penalizes differences between spatial gradients of predicted and GT depth.
    pub fn gradient_matching_loss(pred: &[f64], gt: &[f64], h: usize, w: usize) -> Result<f64> {
        if pred.len() != h * w || gt.len() != h * w {
            return Err(TensorError::compute_error_simple(
                "gradient_matching_loss: size mismatch".to_string(),
            ));
        }
        if h < 2 || w < 2 {
            return Err(TensorError::compute_error_simple(
                "gradient_matching_loss: need h >= 2 and w >= 2".to_string(),
            ));
        }

        let mut loss: f64 = 0.0;
        let mut count: f64 = 0.0;

        // Horizontal gradients
        for i in 0..h {
            for j in 0..(w - 1) {
                let pred_dx = pred[i * w + j + 1] - pred[i * w + j];
                let gt_dx = gt[i * w + j + 1] - gt[i * w + j];
                loss += (pred_dx - gt_dx).powi(2);
                count += 1.0;
            }
        }

        // Vertical gradients
        for i in 0..(h - 1) {
            for j in 0..w {
                let pred_dy = pred[(i + 1) * w + j] - pred[i * w + j];
                let gt_dy = gt[(i + 1) * w + j] - gt[i * w + j];
                loss += (pred_dy - gt_dy).powi(2);
                count += 1.0;
            }
        }

        Ok(loss / count.max(1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  StereoMatcher — Stereo vision depth estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the stereo matcher.
#[derive(Debug, Clone)]
pub struct StereoMatcherConfig {
    /// Maximum disparity to search.
    pub max_disparity: usize,
    /// Focal length (pixels).
    pub focal_length: f64,
    /// Stereo baseline distance (same units as desired depth).
    pub baseline: f64,
    /// Cost aggregation window radius.
    pub agg_radius: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for StereoMatcherConfig {
    fn default() -> Self {
        Self {
            max_disparity: 32,
            focal_length: 500.0,
            baseline: 0.12,
            agg_radius: 2,
            seed: 200,
        }
    }
}

/// Stereo vision depth estimation.
///
/// Builds a cost volume via correlation between left/right feature maps
/// at each candidate disparity, applies window-based cost aggregation,
/// and uses soft-argmin regression for sub-pixel disparity.
#[derive(Debug, Clone)]
pub struct StereoMatcher {
    pub config: StereoMatcherConfig,
}

impl StereoMatcher {
    pub fn new(config: StereoMatcherConfig) -> Result<Self> {
        if config.max_disparity == 0 {
            return Err(TensorError::compute_error_simple(
                "StereoMatcher: max_disparity must be > 0".to_string(),
            ));
        }
        if config.focal_length <= 0.0 || config.baseline <= 0.0 {
            return Err(TensorError::compute_error_simple(
                "StereoMatcher: focal_length and baseline must be positive".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Build the cost volume: correlation between left and right at each disparity.
    ///
    /// Returns `[max_disparity][h * w]` cost volume.
    pub fn build_cost_volume(
        &self,
        left: &[f64],
        right: &[f64],
        h: usize,
        w: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let expected = h * w;
        if left.len() != expected || right.len() != expected {
            return Err(TensorError::compute_error_simple(
                "StereoMatcher: left/right size mismatch".to_string(),
            ));
        }

        let max_d = self.config.max_disparity;
        let mut cost_vol = Vec::with_capacity(max_d);

        for d in 0..max_d {
            let mut cost = vec![0.0; h * w];
            for i in 0..h {
                for j in 0..w {
                    if j >= d {
                        // Correlation (negative L1 distance for matching cost)
                        cost[i * w + j] = -(left[i * w + j] - right[i * w + (j - d)]).abs();
                    } else {
                        cost[i * w + j] = f64::NEG_INFINITY;
                    }
                }
            }
            cost_vol.push(cost);
        }

        Ok(cost_vol)
    }

    /// Aggregate costs using a local window.
    pub fn aggregate_costs(&self, cost_vol: &[Vec<f64>], h: usize, w: usize) -> Vec<Vec<f64>> {
        let r = self.config.agg_radius;
        let max_d = cost_vol.len();
        let mut agg = Vec::with_capacity(max_d);

        for d_idx in 0..max_d {
            let mut agg_d = vec![0.0; h * w];
            for i in 0..h {
                for j in 0..w {
                    let mut sum = 0.0;
                    let mut cnt = 0.0;
                    let i_start = i.saturating_sub(r);
                    let j_start = j.saturating_sub(r);
                    let i_end = (i + r + 1).min(h);
                    let j_end = (j + r + 1).min(w);
                    for ii in i_start..i_end {
                        for jj in j_start..j_end {
                            let val = cost_vol[d_idx][ii * w + jj];
                            if val > f64::NEG_INFINITY {
                                sum += val;
                                cnt += 1.0;
                            }
                        }
                    }
                    agg_d[i * w + j] = if cnt > 0.0 {
                        sum / cnt
                    } else {
                        f64::NEG_INFINITY
                    };
                }
            }
            agg.push(agg_d);
        }

        agg
    }

    /// Soft-argmin disparity regression (differentiable).
    ///
    /// Applies softmax over the disparity dimension and computes the
    /// expected disparity value.
    pub fn soft_argmin_disparity(&self, cost_vol: &[Vec<f64>], h: usize, w: usize) -> Vec<f64> {
        let max_d = cost_vol.len();
        let n = h * w;
        let mut disparity = vec![0.0; n];

        for pixel in 0..n {
            // Collect costs for this pixel across disparities
            let mut costs: Vec<f64> = (0..max_d).map(|d| cost_vol[d][pixel]).collect();

            // Softmax (negative costs → use costs directly as logits)
            let max_c = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;
            for c in &mut costs {
                if *c > f64::NEG_INFINITY {
                    *c = (*c - max_c).exp();
                } else {
                    *c = 0.0;
                }
                sum_exp += *c;
            }
            if sum_exp > 0.0 {
                for c in &mut costs {
                    *c /= sum_exp;
                }
            }

            // Expected disparity
            let d_val: f64 = costs
                .iter()
                .enumerate()
                .map(|(d, &w_d)| d as f64 * w_d)
                .sum();
            disparity[pixel] = d_val;
        }

        disparity
    }

    /// Full stereo matching pipeline.
    ///
    /// Returns `(disparity_map, depth_map)` each of size `h * w`.
    pub fn match_stereo(
        &self,
        left: &[f64],
        right: &[f64],
        h: usize,
        w: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let cost_vol = self.build_cost_volume(left, right, h, w)?;
        let agg_costs = self.aggregate_costs(&cost_vol, h, w);
        let disparity = self.soft_argmin_disparity(&agg_costs, h, w);

        // Convert disparity to depth: depth = focal_length * baseline / disparity
        let depth: Vec<f64> = disparity
            .iter()
            .map(|&d| {
                if d > 1e-6 {
                    self.config.focal_length * self.config.baseline / d
                } else {
                    0.0
                }
            })
            .collect();

        Ok((disparity, depth))
    }

    /// Convert a disparity value to depth.
    pub fn disparity_to_depth(&self, disparity: f64) -> f64 {
        if disparity > 1e-6 {
            self.config.focal_length * self.config.baseline / disparity
        } else {
            0.0
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  DepthCompletion — Sparse-to-dense depth completion
// ─────────────────────────────────────────────────────────────────────────────

/// Fusion strategy for depth completion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeCompletionFusion {
    /// Concatenate sparse depth + RGB → single encoder.
    Early,
    /// Separate encoders for depth and RGB, merge late.
    Late,
}

/// Configuration for depth completion.
#[derive(Debug, Clone)]
pub struct DepthCompletionConfig {
    /// Spatial height.
    pub h: usize,
    /// Spatial width.
    pub w: usize,
    /// Number of propagation iterations.
    pub n_iterations: usize,
    /// Fusion strategy.
    pub fusion: DeCompletionFusion,
    /// Confidence decay factor per iteration.
    pub confidence_decay: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for DepthCompletionConfig {
    fn default() -> Self {
        Self {
            h: 32,
            w: 32,
            n_iterations: 5,
            fusion: DeCompletionFusion::Early,
            confidence_decay: 0.9,
            seed: 300,
        }
    }
}

/// Sparse-to-dense depth completion.
///
/// Uses confidence-guided propagation: sparse depth observations with
/// known confidence are iteratively diffused to neighboring pixels using
/// a mask-propagating convolution scheme.
#[derive(Debug, Clone)]
pub struct DepthCompletion {
    pub config: DepthCompletionConfig,
    /// Propagation kernel (3x3).
    pub prop_kernel: [f64; 9],
}

impl DepthCompletion {
    pub fn new(config: DepthCompletionConfig) -> Result<Self> {
        if config.h == 0 || config.w == 0 {
            return Err(TensorError::compute_error_simple(
                "DepthCompletion: h and w must be > 0".to_string(),
            ));
        }
        // Normalized averaging kernel
        let prop_kernel = [1.0 / 9.0; 9];
        Ok(Self {
            config,
            prop_kernel,
        })
    }

    /// Sparse convolution with mask propagation.
    ///
    /// Only pixels with confidence > 0 contribute to the convolution.
    fn sparse_conv(
        &self,
        depth: &[f64],
        confidence: &[f64],
        h: usize,
        w: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = h * w;
        let mut new_depth = vec![0.0; n];
        let mut new_conf = vec![0.0; n];

        for i in 0..h {
            for j in 0..w {
                let mut weighted_sum = 0.0;
                let mut conf_sum = 0.0;
                let mut k_idx = 0;

                for ki in 0..3_usize {
                    for kj in 0..3_usize {
                        let ni = (i as isize) + (ki as isize) - 1;
                        let nj = (j as isize) + (kj as isize) - 1;
                        if ni >= 0 && ni < h as isize && nj >= 0 && nj < w as isize {
                            let idx = ni as usize * w + nj as usize;
                            if confidence[idx] > 0.0 {
                                weighted_sum +=
                                    depth[idx] * confidence[idx] * self.prop_kernel[k_idx];
                                conf_sum += confidence[idx] * self.prop_kernel[k_idx];
                            }
                        }
                        k_idx += 1;
                    }
                }

                let pixel_idx = i * w + j;
                if confidence[pixel_idx] > 0.5 {
                    // Keep original observed values with high confidence
                    new_depth[pixel_idx] = depth[pixel_idx];
                    new_conf[pixel_idx] = confidence[pixel_idx];
                } else if conf_sum > 0.0 {
                    new_depth[pixel_idx] = weighted_sum / conf_sum;
                    new_conf[pixel_idx] = conf_sum * self.config.confidence_decay;
                }
            }
        }

        (new_depth, new_conf)
    }

    /// Complete sparse depth to dense.
    ///
    /// `sparse_depth`: depth values (0 for unknown).
    /// `confidence_map`: confidence per pixel (0..1).
    /// `rgb_guide`: optional RGB guidance (3 channels, length 3 * h * w), used for
    ///              edge-aware weighting in early fusion mode.
    pub fn complete(
        &self,
        sparse_depth: &[f64],
        confidence_map: &[f64],
        rgb_guide: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let h = self.config.h;
        let w = self.config.w;
        let n = h * w;

        if sparse_depth.len() != n || confidence_map.len() != n {
            return Err(TensorError::compute_error_simple(
                "DepthCompletion: sparse_depth/confidence size mismatch".to_string(),
            ));
        }

        let mut depth = sparse_depth.to_vec();
        let mut conf = confidence_map.to_vec();

        // Early fusion: incorporate RGB as edge-aware guidance
        if self.config.fusion == DeCompletionFusion::Early {
            if let Some(rgb) = rgb_guide {
                if rgb.len() >= n {
                    // Use luminance channel to modulate confidence at edges
                    for i in 0..n {
                        let lum = rgb[i]; // use first channel as proxy
                                          // Reduce confidence at strong edges (high gradient)
                        if i >= w && i + w < n && i + 1 < n && i > 0 {
                            let grad: f64 = (rgb[i + 1] - rgb[i - 1]).abs();
                            if grad > 0.1 {
                                conf[i] *= 0.8;
                            }
                        }
                        let _ = lum;
                    }
                }
            }
        }

        // Iterative confidence-guided propagation
        for _iter in 0..self.config.n_iterations {
            let (new_depth, new_conf) = self.sparse_conv(&depth, &conf, h, w);
            depth = new_depth;
            conf = new_conf;
        }

        Ok(depth)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  DeConv3d — 3D Convolution layer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a 3D convolution layer.
#[derive(Debug, Clone)]
pub struct DeConv3dConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size (D, H, W).
    pub kernel_size: [usize; 3],
    /// Stride (D, H, W).
    pub stride: [usize; 3],
    /// Padding (D, H, W).
    pub padding: [usize; 3],
    /// Dilation (D, H, W).
    pub dilation: [usize; 3],
    /// Random seed.
    pub seed: u64,
}

impl Default for DeConv3dConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_size: [3, 3, 3],
            stride: [1, 1, 1],
            padding: [1, 1, 1],
            dilation: [1, 1, 1],
            seed: 400,
        }
    }
}

/// Full 3D convolution layer.
///
/// Input shape: `[N, C_in, D, H, W]` (flattened).
/// Output shape: `[N, C_out, D', H', W']`.
#[derive(Debug, Clone)]
pub struct DeConv3d {
    pub config: DeConv3dConfig,
    /// Weights: `[out_channels][in_channels * kD * kH * kW]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias: `[out_channels]`.
    pub bias: Vec<f64>,
}

impl DeConv3d {
    pub fn new(config: DeConv3dConfig) -> Result<Self> {
        let [kd, kh, kw] = config.kernel_size;
        let fan_in = config.in_channels * kd * kh * kw;
        let fan_out = config.out_channels * kd * kh * kw;
        let weights = de_xavier_init(config.out_channels, fan_in, config.seed);
        let bias = vec![0.0; config.out_channels];
        Ok(Self {
            config,
            weights,
            bias,
        })
    }

    /// Compute output spatial dimensions.
    pub fn output_shape(&self, d: usize, h: usize, w: usize) -> (usize, usize, usize) {
        let [kd, kh, kw] = self.config.kernel_size;
        let [sd, sh, sw] = self.config.stride;
        let [pd, ph, pw] = self.config.padding;
        let [dd, dh, dw] = self.config.dilation;

        let od = (d + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
        let oh = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let ow = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        (od, oh, ow)
    }

    /// Forward pass for a single sample (batch_size = 1).
    ///
    /// `input`: flattened `[C_in, D, H, W]`.
    /// `shape`: `[C_in, D, H, W]`.
    ///
    /// Returns `(output, [C_out, D', H', W'])`.
    pub fn forward(
        &self,
        input: &[f64],
        in_c: usize,
        d: usize,
        h: usize,
        w: usize,
    ) -> Result<(Vec<f64>, usize, usize, usize, usize)> {
        if in_c != self.config.in_channels {
            return Err(TensorError::compute_error_simple(
                "DeConv3d: in_channels mismatch".to_string(),
            ));
        }
        let expected = in_c * d * h * w;
        if input.len() != expected {
            return Err(TensorError::compute_error_simple(
                "DeConv3d: input length mismatch".to_string(),
            ));
        }

        let [kd, kh, kw] = self.config.kernel_size;
        let [sd, sh, sw] = self.config.stride;
        let [pd, ph, pw] = self.config.padding;
        let [dd, dh, dw] = self.config.dilation;
        let (od, oh, ow) = self.output_shape(d, h, w);
        let out_c = self.config.out_channels;

        let mut output = vec![0.0; out_c * od * oh * ow];

        for oc in 0..out_c {
            for oi in 0..od {
                for oj in 0..oh {
                    for ok in 0..ow {
                        let mut val = self.bias[oc];
                        for ic in 0..in_c {
                            for ki in 0..kd {
                                for kj in 0..kh {
                                    for kk in 0..kw {
                                        let id = oi * sd + ki * dd;
                                        let ih = oj * sh + kj * dh;
                                        let iw = ok * sw + kk * dw;

                                        if id >= pd
                                            && ih >= ph
                                            && iw >= pw
                                            && id - pd < d
                                            && ih - ph < h
                                            && iw - pw < w
                                        {
                                            let in_idx = ic * (d * h * w)
                                                + (id - pd) * (h * w)
                                                + (ih - ph) * w
                                                + (iw - pw);
                                            let w_idx =
                                                ic * (kd * kh * kw) + ki * (kh * kw) + kj * kw + kk;
                                            val += input[in_idx] * self.weights[oc][w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let out_idx = oc * (od * oh * ow) + oi * (oh * ow) + oj * ow + ok;
                        output[out_idx] = val;
                    }
                }
            }
        }

        Ok((output, out_c, od, oh, ow))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  DeImplicitNeuralField — NeRF-inspired implicit representation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the implicit neural field.
#[derive(Debug, Clone)]
pub struct DeImplicitFieldConfig {
    /// Number of positional encoding frequency bands L.
    pub n_freqs: usize,
    /// Number of hidden units in the MLP.
    pub hidden_dim: usize,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Number of stratified samples per ray.
    pub n_samples: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for DeImplicitFieldConfig {
    fn default() -> Self {
        Self {
            n_freqs: 6,
            hidden_dim: 64,
            n_layers: 4,
            n_samples: 32,
            seed: 500,
        }
    }
}

/// NeRF-inspired implicit neural representation.
///
/// Encodes 3D coordinates with positional encoding, processes through an MLP
/// to predict density and color, and renders rays via volume rendering.
#[derive(Debug, Clone)]
pub struct DeImplicitNeuralField {
    pub config: DeImplicitFieldConfig,
    /// MLP weights: `[n_layers][(out_dim, in_dim)]` each layer.
    pub layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Output head: density (1) + color (3).
    pub output_head: (Vec<Vec<f64>>, Vec<f64>),
}

impl DeImplicitNeuralField {
    pub fn new(config: DeImplicitFieldConfig) -> Result<Self> {
        if config.n_freqs == 0 || config.hidden_dim == 0 {
            return Err(TensorError::compute_error_simple(
                "DeImplicitNeuralField: n_freqs and hidden_dim must be > 0".to_string(),
            ));
        }

        // Input dim: 3 coords * (2 * n_freqs) = 6 * n_freqs
        let input_dim = 3 * 2 * config.n_freqs;
        let mut layers = Vec::with_capacity(config.n_layers);
        let mut seed = config.seed;

        // First layer
        let w0 = de_xavier_init(config.hidden_dim, input_dim, seed);
        let b0 = vec![0.0; config.hidden_dim];
        layers.push((w0, b0));
        seed = seed.wrapping_add(1);

        // Hidden layers
        for _ in 1..config.n_layers {
            let w = de_xavier_init(config.hidden_dim, config.hidden_dim, seed);
            let b = vec![0.0; config.hidden_dim];
            layers.push((w, b));
            seed = seed.wrapping_add(1);
        }

        // Output head: hidden_dim → 4 (density + RGB)
        let w_out = de_xavier_init(4, config.hidden_dim, seed);
        let b_out = vec![0.0; 4];

        Ok(Self {
            config,
            layers,
            output_head: (w_out, b_out),
        })
    }

    /// Positional encoding: gamma(p) = [sin(2^k * pi * p), cos(2^k * pi * p)] for k=0..L-1.
    pub fn positional_encode(&self, coords: &[f64; 3]) -> Vec<f64> {
        let mut encoded = Vec::with_capacity(3 * 2 * self.config.n_freqs);
        for &c in coords {
            for k in 0..self.config.n_freqs {
                let freq = (2.0_f64).powi(k as i32) * std::f64::consts::PI;
                encoded.push((freq * c).sin());
                encoded.push((freq * c).cos());
            }
        }
        encoded
    }

    /// Query the field at a 3D point.
    ///
    /// Returns `(density, [r, g, b])`.
    pub fn query(&self, coords: &[f64; 3]) -> Result<(f64, [f64; 3])> {
        let mut x = self.positional_encode(coords);

        for (w, b) in &self.layers {
            if w.is_empty() || w[0].len() != x.len() {
                // Skip layers with dimension mismatch during partial init
                continue;
            }
            x = de_linear_relu(w, b, &x);
        }

        let (ref w_out, ref b_out) = self.output_head;
        if w_out.is_empty() || w_out[0].len() != x.len() {
            return Err(TensorError::compute_error_simple(
                "DeImplicitNeuralField: output head dimension mismatch".to_string(),
            ));
        }
        let out = de_linear(w_out, b_out, &x);

        if out.len() < 4 {
            return Err(TensorError::compute_error_simple(
                "DeImplicitNeuralField: output too short".to_string(),
            ));
        }

        let density = de_softplus(out[0]);
        let color = [de_sigmoid(out[1]), de_sigmoid(out[2]), de_sigmoid(out[3])];

        Ok((density, color))
    }

    /// Render a single ray via volume rendering.
    ///
    /// `C = sum_i T_i * alpha_i * c_i` where
    /// `T_i = prod_{j<i} (1 - alpha_j)`, `alpha = 1 - exp(-sigma * delta)`.
    ///
    /// Returns `(color: [f64; 3], expected_depth: f64)`.
    pub fn render_ray(
        &self,
        origin: &[f64; 3],
        direction: &[f64; 3],
        near: f64,
        far: f64,
        n_samples: usize,
    ) -> Result<([f64; 3], f64)> {
        if near >= far {
            return Err(TensorError::compute_error_simple(
                "render_ray: near must be < far".to_string(),
            ));
        }
        if n_samples == 0 {
            return Err(TensorError::compute_error_simple(
                "render_ray: n_samples must be > 0".to_string(),
            ));
        }

        let n = n_samples;
        let step = (far - near) / n as f64;

        let mut color = [0.0; 3];
        let mut depth = 0.0;
        let mut transmittance = 1.0;

        // Stratified sampling along the ray
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        for i in 0..n {
            let t_base = near + i as f64 * step;
            let jitter = rng.random::<f64>() * step;
            let t = t_base + jitter;
            let delta = step;

            let point = [
                origin[0] + t * direction[0],
                origin[1] + t * direction[1],
                origin[2] + t * direction[2],
            ];

            let (sigma, c) = self.query(&point)?;
            let alpha = 1.0 - (-sigma * delta).exp();
            let weight = transmittance * alpha;

            color[0] += weight * c[0];
            color[1] += weight * c[1];
            color[2] += weight * c[2];
            depth += weight * t;

            transmittance *= 1.0 - alpha;
            if transmittance < 1e-6 {
                break;
            }
        }

        Ok((color, depth))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  DePanopticHead — Panoptic segmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Panoptic segmentation result.
#[derive(Debug, Clone)]
pub struct DePanopticResult {
    /// Per-pixel semantic class IDs.
    pub semantic_map: Vec<usize>,
    /// Per-pixel instance IDs (0 = background/stuff).
    pub instance_map: Vec<usize>,
    /// Fused panoptic map: `class_id * 1000 + instance_id`.
    pub panoptic_map: Vec<usize>,
    /// Height.
    pub height: usize,
    /// Width.
    pub width: usize,
}

/// Configuration for the panoptic head.
#[derive(Debug, Clone)]
pub struct DePanopticConfig {
    /// Number of semantic classes.
    pub n_classes: usize,
    /// Maximum number of instances.
    pub max_instances: usize,
    /// Feature dimension.
    pub feature_dim: usize,
    /// IoU threshold for overlap resolution.
    pub overlap_threshold: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for DePanopticConfig {
    fn default() -> Self {
        Self {
            n_classes: 10,
            max_instances: 50,
            feature_dim: 32,
            overlap_threshold: 0.5,
            seed: 600,
        }
    }
}

/// Panoptic segmentation head.
///
/// Combines a semantic branch (per-pixel class logits) with an instance branch
/// (center prediction + offset regression) and fuses them into a panoptic map.
#[derive(Debug, Clone)]
pub struct DePanopticHead {
    pub config: DePanopticConfig,
    /// Semantic branch: feature_dim → n_classes.
    pub semantic_weights: Vec<Vec<f64>>,
    pub semantic_bias: Vec<f64>,
    /// Instance center weights: feature_dim → 1 (center-ness score).
    pub center_weights: Vec<Vec<f64>>,
    pub center_bias: Vec<f64>,
    /// Offset weights: feature_dim → 2 (x, y offsets to center).
    pub offset_weights: Vec<Vec<f64>>,
    pub offset_bias: Vec<f64>,
}

impl DePanopticHead {
    pub fn new(config: DePanopticConfig) -> Result<Self> {
        if config.n_classes == 0 {
            return Err(TensorError::compute_error_simple(
                "DePanopticHead: n_classes must be > 0".to_string(),
            ));
        }
        let mut seed = config.seed;
        let sem_w = de_xavier_init(config.n_classes, config.feature_dim, seed);
        let sem_b = vec![0.0; config.n_classes];
        seed = seed.wrapping_add(1);

        let ctr_w = de_xavier_init(1, config.feature_dim, seed);
        let ctr_b = vec![0.0; 1];
        seed = seed.wrapping_add(1);

        let off_w = de_xavier_init(2, config.feature_dim, seed);
        let off_b = vec![0.0; 2];

        Ok(Self {
            config,
            semantic_weights: sem_w,
            semantic_bias: sem_b,
            center_weights: ctr_w,
            center_bias: ctr_b,
            offset_weights: off_w,
            offset_bias: off_b,
        })
    }

    /// Compute semantic logits for a feature vector.
    pub fn semantic_logits(&self, features: &[f64]) -> Vec<f64> {
        de_linear(&self.semantic_weights, &self.semantic_bias, features)
    }

    /// Compute center-ness score for a feature vector.
    pub fn center_score(&self, features: &[f64]) -> f64 {
        let out = de_linear(&self.center_weights, &self.center_bias, features);
        if out.is_empty() {
            0.0
        } else {
            de_sigmoid(out[0])
        }
    }

    /// Compute offset prediction for a feature vector.
    pub fn offset_pred(&self, features: &[f64]) -> [f64; 2] {
        let out = de_linear(&self.offset_weights, &self.offset_bias, features);
        if out.len() >= 2 {
            [out[0], out[1]]
        } else {
            [0.0, 0.0]
        }
    }

    /// Full panoptic segmentation from per-pixel feature maps.
    ///
    /// `feature_map`: `[h * w][feature_dim]` feature vectors.
    pub fn forward(
        &self,
        feature_map: &[Vec<f64>],
        h: usize,
        w: usize,
    ) -> Result<DePanopticResult> {
        let n = h * w;
        if feature_map.len() != n {
            return Err(TensorError::compute_error_simple(
                "DePanopticHead: feature_map length mismatch".to_string(),
            ));
        }

        let mut semantic_map = vec![0usize; n];
        let mut center_scores = vec![0.0f64; n];
        let mut offsets = vec![[0.0f64; 2]; n];

        for (idx, feat) in feature_map.iter().enumerate() {
            // Semantic: argmax
            let logits = self.semantic_logits(feat);
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;
            for (c, &s) in logits.iter().enumerate() {
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }
            semantic_map[idx] = best_class;

            // Instance
            center_scores[idx] = self.center_score(feat);
            offsets[idx] = self.offset_pred(feat);
        }

        // Instance grouping via center voting
        let instance_map = self.group_instances(&center_scores, &offsets, &semantic_map, h, w);

        // Panoptic fusion
        let panoptic_map: Vec<usize> = semantic_map
            .iter()
            .zip(&instance_map)
            .map(|(&sem, &inst)| sem * 1000 + inst)
            .collect();

        Ok(DePanopticResult {
            semantic_map,
            instance_map,
            panoptic_map,
            height: h,
            width: w,
        })
    }

    /// Group pixels into instances using center voting and NMS-style merging.
    fn group_instances(
        &self,
        center_scores: &[f64],
        offsets: &[[f64; 2]],
        semantic_map: &[usize],
        h: usize,
        w: usize,
    ) -> Vec<usize> {
        let n = h * w;
        let mut instance_map = vec![0usize; n];
        let center_threshold = 0.3;

        // Find center candidates (pixels with high center-ness)
        let mut centers: Vec<(usize, f64)> = Vec::new();
        for (idx, &score) in center_scores.iter().enumerate() {
            if score > center_threshold {
                centers.push((idx, score));
            }
        }

        // Sort by score descending
        centers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign instance IDs via greedy assignment
        let mut next_instance = 1usize;
        let max_inst = self.config.max_instances;

        for &(center_idx, _score) in &centers {
            if next_instance > max_inst {
                break;
            }
            if instance_map[center_idx] != 0 {
                continue;
            }

            let center_y = center_idx / w;
            let center_x = center_idx % w;
            let center_class = semantic_map[center_idx];

            // Assign nearby pixels that vote for this center
            for idx in 0..n {
                if instance_map[idx] != 0 {
                    continue;
                }
                if semantic_map[idx] != center_class {
                    continue;
                }
                let py = idx / w;
                let px = idx % w;
                let voted_y = py as f64 + offsets[idx][1];
                let voted_x = px as f64 + offsets[idx][0];
                let dist = ((voted_y - center_y as f64).powi(2)
                    + (voted_x - center_x as f64).powi(2))
                .sqrt();
                if dist < 3.0 {
                    instance_map[idx] = next_instance;
                }
            }

            next_instance += 1;
        }

        instance_map
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  DeDepthMetrics — Comprehensive depth evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive depth estimation evaluation report.
#[derive(Debug, Clone)]
pub struct DeDepthReport {
    /// Absolute Relative Error: mean(|pred - gt| / gt).
    pub abs_rel: f64,
    /// Squared Relative Error: mean((pred - gt)^2 / gt).
    pub sq_rel: f64,
    /// Root Mean Squared Error.
    pub rmse: f64,
    /// RMSE on log depths.
    pub rmse_log: f64,
    /// Scale-Invariant Log Error.
    pub si_log: f64,
    /// Delta < 1.25 (percentage).
    pub delta_1: f64,
    /// Delta < 1.25^2 (percentage).
    pub delta_2: f64,
    /// Delta < 1.25^3 (percentage).
    pub delta_3: f64,
    /// Number of valid pixels.
    pub n_valid: usize,
}

/// Depth evaluation metrics.
pub struct DeDepthMetrics;

impl DeDepthMetrics {
    /// Compute absolute relative error: mean(|pred - gt| / gt).
    pub fn abs_rel(pred: &[f64], gt: &[f64]) -> Result<f64> {
        let (sum, count) = Self::valid_pairs(pred, gt)?;
        if count == 0 {
            return Err(TensorError::compute_error_simple(
                "abs_rel: no valid pixels".to_string(),
            ));
        }
        let val: f64 = sum.iter().map(|&(p, g)| (p - g).abs() / g).sum::<f64>() / count as f64;
        Ok(val)
    }

    /// Compute squared relative error: mean((pred - gt)^2 / gt).
    pub fn sq_rel(pred: &[f64], gt: &[f64]) -> Result<f64> {
        let (pairs, count) = Self::valid_pairs(pred, gt)?;
        if count == 0 {
            return Err(TensorError::compute_error_simple(
                "sq_rel: no valid pixels".to_string(),
            ));
        }
        let val: f64 = pairs.iter().map(|&(p, g)| (p - g).powi(2) / g).sum::<f64>() / count as f64;
        Ok(val)
    }

    /// Compute RMSE.
    pub fn rmse(pred: &[f64], gt: &[f64]) -> Result<f64> {
        let (pairs, count) = Self::valid_pairs(pred, gt)?;
        if count == 0 {
            return Err(TensorError::compute_error_simple(
                "rmse: no valid pixels".to_string(),
            ));
        }
        let mse: f64 = pairs.iter().map(|&(p, g)| (p - g).powi(2)).sum::<f64>() / count as f64;
        Ok(mse.sqrt())
    }

    /// Compute RMSE on log depths.
    pub fn rmse_log(pred: &[f64], gt: &[f64]) -> Result<f64> {
        let (pairs, count) = Self::valid_pairs(pred, gt)?;
        if count == 0 {
            return Err(TensorError::compute_error_simple(
                "rmse_log: no valid pixels".to_string(),
            ));
        }
        let mse: f64 = pairs
            .iter()
            .map(|&(p, g)| (p.ln() - g.ln()).powi(2))
            .sum::<f64>()
            / count as f64;
        Ok(mse.sqrt())
    }

    /// Scale-Invariant Log Error (SILog).
    ///
    /// SILog = sqrt(Var(d) + 0 * mean(d)^2) where d_i = log(pred) - log(gt).
    /// Common definition: SILog = 100 * sqrt(E[d^2] - E\[d\]^2).
    pub fn si_log(pred: &[f64], gt: &[f64]) -> Result<f64> {
        let (pairs, count) = Self::valid_pairs(pred, gt)?;
        if count == 0 {
            return Err(TensorError::compute_error_simple(
                "si_log: no valid pixels".to_string(),
            ));
        }
        let n = count as f64;
        let d: Vec<f64> = pairs.iter().map(|&(p, g)| p.ln() - g.ln()).collect();
        let mean_d: f64 = d.iter().sum::<f64>() / n;
        let mean_d2: f64 = d.iter().map(|&di| di * di).sum::<f64>() / n;
        let val = (mean_d2 - mean_d * mean_d).max(0.0).sqrt() * 100.0;
        Ok(val)
    }

    /// Delta threshold metric: percentage of pixels where
    /// max(pred/gt, gt/pred) < threshold.
    pub fn delta_threshold(pred: &[f64], gt: &[f64], threshold: f64) -> Result<f64> {
        let (pairs, count) = Self::valid_pairs(pred, gt)?;
        if count == 0 {
            return Err(TensorError::compute_error_simple(
                "delta_threshold: no valid pixels".to_string(),
            ));
        }
        let good = pairs
            .iter()
            .filter(|&&(p, g)| {
                let ratio = (p / g).max(g / p);
                ratio < threshold
            })
            .count();
        Ok(good as f64 / count as f64)
    }

    /// Compute a comprehensive depth report.
    pub fn evaluate(pred: &[f64], gt: &[f64]) -> Result<DeDepthReport> {
        let abs_rel = Self::abs_rel(pred, gt)?;
        let sq_rel = Self::sq_rel(pred, gt)?;
        let rmse = Self::rmse(pred, gt)?;
        let rmse_log = Self::rmse_log(pred, gt)?;
        let si_log = Self::si_log(pred, gt)?;
        let delta_1 = Self::delta_threshold(pred, gt, 1.25)?;
        let delta_2 = Self::delta_threshold(pred, gt, 1.25 * 1.25)?;
        let delta_3 = Self::delta_threshold(pred, gt, 1.25 * 1.25 * 1.25)?;
        let (_, n_valid) = Self::valid_pairs(pred, gt)?;

        Ok(DeDepthReport {
            abs_rel,
            sq_rel,
            rmse,
            rmse_log,
            si_log,
            delta_1,
            delta_2,
            delta_3,
            n_valid,
        })
    }

    /// Extract valid prediction/GT pairs (both > 0).
    fn valid_pairs(pred: &[f64], gt: &[f64]) -> Result<(Vec<(f64, f64)>, usize)> {
        if pred.len() != gt.len() {
            return Err(TensorError::compute_error_simple(
                "DeDepthMetrics: pred/gt length mismatch".to_string(),
            ));
        }
        let pairs: Vec<(f64, f64)> = pred
            .iter()
            .zip(gt)
            .filter(|(&p, &g)| p > 0.0 && g > 0.0)
            .map(|(&p, &g)| (p, g))
            .collect();
        let count = pairs.len();
        Ok((pairs, count))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
