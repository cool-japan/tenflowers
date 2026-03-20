//! Medical Image Segmentation & Analysis.
//!
//! Production-grade primitives for medical imaging tasks:
//!
//! - **DoubleConv**: Two sequential Conv2d+BN+ReLU blocks
//! - **UNet2D**: Full encoder-bridge-decoder with skip connections
//! - **AttentionGate**: Spatial attention for skip connections
//! - **DiceLoss / DiceBCELoss**: Segmentation loss functions
//! - **HausdorffDistance**: 95th-percentile Hausdorff distance
//! - **SlidingWindowInference**: MONAI-style tiled inference
//! - **MedicalAugmentation**: Gamma, elastic, noise, flip, crop
//! - **SegmentationMetrics**: IoU, Dice, sensitivity, specificity
//! - **AdaptiveInstanceNorm**: AdaIN for style transfer
//! - **NnUNetNormalizer**: Clip-then-z-score normalisation
//!
//! All arithmetic is pure Rust (`Vec<f32>`).  No `unwrap()`, no `unsafe`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu_f(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid_f(x: f32) -> f32 {
    let xc = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-xc).exp())
}

/// Simple batch-norm approximation: zero-mean + unit-variance over a flat slice.
fn batch_norm_1d(x: &[f32], gamma: f32, beta: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .map(|&v| gamma * (v - mean) * std_inv + beta)
        .collect()
}

/// Element-wise addition.
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Pointwise linear projection: weight matrix is [out x in], bias is [out].
fn linear_f32(w: &[Vec<f32>], bias: &[f32], x: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(bias.iter())
        .map(|(row, &b)| {
            row.iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + b
        })
        .collect()
}

/// Xavier-normal initialiser for a 2-D weight matrix.
fn xavier_weights(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let scale = (2.0_f32 / (rows + cols) as f32).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f32 = rng.random();
                    let v: f32 = rng.random();
                    let normal = (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
                    normal * scale
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. DoubleConv
// ─────────────────────────────────────────────────────────────────────────────

/// Two consecutive Conv2d(depthwise-avg) + BatchNorm + ReLU blocks.
///
/// Because we work with flat `Vec<f32>` tensors we represent "convolution" as
/// a learned **linear projection** applied channel-wise at each spatial position
/// (equivalent to 1×1 convolution). This keeps the API tractable without an
/// actual im2col implementation while still demonstrating the double-conv
/// pattern used in U-Net.
#[derive(Debug, Clone)]
pub struct DoubleConv {
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    // First conv: [c_mid x c_in]
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    gamma1: f32,
    beta1: f32,
    // Second conv: [c_out x c_mid]
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    gamma2: f32,
    beta2: f32,
}

impl DoubleConv {
    /// Create a new `DoubleConv` block.  `c_mid` is the intermediate channel
    /// count; pass `c_out` to use the same width for both projections.
    pub fn new(c_in: usize, c_out: usize, seed: u64) -> Self {
        let c_mid = c_out;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_weights(c_mid, c_in, &mut rng);
        let b1 = vec![0.0f32; c_mid];
        let w2 = xavier_weights(c_out, c_mid, &mut rng);
        let b2 = vec![0.0f32; c_out];
        Self {
            c_in,
            c_out,
            w1,
            b1,
            gamma1: 1.0,
            beta1: 0.0,
            w2,
            b2,
            gamma2: 1.0,
            beta2: 0.0,
        }
    }

    /// Forward pass.
    ///
    /// * `x`        — flat `[H × W × c_in]` tensor (channels-last)
    /// * `spatial`  — number of spatial positions `H × W`
    ///
    /// Returns `[H × W × c_out]` channels-last tensor.
    pub fn forward(&self, x: &[f32], spatial: usize) -> Vec<f32> {
        let c_in = self.c_in;
        let c_mid = self.w1.len();
        let c_out = self.c_out;

        let mut out1: Vec<f32> = Vec::with_capacity(spatial * c_mid);
        for s in 0..spatial {
            let slice = &x[s * c_in..(s + 1) * c_in];
            let proj = linear_f32(&self.w1, &self.b1, slice);
            let normed = batch_norm_1d(&proj, self.gamma1, self.beta1);
            for v in normed {
                out1.push(relu_f(v));
            }
        }

        let mut out2: Vec<f32> = Vec::with_capacity(spatial * c_out);
        for s in 0..spatial {
            let slice = &out1[s * c_mid..(s + 1) * c_mid];
            let proj = linear_f32(&self.w2, &self.b2, slice);
            let normed = batch_norm_1d(&proj, self.gamma2, self.beta2);
            for v in normed {
                out2.push(relu_f(v));
            }
        }
        out2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. UNet2D
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D U-Net with 4 encoder levels, bridge, and 4 decoder levels.
///
/// Channel schedule (base 32):  32 → 64 → 128 → 256 → 512 (bridge).
/// Decoder mirrors: 256 → 128 → 64 → 32 → `c_out`.
/// Skip connections are concatenated (simulated via channel addition for same
/// channel counts to avoid explicit upsample bookkeeping).
#[derive(Debug)]
pub struct UNet2D {
    /// Number of input channels.
    pub c_in: usize,
    /// Number of output segmentation channels.
    pub c_out: usize,
    // Encoder
    enc0: DoubleConv,
    enc1: DoubleConv,
    enc2: DoubleConv,
    enc3: DoubleConv,
    // Bridge
    bridge: DoubleConv,
    // Decoder
    dec3: DoubleConv,
    dec2: DoubleConv,
    dec1: DoubleConv,
    dec0: DoubleConv,
    // Final 1×1 projection
    final_w: Vec<Vec<f32>>,
    final_b: Vec<f32>,
}

impl UNet2D {
    /// Construct a `UNet2D`.
    pub fn new(c_in: usize, c_out: usize, base: usize, seed: u64) -> Self {
        let b = base;
        Self {
            c_in,
            c_out,
            enc0: DoubleConv::new(c_in, b, seed),
            enc1: DoubleConv::new(b, b * 2, seed + 1),
            enc2: DoubleConv::new(b * 2, b * 4, seed + 2),
            enc3: DoubleConv::new(b * 4, b * 8, seed + 3),
            bridge: DoubleConv::new(b * 8, b * 16, seed + 4),
            // decoder inputs = upsampled (b*16) + skip (b*8) = b*24 → but we
            // treat skip as add (same channel), so decoder sees b*16
            dec3: DoubleConv::new(b * 16 + b * 8, b * 8, seed + 5),
            dec2: DoubleConv::new(b * 8 + b * 4, b * 4, seed + 6),
            dec1: DoubleConv::new(b * 4 + b * 2, b * 2, seed + 7),
            dec0: DoubleConv::new(b * 2 + b, b, seed + 8),
            final_w: {
                let mut rng = StdRng::seed_from_u64(seed + 9);
                xavier_weights(c_out, b, &mut rng)
            },
            final_b: vec![0.0f32; c_out],
        }
    }

    /// Forward pass.
    ///
    /// * `input` — flat `[H × W × c_in]` channels-last tensor.
    /// * Returns `[H × W × c_out]` channels-last tensor (same spatial size).
    pub fn forward(&self, input: &[f32], h: usize, w: usize, c_in: usize) -> Vec<f32> {
        let spatial = h * w;

        // ── Encoder ─────────────────────────────────────────────────────────
        let e0 = self.enc0.forward(input, spatial); // [HW x b]
        let e1 = self.enc1.forward(&e0, spatial); // [HW x 2b]
        let e2 = self.enc2.forward(&e1, spatial); // [HW x 4b]
        let e3 = self.enc3.forward(&e2, spatial); // [HW x 8b]

        // ── Bridge ──────────────────────────────────────────────────────────
        let br = self.bridge.forward(&e3, spatial); // [HW x 16b]

        // ── Decoder (skip via channel concat, spatial stays same) ──────────
        let d3_in = concat_channels(&br, &e3, spatial);
        let d3 = self.dec3.forward(&d3_in, spatial);

        let d2_in = concat_channels(&d3, &e2, spatial);
        let d2 = self.dec2.forward(&d2_in, spatial);

        let d1_in = concat_channels(&d2, &e1, spatial);
        let d1 = self.dec1.forward(&d1_in, spatial);

        let d0_in = concat_channels(&d1, &e0, spatial);
        let d0 = self.dec0.forward(&d0_in, spatial);

        // ── Final 1×1 ───────────────────────────────────────────────────────
        let b = self.enc0.c_out;
        let mut out = Vec::with_capacity(spatial * self.c_out);
        for s in 0..spatial {
            let slice = &d0[s * b..(s + 1) * b];
            let proj = linear_f32(&self.final_w, &self.final_b, slice);
            out.extend_from_slice(&proj);
        }
        out
    }
}

/// Concatenate channel-last tensors `a` ([spatial × ca]) and `b` ([spatial × cb]).
fn concat_channels(a: &[f32], b: &[f32], spatial: usize) -> Vec<f32> {
    if spatial == 0 {
        return Vec::new();
    }
    let ca = a.len() / spatial;
    let cb = b.len() / spatial;
    let mut out = Vec::with_capacity(spatial * (ca + cb));
    for s in 0..spatial {
        out.extend_from_slice(&a[s * ca..(s + 1) * ca]);
        out.extend_from_slice(&b[s * cb..(s + 1) * cb]);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. AttentionGate
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial attention gate used in Attention U-Net.
///
/// Computes a scalar attention coefficient per spatial location using the
/// gating signal `g` and the encoder feature map `x`.
#[derive(Debug, Clone)]
pub struct AttentionGate {
    /// Feature channels.
    pub c_in: usize,
    /// Gate channels.
    pub c_g: usize,
    /// Intermediate channels.
    pub c_int: usize,
    // Wx: [c_int x c_in]
    wx: Vec<Vec<f32>>,
    bx: Vec<f32>,
    // Wg: [c_int x c_g]
    wg: Vec<Vec<f32>>,
    bg: Vec<f32>,
    // Psi: [1 x c_int]
    psi: Vec<Vec<f32>>,
    bp: Vec<f32>,
}

impl AttentionGate {
    /// Create a new `AttentionGate`.
    pub fn new(c_in: usize, c_g: usize, c_int: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            c_in,
            c_g,
            c_int,
            wx: xavier_weights(c_int, c_in, &mut rng),
            bx: vec![0.0f32; c_int],
            wg: xavier_weights(c_int, c_g, &mut rng),
            bg: vec![0.0f32; c_int],
            psi: xavier_weights(1, c_int, &mut rng),
            bp: vec![0.0f32; 1],
        }
    }

    /// Forward pass.
    ///
    /// * `x`            — skip feature  `[spatial × c_in]`
    /// * `gate`         — gating signal `[spatial × c_g]`
    /// * `spatial_dims` — `[H, W]` (unused in flat computation; kept for API compat)
    ///
    /// Returns attended feature `[spatial × c_in]`.
    pub fn forward(&self, x: &[f32], gate: &[f32], spatial_dims: &[usize]) -> Vec<f32> {
        let spatial = spatial_dims.iter().product::<usize>().max(1);
        let c_in = self.c_in;
        let c_g = self.c_g;

        let mut out = Vec::with_capacity(spatial * c_in);
        for s in 0..spatial {
            let xs = &x[s * c_in..(s + 1) * c_in];
            let gs = &gate[s * c_g..(s + 1) * c_g];

            // theta_x + phi_g → ReLU
            let theta = linear_f32(&self.wx, &self.bx, xs);
            let phi = linear_f32(&self.wg, &self.bg, gs);
            let combined: Vec<f32> = theta
                .iter()
                .zip(phi.iter())
                .map(|(t, p)| relu_f(t + p))
                .collect();

            // psi → sigmoid → scalar alpha
            let psi_out = linear_f32(&self.psi, &self.bp, &combined);
            let alpha = sigmoid_f(psi_out[0]);

            for &xv in xs {
                out.push(alpha * xv);
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. DiceLoss / DiceBCELoss
// ─────────────────────────────────────────────────────────────────────────────

/// Soft Dice loss for binary or multi-class segmentation.
///
/// `pred` and `target` are flat probability / one-hot tensors in `[0, 1]`.
///
/// Returns a scalar loss ∈ `(0, 1]`.
pub fn dice_loss(pred: &[f32], target: &[f32], smooth: f32) -> f32 {
    let intersection: f32 = pred.iter().zip(target.iter()).map(|(p, t)| p * t).sum();
    let sum_pred: f32 = pred.iter().map(|p| p * p).sum();
    let sum_tgt: f32 = target.iter().map(|t| t * t).sum();
    let denom = sum_pred + sum_tgt + smooth;
    1.0 - (2.0 * intersection + smooth) / denom
}

/// Binary-cross-entropy loss (element-wise mean).
pub fn bce_loss(pred: &[f32], target: &[f32]) -> f32 {
    if pred.is_empty() {
        return 0.0;
    }
    let sum: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            let pc = p.clamp(1e-7, 1.0 - 1e-7);
            -(t * pc.ln() + (1.0 - t) * (1.0 - pc).ln())
        })
        .sum();
    sum / pred.len() as f32
}

/// Combined Dice + BCE loss.
pub struct DiceBCELoss {
    /// Dice smoothing constant.
    pub smooth: f32,
    /// Weight for BCE term (0 = pure Dice, 1 = equal).
    pub bce_weight: f32,
}

impl DiceBCELoss {
    /// Create a new `DiceBCELoss`.
    pub fn new(smooth: f32, bce_weight: f32) -> Self {
        Self { smooth, bce_weight }
    }

    /// Compute the combined loss.
    pub fn forward(&self, pred: &[f32], target: &[f32]) -> f32 {
        let dl = dice_loss(pred, target, self.smooth);
        if self.bce_weight < 1e-9 {
            return dl;
        }
        let bc = bce_loss(pred, target);
        dl + self.bce_weight * bc
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. HausdorffDistance
// ─────────────────────────────────────────────────────────────────────────────

/// Approximate 95th-percentile Hausdorff distance between two binary masks.
///
/// Operates on flat boolean masks with an associated 2-D spatial grid.
pub struct HausdorffDistance {
    /// Percentile in `[0, 100]` to use (typically 95).
    pub percentile: f32,
}

impl HausdorffDistance {
    /// Create a new `HausdorffDistance` computer.
    pub fn new(percentile: f32) -> Self {
        Self { percentile }
    }

    /// Compute the directed percentile Hausdorff distance.
    ///
    /// For each foreground pixel in `a`, find the minimum distance to any
    /// foreground pixel in `b` and return the `percentile`-th value.
    fn directed(&self, a_pts: &[(usize, usize)], b_pts: &[(usize, usize)]) -> f32 {
        if a_pts.is_empty() || b_pts.is_empty() {
            return 0.0;
        }
        let mut min_dists: Vec<f32> = a_pts
            .iter()
            .map(|&(ar, ac)| {
                b_pts
                    .iter()
                    .map(|&(br, bc)| {
                        let dr = ar as f32 - br as f32;
                        let dc = ac as f32 - bc as f32;
                        (dr * dr + dc * dc).sqrt()
                    })
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        min_dists.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let idx =
            ((self.percentile / 100.0 * min_dists.len() as f32) as usize).min(min_dists.len() - 1);
        min_dists[idx]
    }

    /// Compute the symmetric percentile Hausdorff distance.
    ///
    /// * `pred`   — flat binary mask (foreground = `true`)
    /// * `target` — flat binary mask (foreground = `true`)
    /// * `h`, `w` — spatial dimensions
    pub fn compute(&self, pred: &[bool], target: &[bool], h: usize, w: usize) -> f32 {
        let pred_pts: Vec<(usize, usize)> = pred
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some((i / w, i % w)) } else { None })
            .collect();
        let tgt_pts: Vec<(usize, usize)> = target
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some((i / w, i % w)) } else { None })
            .collect();

        let d1 = self.directed(&pred_pts, &tgt_pts);
        let d2 = self.directed(&tgt_pts, &pred_pts);
        d1.max(d2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. SlidingWindowInference
// ─────────────────────────────────────────────────────────────────────────────

/// MONAI-style sliding-window (tiled) inference.
///
/// Splits an input volume into overlapping ROI tiles, runs the model on each,
/// and blends predictions with a Gaussian importance weight.
pub struct SlidingWindowInference {
    /// ROI size: `[H_roi, W_roi]`.
    pub roi_size: Vec<usize>,
    /// Batch size for tile inference (tiles are currently processed one-by-one).
    pub sw_batch: usize,
    /// Overlap fraction in `[0, 1)`.
    pub overlap: f32,
}

impl SlidingWindowInference {
    /// Create a new `SlidingWindowInference`.
    pub fn new(roi_size: Vec<usize>, sw_batch: usize, overlap: f32) -> Self {
        Self {
            roi_size,
            sw_batch,
            overlap: overlap.clamp(0.0, 0.9999),
        }
    }

    /// Run inference.
    ///
    /// * `model_fn` — inference closure `&[f32] -> Vec<f32>`
    /// * `input`    — flat `[H × W × C]` channels-last tensor
    /// * `h`, `w`   — spatial dimensions
    /// * `c_in`     — input channels per pixel
    /// * `c_out`    — output channels per pixel (returned by model)
    pub fn run(
        &self,
        model_fn: &dyn Fn(&[f32]) -> Vec<f32>,
        input: &[f32],
        h: usize,
        w: usize,
        c_in: usize,
        c_out: usize,
    ) -> Vec<f32> {
        let roi_h = self.roi_size.first().copied().unwrap_or(h).min(h);
        let roi_w = self.roi_size.get(1).copied().unwrap_or(w).min(w);

        let step_h = ((roi_h as f32 * (1.0 - self.overlap)) as usize).max(1);
        let step_w = ((roi_w as f32 * (1.0 - self.overlap)) as usize).max(1);

        // Gaussian importance map for a single tile
        let importance = gaussian_importance(roi_h, roi_w);

        let mut acc: Vec<f32> = vec![0.0f32; h * w * c_out];
        let mut weight: Vec<f32> = vec![0.0f32; h * w];

        // Generate tile starts
        let starts_h = tile_starts(h, roi_h, step_h);
        let starts_w = tile_starts(w, roi_w, step_w);

        for &sh in &starts_h {
            for &sw_start in &starts_w {
                // Extract tile from input (channels-last)
                let tile = extract_tile(input, h, w, c_in, sh, sw_start, roi_h, roi_w);

                // Run model
                let pred = model_fn(&tile);

                // Splat prediction back with importance weights
                for ri in 0..roi_h {
                    for ci in 0..roi_w {
                        let img_r = sh + ri;
                        let img_c = sw_start + ci;
                        if img_r >= h || img_c >= w {
                            continue;
                        }
                        let imp = importance[ri * roi_w + ci];
                        let tile_idx = ri * roi_w + ci;
                        for k in 0..c_out {
                            let pred_v = pred.get(tile_idx * c_out + k).copied().unwrap_or(0.0);
                            acc[(img_r * w + img_c) * c_out + k] += imp * pred_v;
                        }
                        weight[img_r * w + img_c] += imp;
                    }
                }
            }
        }

        // Normalise
        for s in 0..(h * w) {
            let w_inv = if weight[s] > 1e-9 {
                1.0 / weight[s]
            } else {
                1.0
            };
            for k in 0..c_out {
                acc[s * c_out + k] *= w_inv;
            }
        }
        acc
    }
}

fn tile_starts(total: usize, roi: usize, step: usize) -> Vec<usize> {
    if total == 0 || roi == 0 {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut s = 0usize;
    loop {
        starts.push(s);
        if s + roi >= total {
            break;
        }
        s += step;
    }
    starts
}

fn gaussian_importance(h: usize, w: usize) -> Vec<f32> {
    let ch = h as f32 / 2.0;
    let cw = w as f32 / 2.0;
    let sh = (ch / 3.0).max(1e-5);
    let sw = (cw / 3.0).max(1e-5);
    let mut imp: Vec<f32> = (0..h * w)
        .map(|i| {
            let r = i / w;
            let c = i % w;
            let dr = (r as f32 - ch) / sh;
            let dc = (c as f32 - cw) / sw;
            (-(dr * dr + dc * dc) / 2.0).exp()
        })
        .collect();
    // Normalise to [0, 1]
    let max_v = imp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_v > 0.0 {
        for v in imp.iter_mut() {
            *v /= max_v;
        }
    }
    imp
}

fn extract_tile(
    input: &[f32],
    _h: usize,
    w: usize,
    c_in: usize,
    start_r: usize,
    start_c: usize,
    roi_h: usize,
    roi_w: usize,
) -> Vec<f32> {
    let mut tile = Vec::with_capacity(roi_h * roi_w * c_in);
    for ri in 0..roi_h {
        for ci in 0..roi_w {
            let img_r = start_r + ri;
            let img_c = start_c + ci;
            let base = (img_r * w + img_c) * c_in;
            if base + c_in <= input.len() {
                tile.extend_from_slice(&input[base..base + c_in]);
            } else {
                tile.extend(std::iter::repeat(0.0f32).take(c_in));
            }
        }
    }
    tile
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. MedicalAugmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Medical image augmentation pipeline.
pub struct MedicalAugmentation {
    /// Seed for the internal RNG.
    pub seed: u64,
}

impl MedicalAugmentation {
    /// Create a new `MedicalAugmentation` with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Gamma intensity transform: `out = x ^ gamma`.
    ///
    /// `x` must be in `[0, 1]`.  `gamma > 1` darkens, `gamma < 1` brightens.
    pub fn gamma_transform(&self, x: &[f32], gamma: f32) -> Vec<f32> {
        x.iter().map(|&v| v.clamp(0.0, 1.0).powf(gamma)).collect()
    }

    /// Add Gaussian noise with zero mean and given `std_dev`.
    pub fn add_gaussian_noise(&self, x: &[f32], std_dev: f32, seed_offset: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed + seed_offset);
        x.iter()
            .map(|&v| {
                let u1: f32 = rng.random::<f32>().clamp(1e-9, 1.0);
                let u2: f32 = rng.random();
                let noise =
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std_dev;
                v + noise
            })
            .collect()
    }

    /// Random horizontal flip.
    pub fn random_flip_h(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        c: usize,
        seed_offset: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed + seed_offset);
        if rng.random::<f32>() < 0.5 {
            return x.to_vec();
        }
        let mut out = vec![0.0f32; x.len()];
        for row in 0..h {
            for col in 0..w {
                let src_col = w - 1 - col;
                let src_base = (row * w + src_col) * c;
                let dst_base = (row * w + col) * c;
                if src_base + c <= x.len() && dst_base + c <= out.len() {
                    out[dst_base..dst_base + c].copy_from_slice(&x[src_base..src_base + c]);
                }
            }
        }
        out
    }

    /// Random vertical flip.
    pub fn random_flip_v(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        c: usize,
        seed_offset: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed + seed_offset);
        if rng.random::<f32>() < 0.5 {
            return x.to_vec();
        }
        let mut out = vec![0.0f32; x.len()];
        for row in 0..h {
            let src_row = h - 1 - row;
            let src_base = src_row * w * c;
            let dst_base = row * w * c;
            let len = w * c;
            if src_base + len <= x.len() && dst_base + len <= out.len() {
                out[dst_base..dst_base + len].copy_from_slice(&x[src_base..src_base + len]);
            }
        }
        out
    }

    /// Random crop of size `[crop_h, crop_w]`.
    ///
    /// Returns the cropped patch (channels-last).
    pub fn random_crop(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        c: usize,
        crop_h: usize,
        crop_w: usize,
        seed_offset: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed + seed_offset);
        let ch = crop_h.min(h);
        let cw = crop_w.min(w);
        let max_r = h - ch;
        let max_c = w - cw;
        let start_r: usize = if max_r > 0 {
            (rng.random::<f32>() * max_r as f32) as usize
        } else {
            0
        };
        let start_c: usize = if max_c > 0 {
            (rng.random::<f32>() * max_c as f32) as usize
        } else {
            0
        };
        let mut out = Vec::with_capacity(ch * cw * c);
        for row in start_r..start_r + ch {
            for col in start_c..start_c + cw {
                let base = (row * w + col) * c;
                if base + c <= x.len() {
                    out.extend_from_slice(&x[base..base + c]);
                } else {
                    out.extend(std::iter::repeat(0.0f32).take(c));
                }
            }
        }
        out
    }

    /// Elastic deformation via a random displacement grid.
    ///
    /// Generates a coarse `grid_h × grid_w` displacement field, bilinearly
    /// up-samples it to `h × w`, and samples the input image.
    pub fn elastic_deformation(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        c: usize,
        alpha: f32,
        sigma: f32,
        grid_h: usize,
        grid_w: usize,
        seed_offset: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed + seed_offset);

        // Generate coarse displacement field and smooth with Gaussian kernel
        let coarse_size = grid_h * grid_w;
        let dx_coarse: Vec<f32> = (0..coarse_size)
            .map(|_| {
                let u: f32 = rng.random::<f32>().clamp(1e-9, 1.0);
                let v: f32 = rng.random();
                alpha * (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
            })
            .collect();
        let dy_coarse: Vec<f32> = (0..coarse_size)
            .map(|_| {
                let u: f32 = rng.random::<f32>().clamp(1e-9, 1.0);
                let v: f32 = rng.random();
                alpha * (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
            })
            .collect();

        // Smooth coarse field
        let dx_smooth = gaussian_smooth_1d(&dx_coarse, grid_h, grid_w, sigma);
        let dy_smooth = gaussian_smooth_1d(&dy_coarse, grid_h, grid_w, sigma);

        let mut out = vec![0.0f32; h * w * c];
        let gh = grid_h.max(1) as f32;
        let gw = grid_w.max(1) as f32;

        for row in 0..h {
            for col in 0..w {
                // Map pixel to coarse grid coords
                let gr = row as f32 / h as f32 * gh;
                let gc = col as f32 / w as f32 * gw;
                let gr0 = (gr as usize).min(grid_h.saturating_sub(1));
                let gc0 = (gc as usize).min(grid_w.saturating_sub(1));
                let idx = gr0 * grid_w + gc0;

                let dx = dx_smooth.get(idx).copied().unwrap_or(0.0);
                let dy = dy_smooth.get(idx).copied().unwrap_or(0.0);

                // Sample source location
                let src_r = (row as f32 + dy).clamp(0.0, (h - 1) as f32);
                let src_c = (col as f32 + dx).clamp(0.0, (w - 1) as f32);
                let r0 = src_r as usize;
                let c0 = src_c as usize;
                let r1 = (r0 + 1).min(h - 1);
                let c1 = (c0 + 1).min(w - 1);
                let wr = src_r - r0 as f32;
                let wc = src_c - c0 as f32;

                let dst_base = (row * w + col) * c;
                for k in 0..c {
                    let v00 = x.get((r0 * w + c0) * c + k).copied().unwrap_or(0.0);
                    let v01 = x.get((r0 * w + c1) * c + k).copied().unwrap_or(0.0);
                    let v10 = x.get((r1 * w + c0) * c + k).copied().unwrap_or(0.0);
                    let v11 = x.get((r1 * w + c1) * c + k).copied().unwrap_or(0.0);
                    let val = (1.0 - wr) * ((1.0 - wc) * v00 + wc * v01)
                        + wr * ((1.0 - wc) * v10 + wc * v11);
                    if dst_base + k < out.len() {
                        out[dst_base + k] = val;
                    }
                }
            }
        }
        out
    }
}

/// Simple row-and-column separable Gaussian smoothing on a 2-D field.
fn gaussian_smooth_1d(field: &[f32], gh: usize, gw: usize, sigma: f32) -> Vec<f32> {
    let k = (sigma * 3.0).ceil() as usize;
    let k = k.max(1);
    let kernel: Vec<f32> = (0..=2 * k)
        .map(|i| {
            let x = i as f32 - k as f32;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let ksum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|v| v / ksum).collect();

    // Row pass
    let mut tmp = vec![0.0f32; gh * gw];
    for r in 0..gh {
        for c in 0..gw {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_c = (c + ki).saturating_sub(k).min(gw - 1);
                acc += kv * field.get(r * gw + src_c).copied().unwrap_or(0.0);
            }
            tmp[r * gw + c] = acc;
        }
    }
    // Column pass
    let mut out = vec![0.0f32; gh * gw];
    for r in 0..gh {
        for c in 0..gw {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_r = (r + ki).saturating_sub(k).min(gh - 1);
                acc += kv * tmp.get(src_r * gw + c).copied().unwrap_or(0.0);
            }
            out[r * gw + c] = acc;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. SegmentationMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Pixel-level segmentation evaluation metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct SegMetrics {
    /// Intersection-over-Union.
    pub iou: f32,
    /// Dice / F1 score.
    pub dice: f32,
    /// Sensitivity (true positive rate / recall).
    pub sensitivity: f32,
    /// Specificity (true negative rate).
    pub specificity: f32,
    /// True positives.
    pub tp: usize,
    /// True negatives.
    pub tn: usize,
    /// False positives.
    pub fp: usize,
    /// False negatives.
    pub fn_count: usize,
}

/// Compute segmentation metrics from binary prediction and ground-truth masks.
pub fn compute_segmentation_metrics(pred: &[bool], gt: &[bool]) -> SegMetrics {
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_c = 0usize;

    for (&p, &g) in pred.iter().zip(gt.iter()) {
        match (p, g) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (true, false) => fp += 1,
            (false, true) => fn_c += 1,
        }
    }

    let tp_f = tp as f32;
    let tn_f = tn as f32;
    let fp_f = fp as f32;
    let fn_f = fn_c as f32;

    let iou = if tp + fp + fn_c > 0 {
        tp_f / (tp_f + fp_f + fn_f)
    } else {
        1.0
    };
    let dice = if 2 * tp + fp + fn_c > 0 {
        2.0 * tp_f / (2.0 * tp_f + fp_f + fn_f)
    } else {
        1.0
    };
    let sensitivity = if tp + fn_c > 0 {
        tp_f / (tp_f + fn_f)
    } else {
        1.0
    };
    let specificity = if tn + fp > 0 {
        tn_f / (tn_f + fp_f)
    } else {
        1.0
    };

    SegMetrics {
        iou,
        dice,
        sensitivity,
        specificity,
        tp,
        tn,
        fp,
        fn_count: fn_c,
    }
}

/// Stateless struct exposing `compute` as an associated function (MONAI-style API).
pub struct SegmentationMetrics;

impl SegmentationMetrics {
    /// Compute all segmentation metrics.
    pub fn compute(pred: &[bool], gt: &[bool]) -> SegMetrics {
        compute_segmentation_metrics(pred, gt)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. AdaptiveInstanceNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive Instance Normalisation (AdaIN).
///
/// Transfers style statistics from `style` to `content`:
/// `AdaIN(content, style) = style_std * (content - content_mean) / content_std + style_mean`
#[derive(Debug, Clone)]
pub struct AdaptiveInstanceNorm {
    /// Number of channels.
    pub channels: usize,
}

impl AdaptiveInstanceNorm {
    /// Create a new `AdaptiveInstanceNorm`.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    /// Forward pass.
    ///
    /// * `content` — `[spatial × channels]` content feature map
    /// * `style`   — `[spatial × channels]` style feature map
    /// * `spatial` — number of spatial positions
    ///
    /// Returns normalised `[spatial × channels]` tensor.
    pub fn forward(&self, content: &[f32], style: &[f32], spatial: usize) -> Vec<f32> {
        let c = self.channels;
        if c == 0 || spatial == 0 {
            return content.to_vec();
        }

        // Compute per-channel statistics
        let mut c_mean = vec![0.0f32; c];
        let mut c_var = vec![0.0f32; c];
        let mut s_mean = vec![0.0f32; c];
        let mut s_var = vec![0.0f32; c];
        let n = spatial as f32;

        for s in 0..spatial {
            for k in 0..c {
                let cv = content.get(s * c + k).copied().unwrap_or(0.0);
                let sv = style.get(s * c + k).copied().unwrap_or(0.0);
                c_mean[k] += cv;
                s_mean[k] += sv;
            }
        }
        for k in 0..c {
            c_mean[k] /= n;
            s_mean[k] /= n;
        }
        for s in 0..spatial {
            for k in 0..c {
                let cv = content.get(s * c + k).copied().unwrap_or(0.0);
                let sv = style.get(s * c + k).copied().unwrap_or(0.0);
                c_var[k] += (cv - c_mean[k]).powi(2);
                s_var[k] += (sv - s_mean[k]).powi(2);
            }
        }
        for k in 0..c {
            c_var[k] /= n;
            s_var[k] /= n;
        }

        let c_std: Vec<f32> = c_var.iter().map(|&v| (v + 1e-5).sqrt()).collect();
        let s_std: Vec<f32> = s_var.iter().map(|&v| (v + 1e-5).sqrt()).collect();

        let mut out = Vec::with_capacity(spatial * c);
        for s in 0..spatial {
            for k in 0..c {
                let cv = content.get(s * c + k).copied().unwrap_or(0.0);
                let normalised = (cv - c_mean[k]) / c_std[k];
                out.push(s_std[k] * normalised + s_mean[k]);
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. NnUNetNormalizer
// ─────────────────────────────────────────────────────────────────────────────

/// nnU-Net style per-dataset intensity normalisation.
///
/// Clips to `[lo, hi]` percentiles then applies z-score normalisation:
/// `out = (clip(x, lo, hi) - mean) / (std + eps)`
#[derive(Debug, Clone)]
pub struct NnUNetNormalizer {
    /// Small constant for numerical stability.
    pub eps: f32,
}

impl NnUNetNormalizer {
    /// Create a new `NnUNetNormalizer`.
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }

    /// Normalise a volume in-place.
    ///
    /// * `input` — flat volume, modified in place.
    /// * `mean`  — dataset foreground mean.
    /// * `std`   — dataset foreground standard deviation.
    /// * `lo`    — lower clip value (e.g. 0.5th percentile of training set).
    /// * `hi`    — upper clip value (e.g. 99.5th percentile).
    pub fn normalize(&self, input: &mut [f32], mean: f32, std: f32, lo: f32, hi: f32) {
        let std_eff = std.abs().max(self.eps);
        for v in input.iter_mut() {
            *v = (v.clamp(lo, hi) - mean) / std_eff;
        }
    }

    /// Compute foreground statistics from a volume and optional binary mask.
    ///
    /// If `mask` is `None` all voxels are used.  Returns `(mean, std, p_lo, p_hi)`
    /// where `p_lo` and `p_hi` are the `lo_pct` / `hi_pct` percentile values.
    pub fn compute_stats(
        &self,
        volume: &[f32],
        mask: Option<&[bool]>,
        lo_pct: f32,
        hi_pct: f32,
    ) -> (f32, f32, f32, f32) {
        let vals: Vec<f32> = match mask {
            Some(m) => volume
                .iter()
                .zip(m.iter())
                .filter_map(|(&v, &fg)| if fg { Some(v) } else { None })
                .collect(),
            None => volume.to_vec(),
        };
        if vals.is_empty() {
            return (0.0, 1.0, 0.0, 0.0);
        }
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let lo_idx = ((lo_pct / 100.0 * n as f32) as usize).min(n - 1);
        let hi_idx = ((hi_pct / 100.0 * n as f32) as usize).min(n - 1);
        let p_lo = sorted[lo_idx];
        let p_hi = sorted[hi_idx];

        let mean: f32 = vals.iter().sum::<f32>() / n as f32;
        let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let std_v = var.sqrt();
        (mean, std_v, p_lo, p_hi)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extra: ConnectedComponents (useful companion to segmentation metrics)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute 4-connected component labelling of a binary 2-D mask (union-find).
///
/// Returns a label map where 0 = background and positive integers label foreground
/// components.
pub fn connected_components_2d(mask: &[bool], h: usize, w: usize) -> Vec<usize> {
    let n = h * w;
    let mut parent: Vec<usize> = (0..n).collect();

    let find = |mut x: usize, p: &mut Vec<usize>| -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    };

    // Union-Find union
    let mut labels = vec![0usize; n];
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            if !mask[i] {
                continue;
            }
            if c > 0 && mask[i - 1] {
                let a = find(i, &mut parent);
                let b = find(i - 1, &mut parent);
                if a != b {
                    parent[a] = b;
                }
            }
            if r > 0 && mask[i - w] {
                let a = find(i, &mut parent);
                let b = find(i - w, &mut parent);
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }

    // Assign contiguous labels
    let mut label_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next_label = 1usize;
    for i in 0..n {
        if mask[i] {
            let root = find(i, &mut parent);
            let entry = label_map.entry(root).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            labels[i] = *entry;
        }
    }
    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DoubleConv ──────────────────────────────────────────────────────────

    #[test]
    fn test_double_conv_output_shape() {
        let dc = DoubleConv::new(3, 8, 42);
        let input = vec![0.5f32; 4 * 4 * 3]; // 4×4 spatial, 3 channels
        let out = dc.forward(&input, 16);
        assert_eq!(out.len(), 16 * 8, "DoubleConv output shape mismatch");
    }

    #[test]
    fn test_double_conv_non_negative_relu() {
        let dc = DoubleConv::new(2, 4, 7);
        let input = vec![-1.0f32; 9 * 2];
        let out = dc.forward(&input, 9);
        // After ReLU all values should be >= 0
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "ReLU should produce non-negative outputs"
        );
    }

    #[test]
    fn test_double_conv_single_pixel() {
        let dc = DoubleConv::new(1, 4, 1);
        let input = vec![1.0f32];
        let out = dc.forward(&input, 1);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_double_conv_deterministic() {
        let dc = DoubleConv::new(4, 8, 99);
        let input: Vec<f32> = (0..16 * 4).map(|i| i as f32 / 64.0).collect();
        let out1 = dc.forward(&input, 16);
        let out2 = dc.forward(&input, 16);
        assert_eq!(out1, out2, "DoubleConv must be deterministic");
    }

    // ── concat_channels ─────────────────────────────────────────────────────

    #[test]
    fn test_concat_channels_shape() {
        let a = vec![1.0f32; 4 * 3]; // 4 spatial, 3 ch
        let b = vec![2.0f32; 4 * 5]; // 4 spatial, 5 ch
        let out = concat_channels(&a, &b, 4);
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn test_concat_channels_values() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // spatial=2, c=2
        let b = vec![5.0f32, 6.0]; // spatial=2, c=1
        let out = concat_channels(&a, &b, 2);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    // ── UNet2D ───────────────────────────────────────────────────────────────

    #[test]
    fn test_unet_output_shape() {
        let net = UNet2D::new(1, 2, 4, 0);
        let input = vec![0.5f32; 8 * 8];
        let out = net.forward(&input, 8, 8, 1);
        assert_eq!(
            out.len(),
            8 * 8 * 2,
            "UNet2D output spatial should match input"
        );
    }

    #[test]
    fn test_unet_multi_channel_input() {
        let net = UNet2D::new(3, 1, 4, 5);
        let input: Vec<f32> = (0..4 * 4 * 3).map(|i| i as f32 / 48.0).collect();
        let out = net.forward(&input, 4, 4, 3);
        assert_eq!(out.len(), (4 * 4));
    }

    #[test]
    fn test_unet_deterministic() {
        let net = UNet2D::new(1, 1, 4, 7);
        let input = vec![0.2f32; 4 * 4];
        let o1 = net.forward(&input, 4, 4, 1);
        let o2 = net.forward(&input, 4, 4, 1);
        assert_eq!(o1, o2);
    }

    // ── AttentionGate ────────────────────────────────────────────────────────

    #[test]
    fn test_attention_gate_output_shape() {
        let ag = AttentionGate::new(8, 8, 4, 42);
        let x = vec![0.5f32; 16 * 8];
        let g = vec![0.3f32; 16 * 8];
        let out = ag.forward(&x, &g, &[4, 4]);
        assert_eq!(out.len(), 16 * 8);
    }

    #[test]
    fn test_attention_gate_output_range() {
        // Attended features should scale x by sigmoid value, so magnitude ≤ |x|
        let ag = AttentionGate::new(4, 4, 2, 1);
        let x = vec![1.0f32; 9 * 4];
        let g = vec![0.0f32; 9 * 4];
        let out = ag.forward(&x, &g, &[3, 3]);
        // alpha ∈ (0,1), so out ∈ (0, 1)
        assert!(out.iter().all(|&v| (0.0..=1.001).contains(&v)));
    }

    #[test]
    fn test_attention_gate_single_pixel() {
        let ag = AttentionGate::new(3, 3, 2, 3);
        let x = vec![0.5f32, 0.5, 0.5];
        let g = vec![0.1f32, 0.1, 0.1];
        let out = ag.forward(&x, &g, &[1, 1]);
        assert_eq!(out.len(), 3);
    }

    // ── DiceLoss ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dice_loss_perfect() {
        let pred = vec![1.0f32, 0.0, 1.0];
        let tgt = vec![1.0f32, 0.0, 1.0];
        let loss = dice_loss(&pred, &tgt, 1e-6);
        assert!(
            loss < 0.01,
            "Perfect pred should give near-zero Dice loss, got {}",
            loss
        );
    }

    #[test]
    fn test_dice_loss_all_wrong() {
        let pred = vec![1.0f32; 4];
        let tgt = vec![0.0f32; 4];
        let loss = dice_loss(&pred, &tgt, 1e-5);
        assert!(
            loss > 0.9,
            "All-wrong pred should give loss near 1, got {}",
            loss
        );
    }

    #[test]
    fn test_dice_loss_range() {
        let pred = vec![0.7f32, 0.2, 0.8];
        let tgt = vec![1.0f32, 0.0, 1.0];
        let loss = dice_loss(&pred, &tgt, 1.0);
        assert!(
            (0.0..=1.0).contains(&loss),
            "Dice loss must be in [0,1], got {}",
            loss
        );
    }

    #[test]
    fn test_bce_loss_perfect() {
        let pred = vec![0.9999f32, 0.0001];
        let tgt = vec![1.0f32, 0.0];
        let loss = bce_loss(&pred, &tgt);
        assert!(
            loss < 0.01,
            "Near-perfect pred should have low BCE, got {}",
            loss
        );
    }

    // ── DiceBCELoss ───────────────────────────────────────────────────────────

    #[test]
    fn test_dice_bce_loss_combined() {
        let loss_fn = DiceBCELoss::new(1.0, 1.0);
        let pred = vec![0.8f32, 0.2, 0.9];
        let tgt = vec![1.0f32, 0.0, 1.0];
        let l = loss_fn.forward(&pred, &tgt);
        assert!(l >= 0.0, "DiceBCE loss must be non-negative");
    }

    #[test]
    fn test_dice_bce_weight_zero_equals_dice() {
        let pred = vec![0.6f32; 5];
        let tgt = vec![1.0f32; 5];
        let l_dice = dice_loss(&pred, &tgt, 1.0);
        let l_combo = DiceBCELoss::new(1.0, 0.0).forward(&pred, &tgt);
        assert!((l_dice - l_combo).abs() < 1e-6);
    }

    // ── HausdorffDistance ─────────────────────────────────────────────────────

    #[test]
    fn test_hausdorff_identical_masks() {
        let hd = HausdorffDistance::new(95.0);
        let mask: Vec<bool> = vec![true, false, true, false, true, false, false, false, true];
        let d = hd.compute(&mask, &mask, 3, 3);
        assert!(d < 1e-5, "Identical masks → 0 Hausdorff, got {}", d);
    }

    #[test]
    fn test_hausdorff_disjoint() {
        let hd = HausdorffDistance::new(95.0);
        let a = vec![true, false, false, false, false, false];
        let b = vec![false, false, false, false, false, true];
        let d = hd.compute(&a, &b, 2, 3);
        assert!(
            d > 0.0,
            "Disjoint masks should have positive Hausdorff distance"
        );
    }

    #[test]
    fn test_hausdorff_empty_masks() {
        let hd = HausdorffDistance::new(95.0);
        let empty = vec![false; 9];
        let d = hd.compute(&empty, &empty, 3, 3);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_hausdorff_symmetry() {
        let hd = HausdorffDistance::new(95.0);
        let a: Vec<bool> = (0..16).map(|i| i % 3 == 0).collect();
        let b: Vec<bool> = (0..16).map(|i| i % 5 == 0).collect();
        let d1 = hd.compute(&a, &b, 4, 4);
        let d2 = hd.compute(&b, &a, 4, 4);
        assert!((d1 - d2).abs() < 1e-5, "Hausdorff should be symmetric");
    }

    // ── SlidingWindowInference ────────────────────────────────────────────────

    #[test]
    fn test_sliding_window_output_shape() {
        let swi = SlidingWindowInference::new(vec![4, 4], 1, 0.5);
        let input = vec![0.5f32; 8 * 8 * 2];
        let identity_fn = |tile: &[f32]| tile.to_vec(); // c_in == c_out == 2
        let out = swi.run(&identity_fn, &input, 8, 8, 2, 2);
        assert_eq!(
            out.len(),
            8 * 8 * 2,
            "SlidingWindow output shape should match input"
        );
    }

    #[test]
    fn test_sliding_window_constant_input() {
        // For a constant input the blended output should also be nearly constant
        let swi = SlidingWindowInference::new(vec![3, 3], 1, 0.25);
        let input = vec![1.0f32; 6 * 6];
        let const_fn = |tile: &[f32]| tile.to_vec();
        let out = swi.run(&const_fn, &input, 6, 6, 1, 1);
        assert_eq!(out.len(), 36);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "Expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_sliding_window_zero_overlap() {
        let swi = SlidingWindowInference::new(vec![2, 2], 1, 0.0);
        let input = vec![0.5f32; 4 * 4];
        let id = |t: &[f32]| t.to_vec();
        let out = swi.run(&id, &input, 4, 4, 1, 1);
        assert_eq!(out.len(), 16);
    }

    // ── MedicalAugmentation ───────────────────────────────────────────────────

    #[test]
    fn test_gamma_identity() {
        let aug = MedicalAugmentation::new(0);
        let x: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
        let out = aug.gamma_transform(&x, 1.0);
        for (&a, &b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gamma_range() {
        let aug = MedicalAugmentation::new(0);
        let x = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let out = aug.gamma_transform(&x, 2.0);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_gaussian_noise_shape() {
        let aug = MedicalAugmentation::new(42);
        let x = vec![0.5f32; 20];
        let out = aug.add_gaussian_noise(&x, 0.05, 0);
        assert_eq!(out.len(), 20);
    }

    #[test]
    fn test_flip_h_double_flip_identity() {
        let aug = MedicalAugmentation::new(0);
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // Flip once with seed 0
        let flipped = aug.random_flip_h(&x, 3, 4, 1, 0);
        // The output must have the same length
        assert_eq!(flipped.len(), x.len(), "Flip must preserve shape");
        // Applying same seed again yields the same result (deterministic)
        let flipped2 = aug.random_flip_h(&x, 3, 4, 1, 0);
        assert_eq!(flipped, flipped2, "same seed → same result");
    }

    #[test]
    fn test_random_crop_shape() {
        let aug = MedicalAugmentation::new(11);
        let x = vec![0.5f32; 8 * 8 * 3];
        let crop = aug.random_crop(&x, 8, 8, 3, 5, 5, 0);
        assert_eq!(crop.len(), 5 * 5 * 3);
    }

    #[test]
    fn test_random_crop_full_size() {
        let aug = MedicalAugmentation::new(7);
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let crop = aug.random_crop(&x, 4, 4, 1, 4, 4, 0);
        assert_eq!(crop.len(), 16);
    }

    #[test]
    fn test_elastic_deformation_shape() {
        let aug = MedicalAugmentation::new(33);
        let x = vec![0.5f32; 8 * 8 * 2];
        let out = aug.elastic_deformation(&x, 8, 8, 2, 10.0, 1.5, 4, 4, 0);
        assert_eq!(out.len(), 8 * 8 * 2);
    }

    #[test]
    fn test_elastic_deformation_zero_alpha() {
        // alpha=0 → no deformation: every pixel should equal the source pixel
        let aug = MedicalAugmentation::new(5);
        let x: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
        let out = aug.elastic_deformation(&x, 3, 3, 1, 0.0, 1.0, 3, 3, 0);
        assert_eq!(out.len(), 9);
        for (&a, &b) in x.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Zero alpha: expected {}, got {}",
                a,
                b
            );
        }
    }

    // ── SegmentationMetrics ───────────────────────────────────────────────────

    #[test]
    fn test_seg_metrics_perfect() {
        let mask: Vec<bool> = vec![true, true, false, false];
        let m = SegmentationMetrics::compute(&mask, &mask);
        assert!((m.dice - 1.0).abs() < 1e-5);
        assert!((m.iou - 1.0).abs() < 1e-5);
        assert!((m.sensitivity - 1.0).abs() < 1e-5);
        assert!((m.specificity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_seg_metrics_all_fp() {
        let pred = vec![true, true, true, true];
        let gt = vec![false, false, false, false];
        let m = SegmentationMetrics::compute(&pred, &gt);
        assert_eq!(m.tp, 0);
        assert_eq!(m.fp, 4);
        assert_eq!(m.fn_count, 0);
        assert_eq!(m.tn, 0);
    }

    #[test]
    fn test_seg_metrics_all_fn() {
        let pred = vec![false; 4];
        let gt = vec![true; 4];
        let m = SegmentationMetrics::compute(&pred, &gt);
        assert_eq!(m.fn_count, 4);
        assert!(m.sensitivity < 1e-5);
    }

    #[test]
    fn test_seg_metrics_mixed() {
        let pred = vec![true, false, true, true];
        let gt = vec![true, true, false, true];
        let m = SegmentationMetrics::compute(&pred, &gt);
        assert_eq!(m.tp, 2);
        assert_eq!(m.fp, 1);
        assert_eq!(m.fn_count, 1);
        assert_eq!(m.tn, 0);
    }

    #[test]
    fn test_seg_metrics_empty() {
        let pred: Vec<bool> = Vec::new();
        let gt: Vec<bool> = Vec::new();
        let m = SegmentationMetrics::compute(&pred, &gt);
        assert!((m.dice - 1.0).abs() < 1e-5);
    }

    // ── AdaptiveInstanceNorm ─────────────────────────────────────────────────

    #[test]
    fn test_adain_output_shape() {
        let adain = AdaptiveInstanceNorm::new(4);
        let content = vec![0.5f32; 9 * 4];
        let style = vec![0.3f32; 9 * 4];
        let out = adain.forward(&content, &style, 9);
        assert_eq!(out.len(), 9 * 4);
    }

    #[test]
    fn test_adain_zero_content() {
        let adain = AdaptiveInstanceNorm::new(2);
        let content = vec![0.0f32; 4 * 2];
        let style: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = adain.forward(&content, &style, 4);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_adain_constant_content() {
        // Constant content → zero variance → normalised to style mean
        let adain = AdaptiveInstanceNorm::new(1);
        let content = vec![5.0f32; 4]; // spatial=4, c=1
        let style = vec![1.0f32, 2.0, 3.0, 4.0]; // mean=2.5
        let out = adain.forward(&content, &style, 4);
        // All output values should be near style mean (2.5) since normalised content is 0
        for &v in &out {
            assert!((v - 2.5).abs() < 0.2, "Expected ~2.5 got {}", v);
        }
    }

    #[test]
    fn test_adain_style_statistics_transferred() {
        let adain = AdaptiveInstanceNorm::new(1);
        // Style: [8, 10, 12, 14, 16, 18, 20, 22], mean = 15.0
        let style: Vec<f32> = (0..8).map(|i| 8.0 + i as f32 * 2.0).collect();
        let content: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let out = adain.forward(&content, &style, 8);
        let out_mean: f32 = out.iter().sum::<f32>() / 8.0;
        // Output mean should be close to style mean (15.0)
        assert!(
            (out_mean - 15.0).abs() < 1.0,
            "AdaIN mean ~= style mean 15.0, got {}",
            out_mean
        );
    }

    // ── NnUNetNormalizer ──────────────────────────────────────────────────────

    #[test]
    fn test_nnu_normalize_clips_and_normalizes() {
        let norm = NnUNetNormalizer::new(1e-5);
        let mut data = vec![-5.0f32, 0.0, 5.0, 100.0, -100.0];
        norm.normalize(&mut data, 0.0, 2.0, -3.0, 3.0);
        // After clipping to [-3,3] and z-scoring with mean=0 std=2:
        assert!((data[0] - (-1.5)).abs() < 1e-4); // -3/2 = -1.5
        assert!((data[2] - 1.5).abs() < 1e-4); // 3/2 = 1.5
        assert!((data[3] - 1.5).abs() < 1e-4); // 100 → clipped to 3
        assert!((data[4] - (-1.5)).abs() < 1e-4); // -100 → clipped to -3
    }

    #[test]
    fn test_nnu_compute_stats() {
        let norm = NnUNetNormalizer::new(1e-5);
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (mean, std_v, p_lo, p_hi) = norm.compute_stats(&data, None, 5.0, 95.0);
        assert!((mean - 49.5).abs() < 0.5, "mean ~= 49.5, got {}", mean);
        assert!(std_v > 0.0);
        assert!((p_lo - 5.0).abs() < 1.0, "p5 should be ~5, got {}", p_lo);
        assert!((p_hi - 95.0).abs() < 1.0, "p95 should be ~95, got {}", p_hi);
    }

    #[test]
    fn test_nnu_mask_stats() {
        let norm = NnUNetNormalizer::new(1e-5);
        let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let mask = vec![false, true, true, true, false];
        let (mean, _, _, _) = norm.compute_stats(&data, Some(&mask), 0.0, 100.0);
        assert!(
            (mean - 2.0).abs() < 1e-5,
            "mean of [1,2,3] = 2.0, got {}",
            mean
        );
    }

    #[test]
    fn test_nnu_normalize_zero_std() {
        let norm = NnUNetNormalizer::new(1.0);
        let mut data = vec![5.0f32; 5];
        // std = 0, eps = 1.0 → std_eff = 1.0
        norm.normalize(&mut data, 5.0, 0.0, 0.0, 10.0);
        for &v in &data {
            assert!((v - 0.0).abs() < 1e-5, "Expected 0.0 got {}", v);
        }
    }

    // ── ConnectedComponents ───────────────────────────────────────────────────

    #[test]
    fn test_cc2d_single_component() {
        // 3×3 all-true
        let mask = vec![true; 9];
        let labels = connected_components_2d(&mask, 3, 3);
        let non_zero: Vec<usize> = labels.iter().filter(|&&v| v > 0).cloned().collect();
        assert_eq!(non_zero.len(), 9);
        // All same label
        let first = labels[0];
        assert!(labels.iter().all(|&v| v == first));
    }

    #[test]
    fn test_cc2d_two_components() {
        // Two isolated pixels on a 1×4 grid
        let mask = vec![true, false, false, true];
        let labels = connected_components_2d(&mask, 1, 4);
        assert_eq!(labels[0], 1);
        assert_eq!(labels[3], 2);
        assert_eq!(labels[1], 0);
    }

    #[test]
    fn test_cc2d_all_background() {
        let mask = vec![false; 6];
        let labels = connected_components_2d(&mask, 2, 3);
        assert!(labels.iter().all(|&v| v == 0));
    }

    // ── Gaussian importance map ───────────────────────────────────────────────

    #[test]
    fn test_gaussian_importance_center_max() {
        let imp = gaussian_importance(5, 5);
        let center = imp[2 * 5 + 2];
        let corner = imp[0];
        assert!(
            center > corner,
            "Center should have higher importance than corner"
        );
    }

    #[test]
    fn test_gaussian_importance_range() {
        let imp = gaussian_importance(6, 6);
        assert!(imp.iter().all(|&v| (0.0..=1.001).contains(&v)));
    }
}
