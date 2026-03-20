//! Video Understanding — production-grade video neural networks.
//!
//! Implements a comprehensive set of modern video understanding architectures,
//! operating on flat `Vec<f32>` / `&[f32]` buffers.
//!
//! ## Components
//!
//! - **TemporalShiftModule** — TSM: shifts channels along temporal axis
//! - **VideoTransformerBlock** — factored space-time attention
//! - **ActionRecognitionHead** — temporal mean-pool + linear classifier
//! - **VideoMAE** — masked autoencoder with tube masking
//! - **OpticalFlowRAFT** — simplified RAFT optical flow with correlation volume + GRU refinement
//! - **DeformableConv2D** — learned offset bilinear sampling
//! - **VideoSwinBlock** — 3D window attention with shifted windows
//! - **HeatmapPoseHead** — 2D gaussian heatmap keypoint regression
//! - **TemporalActionDetector** — 1D proposal generation with NMS
//! - **VideoAugmentation** — temporal crop, frame-rate jitter, flip, reversal, cutout3D

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

pub(crate) fn softmax_f32(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s == 0.0 {
        return exps;
    }
    exps.iter().map(|&e| e / s).collect()
}

fn layer_norm_f32(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std = (var + 1e-5_f32).sqrt();
    x.iter().map(|&v| (v - mean) / std).collect()
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Box-Muller normal sample, mean 0 std σ.
fn rand_normal(rng: &mut StdRng, std_dev: f32) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-12);
    let u2: f32 = rng.random::<f32>();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos() * std_dev
}

fn rand_vec_f32(size: usize, std_dev: f32, rng: &mut StdRng) -> Vec<f32> {
    let mut out = Vec::with_capacity(size);
    for _ in 0..size {
        out.push(rand_normal(rng, std_dev));
    }
    out
}

/// Small linear projection: y = W x + b, W is [out x in] row-major.
fn linear(x: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; out_d];
    for o in 0..out_d {
        let row = &w[o * in_d..(o + 1) * in_d];
        y[o] = dot_f32(row, x) + b[o];
    }
    y
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. TemporalShiftModule (TSM)
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal Shift Module — shifts a fraction of channels along the temporal axis.
///
/// Given a clip tensor [T, H, W, C] stored in row-major (T outermost), shifts
/// the first `C / fold_div` channels one step forward in time (from t-1 to t)
/// and the next `C / fold_div` channels one step backward (from t+1 to t).
/// The remaining channels are unchanged.
pub struct TemporalShiftModule {
    /// Number of channels to fold (divisor).
    pub fold_div: usize,
}

impl TemporalShiftModule {
    pub fn new(fold_div: usize) -> Self {
        assert!(fold_div >= 1, "fold_div must be >= 1");
        Self { fold_div }
    }

    /// `x` layout: [T, H, W, C] flattened.  Returns same shape.
    pub fn shift(&self, x: &[f32], t: usize, h: usize, w: usize, c: usize) -> Vec<f32> {
        let spatial = h * w;
        let fold = c / self.fold_div;
        let mut out = x.to_vec();

        // Forward shift: channels [0, fold) at time t come from t-1
        for ti in 0..t {
            for si in 0..spatial {
                let base_dst = (ti * spatial + si) * c;
                // source is t-1 (or zero-pad for t=0)
                for ci in 0..fold {
                    if ti == 0 {
                        out[base_dst + ci] = 0.0;
                    } else {
                        let base_src = ((ti - 1) * spatial + si) * c;
                        out[base_dst + ci] = x[base_src + ci];
                    }
                }
            }
        }

        // Backward shift: channels [fold, 2*fold) at time t come from t+1
        // Source is read from channels [0, fold) of the next frame (ci - fold offset).
        // The upper bound is clamped to c to avoid out-of-bounds when fold_div == 1.
        let back_end = (2 * fold).min(c);
        for ti in 0..t {
            for si in 0..spatial {
                let base_dst = (ti * spatial + si) * c;
                for ci in fold..back_end {
                    if ti + 1 >= t {
                        out[base_dst + ci] = 0.0;
                    } else {
                        let base_src = ((ti + 1) * spatial + si) * c;
                        out[base_dst + ci] = x[base_src + (ci - fold)];
                    }
                }
            }
        }

        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. VideoTransformerBlock — factored space-time attention
// ─────────────────────────────────────────────────────────────────────────────

/// Factored space-then-time attention block.
///
/// Layout convention: `x` is [T*H*W, D] flattened (T*H*W tokens, D features).
/// Spatial attention first (groups tokens by time step), then temporal
/// attention (groups tokens by spatial position).
pub struct VideoTransformerBlock {
    pub d_model: usize,
    pub n_heads: usize,
    // Spatial attention params: [D x D] row-major for Q/K/V/O
    wq_s: Vec<f32>,
    wk_s: Vec<f32>,
    wv_s: Vec<f32>,
    wo_s: Vec<f32>,
    // Temporal attention params
    wq_t: Vec<f32>,
    wk_t: Vec<f32>,
    wv_t: Vec<f32>,
    wo_t: Vec<f32>,
    // FFN params
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl VideoTransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        let dd = d_model * d_model;
        let ffn_dim = d_model * 4;
        Self {
            d_model,
            n_heads,
            wq_s: rand_vec_f32(dd, std, &mut rng),
            wk_s: rand_vec_f32(dd, std, &mut rng),
            wv_s: rand_vec_f32(dd, std, &mut rng),
            wo_s: rand_vec_f32(dd, std, &mut rng),
            wq_t: rand_vec_f32(dd, std, &mut rng),
            wk_t: rand_vec_f32(dd, std, &mut rng),
            wv_t: rand_vec_f32(dd, std, &mut rng),
            wo_t: rand_vec_f32(dd, std, &mut rng),
            w1: rand_vec_f32(d_model * ffn_dim, std, &mut rng),
            b1: vec![0.0; ffn_dim],
            w2: rand_vec_f32(ffn_dim * d_model, std, &mut rng),
            b2: vec![0.0; d_model],
        }
    }

    /// Single-head scaled dot-product attention on a sequence of tokens `seq` [N, D].
    fn attend(
        tokens: &[Vec<f32>],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        d: usize,
        n_heads: usize,
    ) -> Vec<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        let head_dim = d / n_heads;
        let scale = (head_dim as f32).sqrt();

        // Project all tokens
        let qs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear(t, wq, &vec![0.0; d], d, d))
            .collect();
        let ks: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear(t, wk, &vec![0.0; d], d, d))
            .collect();
        let vs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear(t, wv, &vec![0.0; d], d, d))
            .collect();

        let mut out_tokens: Vec<Vec<f32>> = vec![vec![0.0; d]; n];

        for h in 0..n_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            // Compute attention matrix [n x n] for this head
            let mut attn = vec![vec![0.0_f32; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let score = dot_f32(&qs[i][start..end], &ks[j][start..end]) / scale;
                    attn[i][j] = score;
                }
                let row_sm = softmax_f32(&attn[i].clone());
                attn[i] = row_sm;
            }
            // Weighted sum of V
            for i in 0..n {
                for j in 0..n {
                    for k in start..end {
                        out_tokens[i][k] += attn[i][j] * vs[j][k];
                    }
                }
            }
        }

        // Output projection
        out_tokens
            .iter()
            .map(|t| {
                let bias = vec![0.0_f32; d];
                linear(t, wo, &bias, d, d)
            })
            .collect()
    }

    /// Forward: x layout [T*H*W, D].  Returns same shape.
    pub fn forward(&self, x: &[f32], t: usize, h: usize, w: usize, d: usize) -> Vec<f32> {
        let spatial = h * w;
        let total = t * spatial;
        assert_eq!(x.len(), total * d);

        // Unpack into Vec<Vec<f32>>
        let tokens: Vec<Vec<f32>> = (0..total).map(|i| x[i * d..(i + 1) * d].to_vec()).collect();

        // ── Spatial attention ── group by time step
        let mut after_spatial = tokens.clone();
        for ti in 0..t {
            let slice: Vec<Vec<f32>> = (0..spatial)
                .map(|si| tokens[ti * spatial + si].clone())
                .collect();
            let attended = Self::attend(
                &slice,
                &self.wq_s,
                &self.wk_s,
                &self.wv_s,
                &self.wo_s,
                d,
                self.n_heads,
            );
            for si in 0..spatial {
                // residual + layer norm over the full d-dimensional token
                let residual: Vec<f32> = (0..d)
                    .map(|di| tokens[ti * spatial + si][di] + attended[si][di])
                    .collect();
                let normed = layer_norm_f32(&residual);
                after_spatial[ti * spatial + si] = normed;
            }
        }

        // ── Temporal attention ── group by spatial position
        let mut after_temporal = after_spatial.clone();
        for si in 0..spatial {
            let slice: Vec<Vec<f32>> = (0..t)
                .map(|ti| after_spatial[ti * spatial + si].clone())
                .collect();
            let attended = Self::attend(
                &slice,
                &self.wq_t,
                &self.wk_t,
                &self.wv_t,
                &self.wo_t,
                d,
                self.n_heads,
            );
            for ti in 0..t {
                // residual + layer norm over the full d-dimensional token
                let residual: Vec<f32> = (0..d)
                    .map(|di| after_spatial[ti * spatial + si][di] + attended[ti][di])
                    .collect();
                let normed = layer_norm_f32(&residual);
                after_temporal[ti * spatial + si] = normed;
            }
        }

        // ── FFN ──
        let ffn_dim = d * 4;
        let mut result = vec![0.0_f32; total * d];
        for i in 0..total {
            let token = &after_temporal[i];
            let h1: Vec<f32> = (0..ffn_dim)
                .map(|oi| {
                    let row = &self.w1[oi * d..(oi + 1) * d];
                    relu(dot_f32(row, token) + self.b1[oi])
                })
                .collect();
            let h2: Vec<f32> = (0..d)
                .map(|oi| {
                    let row = &self.w2[oi * ffn_dim..(oi + 1) * ffn_dim];
                    dot_f32(row, &h1) + self.b2[oi]
                })
                .collect();
            let ln = layer_norm_f32(
                &h2.iter()
                    .enumerate()
                    .map(|(di, &v)| after_temporal[i][di] + v)
                    .collect::<Vec<_>>(),
            );
            for di in 0..d {
                result[i * d + di] = ln[di];
            }
        }

        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ActionRecognitionHead
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal mean-pool → linear classifier for action recognition.
pub struct ActionRecognitionHead {
    pub d_model: usize,
    pub n_classes: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl ActionRecognitionHead {
    pub fn new(d_model: usize, n_classes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / (d_model + n_classes) as f32).sqrt();
        Self {
            d_model,
            n_classes,
            w: rand_vec_f32(d_model * n_classes, std, &mut rng),
            b: vec![0.0; n_classes],
        }
    }

    /// `features` layout: [T, D] flattened.  Returns logits \[n_classes\].
    pub fn forward(&self, features: &[f32], t: usize, d: usize, n_classes: usize) -> Vec<f32> {
        assert_eq!(features.len(), t * d);
        // Temporal mean pool
        let mut pooled = vec![0.0_f32; d];
        for ti in 0..t {
            for di in 0..d {
                pooled[di] += features[ti * d + di];
            }
        }
        let t_f = t as f32;
        for v in pooled.iter_mut() {
            *v /= t_f;
        }
        // Linear
        linear(&pooled, &self.w, &self.b, d, n_classes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. VideoMAE
// ─────────────────────────────────────────────────────────────────────────────

/// Video Masked Autoencoder with tube masking (consistent spatial mask across time).
pub struct VideoMAE {
    pub patch_t: usize,
    pub patch_size: usize, // spatial patch size (square)
    pub mask_ratio: f32,
    pub encoder_dim: usize,
    pub decoder_dim: usize,
    // Encoder projection: patch_tokens → encoder_dim
    enc_w: Vec<f32>,
    enc_b: Vec<f32>,
    // Decoder projection: encoder_dim → patch_tokens (reconstruction)
    dec_w: Vec<f32>,
    dec_b: Vec<f32>,
}

impl VideoMAE {
    pub fn new(
        patch_t: usize,
        patch_size: usize,
        mask_ratio: f32,
        encoder_dim: usize,
        decoder_dim: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let patch_tokens = patch_t * patch_size * patch_size;
        let std_enc = (2.0_f32 / patch_tokens as f32).sqrt();
        let std_dec = (2.0_f32 / encoder_dim as f32).sqrt();
        Self {
            patch_t,
            patch_size,
            mask_ratio,
            encoder_dim,
            decoder_dim,
            enc_w: rand_vec_f32(patch_tokens * encoder_dim, std_enc, &mut rng),
            enc_b: vec![0.0; encoder_dim],
            dec_w: rand_vec_f32(encoder_dim * decoder_dim, std_dec, &mut rng),
            dec_b: vec![0.0; decoder_dim],
        }
    }

    /// Creates a tube mask: consistent spatial mask repeated across time.
    /// Returns a boolean mask of length `n_patches` (spatial tube level).
    /// `true` = masked (hidden).
    ///
    /// Total patches: (T/patch_t) * (H/patch_size) * (W/patch_size).
    /// The mask is at the tube granularity: one bool per spatial patch tube.
    pub fn create_tube_mask(
        t: usize,
        h: usize,
        w: usize,
        patch_t: usize,
        patch_size: usize,
        mask_ratio: f32,
        rng: &mut StdRng,
    ) -> Vec<bool> {
        let n_t = t / patch_t;
        let n_h = h / patch_size;
        let n_w = w / patch_size;
        let n_tubes = n_t * n_h * n_w;
        let n_mask = ((n_tubes as f32 * mask_ratio) as usize).min(n_tubes);

        // Fisher-Yates shuffle to select masked indices
        let mut indices: Vec<usize> = (0..n_tubes).collect();
        for i in (1..n_tubes).rev() {
            let j: usize = (rng.random::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
        let mut mask = vec![false; n_tubes];
        for &idx in &indices[..n_mask] {
            mask[idx] = true;
        }
        mask
    }

    /// Encode visible patches.  `visible_patches` is [n_visible, patch_tokens].
    /// Returns [n_visible, encoder_dim].
    pub fn forward_encoder(&self, visible_patches: &[f32], n_visible: usize) -> Vec<f32> {
        let patch_tokens = self.patch_t * self.patch_size * self.patch_size;
        assert_eq!(visible_patches.len(), n_visible * patch_tokens);
        let mut out = vec![0.0_f32; n_visible * self.encoder_dim];
        for i in 0..n_visible {
            let patch = &visible_patches[i * patch_tokens..(i + 1) * patch_tokens];
            let enc = linear(
                patch,
                &self.enc_w,
                &self.enc_b,
                patch_tokens,
                self.encoder_dim,
            );
            let normed = layer_norm_f32(&enc);
            for (di, &v) in normed.iter().enumerate() {
                out[i * self.encoder_dim + di] = v;
            }
        }
        out
    }

    /// Reconstruction loss (MSE) on masked patches only.
    /// `pred` and `target` are [n_patches, decoder_dim].
    /// `mask` has length n_patches (true = masked).
    pub fn reconstruction_loss(&self, pred: &[f32], target: &[f32], mask: &[bool]) -> f32 {
        assert_eq!(pred.len(), target.len());
        let n_patches = mask.len();
        let patch_dim = match pred.len().checked_div(n_patches) {
            Some(d) => d,
            None => return 0.0,
        };
        let mut loss = 0.0_f32;
        let mut count = 0usize;
        for (pi, &masked) in mask.iter().enumerate() {
            if masked {
                for di in 0..patch_dim {
                    let diff = pred[pi * patch_dim + di] - target[pi * patch_dim + di];
                    loss += diff * diff;
                }
                count += patch_dim;
            }
        }
        if count == 0 {
            0.0
        } else {
            loss / count as f32
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. OpticalFlowRAFT
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified RAFT optical flow estimator.
///
/// Uses all-pairs correlation limited to radius `r`, plus GRU-based iterative
/// refinement.
pub struct OpticalFlowRAFT {
    pub radius: usize,
    pub n_iters: usize,
    pub hidden_dim: usize,
    // GRU parameters (hidden_dim x corr_dim+hidden_dim+2 for reset/update/new gates)
    corr_channels: usize,
    // Gate weights: W_z, W_r, W_n each [hidden_dim x (hidden_dim + corr_dim + 2)]
    w_z: Vec<f32>,
    w_r: Vec<f32>,
    w_n: Vec<f32>,
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_n: Vec<f32>,
    // Flow head: [2 x hidden_dim]
    w_flow: Vec<f32>,
    b_flow: Vec<f32>,
}

impl OpticalFlowRAFT {
    pub fn new(
        radius: usize,
        n_iters: usize,
        hidden_dim: usize,
        feat_dim: usize,
        seed: u64,
    ) -> Self {
        let corr_channels = (2 * radius + 1) * (2 * radius + 1);
        let input_dim = hidden_dim + corr_channels + 2; // +2 for current flow
        let std = (2.0_f32 / input_dim as f32).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            radius,
            n_iters,
            hidden_dim,
            corr_channels,
            w_z: rand_vec_f32(hidden_dim * input_dim, std, &mut rng),
            w_r: rand_vec_f32(hidden_dim * input_dim, std, &mut rng),
            w_n: rand_vec_f32(hidden_dim * input_dim, std, &mut rng),
            b_z: vec![0.0; hidden_dim],
            b_r: vec![0.0; hidden_dim],
            b_n: vec![0.0; hidden_dim],
            w_flow: rand_vec_f32(2 * hidden_dim, std, &mut rng),
            b_flow: vec![0.0; 2],
        }
    }

    /// Compute dense correlation volume.
    ///
    /// For each pixel (i,j) in frame1, computes dot products with all pixels
    /// within radius r in frame2.  Returns [H, W, (2r+1)^2].
    pub fn compute_correlation(
        feat1: &[f32],
        feat2: &[f32],
        h: usize,
        w: usize,
        d: usize,
        radius: usize,
    ) -> Vec<f32> {
        assert_eq!(feat1.len(), h * w * d);
        assert_eq!(feat2.len(), h * w * d);
        let diam = 2 * radius + 1;
        let corr_dim = diam * diam;
        let mut corr = vec![0.0_f32; h * w * corr_dim];

        for i in 0..h {
            for j in 0..w {
                let f1 = &feat1[(i * w + j) * d..(i * w + j + 1) * d];
                let mut ci = 0usize;
                for di in 0..diam {
                    for dj in 0..diam {
                        let ni = i as isize + di as isize - radius as isize;
                        let nj = j as isize + dj as isize - radius as isize;
                        let val = if ni >= 0 && ni < h as isize && nj >= 0 && nj < w as isize {
                            let nni = ni as usize;
                            let nnj = nj as usize;
                            let f2 = &feat2[(nni * w + nnj) * d..(nni * w + nnj + 1) * d];
                            dot_f32(f1, f2) / (d as f32).sqrt()
                        } else {
                            0.0
                        };
                        corr[(i * w + j) * corr_dim + ci] = val;
                        ci += 1;
                    }
                }
            }
        }
        corr
    }

    /// Iterative GRU-based flow refinement.
    ///
    /// `correlation` is [H, W, corr_dim].
    /// `flow` is [H, W, 2] (updated in place).
    pub fn refine_flow(
        &self,
        correlation: &[f32],
        flow: &mut [f32],
        h: usize,
        w: usize,
        n_iters: usize,
    ) {
        let corr_dim = self.corr_channels;
        let n = h * w;
        assert_eq!(correlation.len(), n * corr_dim);
        assert_eq!(flow.len(), n * 2);

        let input_dim = self.hidden_dim + corr_dim + 2;
        let mut hidden = vec![0.0_f32; n * self.hidden_dim];

        for _ in 0..n_iters {
            for idx in 0..n {
                let h_prev = &hidden[idx * self.hidden_dim..(idx + 1) * self.hidden_dim];
                let corr_feat = &correlation[idx * corr_dim..(idx + 1) * corr_dim];
                let flow_xy = &flow[idx * 2..(idx + 1) * 2];

                // Concatenate [h_prev | corr | flow_xy]
                let mut inp = Vec::with_capacity(input_dim);
                inp.extend_from_slice(h_prev);
                inp.extend_from_slice(corr_feat);
                inp.extend_from_slice(flow_xy);

                // GRU gates
                let z_raw = linear(&inp, &self.w_z, &self.b_z, input_dim, self.hidden_dim);
                let r_raw = linear(&inp, &self.w_r, &self.b_r, input_dim, self.hidden_dim);
                let z: Vec<f32> = z_raw.iter().map(|&v| sigmoid(v)).collect();
                let r: Vec<f32> = r_raw.iter().map(|&v| sigmoid(v)).collect();

                // Candidate: uses r * h_prev
                let mut gated_inp = Vec::with_capacity(input_dim);
                for (ri, &rv) in r.iter().enumerate() {
                    gated_inp.push(rv * h_prev[ri]);
                }
                gated_inp.extend_from_slice(corr_feat);
                gated_inp.extend_from_slice(flow_xy);
                let n_raw = linear(&gated_inp, &self.w_n, &self.b_n, input_dim, self.hidden_dim);
                let n_cand: Vec<f32> = n_raw.iter().map(|&v| tanh_f32(v)).collect();

                // New hidden state
                let mut new_h = vec![0.0_f32; self.hidden_dim];
                for hi in 0..self.hidden_dim {
                    new_h[hi] = (1.0 - z[hi]) * h_prev[hi] + z[hi] * n_cand[hi];
                }

                // Flow delta
                let delta = linear(&new_h, &self.w_flow, &self.b_flow, self.hidden_dim, 2);
                flow[idx * 2] += delta[0];
                flow[idx * 2 + 1] += delta[1];

                for hi in 0..self.hidden_dim {
                    hidden[idx * self.hidden_dim + hi] = new_h[hi];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. DeformableConv2D
// ─────────────────────────────────────────────────────────────────────────────

/// Deformable convolution: learns per-pixel offsets for bilinear sampling.
///
/// For a 3×3 kernel: samples at 9 offset positions per output channel.
pub struct DeformableConv2D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_size: usize, // typically 3
    // Weight: [c_out, c_in, k, k]
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl DeformableConv2D {
    pub fn new(c_in: usize, c_out: usize, kernel_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let fan_in = c_in * kernel_size * kernel_size;
        let std = (2.0_f32 / fan_in as f32).sqrt();
        Self {
            c_in,
            c_out,
            kernel_size,
            weight: rand_vec_f32(c_out * c_in * kernel_size * kernel_size, std, &mut rng),
            bias: vec![0.0; c_out],
        }
    }

    /// Bilinear sampling from `x` [H, W, C_in] at sub-pixel coordinate (y, x).
    fn bilinear_sample(src: &[f32], h: usize, w: usize, c_in: usize, fy: f32, fx: f32) -> Vec<f32> {
        let y0 = fy.floor() as isize;
        let x0 = fx.floor() as isize;
        let y1 = y0 + 1;
        let x1 = x0 + 1;
        let dy = fy - fy.floor();
        let dx = fx - fx.floor();

        let clamp_y = |y: isize| y.max(0).min(h as isize - 1) as usize;
        let clamp_x = |x: isize| x.max(0).min(w as isize - 1) as usize;

        let y0c = clamp_y(y0);
        let x0c = clamp_x(x0);
        let y1c = clamp_y(y1);
        let x1c = clamp_x(x1);

        let w00 = (1.0 - dy) * (1.0 - dx);
        let w01 = (1.0 - dy) * dx;
        let w10 = dy * (1.0 - dx);
        let w11 = dy * dx;

        (0..c_in)
            .map(|ci| {
                w00 * src[(y0c * w + x0c) * c_in + ci]
                    + w01 * src[(y0c * w + x1c) * c_in + ci]
                    + w10 * src[(y1c * w + x0c) * c_in + ci]
                    + w11 * src[(y1c * w + x1c) * c_in + ci]
            })
            .collect()
    }

    /// Forward pass.
    ///
    /// `x` layout: [H, W, C_in].
    /// `offsets` layout: [H, W, 2 * kernel_size^2] — (dy, dx) pairs per kernel point.
    /// Returns [H, W, C_out].
    pub fn forward(&self, x: &[f32], h: usize, w: usize, c_in: usize, offsets: &[f32]) -> Vec<f32> {
        let k = self.kernel_size;
        let k2 = k * k;
        assert_eq!(x.len(), h * w * c_in);
        assert_eq!(offsets.len(), h * w * 2 * k2);

        let mut out = vec![0.0_f32; h * w * self.c_out];

        for hi in 0..h {
            for wi in 0..w {
                let pix_idx = hi * w + wi;
                let offset_base = pix_idx * 2 * k2;

                // Gather sampled features for all kernel points
                let mut sampled = vec![0.0_f32; c_in * k2]; // [k2, c_in]
                for ki in 0..k2 {
                    let kh = (ki / k) as isize - (k as isize / 2);
                    let kw = (ki % k) as isize - (k as isize / 2);
                    let dy = offsets[offset_base + 2 * ki];
                    let dx = offsets[offset_base + 2 * ki + 1];
                    let fy = hi as f32 + kh as f32 + dy;
                    let fx = wi as f32 + kw as f32 + dx;
                    let s = Self::bilinear_sample(x, h, w, c_in, fy, fx);
                    for ci in 0..c_in {
                        sampled[ki * c_in + ci] = s[ci];
                    }
                }

                // Apply weight [c_out, c_in, k2]
                for co in 0..self.c_out {
                    let mut val = self.bias[co];
                    for ki in 0..k2 {
                        for ci in 0..c_in {
                            let w_idx = co * (c_in * k2) + ci * k2 + ki;
                            val += self.weight[w_idx] * sampled[ki * c_in + ci];
                        }
                    }
                    out[pix_idx * self.c_out + co] = val;
                }
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. VideoSwinBlock — 3D window attention with shifted windows
// ─────────────────────────────────────────────────────────────────────────────

/// 3D Swin Transformer block: window-based self-attention with cyclic shift.
pub struct VideoSwinBlock {
    pub d_model: usize,
    pub n_heads: usize,
    pub shift: bool,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl VideoSwinBlock {
    pub fn new(d_model: usize, n_heads: usize, shift: bool, seed: u64) -> Self {
        assert!(d_model % n_heads == 0);
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        let dd = d_model * d_model;
        let ffn_dim = d_model * 4;
        Self {
            d_model,
            n_heads,
            shift,
            wq: rand_vec_f32(dd, std, &mut rng),
            wk: rand_vec_f32(dd, std, &mut rng),
            wv: rand_vec_f32(dd, std, &mut rng),
            wo: rand_vec_f32(dd, std, &mut rng),
            w1: rand_vec_f32(d_model * ffn_dim, std, &mut rng),
            b1: vec![0.0; ffn_dim],
            w2: rand_vec_f32(ffn_dim * d_model, std, &mut rng),
            b2: vec![0.0; d_model],
        }
    }

    /// Cyclic shift along one dimension by `shift_size`.
    fn cyclic_shift_1d(seq: &[f32], len: usize, shift_size: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq.len()];
        let stride = seq.len() / len;
        for i in 0..len {
            let src_i = (i + len - shift_size) % len;
            for k in 0..stride {
                out[i * stride + k] = seq[src_i * stride + k];
            }
        }
        out
    }

    /// Window-partition attention for a sequence of tokens within a window.
    fn window_attention(
        tokens: &[Vec<f32>],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        d: usize,
        n_heads: usize,
    ) -> Vec<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        let head_dim = d / n_heads;
        let scale = (head_dim as f32).sqrt();
        let zero_b = vec![0.0_f32; d];
        let qs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear(t, wq, &zero_b, d, d))
            .collect();
        let ks: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear(t, wk, &zero_b, d, d))
            .collect();
        let vs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear(t, wv, &zero_b, d, d))
            .collect();

        let mut out: Vec<Vec<f32>> = vec![vec![0.0; d]; n];
        for h in 0..n_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let mut attn = vec![0.0_f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    attn[i * n + j] = dot_f32(&qs[i][start..end], &ks[j][start..end]) / scale;
                }
                let row = softmax_f32(&attn[i * n..(i + 1) * n]);
                attn[i * n..(i + 1) * n].copy_from_slice(&row);
            }
            for i in 0..n {
                for j in 0..n {
                    let a = attn[i * n + j];
                    for k in start..end {
                        out[i][k] += a * vs[j][k];
                    }
                }
            }
        }
        out.iter().map(|t| linear(t, wo, &zero_b, d, d)).collect()
    }

    /// Forward: x layout [T, H, W, D] flattened.  Returns same shape.
    pub fn forward(
        &self,
        x: &[f32],
        t: usize,
        h: usize,
        w: usize,
        d: usize,
        win_t: usize,
        win_h: usize,
        win_w: usize,
    ) -> Vec<f32> {
        assert_eq!(x.len(), t * h * w * d);

        // Optionally cyclic shift
        let shifted = if self.shift {
            let s = x.to_vec();
            // Shift along T
            
            Self::cyclic_shift_1d(&s, t * h * w, win_t / 2)
        } else {
            x.to_vec()
        };

        let n_wt = (t + win_t - 1) / win_t;
        let n_wh = (h + win_h - 1) / win_h;
        let n_ww = (w + win_w - 1) / win_w;

        // Output buffer
        let mut out = shifted.clone();

        for wt in 0..n_wt {
            for wh in 0..n_wh {
                for ww in 0..n_ww {
                    // Collect tokens in this 3D window
                    let mut window_tokens: Vec<(usize, Vec<f32>)> = Vec::new();
                    for ti in (wt * win_t)..((wt + 1) * win_t).min(t) {
                        for hi in (wh * win_h)..((wh + 1) * win_h).min(h) {
                            for wi in (ww * win_w)..((ww + 1) * win_w).min(w) {
                                let flat = ((ti * h + hi) * w + wi) * d;
                                let token = shifted[flat..flat + d].to_vec();
                                window_tokens.push((flat, token));
                            }
                        }
                    }

                    let tokens: Vec<Vec<f32>> =
                        window_tokens.iter().map(|(_, t)| t.clone()).collect();
                    let attended = Self::window_attention(
                        &tokens,
                        &self.wq,
                        &self.wk,
                        &self.wv,
                        &self.wo,
                        d,
                        self.n_heads,
                    );

                    // Write back with residual + layernorm
                    let ffn_dim = d * 4;
                    for (idx, (flat, orig)) in window_tokens.iter().enumerate() {
                        let pre_ffn: Vec<f32> = layer_norm_f32(
                            &orig
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| v + attended[idx][i])
                                .collect::<Vec<_>>(),
                        );
                        // FFN
                        let h1: Vec<f32> = (0..ffn_dim)
                            .map(|oi| {
                                relu(
                                    dot_f32(&self.w1[oi * d..(oi + 1) * d], &pre_ffn) + self.b1[oi],
                                )
                            })
                            .collect();
                        let h2: Vec<f32> = (0..d)
                            .map(|oi| {
                                dot_f32(&self.w2[oi * ffn_dim..(oi + 1) * ffn_dim], &h1)
                                    + self.b2[oi]
                            })
                            .collect();
                        let final_tok = layer_norm_f32(
                            &pre_ffn
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| v + h2[i])
                                .collect::<Vec<_>>(),
                        );
                        out[*flat..(d + *flat)].copy_from_slice(&final_tok[..d]);
                    }
                }
            }
        }

        // Reverse cyclic shift
        if self.shift {
            let shift_back = win_t / 2;
            if shift_back > 0 {
                let n_flat = t * h * w;
                let shifted_back = Self::cyclic_shift_1d(&out, n_flat, win_t - shift_back);
                return shifted_back;
            }
        }
        out
    }
}

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;
