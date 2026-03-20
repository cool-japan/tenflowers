//! Video Understanding — Advanced: modern video transformer architectures, self-supervised
//! learning, video object segmentation, and video generation models.
//!
//! ## Components
//!
//! - **VideoSwinBlockV2** / **Hiera** / **TimeSformer** — advanced video transformer blocks
//! - **VideoMaeV2** / **Dino4Video** / **VideoContrastive** — video self-supervised learning
//! - **MemoryBank** / **MemoryReader** / **VosDecoder** — video object segmentation
//! - **VideoLdmUnet** / **ConsistencyModel** / **VuMetrics** — video generation & evaluation

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::{dot_f32, layer_norm_f32, linear, rand_vec_f32, relu, softmax_f32};

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers for advanced.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller normal sample.
fn adv_rand_normal(rng: &mut StdRng, std_dev: f32) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-12);
    let u2: f32 = rng.random::<f32>();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos() * std_dev
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Video Transformers
// ─────────────────────────────────────────────────────────────────────────────

/// 3D relative position bias table for Swin-style attention.
/// Stores biases indexed by relative (t, h, w) offset.
#[derive(Debug, Clone)]
pub struct RelativePositionBias3D {
    /// Bias table [max_t_offsets * max_h_offsets * max_w_offsets].
    pub table: Vec<f32>,
    pub max_t: usize,
    pub max_h: usize,
    pub max_w: usize,
}

impl RelativePositionBias3D {
    /// Create a learnable 3D relative-position bias table.
    pub fn new(max_t: usize, max_h: usize, max_w: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = (2 * max_t - 1) * (2 * max_h - 1) * (2 * max_w - 1);
        let std = 0.02_f32;
        let table = (0..n).map(|_| adv_rand_normal(&mut rng, std)).collect();
        Self { table, max_t, max_h, max_w }
    }

    /// Look up bias for relative offset (dt, dh, dw).
    pub fn lookup(&self, dt: isize, dh: isize, dw: isize) -> f32 {
        let mt = self.max_t as isize;
        let mh = self.max_h as isize;
        let mw = self.max_w as isize;
        let dt_c = (dt + mt - 1).clamp(0, 2 * mt - 2);
        let dh_c = (dh + mh - 1).clamp(0, 2 * mh - 2);
        let dw_c = (dw + mw - 1).clamp(0, 2 * mw - 2);
        let idx = dt_c as usize * (2 * self.max_h - 1) * (2 * self.max_w - 1)
            + dh_c as usize * (2 * self.max_w - 1)
            + dw_c as usize;
        self.table.get(idx).copied().unwrap_or(0.0)
    }
}

/// Swin-style shifted window attention block for video with 3D relative position bias.
///
/// Reference: Liu et al. (2022) Video Swin Transformer.
pub struct VideoSwinBlockV2 {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Whether to apply cyclic shift.
    pub shift: bool,
    /// 3D relative position bias.
    pub pos_bias: RelativePositionBias3D,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl VideoSwinBlockV2 {
    /// Create a new VideoSwinBlockV2 with 3D relative position bias.
    pub fn new(d_model: usize, n_heads: usize, shift: bool, win_t: usize, win_h: usize, win_w: usize, seed: u64) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        let dd = d_model * d_model;
        let ffn_dim = d_model * 4;
        Self {
            d_model,
            n_heads,
            shift,
            pos_bias: RelativePositionBias3D::new(win_t, win_h, win_w, seed.wrapping_add(1)),
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

    /// Forward pass on [T, H, W, D] flattened input; returns same shape.
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
        let n_wt = (t + win_t - 1) / win_t;
        let n_wh = (h + win_h - 1) / win_h;
        let n_ww = (w + win_w - 1) / win_w;
        let head_dim = d / self.n_heads;
        let scale = (head_dim as f32).sqrt();
        let ffn_dim = d * 4;
        let zero_b = vec![0.0_f32; d];

        let mut out = x.to_vec();

        for wt_idx in 0..n_wt {
            for wh_idx in 0..n_wh {
                for ww_idx in 0..n_ww {
                    let mut window_tokens: Vec<(usize, Vec<f32>, (usize, usize, usize))> = Vec::new();
                    for ti in (wt_idx * win_t)..((wt_idx + 1) * win_t).min(t) {
                        for hi in (wh_idx * win_h)..((wh_idx + 1) * win_h).min(h) {
                            for wi in (ww_idx * win_w)..((ww_idx + 1) * win_w).min(w) {
                                let flat = ((ti * h + hi) * w + wi) * d;
                                let token = x[flat..flat + d].to_vec();
                                window_tokens.push((flat, token, (ti, hi, wi)));
                            }
                        }
                    }
                    let n = window_tokens.len();
                    if n == 0 { continue; }

                    let tokens: Vec<&Vec<f32>> = window_tokens.iter().map(|(_, t, _)| t).collect();
                    let qs: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, &self.wq, &zero_b, d, d)).collect();
                    let ks: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, &self.wk, &zero_b, d, d)).collect();
                    let vs: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, &self.wv, &zero_b, d, d)).collect();

                    let mut attn_out: Vec<Vec<f32>> = vec![vec![0.0; d]; n];
                    for head in 0..self.n_heads {
                        let s = head * head_dim;
                        let e = s + head_dim;
                        let mut attn = vec![0.0_f32; n * n];
                        for i in 0..n {
                            for j in 0..n {
                                let qk = dot_f32(&qs[i][s..e], &ks[j][s..e]) / scale;
                                let (ti, hi, wi) = window_tokens[i].2;
                                let (tj, hj, wj) = window_tokens[j].2;
                                let bias = self.pos_bias.lookup(
                                    ti as isize - tj as isize,
                                    hi as isize - hj as isize,
                                    wi as isize - wj as isize,
                                );
                                attn[i * n + j] = qk + bias;
                            }
                            let row_sm = softmax_f32(&attn[i * n..(i + 1) * n]);
                            attn[i * n..(i + 1) * n].copy_from_slice(&row_sm);
                        }
                        for i in 0..n {
                            for j in 0..n {
                                let a = attn[i * n + j];
                                for k in s..e {
                                    attn_out[i][k] += a * vs[j][k];
                                }
                            }
                        }
                    }

                    for (idx, (flat, orig, _)) in window_tokens.iter().enumerate() {
                        let proj = linear(&attn_out[idx], &self.wo, &zero_b, d, d);
                        let pre_ffn: Vec<f32> = layer_norm_f32(
                            &orig.iter().enumerate().map(|(i, &v)| v + proj[i]).collect::<Vec<_>>()
                        );
                        let h1: Vec<f32> = (0..ffn_dim)
                            .map(|oi| relu(dot_f32(&self.w1[oi * d..(oi + 1) * d], &pre_ffn) + self.b1[oi]))
                            .collect();
                        let h2: Vec<f32> = (0..d)
                            .map(|oi| dot_f32(&self.w2[oi * ffn_dim..(oi + 1) * ffn_dim], &h1) + self.b2[oi])
                            .collect();
                        let final_tok = layer_norm_f32(
                            &pre_ffn.iter().enumerate().map(|(i, &v)| v + h2[i]).collect::<Vec<_>>()
                        );
                        out[*flat..(d + *flat)].copy_from_slice(&final_tok[..d]);
                    }
                }
            }
        }
        out
    }
}

/// Hierarchical Vision Transformer for video (Hiera, Ryali 2023).
///
/// Multi-scale masking with spatial pooling attention.  Each stage halves
/// the spatial resolution using a pooling kernel applied before attention.
pub struct Hiera {
    /// Stages: list of (d_model_in, d_model_out, n_heads, pool_stride).
    pub stages: Vec<(usize, usize, usize, usize)>,
    stage_weights: Vec<HieraStage>,
}

struct HieraStage {
    pool_proj: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d_in: usize,
    d_out: usize,
    n_heads: usize,
}

impl Hiera {
    /// Create a new Hiera model with given stage configurations.
    ///
    /// `stages`: list of (d_in, d_out, n_heads, pool_stride) tuples.
    pub fn new(stages: Vec<(usize, usize, usize, usize)>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let stage_weights = stages.iter().map(|&(d_in, d_out, n_heads, _pool)| {
            let std_proj = (2.0_f32 / d_in as f32).sqrt();
            let std_attn = (2.0_f32 / d_out as f32).sqrt();
            let ffn_dim = d_out * 4;
            HieraStage {
                pool_proj: rand_vec_f32(d_in * d_out, std_proj, &mut rng),
                wq: rand_vec_f32(d_out * d_out, std_attn, &mut rng),
                wk: rand_vec_f32(d_out * d_out, std_attn, &mut rng),
                wv: rand_vec_f32(d_out * d_out, std_attn, &mut rng),
                wo: rand_vec_f32(d_out * d_out, std_attn, &mut rng),
                w1: rand_vec_f32(d_out * ffn_dim, std_attn, &mut rng),
                b1: vec![0.0; ffn_dim],
                w2: rand_vec_f32(ffn_dim * d_out, std_attn, &mut rng),
                b2: vec![0.0; d_out],
                d_in,
                d_out,
                n_heads,
            }
        }).collect();
        Self { stages, stage_weights }
    }

    /// Forward pass.  Input `x` is [N, D_in] (N tokens, D_in features).
    /// Returns [N / (pool_stride^2 * prod_stages), D_last_out] flattened.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> (Vec<f32>, usize) {
        let mut current = x.to_vec();
        let mut n = n_tokens;

        for stage in &self.stage_weights {
            let d_in = stage.d_in;
            let d_out = stage.d_out;
            let n_heads = stage.n_heads;
            let head_dim = d_out / n_heads;
            let scale = (head_dim as f32).sqrt();
            let ffn_dim = d_out * 4;
            let zero_b = vec![0.0_f32; d_out];

            // Pool + project: average every pool_stride tokens
            let pool_stride = self.stages[self.stage_weights.iter().position(|s| {
                std::ptr::eq(s as *const _, stage as *const _)
            }).unwrap_or(0)].3;
            let n_out = n / pool_stride.max(1);
            let n_out = n_out.max(1);

            let mut pooled = vec![0.0_f32; n_out * d_out];
            for i in 0..n_out {
                let start = i * pool_stride;
                let end = (start + pool_stride).min(n);
                let mut avg = vec![0.0_f32; d_in];
                let cnt = (end - start) as f32;
                for j in start..end {
                    for k in 0..d_in {
                        avg[k] += current[j * d_in + k] / cnt;
                    }
                }
                let proj = linear(&avg, &stage.pool_proj, &vec![0.0; d_out], d_in, d_out);
                pooled[i * d_out..(i + 1) * d_out].copy_from_slice(&proj);
            }

            // Self-attention over pooled tokens
            let tokens: Vec<Vec<f32>> = (0..n_out)
                .map(|i| pooled[i * d_out..(i + 1) * d_out].to_vec())
                .collect();
            let qs: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, &stage.wq, &zero_b, d_out, d_out)).collect();
            let ks: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, &stage.wk, &zero_b, d_out, d_out)).collect();
            let vs: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, &stage.wv, &zero_b, d_out, d_out)).collect();

            let mut attn_out = vec![0.0_f32; n_out * d_out];
            for head in 0..n_heads {
                let s = head * head_dim;
                let e = s + head_dim;
                let mut attn = vec![0.0_f32; n_out * n_out];
                for i in 0..n_out {
                    for j in 0..n_out {
                        attn[i * n_out + j] = dot_f32(&qs[i][s..e], &ks[j][s..e]) / scale;
                    }
                    let row = softmax_f32(&attn[i * n_out..(i + 1) * n_out]);
                    attn[i * n_out..(i + 1) * n_out].copy_from_slice(&row);
                }
                for i in 0..n_out {
                    for j in 0..n_out {
                        let a = attn[i * n_out + j];
                        for k in s..e {
                            attn_out[i * d_out + k] += a * vs[j][k];
                        }
                    }
                }
            }

            // Output projection + FFN with residual + layernorm
            let mut result = vec![0.0_f32; n_out * d_out];
            for i in 0..n_out {
                let proj = linear(&attn_out[i * d_out..(i + 1) * d_out], &stage.wo, &zero_b, d_out, d_out);
                let res: Vec<f32> = (0..d_out).map(|k| pooled[i * d_out + k] + proj[k]).collect();
                let pre_ffn = layer_norm_f32(&res);
                let h1: Vec<f32> = (0..ffn_dim)
                    .map(|oi| relu(dot_f32(&stage.w1[oi * d_out..(oi + 1) * d_out], &pre_ffn) + stage.b1[oi]))
                    .collect();
                let h2: Vec<f32> = (0..d_out)
                    .map(|oi| dot_f32(&stage.w2[oi * ffn_dim..(oi + 1) * ffn_dim], &h1) + stage.b2[oi])
                    .collect();
                let normed = layer_norm_f32(&(0..d_out).map(|k| pre_ffn[k] + h2[k]).collect::<Vec<_>>());
                result[i * d_out..(i + 1) * d_out].copy_from_slice(&normed);
            }

            current = result;
            n = n_out;
        }
        (current, n)
    }
}

/// Divided Space-Time Attention (TimeSformer, Bertasius 2021).
///
/// Applies temporal attention first (across time at each spatial position),
/// then spatial attention (across positions at each time step).
pub struct TimeSformer {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    // Temporal attention weights
    wq_t: Vec<f32>,
    wk_t: Vec<f32>,
    wv_t: Vec<f32>,
    wo_t: Vec<f32>,
    // Spatial attention weights
    wq_s: Vec<f32>,
    wk_s: Vec<f32>,
    wv_s: Vec<f32>,
    wo_s: Vec<f32>,
    // FFN
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl TimeSformer {
    /// Create a new TimeSformer block.
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        let dd = d_model * d_model;
        let ffn_dim = d_model * 4;
        Self {
            d_model,
            n_heads,
            wq_t: rand_vec_f32(dd, std, &mut rng),
            wk_t: rand_vec_f32(dd, std, &mut rng),
            wv_t: rand_vec_f32(dd, std, &mut rng),
            wo_t: rand_vec_f32(dd, std, &mut rng),
            wq_s: rand_vec_f32(dd, std, &mut rng),
            wk_s: rand_vec_f32(dd, std, &mut rng),
            wv_s: rand_vec_f32(dd, std, &mut rng),
            wo_s: rand_vec_f32(dd, std, &mut rng),
            w1: rand_vec_f32(d_model * ffn_dim, std, &mut rng),
            b1: vec![0.0; ffn_dim],
            w2: rand_vec_f32(ffn_dim * d_model, std, &mut rng),
            b2: vec![0.0; d_model],
        }
    }

    fn attend_seq(
        tokens: &[Vec<f32>], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32],
        d: usize, n_heads: usize,
    ) -> Vec<Vec<f32>> {
        let n = tokens.len();
        if n == 0 { return Vec::new(); }
        let head_dim = d / n_heads;
        let scale = (head_dim as f32).sqrt();
        let zero_b = vec![0.0_f32; d];
        let qs: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, wq, &zero_b, d, d)).collect();
        let ks: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, wk, &zero_b, d, d)).collect();
        let vs: Vec<Vec<f32>> = tokens.iter().map(|t| linear(t, wv, &zero_b, d, d)).collect();
        let mut out_tokens: Vec<Vec<f32>> = vec![vec![0.0; d]; n];
        for head in 0..n_heads {
            let s = head * head_dim;
            let e = s + head_dim;
            let mut attn = vec![0.0_f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    attn[i * n + j] = dot_f32(&qs[i][s..e], &ks[j][s..e]) / scale;
                }
                let row = softmax_f32(&attn[i * n..(i + 1) * n]);
                attn[i * n..(i + 1) * n].copy_from_slice(&row);
            }
            for i in 0..n {
                for j in 0..n {
                    let a = attn[i * n + j];
                    for k in s..e {
                        out_tokens[i][k] += a * vs[j][k];
                    }
                }
            }
        }
        out_tokens.iter().map(|t| linear(t, wo, &zero_b, d, d)).collect()
    }

    /// Forward pass.  `x` is [T*H*W, D] flattened.  Returns same shape.
    pub fn forward(&self, x: &[f32], t: usize, h: usize, w: usize, d: usize) -> Vec<f32> {
        let spatial = h * w;
        let total = t * spatial;
        assert_eq!(x.len(), total * d);

        let tokens: Vec<Vec<f32>> = (0..total).map(|i| x[i * d..(i + 1) * d].to_vec()).collect();
        let mut after_temporal = tokens.clone();

        // Temporal attention: for each spatial position, attend across T
        for si in 0..spatial {
            let seq: Vec<Vec<f32>> = (0..t).map(|ti| tokens[ti * spatial + si].clone()).collect();
            let attended = Self::attend_seq(&seq, &self.wq_t, &self.wk_t, &self.wv_t, &self.wo_t, d, self.n_heads);
            for ti in 0..t {
                let res: Vec<f32> = (0..d).map(|k| tokens[ti * spatial + si][k] + attended[ti][k]).collect();
                after_temporal[ti * spatial + si] = layer_norm_f32(&res);
            }
        }

        // Spatial attention: for each time step, attend across H*W
        let mut after_spatial = after_temporal.clone();
        for ti in 0..t {
            let seq: Vec<Vec<f32>> = (0..spatial).map(|si| after_temporal[ti * spatial + si].clone()).collect();
            let attended = Self::attend_seq(&seq, &self.wq_s, &self.wk_s, &self.wv_s, &self.wo_s, d, self.n_heads);
            for si in 0..spatial {
                let res: Vec<f32> = (0..d).map(|k| after_temporal[ti * spatial + si][k] + attended[si][k]).collect();
                after_spatial[ti * spatial + si] = layer_norm_f32(&res);
            }
        }

        // FFN
        let ffn_dim = d * 4;
        let mut result = vec![0.0_f32; total * d];
        for i in 0..total {
            let token = &after_spatial[i];
            let h1: Vec<f32> = (0..ffn_dim)
                .map(|oi| relu(dot_f32(&self.w1[oi * d..(oi + 1) * d], token) + self.b1[oi]))
                .collect();
            let h2: Vec<f32> = (0..d)
                .map(|oi| dot_f32(&self.w2[oi * ffn_dim..(oi + 1) * ffn_dim], &h1) + self.b2[oi])
                .collect();
            let ln = layer_norm_f32(&(0..d).map(|k| after_spatial[i][k] + h2[k]).collect::<Vec<_>>());
            result[i * d..(i + 1) * d].copy_from_slice(&ln);
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Video Self-Supervised Learning
// ─────────────────────────────────────────────────────────────────────────────

/// VideoMAE V2 — masked autoencoder with tube masking and reconstruction target normalization.
///
/// Uses 90% masking ratio and pixel-space reconstruction with per-patch normalization target.
pub struct VideoMaeV2 {
    /// Spatial patch size (square).
    pub patch_size: usize,
    /// Temporal patch size.
    pub patch_t: usize,
    /// Masking ratio (0..1), default 0.9.
    pub mask_ratio: f32,
    /// Encoder embedding dimension.
    pub encoder_dim: usize,
    enc_w: Vec<f32>,
    enc_b: Vec<f32>,
    dec_w: Vec<f32>,
    dec_b: Vec<f32>,
}

impl VideoMaeV2 {
    /// Create a new VideoMaeV2.
    pub fn new(patch_size: usize, patch_t: usize, mask_ratio: f32, encoder_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let patch_tokens = patch_t * patch_size * patch_size * 3; // RGB channels
        let std_enc = (2.0_f32 / patch_tokens as f32).sqrt();
        let std_dec = (2.0_f32 / encoder_dim as f32).sqrt();
        Self {
            patch_size,
            patch_t,
            mask_ratio,
            encoder_dim,
            enc_w: rand_vec_f32(patch_tokens * encoder_dim, std_enc, &mut rng),
            enc_b: vec![0.0; encoder_dim],
            dec_w: rand_vec_f32(encoder_dim * patch_tokens, std_dec, &mut rng),
            dec_b: vec![0.0; patch_tokens],
        }
    }

    /// Create a tube mask for VideoMAE V2 at 90% masking ratio.
    /// Returns boolean mask of length `n_patches` (true = masked).
    pub fn create_tube_mask(n_patches: usize, mask_ratio: f32, rng: &mut StdRng) -> Vec<bool> {
        let n_mask = ((n_patches as f32 * mask_ratio) as usize).min(n_patches);
        let mut indices: Vec<usize> = (0..n_patches).collect();
        for i in (1..n_patches).rev() {
            let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
        let mut mask = vec![false; n_patches];
        for &idx in &indices[..n_mask] {
            mask[idx] = true;
        }
        mask
    }

    /// Encode visible patches: [n_visible, patch_tokens] → [n_visible, encoder_dim].
    pub fn encode(&self, visible_patches: &[f32], n_visible: usize) -> Vec<f32> {
        let patch_tokens = self.patch_t * self.patch_size * self.patch_size * 3;
        assert_eq!(visible_patches.len(), n_visible * patch_tokens);
        let mut out = vec![0.0_f32; n_visible * self.encoder_dim];
        for i in 0..n_visible {
            let patch = &visible_patches[i * patch_tokens..(i + 1) * patch_tokens];
            let enc = linear(patch, &self.enc_w, &self.enc_b, patch_tokens, self.encoder_dim);
            let normed = layer_norm_f32(&enc);
            out[i * self.encoder_dim..(i + 1) * self.encoder_dim].copy_from_slice(&normed);
        }
        out
    }

    /// Decode encoded tokens: [n_visible, encoder_dim] → [n_visible, patch_tokens].
    pub fn decode(&self, encoded: &[f32], n_visible: usize) -> Vec<f32> {
        let patch_tokens = self.patch_t * self.patch_size * self.patch_size * 3;
        assert_eq!(encoded.len(), n_visible * self.encoder_dim);
        let mut out = vec![0.0_f32; n_visible * patch_tokens];
        for i in 0..n_visible {
            let feat = &encoded[i * self.encoder_dim..(i + 1) * self.encoder_dim];
            let decoded = linear(feat, &self.dec_w, &self.dec_b, self.encoder_dim, patch_tokens);
            out[i * patch_tokens..(i + 1) * patch_tokens].copy_from_slice(&decoded);
        }
        out
    }

    /// Compute per-patch normalized MSE reconstruction loss on masked patches.
    pub fn reconstruction_loss_normalized(&self, pred: &[f32], target: &[f32], mask: &[bool]) -> f32 {
        assert_eq!(pred.len(), target.len());
        let n_patches = mask.len();
        if n_patches == 0 { return 0.0; }
        let patch_dim = pred.len() / n_patches;
        let mut loss = 0.0_f32;
        let mut count = 0usize;
        for (pi, &masked) in mask.iter().enumerate() {
            if !masked { continue; }
            let p_slice = &pred[pi * patch_dim..(pi + 1) * patch_dim];
            let t_slice = &target[pi * patch_dim..(pi + 1) * patch_dim];
            // Per-patch normalization of target
            let t_mean = t_slice.iter().sum::<f32>() / patch_dim as f32;
            let t_var = t_slice.iter().map(|&v| (v - t_mean) * (v - t_mean)).sum::<f32>() / patch_dim as f32;
            let t_std = (t_var + 1e-6).sqrt();
            for di in 0..patch_dim {
                let t_norm = (t_slice[di] - t_mean) / t_std;
                loss += (p_slice[di] - t_norm) * (p_slice[di] - t_norm);
            }
            count += patch_dim;
        }
        if count == 0 { 0.0 } else { loss / count as f32 }
    }
}

/// DINO-style self-distillation for video with spatiotemporal crops.
///
/// Maintains an exponential-moving-average (EMA) teacher and a student network,
/// training the student to match teacher softmax outputs via cross-entropy.
pub struct Dino4Video {
    /// Embedding dimension.
    pub d_model: usize,
    /// Number of projection output dimensions.
    pub proj_dim: usize,
    /// EMA momentum for teacher update.
    pub momentum: f32,
    /// Student projection head weights.
    pub student_w: Vec<f32>,
    /// Student projection head biases.
    pub student_b: Vec<f32>,
    /// Teacher projection head weights (EMA of student).
    pub teacher_w: Vec<f32>,
    /// Teacher projection head biases (EMA of student).
    pub teacher_b: Vec<f32>,
    // Teacher center (running mean of teacher outputs for centering)
    center: Vec<f32>,
    /// Sharpening temperature for teacher softmax.
    pub teacher_temp: f32,
    /// Student temperature.
    pub student_temp: f32,
}

impl Dino4Video {
    /// Create a new Dino4Video model.
    pub fn new(d_model: usize, proj_dim: usize, momentum: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        let w = rand_vec_f32(d_model * proj_dim, std, &mut rng);
        let b = vec![0.0_f32; proj_dim];
        Self {
            d_model,
            proj_dim,
            momentum,
            student_w: w.clone(),
            student_b: b.clone(),
            teacher_w: w,
            teacher_b: b,
            center: vec![0.0_f32; proj_dim],
            teacher_temp: 0.04,
            student_temp: 0.1,
        }
    }

    /// Student forward: project token → softmax logits.
    pub fn student_forward(&self, x: &[f32]) -> Vec<f32> {
        let proj = linear(x, &self.student_w, &self.student_b, self.d_model, self.proj_dim);
        let scaled: Vec<f32> = proj.iter().map(|&v| v / self.student_temp).collect();
        softmax_f32(&scaled)
    }

    /// Teacher forward: project token → sharpened softmax with centering.
    pub fn teacher_forward(&self, x: &[f32]) -> Vec<f32> {
        let proj = linear(x, &self.teacher_w, &self.teacher_b, self.d_model, self.proj_dim);
        let centered: Vec<f32> = proj.iter().zip(self.center.iter())
            .map(|(&p, &c)| (p - c) / self.teacher_temp)
            .collect();
        softmax_f32(&centered)
    }

    /// DINO cross-entropy loss: -sum(teacher_prob * log(student_prob + eps)).
    pub fn dino_loss(&self, teacher_probs: &[f32], student_probs: &[f32]) -> f32 {
        teacher_probs.iter().zip(student_probs.iter())
            .map(|(&t, &s)| -t * (s + 1e-8).ln())
            .sum::<f32>()
    }

    /// EMA update of teacher weights.
    pub fn update_teacher(&mut self) {
        for (tw, sw) in self.teacher_w.iter_mut().zip(self.student_w.iter()) {
            *tw = self.momentum * *tw + (1.0 - self.momentum) * sw;
        }
        for (tb, sb) in self.teacher_b.iter_mut().zip(self.student_b.iter()) {
            *tb = self.momentum * *tb + (1.0 - self.momentum) * sb;
        }
    }

    /// Update center using EMA of teacher output.
    pub fn update_center(&mut self, teacher_output: &[f32]) {
        let m = 0.9_f32;
        for (c, &v) in self.center.iter_mut().zip(teacher_output.iter()) {
            *c = m * *c + (1.0 - m) * v;
        }
    }
}

/// Contrastive learning for video using temporal positive pairs and spatial augmentations.
///
/// Adjacent clips from the same video are treated as positives;
/// NT-Xent (normalized temperature-scaled cross-entropy) loss is used.
pub struct VideoContrastive {
    /// Embedding dimension.
    pub d_model: usize,
    /// Projection head output dimension.
    pub proj_dim: usize,
    /// Temperature for NT-Xent.
    pub temperature: f32,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
}

impl VideoContrastive {
    /// Create a new VideoContrastive model.
    pub fn new(d_model: usize, proj_dim: usize, temperature: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        Self {
            d_model,
            proj_dim,
            temperature,
            proj_w: rand_vec_f32(d_model * proj_dim, std, &mut rng),
            proj_b: vec![0.0; proj_dim],
        }
    }

    /// Project a clip embedding to the contrastive space (L2 normalized).
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let p = linear(x, &self.proj_w, &self.proj_b, self.d_model, self.proj_dim);
        let norm = p.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8);
        p.iter().map(|&v| v / norm).collect()
    }

    /// NT-Xent loss for a batch of (anchor, positive) L2-normalized embedding pairs.
    ///
    /// `embeddings` is [2N, proj_dim] where row `i` and row `i+N` are a positive pair.
    pub fn nt_xent_loss(&self, embeddings: &[Vec<f32>]) -> f32 {
        let n2 = embeddings.len();
        if n2 < 2 { return 0.0; }
        let n = n2 / 2;
        let mut sim = vec![0.0_f32; n2 * n2];
        for i in 0..n2 {
            for j in 0..n2 {
                if i == j { continue; }
                let s = dot_f32(&embeddings[i], &embeddings[j]) / self.temperature;
                sim[i * n2 + j] = s;
            }
        }

        let mut loss = 0.0_f32;
        for i in 0..n {
            // Positive for i is i+n
            let pos_j = i + n;
            let row_i: Vec<f32> = (0..n2).filter(|&j| j != i)
                .map(|j| sim[i * n2 + j])
                .collect();
            let log_sum_exp = {
                let max_v = row_i.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                max_v + row_i.iter().map(|&v| (v - max_v).exp()).sum::<f32>().ln()
            };
            loss += -(sim[i * n2 + pos_j] - log_sum_exp);

            // Positive for i+n is i
            let row_j: Vec<f32> = (0..n2).filter(|&k| k != pos_j)
                .map(|k| sim[pos_j * n2 + k])
                .collect();
            let log_sum_exp_j = {
                let max_v = row_j.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                max_v + row_j.iter().map(|&v| (v - max_v).exp()).sum::<f32>().ln()
            };
            loss += -(sim[pos_j * n2 + i] - log_sum_exp_j);
        }
        loss / (2.0 * n as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Video Object Segmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Key-value memory bank for video object segmentation.
///
/// Based on Space-Time Memory Networks (STM, Oh 2019).
/// Stores per-frame (key, value) pairs for memory-based VOS.
pub struct MemoryBank {
    /// Stored keys: [n_frames, n_pixels, key_dim].
    pub keys: Vec<Vec<f32>>,
    /// Stored values: [n_frames, n_pixels, val_dim].
    pub values: Vec<Vec<f32>>,
    /// Key dimension.
    pub key_dim: usize,
    /// Value dimension.
    pub val_dim: usize,
}

impl MemoryBank {
    /// Create an empty memory bank.
    pub fn new(key_dim: usize, val_dim: usize) -> Self {
        Self { keys: Vec::new(), values: Vec::new(), key_dim, val_dim }
    }

    /// Add a frame's key-value pair to the memory.
    ///
    /// `key` is [n_pixels, key_dim] flattened; `value` is [n_pixels, val_dim] flattened.
    pub fn push(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(key);
        self.values.push(value);
    }

    /// Number of memory frames.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Attention-based memory reader with top-k sparse retrieval.
///
/// Given a query key and a memory bank, retrieves the value using
/// dot-product attention over all stored keys.
pub struct MemoryReader {
    /// Number of top-k keys to attend over (sparse retrieval).
    pub top_k: usize,
}

impl MemoryReader {
    /// Create a new MemoryReader.
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    /// Read from memory bank.
    ///
    /// `query_key` is [n_pixels, key_dim] flattened.
    /// Returns [n_pixels, val_dim] flattened retrieved value.
    pub fn read(&self, bank: &MemoryBank, query_key: &[f32], n_pixels: usize) -> Vec<f32> {
        if bank.is_empty() {
            return vec![0.0_f32; n_pixels * bank.val_dim];
        }
        let key_dim = bank.key_dim;
        let val_dim = bank.val_dim;
        let n_frames = bank.len();

        let top_k = self.top_k.min(n_frames);
        let mut result = vec![0.0_f32; n_pixels * val_dim];

        for pi in 0..n_pixels {
            let q = &query_key[pi * key_dim..(pi + 1) * key_dim];
            let scale = (key_dim as f32).sqrt();

            // Compute similarity with all memory frame keys (use first pixel per frame for simplicity)
            let scores: Vec<f32> = bank.keys.iter().map(|k| {
                let k_slice = if k.len() >= key_dim { &k[..key_dim] } else { &k[..] };
                dot_f32(q, k_slice) / scale
            }).collect();

            // Top-k selection
            let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<usize> = indexed.iter().take(top_k).map(|&(i, _)| i).collect();
            let top_scores: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
            let attn = softmax_f32(&top_scores);

            // Aggregate values
            for (ai, &frame_idx) in top.iter().enumerate() {
                let v = &bank.values[frame_idx];
                let val_slice = if v.len() >= val_dim { &v[..val_dim] } else { &v[..] };
                let len = val_slice.len().min(val_dim);
                for vi in 0..len {
                    result[pi * val_dim + vi] += attn[ai] * val_slice[vi];
                }
            }
        }
        result
    }
}

/// VOS decoder: fuses query features with memory-retrieved values to produce a segmentation mask.
///
/// Takes [n_pixels, query_dim + val_dim] concatenated features and produces
/// \[n_pixels\] binary segmentation logits.
pub struct VosDecoder {
    /// Query feature dimension.
    pub query_dim: usize,
    /// Memory value dimension.
    pub val_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl VosDecoder {
    /// Create a new VosDecoder.
    pub fn new(query_dim: usize, val_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let in_dim = query_dim + val_dim;
        let hidden = in_dim * 2;
        let std = (2.0_f32 / in_dim as f32).sqrt();
        Self {
            query_dim,
            val_dim,
            w1: rand_vec_f32(in_dim * hidden, std, &mut rng),
            b1: vec![0.0; hidden],
            w2: rand_vec_f32(hidden, std, &mut rng),
            b2: vec![0.0; 1],
        }
    }

    /// Forward pass.
    ///
    /// `query_feat` is [n_pixels, query_dim] flattened.
    /// `memory_feat` is [n_pixels, val_dim] flattened.
    /// Returns \[n_pixels\] logits (positive = foreground).
    pub fn forward(&self, query_feat: &[f32], memory_feat: &[f32], n_pixels: usize) -> Vec<f32> {
        let in_dim = self.query_dim + self.val_dim;
        let hidden = in_dim * 2;
        let mut out = vec![0.0_f32; n_pixels];
        for pi in 0..n_pixels {
            let mut inp = Vec::with_capacity(in_dim);
            inp.extend_from_slice(&query_feat[pi * self.query_dim..(pi + 1) * self.query_dim]);
            inp.extend_from_slice(&memory_feat[pi * self.val_dim..(pi + 1) * self.val_dim]);
            let h1: Vec<f32> = (0..hidden)
                .map(|oi| relu(dot_f32(&self.w1[oi * in_dim..(oi + 1) * in_dim], &inp) + self.b1[oi]))
                .collect();
            let logit = dot_f32(&self.w2, &h1) + self.b2[0];
            out[pi] = logit;
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Video Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Latent Diffusion UNet for video (VideoLDM).
///
/// Uses 3D convolution blocks (simulated as linear layers over flattened 3D patches)
/// interleaved with temporal attention layers for video-coherent generation.
pub struct VideoLdmUnet {
    /// Latent dimension per token.
    pub latent_dim: usize,
    /// Number of diffusion timesteps.
    pub n_timesteps: usize,
    /// Number of encoder/decoder levels.
    pub n_levels: usize,
    // Encoder projection weights per level
    enc_ws: Vec<Vec<f32>>,
    enc_bs: Vec<Vec<f32>>,
    // Temporal attention weights per level
    attn_ws: Vec<Vec<f32>>,
    // Decoder projection weights per level (reverse)
    dec_ws: Vec<Vec<f32>>,
    dec_bs: Vec<Vec<f32>>,
    // Timestep embedding
    t_embed_w: Vec<f32>,
    t_embed_b: Vec<f32>,
    /// Timestep embedding dimension.
    pub t_embed_dim: usize,
}

impl VideoLdmUnet {
    /// Create a new VideoLdmUnet.
    pub fn new(latent_dim: usize, n_timesteps: usize, n_levels: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let t_embed_dim = latent_dim * 4;
        let std = (2.0_f32 / latent_dim as f32).sqrt();

        let mut enc_ws = Vec::with_capacity(n_levels);
        let mut enc_bs = Vec::with_capacity(n_levels);
        let mut attn_ws = Vec::with_capacity(n_levels);
        let mut dec_ws = Vec::with_capacity(n_levels);
        let mut dec_bs = Vec::with_capacity(n_levels);

        for _ in 0..n_levels {
            enc_ws.push(rand_vec_f32(latent_dim * latent_dim, std, &mut rng));
            enc_bs.push(vec![0.0; latent_dim]);
            attn_ws.push(rand_vec_f32(latent_dim * latent_dim, std, &mut rng));
            dec_ws.push(rand_vec_f32(latent_dim * latent_dim, std, &mut rng));
            dec_bs.push(vec![0.0; latent_dim]);
        }

        Self {
            latent_dim,
            n_timesteps,
            n_levels,
            enc_ws,
            enc_bs,
            attn_ws,
            dec_ws,
            dec_bs,
            t_embed_w: rand_vec_f32(t_embed_dim * latent_dim, std, &mut rng),
            t_embed_b: vec![0.0; latent_dim],
            t_embed_dim,
        }
    }

    /// Sinusoidal timestep embedding.
    fn timestep_embed(&self, t: usize) -> Vec<f32> {
        let d = self.t_embed_dim;
        let mut embed = vec![0.0_f32; d];
        for i in 0..d / 2 {
            let freq = 10000.0_f32.powf(2.0 * i as f32 / d as f32);
            embed[2 * i] = (t as f32 / freq).sin();
            embed[2 * i + 1] = (t as f32 / freq).cos();
        }
        embed
    }

    /// Denoise step: given noisy latents [n_tokens, latent_dim] and timestep t,
    /// returns denoised prediction of same shape.
    pub fn denoise(&self, x: &[f32], n_tokens: usize, t: usize) -> Vec<f32> {
        assert_eq!(x.len(), n_tokens * self.latent_dim);
        let d = self.latent_dim;

        // Timestep embedding
        let t_raw = self.timestep_embed(t);
        let t_emb = linear(&t_raw, &self.t_embed_w, &self.t_embed_b, self.t_embed_dim, d);

        // Encoder path
        let mut skips: Vec<Vec<f32>> = Vec::new();
        let mut h = x.to_vec();

        for level in 0..self.n_levels {
            let mut new_h = vec![0.0_f32; n_tokens * d];
            for i in 0..n_tokens {
                let tok = &h[i * d..(i + 1) * d];
                let proj = linear(tok, &self.enc_ws[level], &self.enc_bs[level], d, d);
                let cond: Vec<f32> = (0..d).map(|k| proj[k] + t_emb[k]).collect();
                let normed = layer_norm_f32(&cond);
                // Temporal attention via linear approximation
                let attn: Vec<f32> = (0..d).map(|k| {
                    gelu(normed[k] * self.attn_ws[level][k * d + k.min(d - 1)])
                }).collect();
                for k in 0..d {
                    new_h[i * d + k] = normed[k] + attn[k];
                }
            }
            skips.push(h.clone());
            h = new_h;
        }

        // Decoder path with skip connections
        for level in (0..self.n_levels).rev() {
            let skip = &skips[level];
            let mut new_h = vec![0.0_f32; n_tokens * d];
            for i in 0..n_tokens {
                let tok = &h[i * d..(i + 1) * d];
                let sk = &skip[i * d..(i + 1) * d];
                let combined: Vec<f32> = (0..d).map(|k| tok[k] + sk[k]).collect();
                let proj = linear(&combined, &self.dec_ws[level], &self.dec_bs[level], d, d);
                let cond: Vec<f32> = (0..d).map(|k| proj[k] + t_emb[k]).collect();
                let normed = layer_norm_f32(&cond);
                for k in 0..d {
                    new_h[i * d + k] = normed[k];
                }
            }
            h = new_h;
        }
        h
    }
}

/// Consistency Model for fast video generation (Song 2023).
///
/// Consistency models support 1-step or multi-step generation by training
/// the model to map any noisy sample to the corresponding clean sample.
pub struct ConsistencyModel {
    /// Embedding dimension.
    pub d_model: usize,
    /// Minimum noise level (sigma_min).
    pub sigma_min: f32,
    /// Maximum noise level (sigma_max).
    pub sigma_max: f32,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    // Noise conditioning
    noise_embed_w: Vec<f32>,
    noise_embed_b: Vec<f32>,
}

impl ConsistencyModel {
    /// Create a new ConsistencyModel.
    pub fn new(d_model: usize, sigma_min: f32, sigma_max: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0_f32 / d_model as f32).sqrt();
        let ffn = d_model * 4;
        Self {
            d_model,
            sigma_min,
            sigma_max,
            w1: rand_vec_f32(d_model * ffn, std, &mut rng),
            b1: vec![0.0; ffn],
            w2: rand_vec_f32(ffn * d_model, std, &mut rng),
            b2: vec![0.0; d_model],
            noise_embed_w: rand_vec_f32(d_model, std, &mut rng),
            noise_embed_b: vec![0.0; d_model],
        }
    }

    /// Consistency function f(x, sigma): maps noisy x at noise level sigma to clean x_0.
    ///
    /// Uses the skip-scaling trick: f = c_skip(sigma) * x + c_out(sigma) * F_theta(x, sigma).
    pub fn consistency_fn(&self, x: &[f32], sigma: f32) -> Vec<f32> {
        let d = self.d_model;
        assert_eq!(x.len(), d);
        let sigma_data = 0.5_f32;
        let c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        let c_out = sigma * sigma_data / (sigma * sigma + sigma_data * sigma_data).sqrt();

        // Noise embedding
        let log_sigma = sigma.ln();
        let noise_emb: Vec<f32> = (0..d)
            .map(|k| relu(log_sigma * self.noise_embed_w[k] + self.noise_embed_b[k]))
            .collect();

        // Conditioned input
        let cond: Vec<f32> = (0..d).map(|k| x[k] + noise_emb[k]).collect();

        let ffn = d * 4;
        let h1: Vec<f32> = (0..ffn)
            .map(|oi| gelu(dot_f32(&self.w1[oi * d..(oi + 1) * d], &cond) + self.b1[oi]))
            .collect();
        let net_out: Vec<f32> = (0..d)
            .map(|oi| dot_f32(&self.w2[oi * ffn..(oi + 1) * ffn], &h1) + self.b2[oi])
            .collect();

        (0..d).map(|k| c_skip * x[k] + c_out * net_out[k]).collect()
    }

    /// Consistency distillation loss between student and teacher outputs at two noise levels.
    pub fn consistency_loss(&self, x_t1: &[f32], x_t2: &[f32], sigma1: f32, sigma2: f32) -> f32 {
        let f1 = self.consistency_fn(x_t1, sigma1);
        let f2 = self.consistency_fn(x_t2, sigma2);
        f1.iter().zip(f2.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>() / f1.len() as f32
    }

    /// Multi-step generation: apply consistency model iteratively from sigma_max to sigma_min.
    ///
    /// `n_steps`: number of denoising steps (1 for single-step, up to 4 for quality).
    pub fn generate(&self, noise: &[f32], n_steps: usize, rng: &mut StdRng) -> Vec<f32> {
        let d = self.d_model;
        assert_eq!(noise.len(), d);
        let n_steps = n_steps.clamp(1, 4);

        let sigmas: Vec<f32> = (0..n_steps).map(|i| {
            let t = i as f32 / (n_steps - 1).max(1) as f32;
            self.sigma_max * (self.sigma_min / self.sigma_max).powf(t)
        }).collect();

        let mut x = noise.to_vec();
        for &sigma in &sigmas {
            x = self.consistency_fn(&x, sigma);
            if sigma > self.sigma_min {
                // Add small noise for multi-step quality
                let noise_scale = (sigma * 0.1).min(0.05);
                for v in x.iter_mut() {
                    *v += adv_rand_normal(rng, noise_scale);
                }
            }
        }
        x
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Video Evaluation Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Video Understanding / Generation Metrics.
///
/// - FVD proxy: feature L2 distance between generated and real feature distributions.
/// - Temporal coherence: optical-flow warp error between consecutive frames.
pub struct VuMetrics;

impl VuMetrics {
    /// FVD proxy: mean L2 distance between real and generated feature vectors.
    ///
    /// `real_feats` and `gen_feats` are each [n_samples, feat_dim] flattened.
    pub fn fvd_proxy(real_feats: &[f32], gen_feats: &[f32], n_samples: usize, feat_dim: usize) -> f32 {
        assert_eq!(real_feats.len(), n_samples * feat_dim);
        assert_eq!(gen_feats.len(), n_samples * feat_dim);
        let mut total = 0.0_f32;
        for i in 0..n_samples {
            let r = &real_feats[i * feat_dim..(i + 1) * feat_dim];
            let g = &gen_feats[i * feat_dim..(i + 1) * feat_dim];
            let l2_sq: f32 = r.iter().zip(g.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
            total += l2_sq.sqrt();
        }
        total / n_samples as f32
    }

    /// Temporal coherence via warp error.
    ///
    /// Given consecutive frames `f1` and `f2` \[H*W*C\] and optical flow `flow` \[H*W*2\],
    /// warps `f1` according to `flow` and measures L2 distance to `f2`.
    pub fn temporal_coherence(
        f1: &[f32],
        f2: &[f32],
        flow: &[f32],
        h: usize,
        w: usize,
        c: usize,
    ) -> f32 {
        assert_eq!(f1.len(), h * w * c);
        assert_eq!(f2.len(), h * w * c);
        assert_eq!(flow.len(), h * w * 2);
        let mut warp_error = 0.0_f32;
        let mut count = 0usize;
        for yi in 0..h {
            for xi in 0..w {
                let pi = yi * w + xi;
                let dy = flow[pi * 2];
                let dx = flow[pi * 2 + 1];
                let src_y = yi as f32 + dy;
                let src_x = xi as f32 + dx;
                let y0 = src_y.floor().clamp(0.0, (h - 1) as f32) as usize;
                let x0 = src_x.floor().clamp(0.0, (w - 1) as f32) as usize;
                for ci in 0..c {
                    let warped = f1[(y0 * w + x0) * c + ci];
                    let target = f2[pi * c + ci];
                    warp_error += (warped - target) * (warped - target);
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { (warp_error / count as f32).sqrt() }
    }

    /// Top-1 accuracy for action recognition.
    pub fn top1_accuracy(preds: &[usize], labels: &[usize]) -> f32 {
        if preds.is_empty() { return 0.0; }
        let correct = preds.iter().zip(labels.iter()).filter(|(&p, &l)| p == l).count();
        correct as f32 / preds.len() as f32
    }

    /// Mean IoU for segmentation masks.
    ///
    /// `pred_masks` and `gt_masks` are binary \[H*W\] masks for each frame.
    pub fn mean_iou(pred_masks: &[Vec<bool>], gt_masks: &[Vec<bool>]) -> f32 {
        if pred_masks.is_empty() { return 0.0; }
        let ious: Vec<f32> = pred_masks.iter().zip(gt_masks.iter()).map(|(p, g)| {
            let inter = p.iter().zip(g.iter()).filter(|(&a, &b)| a && b).count() as f32;
            let union = p.iter().zip(g.iter()).filter(|(&a, &b)| a || b).count() as f32;
            if union == 0.0 { 1.0 } else { inter / union }
        }).collect();
        ious.iter().sum::<f32>() / ious.len() as f32
    }
}
