//! Climate and Earth Science ML — climate_ml module.
//!
//! This module provides production-grade ML primitives for climate modelling:
//!
//! - **FourCastNet**: Fourier neural operator for global weather forecasting
//! - **PanGu-Weather**: Hierarchical transformer with Earth position bias
//! - **ClimateDownscaler**: Statistical downscaling via super-resolution
//! - **ExtremeEventDetector**: Rare event prediction with focal loss and oversampling
//! - **AtmosphericEmbedding**: Multi-variable atmospheric state encoding
//! - **OceanCurrentPredictor**: LSTM-based ocean dynamics forecasting
//! - **CarbonFluxEstimator**: Process-based ML hybrid for GPP and respiration
//! - **ClimateProjectionEnsemble**: Multi-model ensemble with uncertainty quantification
//! - **TeleconnectionAnalyzer**: Remote climate pattern detection via EOFs
//! - **ClimateMetrics**: Skill scores, ACC, reliability diagrams, Brier score

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Shared math helpers (f32)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-88.0, 88.0)).exp())
}

#[inline]
fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

fn xavier_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (1.0 / n as f64).sqrt() as f32;
    (0..n)
        .map(|_| {
            let u: f32 = rng.random();
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn normal_samples_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        if i + 1 < n {
            out.push((r * theta.sin()) as f32);
        }
        i += 2;
    }
    out.truncate(n);
    out
}

/// Dense linear layer forward pass (row-major weight matrix).
fn dense_linear_f32(weights: &[f32], bias: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|o| {
            let b = if o < bias.len() { bias[o] } else { 0.0 };
            let s: f32 = (0..in_dim)
                .map(|i| weights[o * in_dim + i] * input[i])
                .sum();
            s + b
        })
        .collect()
}

fn layer_norm_f32(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let g = if i < gamma.len() { gamma[i] } else { 1.0 };
            let b = if i < beta.len() { beta[i] } else { 0.0 };
            g * (v - mean) * inv_std + b
        })
        .collect()
}

/// Discrete Fourier Transform (real→complex) along the first axis of a batched signal.
/// `x` has length `n * channels`; returns (real, imag) each of the same length,
/// retaining all n/2+1 modes (one-sided).
fn rfft_1d(x: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let channels = x.len().checked_div(n).unwrap_or(0);
    let half = n / 2 + 1;
    let mut re = vec![0.0_f32; half * channels];
    let mut im = vec![0.0_f32; half * channels];
    for c in 0..channels {
        for k in 0..half {
            let mut r = 0.0_f32;
            let mut ig = 0.0_f32;
            for t in 0..n {
                let angle = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                r += x[t * channels + c] * angle.cos();
                ig += x[t * channels + c] * angle.sin();
            }
            re[k * channels + c] = r;
            im[k * channels + c] = ig;
        }
    }
    (re, im)
}

/// Inverse DFT (one-sided complex → real).
fn irfft_1d(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    let channels = re.len().checked_div(n / 2 + 1).unwrap_or(0);
    let half = n / 2 + 1;
    let mut out = vec![0.0_f32; n * channels];
    let inv_n = 1.0 / n as f32;
    for t in 0..n {
        for c in 0..channels {
            let mut val = 0.0_f32;
            for k in 0..half {
                let angle = 2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                let factor = if k == 0 || (n % 2 == 0 && k == n / 2) {
                    1.0
                } else {
                    2.0
                };
                val += factor
                    * (re[k * channels + c] * angle.cos() - im[k * channels + c] * angle.sin());
            }
            out[t * channels + c] = val * inv_n;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. FourCastNet — Fourier Neural Operator for weather forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the FourCastNet model.
#[derive(Debug, Clone)]
pub struct FourCastConfig {
    /// Number of latitude grid points.
    pub lat_size: usize,
    /// Number of longitude grid points.
    pub lon_size: usize,
    /// Number of atmospheric channels (variables).
    pub n_channels: usize,
    /// Number of Fourier modes retained (truncation).
    pub n_modes: usize,
    /// Embedding dimension for the lifting layer.
    pub embed_dim: usize,
    /// Number of spectral layers.
    pub n_layers: usize,
}

/// One spherical Fourier layer with complex-valued spectral weights.
/// Weights are stored in row-major (n_modes × embed_dim × 2) layout
/// where the last dim is [real, imag].
#[derive(Debug, Clone)]
pub struct SphericalFourierLayer {
    /// Real parts of spectral weights: shape [n_modes * embed_dim].
    pub weights_r: Vec<f32>,
    /// Imaginary parts of spectral weights: shape [n_modes * embed_dim].
    pub weights_i: Vec<f32>,
    /// Bias for the mixing MLP after spectral operation.
    pub bias: Vec<f32>,
    pub n_modes: usize,
    pub embed_dim: usize,
}

impl SphericalFourierLayer {
    /// Construct with random Xavier initialisation.
    pub fn new(n_modes: usize, embed_dim: usize, seed: u64) -> Self {
        let n = n_modes * embed_dim;
        let weights_r = xavier_vec(n, seed);
        let weights_i = xavier_vec(n, seed.wrapping_add(1));
        let bias = vec![0.0_f32; embed_dim];
        Self {
            weights_r,
            weights_i,
            bias,
            n_modes,
            embed_dim,
        }
    }

    /// Apply spectral mixing on a 1-D lat strip: `x` has length `lat * embed_dim`.
    pub fn forward(&self, x: &[f32], lat: usize) -> Vec<f32> {
        // DFT along lat axis
        let (re, im) = rfft_1d(x, lat);
        // Truncate to n_modes and apply learned complex weights
        let modes = self.n_modes.min(lat / 2 + 1);
        let embed = self.embed_dim;
        let mut re_out = vec![0.0_f32; (lat / 2 + 1) * embed];
        let mut im_out = vec![0.0_f32; (lat / 2 + 1) * embed];
        for k in 0..modes {
            for c in 0..embed {
                let idx = k * embed + c;
                let wr = if idx < self.weights_r.len() {
                    self.weights_r[idx]
                } else {
                    1.0
                };
                let wi = if idx < self.weights_i.len() {
                    self.weights_i[idx]
                } else {
                    0.0
                };
                let r_in = if k * embed + c < re.len() {
                    re[k * embed + c]
                } else {
                    0.0
                };
                let i_in = if k * embed + c < im.len() {
                    im[k * embed + c]
                } else {
                    0.0
                };
                // Complex multiply: (r_in + i*i_in)(wr + i*wi)
                re_out[k * embed + c] = r_in * wr - i_in * wi;
                im_out[k * embed + c] = r_in * wi + i_in * wr;
            }
        }
        // IDFT back
        let mut result = irfft_1d(&re_out, &im_out, lat);
        // Add bias
        for t in 0..lat {
            for c in 0..embed {
                if c < self.bias.len() {
                    result[t * embed + c] += self.bias[c];
                }
            }
        }
        result
    }
}

/// FourCastNet model for global medium-range weather forecasting.
#[derive(Debug, Clone)]
pub struct FourCastNet {
    pub config: FourCastConfig,
    /// Lifting weights: [embed_dim × n_channels]
    pub lift_weights: Vec<f32>,
    pub lift_bias: Vec<f32>,
    pub layers: Vec<SphericalFourierLayer>,
    /// Projection weights: [n_channels × embed_dim]
    pub proj_weights: Vec<f32>,
    pub proj_bias: Vec<f32>,
}

impl FourCastNet {
    /// Create a new FourCastNet with random weights.
    pub fn new(config: FourCastConfig) -> Self {
        let seed_base = 42_u64;
        let lift_weights = normal_samples_f32(config.embed_dim * config.n_channels, seed_base);
        let lift_bias = vec![0.0_f32; config.embed_dim];
        let layers = (0..config.n_layers)
            .map(|i| {
                SphericalFourierLayer::new(
                    config.n_modes,
                    config.embed_dim,
                    seed_base + i as u64 + 10,
                )
            })
            .collect();
        let proj_weights =
            normal_samples_f32(config.n_channels * config.embed_dim, seed_base + 100);
        let proj_bias = vec![0.0_f32; config.n_channels];
        Self {
            config,
            lift_weights,
            lift_bias,
            layers,
            proj_weights,
            proj_bias,
        }
    }

    /// Forward pass: input `x` has shape [lat × lon × n_channels] in row-major.
    /// Returns output of the same shape.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let spatial = cfg.lat_size * cfg.lon_size;
        let expected = spatial * cfg.n_channels;
        if x.len() != expected {
            return Err(TensorError::invalid_shape_simple(format!(
                "FourCastNet expects {} elements, got {}",
                expected,
                x.len()
            )));
        }
        // Lift: project n_channels → embed_dim at each grid point
        let mut h: Vec<f32> = (0..spatial)
            .flat_map(|s| {
                let in_slice = &x[s * cfg.n_channels..(s + 1) * cfg.n_channels];
                dense_linear_f32(&self.lift_weights, &self.lift_bias, in_slice, cfg.embed_dim)
            })
            .collect();
        // Apply spectral layers along lat dimension (one lon slice at a time)
        for layer in &self.layers {
            let mut new_h = vec![0.0_f32; h.len()];
            for lon in 0..cfg.lon_size {
                // Extract lat strip for this longitude (embed_dim per lat)
                let mut strip = vec![0.0_f32; cfg.lat_size * cfg.embed_dim];
                for lat in 0..cfg.lat_size {
                    let src = (lat * cfg.lon_size + lon) * cfg.embed_dim;
                    let dst = lat * cfg.embed_dim;
                    strip[dst..dst + cfg.embed_dim].copy_from_slice(&h[src..src + cfg.embed_dim]);
                }
                let out_strip = layer.forward(&strip, cfg.lat_size);
                // Write back with residual
                for lat in 0..cfg.lat_size {
                    let dst = (lat * cfg.lon_size + lon) * cfg.embed_dim;
                    let src = lat * cfg.embed_dim;
                    for c in 0..cfg.embed_dim {
                        new_h[dst + c] = h[dst + c] + relu_f32(out_strip[src + c]);
                    }
                }
            }
            h = new_h;
        }
        // Project: embed_dim → n_channels
        let output: Vec<f32> = (0..spatial)
            .flat_map(|s| {
                let in_slice = &h[s * cfg.embed_dim..(s + 1) * cfg.embed_dim];
                dense_linear_f32(
                    &self.proj_weights,
                    &self.proj_bias,
                    in_slice,
                    cfg.n_channels,
                )
            })
            .collect();
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PanGu-Weather — Hierarchical Transformer
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D relative position bias for toroidal Earth geometry.
/// The longitude wraps around; latitude does not.
#[derive(Debug, Clone)]
pub struct EarthPositionBias {
    /// Learnable bias table: [(2*lat_size-1) × (2*lon_size-1)].
    pub bias_table: Vec<f32>,
    pub lat_size: usize,
    pub lon_size: usize,
}

impl EarthPositionBias {
    pub fn new(lat_size: usize, lon_size: usize) -> Self {
        let table_size = (2 * lat_size - 1) * (2 * lon_size - 1);
        let bias_table = vec![0.0_f32; table_size];
        Self {
            bias_table,
            lat_size,
            lon_size,
        }
    }

    /// Return bias value for a pair of grid cells.
    pub fn get_bias(&self, lat1: usize, lon1: usize, lat2: usize, lon2: usize) -> f32 {
        let dlat = (lat1 as isize - lat2 as isize + self.lat_size as isize - 1) as usize;
        // Longitude wraps toroidally
        let dlon_raw = (lon1 as isize - lon2 as isize).rem_euclid(self.lon_size as isize) as usize;
        let idx = dlat * (2 * self.lon_size - 1) + dlon_raw;
        if idx < self.bias_table.len() {
            self.bias_table[idx]
        } else {
            0.0
        }
    }
}

/// Pressure-level attention block (simplified multi-head for upper/lower atmosphere).
#[derive(Debug, Clone)]
pub struct PressureLevelAttention {
    /// Weight matrices for Q/K/V projections, shape [hidden × hidden] each.
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub hidden: usize,
    pub n_heads: usize,
}

impl PressureLevelAttention {
    pub fn new(hidden: usize, n_heads: usize, seed: u64) -> Self {
        let n = hidden * hidden;
        let wq = xavier_vec(n, seed);
        let wk = xavier_vec(n, seed + 1);
        let wv = xavier_vec(n, seed + 2);
        let wo = xavier_vec(n, seed + 3);
        Self {
            wq,
            wk,
            wv,
            wo,
            hidden,
            n_heads,
        }
    }

    fn project(&self, weights: &[f32], x: &[f32]) -> Vec<f32> {
        dense_linear_f32(weights, &[], x, self.hidden)
    }

    /// Scaled dot-product attention over a sequence of tokens.
    /// `x`: [seq_len × hidden]
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden;
        let scale = 1.0 / (h as f32).sqrt();
        // Project each token
        let q: Vec<f32> = (0..seq_len)
            .flat_map(|t| self.project(&self.wq, &x[t * h..(t + 1) * h]))
            .collect();
        let k: Vec<f32> = (0..seq_len)
            .flat_map(|t| self.project(&self.wk, &x[t * h..(t + 1) * h]))
            .collect();
        let v: Vec<f32> = (0..seq_len)
            .flat_map(|t| self.project(&self.wv, &x[t * h..(t + 1) * h]))
            .collect();
        // Attention scores: [seq_len × seq_len]
        let mut attn = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f32 = (0..h).map(|d| q[i * h + d] * k[j * h + d]).sum();
                attn[i * seq_len + j] = dot * scale;
            }
        }
        // Softmax per row
        for i in 0..seq_len {
            let row = &mut attn[i * seq_len..(i + 1) * seq_len];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row
                .iter()
                .map(|&a| (a - mx).exp())
                .sum::<f32>()
                .max(f32::EPSILON);
            for a in row.iter_mut() {
                *a = (*a - mx).exp() / sum;
            }
        }
        // Weighted sum of values → output projection
        let mut ctx = vec![0.0_f32; seq_len * h];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let w = attn[i * seq_len + j];
                for d in 0..h {
                    ctx[i * h + d] += w * v[j * h + d];
                }
            }
        }
        (0..seq_len)
            .flat_map(|t| self.project(&self.wo, &ctx[t * h..(t + 1) * h]))
            .collect()
    }
}

/// PanGu hierarchical weather model (upper + surface branch).
#[derive(Debug, Clone)]
pub struct PanGuWeather {
    pub upper_attn: PressureLevelAttention,
    pub surface_attn: PressureLevelAttention,
    pub position_bias: EarthPositionBias,
    /// Layer norm params for upper.
    pub ln_gamma_u: Vec<f32>,
    pub ln_beta_u: Vec<f32>,
    /// Layer norm params for surface.
    pub ln_gamma_s: Vec<f32>,
    pub ln_beta_s: Vec<f32>,
    pub hidden: usize,
}

impl PanGuWeather {
    pub fn new(lat_size: usize, lon_size: usize, hidden: usize, n_heads: usize) -> Self {
        let upper_attn = PressureLevelAttention::new(hidden, n_heads, 7);
        let surface_attn = PressureLevelAttention::new(hidden, n_heads, 13);
        let position_bias = EarthPositionBias::new(lat_size, lon_size);
        let ln_gamma_u = vec![1.0_f32; hidden];
        let ln_beta_u = vec![0.0_f32; hidden];
        let ln_gamma_s = vec![1.0_f32; hidden];
        let ln_beta_s = vec![0.0_f32; hidden];
        Self {
            upper_attn,
            surface_attn,
            position_bias,
            ln_gamma_u,
            ln_beta_u,
            ln_gamma_s,
            ln_beta_s,
            hidden,
        }
    }

    /// Forward: processes upper atmosphere and surface fields.
    /// `upper`: [upper_seq × hidden], `surface`: [surface_seq × hidden].
    pub fn forward(&self, upper: &[f32], surface: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if upper.is_empty() || upper.len() % self.hidden != 0 {
            return Err(TensorError::invalid_argument_op(
                "PanGuWeather::forward",
                "upper tensor dim mismatch",
            ));
        }
        if surface.is_empty() || surface.len() % self.hidden != 0 {
            return Err(TensorError::invalid_argument_op(
                "PanGuWeather::forward",
                "surface tensor dim mismatch",
            ));
        }
        let upper_seq = upper.len() / self.hidden;
        let surface_seq = surface.len() / self.hidden;
        // Pre-norm + attention + residual for upper
        let upper_normed: Vec<f32> = upper
            .chunks(self.hidden)
            .flat_map(|tok| layer_norm_f32(tok, &self.ln_gamma_u, &self.ln_beta_u))
            .collect();
        let upper_attn_out = self.upper_attn.forward(&upper_normed, upper_seq);
        let upper_out: Vec<f32> = upper
            .iter()
            .zip(upper_attn_out.iter())
            .map(|(&r, &a)| r + a)
            .collect();
        // Pre-norm + attention + residual for surface
        let surface_normed: Vec<f32> = surface
            .chunks(self.hidden)
            .flat_map(|tok| layer_norm_f32(tok, &self.ln_gamma_s, &self.ln_beta_s))
            .collect();
        let surface_attn_out = self.surface_attn.forward(&surface_normed, surface_seq);
        let surface_out: Vec<f32> = surface
            .iter()
            .zip(surface_attn_out.iter())
            .map(|(&r, &a)| r + a)
            .collect();
        Ok((upper_out, surface_out))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ClimateDownscaler — Statistical downscaling via super-resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Bicubic upsampling by integer scale factor (simple bilinear approximation
/// using 4-point cubic Catmull-Rom in each axis).
#[derive(Debug, Clone)]
pub struct BicubicUpsample {
    pub scale: usize,
}

impl BicubicUpsample {
    pub fn new(scale: usize) -> Self {
        Self { scale }
    }

    /// `input`: [h × w × c] row-major. Returns [h*scale × w*scale × c].
    pub fn forward(&self, input: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        let oh = h * self.scale;
        let ow = w * self.scale;
        let mut out = vec![0.0_f32; oh * ow * c];
        for oy in 0..oh {
            for ox in 0..ow {
                // Map output pixel back to input coordinate
                let iy_f = oy as f32 / self.scale as f32;
                let ix_f = ox as f32 / self.scale as f32;
                // Bilinear (4 neighbour) as a simpler stand-in for Catmull-Rom
                let iy0 = (iy_f.floor() as usize).min(h.saturating_sub(1));
                let ix0 = (ix_f.floor() as usize).min(w.saturating_sub(1));
                let iy1 = (iy0 + 1).min(h.saturating_sub(1));
                let ix1 = (ix0 + 1).min(w.saturating_sub(1));
                let ty = iy_f - iy0 as f32;
                let tx = ix_f - ix0 as f32;
                for ch in 0..c {
                    let v00 = input[(iy0 * w + ix0) * c + ch];
                    let v01 = input[(iy0 * w + ix1) * c + ch];
                    let v10 = input[(iy1 * w + ix0) * c + ch];
                    let v11 = input[(iy1 * w + ix1) * c + ch];
                    out[(oy * ow + ox) * c + ch] = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v01)
                        + ty * ((1.0 - tx) * v10 + tx * v11);
                }
            }
        }
        out
    }
}

/// Residual Dense Block with dense (concat) connections.
#[derive(Debug, Clone)]
pub struct ResidualDenseBlock {
    pub n_feat: usize,
    pub n_dense_layers: usize,
    /// Weights for each dense layer: grows by n_feat each step.
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    /// Final compression: (n_feat * n_dense_layers) → n_feat
    pub compress_w: Vec<f32>,
    pub compress_b: Vec<f32>,
}

impl ResidualDenseBlock {
    pub fn new(n_feat: usize, n_dense_layers: usize, seed: u64) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..n_dense_layers {
            let in_f = n_feat * (l + 1);
            weights.push(normal_samples_f32(n_feat * in_f, seed + l as u64));
            biases.push(vec![0.0_f32; n_feat]);
        }
        let total_in = n_feat * n_dense_layers;
        let compress_w = normal_samples_f32(n_feat * total_in, seed + 100);
        let compress_b = vec![0.0_f32; n_feat];
        Self {
            n_feat,
            n_dense_layers,
            weights,
            biases,
            compress_w,
            compress_b,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut concat = x.to_vec(); // starts as [n_feat]
        let mut all_feats: Vec<f32> = Vec::new();
        for l in 0..self.n_dense_layers {
            let in_f = concat.len();
            let out = dense_linear_f32(&self.weights[l], &self.biases[l], &concat, self.n_feat);
            let activated: Vec<f32> = out.iter().map(|&v| relu_f32(v)).collect();
            all_feats.extend_from_slice(&activated);
            concat.extend_from_slice(&activated);
        }
        // Compress all dense outputs → n_feat, add residual
        let compressed =
            dense_linear_f32(&self.compress_w, &self.compress_b, &all_feats, self.n_feat);
        x.iter()
            .zip(compressed.iter())
            .map(|(&r, &c)| r + c)
            .collect()
    }
}

/// Climate downscaling model combining upsampling with residual dense blocks.
#[derive(Debug, Clone)]
pub struct DownscalerModel {
    pub low_res_size: usize,
    pub high_res_size: usize,
    pub n_channels: usize,
    pub upsample: BicubicUpsample,
    pub rdb: ResidualDenseBlock,
    /// Final convolution-like dense projection per pixel.
    pub final_w: Vec<f32>,
    pub final_b: Vec<f32>,
}

impl DownscalerModel {
    pub fn new(low_res_size: usize, high_res_size: usize, n_channels: usize) -> Result<Self> {
        if low_res_size == 0 || high_res_size == 0 || n_channels == 0 {
            return Err(TensorError::invalid_argument_op(
                "DownscalerModel::new",
                "sizes must be > 0",
            ));
        }
        let scale = high_res_size / low_res_size;
        if scale == 0 || low_res_size * scale != high_res_size {
            return Err(TensorError::invalid_argument_op(
                "DownscalerModel::new",
                "high_res_size must be integer multiple of low_res_size",
            ));
        }
        let upsample = BicubicUpsample::new(scale);
        let rdb = ResidualDenseBlock::new(n_channels, 4, 99);
        let final_w = normal_samples_f32(n_channels * n_channels, 200);
        let final_b = vec![0.0_f32; n_channels];
        Ok(Self {
            low_res_size,
            high_res_size,
            n_channels,
            upsample,
            rdb,
            final_w,
            final_b,
        })
    }

    /// `low_res`: [low_res_size × low_res_size × n_channels]. Returns high-res output.
    pub fn forward(&self, low_res: &[f32]) -> Result<Vec<f32>> {
        let expected = self.low_res_size * self.low_res_size * self.n_channels;
        if low_res.len() != expected {
            return Err(TensorError::invalid_shape_simple(format!(
                "DownscalerModel expects {} elements, got {}",
                expected,
                low_res.len()
            )));
        }
        let upsampled = self.upsample.forward(
            low_res,
            self.low_res_size,
            self.low_res_size,
            self.n_channels,
        );
        let hr_pixels = self.high_res_size * self.high_res_size;
        let refined: Vec<f32> = upsampled
            .chunks(self.n_channels)
            .take(hr_pixels)
            .flat_map(|pix| self.rdb.forward(pix))
            .collect();
        let output: Vec<f32> = refined
            .chunks(self.n_channels)
            .flat_map(|pix| dense_linear_f32(&self.final_w, &self.final_b, pix, self.n_channels))
            .collect();
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. ExtremeEventDetector — Rare event prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Focal loss for addressing severe class imbalance in extreme event prediction.
#[derive(Debug, Clone)]
pub struct FocalLoss {
    /// Alpha weighting for positive class.
    pub alpha: f32,
    /// Focusing parameter (gamma ≥ 0).
    pub gamma: f32,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        Self { alpha, gamma }
    }

    /// Compute focal loss for a single sample.
    /// `pred`: sigmoid probability, `target`: 0 or 1.
    pub fn focal_loss(&self, pred: f32, target: f32) -> f32 {
        let eps = 1e-7_f32;
        let p = pred.clamp(eps, 1.0 - eps);
        let (pt, at) = if target > 0.5 {
            (p, self.alpha)
        } else {
            (1.0 - p, 1.0 - self.alpha)
        };
        -at * (1.0 - pt).powf(self.gamma) * pt.ln()
    }

    /// Batch focal loss (mean over samples).
    pub fn batch_loss(&self, preds: &[f32], targets: &[f32]) -> f32 {
        if preds.is_empty() {
            return 0.0;
        }
        let total: f32 = preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| self.focal_loss(p, t))
            .sum();
        total / preds.len() as f32
    }
}

/// Oversampling strategy for rare climate extreme events.
#[derive(Debug, Clone)]
pub struct ClimateSampler {
    /// Probability threshold above which an event is considered "extreme".
    pub event_threshold: f32,
    /// Oversampling ratio for positive class.
    pub oversample_ratio: usize,
}

impl ClimateSampler {
    pub fn new(event_threshold: f32, oversample_ratio: usize) -> Self {
        Self {
            event_threshold,
            oversample_ratio,
        }
    }

    /// Given feature rows and labels, return indices with positive events
    /// oversampled. `labels`: event probability or binary flag per sample.
    pub fn sample_indices(&self, labels: &[f32], n_samples: usize) -> Vec<usize> {
        let pos_idx: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l >= self.event_threshold)
            .map(|(i, _)| i)
            .collect();
        let neg_idx: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l < self.event_threshold)
            .map(|(i, _)| i)
            .collect();
        let mut indices = neg_idx.clone();
        // Oversample positives
        for _ in 0..self.oversample_ratio {
            indices.extend_from_slice(&pos_idx);
        }
        // Truncate to n_samples
        indices.truncate(n_samples);
        indices
    }
}

/// Extreme event detection model with sigmoid output and focal loss training.
#[derive(Debug, Clone)]
pub struct ExtremeEventModel {
    /// Threshold for classifying as an extreme event.
    pub threshold: f32,
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub layer1_w: Vec<f32>,
    pub layer1_b: Vec<f32>,
    pub layer2_w: Vec<f32>,
    pub layer2_b: Vec<f32>,
    pub focal_loss: FocalLoss,
}

impl ExtremeEventModel {
    pub fn new(input_dim: usize, hidden_dim: usize, threshold: f32) -> Self {
        let layer1_w = normal_samples_f32(hidden_dim * input_dim, 55);
        let layer1_b = vec![0.0_f32; hidden_dim];
        let layer2_w = normal_samples_f32(hidden_dim, 56);
        let layer2_b = vec![0.0_f32; 1];
        Self {
            threshold,
            hidden_dim,
            input_dim,
            layer1_w,
            layer1_b,
            layer2_w,
            layer2_b,
            focal_loss: FocalLoss::new(0.25, 2.0),
        }
    }

    /// Returns probability of extreme event ∈ (0, 1).
    pub fn predict_probability(&self, features: &[f32]) -> f32 {
        let h = dense_linear_f32(&self.layer1_w, &self.layer1_b, features, self.hidden_dim);
        let h_act: Vec<f32> = h.iter().map(|&v| relu_f32(v)).collect();
        let logit = dense_linear_f32(&self.layer2_w, &self.layer2_b, &h_act, 1);
        sigmoid_f32(logit[0])
    }

    pub fn predict(&self, features: &[f32]) -> bool {
        self.predict_probability(features) >= self.threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. AtmosphericEmbedding — Multi-variable atmospheric state encoder
// ─────────────────────────────────────────────────────────────────────────────

/// Learnable embedding for a single atmospheric variable.
#[derive(Debug, Clone)]
pub struct VariableEmbedding {
    pub var_name: String,
    pub embed_dim: usize,
    /// Weight vector \[embed_dim\], projects scalar → embedding.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl VariableEmbedding {
    pub fn new(var_name: String, embed_dim: usize, seed: u64) -> Self {
        let weights = xavier_vec(embed_dim, seed);
        let bias = vec![0.0_f32; embed_dim];
        Self {
            var_name,
            embed_dim,
            weights,
            bias,
        }
    }

    pub fn embed(&self, value: f32) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(&w, &b)| value * w + b)
            .collect()
    }
}

/// Encoder that combines multiple atmospheric variable embeddings.
#[derive(Debug, Clone)]
pub struct AtmosphericEncoder {
    pub variables: Vec<VariableEmbedding>,
    /// Position embedding weights: \[embed_dim\] (added after summation).
    pub pos_embedding: Vec<f32>,
}

impl AtmosphericEncoder {
    pub fn new(variable_names: &[&str], embed_dim: usize) -> Self {
        let variables = variable_names
            .iter()
            .enumerate()
            .map(|(i, &name)| VariableEmbedding::new(name.to_string(), embed_dim, 1000 + i as u64))
            .collect();
        let pos_embedding = vec![0.0_f32; embed_dim];
        Self {
            variables,
            pos_embedding,
        }
    }

    /// Encodes a set of named variable values into a single embedding vector.
    /// Unknown variables are ignored; missing ones contribute zero.
    pub fn encode(&self, variable_values: &[(String, f32)]) -> Vec<f32> {
        let embed_dim = self.variables.first().map(|v| v.embed_dim).unwrap_or(0);
        if embed_dim == 0 {
            return Vec::new();
        }
        let mut result = self.pos_embedding.clone();
        for ve in &self.variables {
            if let Some((_, val)) = variable_values
                .iter()
                .find(|(name, _)| name == &ve.var_name)
            {
                let emb = ve.embed(*val);
                for (r, e) in result.iter_mut().zip(emb.iter()) {
                    *r += e;
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. OceanCurrentPredictor — LSTM-based ocean dynamics
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for ocean LSTM.
#[derive(Debug, Clone)]
pub struct OceanLstmConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_steps_ahead: usize,
}

/// A single LSTM layer with all gate parameters.
#[derive(Debug, Clone)]
pub struct LstmLayer {
    // Input gate
    pub w_i: Vec<f32>,
    pub b_i: Vec<f32>,
    // Forget gate
    pub w_f: Vec<f32>,
    pub b_f: Vec<f32>,
    // Cell gate
    pub w_g: Vec<f32>,
    pub b_g: Vec<f32>,
    // Output gate
    pub w_o: Vec<f32>,
    pub b_o: Vec<f32>,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

impl LstmLayer {
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let param_size = hidden_dim * (input_dim + hidden_dim);
        Self {
            w_i: normal_samples_f32(param_size, seed),
            b_i: vec![0.0_f32; hidden_dim],
            w_f: normal_samples_f32(param_size, seed + 1),
            b_f: vec![1.0_f32; hidden_dim], // forget gate bias = 1
            w_g: normal_samples_f32(param_size, seed + 2),
            b_g: vec![0.0_f32; hidden_dim],
            w_o: normal_samples_f32(param_size, seed + 3),
            b_o: vec![0.0_f32; hidden_dim],
            input_dim,
            hidden_dim,
        }
    }

    /// Single LSTM cell step. Returns (new_h, new_c).
    pub fn step(&self, x: &[f32], h: &[f32], c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let hd = self.hidden_dim;
        // Concatenate input and hidden state
        let mut xh = Vec::with_capacity(x.len() + h.len());
        xh.extend_from_slice(x);
        xh.extend_from_slice(h);
        let gate_i: Vec<f32> = dense_linear_f32(&self.w_i, &self.b_i, &xh, hd)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let gate_f: Vec<f32> = dense_linear_f32(&self.w_f, &self.b_f, &xh, hd)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let gate_g: Vec<f32> = dense_linear_f32(&self.w_g, &self.b_g, &xh, hd)
            .iter()
            .map(|&v| tanh_f32(v))
            .collect();
        let gate_o: Vec<f32> = dense_linear_f32(&self.w_o, &self.b_o, &xh, hd)
            .iter()
            .map(|&v| sigmoid_f32(v))
            .collect();
        let new_c: Vec<f32> = (0..hd)
            .map(|i| gate_f[i] * c[i] + gate_i[i] * gate_g[i])
            .collect();
        let new_h: Vec<f32> = (0..hd).map(|i| gate_o[i] * tanh_f32(new_c[i])).collect();
        (new_h, new_c)
    }
}

/// Multi-layer LSTM for ocean current prediction.
#[derive(Debug, Clone)]
pub struct OceanLstm {
    pub layers: Vec<LstmLayer>,
    pub config: OceanLstmConfig,
    /// Output projection: [n_steps_ahead × input_dim] × hidden_dim
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

impl OceanLstm {
    pub fn new(config: OceanLstmConfig) -> Self {
        let layers: Vec<LstmLayer> = (0..config.n_layers)
            .map(|i| {
                let in_d = if i == 0 {
                    config.input_dim
                } else {
                    config.hidden_dim
                };
                LstmLayer::new(in_d, config.hidden_dim, 300 + i as u64 * 10)
            })
            .collect();
        let out_dim = config.n_steps_ahead * config.input_dim;
        let out_w = normal_samples_f32(out_dim * config.hidden_dim, 400);
        let out_b = vec![0.0_f32; out_dim];
        Self {
            layers,
            config,
            out_w,
            out_b,
        }
    }

    /// `x`: [seq_len × input_dim] row-major. Returns [n_steps_ahead × input_dim].
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let id = self.config.input_dim;
        if x.len() != seq_len * id {
            return Err(TensorError::invalid_argument_op(
                "OceanLstm::forward",
                "x length mismatch",
            ));
        }
        let hd = self.config.hidden_dim;
        // Initialize hidden states
        let mut hs: Vec<Vec<f32>> = (0..self.config.n_layers)
            .map(|_| vec![0.0_f32; hd])
            .collect();
        let mut cs: Vec<Vec<f32>> = (0..self.config.n_layers)
            .map(|_| vec![0.0_f32; hd])
            .collect();
        for t in 0..seq_len {
            let mut layer_in = x[t * id..(t + 1) * id].to_vec();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (new_h, new_c) = layer.step(&layer_in, &hs[layer_idx], &cs[layer_idx]);
                hs[layer_idx] = new_h.clone();
                cs[layer_idx] = new_c;
                layer_in = new_h;
            }
        }
        // Project last hidden state to multi-step prediction
        let out_dim = self.config.n_steps_ahead * id;
        let output = dense_linear_f32(
            &self.out_w,
            &self.out_b,
            hs.last().unwrap_or(&vec![]),
            out_dim,
        );
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CarbonFluxEstimator — Process-based ML hybrid
// ─────────────────────────────────────────────────────────────────────────────

/// Ecosystem input features for carbon flux estimation.
#[derive(Debug, Clone)]
pub struct EcosystemFeatures {
    /// Leaf Area Index [m²/m²].
    pub lai: f32,
    /// Volumetric soil moisture [m³/m³].
    pub soil_moisture: f32,
    /// Air temperature [°C].
    pub air_temp: f32,
    /// Vapour Pressure Deficit \[kPa\].
    pub vpd: f32,
    /// Photosynthetically Active Radiation [W/m²].
    pub par: f32,
}

/// Photosynthesis model with Michaelis-Menten kinetics and ML-corrected parameters.
#[derive(Debug, Clone)]
pub struct PhotosynthesisModel {
    /// Maximum carboxylation capacity [μmol CO₂/m²/s].
    pub vcmax_base: f32,
    /// Half-saturation constant for light [W/m²].
    pub km_light: f32,
    /// ML correction network weights (small MLP on [lai, vpd, temp] → correction factor).
    pub corr_w1: Vec<f32>,
    pub corr_b1: Vec<f32>,
    pub corr_w2: Vec<f32>,
    pub corr_b2: Vec<f32>,
}

impl PhotosynthesisModel {
    pub fn new() -> Self {
        Self {
            vcmax_base: 50.0,
            km_light: 200.0,
            corr_w1: normal_samples_f32(8 * 3, 500),
            corr_b1: vec![0.0_f32; 8],
            corr_w2: normal_samples_f32(8, 501),
            corr_b2: vec![0.1_f32; 1],
        }
    }

    /// Estimate GPP [g C/m²/day] using Michaelis-Menten + ML correction.
    pub fn estimate_gpp(&self, f: &EcosystemFeatures) -> f32 {
        // Michaelis-Menten light response
        let par_response = f.par / (f.par + self.km_light);
        // Temperature scaling (optimal around 25°C)
        let temp_scale = (1.0 - ((f.air_temp - 25.0) / 15.0).powi(2)).max(0.0);
        // Water stress
        let water_scale = f.soil_moisture.clamp(0.0, 1.0);
        // Base GPP
        let gpp_base = self.vcmax_base * par_response * temp_scale * water_scale * f.lai;
        // ML correction
        let features = [f.lai, f.vpd, f.air_temp];
        let h = dense_linear_f32(&self.corr_w1, &self.corr_b1, &features, 8);
        let h_act: Vec<f32> = h.iter().map(|&v| relu_f32(v)).collect();
        let corr = dense_linear_f32(&self.corr_w2, &self.corr_b2, &h_act, 1);
        let correction = 1.0 + 0.1 * sigmoid_f32(corr[0]);
        gpp_base * correction
    }
}

impl Default for PhotosynthesisModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Respiration model using Q10 temperature sensitivity.
#[derive(Debug, Clone)]
pub struct RespirationModel {
    /// Base respiration rate at 10°C [g C/m²/day].
    pub r_base: f32,
    /// Q10 coefficient (typically 2.0).
    pub q10: f32,
    /// Reference temperature [°C].
    pub t_ref: f32,
}

impl RespirationModel {
    pub fn new() -> Self {
        Self {
            r_base: 2.0,
            q10: 2.0,
            t_ref: 10.0,
        }
    }

    /// Estimate ecosystem respiration [g C/m²/day].
    pub fn estimate_respiration(&self, f: &EcosystemFeatures) -> f32 {
        let exp = (f.air_temp - self.t_ref) / 10.0;
        let q10_scale = self.q10.powf(exp);
        let moisture_effect = (f.soil_moisture * 2.0).min(1.0);
        self.r_base * q10_scale * moisture_effect
    }
}

impl Default for RespirationModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined carbon flux estimator: GPP and ecosystem respiration.
#[derive(Debug, Clone)]
pub struct CarbonFluxEstimator {
    pub photosynthesis: PhotosynthesisModel,
    pub respiration: RespirationModel,
}

impl CarbonFluxEstimator {
    pub fn new() -> Self {
        Self {
            photosynthesis: PhotosynthesisModel::new(),
            respiration: RespirationModel::new(),
        }
    }

    /// Predict (GPP, respiration) for given ecosystem features.
    pub fn predict(&self, features: &EcosystemFeatures) -> (f32, f32) {
        let gpp = self.photosynthesis.estimate_gpp(features);
        let resp = self.respiration.estimate_respiration(features);
        (gpp, resp)
    }
}

impl Default for CarbonFluxEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. ClimateProjectionEnsemble — Multi-model ensemble with uncertainty
// ─────────────────────────────────────────────────────────────────────────────

/// A single linear climate projection model.
#[derive(Debug, Clone)]
pub struct ClimateModel {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl ClimateModel {
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let weights = normal_samples_f32(output_dim * input_dim, seed);
        let bias = vec![0.0_f32; output_dim];
        Self {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        dense_linear_f32(&self.weights, &self.bias, x, self.output_dim)
    }
}

/// Ensemble of climate projection models.
#[derive(Debug, Clone)]
pub struct ProjectionEnsemble {
    pub models: Vec<ClimateModel>,
}

impl ProjectionEnsemble {
    pub fn new(n_models: usize, input_dim: usize, output_dim: usize) -> Self {
        let models = (0..n_models)
            .map(|i| ClimateModel::new(input_dim, output_dim, 700 + i as u64 * 17))
            .collect();
        Self { models }
    }

    /// Returns (mean_projection, std_projection) across the ensemble.
    pub fn predict(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.models.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ProjectionEnsemble::predict",
                "no models in ensemble",
            ));
        }
        let out_dim = self.models[0].output_dim;
        let preds: Vec<Vec<f32>> = self.models.iter().map(|m| m.predict(x)).collect();
        let mut mean = vec![0.0_f32; out_dim];
        for p in &preds {
            for (m, &v) in mean.iter_mut().zip(p.iter()) {
                *m += v;
            }
        }
        let n = preds.len() as f32;
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut var = vec![0.0_f32; out_dim];
        for p in &preds {
            for (v, (&val, &mu)) in var.iter_mut().zip(p.iter().zip(mean.iter())) {
                *v += (val - mu).powi(2);
            }
        }
        let std: Vec<f32> = var.iter().map(|&v| (v / n).sqrt()).collect();
        Ok((mean, std))
    }

    /// Compute weighted ensemble mean.
    pub fn weighted_ensemble(&self, x: &[f32], weights: &[f32]) -> Vec<f32> {
        if self.models.is_empty() {
            return Vec::new();
        }
        let out_dim = self.models[0].output_dim;
        let wsum: f32 = weights.iter().sum::<f32>().max(f32::EPSILON);
        let mut result = vec![0.0_f32; out_dim];
        for (m, &w) in self.models.iter().zip(weights.iter()) {
            let pred = m.predict(x);
            for (r, &v) in result.iter_mut().zip(pred.iter()) {
                *r += (w / wsum) * v;
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. TeleconnectionAnalyzer — Remote climate pattern detection
// ─────────────────────────────────────────────────────────────────────────────

/// Teleconnection index: projects a climate field onto a fixed pattern.
#[derive(Debug, Clone)]
pub struct TeleconnectionIndex {
    /// Spatial pattern weights (e.g. ENSO loading pattern).
    pub pattern_weights: Vec<f32>,
    pub name: String,
}

impl TeleconnectionIndex {
    pub fn new(name: String, pattern_weights: Vec<f32>) -> Self {
        Self {
            pattern_weights,
            name,
        }
    }

    /// Compute the index as the inner product of the field with the pattern.
    pub fn compute_index(&self, field: &[f32]) -> f32 {
        field
            .iter()
            .zip(self.pattern_weights.iter())
            .map(|(&f, &w)| f * w)
            .sum()
    }
}

/// Empirical Orthogonal Function decomposition (EOF/PCA).
#[derive(Debug, Clone)]
pub struct EmpiricalOrthogonalFunction {
    pub n_modes: usize,
    /// EOF vectors [n_modes × n_spatial].
    pub eigenvectors: Vec<Vec<f32>>,
    /// Explained variance for each mode.
    pub eigenvalues: Vec<f32>,
}

impl EmpiricalOrthogonalFunction {
    /// Construct with provided eigenvectors and eigenvalues.
    pub fn new(n_modes: usize, eigenvectors: Vec<Vec<f32>>, eigenvalues: Vec<f32>) -> Result<Self> {
        if eigenvectors.len() != n_modes || eigenvalues.len() != n_modes {
            return Err(TensorError::invalid_argument_op(
                "EmpiricalOrthogonalFunction::new",
                "eigenvectors/eigenvalues length must equal n_modes",
            ));
        }
        Ok(Self {
            n_modes,
            eigenvectors,
            eigenvalues,
        })
    }

    /// Compute empirical eigenvectors from data using power iteration.
    /// `data`: [n_time × n_spatial]. Returns an EOF with `n_modes` leading modes.
    pub fn fit_from_data(data: &[Vec<f32>], n_modes: usize) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "EmpiricalOrthogonalFunction::fit_from_data",
                "empty data",
            ));
        }
        let n_spatial = data[0].len();
        if n_spatial == 0 {
            return Err(TensorError::invalid_argument_op(
                "EmpiricalOrthogonalFunction::fit_from_data",
                "zero spatial dimension",
            ));
        }
        let n_time = data.len();
        // Compute anomalies (remove temporal mean)
        let mean: Vec<f32> = (0..n_spatial)
            .map(|s| data.iter().map(|t| t[s]).sum::<f32>() / n_time as f32)
            .collect();
        let anom: Vec<Vec<f32>> = data
            .iter()
            .map(|t| t.iter().zip(mean.iter()).map(|(&v, &m)| v - m).collect())
            .collect();
        // Covariance matrix C = (1/T) * A^T * A  [n_spatial × n_spatial]
        // Use power iteration to find leading n_modes eigenvectors
        let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(n_modes);
        let mut eigenvalues: Vec<f32> = Vec::with_capacity(n_modes);
        let mut deflated = anom.clone();
        let mut rng = StdRng::seed_from_u64(999);
        for _mode in 0..n_modes.min(n_spatial) {
            // Random initialisation
            let mut v: Vec<f32> = (0..n_spatial)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect();
            // Power iteration
            for _ in 0..50 {
                // Av = A * v  [n_time]
                let av: Vec<f32> = deflated
                    .iter()
                    .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f32>())
                    .collect();
                // A^T * Av  [n_spatial]
                let mut new_v = vec![0.0_f32; n_spatial];
                for (t, &av_t) in deflated.iter().zip(av.iter()) {
                    for (nv, &a) in new_v.iter_mut().zip(t.iter()) {
                        *nv += av_t * a;
                    }
                }
                // Normalise
                let norm = new_v
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(f32::EPSILON);
                for x in new_v.iter_mut() {
                    *x /= norm;
                }
                v = new_v;
            }
            // Compute eigenvalue
            let av: Vec<f32> = deflated
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f32>())
                .collect();
            let eigenvalue = av.iter().map(|&x| x * x).sum::<f32>().sqrt() / n_time as f32;
            // Deflate: remove this mode from data
            let projections: Vec<f32> = deflated
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f32>())
                .collect();
            for (row, &proj) in deflated.iter_mut().zip(projections.iter()) {
                for (a, &vi) in row.iter_mut().zip(v.iter()) {
                    *a -= proj * vi;
                }
            }
            eigenvectors.push(v);
            eigenvalues.push(eigenvalue);
        }
        // Pad if fewer modes found
        while eigenvectors.len() < n_modes {
            eigenvectors.push(vec![0.0_f32; n_spatial]);
            eigenvalues.push(0.0);
        }
        Ok(Self {
            n_modes,
            eigenvectors,
            eigenvalues,
        })
    }

    /// Project data onto EOFs to obtain principal components.
    /// `data`: [n_time × n_spatial]. Returns [n_modes × n_time].
    pub fn compute_pcs(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.eigenvectors
            .iter()
            .map(|eof| {
                data.iter()
                    .map(|t| t.iter().zip(eof.iter()).map(|(&d, &e)| d * e).sum())
                    .collect()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. ClimateMetrics — Evaluation scores for climate predictions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the RMSE skill score (1 - RMSE/RMSE_clim).
/// A score of 1.0 is perfect; 0.0 matches climatology; negative is worse than climatology.
pub fn rmse_skill_score(pred: &[f32], obs: &[f32], climatology: &[f32]) -> f32 {
    if pred.is_empty() || pred.len() != obs.len() || pred.len() != climatology.len() {
        return f32::NAN;
    }
    let rmse_pred = (pred
        .iter()
        .zip(obs.iter())
        .map(|(&p, &o)| (p - o).powi(2))
        .sum::<f32>()
        / pred.len() as f32)
        .sqrt();
    let rmse_clim = (climatology
        .iter()
        .zip(obs.iter())
        .map(|(&c, &o)| (c - o).powi(2))
        .sum::<f32>()
        / pred.len() as f32)
        .sqrt();
    if rmse_clim.abs() < f32::EPSILON {
        return 0.0;
    }
    1.0 - rmse_pred / rmse_clim
}

/// Anomaly Correlation Coefficient (ACC) — standard metric for NWP skill.
pub fn anomaly_correlation(pred_anom: &[f32], obs_anom: &[f32]) -> f32 {
    if pred_anom.len() != obs_anom.len() || pred_anom.is_empty() {
        return f32::NAN;
    }
    let num: f32 = pred_anom
        .iter()
        .zip(obs_anom.iter())
        .map(|(&p, &o)| p * o)
        .sum();
    let denom_p: f32 = pred_anom.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let denom_o: f32 = obs_anom.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let denom = denom_p * denom_o;
    if denom < f32::EPSILON {
        return 0.0;
    }
    num / denom
}

/// Reliability diagram: returns (mean_forecast_prob, observed_frequency) per bin.
pub fn reliability_diagram(probs: &[f32], labels: &[bool], n_bins: usize) -> Vec<(f32, f32)> {
    if probs.len() != labels.len() || n_bins == 0 {
        return Vec::new();
    }
    let mut bin_prob_sum = vec![0.0_f32; n_bins];
    let mut bin_obs_sum = vec![0_u32; n_bins];
    let mut bin_count = vec![0_u32; n_bins];
    for (&p, &l) in probs.iter().zip(labels.iter()) {
        let bin = ((p * n_bins as f32).floor() as usize).min(n_bins - 1);
        bin_prob_sum[bin] += p;
        if l {
            bin_obs_sum[bin] += 1;
        }
        bin_count[bin] += 1;
    }
    (0..n_bins)
        .filter_map(|b| {
            if bin_count[b] > 0 {
                let mean_p = bin_prob_sum[b] / bin_count[b] as f32;
                let obs_freq = bin_obs_sum[b] as f32 / bin_count[b] as f32;
                Some((mean_p, obs_freq))
            } else {
                None
            }
        })
        .collect()
}

/// Brier score: mean squared error between probabilities and binary labels.
pub fn brier_score(probs: &[f32], labels: &[bool]) -> f32 {
    if probs.is_empty() || probs.len() != labels.len() {
        return f32::NAN;
    }
    let sum: f32 = probs
        .iter()
        .zip(labels.iter())
        .map(|(&p, &l)| (p - if l { 1.0 } else { 0.0 }).powi(2))
        .sum();
    sum / probs.len() as f32
}

/// Compute Continuous Ranked Probability Score (CRPS) for ensemble forecasts.
/// `ensemble_preds`: sorted ensemble members per sample. `obs`: scalar observations.
pub fn crps_ensemble(ensemble_preds: &[Vec<f32>], obs: &[f32]) -> f32 {
    if ensemble_preds.len() != obs.len() || ensemble_preds.is_empty() {
        return f32::NAN;
    }
    let mut total = 0.0_f32;
    for (preds, &y) in ensemble_preds.iter().zip(obs.iter()) {
        if preds.is_empty() {
            continue;
        }
        let m = preds.len() as f32;
        // Energy score form: E|X - y| - 0.5 * E|X - X'|
        let mae: f32 = preds.iter().map(|&x| (x - y).abs()).sum::<f32>() / m;
        let mut spread = 0.0_f32;
        for i in 0..preds.len() {
            for j in 0..preds.len() {
                spread += (preds[i] - preds[j]).abs();
            }
        }
        spread /= m * m;
        total += mae - 0.5 * spread;
    }
    total / ensemble_preds.len() as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (in separate file to satisfy the 1900-line limit)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "climate_ml/tests.rs"]
mod tests;
