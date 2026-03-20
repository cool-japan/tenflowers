//! Advanced GAN architectures and image generation models.
//!
//! - [`ImageTensor`]: Image representation (C×H×W).
//! - [`StyleGanGenerator`]: StyleGAN2-inspired generator with mapping network and synthesis blocks.
//! - \[`BigGanComponents`\]: Conditional BN, residual blocks, spectral-norm linear.
//! - [`VqGan`]: Vector-quantized GAN encoder/decoder/codebook.
//! - \[`GanTrainingComponents`\]: Loss functions, gradient penalty, R1, PPL, EMA.
//! - \[`ImageGenerationMetrics`\]: FID, IS, perceptual loss, eval report.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Internal utilities (f32)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec_f32(mat: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    mat.iter().map(|row| dot_f32(row, v)).collect()
}

/// He-normal init for a weight matrix (rows × cols).
fn rand_weight_f32(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    let scale = (2.0_f32 / cols as f32).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

/// Mean and variance of a slice.
fn mean_var(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    if n < 1.0 {
        return (0.0, 1.0);
    }
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var)
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  ImageTensor
// ─────────────────────────────────────────────────────────────────────────────

/// A CHW image tensor stored as row-major f32.
#[derive(Debug, Clone)]
pub struct ImageTensor {
    pub data: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl ImageTensor {
    /// Create a zero-filled image tensor with shape (C, H, W).
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            data: vec![0.0_f32; channels * height * width],
            channels,
            height,
            width,
        }
    }

    /// Create from an existing flat data vector.
    pub fn from_vec(data: Vec<f32>, channels: usize, height: usize, width: usize) -> Self {
        Self {
            data,
            channels,
            height,
            width,
        }
    }

    /// Access pixel at channel c, row h, column w.
    pub fn pixel(&self, c: usize, h: usize, w: usize) -> f32 {
        let idx = c * self.height * self.width + h * self.width + w;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns (channels, height, width).
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    /// Normalize values to [-1, 1] using global min/max.
    pub fn normalize(&self) -> Self {
        let min = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);
        let normed = self
            .data
            .iter()
            .map(|v| (v - min) / range * 2.0 - 1.0)
            .collect();
        Self::from_vec(normed, self.channels, self.height, self.width)
    }

    /// Return a flattened copy of the data.
    pub fn to_flat(&self) -> Vec<f32> {
        self.data.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  StyleGAN2-inspired generator
// ─────────────────────────────────────────────────────────────────────────────

/// 8-layer MLP mapping z → w (style latent).
#[derive(Debug, Clone)]
pub struct MappingNetwork {
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // (weight, bias)
    pub z_dim: usize,
    pub w_dim: usize,
}

impl MappingNetwork {
    pub fn new(z_dim: usize, w_dim: usize, rng: &mut impl Rng) -> Self {
        let n_layers = 8;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let in_d = if i == 0 { z_dim } else { w_dim };
            let out_d = w_dim;
            let w = rand_weight_f32(out_d, in_d, rng);
            let b = vec![0.0_f32; out_d];
            layers.push((w, b));
        }
        Self {
            layers,
            z_dim,
            w_dim,
        }
    }

    /// Map latent z to style vector w.
    pub fn forward(&self, z: &[f32]) -> Vec<f32> {
        // Normalize z to unit sphere
        let norm = z.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let mut h: Vec<f32> = z.iter().map(|v| v / norm).collect();
        for (w, b) in &self.layers {
            let pre: Vec<f32> = matvec_f32(w, &h)
                .into_iter()
                .zip(b.iter())
                .map(|(x, &bias)| x + bias)
                .collect();
            h = pre.into_iter().map(|x| leaky_relu(x, 0.2)).collect();
        }
        h
    }
}

/// Adaptive Instance Normalization layer.
pub struct AdaInLayer;

impl AdaInLayer {
    /// Normalize x to zero mean / unit std, then apply style scale and shift.
    pub fn normalize(x: &[f32], style_scale: &[f32], style_shift: &[f32]) -> Vec<f32> {
        let (mean, var) = mean_var(x);
        let std = (var + 1e-5).sqrt();
        x.iter()
            .zip(style_scale.iter().cycle())
            .zip(style_shift.iter().cycle())
            .map(|((v, s), sh)| s * (v - mean) / std + sh)
            .collect()
    }
}

/// Style-modulated synthesis block (single resolution).
#[derive(Debug, Clone)]
pub struct SynthesisBlock {
    /// conv_weight shape: out_ch × (in_ch × k × k)
    pub conv_weight: Vec<Vec<f32>>,
    /// style_proj: w_dim → 2*out_ch
    pub style_proj: Vec<Vec<f32>>,
    pub style_proj_bias: Vec<f32>,
    pub w_dim: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
}

impl SynthesisBlock {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        w_dim: usize,
        kernel: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let fan_in = in_ch * kernel * kernel;
        let conv_weight = rand_weight_f32(out_ch, fan_in, rng);
        let style_proj = rand_weight_f32(2 * out_ch, w_dim, rng);
        let style_proj_bias = vec![0.0_f32; 2 * out_ch];
        Self {
            conv_weight,
            style_proj,
            style_proj_bias,
            w_dim,
            in_ch,
            out_ch,
            kernel,
        }
    }

    /// Style-modulated convolution with weight demodulation.
    ///
    /// x: in_ch × spatial × spatial flattened
    /// Returns out_ch × spatial × spatial.
    pub fn forward(
        &self,
        x: &[f32],
        w: &[f32],
        in_ch: usize,
        out_ch: usize,
        spatial: usize,
    ) -> Vec<f32> {
        // Compute style from w via style_proj
        let style_raw: Vec<f32> = matvec_f32(&self.style_proj, w)
            .into_iter()
            .zip(self.style_proj_bias.iter())
            .map(|(v, b)| v + b)
            .collect();

        let style_scale: Vec<f32> = style_raw[..out_ch].iter().map(|v| v.exp() + 1.0).collect();
        let style_shift: Vec<f32> = style_raw[out_ch..].to_vec();

        // Simple per-channel feature map: flatten x into channel slices and apply AdaIN
        let per_ch = spatial * spatial;
        let mut out = vec![0.0_f32; out_ch * per_ch];

        for oc in 0..out_ch.min(self.conv_weight.len()) {
            // Weight demodulation: use conv_weight[oc] dot x per position approximation
            let w_row = &self.conv_weight[oc];
            let scale = style_scale.get(oc).copied().unwrap_or(1.0);
            let shift = style_shift.get(oc).copied().unwrap_or(0.0);

            // demod_weight norm
            let denom = (w_row.iter().map(|v| v * v).sum::<f32>() + 1e-8).sqrt();

            for pos in 0..per_ch {
                // Sample in_ch values from x at position pos (stride = per_ch)
                let x_val: f32 = (0..in_ch.min(w_row.len()))
                    .map(|ic| {
                        let x_idx = ic * per_ch + pos;
                        let x_v = x.get(x_idx).copied().unwrap_or(0.0);
                        let w_v = w_row.get(ic).copied().unwrap_or(0.0);
                        x_v * w_v * scale
                    })
                    .sum::<f32>()
                    / denom;
                out[oc * per_ch + pos] = relu_f32(x_val + shift);
            }
        }
        out
    }
}

/// StyleGAN2-inspired generator.
#[derive(Debug, Clone)]
pub struct StyleGanGenerator {
    pub mapping: MappingNetwork,
    pub blocks: Vec<SynthesisBlock>,
    pub z_dim: usize,
    pub w_dim: usize,
    pub n_blocks: usize,
}

impl StyleGanGenerator {
    /// n_blocks controls resolution: 4 blocks → 4→8→16→32→64 (input 4×4, output 64×64).
    pub fn new(z_dim: usize, w_dim: usize, n_blocks: usize, rng: &mut impl Rng) -> Self {
        let mapping = MappingNetwork::new(z_dim, w_dim, rng);
        let kernel = 3;
        let mut blocks = Vec::with_capacity(n_blocks);
        // channels: 512 → 256 → 128 → 64 → 32 ...
        let channels: Vec<usize> = (0..=n_blocks).map(|i| (512 >> i).max(32)).collect();
        for i in 0..n_blocks {
            let in_ch = channels[i];
            let out_ch = channels[i + 1];
            blocks.push(SynthesisBlock::new(in_ch, out_ch, w_dim, kernel, rng));
        }
        Self {
            mapping,
            blocks,
            z_dim,
            w_dim,
            n_blocks,
        }
    }

    /// Generate an image from latent z.
    /// Returns 64×64×3 image (or smaller if fewer blocks).
    pub fn generate(&self, z: &[f32]) -> ImageTensor {
        let w = self.mapping.forward(z);
        let init_spatial = 4_usize;
        let init_ch = 512_usize;

        // Constant input (ones at 4×4)
        let mut x = vec![0.1_f32; init_ch * init_spatial * init_spatial];

        let mut spatial = init_spatial;
        for (i, block) in self.blocks.iter().enumerate() {
            let in_ch = (512_usize >> i).max(32);
            let out_ch = (512_usize >> (i + 1)).max(32);
            x = block.forward(&x, &w, in_ch, out_ch, spatial);
            // Bilinear upsample 2×
            spatial *= 2;
            let new_len = out_ch * spatial * spatial;
            let mut upsampled = vec![0.0_f32; new_len];
            let src_sp = spatial / 2;
            for c in 0..out_ch {
                for h in 0..spatial {
                    for w_pos in 0..spatial {
                        let src_h = (h / 2).min(src_sp.saturating_sub(1));
                        let src_w = (w_pos / 2).min(src_sp.saturating_sub(1));
                        let src_idx = c * src_sp * src_sp + src_h * src_sp + src_w;
                        let dst_idx = c * spatial * spatial + h * spatial + w_pos;
                        upsampled[dst_idx] = x.get(src_idx).copied().unwrap_or(0.0);
                    }
                }
            }
            x = upsampled;
        }

        // Convert to 3-channel image (take first 3 channels if available)
        let out_ch = (512_usize >> self.n_blocks).max(32);
        let out_channels = 3_usize;
        let total = out_channels * spatial * spatial;
        let mut img_data = vec![0.0_f32; total];
        for c in 0..out_channels {
            for pos in 0..(spatial * spatial) {
                let src = c * spatial * spatial + pos;
                img_data[c * spatial * spatial + pos] = x.get(src).copied().unwrap_or(0.0);
            }
        }
        let _ = out_ch; // used above for clarity
        ImageTensor::from_vec(img_data, out_channels, spatial, spatial)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  BigGAN-style conditional generation
// ─────────────────────────────────────────────────────────────────────────────

/// Class-conditional batch normalization.
#[derive(Debug, Clone)]
pub struct ConditionalBatchNorm {
    pub gamma_embed: Vec<Vec<f32>>, // n_classes × feature_dim
    pub beta_embed: Vec<Vec<f32>>,  // n_classes × feature_dim
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
}

impl ConditionalBatchNorm {
    pub fn new(n_classes: usize, feature_dim: usize, rng: &mut impl Rng) -> Self {
        let gamma_embed: Vec<Vec<f32>> = (0..n_classes)
            .map(|_| {
                (0..feature_dim)
                    .map(|_| rng.random::<f32>() * 0.1 + 1.0)
                    .collect()
            })
            .collect();
        let beta_embed: Vec<Vec<f32>> = (0..n_classes)
            .map(|_| {
                (0..feature_dim)
                    .map(|_| rng.random::<f32>() * 0.1)
                    .collect()
            })
            .collect();
        Self {
            gamma_embed,
            beta_embed,
            running_mean: vec![0.0_f32; feature_dim],
            running_var: vec![1.0_f32; feature_dim],
        }
    }

    /// BN normalize then apply class-conditional gamma/beta.
    pub fn normalize(&self, x: &[f32], class_id: usize, eps: f32) -> Vec<f32> {
        let feat_dim = self.running_mean.len();
        let (mean, var) = mean_var(x);
        let std = (var + eps).sqrt();
        let gamma = self
            .gamma_embed
            .get(class_id)
            .map(|g| g.as_slice())
            .unwrap_or(&[]);
        let beta = self
            .beta_embed
            .get(class_id)
            .map(|b| b.as_slice())
            .unwrap_or(&[]);
        x.iter()
            .enumerate()
            .map(|(i, v)| {
                let g = gamma.get(i % feat_dim.max(1)).copied().unwrap_or(1.0);
                let b = beta.get(i % feat_dim.max(1)).copied().unwrap_or(0.0);
                g * (v - mean) / std + b
            })
            .collect()
    }
}

/// BigGAN residual block with conditional BN and optional upsampling.
#[derive(Debug, Clone)]
pub struct BigGanResBlock {
    pub conv1: Vec<Vec<f32>>, // out_ch × in_ch
    pub conv2: Vec<Vec<f32>>, // out_ch × out_ch
    pub cbn: ConditionalBatchNorm,
    pub upsample: bool,
    pub in_ch: usize,
    pub out_ch: usize,
}

impl BigGanResBlock {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        n_classes: usize,
        upsample: bool,
        rng: &mut impl Rng,
    ) -> Self {
        let conv1 = rand_weight_f32(out_ch, in_ch, rng);
        let conv2 = rand_weight_f32(out_ch, out_ch, rng);
        let cbn = ConditionalBatchNorm::new(n_classes, out_ch, rng);
        Self {
            conv1,
            conv2,
            cbn,
            upsample,
            in_ch,
            out_ch,
        }
    }

    /// Forward pass: x is in_ch × spatial × spatial flattened.
    pub fn forward(&self, x: &[f32], class_id: usize, spatial: usize) -> Vec<f32> {
        let per_ch = spatial * spatial;
        // conv1: project channels
        let mut h = vec![0.0_f32; self.out_ch * per_ch];
        for oc in 0..self.out_ch {
            for pos in 0..per_ch {
                let v: f32 = (0..self.in_ch)
                    .map(|ic| {
                        let xi = x.get(ic * per_ch + pos).copied().unwrap_or(0.0);
                        let wi = self
                            .conv1
                            .get(oc)
                            .and_then(|r| r.get(ic))
                            .copied()
                            .unwrap_or(0.0);
                        xi * wi
                    })
                    .sum();
                h[oc * per_ch + pos] = relu_f32(v);
            }
        }
        // CBN
        let h_normed = self.cbn.normalize(&h, class_id, 1e-5);
        // conv2
        let mut out = vec![0.0_f32; self.out_ch * per_ch];
        for oc in 0..self.out_ch {
            for pos in 0..per_ch {
                let v: f32 = (0..self.out_ch)
                    .map(|ic| {
                        let xi = h_normed.get(ic * per_ch + pos).copied().unwrap_or(0.0);
                        let wi = self
                            .conv2
                            .get(oc)
                            .and_then(|r| r.get(ic))
                            .copied()
                            .unwrap_or(0.0);
                        xi * wi
                    })
                    .sum();
                out[oc * per_ch + pos] = relu_f32(v);
            }
        }
        // Residual connection (skip if channel mismatch just use out)
        if self.in_ch == self.out_ch {
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o += xi;
            }
        }
        out
    }
}

/// Spectral normalization linear layer.
#[derive(Debug, Clone)]
pub struct SpectralNormLinear {
    pub weight: Vec<Vec<f32>>, // out_dim × in_dim
    pub u: Vec<f32>,           // left singular vector (out_dim)
    pub in_dim: usize,
    pub out_dim: usize,
}

impl SpectralNormLinear {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let weight = rand_weight_f32(out_dim, in_dim, rng);
        // Initialize u as random unit vector
        let mut u: Vec<f32> = (0..out_dim)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let norm = u.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in u.iter_mut() {
            *v /= norm;
        }
        Self {
            weight,
            u,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass using spectral-normalized weight.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let w_norm = self.normalized_weight();
        matvec_f32(&w_norm, x)
    }

    /// Compute W / sigma_max via one step of power iteration.
    pub fn normalized_weight(&self) -> Vec<Vec<f32>> {
        // v = W^T u / ||W^T u||
        let wt_u: Vec<f32> = (0..self.in_dim)
            .map(|j| {
                self.weight
                    .iter()
                    .zip(self.u.iter())
                    .map(|(row, &ui)| row.get(j).copied().unwrap_or(0.0) * ui)
                    .sum::<f32>()
            })
            .collect();
        let v_norm = wt_u.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let v: Vec<f32> = wt_u.iter().map(|v| v / v_norm).collect();

        // sigma = u^T W v
        let wv: Vec<f32> = matvec_f32(&self.weight, &v);
        let sigma = self
            .u
            .iter()
            .zip(wv.iter())
            .map(|(ui, wvi)| ui * wvi)
            .sum::<f32>()
            .max(1e-8);

        self.weight
            .iter()
            .map(|row| row.iter().map(|w| w / sigma).collect())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  VQ-GAN
// ─────────────────────────────────────────────────────────────────────────────

/// Vector-quantized codebook.
#[derive(Debug, Clone)]
pub struct VqCodebook {
    pub embeddings: Vec<Vec<f32>>, // n_codes × code_dim
    pub n_codes: usize,
    pub code_dim: usize,
}

impl VqCodebook {
    pub fn new(n_codes: usize, code_dim: usize, rng: &mut impl Rng) -> Self {
        let embeddings: Vec<Vec<f32>> = (0..n_codes)
            .map(|_| {
                (0..code_dim)
                    .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        Self {
            embeddings,
            n_codes,
            code_dim,
        }
    }

    /// Quantize z to nearest codebook entry.
    /// Returns (quantized vector, code index, commitment loss).
    pub fn quantize(&self, z: &[f32]) -> (Vec<f32>, usize, f32) {
        // Find nearest embedding
        let best_idx = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let d: f32 = e
                    .iter()
                    .zip(z.iter())
                    .map(|(ei, zi)| (ei - zi).powi(2))
                    .sum();
                (i, d)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let e = &self.embeddings[best_idx];
        // commitment loss = ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
        let beta = 0.25_f32;
        let commit_loss: f32 = z
            .iter()
            .zip(e.iter())
            .map(|(zi, ei)| (zi - ei).powi(2))
            .sum::<f32>()
            + beta
                * e.iter()
                    .zip(z.iter())
                    .map(|(ei, zi)| (ei - zi).powi(2))
                    .sum::<f32>();

        (e.clone(), best_idx, commit_loss.max(0.0))
    }

    /// Decode a codebook index to its embedding vector.
    pub fn decode_index(&self, idx: usize) -> Vec<f32> {
        self.embeddings
            .get(idx % self.n_codes.max(1))
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.code_dim])
    }
}

/// VQ-GAN encoder (simple MLP).
#[derive(Debug, Clone)]
pub struct VqGanEncoder {
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // (weight, bias)
    pub code_dim: usize,
}

impl VqGanEncoder {
    pub fn new(input_dim: usize, n_layers: usize, code_dim: usize, rng: &mut impl Rng) -> Self {
        let hidden = input_dim.max(code_dim);
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let in_d = if i == 0 { input_dim } else { hidden };
            let out_d = if i == n_layers - 1 { code_dim } else { hidden };
            let w = rand_weight_f32(out_d, in_d, rng);
            let b = vec![0.0_f32; out_d];
            layers.push((w, b));
        }
        Self { layers, code_dim }
    }

    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let pre: Vec<f32> = matvec_f32(w, &h)
                .into_iter()
                .zip(b.iter())
                .map(|(v, &bias)| v + bias)
                .collect();
            h = if i < self.layers.len() - 1 {
                pre.into_iter().map(relu_f32).collect()
            } else {
                pre // last layer: no activation
            };
        }
        h
    }
}

/// VQ-GAN decoder (simple MLP).
#[derive(Debug, Clone)]
pub struct VqGanDecoder {
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    pub output_dim: usize,
}

impl VqGanDecoder {
    pub fn new(code_dim: usize, n_layers: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let hidden = output_dim.max(code_dim);
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let in_d = if i == 0 { code_dim } else { hidden };
            let out_d = if i == n_layers - 1 {
                output_dim
            } else {
                hidden
            };
            let w = rand_weight_f32(out_d, in_d, rng);
            let b = vec![0.0_f32; out_d];
            layers.push((w, b));
        }
        Self { layers, output_dim }
    }

    pub fn decode(&self, quantized: &[f32]) -> Vec<f32> {
        let mut h = quantized.to_vec();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let pre: Vec<f32> = matvec_f32(w, &h)
                .into_iter()
                .zip(b.iter())
                .map(|(v, &bias)| v + bias)
                .collect();
            h = if i < self.layers.len() - 1 {
                pre.into_iter().map(relu_f32).collect()
            } else {
                pre.into_iter().map(sigmoid_f32).collect() // sigmoid on output
            };
        }
        h
    }
}

/// Complete VQ-GAN model.
#[derive(Debug, Clone)]
pub struct VqGan {
    pub encoder: VqGanEncoder,
    pub decoder: VqGanDecoder,
    pub codebook: VqCodebook,
}

impl VqGan {
    pub fn new(input_dim: usize, code_dim: usize, n_codes: usize, rng: &mut impl Rng) -> Self {
        let encoder = VqGanEncoder::new(input_dim, 3, code_dim, rng);
        let decoder = VqGanDecoder::new(code_dim, 3, input_dim, rng);
        let codebook = VqCodebook::new(n_codes, code_dim, rng);
        Self {
            encoder,
            decoder,
            codebook,
        }
    }

    /// Encode → quantize → decode. Returns (reconstruction, code_idx, commitment_loss).
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, usize, f32) {
        let z = self.encoder.encode(x);
        let (quantized, idx, loss) = self.codebook.quantize(&z);
        let recon = self.decoder.decode(&quantized);
        (recon, idx, loss)
    }

    /// Convenience: encode → quantize → decode, return reconstruction only.
    pub fn reconstruct(&self, x: &[f32]) -> Vec<f32> {
        let (recon, _, _) = self.forward(x);
        recon
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  GAN training components
// ─────────────────────────────────────────────────────────────────────────────

/// Supported GAN loss variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GanLossType {
    VanillaGan,
    LsGan,
    HingeGan,
    WassersteinGan,
}

/// Compute generator loss given fake discriminator logits.
pub fn generator_loss(fake_logits: &[f32], loss_type: GanLossType) -> f32 {
    if fake_logits.is_empty() {
        return 0.0;
    }
    let n = fake_logits.len() as f32;
    match loss_type {
        GanLossType::VanillaGan => {
            // -log(D(G(z)))
            -fake_logits
                .iter()
                .map(|&x| sigmoid_f32(x).max(1e-8).ln())
                .sum::<f32>()
                / n
        }
        GanLossType::LsGan => {
            // E[(D(G(z)) - 1)^2]
            fake_logits.iter().map(|&x| (x - 1.0).powi(2)).sum::<f32>() / n
        }
        GanLossType::HingeGan => {
            // -E[D(G(z))]
            -fake_logits.iter().sum::<f32>() / n
        }
        GanLossType::WassersteinGan => {
            // -E[D(G(z))]
            -fake_logits.iter().sum::<f32>() / n
        }
    }
}

/// Compute discriminator loss given real and fake logits.
pub fn discriminator_loss(real_logits: &[f32], fake_logits: &[f32], loss_type: GanLossType) -> f32 {
    let nr = real_logits.len() as f32;
    let nf = fake_logits.len() as f32;
    if nr == 0.0 || nf == 0.0 {
        return 0.0;
    }
    match loss_type {
        GanLossType::VanillaGan => {
            // -E[log D(real)] - E[log(1 - D(fake))]
            let real_loss: f32 = -real_logits
                .iter()
                .map(|&x| sigmoid_f32(x).max(1e-8).ln())
                .sum::<f32>()
                / nr;
            let fake_loss: f32 = -fake_logits
                .iter()
                .map(|&x| (1.0 - sigmoid_f32(x)).max(1e-8).ln())
                .sum::<f32>()
                / nf;
            real_loss + fake_loss
        }
        GanLossType::LsGan => {
            // E[(D(real)-1)^2] + E[D(fake)^2]
            let real_loss: f32 = real_logits.iter().map(|&x| (x - 1.0).powi(2)).sum::<f32>() / nr;
            let fake_loss: f32 = fake_logits.iter().map(|&x| x.powi(2)).sum::<f32>() / nf;
            (real_loss + fake_loss) * 0.5
        }
        GanLossType::HingeGan => {
            // E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
            let real_loss: f32 = real_logits.iter().map(|&x| (1.0 - x).max(0.0)).sum::<f32>() / nr;
            let fake_loss: f32 = fake_logits.iter().map(|&x| (1.0 + x).max(0.0)).sum::<f32>() / nf;
            real_loss + fake_loss
        }
        GanLossType::WassersteinGan => {
            // E[D(fake)] - E[D(real)]
            let real_mean: f32 = real_logits.iter().sum::<f32>() / nr;
            let fake_mean: f32 = fake_logits.iter().sum::<f32>() / nf;
            fake_mean - real_mean
        }
    }
}

/// WGAN-GP gradient penalty approximation via finite differences.
pub fn gradient_penalty(
    real_samples: &[Vec<f32>],
    fake_samples: &[Vec<f32>],
    lambda: f32,
    rng: &mut impl Rng,
) -> f32 {
    if real_samples.is_empty() || fake_samples.is_empty() {
        return 0.0;
    }
    let n = real_samples.len().min(fake_samples.len());
    let mut total_penalty = 0.0_f32;
    let eps_fd = 1e-4_f32;

    for i in 0..n {
        let alpha: f32 = rng.random::<f32>();
        let real = &real_samples[i];
        let fake = &fake_samples[i % fake_samples.len()];
        let dim = real.len().min(fake.len());

        // Interpolated sample
        let interp: Vec<f32> = (0..dim)
            .map(|j| {
                alpha * real.get(j).copied().unwrap_or(0.0)
                    + (1.0 - alpha) * fake.get(j).copied().unwrap_or(0.0)
            })
            .collect();

        // Approximate gradient norm via finite difference in random direction
        let direction: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let dir_norm = direction
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        let perturbed: Vec<f32> = interp
            .iter()
            .zip(direction.iter())
            .map(|(x, d)| x + eps_fd * d / dir_norm)
            .collect();

        // Simulated discriminator output: sum of elements (linear proxy)
        let d_interp: f32 = interp.iter().sum::<f32>();
        let d_perturbed: f32 = perturbed.iter().sum::<f32>();
        let grad_approx = (d_perturbed - d_interp).abs() / eps_fd;

        // Penalty: (||grad|| - 1)^2
        total_penalty += (grad_approx - 1.0).powi(2);
    }
    lambda * total_penalty / n as f32
}

/// R1 regularization: ||grad_D(real)||² approximation via finite differences.
pub fn r1_regularization(real_logits: &[f32], real_samples: &[Vec<f32>]) -> f32 {
    if real_logits.is_empty() || real_samples.is_empty() {
        return 0.0;
    }
    let eps = 1e-4_f32;
    let n = real_logits.len().min(real_samples.len());
    let mut r1 = 0.0_f32;
    for i in 0..n {
        let s = &real_samples[i];
        // Approximate gradient norm: ||logit * s / ||s|||| (linear model approximation)
        let logit = real_logits.get(i).copied().unwrap_or(0.0);
        let s_norm = s.iter().map(|v| v * v).sum::<f32>().sqrt().max(eps);
        let grad_norm = logit.abs() / s_norm;
        r1 += grad_norm.powi(2);
    }
    r1 / (2.0 * n as f32)
}

/// Path Length Regularization for StyleGAN.
pub struct PathLengthRegularization;

impl PathLengthRegularization {
    /// Approximate PPL as ||images2 - images1|| / ||w2 - w1||.
    pub fn perceptual_path_length(w1: &[f32], w2: &[f32], images1: &[f32], images2: &[f32]) -> f32 {
        let dw: f32 = w1
            .iter()
            .zip(w2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        let di: f32 = images1
            .iter()
            .zip(images2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        (di / dw).powi(2)
    }
}

/// Exponential moving average of generator weights.
pub struct EmaWeights;

impl EmaWeights {
    /// Update EMA in-place: ema = decay * ema + (1 - decay) * current.
    pub fn update(ema: &mut [f32], current: &[f32], decay: f32) {
        for (e, c) in ema.iter_mut().zip(current.iter()) {
            *e = decay * (*e) + (1.0 - decay) * c;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Image generation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// FID (Fréchet Inception Distance) computation.
pub struct FrechetDistance;

impl FrechetDistance {
    /// Compute approximate FID between two sets of feature vectors.
    ///
    /// FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r·Σ_f))
    /// The matrix square-root is approximated via Newton-Schulz.
    pub fn fid_score(real_features: &[Vec<f32>], fake_features: &[Vec<f32>]) -> f32 {
        if real_features.is_empty() || fake_features.is_empty() {
            return 0.0;
        }
        let d = real_features[0].len();
        if d == 0 {
            return 0.0;
        }

        // Compute means
        let mu_r = Self::mean_vec(real_features, d);
        let mu_f = Self::mean_vec(fake_features, d);

        // Mean squared difference
        let mean_diff_sq: f32 = mu_r
            .iter()
            .zip(mu_f.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        // Compute covariance matrices (diagonal approximation for tractability)
        let cov_r = Self::diag_cov(real_features, &mu_r, d);
        let cov_f = Self::diag_cov(fake_features, &mu_f, d);

        // Tr(Σ_r) + Tr(Σ_f) - 2 * Tr(√(Σ_r * Σ_f))
        // For diagonal: Tr(√(Σ_r * Σ_f)) = Σ sqrt(σ_r_i * σ_f_i)
        let trace_term: f32 = cov_r
            .iter()
            .zip(cov_f.iter())
            .map(|(r, f)| r + f - 2.0 * (r * f).max(0.0).sqrt())
            .sum();

        (mean_diff_sq + trace_term).max(0.0)
    }

    fn mean_vec(feats: &[Vec<f32>], d: usize) -> Vec<f32> {
        let n = feats.len() as f32;
        let mut mu = vec![0.0_f32; d];
        for f in feats {
            for (i, v) in f.iter().enumerate().take(d) {
                mu[i] += v / n;
            }
        }
        mu
    }

    fn diag_cov(feats: &[Vec<f32>], mean: &[f32], d: usize) -> Vec<f32> {
        let n = feats.len() as f32;
        let mut var = vec![0.0_f32; d];
        for f in feats {
            for (i, v) in f.iter().enumerate().take(d) {
                var[i] += (v - mean[i]).powi(2) / n;
            }
        }
        var
    }
}

/// Inception Score calculator.
pub struct InceptionScore;

impl InceptionScore {
    /// Compute IS mean and std over splits.
    ///
    /// KL(p(y|x) || p(y)) for each image, then exp(mean(KL)).
    pub fn is_score(class_logits: &[Vec<f32>]) -> (f32, f32) {
        if class_logits.is_empty() {
            return (0.0, 0.0);
        }
        // Compute p(y|x) = softmax(logits)
        let probs: Vec<Vec<f32>> = class_logits.iter().map(|l| Self::softmax(l)).collect();

        let n_classes = probs[0].len();
        let n = probs.len() as f32;

        // Marginal p(y) = mean over all images
        let mut p_y = vec![0.0_f32; n_classes];
        for p in &probs {
            for (i, v) in p.iter().enumerate() {
                p_y[i] += v / n;
            }
        }

        // Compute KL for each image and split IS into 10 splits
        let n_splits = 10_usize.min(probs.len());
        let split_size = (probs.len() / n_splits).max(1);
        let mut is_scores = Vec::with_capacity(n_splits);

        for split in 0..n_splits {
            let start = split * split_size;
            let end = ((split + 1) * split_size).min(probs.len());
            let split_probs = &probs[start..end];

            // Marginal for this split
            let sn = split_probs.len() as f32;
            let mut split_py = vec![0.0_f32; n_classes];
            for p in split_probs {
                for (i, v) in p.iter().enumerate() {
                    split_py[i] += v / sn;
                }
            }

            // Mean KL divergence
            let kl_sum: f32 = split_probs
                .iter()
                .map(|p| {
                    p.iter()
                        .zip(split_py.iter())
                        .map(|(pi, qi)| {
                            if *pi > 1e-10 && *qi > 1e-10 {
                                pi * (pi / qi).ln()
                            } else {
                                0.0
                            }
                        })
                        .sum::<f32>()
                })
                .sum();
            let mean_kl = kl_sum / sn;
            is_scores.push(mean_kl.exp());
        }

        let _ = p_y; // computed for reference
        let mean = is_scores.iter().sum::<f32>() / is_scores.len() as f32;
        let var =
            is_scores.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / is_scores.len() as f32;
        (mean, var.sqrt())
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![];
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum = exps.iter().sum::<f32>().max(1e-8);
        exps.into_iter().map(|v| v / sum).collect()
    }
}

/// Simplified perceptual loss (L2 in feature space approximated by direct L2).
pub struct PerceptualLoss;

impl PerceptualLoss {
    /// Compute perceptual loss as MSE of mid-layer feature approximation.
    pub fn compute(img1: &[f32], img2: &[f32], dim: usize) -> f32 {
        if img1.is_empty() || img2.is_empty() {
            return 0.0;
        }
        // Apply a simple non-linear feature extraction (ReLU of linear projection)
        let feat1 = Self::extract_features(img1, dim);
        let feat2 = Self::extract_features(img2, dim);
        let n = feat1.len() as f32;
        feat1
            .iter()
            .zip(feat2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n.max(1.0)
    }

    fn extract_features(img: &[f32], dim: usize) -> Vec<f32> {
        // Simple pseudo-feature: ReLU applied to mean-pooled blocks
        let block = (img.len() / dim.max(1)).max(1);
        (0..dim)
            .map(|i| {
                let start = i * block;
                let end = ((i + 1) * block).min(img.len());
                if start >= img.len() {
                    return 0.0;
                }
                let v: f32 = img[start..end].iter().sum::<f32>() / (end - start) as f32;
                relu_f32(v)
            })
            .collect()
    }
}

/// Evaluation report for an image generation model.
#[derive(Debug, Clone)]
pub struct ImageGenEvalReport {
    pub fid: f32,
    pub is_mean: f32,
    pub is_std: f32,
    pub ppl: f32,
    pub lpips: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ── ImageTensor ──────────────────────────────────────────────────────────

    #[test]
    fn test_image_tensor_creation() {
        let img = ImageTensor::new(3, 16, 16);
        assert_eq!(img.data.len(), 3 * 16 * 16);
        assert!(img.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_image_tensor_shape() {
        let img = ImageTensor::new(3, 32, 32);
        assert_eq!(img.shape(), (3, 32, 32));
    }

    #[test]
    fn test_image_tensor_normalize_range() {
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let img = ImageTensor::from_vec(data, 3, 4, 4);
        let normed = img.normalize();
        assert!(normed
            .data
            .iter()
            .all(|&v| (-1.0 - 1e-5..=1.0 + 1e-5).contains(&v)));
    }

    #[test]
    fn test_image_tensor_to_flat_length() {
        let img = ImageTensor::new(3, 8, 8);
        assert_eq!(img.to_flat().len(), 3 * 8 * 8);
    }

    // ── MappingNetwork ───────────────────────────────────────────────────────

    #[test]
    fn test_mapping_network_forward_shape() {
        let mut rng = make_rng(1);
        let mn = MappingNetwork::new(64, 128, &mut rng);
        let z: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let w = mn.forward(&z);
        assert_eq!(w.len(), 128);
    }

    #[test]
    fn test_mapping_network_w_dim() {
        let mut rng = make_rng(2);
        let mn = MappingNetwork::new(32, 64, &mut rng);
        let z = vec![0.5_f32; 32];
        let w = mn.forward(&z);
        assert_eq!(w.len(), mn.w_dim);
    }

    // ── AdaInLayer ───────────────────────────────────────────────────────────

    #[test]
    fn test_adain_normalize_zero_mean() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let scale = vec![1.0_f32; 4];
        let shift = vec![0.0_f32; 4];
        let out = AdaInLayer::normalize(&x, &scale, &shift);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-4, "mean={}", mean);
    }

    #[test]
    fn test_adain_applies_style() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let scale = vec![2.0_f32; 4];
        let shift = vec![1.0_f32; 4];
        let out = AdaInLayer::normalize(&x, &scale, &shift);
        assert_eq!(out.len(), 4);
        // With scale=2, shift=1: output should differ from input
        let same = x.iter().zip(out.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!same, "Style should be applied");
    }

    // ── SynthesisBlock ───────────────────────────────────────────────────────

    #[test]
    fn test_synthesis_block_forward_shape() {
        let mut rng = make_rng(3);
        let block = SynthesisBlock::new(8, 4, 16, 3, &mut rng);
        let spatial = 4;
        let x = vec![0.1_f32; 8 * spatial * spatial];
        let w = vec![0.5_f32; 16];
        let out = block.forward(&x, &w, 8, 4, spatial);
        assert_eq!(out.len(), 4 * spatial * spatial);
    }

    // ── StyleGanGenerator ────────────────────────────────────────────────────

    #[test]
    fn test_stylegan_generator_creation() {
        let mut rng = make_rng(10);
        let gen = StyleGanGenerator::new(32, 64, 3, &mut rng);
        assert_eq!(gen.n_blocks, 3);
        assert_eq!(gen.z_dim, 32);
        assert_eq!(gen.w_dim, 64);
    }

    #[test]
    fn test_stylegan_generate_shape() {
        let mut rng = make_rng(11);
        let gen = StyleGanGenerator::new(16, 32, 2, &mut rng);
        let z = vec![0.1_f32; 16];
        let img = gen.generate(&z);
        // 2 blocks: 4→8→16, then one more upsample → 32; final 3 channels
        assert_eq!(img.channels, 3);
        assert!(img.height > 0 && img.width > 0);
        assert_eq!(img.data.len(), 3 * img.height * img.width);
    }

    #[test]
    fn test_stylegan_generate_deterministic_seed() {
        let mut rng1 = make_rng(42);
        let gen = StyleGanGenerator::new(16, 32, 2, &mut rng1);
        let z = vec![0.5_f32; 16];
        let img1 = gen.generate(&z);
        let img2 = gen.generate(&z);
        // Same z → same output
        assert_eq!(img1.data, img2.data);
    }

    #[test]
    fn test_stylegan_different_z_different_output() {
        let mut rng = make_rng(99);
        let gen = StyleGanGenerator::new(16, 32, 2, &mut rng);
        let z1 = vec![0.1_f32; 16];
        let z2 = vec![-0.1_f32; 16];
        let img1 = gen.generate(&z1);
        let img2 = gen.generate(&z2);
        let diff: f32 = img1
            .data
            .iter()
            .zip(img2.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Different z should produce different outputs");
    }

    // ── ConditionalBatchNorm ─────────────────────────────────────────────────

    #[test]
    fn test_conditional_bn_creation() {
        let mut rng = make_rng(20);
        let cbn = ConditionalBatchNorm::new(10, 32, &mut rng);
        assert_eq!(cbn.gamma_embed.len(), 10);
        assert_eq!(cbn.gamma_embed[0].len(), 32);
    }

    #[test]
    fn test_conditional_bn_normalize_shape() {
        let mut rng = make_rng(21);
        let cbn = ConditionalBatchNorm::new(5, 16, &mut rng);
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let out = cbn.normalize(&x, 2, 1e-5);
        assert_eq!(out.len(), x.len());
    }

    // ── BigGanResBlock ───────────────────────────────────────────────────────

    #[test]
    fn test_biggan_res_block_forward_shape() {
        let mut rng = make_rng(30);
        let block = BigGanResBlock::new(4, 4, 5, false, &mut rng);
        let spatial = 8;
        let x = vec![0.2_f32; 4 * spatial * spatial];
        let out = block.forward(&x, 1, spatial);
        assert_eq!(out.len(), 4 * spatial * spatial);
    }

    // ── SpectralNormLinear ───────────────────────────────────────────────────

    #[test]
    fn test_spectral_norm_forward_shape() {
        let mut rng = make_rng(40);
        let snl = SpectralNormLinear::new(16, 8, &mut rng);
        let x = vec![0.1_f32; 16];
        let out = snl.forward(&x);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_spectral_norm_normalized_weight_shape() {
        let mut rng = make_rng(41);
        let snl = SpectralNormLinear::new(8, 4, &mut rng);
        let wn = snl.normalized_weight();
        assert_eq!(wn.len(), 4);
        assert_eq!(wn[0].len(), 8);
    }

    // ── VqCodebook ───────────────────────────────────────────────────────────

    #[test]
    fn test_vq_codebook_creation() {
        let mut rng = make_rng(50);
        let cb = VqCodebook::new(256, 64, &mut rng);
        assert_eq!(cb.n_codes, 256);
        assert_eq!(cb.embeddings.len(), 256);
        assert_eq!(cb.embeddings[0].len(), 64);
    }

    #[test]
    fn test_vq_codebook_quantize_shape() {
        let mut rng = make_rng(51);
        let cb = VqCodebook::new(32, 16, &mut rng);
        let z: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let (q, idx, _loss) = cb.quantize(&z);
        assert_eq!(q.len(), 16);
        assert!(idx < 32);
    }

    #[test]
    fn test_vq_codebook_commitment_loss_nonneg() {
        let mut rng = make_rng(52);
        let cb = VqCodebook::new(16, 8, &mut rng);
        let z = vec![0.3_f32; 8];
        let (_q, _idx, loss) = cb.quantize(&z);
        assert!(loss >= 0.0, "Commitment loss must be non-negative");
    }

    #[test]
    fn test_vq_codebook_decode_index_shape() {
        let mut rng = make_rng(53);
        let cb = VqCodebook::new(64, 32, &mut rng);
        let decoded = cb.decode_index(5);
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_vq_codebook_nearest_neighbor() {
        let mut rng = make_rng(54);
        // Small codebook: check that decode(quantize(e_i)) ≈ e_i for an exact codebook entry
        let n_codes = 8;
        let code_dim = 4;
        let cb = VqCodebook::new(n_codes, code_dim, &mut rng);
        // Use exact codebook entry as query
        let z = cb.embeddings[3].clone();
        let (q, idx, _) = cb.quantize(&z);
        assert_eq!(idx, 3, "Should map to exact codebook entry");
        let decoded = cb.decode_index(idx);
        let err: f32 = q
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(err < 1e-5, "decode_index should match quantized");
    }

    // ── VqGanEncoder / Decoder ───────────────────────────────────────────────

    #[test]
    fn test_vqgan_encoder_shape() {
        let mut rng = make_rng(60);
        let enc = VqGanEncoder::new(64, 3, 16, &mut rng);
        let x = vec![0.1_f32; 64];
        let z = enc.encode(&x);
        assert_eq!(z.len(), 16);
    }

    #[test]
    fn test_vqgan_decoder_shape() {
        let mut rng = make_rng(61);
        let dec = VqGanDecoder::new(16, 3, 64, &mut rng);
        let q = vec![0.5_f32; 16];
        let out = dec.decode(&q);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_vqgan_forward_reconstruction_shape() {
        let mut rng = make_rng(62);
        let vqgan = VqGan::new(32, 8, 16, &mut rng);
        let x = vec![0.2_f32; 32];
        let (recon, _idx, _loss) = vqgan.forward(&x);
        assert_eq!(recon.len(), 32);
    }

    #[test]
    fn test_vqgan_reconstruct_shape() {
        let mut rng = make_rng(63);
        let vqgan = VqGan::new(32, 8, 16, &mut rng);
        let x = vec![0.2_f32; 32];
        let recon = vqgan.reconstruct(&x);
        assert_eq!(recon.len(), 32);
    }

    #[test]
    fn test_vqgan_code_idx_in_range() {
        let mut rng = make_rng(64);
        let n_codes = 32;
        let vqgan = VqGan::new(16, 4, n_codes, &mut rng);
        let x = vec![0.5_f32; 16];
        let (_recon, idx, _loss) = vqgan.forward(&x);
        assert!(idx < n_codes, "Code index {} out of range {}", idx, n_codes);
    }

    // ── GAN losses ───────────────────────────────────────────────────────────

    #[test]
    fn test_generator_loss_vanilla() {
        let fake_logits = vec![0.0_f32, 1.0, -1.0];
        let loss = generator_loss(&fake_logits, GanLossType::VanillaGan);
        assert!(loss.is_finite(), "Loss should be finite");
        assert!(loss >= 0.0, "Vanilla gen loss should be non-negative");
    }

    #[test]
    fn test_discriminator_loss_vanilla_positive() {
        let real = vec![2.0_f32, 1.5, 1.0];
        let fake = vec![-1.0_f32, -0.5, 0.0];
        let loss = discriminator_loss(&real, &fake, GanLossType::VanillaGan);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_generator_loss_lsgan() {
        let fake_logits = vec![0.8_f32, 0.9, 1.0];
        let loss = generator_loss(&fake_logits, GanLossType::LsGan);
        assert!(loss.is_finite());
        // D(G(z)) near 1 → loss near 0
        assert!(loss < 0.1);
    }

    #[test]
    fn test_discriminator_loss_hinge() {
        let real = vec![1.5_f32, 2.0];
        let fake = vec![-1.5_f32, -2.0];
        let loss = discriminator_loss(&real, &fake, GanLossType::HingeGan);
        assert!(loss.is_finite());
        // real >> 1 and fake << -1 → small loss
        assert!(
            loss < 0.5,
            "Hinge loss should be small when margins are satisfied"
        );
    }

    #[test]
    fn test_discriminator_loss_wgan() {
        let real = vec![1.0_f32, 1.0];
        let fake = vec![-1.0_f32, -1.0];
        let loss = discriminator_loss(&real, &fake, GanLossType::WassersteinGan);
        assert!(loss.is_finite());
        // E[D(fake)] - E[D(real)] = -1 - 1 = -2
        assert!((loss - (-2.0)).abs() < 1e-5);
    }

    // ── Gradient penalty ─────────────────────────────────────────────────────

    #[test]
    fn test_gradient_penalty_nonneg() {
        let mut rng = make_rng(70);
        let real: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0_f32; 8]).collect();
        let fake: Vec<Vec<f32>> = (0..4).map(|_| vec![-1.0_f32; 8]).collect();
        let penalty = gradient_penalty(&real, &fake, 10.0, &mut rng);
        assert!(penalty >= 0.0, "Gradient penalty must be non-negative");
    }

    // ── R1 regularization ────────────────────────────────────────────────────

    #[test]
    fn test_r1_regularization_nonneg() {
        let real_logits = vec![1.0_f32, 0.5, -0.5];
        let real_samples: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let r1 = r1_regularization(&real_logits, &real_samples);
        assert!(r1 >= 0.0, "R1 must be non-negative");
    }

    // ── Path length / EMA ────────────────────────────────────────────────────

    #[test]
    fn test_path_length_finite() {
        let w1 = vec![0.1_f32; 8];
        let w2 = vec![0.2_f32; 8];
        let img1 = vec![0.5_f32; 16];
        let img2 = vec![0.6_f32; 16];
        let ppl = PathLengthRegularization::perceptual_path_length(&w1, &w2, &img1, &img2);
        assert!(ppl.is_finite());
        assert!(ppl >= 0.0);
    }

    #[test]
    fn test_ema_weights_update() {
        let mut ema = vec![0.0_f32; 4];
        let current = vec![1.0_f32; 4];
        EmaWeights::update(&mut ema, &current, 0.9);
        // 0.9 * 0 + 0.1 * 1 = 0.1
        for v in &ema {
            assert!((v - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ema_weights_decay_rate() {
        let old = vec![10.0_f32; 4];
        let mut ema = old.clone();
        let current = vec![0.0_f32; 4];
        let decay = 0.9_f32;
        EmaWeights::update(&mut ema, &current, decay);
        // Result should be between old and current (closer to old)
        for (&e, &o) in ema.iter().zip(old.iter()) {
            assert!(e < o, "EMA should decay toward current");
            assert!(
                e > 0.0,
                "Should not reach current in one step with decay 0.9"
            );
        }
    }

    // ── FrechetDistance ───────────────────────────────────────────────────────

    #[test]
    fn test_fid_score_zero_same() {
        let feats: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let fid = FrechetDistance::fid_score(&feats, &feats);
        assert!(
            fid < 1e-3,
            "FID of identical distributions should be ~0, got {}",
            fid
        );
    }

    #[test]
    fn test_fid_score_positive_different() {
        let real: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let fake: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 + 100.0, (i + 1) as f32 + 100.0])
            .collect();
        let fid = FrechetDistance::fid_score(&real, &fake);
        assert!(
            fid > 0.0,
            "FID of different distributions should be positive"
        );
    }

    // ── InceptionScore ────────────────────────────────────────────────────────

    #[test]
    fn test_inception_score_shape() {
        let logits: Vec<Vec<f32>> = (0..20).map(|i| vec![(i % 5) as f32 * 2.0; 5]).collect();
        let (mean, std) = InceptionScore::is_score(&logits);
        let _ = (mean, std); // just verifying it returns a tuple
    }

    #[test]
    fn test_inception_score_range() {
        // Peaked distributions → high IS
        let logits: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let mut v = vec![-10.0_f32; 10];
                v[i % 10] = 10.0;
                v
            })
            .collect();
        let (mean, _std) = InceptionScore::is_score(&logits);
        assert!(mean >= 1.0, "IS should be at least 1.0, got {}", mean);
    }

    // ── PerceptualLoss ────────────────────────────────────────────────────────

    #[test]
    fn test_perceptual_loss_zero_same() {
        let img = vec![0.5_f32; 64];
        let loss = PerceptualLoss::compute(&img, &img, 8);
        assert!(
            loss < 1e-6,
            "Perceptual loss of identical images should be 0"
        );
    }

    #[test]
    fn test_perceptual_loss_positive_different() {
        let img1 = vec![0.0_f32; 64];
        let img2 = vec![1.0_f32; 64];
        let loss = PerceptualLoss::compute(&img1, &img2, 8);
        assert!(
            loss > 0.0,
            "Perceptual loss of different images should be positive"
        );
    }

    // ── ImageGenEvalReport ───────────────────────────────────────────────────

    #[test]
    fn test_image_gen_eval_report_fields() {
        let report = ImageGenEvalReport {
            fid: 12.5,
            is_mean: 3.2,
            is_std: 0.4,
            ppl: 250.0,
            lpips: 0.12,
        };
        assert!((report.fid - 12.5).abs() < 1e-6);
        assert!((report.is_mean - 3.2).abs() < 1e-6);
        assert!((report.is_std - 0.4).abs() < 1e-6);
        assert!((report.ppl - 250.0).abs() < 1e-6);
        assert!((report.lpips - 0.12).abs() < 1e-6);
    }
}
