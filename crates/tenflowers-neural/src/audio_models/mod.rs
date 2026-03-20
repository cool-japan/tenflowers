//! Speech & Audio Neural Networks — Track AU.
//!
//! Implements a comprehensive set of modern speech and audio neural network
//! architectures, all in pure Rust, operating on flat `Vec<f32>` / `&[f32]`
//! buffers for zero external-tensor overhead.
//!
//! ## Modules
//!
//! - **Wav2Vec 2.0** — self-supervised speech representation learning
//! - **Conformer** — convolution-augmented transformer for ASR
//! - **FastSpeech 2 / Mel** — non-autoregressive TTS with mel-spectrogram
//! - **Speaker Recognition** — TDNN / x-vector / GE2E loss
//! - **Audio Augmentation** — SpecAugment, noise injection, pitch/time distortion
//! - **Audio Foundation Models** — HuBERT, Data2Vec, SoundStream, DAC, MIR, TTS components

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// Construct an InvalidArgument error concisely.
#[inline]
pub(crate) fn inval(op: &str, reason: impl Into<String>) -> TensorError {
    TensorError::InvalidArgument {
        operation: op.to_string(),
        reason: reason.into(),
        context: None,
    }
}

#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    x * 0.5 * (1.0 + inner.tanh())
}

#[inline]
pub(crate) fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Softmax over a mutable slice (in-place).
pub(crate) fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Dot product of two equally-sized slices.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Numerically stable log-sum-exp.
#[inline]
pub(crate) fn log_sum_exp(v: &[f32]) -> f32 {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f32 = v.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Layer normalisation over the last dimension.
/// `data`: flat [T × D], `d`: feature dim, `eps`: stability.
pub(crate) fn layer_norm_rows(data: &[f32], d: usize, eps: f32) -> Vec<f32> {
    if d == 0 || data.is_empty() {
        return data.to_vec();
    }
    let t = data.len() / d;
    let mut out = vec![0.0_f32; data.len()];
    for i in 0..t {
        let row = &data[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            out[i * d + j] = (row[j] - mean) * inv_std;
        }
    }
    out
}

/// Simple linear projection: y[i*out_d .. (i+1)*out_d] = W·x[i*in_d..] + b
pub(crate) fn linear_fwd(x: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let t = x.len() / in_d;
    let mut out = vec![0.0_f32; t * out_d];
    for i in 0..t {
        let xi = &x[i * in_d..(i + 1) * in_d];
        for j in 0..out_d {
            let mut s = b[j];
            for k in 0..in_d {
                s += w[j * in_d + k] * xi[k];
            }
            out[i * out_d + j] = s;
        }
    }
    out
}

/// Same-padded 1D convolution: `[T, in_ch]` → `[T, out_ch]`.
pub(crate) fn conv1d_same(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    t: usize,
    in_ch: usize,
    out_ch: usize,
    ks: usize,
) -> Vec<f32> {
    let pad = (ks - 1) / 2;
    let mut out = vec![0.0_f32; t * out_ch];
    for ti in 0..t {
        for oc in 0..out_ch {
            let mut s = b[oc];
            for ic in 0..in_ch {
                for ki in 0..ks {
                    let src = ti + ki;
                    if src < pad || src - pad >= t {
                        continue;
                    }
                    s += w[oc * in_ch * ks + ic * ks + ki] * x[(src - pad) * in_ch + ic];
                }
            }
            out[ti * out_ch + oc] = s;
        }
    }
    out
}

/// Multi-head attention: `[T, D]` → `[T, D]` with `nh` heads.
pub(crate) fn mha_fwd(
    x: &[f32],
    t: usize,
    d: usize,
    nh: usize,
    qkv_w: &[f32],
    out_w: &[f32],
    pos_bias: bool,
) -> Vec<f32> {
    let head_d = d / nh;
    let scale = (head_d as f32).sqrt().max(1e-8);
    let qkv = linear_fwd(x, qkv_w, &vec![0.0_f32; 3 * d], d, 3 * d);
    let q: Vec<f32> = (0..t)
        .flat_map(|i| qkv[i * 3 * d..i * 3 * d + d].to_vec())
        .collect();
    let k: Vec<f32> = (0..t)
        .flat_map(|i| qkv[i * 3 * d + d..i * 3 * d + 2 * d].to_vec())
        .collect();
    let v: Vec<f32> = (0..t)
        .flat_map(|i| qkv[i * 3 * d + 2 * d..i * 3 * d + 3 * d].to_vec())
        .collect();
    let mut attn_out = vec![0.0_f32; t * d];
    for h_idx in 0..nh {
        let h_off = h_idx * head_d;
        for qi in 0..t {
            let q_row = &q[qi * d + h_off..qi * d + h_off + head_d];
            let mut scores = vec![0.0_f32; t];
            for ki in 0..t {
                let k_row = &k[ki * d + h_off..ki * d + h_off + head_d];
                let bias = if pos_bias {
                    ((qi as f32 - ki as f32) / d as f32).sin() * 0.1
                } else {
                    0.0
                };
                scores[ki] = dot(q_row, k_row) / scale + bias;
            }
            softmax_inplace(&mut scores);
            for ki in 0..t {
                let v_row = &v[ki * d + h_off..ki * d + h_off + head_d];
                for j in 0..head_d {
                    attn_out[qi * d + h_off + j] += scores[ki] * v_row[j];
                }
            }
        }
    }
    linear_fwd(&attn_out, out_w, &vec![0.0_f32; d], d, d)
}

/// Kaiming-He uniform initialisation for weight buffers.
pub(crate) fn he_init(size: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let bound = (6.0 / fan_in as f64).sqrt() as f32;
    (0..size)
        .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
        .collect()
}

/// Configuration for the 7-layer 1D CNN feature extractor.
#[derive(Debug, Clone)]
pub struct FeatureExtractorConfig {
    /// Number of output channels per layer.
    pub channels: Vec<usize>,
    /// Kernel sizes per layer.
    pub kernel_sizes: Vec<usize>,
    /// Strides per layer.
    pub strides: Vec<usize>,
}

impl Default for FeatureExtractorConfig {
    fn default() -> Self {
        Self {
            channels: vec![512, 512, 512, 512, 512, 512, 512],
            kernel_sizes: vec![10, 3, 3, 3, 3, 2, 2],
            strides: vec![5, 2, 2, 2, 2, 2, 2],
        }
    }
}

/// 7-layer 1D CNN (conv1d + GELU) feature extractor for raw waveform input.
///
/// Each layer applies:  `Conv1d(in, out, k, stride) → GELU`
///
/// Input: `[T]` (mono waveform samples)
/// Output: `[T', hidden_dim]` flat buffer, where `T'` is the reduced time axis
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Convolution weights per layer: `[out_ch × in_ch × kernel]`
    pub weights: Vec<Vec<f32>>,
    /// Biases per layer: `[out_ch]`
    pub biases: Vec<Vec<f32>>,
    pub config: FeatureExtractorConfig,
}

impl FeatureExtractor {
    /// Construct with random initialisation using `seed`.
    pub fn new(config: FeatureExtractorConfig, seed: u64) -> Result<Self> {
        let n = config.channels.len();
        if n != config.kernel_sizes.len() || n != config.strides.len() {
            return Err(inval(
                "FeatureExtractor::new",
                "channels / kernel_sizes / strides must have equal length",
            ));
        }
        let mut weights = Vec::with_capacity(n);
        let mut biases = Vec::with_capacity(n);
        let mut in_ch = 1_usize;
        for (i, (&out_ch, &ks)) in config
            .channels
            .iter()
            .zip(config.kernel_sizes.iter())
            .enumerate()
        {
            let w_size = out_ch * in_ch * ks;
            let w = he_init(w_size, in_ch * ks, seed.wrapping_add(i as u64));
            let b = vec![0.0_f32; out_ch];
            weights.push(w);
            biases.push(b);
            in_ch = out_ch;
        }
        Ok(Self {
            weights,
            biases,
            config,
        })
    }

    /// Forward pass: raw waveform `[T]` → latent features `[T', out_ch]`.
    ///
    /// Returns `(features, t_prime)` where features is flat `[T' × out_ch]`.
    pub fn forward(&self, waveform: &[f32]) -> Result<(Vec<f32>, usize)> {
        if waveform.is_empty() {
            return Err(inval("FeatureExtractor::forward", "waveform is empty"));
        }
        let mut current: Vec<f32> = waveform.to_vec();
        let mut t = waveform.len();
        let mut in_ch = 1_usize;

        for (layer_idx, (&out_ch, (&ks, &stride))) in self
            .config
            .channels
            .iter()
            .zip(
                self.config
                    .kernel_sizes
                    .iter()
                    .zip(self.config.strides.iter()),
            )
            .enumerate()
        {
            let w = &self.weights[layer_idx];
            let b = &self.biases[layer_idx];
            if t < ks {
                return Err(inval(
                    "FeatureExtractor::forward",
                    format!("layer {layer_idx}: time dim {t} < kernel size {ks}"),
                ));
            }
            let t_out = (t - ks) / stride + 1;
            let mut out = vec![0.0_f32; t_out * out_ch];
            for ti in 0..t_out {
                let t_start = ti * stride;
                for oc in 0..out_ch {
                    let mut s = b[oc];
                    for ic in 0..in_ch {
                        for ki in 0..ks {
                            let src_t = t_start + ki;
                            s += w[oc * in_ch * ks + ic * ks + ki] * current[src_t * in_ch + ic];
                        }
                    }
                    out[ti * out_ch + oc] = gelu(s);
                }
            }
            current = out;
            t = t_out;
            in_ch = out_ch;
        }
        Ok((current, t))
    }

    /// Output channel dimensionality (last layer's out_ch).
    pub fn output_dim(&self) -> usize {
        *self.config.channels.last().unwrap_or(&512)
    }
}

/// Output of a quantizer forward pass.
#[derive(Debug, Clone)]
pub struct QuantizerOutput {
    /// Codebook entry indices per group: `[T, G]` flat.
    pub indices: Vec<usize>,
    /// Quantized representation: `[T, G*sub_d]` flat.
    pub quantized: Vec<f32>,
    /// Diversity loss (entropy regularisation): scalar.
    pub diversity_loss: f32,
}

/// Product quantizer: `G` groups × `V` codebook entries per group.
///
/// The input vector of dimension `D` is split into `G` sub-vectors of
/// dimension `D/G`.  Each sub-vector is compared against `V` entries;
/// the nearest entry index is selected via Gumbel-softmax during training.
#[derive(Debug, Clone)]
pub struct QuantizerCodebook {
    /// `[G, V, sub_d]` codebook entries (flat).
    pub codebook: Vec<f32>,
    /// Number of groups.
    pub num_groups: usize,
    /// Number of entries per group.
    pub num_entries: usize,
    /// Sub-vector dimension = input_dim / num_groups.
    pub sub_dim: usize,
    /// Temperature for Gumbel-softmax.
    pub temperature: f32,
}

impl QuantizerCodebook {
    /// Create a new product quantizer.
    pub fn new(
        input_dim: usize,
        num_groups: usize,
        num_entries: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Self> {
        if input_dim == 0 || num_groups == 0 || num_entries == 0 {
            return Err(inval("QuantizerCodebook::new", "dimensions must be > 0"));
        }
        if input_dim % num_groups != 0 {
            return Err(inval(
                "QuantizerCodebook::new",
                format!("input_dim {input_dim} must be divisible by num_groups {num_groups}"),
            ));
        }
        let sub_dim = input_dim / num_groups;
        let size = num_groups * num_entries * sub_dim;
        let codebook = he_init(size, sub_dim, seed);
        Ok(Self {
            codebook,
            num_groups,
            num_entries,
            sub_dim,
            temperature,
        })
    }

    /// Quantize a batch of feature vectors.
    ///
    /// `z`: flat `[T, D]` where `D = num_groups * sub_dim`.
    /// Returns [`QuantizerOutput`].
    pub fn quantize(&self, z: &[f32]) -> Result<QuantizerOutput> {
        let d = self.num_groups * self.sub_dim;
        if z.is_empty() || z.len() % d != 0 {
            return Err(inval(
                "QuantizerCodebook::quantize",
                format!("z.len() {} not divisible by D={d}", z.len()),
            ));
        }
        let (t, g, v, sd) = (z.len() / d, self.num_groups, self.num_entries, self.sub_dim);
        let mut indices = vec![0_usize; t * g];
        let mut quantized = vec![0.0_f32; t * d];
        let mut soft_counts = vec![0.0_f32; g * v];
        for ti in 0..t {
            for gi in 0..g {
                let sub = &z[ti * d + gi * sd..ti * d + (gi + 1) * sd];
                let mut probs: Vec<f32> = (0..v)
                    .map(|vi| {
                        let e = &self.codebook[gi * v * sd + vi * sd..gi * v * sd + vi * sd + sd];
                        let dist: f32 = sub.iter().zip(e).map(|(a, b)| (a - b) * (a - b)).sum();
                        -dist / self.temperature
                    })
                    .collect();
                softmax_inplace(&mut probs);
                indices[ti * g + gi] = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                for vi in 0..v {
                    let e = &self.codebook[gi * v * sd + vi * sd..gi * v * sd + vi * sd + sd];
                    for (j, &ev) in e.iter().enumerate() {
                        quantized[ti * d + gi * sd + j] += probs[vi] * ev;
                    }
                    soft_counts[gi * v + vi] += probs[vi];
                }
            }
        }
        let diversity_loss = -(0..g)
            .map(|gi| {
                let total: f32 = soft_counts[gi * v..(gi + 1) * v].iter().sum();
                let inv = if total > 0.0 { 1.0 / total } else { 1.0 };
                soft_counts[gi * v..(gi + 1) * v]
                    .iter()
                    .map(|&c| {
                        let p = c * inv;
                        if p > 1e-10 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>()
            })
            .sum::<f32>()
            / g as f32;

        Ok(QuantizerOutput {
            indices,
            quantized,
            diversity_loss,
        })
    }
}

/// Minimal sinusoidal positional encoding added to a flat `[T, D]` buffer.
pub(crate) fn sinusoidal_pos_enc(t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * d];
    for ti in 0..t {
        for di in 0..d {
            let angle = ti as f64 / 10000_f64.powf(2.0 * (di / 2) as f64 / d as f64);
            out[ti * d + di] = if di % 2 == 0 {
                angle.sin() as f32
            } else {
                angle.cos() as f32
            };
        }
    }
    out
}

/// Configuration for the context Transformer.
#[derive(Debug, Clone)]
pub struct ContextNetworkConfig {
    /// Hidden dimensionality of the transformer.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Feed-forward network inner dimension.
    pub ffn_dim: usize,
}

impl Default for ContextNetworkConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            num_heads: 8,
            num_layers: 6,
            ffn_dim: 2048,
        }
    }
}

/// Lightweight Transformer context network (self-attention + FFN, pre-norm).
///
/// Input/Output: flat `[T, hidden_dim]`.
#[derive(Debug, Clone)]
pub struct TransformerContextNetwork {
    /// Network configuration.
    pub config: ContextNetworkConfig,
    /// Per-layer: QKV projection `[3*D, D]`, output `[D, D]`, FFN w1 `[ffn, D]`, FFN w2 `[D, ffn]`
    pub qkv_w: Vec<Vec<f32>>,
    /// Per-layer output projection weights.
    pub out_w: Vec<Vec<f32>>,
    /// Per-layer FFN first layer weights.
    pub ffn_w1: Vec<Vec<f32>>,
    /// Per-layer FFN second layer weights.
    pub ffn_w2: Vec<Vec<f32>>,
    /// Per-layer FFN first layer biases.
    pub ffn_b1: Vec<Vec<f32>>,
    /// Per-layer FFN second layer biases.
    pub ffn_b2: Vec<Vec<f32>>,
}

impl TransformerContextNetwork {
    /// Create a new transformer context network with random weights.
    pub fn new(config: ContextNetworkConfig, seed: u64) -> Self {
        let d = config.hidden_dim;
        let f = config.ffn_dim;
        let nl = config.num_layers;
        let mut qkv_w = Vec::with_capacity(nl);
        let mut out_w = Vec::with_capacity(nl);
        let mut ffn_w1 = Vec::with_capacity(nl);
        let mut ffn_w2 = Vec::with_capacity(nl);
        let mut ffn_b1 = Vec::with_capacity(nl);
        let mut ffn_b2 = Vec::with_capacity(nl);
        for i in 0..nl {
            let base = seed.wrapping_add(i as u64 * 1000);
            qkv_w.push(he_init(3 * d * d, d, base));
            out_w.push(he_init(d * d, d, base + 1));
            ffn_w1.push(he_init(f * d, d, base + 2));
            ffn_w2.push(he_init(d * f, f, base + 3));
            ffn_b1.push(vec![0.0_f32; f]);
            ffn_b2.push(vec![0.0_f32; d]);
        }
        Self {
            config,
            qkv_w,
            out_w,
            ffn_w1,
            ffn_w2,
            ffn_b1,
            ffn_b2,
        }
    }

    /// Forward pass. Input: `[T, D]` flat. Returns `[T, D]`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.config.hidden_dim;
        if x.len() % d != 0 {
            return Err(inval(
                "TransformerContextNetwork::forward",
                format!("x.len() {} not divisible by hidden_dim {d}", x.len()),
            ));
        }
        let t = x.len() / d;
        let pe = sinusoidal_pos_enc(t, d);
        let mut h: Vec<f32> = x.iter().zip(pe.iter()).map(|(a, b)| a + b).collect();
        for layer_idx in 0..self.config.num_layers {
            let h_norm = layer_norm_rows(&h, d, 1e-5);
            let projected = mha_fwd(
                &h_norm,
                t,
                d,
                self.config.num_heads,
                &self.qkv_w[layer_idx],
                &self.out_w[layer_idx],
                false,
            );
            for i in 0..h.len() {
                h[i] += projected[i];
            }
            let h2_norm = layer_norm_rows(&h, d, 1e-5);
            let ffn1: Vec<f32> = linear_fwd(
                &h2_norm,
                &self.ffn_w1[layer_idx],
                &self.ffn_b1[layer_idx],
                d,
                self.config.ffn_dim,
            )
            .into_iter()
            .map(gelu)
            .collect();
            let ffn2 = linear_fwd(
                &ffn1,
                &self.ffn_w2[layer_idx],
                &self.ffn_b2[layer_idx],
                self.config.ffn_dim,
                d,
            );
            for i in 0..h.len() {
                h[i] += ffn2[i];
            }
        }
        Ok(h)
    }
}

/// InfoNCE contrastive loss for Wav2Vec 2.0 masked prediction.
///
/// For each masked time-step, the loss maximises cosine similarity between
/// the context representation `c_t` and the quantized target `q_t` versus
/// `K` distractors drawn uniformly from other time-steps in the batch.
#[derive(Debug, Clone)]
pub struct ContrastiveWav2VecLoss {
    /// Number of distractor samples.
    pub num_negatives: usize,
    /// Temperature for InfoNCE.
    pub temperature: f32,
}

impl ContrastiveWav2VecLoss {
    /// Create a new contrastive loss with given number of negatives and temperature.
    pub fn new(num_negatives: usize, temperature: f32) -> Self {
        Self {
            num_negatives,
            temperature,
        }
    }

    /// Compute InfoNCE loss over masked positions.
    pub fn compute(
        &self,
        context: &[f32],
        quantized: &[f32],
        mask: &[bool],
        seed: u64,
    ) -> Result<f32> {
        if context.len() != quantized.len() || mask.is_empty() || context.len() / mask.len() == 0 {
            return Err(inval("ContrastiveWav2VecLoss::compute", "shape mismatch"));
        }
        let t = mask.len();
        let d = context.len() / t;
        if d == 0 {
            return Err(inval("ContrastiveWav2VecLoss::compute", "feature dim is 0"));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut total_loss, mut count) = (0.0_f64, 0_usize);
        for ti in 0..t {
            if !mask[ti] {
                continue;
            }
            let c = &context[ti * d..(ti + 1) * d];
            let q = &quantized[ti * d..(ti + 1) * d];
            let (c_norm, q_norm) = (l2_norm(c).max(1e-8), l2_norm(q).max(1e-8));
            let pos_sim = dot(c, q) / (c_norm * q_norm * self.temperature);

            let k = self.num_negatives.min(t.saturating_sub(1));
            let mut seen = std::collections::HashSet::new();
            seen.insert(ti);
            let mut neg_sims: Vec<f32> = Vec::with_capacity(k);
            let mut attempts = 0_usize;
            while neg_sims.len() < k && attempts < 4 * k + 16 {
                let ni = rng.random_range(0..t);
                attempts += 1;
                if !seen.insert(ni) {
                    continue;
                }
                let n = &quantized[ni * d..(ni + 1) * d];
                neg_sims.push(dot(c, n) / (c_norm * l2_norm(n).max(1e-8) * self.temperature));
            }
            let all_sims: Vec<f32> = std::iter::once(pos_sim).chain(neg_sims).collect();
            total_loss += (log_sum_exp(&all_sims) - pos_sim) as f64;
            count += 1;
        }
        Ok(if count > 0 {
            (total_loss / count as f64) as f32
        } else {
            0.0
        })
    }
}

/// Full Wav2Vec 2.0 model: feature extractor + quantizer + context network.
#[derive(Debug, Clone)]
pub struct Wav2Vec2Model {
    /// CNN feature extractor for raw waveform.
    pub feature_extractor: FeatureExtractor,
    /// Product quantizer for discrete targets.
    pub quantizer: QuantizerCodebook,
    /// Transformer context network.
    pub context_network: TransformerContextNetwork,
    /// Projection from feature_extractor output dim → quantizer input dim.
    pub proj_w: Vec<f32>,
    /// Projection bias.
    pub proj_b: Vec<f32>,
}

impl Wav2Vec2Model {
    /// Create a default Wav2Vec 2.0 model with random initialisation.
    pub fn new(seed: u64) -> Result<Self> {
        let feat_cfg = FeatureExtractorConfig::default();
        let feature_extractor = FeatureExtractor::new(feat_cfg, seed)?;
        let feat_dim = feature_extractor.output_dim(); // 512
        let quantizer = QuantizerCodebook::new(feat_dim, 2, 320, 2.0, seed + 100)?;
        let ctx_cfg = ContextNetworkConfig {
            hidden_dim: feat_dim,
            num_heads: 8,
            num_layers: 2,
            ffn_dim: 2048,
        };
        let context_network = TransformerContextNetwork::new(ctx_cfg, seed + 200);
        let proj_w = he_init(feat_dim * feat_dim, feat_dim, seed + 300);
        let proj_b = vec![0.0_f32; feat_dim];
        Ok(Self {
            feature_extractor,
            quantizer,
            context_network,
            proj_w,
            proj_b,
        })
    }

    /// Forward: waveform → (context representations, quantizer output).
    pub fn forward(&self, waveform: &[f32]) -> Result<(Vec<f32>, QuantizerOutput)> {
        let (features, _t) = self.feature_extractor.forward(waveform)?;
        let d = self.feature_extractor.output_dim();
        let projected = linear_fwd(&features, &self.proj_w, &self.proj_b, d, d);
        let quant_out = self.quantizer.quantize(&projected)?;
        let context = self.context_network.forward(&projected)?;
        Ok((context, quant_out))
    }
}

/// Conformer ConvModule: LayerNorm → pointwise → GLU → depthwise → BN → Swish → pointwise + residual.
#[derive(Debug, Clone)]
pub struct ConvModule {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Depthwise convolution kernel size.
    pub kernel_size: usize,
    /// Pointwise expansion weight: `[2*D, D]` (for GLU → expand ×2)
    pub pw_expand_w: Vec<f32>,
    /// Depthwise conv weight: `[D, 1, K]`
    pub dw_w: Vec<f32>,
    /// Pointwise contraction weight: `[D, D]`
    pub pw_contract_w: Vec<f32>,
    /// Batch norm scale per channel.
    pub bn_scale: Vec<f32>,
    /// Batch norm bias per channel.
    pub bn_bias: Vec<f32>,
}

impl ConvModule {
    /// Create a new ConvModule with given input dim and kernel size.
    pub fn new(input_dim: usize, kernel_size: usize, seed: u64) -> Self {
        let d = input_dim;
        Self {
            input_dim: d,
            kernel_size,
            pw_expand_w: he_init(2 * d * d, d, seed),
            dw_w: he_init(d * kernel_size, d, seed + 1),
            pw_contract_w: he_init(d * d, d, seed + 2),
            bn_scale: vec![1.0_f32; d],
            bn_bias: vec![0.0_f32; d],
        }
    }

    /// Forward: `[T, D]` → `[T, D]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let d = self.input_dim;
        let t = x.len() / d;
        if t == 0 || d == 0 {
            return x.to_vec();
        }
        let xn = layer_norm_rows(x, d, 1e-5);
        let expanded = linear_fwd(&xn, &self.pw_expand_w, &vec![0.0_f32; 2 * d], d, 2 * d);
        let mut glu_out = vec![0.0_f32; t * d];
        for ti in 0..t {
            for j in 0..d {
                let a = expanded[ti * 2 * d + j];
                let b = expanded[ti * 2 * d + d + j];
                glu_out[ti * d + j] = a / (1.0 + (-b).exp());
            }
        }
        let ks = self.kernel_size;
        let pad = (ks - 1) / 2;
        let mut dw_out = vec![0.0_f32; t * d];
        for ti in 0..t {
            for ch in 0..d {
                let mut s = 0.0_f32;
                for ki in 0..ks {
                    let src = ti + ki;
                    if src < pad || src - pad >= t {
                        continue;
                    }
                    s += self.dw_w[ch * ks + ki] * glu_out[(src - pad) * d + ch];
                }
                dw_out[ti * d + ch] = s;
            }
        }
        let swish_out: Vec<f32> = dw_out
            .iter()
            .enumerate()
            .map(|(i, &v)| swish(v * self.bn_scale[i % d] + self.bn_bias[i % d]))
            .collect();
        let contracted = linear_fwd(&swish_out, &self.pw_contract_w, &vec![0.0_f32; d], d, d);
        x.iter()
            .zip(contracted.iter())
            .map(|(a, b)| a + b)
            .collect()
    }
}

/// Conformer FeedForwardModule: pre-norm FFN with swish, half-step residual (×0.5).
#[derive(Debug, Clone)]
pub struct FeedForwardModule {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Feed-forward inner dimension.
    pub ffn_dim: usize,
    /// First linear layer weights.
    pub w1: Vec<f32>,
    /// First linear layer biases.
    pub b1: Vec<f32>,
    /// Second linear layer weights.
    pub w2: Vec<f32>,
    /// Second linear layer biases.
    pub b2: Vec<f32>,
}

impl FeedForwardModule {
    /// Create a new FeedForwardModule.
    pub fn new(input_dim: usize, ffn_dim: usize, seed: u64) -> Self {
        Self {
            input_dim,
            ffn_dim,
            w1: he_init(ffn_dim * input_dim, input_dim, seed),
            b1: vec![0.0_f32; ffn_dim],
            w2: he_init(input_dim * ffn_dim, ffn_dim, seed + 1),
            b2: vec![0.0_f32; input_dim],
        }
    }

    /// Forward: `[T, D]` → `[T, D]` with half-step residual.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let d = self.input_dim;
        let xn = layer_norm_rows(x, d, 1e-5);
        let h1 = linear_fwd(&xn, &self.w1, &self.b1, d, self.ffn_dim);
        let h1s: Vec<f32> = h1.iter().map(|&v| swish(v)).collect();
        let h2 = linear_fwd(&h1s, &self.w2, &self.b2, self.ffn_dim, d);
        x.iter().zip(h2.iter()).map(|(a, b)| a + 0.5 * b).collect()
    }
}

/// Relative-position multi-head self-attention module (simplified).
#[derive(Debug, Clone)]
pub struct MultiHeadSelfAttentionModule {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// QKV projection weights.
    pub qkv_w: Vec<f32>,
    /// Output projection weights.
    pub out_w: Vec<f32>,
}

impl MultiHeadSelfAttentionModule {
    /// Create a new MultiHeadSelfAttentionModule.
    pub fn new(input_dim: usize, num_heads: usize, seed: u64) -> Self {
        let d = input_dim;
        Self {
            input_dim: d,
            num_heads,
            qkv_w: he_init(3 * d * d, d, seed),
            out_w: he_init(d * d, d, seed + 1),
        }
    }

    /// Forward: `[T, D]` → `[T, D]` with relative positional bias.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let d = self.input_dim;
        let t = x.len() / d;
        if t == 0 || d == 0 {
            return x.to_vec();
        }
        let xn = layer_norm_rows(x, d, 1e-5);
        let projected = mha_fwd(&xn, t, d, self.num_heads, &self.qkv_w, &self.out_w, true);
        x.iter().zip(projected.iter()).map(|(a, b)| a + b).collect()
    }
}

/// Conformer block: FFN/2 → MHSA → Conv → FFN/2 (Macaron-Net style).
#[derive(Debug, Clone)]
pub struct ConformerBlock {
    /// First feed-forward module (half-step).
    pub ffn1: FeedForwardModule,
    /// Multi-head self-attention module.
    pub mhsa: MultiHeadSelfAttentionModule,
    /// Convolution module.
    pub conv_module: ConvModule,
    /// Second feed-forward module (half-step).
    pub ffn2: FeedForwardModule,
}

impl ConformerBlock {
    /// Create a new ConformerBlock.
    pub fn new(
        input_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        conv_kernel: usize,
        seed: u64,
    ) -> Self {
        Self {
            ffn1: FeedForwardModule::new(input_dim, ffn_dim, seed),
            mhsa: MultiHeadSelfAttentionModule::new(input_dim, num_heads, seed + 10),
            conv_module: ConvModule::new(input_dim, conv_kernel, seed + 20),
            ffn2: FeedForwardModule::new(input_dim, ffn_dim, seed + 30),
        }
    }

    /// Forward: `[T, D]` → `[T, D]` (shape-preserving).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h1 = self.ffn1.forward(x);
        let h2 = self.mhsa.forward(&h1);
        let h3 = self.conv_module.forward(&h2);
        self.ffn2.forward(&h3)
    }
}

/// Conformer encoder: N ConformerBlocks + linear projection.
#[derive(Debug, Clone)]
pub struct ConformerEncoder {
    /// Stack of conformer blocks.
    pub blocks: Vec<ConformerBlock>,
    /// Output projection weights.
    pub proj_w: Vec<f32>,
    /// Output projection bias.
    pub proj_b: Vec<f32>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output feature dimension.
    pub output_dim: usize,
}

impl ConformerEncoder {
    /// Create a new ConformerEncoder.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        num_blocks: usize,
        num_heads: usize,
        ffn_dim: usize,
        conv_kernel: usize,
        seed: u64,
    ) -> Self {
        let blocks = (0..num_blocks)
            .map(|i| {
                ConformerBlock::new(
                    input_dim,
                    num_heads,
                    ffn_dim,
                    conv_kernel,
                    seed + i as u64 * 100,
                )
            })
            .collect();
        let proj_w = he_init(output_dim * input_dim, input_dim, seed + 9999);
        let proj_b = vec![0.0_f32; output_dim];
        Self {
            blocks,
            proj_w,
            proj_b,
            input_dim,
            output_dim,
        }
    }

    /// Forward: `[T, input_dim]` → `[T, output_dim]`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.input_dim;
        if x.len() % d != 0 {
            return Err(inval(
                "ConformerEncoder::forward",
                format!("x.len() {} not divisible by input_dim {d}", x.len()),
            ));
        }
        let mut h = x.to_vec();
        for block in &self.blocks {
            h = block.forward(&h);
        }
        Ok(linear_fwd(
            &h,
            &self.proj_w,
            &self.proj_b,
            d,
            self.output_dim,
        ))
    }
}

/// Greedy CTC decoder: argmax over vocab, collapse repeats, remove blank (token 0).
#[derive(Debug, Clone)]
pub struct CtcDecoder {
    /// Blank token index (always 0).
    pub blank_token: usize,
    /// Total vocabulary size including blank.
    pub vocab_size: usize,
}

impl CtcDecoder {
    /// Create a CTC decoder with given vocabulary size (blank = token 0).
    pub fn new(vocab_size: usize) -> Self {
        Self {
            blank_token: 0,
            vocab_size,
        }
    }

    /// Decode `logits`: flat `[T, V]` raw scores.
    /// Returns decoded token sequence (no repeats, no blanks).
    pub fn decode(&self, logits: &[f32]) -> Result<Vec<usize>> {
        let v = self.vocab_size;
        if v == 0 || logits.len() % v != 0 {
            return Err(inval(
                "CtcDecoder::decode",
                format!(
                    "logits.len() {} not divisible by vocab_size {v}",
                    logits.len()
                ),
            ));
        }
        let t = logits.len() / v;
        let mut raw: Vec<usize> = (0..t)
            .map(|ti| {
                logits[ti * v..(ti + 1) * v]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        raw.dedup();
        Ok(raw
            .into_iter()
            .filter(|&tok| tok != self.blank_token)
            .collect())
    }
}

/// Duration predictor: 2-layer 1D conv + ReLU + linear → scalar duration per frame.
#[derive(Debug, Clone)]
pub struct FastSpeechDuration {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Conv layer 1 weights: `[D, D, K]`
    pub conv1_w: Vec<f32>,
    /// Conv layer 1 biases.
    pub conv1_b: Vec<f32>,
    /// Conv layer 2 weights: `[D, D, K]`
    pub conv2_w: Vec<f32>,
    /// Conv layer 2 biases.
    pub conv2_b: Vec<f32>,
    /// Final linear weights: `[1, D]`
    pub linear_w: Vec<f32>,
    /// Final linear biases.
    pub linear_b: Vec<f32>,
    /// Convolution kernel size.
    pub kernel_size: usize,
}

impl FastSpeechDuration {
    /// Create a new duration predictor.
    pub fn new(input_dim: usize, kernel_size: usize, seed: u64) -> Self {
        let d = input_dim;
        let ks = kernel_size;
        Self {
            input_dim: d,
            conv1_w: he_init(d * d * ks, d * ks, seed),
            conv1_b: vec![0.0_f32; d],
            conv2_w: he_init(d * d * ks, d * ks, seed + 1),
            conv2_b: vec![0.0_f32; d],
            linear_w: he_init(d, d, seed + 2),
            linear_b: vec![0.0_f32; 1],
            kernel_size,
        }
    }

    /// Forward: `[T, D]` → `Vec<f64>` predicted durations (softplus-activated).
    pub fn predict_duration(&self, x: &[f32]) -> Result<Vec<f64>> {
        let d = self.input_dim;
        let ks = self.kernel_size;
        if x.len() % d != 0 {
            return Err(inval(
                "FastSpeechDuration::predict_duration",
                format!("x.len() {} not divisible by input_dim {d}", x.len()),
            ));
        }
        let t = x.len() / d;
        let h1: Vec<f32> = conv1d_same(x, &self.conv1_w, &self.conv1_b, t, d, d, ks)
            .into_iter()
            .map(relu)
            .collect();
        let h1n = layer_norm_rows(&h1, d, 1e-5);
        let h2: Vec<f32> = conv1d_same(&h1n, &self.conv2_w, &self.conv2_b, t, d, d, ks)
            .into_iter()
            .map(relu)
            .collect();
        let h2n = layer_norm_rows(&h2, d, 1e-5);
        Ok((0..t)
            .map(|ti| {
                let row = &h2n[ti * d..(ti + 1) * d];
                let raw: f32 = row
                    .iter()
                    .zip(self.linear_w.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + self.linear_b[0];
                (1.0_f32 + raw.exp()).ln() as f64
            })
            .collect())
    }
}

/// Length regulator: upsample frame sequence by integer durations.
#[derive(Debug, Clone)]
pub struct LengthRegulator;

impl LengthRegulator {
    /// Create a new LengthRegulator.
    pub fn new() -> Self {
        Self
    }

    /// Upsample `x` (`[T, D]`) by `durations` (length T), repeating each frame `d_i` times.
    pub fn regulate(&self, x: &[f32], durations: &[usize], d: usize) -> Result<Vec<f32>> {
        if x.len() % d != 0 {
            return Err(inval(
                "LengthRegulator::regulate",
                format!("x.len() {} not divisible by feature_dim {d}", x.len()),
            ));
        }
        let t = x.len() / d;
        if durations.len() != t {
            return Err(inval(
                "LengthRegulator::regulate",
                format!("durations.len() {} != T={t}", durations.len()),
            ));
        }
        let total: usize = durations.iter().sum();
        let mut out = Vec::with_capacity(total * d);
        for (ti, &dur) in durations.iter().enumerate() {
            let frame = &x[ti * d..(ti + 1) * d];
            for _ in 0..dur {
                out.extend_from_slice(frame);
            }
        }
        Ok(out)
    }
}

impl Default for LengthRegulator {
    fn default() -> Self {
        Self::new()
    }
}

/// FastSpeech 2 FFN-Transformer block for mel generation.
#[derive(Debug, Clone)]
pub struct FastSpeech2Block {
    /// Multi-head self-attention module.
    pub mhsa: MultiHeadSelfAttentionModule,
    /// First conv weights for FFN.
    pub conv1_w: Vec<f32>,
    /// First conv biases.
    pub conv1_b: Vec<f32>,
    /// Second conv weights for FFN.
    pub conv2_w: Vec<f32>,
    /// Second conv biases.
    pub conv2_b: Vec<f32>,
    /// Input / output feature dimension.
    pub input_dim: usize,
    /// Inner FFN dimension.
    pub ffn_dim: usize,
    /// Conv kernel size.
    pub kernel_size: usize,
}

impl FastSpeech2Block {
    /// Create a new FastSpeech2Block.
    pub fn new(
        input_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        kernel_size: usize,
        seed: u64,
    ) -> Self {
        let d = input_dim;
        let ks = kernel_size;
        Self {
            mhsa: MultiHeadSelfAttentionModule::new(d, num_heads, seed),
            conv1_w: he_init(ffn_dim * d * ks, d * ks, seed + 10),
            conv1_b: vec![0.0_f32; ffn_dim],
            conv2_w: he_init(d * ffn_dim * ks, ffn_dim * ks, seed + 11),
            conv2_b: vec![0.0_f32; d],
            input_dim: d,
            ffn_dim,
            kernel_size: ks,
        }
    }

    /// Forward: `[T, D]` → `[T, D]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let d = self.input_dim;
        let t = x.len() / d;
        let ks = self.kernel_size;
        let f = self.ffn_dim;
        let attn_out = self.mhsa.forward(x);
        let attn_n = layer_norm_rows(&attn_out, d, 1e-5);
        let h1: Vec<f32> = conv1d_same(&attn_n, &self.conv1_w, &self.conv1_b, t, d, f, ks)
            .into_iter()
            .map(relu)
            .collect();
        let h2 = conv1d_same(&h1, &self.conv2_w, &self.conv2_b, t, f, d, ks);
        attn_out.iter().zip(h2.iter()).map(|(a, b)| a + b).collect()
    }
}

/// Mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    /// Number of mel filter banks.
    pub n_mels: usize,
    /// FFT size.
    pub n_fft: usize,
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Mel filterbank: `[n_mels, n_fft/2+1]`
    pub filterbank: Vec<f32>,
}

impl MelSpectrogram {
    /// Construct with a computed mel filterbank.
    pub fn new(n_mels: usize, n_fft: usize, sample_rate: f64) -> Result<Self> {
        if n_mels == 0 || n_fft == 0 {
            return Err(inval("MelSpectrogram::new", "n_mels and n_fft must be > 0"));
        }
        let filterbank = Self::compute_mel_filterbank(n_mels, n_fft, sample_rate);
        Ok(Self {
            n_mels,
            n_fft,
            sample_rate,
            filterbank,
        })
    }

    /// Compute triangular mel filterbank: `[n_mels, n_fft/2+1]`.
    pub fn compute_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f64) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        fn hz_to_mel(hz: f64) -> f64 {
            2595.0 * (1.0 + hz / 700.0).log10()
        }
        fn mel_to_hz(mel: f64) -> f64 {
            700.0 * (10_f64.powf(mel / 2595.0) - 1.0)
        }

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(sample_rate / 2.0);
        let mel_points: Vec<f64> = (0..n_mels + 2)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
            .collect();
        let bin_freqs: Vec<f64> = (0..n_freqs)
            .map(|i| sample_rate * i as f64 / n_fft as f64)
            .collect();

        let mut fb = vec![0.0_f32; n_mels * n_freqs];
        for m in 0..n_mels {
            let f_m_minus = mel_points[m];
            let f_m = mel_points[m + 1];
            let f_m_plus = mel_points[m + 2];
            for k in 0..n_freqs {
                let f = bin_freqs[k];
                let val = if f >= f_m_minus && f <= f_m {
                    if (f_m - f_m_minus).abs() < 1e-10 {
                        0.0
                    } else {
                        ((f - f_m_minus) / (f_m - f_m_minus)) as f32
                    }
                } else if f >= f_m && f <= f_m_plus {
                    if (f_m_plus - f_m).abs() < 1e-10 {
                        0.0
                    } else {
                        ((f_m_plus - f) / (f_m_plus - f_m)) as f32
                    }
                } else {
                    0.0
                };
                fb[m * n_freqs + k] = val;
            }
        }
        fb
    }

    /// Apply filterbank to STFT magnitude spectrogram.
    ///
    /// `spectrogram`: `[T, n_fft/2+1]` → returns `[T, n_mels]`.
    pub fn stft_to_mel(&self, spectrogram: &[f32]) -> Result<Vec<f32>> {
        let n_freqs = self.n_fft / 2 + 1;
        if spectrogram.len() % n_freqs != 0 {
            return Err(inval(
                "MelSpectrogram::stft_to_mel",
                format!(
                    "spectrogram.len() {} not divisible by n_freqs {n_freqs}",
                    spectrogram.len()
                ),
            ));
        }
        let t = spectrogram.len() / n_freqs;
        let mut mel = vec![0.0_f32; t * self.n_mels];
        for ti in 0..t {
            let frame = &spectrogram[ti * n_freqs..(ti + 1) * n_freqs];
            for m in 0..self.n_mels {
                let s: f32 = frame
                    .iter()
                    .zip(self.filterbank[m * n_freqs..(m + 1) * n_freqs].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                mel[ti * self.n_mels + m] = (s.max(1e-8)).ln(); // log mel
            }
        }
        Ok(mel)
    }
}

