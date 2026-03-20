//! Multimodal Foundation Model Components.
//!
//! Implements Vision-Language alignment (CLIP/Flamingo-style), LLaVA-style
//! architecture, Audio-Visual models, Document Understanding, and Video
//! Understanding components — all backed by plain `Vec<f64>` computations
//! (no tensor runtime dependency) with `scirs2_core::random` for weight init.
//!
//! # Modules
//!
//! | Section | Key Types |
//! |---------|-----------|
//! | VL Alignment | [`VisualTokenizer`], [`LanguageProjector`], [`VisualLanguageAligner`], [`GatedCrossAttention`], [`PerceiverResampler`] |
//! | LLaVA | [`VisionEncoder`], [`MlpProjector`], [`LlavaModel`], [`ImageInstructionFormatter`], [`LlavaLoss`] |
//! | Audio-Visual | [`AudioSpectrogram`], [`AudioVisualAttention`], [`AvEncoder`], [`AvContrastiveLoss`] |
//! | Document | [`LayoutAwareAttention`], [`DocumentTokenizer`], [`TableParser`], [`ReadingOrderPrediction`], [`DocumentQaModel`] |
//! | Video | [`TemporalPositionEmbedding`], [`TimesFormerBlock`], [`VideoSlowFast`], [`VideoTextAlignment`], [`VideoQaModel`] |

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::TensorError;

// ── Internal math helpers ───────────────────────────────────────────────────

#[inline]
pub(crate) fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).tanh())
}

#[inline]
pub(crate) fn tanh_safe(x: f64) -> f64 {
    x.clamp(-15.0, 15.0).tanh()
}

pub(crate) fn layer_norm(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    if n == 0.0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub(crate) fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < f64::EPSILON {
        return vec![1.0 / x.len() as f64; x.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

pub(crate) fn l2_normalize(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < f64::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

pub(crate) fn rand_vec(size: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut out = Vec::with_capacity(size);
    let mut i = 0;
    while i < size {
        let u1: f64 = rng.random::<f64>().max(1e-12);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push(r * theta.cos() * 0.02);
        if i + 1 < size {
            out.push(r * theta.sin() * 0.02);
        }
        i += 2;
    }
    out.truncate(size);
    out
}

pub(crate) fn rand_mat(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..rows).map(|_| rand_vec(cols, rng)).collect()
}

pub(crate) fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

pub(crate) fn sdp_attention(q: &[Vec<f64>], k: &[Vec<f64>], v: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = q.first().map(|r| r.len()).unwrap_or(1) as f64;
    let scale = d.sqrt();
    q.iter()
        .map(|qi| {
            let scores: Vec<f64> = k.iter().map(|kj| dot(qi, kj) / scale).collect();
            let weights = softmax(&scores);
            if v.is_empty() {
                return Vec::new();
            }
            let dv = v[0].len();
            let mut out = vec![0.0_f64; dv];
            for (w, vj) in weights.iter().zip(v.iter()) {
                for (o, &vval) in out.iter_mut().zip(vj.iter()) {
                    *o += w * vval;
                }
            }
            out
        })
        .collect()
}

pub(crate) fn sinusoidal_pe(pos: usize, d_model: usize) -> Vec<f64> {
    (0..d_model)
        .map(|i| {
            let denom = 10000.0_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
            if i % 2 == 0 {
                (pos as f64 / denom).sin()
            } else {
                (pos as f64 / denom).cos()
            }
        })
        .collect()
}

// ── 1. Vision-Language Alignment ───────────────────────────────────────────

/// Average-pool variable-length visual features to exactly `n_tokens` tokens.
#[derive(Debug, Clone)]
pub struct VisualTokenizer {
    /// Target number of output tokens.
    pub n_tokens: usize,
}

impl VisualTokenizer {
    /// Create a new `VisualTokenizer` with the given target token count.
    pub fn new(n_tokens: usize) -> Self {
        Self { n_tokens }
    }

    /// Pool `image_features` down to `n_tokens` output tokens via average pooling.
    pub fn tokenize(&self, image_features: &[Vec<f64>], n_tokens: usize) -> Vec<Vec<f64>> {
        let n_tokens = if n_tokens == 0 {
            self.n_tokens
        } else {
            n_tokens
        };
        if image_features.is_empty() || n_tokens == 0 {
            return Vec::new();
        }
        let feat_dim = image_features[0].len();
        let n_src = image_features.len();

        if n_src == n_tokens {
            return image_features.to_vec();
        }
        if n_src < n_tokens {
            let mut out = image_features.to_vec();
            out.resize(n_tokens, vec![0.0; feat_dim]);
            return out;
        }
        let mut out = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let start = t * n_src / n_tokens;
            let end = (t + 1) * n_src / n_tokens;
            let end = end.max(start + 1).min(n_src);
            let count = (end - start) as f64;
            let mut avg = vec![0.0_f64; feat_dim];
            for feat in &image_features[start..end] {
                for (a, &f) in avg.iter_mut().zip(feat.iter()) {
                    *a += f / count;
                }
            }
            out.push(avg);
        }
        out
    }
}

/// Project language features to visual space: linear + GELU.
#[derive(Debug, Clone)]
pub struct LanguageProjector {
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
    out_dim: usize,
}

impl LanguageProjector {
    /// Create a new `LanguageProjector` mapping `in_dim` → `out_dim`.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            weight: rand_mat(out_dim, in_dim, rng),
            bias: vec![0.0; out_dim],
            out_dim,
        }
    }

    /// Project `x` from language space to visual space via linear + GELU.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        matvec(&self.weight, x)
            .iter()
            .zip(self.bias.iter())
            .map(|(v, b)| gelu(v + b))
            .collect()
    }
}

/// CLIP-style contrastive alignment loss (InfoNCE) + optional captioning loss.
#[derive(Debug, Clone)]
pub struct VisualLanguageAligner {
    /// Embedding dimension shared between image and text encoders.
    pub embed_dim: usize,
    /// Log-temperature for InfoNCE (initialized 0 → temperature=1).
    pub log_temperature: f64,
    /// Whether to include captioning loss.
    pub use_captioning: bool,
}

impl VisualLanguageAligner {
    /// Create a new aligner with the given embedding dimension.
    pub fn new(embed_dim: usize, use_captioning: bool) -> Self {
        Self {
            embed_dim,
            log_temperature: 0.0, // temperature = 1.0 initially
            use_captioning,
        }
    }

    /// InfoNCE loss for a batch of (image, text) embedding pairs.
    /// `img_embeds` and `txt_embeds` are each `[batch, embed_dim]`.
    pub fn contrastive_loss(
        &self,
        img_embeds: &[Vec<f64>],
        txt_embeds: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        let n = img_embeds.len();
        if n == 0 || n != txt_embeds.len() {
            return Err(TensorError::invalid_argument_op(
                "contrastive_loss",
                "batch sizes must match and be non-zero",
            ));
        }
        let temp = self.log_temperature.exp();
        // compute similarity matrix
        let mut sim = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let img_n = l2_normalize(&img_embeds[i]);
            for j in 0..n {
                let txt_n = l2_normalize(&txt_embeds[j]);
                sim[i][j] = dot(&img_n, &txt_n) / temp;
            }
        }
        let targets: Vec<usize> = (0..n).collect();
        let mut loss = 0.0_f64;
        // image → text direction
        for i in 0..n {
            let row_sm = softmax(&sim[i]);
            loss -= row_sm[targets[i]].max(1e-12).ln();
        }
        // text → image direction
        let sim_t: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..n).map(|i| sim[i][j]).collect())
            .collect();
        for j in 0..n {
            let col_sm = softmax(&sim_t[j]);
            loss -= col_sm[targets[j]].max(1e-12).ln();
        }
        Ok(loss / (2.0 * n as f64))
    }

    /// Optional captioning loss: cross-entropy over vocabulary logits.
    pub fn captioning_loss(
        &self,
        logits: &[Vec<f64>],
        targets: &[usize],
    ) -> Result<f64, TensorError> {
        if !self.use_captioning {
            return Ok(0.0);
        }
        if logits.len() != targets.len() || logits.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "captioning_loss",
                "logits and targets length mismatch or empty",
            ));
        }
        let mut loss = 0.0_f64;
        for (row, &t) in logits.iter().zip(targets.iter()) {
            let sm = softmax(row);
            let p = if t < sm.len() { sm[t] } else { 1e-12 };
            loss -= p.max(1e-12).ln();
        }
        Ok(loss / logits.len() as f64)
    }
}

/// Flamingo-style gated cross-attention: `tanh(α) * cross_attn(x_lang, x_vis)`.
/// α is a learned scalar initialized to 0 so the gate starts closed.
#[derive(Debug, Clone)]
pub struct GatedCrossAttention {
    /// Model dimension.
    pub d_model: usize,
    /// Learned gate scalar (initialized 0 → tanh(0)=0).
    pub alpha: f64,
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
}

impl GatedCrossAttention {
    /// Create a new gated cross-attention layer.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            alpha: 0.0,
            wq: rand_mat(d_model, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
        }
    }

    /// Forward: language tokens attend to visual tokens, gated by tanh(α).
    /// `x_lang`: [lang_len, d_model], `x_vis`: [vis_len, d_model]
    /// Returns updated language tokens of the same shape.
    pub fn forward(&self, x_lang: &[Vec<f64>], x_vis: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if x_lang.is_empty() || x_vis.is_empty() {
            return x_lang.to_vec();
        }
        let gate = tanh_safe(self.alpha);
        let q: Vec<Vec<f64>> = x_lang.iter().map(|x| matvec(&self.wq, x)).collect();
        let k: Vec<Vec<f64>> = x_vis.iter().map(|x| matvec(&self.wk, x)).collect();
        let v: Vec<Vec<f64>> = x_vis.iter().map(|x| matvec(&self.wv, x)).collect();
        let attn = sdp_attention(&q, &k, &v);
        let out: Vec<Vec<f64>> = attn.iter().map(|a| matvec(&self.wo, a)).collect();
        // residual + gate
        x_lang
            .iter()
            .zip(out.iter())
            .map(|(xl, o)| xl.iter().zip(o.iter()).map(|(a, b)| a + gate * b).collect())
            .collect()
    }
}

/// Perceiver Resampler: fixed learned queries attend to visual features,
/// producing a fixed-size set of latent tokens regardless of input length.
#[derive(Debug, Clone)]
pub struct PerceiverResampler {
    /// Number of output latent tokens.
    pub n_latents: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Learned latent queries: [n_latents, d_model].
    latent_queries: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
}

impl PerceiverResampler {
    /// Create a new perceiver resampler.
    pub fn new(n_latents: usize, d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            n_latents,
            d_model,
            latent_queries: rand_mat(n_latents, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
        }
    }

    /// Resample visual features to `n_latents` fixed-size latents.
    /// `vis_features`: [n_vis, d_model] → output: [n_latents, d_model].
    pub fn resample(&self, vis_features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if vis_features.is_empty() {
            return self.latent_queries.clone();
        }
        let k: Vec<Vec<f64>> = vis_features.iter().map(|x| matvec(&self.wk, x)).collect();
        let v: Vec<Vec<f64>> = vis_features.iter().map(|x| matvec(&self.wv, x)).collect();
        let attn = sdp_attention(&self.latent_queries, &k, &v);
        attn.iter().map(|a| matvec(&self.wo, a)).collect()
    }
}

// ── 2. LLaVA-Style Architecture ────────────────────────────────────────────

/// ViT-based vision encoder: processes image patches → CLS + patch features.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    /// Patch size (square patches).
    pub patch_size: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Patch projection weight: [d_model, patch_dim].
    patch_proj: Vec<Vec<f64>>,
    /// CLS token: [d_model].
    cls_token: Vec<f64>,
    /// Single transformer block for simplicity.
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    ff_w1: Vec<Vec<f64>>,
    ff_w2: Vec<Vec<f64>>,
}

impl VisionEncoder {
    /// Create a new vision encoder.
    pub fn new(patch_size: usize, d_model: usize, n_heads: usize, rng: &mut StdRng) -> Self {
        let patch_dim = patch_size * patch_size * 3; // RGB patches
        Self {
            patch_size,
            d_model,
            n_heads,
            patch_proj: rand_mat(d_model, patch_dim, rng),
            cls_token: rand_vec(d_model, rng),
            wq: rand_mat(d_model, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
            ff_w1: rand_mat(d_model * 4, d_model, rng),
            ff_w2: rand_mat(d_model, d_model * 4, rng),
        }
    }

    /// Returns CLS token + patch embeddings: [1 + n_patches, d_model].
    pub fn encode(&self, patches: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut tokens: Vec<Vec<f64>> = std::iter::once(self.cls_token.clone())
            .chain(patches.iter().map(|p| matvec(&self.patch_proj, p)))
            .collect();
        for (i, tok) in tokens.iter_mut().enumerate() {
            let pe = sinusoidal_pe(i, self.d_model);
            for (t, p) in tok.iter_mut().zip(pe.iter()) {
                *t += p;
            }
        }
        let q: Vec<Vec<f64>> = tokens.iter().map(|x| matvec(&self.wq, x)).collect();
        let k: Vec<Vec<f64>> = tokens.iter().map(|x| matvec(&self.wk, x)).collect();
        let v: Vec<Vec<f64>> = tokens.iter().map(|x| matvec(&self.wv, x)).collect();
        let proj: Vec<Vec<f64>> = sdp_attention(&q, &k, &v)
            .iter()
            .map(|a| matvec(&self.wo, a))
            .collect();
        tokens
            .iter()
            .zip(proj.iter())
            .map(|(t, p)| {
                let res: Vec<f64> = t.iter().zip(p.iter()).map(|(a, b)| a + b).collect();
                let normed = layer_norm(&res);
                let h: Vec<f64> = matvec(&self.ff_w1, &normed)
                    .iter()
                    .map(|v| gelu(*v))
                    .collect();
                let ff: Vec<f64> = matvec(&self.ff_w2, &h);
                layer_norm(
                    &normed
                        .iter()
                        .zip(ff.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

/// 2-layer MLP connector between visual encoder and language model spaces.
#[derive(Debug, Clone)]
pub struct MlpProjector {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    /// Output dimension.
    pub out_dim: usize,
}

impl MlpProjector {
    /// Create a new 2-layer MLP projector.
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            w1: rand_mat(hidden_dim, in_dim, rng),
            b1: vec![0.0; hidden_dim],
            w2: rand_mat(out_dim, hidden_dim, rng),
            b2: vec![0.0; out_dim],
            out_dim,
        }
    }

    /// Project input `x` through two linear layers with GELU activation.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        let h1: Vec<f64> = matvec(&self.w1, x)
            .iter()
            .zip(self.b1.iter())
            .map(|(v, b)| gelu(v + b))
            .collect();
        matvec(&self.w2, &h1)
            .iter()
            .zip(self.b2.iter())
            .map(|(v, b)| v + b)
            .collect()
    }
}

/// LLaVA model: VisionEncoder + MlpProjector + causal language decoder.
#[derive(Debug, Clone)]
pub struct LlavaModel {
    /// Vision encoder component.
    pub vision_encoder: VisionEncoder,
    /// MLP projector mapping visual to language space.
    pub projector: MlpProjector,
    /// Language model vocabulary size for output head.
    pub vocab_size: usize,
    lm_head: Vec<Vec<f64>>,
}

impl LlavaModel {
    /// Create a new LLaVA model.
    pub fn new(
        patch_size: usize,
        vis_d_model: usize,
        n_heads: usize,
        hidden_dim: usize,
        lang_d_model: usize,
        vocab_size: usize,
        rng: &mut StdRng,
    ) -> Self {
        let vision_encoder = VisionEncoder::new(patch_size, vis_d_model, n_heads, rng);
        let projector = MlpProjector::new(vis_d_model, hidden_dim, lang_d_model, rng);
        let lm_head = rand_mat(vocab_size, lang_d_model, rng);
        Self {
            vision_encoder,
            projector,
            vocab_size,
            lm_head,
        }
    }

    /// Forward pass: encode image patches and project, then run over text tokens.
    /// Returns logits over vocabulary for each position: [n_positions, vocab_size].
    pub fn forward(&self, patches: &[Vec<f64>], _tokens: &[usize]) -> Vec<Vec<f64>> {
        // Encode visual tokens
        let vis_tokens = self.vision_encoder.encode(patches);
        // Project each visual token
        let projected: Vec<Vec<f64>> = vis_tokens
            .iter()
            .map(|t| self.projector.project(t))
            .collect();
        // For each projected token, compute logits via lm_head
        projected.iter().map(|t| matvec(&self.lm_head, t)).collect()
    }
}

/// Format `<image>` token + system prompt + user instruction for LLaVA inference.
#[derive(Debug, Clone)]
pub struct ImageInstructionFormatter {
    /// The image token marker string.
    pub image_token: String,
    /// System prompt prepended to every conversation.
    pub system_prompt: String,
}

impl ImageInstructionFormatter {
    /// Create a new formatter with the given system prompt.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            image_token: "<image>".to_string(),
            system_prompt: system_prompt.into(),
        }
    }

    /// Returns formatted string: "\[SYSTEM\] {system}\n{image_token}\n\[USER\] {instruction}"
    pub fn format(&self, instruction: &str) -> String {
        format!(
            "[SYSTEM] {}\n{}\n[USER] {}",
            self.system_prompt, self.image_token, instruction
        )
    }

    /// Returns a (token_segments, image_positions) pair indicating which
    /// positions are image tokens vs. text tokens.
    pub fn tokenize_with_positions(&self, instruction: &str) -> (Vec<String>, Vec<bool>) {
        let formatted = self.format(instruction);
        let segments: Vec<String> = formatted.split_whitespace().map(String::from).collect();
        let is_image: Vec<bool> = segments.iter().map(|s| s == &self.image_token).collect();
        (segments, is_image)
    }
}

/// Next-token prediction loss computed only on text tokens (not image tokens).
#[derive(Debug, Clone)]
pub struct LlavaLoss;

impl LlavaLoss {
    /// Create a new LLaVA loss instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute cross-entropy loss only on text-token positions.
    /// `logits`: [seq_len, vocab_size], `targets`: \[seq_len\], `is_image`: mask.
    pub fn compute(
        &self,
        logits: &[Vec<f64>],
        targets: &[usize],
        is_image: &[bool],
    ) -> Result<f64, TensorError> {
        if logits.len() != targets.len() || logits.len() != is_image.len() {
            return Err(TensorError::invalid_argument_op(
                "LlavaLoss::compute",
                "logits, targets, and is_image must have equal length",
            ));
        }
        let mut loss = 0.0_f64;
        let mut count = 0usize;
        for ((row, &t), &img) in logits.iter().zip(targets.iter()).zip(is_image.iter()) {
            if img {
                continue; // skip image tokens
            }
            let sm = softmax(row);
            let p = if t < sm.len() { sm[t] } else { 1e-12 };
            loss -= p.max(1e-12).ln();
            count += 1;
        }
        if count == 0 {
            return Ok(0.0);
        }
        Ok(loss / count as f64)
    }
}

impl Default for LlavaLoss {
    fn default() -> Self {
        Self::new()
    }
}

// ── 3. Audio-Visual Models ─────────────────────────────────────────────────

/// Spectrogram feature extraction from a raw waveform.
#[derive(Debug, Clone)]
pub struct AudioSpectrogram {
    /// FFT window size.
    pub n_fft: usize,
    /// Hop size between frames.
    pub hop: usize,
}

impl AudioSpectrogram {
    /// Create a new spectrogram extractor.
    pub fn new(n_fft: usize, hop: usize) -> Self {
        Self { n_fft, hop }
    }

    /// Extract log-magnitude spectrogram frames.
    /// Returns `[n_frames, n_fft/2 + 1]` frames.
    pub fn extract(&self, waveform: &[f64], n_fft: usize, hop: usize) -> Vec<Vec<f64>> {
        let n_fft = if n_fft == 0 { self.n_fft } else { n_fft };
        let hop = if hop == 0 { self.hop } else { hop };
        if waveform.is_empty() || n_fft == 0 || hop == 0 {
            return Vec::new();
        }
        let n_bins = n_fft / 2 + 1;
        let n_frames = if waveform.len() >= n_fft {
            (waveform.len() - n_fft) / hop + 1
        } else {
            0
        };

        (0..n_frames)
            .map(|f| {
                let start = f * hop;
                let frame = &waveform[start..(start + n_fft).min(waveform.len())];
                (0..n_bins)
                    .map(|k| {
                        let (re, im): (f64, f64) =
                            frame
                                .iter()
                                .enumerate()
                                .fold((0.0, 0.0), |(re, im), (n, &s)| {
                                    let a = 2.0 * std::f64::consts::PI * k as f64 * n as f64
                                        / n_fft as f64;
                                    (re + s * a.cos(), im - s * a.sin())
                                });
                        ((re * re + im * im).sqrt() + 1e-8).ln()
                    })
                    .collect()
            })
            .collect()
    }
}

/// Cross-modal attention between audio features and video features.
#[derive(Debug, Clone)]
pub struct AudioVisualAttention {
    /// Model dimension.
    pub d_model: usize,
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
}

impl AudioVisualAttention {
    /// Create a new audio-visual attention layer.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            wq: rand_mat(d_model, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
        }
    }

    /// Attend audio features to video features.
    pub fn attend(&self, audio_feats: &[Vec<f64>], video_feats: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if audio_feats.is_empty() || video_feats.is_empty() {
            return audio_feats.to_vec();
        }
        let q: Vec<Vec<f64>> = audio_feats.iter().map(|x| matvec(&self.wq, x)).collect();
        let k: Vec<Vec<f64>> = video_feats.iter().map(|x| matvec(&self.wk, x)).collect();
        let v: Vec<Vec<f64>> = video_feats.iter().map(|x| matvec(&self.wv, x)).collect();
        let attn = sdp_attention(&q, &k, &v);
        attn.iter().map(|a| matvec(&self.wo, a)).collect()
    }
}

/// Joint audio-visual encoder with modality-specific positional encoding.
#[derive(Debug, Clone)]
pub struct AvEncoder {
    /// Model dimension.
    pub d_model: usize,
    /// Number of audio tokens.
    pub n_audio_tokens: usize,
    /// Number of video tokens.
    pub n_video_tokens: usize,
    audio_proj: Vec<Vec<f64>>,
    video_proj: Vec<Vec<f64>>,
    cross_attn: AudioVisualAttention,
}

impl AvEncoder {
    /// Create a new joint audio-visual encoder.
    pub fn new(
        audio_in: usize,
        video_in: usize,
        d_model: usize,
        n_audio_tokens: usize,
        n_video_tokens: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            d_model,
            n_audio_tokens,
            n_video_tokens,
            audio_proj: rand_mat(d_model, audio_in, rng),
            video_proj: rand_mat(d_model, video_in, rng),
            cross_attn: AudioVisualAttention::new(d_model, rng),
        }
    }

    /// Encode audio and video features jointly.
    pub fn encode(&self, audio_feats: &[Vec<f64>], video_feats: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let audio_emb: Vec<Vec<f64>> = audio_feats
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let proj = matvec(&self.audio_proj, a);
                let pe = sinusoidal_pe(i, self.d_model);
                proj.iter().zip(pe.iter()).map(|(p, e)| p + e).collect()
            })
            .collect();
        let video_emb: Vec<Vec<f64>> = video_feats
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let proj = matvec(&self.video_proj, v);
                let pe = sinusoidal_pe(i + 1000, self.d_model); // offset for modality distinction
                proj.iter().zip(pe.iter()).map(|(p, e)| p + e).collect()
            })
            .collect();
        self.cross_attn.attend(&audio_emb, &video_emb)
    }
}

/// Audio-visual contrastive loss: positive pairs = same clip, negatives = different.
#[derive(Debug, Clone)]
pub struct AvContrastiveLoss {
    /// Temperature for InfoNCE scaling.
    pub temperature: f64,
}

impl AvContrastiveLoss {
    /// Create a new AV contrastive loss with the given temperature.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Compute symmetric InfoNCE loss between audio and video embeddings.
    pub fn compute(
        &self,
        audio_embeds: &[Vec<f64>],
        video_embeds: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        let n = audio_embeds.len();
        if n == 0 || n != video_embeds.len() {
            return Err(TensorError::invalid_argument_op(
                "AvContrastiveLoss",
                "batch sizes must match and be non-zero",
            ));
        }
        let t = self.temperature.max(1e-8);
        let mut sim = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let an = l2_normalize(&audio_embeds[i]);
            for j in 0..n {
                let vn = l2_normalize(&video_embeds[j]);
                sim[i][j] = dot(&an, &vn) / t;
            }
        }
        let mut loss = 0.0_f64;
        for i in 0..n {
            let sm_a = softmax(&sim[i]);
            loss -= sm_a[i].max(1e-12).ln();
            let col: Vec<f64> = (0..n).map(|j| sim[j][i]).collect();
            let sm_v = softmax(&col);
            loss -= sm_v[i].max(1e-12).ln();
        }
        Ok(loss / (2.0 * n as f64))
    }
}

/// Align spoken words (audio segments) to visual regions via attention.
#[derive(Debug, Clone)]
pub struct SpeechVisualGrounding {
    /// Model dimension.
    pub d_model: usize,
    audio_proj: Vec<Vec<f64>>,
    visual_proj: Vec<Vec<f64>>,
}

impl SpeechVisualGrounding {
    /// Create a new speech-visual grounder.
    pub fn new(audio_dim: usize, visual_dim: usize, d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            audio_proj: rand_mat(d_model, audio_dim, rng),
            visual_proj: rand_mat(d_model, visual_dim, rng),
        }
    }

    /// Compute attention weights from audio words to visual regions.
    pub fn ground(&self, audio_words: &[Vec<f64>], visual_regions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let aq: Vec<Vec<f64>> = audio_words
            .iter()
            .map(|a| matvec(&self.audio_proj, a))
            .collect();
        let vk: Vec<Vec<f64>> = visual_regions
            .iter()
            .map(|v| matvec(&self.visual_proj, v))
            .collect();
        let scale = (self.d_model as f64).sqrt();
        aq.iter()
            .map(|q| softmax(&vk.iter().map(|k| dot(q, k) / scale).collect::<Vec<_>>()))
            .collect()
    }
}

// ── 4. Document Understanding ──────────────────────────────────────────────

/// Layout embedding that maps bounding box coordinates to a dense vector.
#[derive(Debug, Clone)]
pub struct LayoutEmbedding {
    /// Output dimension.
    pub d_model: usize,
    weight: Vec<Vec<f64>>,
}
impl LayoutEmbedding {
    /// Create a new layout embedding layer.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            weight: rand_mat(d_model, 4, rng),
        }
    }
    /// Embed a bounding box [x, y, w, h] into a dense vector.
    pub fn embed(&self, bbox: &[f64; 4]) -> Vec<f64> {
        matvec(&self.weight, bbox)
    }
}

/// 2D position-aware attention: adds layout (x,y,w,h) embeddings to Q and K.
#[derive(Debug, Clone)]
pub struct LayoutAwareAttention {
    /// Model dimension.
    pub d_model: usize,
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    layout_emb: LayoutEmbedding,
}

impl LayoutAwareAttention {
    /// Create a new layout-aware attention layer.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            wq: rand_mat(d_model, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
            layout_emb: LayoutEmbedding::new(d_model, rng),
        }
    }

    /// Forward pass with layout-aware Q and K computation.
    pub fn forward(&self, tokens: &[Vec<f64>], bboxes: &[[f64; 4]]) -> Vec<Vec<f64>> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let n = tokens.len().min(bboxes.len());
        let q: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let qv = matvec(&self.wq, &tokens[i]);
                let lv = self.layout_emb.embed(&bboxes[i]);
                qv.iter().zip(lv.iter()).map(|(a, b)| a + b).collect()
            })
            .collect();
        let k: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let kv = matvec(&self.wk, &tokens[i]);
                let lv = self.layout_emb.embed(&bboxes[i]);
                kv.iter().zip(lv.iter()).map(|(a, b)| a + b).collect()
            })
            .collect();
        let v: Vec<Vec<f64>> = (0..n).map(|i| matvec(&self.wv, &tokens[i])).collect();
        let attn = sdp_attention(&q, &k, &v);
        let out: Vec<Vec<f64>> = attn.iter().map(|a| matvec(&self.wo, a)).collect();
        (0..n)
            .map(|i| {
                let res: Vec<f64> = tokens[i]
                    .iter()
                    .zip(out[i].iter())
                    .map(|(a, b)| a + b)
                    .collect();
                layer_norm(&res)
            })
            .collect()
    }
}

/// Tokenize a document with layout: returns (token_ids, position_embeddings).
#[derive(Debug, Clone)]
pub struct DocumentTokenizer {
    /// Token vocabulary mapping words to IDs.
    pub vocab: HashMap<String, usize>,
    /// Embedding dimension.
    pub d_model: usize,
    layout_emb: LayoutEmbedding,
}

impl DocumentTokenizer {
    /// Create a new document tokenizer with basic special tokens.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        Self {
            vocab,
            d_model,
            layout_emb: LayoutEmbedding::new(d_model, rng),
        }
    }

    /// Register words into vocabulary (used in tests / training).
    pub fn add_tokens(&mut self, tokens: &[&str]) {
        let next_id = self.vocab.len();
        for (i, &tok) in tokens.iter().enumerate() {
            self.vocab.entry(tok.to_string()).or_insert(next_id + i);
        }
    }

    /// Tokenize: (text_words, bbox) pairs → (token_ids, position_embeds).
    /// `text_bbox_pairs`: list of (word, [x, y, w, h]).
    pub fn tokenize(&self, text_bbox_pairs: &[(String, [f64; 4])]) -> (Vec<usize>, Vec<Vec<f64>>) {
        let token_ids: Vec<usize> = text_bbox_pairs
            .iter()
            .map(|(word, _)| {
                *self
                    .vocab
                    .get(word.as_str())
                    .unwrap_or_else(|| self.vocab.get("[UNK]").unwrap_or(&1))
            })
            .collect();
        let position_embeds: Vec<Vec<f64>> = text_bbox_pairs
            .iter()
            .map(|(_, bbox)| self.layout_emb.embed(bbox))
            .collect();
        (token_ids, position_embeds)
    }
}
