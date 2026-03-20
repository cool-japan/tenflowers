//! Advanced audio ML algorithms: foundation models, neural codecs, MIR, and TTS components.
//!
//! ## New algorithms
//!
//! - **HubertModel** (Hsu 2021) — self-supervised speech representation via offline k-means
//! - **Data2VecAudio** (Baevski 2022) — teacher-student EMA framework
//! - **UniSpeechEncoder** — unified speech representation with CTC auxiliary loss
//! - **RvqCodebook** — Residual Vector Quantization with straight-through estimator
//! - **SoundStreamCodec** — streaming audio codec (encoder + RVQ + decoder)
//! - **DacCodec** — Descript Audio Codec with snake activation
//! - **BeatTracker** — onset-strength + DP beat tracking
//! - **ChordRecognizer** — chromagram + HMM chord recognition (Viterbi)
//! - **MusicSeparator** — harmonic-percussive source separation
//! - **KeyDetector** — Krumhansl-Schmuckler key-finding
//! - **FastSpeech2Duration** — non-autoregressive TTS duration predictor
//! - **LengthRegulatorAdv** — advanced length regulation with fractional durations
//! - **VocoderHiFi** — HiFi-GAN style vocoder generator
//! - **TtsMetrics** — MOS proxy, WER-based intelligibility, MCD

use super::{
    conv1d_same, gelu, he_init, inval, l2_norm, layer_norm_rows, linear_fwd, mha_fwd, relu,
    softmax_inplace,
};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─── Utility ────────────────────────────────────────────────────────────────

/// Snake activation: x + sin²(a·x) / a — anti-aliased periodic nonlinearity.
#[inline]
fn snake(x: f32, a: f32) -> f32 {
    let ax = a * x;
    x + ax.sin() * ax.sin() / a.max(1e-8)
}

/// Cosine distance (1 - cosine similarity) between two vectors.
#[inline]
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm(a).max(1e-8);
    let nb = l2_norm(b).max(1e-8);
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot / (na * nb)
}

/// Nearest centroid index in flat codebook `[k, d]`.
fn nearest_centroid(x: &[f32], codebook: &[f32], k: usize, d: usize) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for i in 0..k {
        let c = &codebook[i * d..(i + 1) * d];
        let dist: f32 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    best
}

// ─── Audio Foundation Models ──────────────────────────────────────────────

/// HuBERT (Hsu 2021): self-supervised speech representation learning.
///
/// Uses offline k-means clustering of MFCC features to generate pseudo-label
/// targets. A transformer encoder is trained with a masked prediction loss
/// against the cluster assignments.
#[derive(Debug, Clone)]
pub struct HubertModel {
    /// Number of cluster centroids for pseudo-labels.
    pub n_clusters: usize,
    /// Feature dimension.
    pub feat_dim: usize,
    /// K-means centroids: `[n_clusters, feat_dim]` flat.
    pub centroids: Vec<f32>,
    /// Transformer encoder QKV weights per layer.
    pub qkv_w: Vec<Vec<f32>>,
    /// Transformer encoder output projection weights per layer.
    pub out_w: Vec<Vec<f32>>,
    /// Transformer FFN weights per layer (w1, b1, w2, b2).
    pub ffn: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// FFN inner dimension.
    pub ffn_dim: usize,
    /// Projection head weights for masked prediction: `[n_clusters, feat_dim]`.
    pub head_w: Vec<f32>,
    /// Projection head biases.
    pub head_b: Vec<f32>,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl HubertModel {
    /// Construct a HuBERT model with random weights.
    pub fn new(n_clusters: usize, feat_dim: usize, n_layers: usize, n_heads: usize, seed: u64) -> Self {
        let ffn_dim = feat_dim * 4;
        let mut qkv_w = Vec::with_capacity(n_layers);
        let mut out_w = Vec::with_capacity(n_layers);
        let mut ffn = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let b = seed.wrapping_add(i as u64 * 100);
            qkv_w.push(he_init(3 * feat_dim * feat_dim, feat_dim, b));
            out_w.push(he_init(feat_dim * feat_dim, feat_dim, b + 1));
            ffn.push((
                he_init(ffn_dim * feat_dim, feat_dim, b + 2),
                vec![0.0_f32; ffn_dim],
                he_init(feat_dim * ffn_dim, ffn_dim, b + 3),
                vec![0.0_f32; feat_dim],
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed + 9000);
        let centroids: Vec<f32> = (0..n_clusters * feat_dim)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let head_w = he_init(n_clusters * feat_dim, feat_dim, seed + 9001);
        let head_b = vec![0.0_f32; n_clusters];
        Self {
            n_clusters,
            feat_dim,
            centroids,
            qkv_w,
            out_w,
            ffn,
            n_layers,
            ffn_dim,
            head_w,
            head_b,
            n_heads,
        }
    }

    /// Assign pseudo-labels via nearest-centroid for each frame in `[T, D]`.
    pub fn cluster_assign(&self, features: &[f32]) -> Result<Vec<usize>> {
        let d = self.feat_dim;
        if features.len() % d != 0 {
            return Err(inval("HubertModel::cluster_assign", "shape mismatch"));
        }
        let t = features.len() / d;
        let labels: Vec<usize> = (0..t)
            .map(|ti| {
                nearest_centroid(
                    &features[ti * d..(ti + 1) * d],
                    &self.centroids,
                    self.n_clusters,
                    d,
                )
            })
            .collect();
        Ok(labels)
    }

    /// Forward encoder: `[T, D]` → `[T, D]` contextual representations.
    pub fn encode(&self, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.feat_dim;
        if x.len() % d != 0 {
            return Err(inval("HubertModel::encode", "shape mismatch"));
        }
        let t = x.len() / d;
        let mut h = x.to_vec();
        for li in 0..self.n_layers {
            let hn = layer_norm_rows(&h, d, 1e-5);
            let attn = mha_fwd(&hn, t, d, self.n_heads, &self.qkv_w[li], &self.out_w[li], false);
            for i in 0..h.len() {
                h[i] += attn[i];
            }
            let hn2 = layer_norm_rows(&h, d, 1e-5);
            let (w1, b1, w2, b2) = &self.ffn[li];
            let f1: Vec<f32> = linear_fwd(&hn2, w1, b1, d, self.ffn_dim)
                .into_iter()
                .map(gelu)
                .collect();
            let f2 = linear_fwd(&f1, w2, b2, self.ffn_dim, d);
            for i in 0..h.len() {
                h[i] += f2[i];
            }
        }
        Ok(h)
    }

    /// Compute masked prediction loss (cross-entropy over pseudo-labels at masked positions).
    pub fn masked_prediction_loss(
        &self,
        features: &[f32],
        mask: &[bool],
    ) -> Result<f32> {
        let d = self.feat_dim;
        if features.len() % d != 0 || mask.is_empty() {
            return Err(inval("HubertModel::masked_prediction_loss", "shape mismatch"));
        }
        let targets = self.cluster_assign(features)?;
        let encoded = self.encode(features)?;
        let t = mask.len().min(encoded.len() / d);
        let mut loss = 0.0_f32;
        let mut cnt = 0usize;
        for ti in 0..t {
            if !mask[ti] {
                continue;
            }
            let rep = &encoded[ti * d..(ti + 1) * d];
            let mut logits = linear_fwd(rep, &self.head_w, &self.head_b, d, self.n_clusters);
            softmax_inplace(&mut logits);
            let p = logits[targets[ti]].max(1e-10);
            loss -= p.ln();
            cnt += 1;
        }
        Ok(if cnt > 0 { loss / cnt as f32 } else { 0.0 })
    }
}

/// Data2Vec Audio (Baevski 2022): teacher-student EMA self-supervised learning.
///
/// The student encoder's top-K layer outputs are averaged and normalized by an
/// EMA teacher; the student regresses those targets with a cosine similarity loss.
#[derive(Debug, Clone)]
pub struct Data2VecAudio {
    /// Feature dimension.
    pub feat_dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Student encoder QKV weights per layer.
    pub student_qkv: Vec<Vec<f32>>,
    /// Student encoder output projection weights per layer.
    pub student_out: Vec<Vec<f32>>,
    /// Teacher (EMA) QKV weights per layer.
    pub teacher_qkv: Vec<Vec<f32>>,
    /// Teacher (EMA) output projection weights per layer.
    pub teacher_out: Vec<Vec<f32>>,
    /// EMA decay rate.
    pub ema_decay: f32,
    /// Number of top-K layers to average for teacher targets.
    pub top_k: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl Data2VecAudio {
    /// Create a new Data2VecAudio model.
    pub fn new(feat_dim: usize, n_layers: usize, n_heads: usize, ema_decay: f32, top_k: usize, seed: u64) -> Self {
        let mut student_qkv = Vec::with_capacity(n_layers);
        let mut student_out = Vec::with_capacity(n_layers);
        let mut teacher_qkv = Vec::with_capacity(n_layers);
        let mut teacher_out = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let b = seed.wrapping_add(i as u64 * 50);
            student_qkv.push(he_init(3 * feat_dim * feat_dim, feat_dim, b));
            student_out.push(he_init(feat_dim * feat_dim, feat_dim, b + 1));
            teacher_qkv.push(student_qkv[i].clone());
            teacher_out.push(student_out[i].clone());
        }
        Self {
            feat_dim,
            n_layers,
            student_qkv,
            student_out,
            teacher_qkv,
            teacher_out,
            ema_decay,
            top_k: top_k.min(n_layers).max(1),
            n_heads,
        }
    }

    /// Run student or teacher encoder, returning all intermediate layer outputs.
    fn encode_layers(&self, x: &[f32], use_teacher: bool) -> Result<Vec<Vec<f32>>> {
        let d = self.feat_dim;
        if x.len() % d != 0 {
            return Err(inval("Data2VecAudio::encode_layers", "shape mismatch"));
        }
        let t = x.len() / d;
        let mut h = x.to_vec();
        let mut layers_out = Vec::with_capacity(self.n_layers);
        for li in 0..self.n_layers {
            let (qkv, out_p) = if use_teacher {
                (&self.teacher_qkv[li], &self.teacher_out[li])
            } else {
                (&self.student_qkv[li], &self.student_out[li])
            };
            let hn = layer_norm_rows(&h, d, 1e-5);
            let attn = mha_fwd(&hn, t, d, self.n_heads, qkv, out_p, false);
            for i in 0..h.len() {
                h[i] += attn[i];
            }
            layers_out.push(h.clone());
        }
        Ok(layers_out)
    }

    /// Compute teacher targets: average of top-K layer outputs, instance-normalized.
    pub fn teacher_targets(&self, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.feat_dim;
        let layers = self.encode_layers(x, true)?;
        let n = layers.len();
        let start = n.saturating_sub(self.top_k);
        let len = x.len();
        let mut avg = vec![0.0_f32; len];
        let count = (n - start) as f32;
        for li in start..n {
            for i in 0..len {
                avg[i] += layers[li][i];
            }
        }
        for v in avg.iter_mut() {
            *v /= count;
        }
        // Instance normalization per frame
        Ok(layer_norm_rows(&avg, d, 1e-5))
    }

    /// Compute cosine similarity regression loss between student output and teacher targets.
    pub fn regression_loss(&self, x: &[f32], mask: &[bool]) -> Result<f32> {
        let d = self.feat_dim;
        if x.len() % d != 0 || mask.is_empty() {
            return Err(inval("Data2VecAudio::regression_loss", "shape mismatch"));
        }
        let student_layers = self.encode_layers(x, false)?;
        let student_out = student_layers.last().ok_or_else(|| inval("Data2VecAudio", "no layers"))?;
        let targets = self.teacher_targets(x)?;
        let t = mask.len().min(student_out.len() / d);
        let mut loss = 0.0_f32;
        let mut cnt = 0usize;
        for ti in 0..t {
            if !mask[ti] {
                continue;
            }
            let s = &student_out[ti * d..(ti + 1) * d];
            let tgt = &targets[ti * d..(ti + 1) * d];
            loss += cosine_dist(s, tgt);
            cnt += 1;
        }
        Ok(if cnt > 0 { loss / cnt as f32 } else { 0.0 })
    }

    /// Update teacher weights via EMA: θ_teacher ← decay·θ_teacher + (1−decay)·θ_student.
    pub fn ema_update(&mut self) {
        let decay = self.ema_decay;
        for li in 0..self.n_layers {
            for (t, s) in self.teacher_qkv[li].iter_mut().zip(&self.student_qkv[li]) {
                *t = decay * *t + (1.0 - decay) * s;
            }
            for (t, s) in self.teacher_out[li].iter_mut().zip(&self.student_out[li]) {
                *t = decay * *t + (1.0 - decay) * s;
            }
        }
    }
}

/// UniSpeech encoder: unified speech representation with phone-level CTC auxiliary loss.
///
/// A conformer-style encoder with a CTC head for auxiliary phone recognition.
#[derive(Debug, Clone)]
pub struct UniSpeechEncoder {
    /// Feature dimension.
    pub feat_dim: usize,
    /// Number of encoder layers.
    pub n_layers: usize,
    /// Number of phone classes (for CTC).
    pub n_phones: usize,
    /// QKV weights per layer.
    pub qkv_w: Vec<Vec<f32>>,
    /// Output projection per layer.
    pub out_w: Vec<Vec<f32>>,
    /// CTC head weights: `[n_phones, feat_dim]`.
    pub ctc_w: Vec<f32>,
    /// CTC head biases.
    pub ctc_b: Vec<f32>,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl UniSpeechEncoder {
    /// Create a new UniSpeech encoder.
    pub fn new(feat_dim: usize, n_layers: usize, n_phones: usize, n_heads: usize, seed: u64) -> Self {
        let mut qkv_w = Vec::with_capacity(n_layers);
        let mut out_w = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let b = seed.wrapping_add(i as u64 * 80);
            qkv_w.push(he_init(3 * feat_dim * feat_dim, feat_dim, b));
            out_w.push(he_init(feat_dim * feat_dim, feat_dim, b + 1));
        }
        let ctc_w = he_init(n_phones * feat_dim, feat_dim, seed + 5000);
        let ctc_b = vec![0.0_f32; n_phones];
        Self {
            feat_dim,
            n_layers,
            n_phones,
            qkv_w,
            out_w,
            ctc_w,
            ctc_b,
            n_heads,
        }
    }

    /// Encode: `[T, D]` → `[T, D]`.
    pub fn encode(&self, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.feat_dim;
        if x.len() % d != 0 {
            return Err(inval("UniSpeechEncoder::encode", "shape mismatch"));
        }
        let t = x.len() / d;
        let mut h = x.to_vec();
        for li in 0..self.n_layers {
            let hn = layer_norm_rows(&h, d, 1e-5);
            let attn = mha_fwd(&hn, t, d, self.n_heads, &self.qkv_w[li], &self.out_w[li], false);
            for i in 0..h.len() {
                h[i] += attn[i];
            }
        }
        Ok(h)
    }

    /// Compute CTC phone logits: `[T, D]` → `[T, n_phones]`.
    pub fn ctc_logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        let encoded = self.encode(x)?;
        Ok(linear_fwd(&encoded, &self.ctc_w, &self.ctc_b, self.feat_dim, self.n_phones))
    }

    /// CTC auxiliary loss: average negative log-likelihood at non-blank positions.
    pub fn ctc_loss(&self, x: &[f32], phone_targets: &[usize]) -> Result<f32> {
        let logits = self.ctc_logits(x)?;
        let t = logits.len() / self.n_phones;
        let t_target = phone_targets.len().min(t);
        let mut loss = 0.0_f32;
        for ti in 0..t_target {
            let mut frame = logits[ti * self.n_phones..(ti + 1) * self.n_phones].to_vec();
            softmax_inplace(&mut frame);
            let target = phone_targets[ti].min(self.n_phones - 1);
            loss -= frame[target].max(1e-10).ln();
        }
        Ok(if t_target > 0 { loss / t_target as f32 } else { 0.0 })
    }
}

// ─── Neural Audio Codecs ─────────────────────────────────────────────────

/// Residual Vector Quantization (RVQ) codebook layer.
///
/// Each layer quantizes the residual from the previous stage.
/// Uses a straight-through estimator for gradient flow.
#[derive(Debug, Clone)]
pub struct RvqCodebook {
    /// Number of quantization stages.
    pub n_stages: usize,
    /// Number of entries per codebook.
    pub codebook_size: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Codebooks: `[n_stages, codebook_size, dim]` flat.
    pub codebooks: Vec<f32>,
    /// Commitment loss weight.
    pub commitment_weight: f32,
}

impl RvqCodebook {
    /// Create a new RVQ codebook.
    pub fn new(n_stages: usize, codebook_size: usize, dim: usize, commitment_weight: f32, seed: u64) -> Self {
        let codebooks = he_init(n_stages * codebook_size * dim, dim, seed);
        Self {
            n_stages,
            codebook_size,
            dim,
            codebooks,
            commitment_weight,
        }
    }

    /// Quantize input `[T, D]` through all RVQ stages.
    ///
    /// Returns (quantized `[T, D]`, commitment_loss, indices per stage `[n_stages, T]`).
    pub fn quantize(&self, x: &[f32]) -> Result<(Vec<f32>, f32, Vec<Vec<usize>>)> {
        let d = self.dim;
        if x.len() % d != 0 {
            return Err(inval("RvqCodebook::quantize", "shape mismatch"));
        }
        let t = x.len() / d;
        let mut residual: Vec<f32> = x.to_vec();
        let mut quantized = vec![0.0_f32; t * d];
        let mut all_indices: Vec<Vec<usize>> = Vec::with_capacity(self.n_stages);
        let mut commitment_loss = 0.0_f32;

        for stage in 0..self.n_stages {
            let cb_off = stage * self.codebook_size * d;
            let cb = &self.codebooks[cb_off..cb_off + self.codebook_size * d];
            let mut stage_indices = Vec::with_capacity(t);
            let mut stage_q = vec![0.0_f32; t * d];
            for ti in 0..t {
                let r = &residual[ti * d..(ti + 1) * d];
                let idx = nearest_centroid(r, cb, self.codebook_size, d);
                stage_indices.push(idx);
                let entry = &cb[idx * d..(idx + 1) * d];
                for j in 0..d {
                    stage_q[ti * d + j] = entry[j];
                    quantized[ti * d + j] += entry[j];
                }
                // Commitment loss: ||residual - quantized||²
                let sq: f32 = r.iter().zip(entry).map(|(a, b)| (a - b) * (a - b)).sum();
                commitment_loss += sq;
            }
            // Update residual: r_{k+1} = r_k - q_k
            for i in 0..residual.len() {
                residual[i] -= stage_q[i];
            }
            all_indices.push(stage_indices);
        }

        let total = (t * self.n_stages) as f32;
        let commitment_loss = self.commitment_weight * commitment_loss / total.max(1.0);
        Ok((quantized, commitment_loss, all_indices))
    }
}

/// SoundStream codec: streaming audio encoder → RVQ → decoder.
///
/// Encoder uses strided convolutions to downsample, decoder uses transposed convolutions.
#[derive(Debug, Clone)]
pub struct SoundStreamCodec {
    /// Input audio channels (1 for mono).
    pub in_channels: usize,
    /// Encoder hidden dimension.
    pub hidden_dim: usize,
    /// RVQ bottleneck.
    pub rvq: RvqCodebook,
    /// Encoder conv weights per layer: `[hidden, in, kernel]`.
    pub enc_weights: Vec<Vec<f32>>,
    /// Encoder conv biases per layer.
    pub enc_biases: Vec<Vec<f32>>,
    /// Decoder transposed conv weights per layer.
    pub dec_weights: Vec<Vec<f32>>,
    /// Decoder transposed conv biases.
    pub dec_biases: Vec<Vec<f32>>,
    /// Strides per encoder layer.
    pub strides: Vec<usize>,
}

impl SoundStreamCodec {
    /// Create a new SoundStream codec.
    pub fn new(hidden_dim: usize, n_stages: usize, codebook_size: usize, seed: u64) -> Self {
        let strides = vec![2usize, 4, 5, 8];
        let channels = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8];
        let mut enc_weights = Vec::new();
        let mut enc_biases = Vec::new();
        let mut dec_weights = Vec::new();
        let mut dec_biases = Vec::new();
        let mut in_ch = 1usize;
        for (i, &out_ch) in channels.iter().enumerate() {
            let ks = strides[i] * 2;
            enc_weights.push(he_init(out_ch * in_ch * ks, in_ch * ks, seed + i as u64 * 10));
            enc_biases.push(vec![0.0_f32; out_ch]);
            dec_weights.push(he_init(in_ch * out_ch * ks, out_ch * ks, seed + 100 + i as u64 * 10));
            dec_biases.push(vec![0.0_f32; in_ch]);
            in_ch = out_ch;
        }
        let rvq = RvqCodebook::new(n_stages, codebook_size, *channels.last().unwrap_or(&hidden_dim), 0.25, seed + 999);
        Self {
            in_channels: 1,
            hidden_dim,
            rvq,
            enc_weights,
            enc_biases,
            dec_weights,
            dec_biases,
            strides,
        }
    }

    /// Encode waveform `[T]` → quantized codes and commitment loss.
    pub fn encode(&self, waveform: &[f32]) -> Result<(Vec<f32>, f32, Vec<Vec<usize>>)> {
        if waveform.is_empty() {
            return Err(inval("SoundStreamCodec::encode", "empty waveform"));
        }
        let mut h: Vec<f32> = waveform.to_vec();
        let mut t = h.len();
        let mut in_ch = 1usize;
        for (li, stride) in self.strides.iter().enumerate() {
            let ks = stride * 2;
            let out_ch = self.enc_weights[li].len() / (in_ch * ks);
            if t < ks {
                return Err(inval("SoundStreamCodec::encode", format!("layer {li}: sequence too short")));
            }
            let t_out = (t - ks) / stride + 1;
            let w = &self.enc_weights[li];
            let b = &self.enc_biases[li];
            let mut out = vec![0.0_f32; t_out * out_ch];
            for ti in 0..t_out {
                let t_start = ti * stride;
                for oc in 0..out_ch {
                    let mut s = b[oc];
                    for ic in 0..in_ch {
                        for ki in 0..ks {
                            s += w[oc * in_ch * ks + ic * ks + ki] * h[(t_start + ki) * in_ch + ic];
                        }
                    }
                    out[ti * out_ch + oc] = relu(s);
                }
            }
            h = out;
            t = t_out;
            in_ch = out_ch;
        }
        self.rvq.quantize(&h)
    }

    /// Compute total codec loss = commitment loss (codebook loss to be added in training).
    pub fn codec_loss(&self, waveform: &[f32]) -> Result<f32> {
        let (_, commitment_loss, _) = self.encode(waveform)?;
        Ok(commitment_loss)
    }
}

/// DAC (Descript Audio Codec, Kumar 2023): uses snake activation and anti-aliased conv.
///
/// Features multi-scale STFT discriminator loss terms and improved codebook utilization.
#[derive(Debug, Clone)]
pub struct DacCodec {
    /// Encoder hidden dimension.
    pub hidden_dim: usize,
    /// RVQ bottleneck.
    pub rvq: RvqCodebook,
    /// Encoder weights per layer (strided conv).
    pub enc_weights: Vec<Vec<f32>>,
    /// Encoder biases per layer.
    pub enc_biases: Vec<Vec<f32>>,
    /// Snake activation alpha per channel per layer.
    pub snake_alphas: Vec<Vec<f32>>,
    /// STFT window sizes for multi-scale loss.
    pub stft_windows: Vec<usize>,
    /// Strides per encoder layer.
    pub strides: Vec<usize>,
}

impl DacCodec {
    /// Create a new DAC codec.
    pub fn new(hidden_dim: usize, n_stages: usize, codebook_size: usize, seed: u64) -> Self {
        let strides = vec![2usize, 4, 8, 8];
        let channels = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8];
        let mut enc_weights = Vec::new();
        let mut enc_biases = Vec::new();
        let mut snake_alphas = Vec::new();
        let mut in_ch = 1usize;
        for (i, &out_ch) in channels.iter().enumerate() {
            let ks = strides[i] * 2;
            enc_weights.push(he_init(out_ch * in_ch * ks, in_ch * ks, seed + i as u64 * 7));
            enc_biases.push(vec![0.0_f32; out_ch]);
            snake_alphas.push(vec![1.0_f32; out_ch]); // learned freq; init to 1.0
            in_ch = out_ch;
        }
        let latent_dim = *channels.last().unwrap_or(&hidden_dim);
        let rvq = RvqCodebook::new(n_stages, codebook_size, latent_dim, 0.25, seed + 777);
        Self {
            hidden_dim,
            rvq,
            enc_weights,
            enc_biases,
            snake_alphas,
            stft_windows: vec![2048, 1024, 512],
            strides,
        }
    }

    /// Compute per-frame STFT magnitude (simplified DFT).
    fn stft_magnitudes(&self, signal: &[f32], win: usize) -> Vec<f32> {
        let hop = win / 4;
        let n_freqs = win / 2 + 1;
        if signal.len() < win {
            return vec![0.0_f32; n_freqs];
        }
        let n_frames = (signal.len() - win) / hop + 1;
        let mut mags = vec![0.0_f32; n_frames * n_freqs];
        let pi2 = 2.0 * std::f32::consts::PI;
        for ti in 0..n_frames {
            let off = ti * hop;
            for k in 0..n_freqs {
                let (mut re, mut im) = (0.0_f32, 0.0_f32);
                for j in 0..win {
                    if off + j < signal.len() {
                        let hann = 0.5 * (1.0 - (pi2 * j as f32 / (win - 1) as f32).cos());
                        let angle = pi2 * k as f32 * j as f32 / win as f32;
                        re += signal[off + j] * hann * angle.cos();
                        im -= signal[off + j] * hann * angle.sin();
                    }
                }
                mags[ti * n_freqs + k] = (re * re + im * im).sqrt();
            }
        }
        mags
    }

    /// Multi-scale STFT reconstruction loss between original and reconstructed signal.
    pub fn stft_loss(&self, original: &[f32], reconstructed: &[f32]) -> f32 {
        let len = original.len().min(reconstructed.len());
        if len == 0 {
            return 0.0;
        }
        let orig = &original[..len];
        let recon = &reconstructed[..len];
        let mut total = 0.0_f32;
        for &win in &self.stft_windows {
            let m1 = self.stft_magnitudes(orig, win);
            let m2 = self.stft_magnitudes(recon, win);
            let n = m1.len().min(m2.len());
            if n == 0 {
                continue;
            }
            let sc: f32 = m1[..n].iter().zip(&m2[..n]).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / n as f32;
            total += sc.sqrt();
        }
        total / self.stft_windows.len() as f32
    }

    /// Encode waveform with snake activation, returning (quantized, commitment_loss, indices).
    pub fn encode(&self, waveform: &[f32]) -> Result<(Vec<f32>, f32, Vec<Vec<usize>>)> {
        if waveform.is_empty() {
            return Err(inval("DacCodec::encode", "empty waveform"));
        }
        let mut h: Vec<f32> = waveform.to_vec();
        let mut t = h.len();
        let mut in_ch = 1usize;
        for (li, stride) in self.strides.iter().enumerate() {
            let ks = stride * 2;
            let out_ch = self.enc_weights[li].len() / (in_ch * ks);
            if t < ks {
                return Err(inval("DacCodec::encode", format!("layer {li}: sequence too short")));
            }
            let t_out = (t - ks) / stride + 1;
            let w = &self.enc_weights[li];
            let b = &self.enc_biases[li];
            let alphas = &self.snake_alphas[li];
            let mut out = vec![0.0_f32; t_out * out_ch];
            for ti in 0..t_out {
                let t_start = ti * stride;
                for oc in 0..out_ch {
                    let mut s = b[oc];
                    for ic in 0..in_ch {
                        for ki in 0..ks {
                            s += w[oc * in_ch * ks + ic * ks + ki] * h[(t_start + ki) * in_ch + ic];
                        }
                    }
                    out[ti * out_ch + oc] = snake(s, alphas[oc]);
                }
            }
            h = out;
            t = t_out;
            in_ch = out_ch;
        }
        self.rvq.quantize(&h)
    }
}

// ─── Music Information Retrieval ─────────────────────────────────────────

/// Beat tracker via onset strength + dynamic programming (Davies & Plumbley 2007).
#[derive(Debug, Clone)]
pub struct BeatTracker {
    /// Expected tempo in beats-per-minute (used as prior).
    pub tempo_prior_bpm: f32,
    /// Audio sample rate.
    pub sample_rate: f32,
    /// Hop size between frames (in samples).
    pub hop_size: usize,
}

impl BeatTracker {
    /// Create a beat tracker with the given tempo prior and sample rate.
    pub fn new(tempo_prior_bpm: f32, sample_rate: f32, hop_size: usize) -> Self {
        Self { tempo_prior_bpm, sample_rate, hop_size }
    }

    /// Compute onset strength function from `[T, F]` magnitude spectrogram.
    ///
    /// Uses half-wave rectified spectral flux.
    pub fn onset_strength(&self, spectrogram: &[f32], n_freqs: usize) -> Vec<f32> {
        let t = spectrogram.len().checked_div(n_freqs).unwrap_or(0);
        if t < 2 {
            return vec![0.0_f32; t];
        }
        let mut odf = vec![0.0_f32; t];
        for ti in 1..t {
            let cur = &spectrogram[ti * n_freqs..(ti + 1) * n_freqs];
            let prev = &spectrogram[(ti - 1) * n_freqs..ti * n_freqs];
            odf[ti] = cur.iter().zip(prev).map(|(c, p)| (c - p).max(0.0)).sum::<f32>() / n_freqs as f32;
        }
        // Normalize
        let max_v = odf.iter().cloned().fold(0.0_f32, f32::max);
        if max_v > 1e-8 {
            odf.iter_mut().for_each(|v| *v /= max_v);
        }
        odf
    }

    /// Track beats in an onset strength function using DP.
    ///
    /// Returns frame indices of detected beats.
    pub fn track_beats(&self, odf: &[f32]) -> Vec<usize> {
        let t = odf.len();
        if t < 2 {
            return Vec::new();
        }
        // Expected period in frames
        let fps = self.sample_rate / self.hop_size as f32;
        let period = (fps * 60.0 / self.tempo_prior_bpm).round() as usize;
        let period = period.max(2);
        // DP: score[t] = odf[t] + max_{t'=t-2*period..t-period/2} (score[t'] - alpha*(1 - period/(t-t')))
        let mut dp = vec![0.0_f32; t];
        let mut prev = vec![0usize; t];
        for i in 0..t {
            dp[i] = odf[i];
            prev[i] = i;
        }
        for i in 1..t {
            let lo = i.saturating_sub(2 * period);
            let hi = i.saturating_sub(period / 2);
            let mut best_score = f32::NEG_INFINITY;
            let mut best_j = lo;
            for j in lo..=hi.min(i.saturating_sub(1)) {
                let gap = (i - j) as f32;
                let penalty = ((gap / period as f32) - 1.0).powi(2);
                let score = dp[j] - penalty;
                if score > best_score {
                    best_score = score;
                    best_j = j;
                }
            }
            if best_score > f32::NEG_INFINITY {
                dp[i] = odf[i] + best_score;
                prev[i] = best_j;
            }
        }
        // Backtrack from best end
        let mut best_end = 0;
        for i in 1..t {
            if dp[i] > dp[best_end] {
                best_end = i;
            }
        }
        let mut beats = Vec::new();
        let mut cur = best_end;
        loop {
            beats.push(cur);
            let p = prev[cur];
            if p == cur {
                break;
            }
            cur = p;
        }
        beats.sort_unstable();
        beats.dedup();
        beats
    }
}

/// Chord recognizer using chromagram features + Viterbi HMM (24 major/minor chords).
#[derive(Debug, Clone)]
pub struct ChordRecognizer {
    /// Chord emission probabilities: `[24, 12]` chromagram templates.
    pub templates: Vec<f32>,
    /// HMM transition matrix: `[24, 24]`.
    pub transitions: Vec<f32>,
    /// Prior probabilities over 24 chords.
    pub prior: Vec<f32>,
}

impl ChordRecognizer {
    /// Create a chord recognizer with Krumhansl chroma templates.
    pub fn new() -> Self {
        // Major and minor chroma templates (Fujishima 1999)
        let major_template = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let minor_template = [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut templates = Vec::with_capacity(24 * 12);
        for root in 0..12usize {
            let mut t = [0.0f32; 12];
            for i in 0..12 {
                t[(i + root) % 12] = major_template[i];
            }
            let norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            templates.extend(t.iter().map(|&x| x / norm));
        }
        for root in 0..12usize {
            let mut t = [0.0f32; 12];
            for i in 0..12 {
                t[(i + root) % 12] = minor_template[i];
            }
            let norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            templates.extend(t.iter().map(|&x| x / norm));
        }
        // Uniform self-transition with small probability of change
        let self_prob = 0.9f32;
        let other_prob = (1.0 - self_prob) / 23.0;
        let mut transitions = vec![other_prob; 24 * 24];
        for i in 0..24 {
            transitions[i * 24 + i] = self_prob;
        }
        let prior = vec![1.0 / 24.0; 24];
        Self { templates, transitions, prior }
    }

    /// Compute chroma vector (12-bin) from a magnitude spectrum `[n_freqs]`.
    pub fn chromagram_frame(&self, spectrum: &[f32], sample_rate: f32, n_fft: usize) -> Vec<f32> {
        let n_freqs = spectrum.len();
        let mut chroma = [0.0_f32; 12];
        let a4_hz = 440.0f32;
        for k in 1..n_freqs {
            let freq = k as f32 * sample_rate / n_fft as f32;
            if freq < 20.0 {
                continue;
            }
            // MIDI pitch
            let midi = 69.0 + 12.0 * (freq / a4_hz).log2();
            let pc = (midi.round() as i32).rem_euclid(12) as usize;
            chroma[pc] += spectrum[k];
        }
        let norm: f32 = chroma.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        chroma.iter().map(|&x| x / norm).collect()
    }

    /// Recognize chords via Viterbi decoding over a sequence of chromagram frames `[T, 12]`.
    ///
    /// Returns chord indices (0-11: C–B major, 12-23: C–B minor).
    pub fn viterbi_decode(&self, chromagrams: &[f32]) -> Result<Vec<usize>> {
        if chromagrams.len() % 12 != 0 {
            return Err(inval("ChordRecognizer::viterbi_decode", "input must be [T, 12]"));
        }
        let t = chromagrams.len() / 12;
        if t == 0 {
            return Ok(Vec::new());
        }
        let n = 24usize;
        let mut viterbi = vec![f32::NEG_INFINITY; t * n];
        let mut backptr = vec![0usize; t * n];

        // Compute emission: cosine similarity with template
        let emission = |frame: &[f32], chord: usize| -> f32 {
            let tmpl = &self.templates[chord * 12..(chord + 1) * 12];
            frame.iter().zip(tmpl).map(|(a, b)| a * b).sum::<f32>().max(0.0)
        };

        let frame0 = &chromagrams[0..12];
        for s in 0..n {
            viterbi[s] = self.prior[s].ln() + (emission(frame0, s) + 1e-10).ln();
        }

        for ti in 1..t {
            let frame = &chromagrams[ti * 12..(ti + 1) * 12];
            for s in 0..n {
                let obs = (emission(frame, s) + 1e-10).ln();
                let mut best_prob = f32::NEG_INFINITY;
                let mut best_prev = 0;
                for prev in 0..n {
                    let p = viterbi[(ti - 1) * n + prev] + (self.transitions[prev * n + s] + 1e-15).ln();
                    if p > best_prob {
                        best_prob = p;
                        best_prev = prev;
                    }
                }
                viterbi[ti * n + s] = best_prob + obs;
                backptr[ti * n + s] = best_prev;
            }
        }

        // Backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = (0..n).max_by(|&a, &b| viterbi[(t - 1) * n + a].partial_cmp(&viterbi[(t - 1) * n + b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
        for ti in (0..t - 1).rev() {
            path[ti] = backptr[(ti + 1) * n + path[ti + 1]];
        }
        Ok(path)
    }
}

impl Default for ChordRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Harmonic-Percussive Source Separation (HPSS) via median filtering on spectrogram.
#[derive(Debug, Clone)]
pub struct MusicSeparator {
    /// Median filter length in time dimension (for harmonic component).
    pub harmonic_len: usize,
    /// Median filter length in frequency dimension (for percussive component).
    pub percussive_len: usize,
}

impl MusicSeparator {
    /// Create a new HPSS separator.
    pub fn new(harmonic_len: usize, percussive_len: usize) -> Self {
        Self { harmonic_len, percussive_len }
    }

    /// Separate a magnitude spectrogram `[T, F]` into harmonic and percussive components.
    ///
    /// Returns (harmonic `[T, F]`, percussive `[T, F]`).
    pub fn separate(&self, spectrogram: &[f32], n_freqs: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = spectrogram.len().checked_div(n_freqs).unwrap_or(0);
        if t == 0 || spectrogram.len() != t * n_freqs {
            return Err(inval("MusicSeparator::separate", "shape mismatch"));
        }

        // Harmonic: median filter along time axis for each frequency bin
        let mut harmonic = vec![0.0_f32; t * n_freqs];
        let hl = self.harmonic_len;
        for fi in 0..n_freqs {
            for ti in 0..t {
                let lo = ti.saturating_sub(hl / 2);
                let hi = (ti + hl / 2 + 1).min(t);
                let mut vals: Vec<f32> = (lo..hi).map(|t2| spectrogram[t2 * n_freqs + fi]).collect();
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                harmonic[ti * n_freqs + fi] = vals[vals.len() / 2];
            }
        }

        // Percussive: median filter along frequency axis for each time frame
        let mut percussive = vec![0.0_f32; t * n_freqs];
        let pl = self.percussive_len;
        for ti in 0..t {
            for fi in 0..n_freqs {
                let lo = fi.saturating_sub(pl / 2);
                let hi = (fi + pl / 2 + 1).min(n_freqs);
                let mut vals: Vec<f32> = (lo..hi).map(|f2| spectrogram[ti * n_freqs + f2]).collect();
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                percussive[ti * n_freqs + fi] = vals[vals.len() / 2];
            }
        }

        // Wiener-like soft masking
        let mut out_h = vec![0.0_f32; t * n_freqs];
        let mut out_p = vec![0.0_f32; t * n_freqs];
        for i in 0..spectrogram.len() {
            let h = harmonic[i];
            let p = percussive[i];
            let total = h + p + 1e-10;
            out_h[i] = spectrogram[i] * h / total;
            out_p[i] = spectrogram[i] * p / total;
        }
        Ok((out_h, out_p))
    }
}

/// Krumhansl-Schmuckler key-finding algorithm.
///
/// Computes a pitch-class distribution and correlates it with major/minor profiles.
#[derive(Debug, Clone)]
pub struct KeyDetector {
    /// Major key profile (Krumhansl & Kessler 1982).
    pub major_profile: Vec<f32>,
    /// Minor key profile.
    pub minor_profile: Vec<f32>,
}

impl KeyDetector {
    /// Create a key detector with the Krumhansl-Kessler tonal hierarchy profiles.
    pub fn new() -> Self {
        // Krumhansl-Kessler profiles (normalized)
        let major = vec![6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88f64];
        let minor = vec![6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17f64];
        let normalize = |v: Vec<f64>| -> Vec<f32> {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let std = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64).sqrt().max(1e-10);
            v.into_iter().map(|x| ((x - mean) / std) as f32).collect()
        };
        Self {
            major_profile: normalize(major),
            minor_profile: normalize(minor),
        }
    }

    /// Detect the most likely key from a pitch-class distribution `[12]`.
    ///
    /// Returns (key_index 0-11, is_minor).
    pub fn detect_key(&self, pcd: &[f32]) -> Result<(usize, bool)> {
        if pcd.len() < 12 {
            return Err(inval("KeyDetector::detect_key", "pcd must have at least 12 elements"));
        }
        let pcd12 = &pcd[..12];
        let mean: f32 = pcd12.iter().sum::<f32>() / 12.0;
        let std: f32 = (pcd12.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 12.0).sqrt().max(1e-10);
        let pcd_norm: Vec<f32> = pcd12.iter().map(|&x| (x - mean) / std).collect();

        let mut best_corr = f32::NEG_INFINITY;
        let mut best_key = 0;
        let mut best_minor = false;

        for root in 0..12usize {
            // Major correlation
            let corr_maj: f32 = (0..12).map(|i| pcd_norm[i] * self.major_profile[(i + 12 - root) % 12]).sum();
            if corr_maj > best_corr {
                best_corr = corr_maj;
                best_key = root;
                best_minor = false;
            }
            // Minor correlation
            let corr_min: f32 = (0..12).map(|i| pcd_norm[i] * self.minor_profile[(i + 12 - root) % 12]).sum();
            if corr_min > best_corr {
                best_corr = corr_min;
                best_key = root;
                best_minor = true;
            }
        }
        Ok((best_key, best_minor))
    }
}

impl Default for KeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Text-to-Speech Components ────────────────────────────────────────────

/// FastSpeech2 duration predictor (non-autoregressive TTS).
///
/// Two-layer depthwise-separable conv + linear head predicts log duration per token.
#[derive(Debug, Clone)]
pub struct FastSpeech2Duration {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Depthwise conv weights layer 1.
    pub dw1_w: Vec<f32>,
    /// Pointwise conv weights layer 1.
    pub pw1_w: Vec<f32>,
    /// Bias layer 1.
    pub b1: Vec<f32>,
    /// Depthwise conv weights layer 2.
    pub dw2_w: Vec<f32>,
    /// Pointwise conv weights layer 2.
    pub pw2_w: Vec<f32>,
    /// Bias layer 2.
    pub b2: Vec<f32>,
    /// Linear head weights `[1, D]`.
    pub head_w: Vec<f32>,
    /// Linear head bias.
    pub head_b: Vec<f32>,
}

impl FastSpeech2Duration {
    /// Create a new FastSpeech2 duration predictor.
    pub fn new(feat_dim: usize, kernel_size: usize, seed: u64) -> Self {
        let d = feat_dim;
        let ks = kernel_size;
        Self {
            feat_dim: d,
            kernel_size: ks,
            dw1_w: he_init(d * ks, ks, seed),
            pw1_w: he_init(d * d, d, seed + 1),
            b1: vec![0.0_f32; d],
            dw2_w: he_init(d * ks, ks, seed + 2),
            pw2_w: he_init(d * d, d, seed + 3),
            b2: vec![0.0_f32; d],
            head_w: he_init(d, d, seed + 4),
            head_b: vec![0.0_f32; 1],
        }
    }

    /// Predict log-durations: `[T, D]` → `[T]` (softplus-activated, positive).
    pub fn predict(&self, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.feat_dim;
        let ks = self.kernel_size;
        if x.len() % d != 0 {
            return Err(inval("FastSpeech2Duration::predict", "shape mismatch"));
        }
        let t = x.len() / d;
        // Layer 1: same-pad depthwise conv per channel + pointwise
        let dw1: Vec<f32> = conv1d_same(x, &self.dw1_w, &vec![0.0_f32; d], t, 1, d, ks);
        let h1: Vec<f32> = linear_fwd(&dw1, &self.pw1_w, &self.b1, d, d)
            .into_iter()
            .map(relu)
            .collect();
        let h1n = layer_norm_rows(&h1, d, 1e-5);
        // Layer 2
        let dw2: Vec<f32> = conv1d_same(&h1n, &self.dw2_w, &vec![0.0_f32; d], t, 1, d, ks);
        let h2: Vec<f32> = linear_fwd(&dw2, &self.pw2_w, &self.b2, d, d)
            .into_iter()
            .map(relu)
            .collect();
        let h2n = layer_norm_rows(&h2, d, 1e-5);
        // Head: scalar per frame, softplus
        Ok((0..t)
            .map(|ti| {
                let row = &h2n[ti * d..(ti + 1) * d];
                let raw: f32 = row.iter().zip(&self.head_w).map(|(a, b)| a * b).sum::<f32>() + self.head_b[0];
                (1.0_f32 + raw.exp()).ln()
            })
            .collect())
    }
}

/// Advanced length regulator: expand encoder outputs by (possibly fractional) predicted durations.
///
/// Fractional durations are rounded to nearest integer ≥ 1.
#[derive(Debug, Clone)]
pub struct LengthRegulatorAdv {
    /// Minimum duration per frame (frames with 0 duration are silenced).
    pub min_dur: usize,
}

impl LengthRegulatorAdv {
    /// Create a new advanced length regulator.
    pub fn new(min_dur: usize) -> Self {
        Self { min_dur }
    }

    /// Regulate `x` (`[T, D]`) using continuous durations, rounding to integers.
    ///
    /// Frames with rounded duration 0 are skipped if `min_dur` == 0, or repeated once.
    pub fn regulate(&self, x: &[f32], durations: &[f32], d: usize) -> Result<Vec<f32>> {
        if x.len() % d != 0 {
            return Err(inval("LengthRegulatorAdv::regulate", "x shape mismatch"));
        }
        let t = x.len() / d;
        if durations.len() != t {
            return Err(inval("LengthRegulatorAdv::regulate", "durations length mismatch"));
        }
        let mut out = Vec::new();
        for (ti, &dur) in durations.iter().enumerate() {
            let n = (dur.round() as usize).max(self.min_dur);
            let frame = &x[ti * d..(ti + 1) * d];
            for _ in 0..n {
                out.extend_from_slice(frame);
            }
        }
        Ok(out)
    }
}

/// HiFi-GAN style vocoder generator (simplified, Kumar 2019).
///
/// Multi-receptive-field fusion using three residual blocks with different dilation patterns,
/// followed by transposed conv upsampling stages.
#[derive(Debug, Clone)]
pub struct VocoderHiFi {
    /// Input mel dimension.
    pub mel_dim: usize,
    /// Initial hidden dimension.
    pub hidden_dim: usize,
    /// Number of upsampling stages.
    pub n_stages: usize,
    /// Upsampling weights per stage.
    pub upsample_w: Vec<Vec<f32>>,
    /// Residual block weights per stage × block (3 MRF blocks per stage).
    pub res_w: Vec<Vec<f32>>,
    /// Dilation patterns per residual block.
    pub dilations: Vec<Vec<usize>>,
    /// Output projection weights `[1, hidden]`.
    pub out_w: Vec<f32>,
}

impl VocoderHiFi {
    /// Create a new HiFi-GAN vocoder generator.
    pub fn new(mel_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let n_stages = 3;
        let mut upsample_w = Vec::with_capacity(n_stages);
        let mut res_w = Vec::with_capacity(n_stages * 3);
        for i in 0..n_stages {
            let in_d = if i == 0 { mel_dim } else { hidden_dim };
            let ks = 2usize.pow((i + 2) as u32);
            upsample_w.push(he_init(hidden_dim * in_d * ks, in_d * ks, seed + i as u64 * 20));
            // 3 MRF residual blocks per stage
            for bi in 0..3usize {
                res_w.push(he_init(hidden_dim * hidden_dim * 3, hidden_dim * 3, seed + i as u64 * 100 + bi as u64));
            }
        }
        let dilations = vec![
            vec![1, 3, 5],
            vec![1, 3, 5],
            vec![1, 3, 5],
        ];
        let out_w = he_init(hidden_dim, hidden_dim, seed + 9999);
        Self {
            mel_dim,
            hidden_dim,
            n_stages,
            upsample_w,
            res_w,
            dilations,
            out_w,
        }
    }

    /// Apply a single MRF residual block with given dilation.
    fn mrf_block(&self, x: &[f32], w: &[f32], dilations: &[usize]) -> Vec<f32> {
        let d = self.hidden_dim;
        if x.len() / d == 0 {
            return x.to_vec();
        }
        let mut h = x.to_vec();
        for &dil in dilations {
            let t = h.len() / d;
            let ks = 3;
            let span = (ks - 1) * dil + 1;
            if t < span {
                continue;
            }
            let t_out = t - span + 1;
            let mut out = vec![0.0_f32; t_out * d];
            for ti in 0..t_out {
                for oc in 0..d {
                    let mut s = 0.0_f32;
                    for ic in 0..d {
                        for ki in 0..ks {
                            let src = ti + ki * dil;
                            s += w[oc * d * ks + ic * ks + ki] * h[src * d + ic];
                        }
                    }
                    out[ti * d + oc] = relu(s);
                }
            }
            h = out;
        }
        h
    }

    /// Generate waveform from mel spectrogram `[T, mel_dim]`.
    ///
    /// Returns a 1D waveform (shape `[T']`).
    pub fn generate(&self, mel: &[f32]) -> Result<Vec<f32>> {
        let d_in = self.mel_dim;
        if mel.len() % d_in != 0 {
            return Err(inval("VocoderHiFi::generate", "mel shape mismatch"));
        }
        let mut h = mel.to_vec();
        let mut t = h.len() / d_in;
        let mut in_d = d_in;

        for stage in 0..self.n_stages {
            let ks = 2usize.pow((stage + 2) as u32);
            let w = &self.upsample_w[stage];
            let out_d = self.hidden_dim;
            // Transposed conv: upsample by factor `ks/2`
            let factor = ks / 2;
            let t_out = t * factor;
            let mut up = vec![0.0_f32; t_out * out_d];
            for ti in 0..t {
                for oc in 0..out_d {
                    let mut s = 0.0_f32;
                    for ic in 0..in_d {
                        for ki in 0..ks {
                            s += w[oc * in_d * ks + ic * ks + ki] * h[ti * in_d + ic];
                        }
                    }
                    let base = ti * factor;
                    for f in 0..factor {
                        if base + f < t_out {
                            up[(base + f) * out_d + oc] += relu(s);
                        }
                    }
                }
            }
            h = up;
            t = t_out;
            in_d = out_d;
            // Apply 3 MRF blocks
            for bi in 0..3usize {
                let idx = stage * 3 + bi;
                let dil = &self.dilations[bi];
                let mrf_out = self.mrf_block(&h, &self.res_w[idx], dil);
                // Use MRF output if shape is preserved, else skip
                if mrf_out.len() == h.len() {
                    for i in 0..h.len() {
                        h[i] = (h[i] + mrf_out[i]) * 0.5;
                    }
                }
            }
        }
        // Final tanh + linear mix to mono
        let out: Vec<f32> = (0..t)
            .map(|ti| {
                let frame = &h[ti * in_d..(ti + 1) * in_d];
                let raw: f32 = frame.iter().zip(&self.out_w).map(|(a, b)| a * b).sum();
                raw.tanh()
            })
            .collect();
        Ok(out)
    }
}

/// TTS evaluation metrics: MCD, WER-based intelligibility proxy, MOS proxy.
#[derive(Debug, Clone, Default)]
pub struct TtsMetrics {
    /// Mel-cepstral distortion (dB) values collected.
    pub mcd_values: Vec<f32>,
    /// WER-based intelligibility scores (lower = better).
    pub wer_scores: Vec<f32>,
}

impl TtsMetrics {
    /// Create a new TtsMetrics accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute Mel-Cepstral Distortion between two mel-cepstrum sequences `[T, D]`.
    ///
    /// MCD (dB) = (10 / ln(10)) * sqrt(2 * sum_d (c1_d - c2_d)^2) per frame, averaged.
    pub fn mel_cepstral_distortion(ref_mfcc: &[f32], syn_mfcc: &[f32], d: usize) -> Result<f32> {
        if ref_mfcc.len() != syn_mfcc.len() || ref_mfcc.len() % d != 0 {
            return Err(inval("TtsMetrics::mel_cepstral_distortion", "shape mismatch"));
        }
        let t = ref_mfcc.len() / d;
        if t == 0 {
            return Ok(0.0);
        }
        let factor = 10.0 / std::f32::consts::LN_10;
        let mcd: f32 = (0..t)
            .map(|ti| {
                let sq: f32 = (1..d.min(13))
                    .map(|j| {
                        let diff = ref_mfcc[ti * d + j] - syn_mfcc[ti * d + j];
                        diff * diff
                    })
                    .sum();
                factor * (2.0 * sq).sqrt()
            })
            .sum::<f32>()
            / t as f32;
        Ok(mcd)
    }

    /// Estimate a proxy MOS score based on MCD (lower MCD → higher MOS, clamped to [1, 5]).
    ///
    /// Uses the empirical approximation MOS ≈ 5 - 0.4 * MCD.
    pub fn mos_proxy(mcd_db: f32) -> f32 {
        (5.0 - 0.4 * mcd_db).clamp(1.0, 5.0)
    }

    /// Compute a character-error-rate-based intelligibility proxy from two symbol sequences.
    ///
    /// Returns CER ∈ [0, 1] (0 = perfect, 1 = all wrong).
    pub fn cer(reference: &[usize], hypothesis: &[usize]) -> f32 {
        let r = reference.len();
        let h = hypothesis.len();
        if r == 0 {
            return if h == 0 { 0.0 } else { 1.0 };
        }
        // Levenshtein distance via DP
        let mut dp = vec![0usize; (r + 1) * (h + 1)];
        for i in 0..=r {
            dp[i * (h + 1)] = i;
        }
        for j in 0..=h {
            dp[j] = j;
        }
        for i in 1..=r {
            for j in 1..=h {
                let cost = if reference[i - 1] == hypothesis[j - 1] { 0 } else { 1 };
                dp[i * (h + 1) + j] = (dp[(i - 1) * (h + 1) + j] + 1)
                    .min(dp[i * (h + 1) + j - 1] + 1)
                    .min(dp[(i - 1) * (h + 1) + j - 1] + cost);
            }
        }
        dp[r * (h + 1) + h] as f32 / r as f32
    }

    /// Add an MCD measurement.
    pub fn add_mcd(&mut self, mcd: f32) {
        self.mcd_values.push(mcd);
    }

    /// Add a WER score.
    pub fn add_wer(&mut self, wer: f32) {
        self.wer_scores.push(wer);
    }

    /// Compute mean MCD over all accumulated samples.
    pub fn mean_mcd(&self) -> f32 {
        if self.mcd_values.is_empty() {
            return 0.0;
        }
        self.mcd_values.iter().sum::<f32>() / self.mcd_values.len() as f32
    }

    /// Compute mean WER over all accumulated samples.
    pub fn mean_wer(&self) -> f32 {
        if self.wer_scores.is_empty() {
            return 0.0;
        }
        self.wer_scores.iter().sum::<f32>() / self.wer_scores.len() as f32
    }
}
