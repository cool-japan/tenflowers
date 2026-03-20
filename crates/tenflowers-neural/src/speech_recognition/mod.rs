//! End-to-end speech recognition components.
//!
//! Implements:
//! - **WhisperEncoder** — transformer encoder for speech (log-mel spectrogram → contextual embeddings)
//! - **WhisperDecoder** — autoregressive transformer decoder with cross-attention
//! - **CtcBeamDecoder** — CTC beam-search and greedy decoding
//! - **RnntDecoder** — RNN-T joint network with greedy decoding
//! - **NgramLm / LmRescorer** — N-gram language model for shallow fusion
//! - **VoiceActivityDetector** — energy-based VAD with segmentation
//! - **SpeakerDiarizer** — spectral clustering for speaker change detection
//! - **SpeechAugmentation** — noise, time-stretch, pitch-shift, RIR convolution
//! - **WordErrorRate** — standard WER/CER Levenshtein metrics
//! - **SpeechPipeline** — end-to-end VAD → mel → encode → decode pipeline
//!
//! All routines obey the project policies: no `unwrap()`, no `ndarray`, no `rand`.
//! Random numbers use `scirs2_core::random::{rngs::StdRng, Rng, SeedableRng}`.

use std::collections::HashMap;
use std::f32::consts::PI;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn inval(op: &str, reason: impl Into<String>) -> TensorError {
    TensorError::InvalidArgument {
        operation: op.to_string(),
        reason: reason.into(),
        context: None,
    }
}

#[inline]
fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    x * 0.5 * (1.0 + inner.tanh())
}

#[inline]
fn softmax_vec(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    exp.iter().map(|&e| e * inv).collect()
}

#[inline]
fn log_softmax_vec(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = sum.ln();
    v.iter().map(|&x| x - max - log_sum).collect()
}

/// Dot product.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Matrix-vector multiply: A (rows×cols) · x (cols) → y (rows).
fn matvec(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += a[r * cols + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

/// Layer norm on a slice.
fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|&v| (v - mean) * inv).collect()
}

/// Add two vecs element-wise.
fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. WhisperEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Whisper-style encoder.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of mel filterbank channels (typically 80).
    pub n_mels: usize,
    /// FFT size for spectrogram computation.
    pub n_fft: usize,
    /// Hop length in samples for STFT.
    pub hop_length: usize,
    /// Maximum audio context length in frames.
    pub n_audio_ctx: usize,
    /// Transformer hidden state dimension.
    pub n_audio_state: usize,
    /// Number of attention heads.
    pub n_audio_heads: usize,
    /// Number of transformer encoder layers.
    pub n_audio_layers: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_heads: 6,
            n_audio_layers: 4,
        }
    }
}

/// Compute log-mel spectrogram from raw audio samples.
///
/// Steps:
/// 1. Frame with Hann window (n_fft, hop_length).
/// 2. Compute power spectrum.
/// 3. Apply mel filterbank (80 bins, 0–8000 Hz or 0–sample_rate/2).
/// 4. Log-scale: `log(max(mel, 1e-10))`.
///
/// Returns `Vec<Vec<f32>>` of shape `[n_frames][n_mels]`.
pub fn log_mel_spectrogram(
    audio: &[f32],
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    if audio.is_empty() || n_fft == 0 || hop_length == 0 || n_mels == 0 {
        return Vec::new();
    }
    // Build Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft as f32 - 1.0)).cos()))
        .collect();

    // Number of frequency bins: n_fft/2 + 1
    let n_bins = n_fft / 2 + 1;

    // Build mel filterbank
    let fmin = 0.0_f32;
    let fmax = sample_rate as f32 * 0.5;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let bin_freqs: Vec<f32> = (0..n_bins)
        .map(|k| k as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    // Filterbank matrix: [n_mels][n_bins]
    let mut filterbank = vec![vec![0.0_f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let f_low = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_high = mel_points[m + 2];
        for (k, &fk) in bin_freqs.iter().enumerate() {
            if fk >= f_low && fk <= f_center && (f_center - f_low) > 1e-10 {
                filterbank[m][k] = (fk - f_low) / (f_center - f_low);
            } else if fk > f_center && fk <= f_high && (f_high - f_center) > 1e-10 {
                filterbank[m][k] = (f_high - fk) / (f_high - f_center);
            }
        }
    }

    // Frame and compute power spectrum
    let mut frames: Vec<Vec<f32>> = Vec::new();
    let mut pos = 0usize;
    while pos + n_fft <= audio.len() {
        let frame: Vec<f32> = (0..n_fft).map(|i| audio[pos + i] * window[i]).collect();
        let power = naive_rfft_power(&frame);
        frames.push(power);
        pos += hop_length;
    }
    // Handle final partial frame with zero-padding
    if pos < audio.len() {
        let mut frame = vec![0.0_f32; n_fft];
        let rem = audio.len() - pos;
        for i in 0..rem {
            frame[i] = audio[pos + i] * window[i];
        }
        let power = naive_rfft_power(&frame);
        frames.push(power);
    }

    // Apply mel filterbank and log
    frames
        .iter()
        .map(|power| {
            (0..n_mels)
                .map(|m| {
                    let mel_val: f32 = filterbank[m]
                        .iter()
                        .zip(power.iter())
                        .map(|(h, p)| h * p)
                        .sum();
                    (mel_val.max(1e-10)).ln()
                })
                .collect()
        })
        .collect()
}

/// Hz → Mel (O'Shaughnessy formula).
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Mel → Hz.
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Naïve O(N²) real FFT power spectrum (one-sided, N/2+1 bins).
fn naive_rfft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_bins = n / 2 + 1;
    let n_f = n as f32;
    (0..n_bins)
        .map(|k| {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for (i, &x) in frame.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * i as f32 / n_f;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            re * re + im * im
        })
        .collect()
}

/// Two 1D convolutional layers with GELU activation, forming the audio stem.
#[derive(Debug, Clone)]
pub struct AudioConvStem {
    /// First conv weights: [out_channels][in_channels * kernel_size].
    pub conv1_weights: Vec<Vec<f32>>,
    /// Second conv weights: [out_channels][out_channels * kernel_size].
    pub conv2_weights: Vec<Vec<f32>>,
    pub conv1_bias: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub kernel_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl AudioConvStem {
    /// Create a randomly initialised stem.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f32 / (in_channels * kernel_size) as f32).sqrt();
        let conv1_weights = (0..out_channels)
            .map(|_| {
                (0..in_channels * kernel_size)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let scale2 = (2.0_f32 / (out_channels * kernel_size) as f32).sqrt();
        let conv2_weights = (0..out_channels)
            .map(|_| {
                (0..out_channels * kernel_size)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale2)
                    .collect()
            })
            .collect();
        Self {
            conv1_weights,
            conv2_weights,
            conv1_bias: vec![0.0; out_channels],
            conv2_bias: vec![0.0; out_channels],
            kernel_size,
            in_channels,
            out_channels,
        }
    }

    /// Forward: input shape \[T\]\[in_channels\] → output [T']\[out_channels\].
    ///
    /// Uses causal-padded 1D convolution (same length, pad_left = kernel_size-1, pad_right = 0).
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // After conv1: [T][out_channels]
        let after1 = self.conv1d(
            input,
            &self.conv1_weights,
            &self.conv1_bias,
            self.in_channels,
        );
        // GELU activation
        let after_gelu: Vec<Vec<f32>> = after1
            .iter()
            .map(|frame| frame.iter().map(|&x| gelu(x)).collect())
            .collect();
        // After conv2: [T][out_channels]
        let after2 = self.conv1d(
            &after_gelu,
            &self.conv2_weights,
            &self.conv2_bias,
            self.out_channels,
        );
        after2
            .into_iter()
            .map(|frame| frame.into_iter().map(gelu).collect())
            .collect()
    }

    fn conv1d(
        &self,
        input: &[Vec<f32>],
        weights: &[Vec<f32>],
        bias: &[f32],
        in_ch: usize,
    ) -> Vec<Vec<f32>> {
        let t = input.len();
        let ks = self.kernel_size;
        let out_ch = weights.len();
        let pad = ks - 1; // causal: pad left
        (0..t)
            .map(|ti| {
                (0..out_ch)
                    .map(|o| {
                        let mut val = bias[o];
                        for k in 0..ks {
                            let src = ti + k;
                            if src >= pad && src - pad < t {
                                let inp_idx = src - pad;
                                for c in 0..in_ch {
                                    val += weights[o][k * in_ch + c] * input[inp_idx][c];
                                }
                            }
                        }
                        val
                    })
                    .collect()
            })
            .collect()
    }
}

/// A single transformer encoder layer (self-attention + feed-forward).
#[derive(Debug, Clone)]
pub struct WhisperEncoderLayer {
    d_model: usize,
    n_heads: usize,
    /// Q, K, V projections concatenated: [3 * d_model * d_model]
    qkv_weight: Vec<f32>,
    out_weight: Vec<f32>,
    ff1_weight: Vec<f32>,
    ff2_weight: Vec<f32>,
}

impl WhisperEncoderLayer {
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / d_model as f32).sqrt();
        let total_qkv = 3 * d_model * d_model;
        let qkv_weight: Vec<f32> = (0..total_qkv)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let out_weight: Vec<f32> = (0..d_model * d_model)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let ff_dim = 4 * d_model;
        let ff1_weight: Vec<f32> = (0..ff_dim * d_model)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let ff2_weight: Vec<f32> = (0..d_model * ff_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            d_model,
            n_heads,
            qkv_weight,
            out_weight,
            ff1_weight,
            ff2_weight,
        }
    }

    /// Forward: input `[T][d_model]` → `[T][d_model]`.
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = input.len();
        if t == 0 {
            return Vec::new();
        }
        let d = self.d_model;
        let h = self.n_heads;
        let head_dim = d / h;
        let scale = (head_dim as f32).sqrt().recip();

        // Multi-head self-attention
        // Project Q, K, V
        let mut q_mat = vec![vec![0.0f32; d]; t];
        let mut k_mat = vec![vec![0.0f32; d]; t];
        let mut v_mat = vec![vec![0.0f32; d]; t];
        for ti in 0..t {
            let x = &input[ti];
            // Q weight: rows 0..d, K: d..2d, V: 2d..3d  (each d×d)
            for r in 0..d {
                let mut qval = 0.0f32;
                let mut kval = 0.0f32;
                let mut vval = 0.0f32;
                for c in 0..d {
                    qval += self.qkv_weight[r * d + c] * x[c];
                    kval += self.qkv_weight[(d + r) * d + c] * x[c];
                    vval += self.qkv_weight[(2 * d + r) * d + c] * x[c];
                }
                q_mat[ti][r] = qval;
                k_mat[ti][r] = kval;
                v_mat[ti][r] = vval;
            }
        }

        // Compute multi-head attention output
        let mut attn_out = vec![vec![0.0f32; d]; t];
        for hi in 0..h {
            let start = hi * head_dim;
            let end = start + head_dim;
            // Compute attention scores and weighted values for this head
            for qi in 0..t {
                let mut scores = vec![0.0f32; t];
                for ki in 0..t {
                    let s: f32 = (start..end)
                        .map(|r| q_mat[qi][r] * k_mat[ki][r])
                        .sum::<f32>()
                        * scale;
                    scores[ki] = s;
                }
                let attn_weights = softmax_vec(&scores);
                for r in start..end {
                    let val: f32 = (0..t).map(|ki| attn_weights[ki] * v_mat[ki][r]).sum();
                    attn_out[qi][r] += val;
                }
            }
        }

        // Output projection + residual + LN
        let attn_proj: Vec<Vec<f32>> = (0..t)
            .map(|ti| {
                let res = matvec(&self.out_weight, &attn_out[ti], d, d);
                let added = add_vecs(&res, &input[ti]);
                layer_norm(&added, 1e-5)
            })
            .collect();

        // Feed-forward
        let ff_dim = 4 * d;
        (0..t)
            .map(|ti| {
                let ff1 = matvec(&self.ff1_weight, &attn_proj[ti], ff_dim, d);
                let ff1_act: Vec<f32> = ff1.iter().map(|&x| gelu(x)).collect();
                let ff2 = matvec(&self.ff2_weight, &ff1_act, d, ff_dim);
                let added = add_vecs(&ff2, &attn_proj[ti]);
                layer_norm(&added, 1e-5)
            })
            .collect()
    }
}

/// Whisper-style transformer encoder for speech.
#[derive(Debug, Clone)]
pub struct WhisperEncoder {
    pub config: WhisperConfig,
    pub stem: AudioConvStem,
    pub transformer_layers: Vec<WhisperEncoderLayer>,
    pos_embed: Vec<Vec<f32>>,
}

impl WhisperEncoder {
    /// Create a new WhisperEncoder with random weights.
    pub fn new(config: WhisperConfig, seed: u64) -> Self {
        let stem = AudioConvStem::new(config.n_mels, config.n_audio_state, 3, seed);
        let layers = (0..config.n_audio_layers)
            .map(|i| {
                WhisperEncoderLayer::new(
                    config.n_audio_state,
                    config.n_audio_heads,
                    seed + i as u64 + 1,
                )
            })
            .collect();
        // Sinusoidal positional embeddings
        let d = config.n_audio_state;
        let pos_embed = (0..config.n_audio_ctx)
            .map(|pos| {
                (0..d)
                    .map(|i| {
                        let denom = 10000.0_f32.powf(2.0 * (i / 2) as f32 / d as f32);
                        if i % 2 == 0 {
                            (pos as f32 / denom).sin()
                        } else {
                            (pos as f32 / denom).cos()
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            config,
            stem,
            transformer_layers: layers,
            pos_embed,
        }
    }

    /// Encode log-mel spectrogram frames.
    ///
    /// Input: `[T_frames][n_mels]`
    /// Output: `[T_enc][n_audio_state]`
    pub fn encode(&self, log_mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Audio conv stem: treats each frame as a time-step with n_mels channels
        let stem_out = self.stem.forward(log_mel);

        // Add positional embeddings (up to n_audio_ctx frames)
        let n_ctx = self.config.n_audio_ctx;
        let d = self.config.n_audio_state;
        let t = stem_out.len().min(n_ctx);
        let mut x: Vec<Vec<f32>> = (0..t)
            .map(|ti| {
                if stem_out[ti].len() == d {
                    add_vecs(&stem_out[ti], &self.pos_embed[ti])
                } else {
                    // Padding if mismatch
                    let mut v = stem_out[ti].clone();
                    v.resize(d, 0.0);
                    add_vecs(&v, &self.pos_embed[ti])
                }
            })
            .collect();

        // Transformer layers
        for layer in &self.transformer_layers {
            x = layer.forward(&x);
        }
        x
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. WhisperDecoder
// ─────────────────────────────────────────────────────────────────────────────

/// A single decoder layer with masked self-attention + cross-attention + FF.
#[derive(Debug, Clone)]
pub struct WhisperDecoderLayer {
    d_model: usize,
    n_heads: usize,
    self_qkv_weight: Vec<f32>,
    self_out_weight: Vec<f32>,
    cross_q_weight: Vec<f32>,
    cross_kv_weight: Vec<f32>,
    cross_out_weight: Vec<f32>,
    ff1_weight: Vec<f32>,
    ff2_weight: Vec<f32>,
}

impl WhisperDecoderLayer {
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / d_model as f32).sqrt();
        let qkv = 3 * d_model * d_model;
        let ff_dim = 4 * d_model;

        macro_rules! rand_vec {
            ($n:expr) => {
                (0..$n)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect::<Vec<f32>>()
            };
        }

        Self {
            d_model,
            n_heads,
            self_qkv_weight: rand_vec!(qkv),
            self_out_weight: rand_vec!(d_model * d_model),
            cross_q_weight: rand_vec!(d_model * d_model),
            cross_kv_weight: rand_vec!(2 * d_model * d_model),
            cross_out_weight: rand_vec!(d_model * d_model),
            ff1_weight: rand_vec!(ff_dim * d_model),
            ff2_weight: rand_vec!(d_model * ff_dim),
        }
    }

    /// Forward with causal mask on self-attention.
    ///
    /// `tgt`: decoder input `[T_dec][d_model]`
    /// `enc`: encoder output `[T_enc][d_model]`
    pub fn forward(&self, tgt: &[Vec<f32>], enc: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = tgt.len();
        if t == 0 {
            return Vec::new();
        }
        let d = self.d_model;
        let h = self.n_heads;
        let head_dim = d / h;
        let scale = (head_dim as f32).sqrt().recip();

        // Masked self-attention
        let mut q = vec![vec![0.0f32; d]; t];
        let mut k = vec![vec![0.0f32; d]; t];
        let mut v = vec![vec![0.0f32; d]; t];
        for ti in 0..t {
            let x = &tgt[ti];
            for r in 0..d {
                let mut qv = 0.0f32;
                let mut kv = 0.0f32;
                let mut vv = 0.0f32;
                for c in 0..d {
                    qv += self.self_qkv_weight[r * d + c] * x[c];
                    kv += self.self_qkv_weight[(d + r) * d + c] * x[c];
                    vv += self.self_qkv_weight[(2 * d + r) * d + c] * x[c];
                }
                q[ti][r] = qv;
                k[ti][r] = kv;
                v[ti][r] = vv;
            }
        }

        let mut self_attn_out = vec![vec![0.0f32; d]; t];
        for hi in 0..h {
            let start = hi * head_dim;
            let end = start + head_dim;
            for qi in 0..t {
                let mut scores = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {
                    // causal mask
                    let s: f32 = (start..end).map(|r| q[qi][r] * k[ki][r]).sum::<f32>() * scale;
                    scores[ki] = s;
                }
                let attn_weights = softmax_vec(&scores);
                for r in start..end {
                    let val: f32 = (0..=qi).map(|ki| attn_weights[ki] * v[ki][r]).sum();
                    self_attn_out[qi][r] += val;
                }
            }
        }

        // Self-attn output projection + residual + LN
        let x1: Vec<Vec<f32>> = (0..t)
            .map(|ti| {
                let proj = matvec(&self.self_out_weight, &self_attn_out[ti], d, d);
                let added = add_vecs(&proj, &tgt[ti]);
                layer_norm(&added, 1e-5)
            })
            .collect();

        // Cross-attention: Q from decoder, K/V from encoder
        let t_enc = enc.len();
        if t_enc == 0 {
            return x1; // No encoder output, skip cross-attn
        }

        let mut cq = vec![vec![0.0f32; d]; t];
        let mut ck = vec![vec![0.0f32; d]; t_enc];
        let mut cv = vec![vec![0.0f32; d]; t_enc];
        for ti in 0..t {
            let x = &x1[ti];
            for r in 0..d {
                let mut qv = 0.0f32;
                for c in 0..d {
                    qv += self.cross_q_weight[r * d + c] * x[c];
                }
                cq[ti][r] = qv;
            }
        }
        for ti in 0..t_enc {
            let e = &enc[ti];
            for r in 0..d {
                let mut kv = 0.0f32;
                let mut vv = 0.0f32;
                for c in 0..d {
                    kv += self.cross_kv_weight[r * d + c] * e[c];
                    vv += self.cross_kv_weight[(d + r) * d + c] * e[c];
                }
                ck[ti][r] = kv;
                cv[ti][r] = vv;
            }
        }

        let mut cross_attn_out = vec![vec![0.0f32; d]; t];
        for hi in 0..h {
            let start = hi * head_dim;
            let end = start + head_dim;
            for qi in 0..t {
                let mut scores = vec![0.0f32; t_enc];
                for ki in 0..t_enc {
                    scores[ki] = (start..end).map(|r| cq[qi][r] * ck[ki][r]).sum::<f32>() * scale;
                }
                let attn_weights = softmax_vec(&scores);
                for r in start..end {
                    let val: f32 = (0..t_enc).map(|ki| attn_weights[ki] * cv[ki][r]).sum();
                    cross_attn_out[qi][r] += val;
                }
            }
        }

        // Cross-attn projection + residual + LN
        let x2: Vec<Vec<f32>> = (0..t)
            .map(|ti| {
                let proj = matvec(&self.cross_out_weight, &cross_attn_out[ti], d, d);
                let added = add_vecs(&proj, &x1[ti]);
                layer_norm(&added, 1e-5)
            })
            .collect();

        // Feed-forward
        let ff_dim = 4 * d;
        (0..t)
            .map(|ti| {
                let ff1 = matvec(&self.ff1_weight, &x2[ti], ff_dim, d);
                let ff1_act: Vec<f32> = ff1.iter().map(|&x| gelu(x)).collect();
                let ff2 = matvec(&self.ff2_weight, &ff1_act, d, ff_dim);
                let added = add_vecs(&ff2, &x2[ti]);
                layer_norm(&added, 1e-5)
            })
            .collect()
    }
}

/// Whisper-style autoregressive transformer decoder.
#[derive(Debug, Clone)]
pub struct WhisperDecoder {
    d_model: usize,
    vocab_size: usize,
    pub token_embed: Vec<Vec<f32>>,
    pub pos_embed: Vec<Vec<f32>>,
    pub cross_attn_layers: Vec<WhisperDecoderLayer>,
    lm_head: Vec<f32>,
}

impl WhisperDecoder {
    /// Create a new WhisperDecoder.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_len: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / d_model as f32).sqrt();
        let token_embed = (0..vocab_size)
            .map(|_| {
                (0..d_model)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let pos_embed = (0..max_len)
            .map(|pos| {
                (0..d_model)
                    .map(|i| {
                        let denom = 10000.0_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                        if i % 2 == 0 {
                            (pos as f32 / denom).sin()
                        } else {
                            (pos as f32 / denom).cos()
                        }
                    })
                    .collect()
            })
            .collect();
        let layers = (0..n_layers)
            .map(|i| WhisperDecoderLayer::new(d_model, n_heads, seed + i as u64 + 100))
            .collect();
        let lm_head: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            d_model,
            vocab_size,
            token_embed,
            pos_embed,
            cross_attn_layers: layers,
            lm_head,
        }
    }

    /// One decoding step.
    ///
    /// `tokens`: token indices seen so far (prefix)
    /// `encoder_output`: `[T_enc][d_model]`
    /// Returns logits `[vocab_size]` for the next token.
    pub fn decode_step(&self, tokens: &[usize], encoder_output: &[Vec<f32>]) -> Vec<f32> {
        if tokens.is_empty() {
            return vec![0.0; self.vocab_size];
        }
        let d = self.d_model;
        // Embed tokens + positional
        let mut x: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok)| {
                let tok_idx = tok.min(self.vocab_size.saturating_sub(1));
                let pos_idx = pos.min(self.pos_embed.len().saturating_sub(1));
                add_vecs(&self.token_embed[tok_idx], &self.pos_embed[pos_idx])
            })
            .collect();

        // Decoder layers
        for layer in &self.cross_attn_layers {
            x = layer.forward(&x, encoder_output);
        }

        // LM head: last token position → logits
        let last = &x[x.len() - 1];
        matvec(&self.lm_head, last, self.vocab_size, d)
    }

    /// Greedy decoding up to `max_tokens` or until `eos_id`.
    pub fn greedy_decode(
        &self,
        encoder_output: &[Vec<f32>],
        bos_id: usize,
        eos_id: usize,
        max_tokens: usize,
    ) -> Vec<usize> {
        let mut tokens = vec![bos_id];
        for _ in 0..max_tokens {
            let logits = self.decode_step(&tokens, encoder_output);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if next == eos_id {
                break;
            }
            tokens.push(next);
        }
        tokens
    }
}

pub mod extensions;
pub use extensions::*;

#[cfg(test)]
mod tests;
