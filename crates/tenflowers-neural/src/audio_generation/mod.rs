//! # Neural Audio Generation (audio_generation)
//!
//! Production-grade neural audio generation models in pure Rust.
//! All computations use `Vec<f64>` with manual PRNG — no ndarray, no rand crate.
//!
//! ## Modules / Sections
//!
//! - §1 [`AgDilatedCausalConv1D`] — WaveNet causal dilated convolution
//! - §2 [`AgGatedActivation`] — WaveNet gated tanh-sigmoid unit
//! - §3 [`AgWaveNet`] — Oord et al. 2016 autoregressive waveform model
//! - §4 [`AgNoiseSchedule`] — Diffusion noise schedules (linear / cosine)
//! - §5 [`AgDiffWave`] — Kong et al. 2021 diffusion waveform model
//! - §6 [`AgGriffinLim`] — Griffin-Lim phase reconstruction vocoder
//! - §7 [`AgHifiGanGenerator`] — HiFi-GAN MRF upsampling vocoder
//! - §8 [`AgTextToSpeechPipeline`] — Simple TTS pipeline
//! - §9 [`AgAudioCodec`] — SoundStream-inspired neural audio codec
//! - §10 [`AgMetrics`] — SNR, SI-SDR, spectral convergence, log spectral distance, MOS
//!
//! ## References
//!
//! - Van den Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio"
//! - Kong et al. (2021) "DiffWave: A Versatile Diffusion Model for Audio Synthesis"
//! - Griffin & Lim (1984) "Signal estimation from modified short-time Fourier transform"
//! - Kong et al. (2020) "HiFi-GAN: Generative Adversarial Networks for Efficient and
//!   High Fidelity Speech Synthesis"
//! - Zeghidour et al. (2021) "SoundStream: An End-to-End Neural Audio Codec"

#[cfg(test)]
mod tests;

use std::f64::consts::PI;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error Type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors emitted by the `audio_generation` module.
#[derive(Debug, Clone, PartialEq)]
pub enum AgError {
    /// Invalid configuration parameter (e.g. zero channels).
    InvalidConfig(String),
    /// Dimension mismatch between tensors.
    DimensionError(String),
    /// Numerical issue (NaN, Inf, singular matrix).
    NumericalError(String),
}

impl fmt::Display for AgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgError::InvalidConfig(m) => write!(f, "AgError::InvalidConfig: {m}"),
            AgError::DimensionError(m) => write!(f, "AgError::DimensionError: {m}"),
            AgError::NumericalError(m) => write!(f, "AgError::NumericalError: {m}"),
        }
    }
}

impl std::error::Error for AgError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal PRNG utilities
// ─────────────────────────────────────────────────────────────────────────────

/// xorshift64 step — advances the seed and returns a uniform u64.
#[inline]
fn ag_xorshift(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

/// Box-Muller transform: returns one standard-normal sample, advances `seed` twice.
pub fn ag_randn(seed: &mut u64) -> f64 {
    loop {
        let u1 = (ag_xorshift(seed) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        let u2 = (ag_xorshift(seed) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        if u1 > 0.0 && u2 > 0.0 {
            return (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        }
    }
}

/// Uniform sample in [0, 1).
#[inline]
fn ag_uniform(seed: &mut u64) -> f64 {
    ag_xorshift(seed) as f64 / (u64::MAX as f64 + 1.0)
}

/// Glorot / Xavier uniform initializer for a weight matrix of shape [out × in].
fn glorot_uniform(out: usize, in_: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    let limit = (6.0 / (in_ + out) as f64).sqrt();
    (0..out)
        .map(|_| {
            (0..in_)
                .map(|_| {
                    let u = ag_uniform(seed);
                    u * 2.0 * limit - limit
                })
                .collect()
        })
        .collect()
}

/// 1-D dot product.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Matrix–vector product: M [out × in_] @ v [in_] → [out].
fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

/// Element-wise ReLU.
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Softmax over a slice; returns a new Vec<f64>.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let inv = if sum > 1e-40 { 1.0 / sum } else { 1.0 };
    exps.iter().map(|&e| e * inv).collect()
}

/// Sample a discrete index proportional to `probs`.
fn sample_categorical(probs: &[f64], seed: &mut u64) -> usize {
    let u = ag_uniform(seed);
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if u < cdf {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  AgDilatedCausalConv1D
// ─────────────────────────────────────────────────────────────────────────────

/// WaveNet causal dilated 1-D convolution.
///
/// Output at time `t` depends only on inputs at times `t, t-d, t-2d, …`.
/// Implemented by left-padding with `(kernel_size - 1) * dilation` zeros, then
/// sliding a strided window of step `dilation`.
#[derive(Clone, Debug)]
pub struct AgDilatedCausalConv1D {
    /// Weight matrix: `[out_channels × (in_channels * kernel_size)]`.
    pub kernel: Vec<Vec<f64>>,
    /// Bias vector: `[out_channels]`.
    pub bias: Vec<f64>,
    /// Dilation factor.
    pub dilation: usize,
    /// Kernel size.
    pub kernel_size: usize,
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count.
    pub out_channels: usize,
}

impl AgDilatedCausalConv1D {
    /// Construct with Glorot-uniform initialised weights.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        dilation: usize,
        seed: &mut u64,
    ) -> Self {
        let fan_in = in_ch * kernel_size;
        Self {
            kernel: glorot_uniform(out_ch, fan_in, seed),
            bias: vec![0.0; out_ch],
            dilation,
            kernel_size,
            in_channels: in_ch,
            out_channels: out_ch,
        }
    }

    /// Receptive field of this single conv layer: `(kernel_size - 1) * dilation + 1`.
    pub fn receptive_field(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    /// Forward pass.
    ///
    /// `x`: `[in_channels × length]`
    /// Returns `[out_channels × length]` (causal — output length equals input length).
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let length = if x.is_empty() { 0 } else { x[0].len() };
        let pad = (self.kernel_size - 1) * self.dilation;

        // Build padded input: [in_channels × (length + pad)]
        let padded_len = length + pad;
        let mut padded: Vec<Vec<f64>> = (0..self.in_channels)
            .map(|ch| {
                let mut v = vec![0.0_f64; padded_len];
                for t in 0..length {
                    v[pad + t] = if ch < x.len() { x[ch][t] } else { 0.0 };
                }
                v
            })
            .collect();

        // Allocate output
        let mut out: Vec<Vec<f64>> = vec![vec![0.0; length]; self.out_channels];

        for t in 0..length {
            // Build the flattened input window: [in_channels * kernel_size]
            let mut window = Vec::with_capacity(self.in_channels * self.kernel_size);
            for k in 0..self.kernel_size {
                let src_t = t + pad - k * self.dilation; // causal: look back
                for ch in 0..self.in_channels {
                    window.push(padded[ch][src_t]);
                }
            }
            // Apply each output filter
            for oc in 0..self.out_channels {
                out[oc][t] = dot(&self.kernel[oc], &window) + self.bias[oc];
            }
        }

        // Drop the padded variable to avoid unused-mut warning
        drop(padded);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  AgGatedActivation
// ─────────────────────────────────────────────────────────────────────────────

/// WaveNet gated activation unit.
///
/// Implements:
/// ```text
/// gated  = tanh(filter_conv(x)) * sigmoid(gate_conv(x))
/// residual_out = x + 1×1_conv(gated)
/// skip_out     = skip_1×1_conv(gated)
/// ```
/// Optionally adds a conditioning signal to the pre-activation.
#[derive(Clone, Debug)]
pub struct AgGatedActivation {
    /// Filter branch (tanh path).
    pub filter_conv: AgDilatedCausalConv1D,
    /// Gate branch (sigmoid path).
    pub gate_conv: AgDilatedCausalConv1D,
    /// 1×1 residual projection: `[channels × channels]`.
    pub residual_w: Vec<Vec<f64>>,
    /// 1×1 skip projection: `[skip_channels × channels]`.
    pub skip_w: Vec<Vec<f64>>,
    /// Residual channel count.
    pub channels: usize,
    /// Skip channel count.
    pub skip_channels: usize,
}

impl AgGatedActivation {
    /// Construct a new gated activation block.
    pub fn new(
        channels: usize,
        skip_ch: usize,
        kernel_size: usize,
        dilation: usize,
        seed: &mut u64,
    ) -> Self {
        Self {
            filter_conv: AgDilatedCausalConv1D::new(
                channels,
                channels,
                kernel_size,
                dilation,
                seed,
            ),
            gate_conv: AgDilatedCausalConv1D::new(channels, channels, kernel_size, dilation, seed),
            residual_w: glorot_uniform(channels, channels, seed),
            skip_w: glorot_uniform(skip_ch, channels, seed),
            channels,
            skip_channels: skip_ch,
        }
    }

    /// Forward.
    ///
    /// - `x`: `[channels × length]`
    /// - `cond`: optional conditioning `[channels × length]`
    ///
    /// Returns `(residual_output, skip_output)` both `[channels × length]`.
    pub fn forward(
        &self,
        x: &[Vec<f64>],
        cond: Option<&Vec<Vec<f64>>>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let length = if x.is_empty() { 0 } else { x[0].len() };

        let f_out = self.filter_conv.forward(x);
        let g_out = self.gate_conv.forward(x);

        // Gated activation: tanh(f) * sigmoid(g)  [+cond if provided]
        let mut gated: Vec<Vec<f64>> = vec![vec![0.0; length]; self.channels];
        for ch in 0..self.channels {
            for t in 0..length {
                let fv =
                    f_out[ch][t] + cond.map_or(0.0, |c| if ch < c.len() { c[ch][t] } else { 0.0 });
                let gv = g_out[ch][t];
                gated[ch][t] = fv.tanh() * (1.0 / (1.0 + (-gv).exp()));
            }
        }

        // Residual: x + 1×1(gated)
        let mut residual: Vec<Vec<f64>> = vec![vec![0.0; length]; self.channels];
        for t in 0..length {
            let gated_t: Vec<f64> = gated.iter().map(|ch| ch[t]).collect();
            let proj = mat_vec(&self.residual_w, &gated_t);
            for ch in 0..self.channels {
                residual[ch][t] = x[ch][t] + proj[ch];
            }
        }

        // Skip: 1×1(gated)
        let mut skip: Vec<Vec<f64>> = vec![vec![0.0; length]; self.skip_channels];
        for t in 0..length {
            let gated_t: Vec<f64> = gated.iter().map(|ch| ch[t]).collect();
            let proj = mat_vec(&self.skip_w, &gated_t);
            for ch in 0..self.skip_channels {
                skip[ch][t] = proj[ch];
            }
        }

        (residual, skip)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  AgWaveNet
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`AgWaveNet`].
#[derive(Clone, Debug)]
pub struct AgWaveNetConfig {
    /// Number of layers per dilation stack (e.g., 10).
    pub n_layers: usize,
    /// Number of stacks (each stack repeats dilations 1, 2, 4, …, 2^(n_layers-1)).
    pub n_stacks: usize,
    /// Residual channel width.
    pub channels: usize,
    /// Skip-connection channel width.
    pub skip_channels: usize,
    /// Conv kernel size (typically 2).
    pub kernel_size: usize,
    /// Mu-law quantization levels (e.g., 256).
    pub n_quantization: usize,
}

impl Default for AgWaveNetConfig {
    fn default() -> Self {
        Self {
            n_layers: 10,
            n_stacks: 3,
            channels: 64,
            skip_channels: 64,
            kernel_size: 2,
            n_quantization: 256,
        }
    }
}

/// WaveNet autoregressive waveform model (Oord et al. 2016).
///
/// Generates audio one mu-law-quantized sample at a time from a categorical
/// distribution over 256 levels.
#[derive(Clone, Debug)]
pub struct AgWaveNet {
    /// Configuration.
    pub config: AgWaveNetConfig,
    /// Embedding table: `[channels × n_quantization]`.
    pub input_embed: Vec<Vec<f64>>,
    /// Stack of gated activation layers.
    pub layers: Vec<AgGatedActivation>,
    /// First output 1×1 conv: `[skip_channels × skip_channels]`.
    pub out_w1: Vec<Vec<f64>>,
    /// Second output 1×1 conv: `[n_quant × skip_channels]`.
    pub out_w2: Vec<Vec<f64>>,
    /// Bias for `out_w1`.
    pub out_b1: Vec<f64>,
    /// Bias for `out_w2`.
    pub out_b2: Vec<f64>,
}

impl AgWaveNet {
    /// Build a new WaveNet with random weights.
    pub fn new(config: AgWaveNetConfig, seed: &mut u64) -> Self {
        let limit = (6.0 / (config.channels + config.n_quantization) as f64).sqrt();
        let input_embed: Vec<Vec<f64>> = (0..config.channels)
            .map(|_| {
                (0..config.n_quantization)
                    .map(|_| {
                        let u = ag_uniform(seed);
                        u * 2.0 * limit - limit
                    })
                    .collect()
            })
            .collect();

        let mut layers = Vec::new();
        for _stack in 0..config.n_stacks {
            for layer_idx in 0..config.n_layers {
                let dilation = 1usize << layer_idx;
                layers.push(AgGatedActivation::new(
                    config.channels,
                    config.skip_channels,
                    config.kernel_size,
                    dilation,
                    seed,
                ));
            }
        }

        let out_w1 = glorot_uniform(config.skip_channels, config.skip_channels, seed);
        let out_w2 = glorot_uniform(config.n_quantization, config.skip_channels, seed);
        let out_b1 = vec![0.0; config.skip_channels];
        let out_b2 = vec![0.0; config.n_quantization];

        Self {
            config,
            input_embed,
            layers,
            out_w1,
            out_w2,
            out_b1,
            out_b2,
        }
    }

    /// Total receptive field: sum over all layers.
    pub fn total_receptive_field(&self) -> usize {
        let mut total = 0usize;
        for _stack in 0..self.config.n_stacks {
            for layer_idx in 0..self.config.n_layers {
                let dilation = 1usize << layer_idx;
                let rf = (self.config.kernel_size - 1) * dilation + 1;
                total += rf;
            }
        }
        total
    }

    /// Single-step forward pass.
    ///
    /// `x_quantized`: integer in `[0, n_quantization)` representing the current sample.
    /// `context`: `[channels × context_len]` — previous embedded samples (may be empty).
    /// Returns a softmax probability vector of length `n_quantization`.
    pub fn forward_step(&self, x_quantized: usize, context: &[Vec<f64>]) -> Vec<f64> {
        let nq = self.config.n_quantization;
        let ch = self.config.channels;
        let idx = x_quantized.min(nq - 1);

        // Embed: look up column `idx` of input_embed → [channels × 1]
        let embedded: Vec<Vec<f64>> = (0..ch).map(|c| vec![self.input_embed[c][idx]]).collect();

        // Run all layers, accumulate skip connections
        let mut skip_sum: Vec<f64> = vec![0.0; self.config.skip_channels];
        let mut residual = embedded;
        for layer in &self.layers {
            let (new_res, skip) = layer.forward(&residual, None);
            residual = new_res;
            // skip is [skip_channels × 1]; accumulate
            for sc in 0..self.config.skip_channels {
                skip_sum[sc] += skip[sc][0];
            }
        }

        // Output network: relu → W1 → relu → W2 → softmax
        let after_relu1: Vec<f64> = skip_sum.iter().map(|&v| relu(v)).collect();
        let after_w1: Vec<f64> = mat_vec(&self.out_w1, &after_relu1)
            .iter()
            .zip(&self.out_b1)
            .map(|(&w, &b)| relu(w + b))
            .collect();
        let logits: Vec<f64> = mat_vec(&self.out_w2, &after_w1)
            .iter()
            .zip(&self.out_b2)
            .map(|(&w, &b)| w + b)
            .collect();

        softmax(&logits)
    }

    /// Autoregressive generation of `n_samples` waveform samples.
    ///
    /// Returns raw audio in `[-1, 1]` decoded from mu-law.
    pub fn generate(&self, n_samples: usize, seed: &mut u64) -> Vec<f64> {
        let mu = self.config.n_quantization - 1;
        let mut audio = Vec::with_capacity(n_samples);
        let mut current = mu / 2; // start from silence (mid-level)

        for _ in 0..n_samples {
            let probs = self.forward_step(current, &[]);
            current = sample_categorical(&probs, seed);
            audio.push(Self::mu_law_decode(current, mu));
        }
        audio
    }

    /// Mu-law encode: `x ∈ [-1, 1]` → `{0, …, mu}`.
    pub fn mu_law_encode(x: f64, mu: usize) -> usize {
        let mu_f = mu as f64;
        let x_clamp = x.clamp(-1.0, 1.0);
        let compressed = x_clamp.signum() * (1.0 + mu_f * x_clamp.abs()).ln() / (1.0 + mu_f).ln();
        // Map [-1, 1] → [0, mu]
        let q = ((compressed + 1.0) * 0.5 * mu_f + 0.5) as usize;
        q.min(mu)
    }

    /// Mu-law decode: `q ∈ {0, …, mu}` → `f64 ∈ [-1, 1]`.
    pub fn mu_law_decode(q: usize, mu: usize) -> f64 {
        let mu_f = mu as f64;
        let x = q as f64 / mu_f * 2.0 - 1.0; // back to [-1, 1]
        let sign = x.signum();
        sign * ((1.0 + mu_f).powf(sign * x) - 1.0) / mu_f
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  AgNoiseSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// Diffusion noise schedule storing β, α, ᾱ for each timestep.
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct AgNoiseSchedule {
    /// β_t schedule.
    pub betas: Vec<f64>,
    /// α_t = 1 − β_t.
    pub alphas: Vec<f64>,
    /// ᾱ_t = ∏_{s≤t} α_s (cumulative product).
    pub alpha_bar: Vec<f64>,
    /// Total diffusion steps.
    pub T: usize,
}

impl AgNoiseSchedule {
    /// Linear schedule: β_t linearly spaced from `beta_start` to `beta_end`.
    pub fn linear(t_steps: usize, beta_start: f64, beta_end: f64) -> Self {
        let betas: Vec<f64> = (0..t_steps)
            .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (t_steps - 1).max(1) as f64)
            .collect();
        Self::from_betas(betas, t_steps)
    }

    /// Cosine schedule (Nichol & Dhariwal 2021), `s = 0.008`.
    pub fn cosine(t_steps: usize) -> Self {
        let s = 0.008;
        let f0 = ((s / (1.0 + s)) * PI * 0.5).cos().powi(2);
        let alpha_bar: Vec<f64> = (0..=t_steps)
            .map(|t| {
                let ft = (((t as f64 / t_steps as f64) + s) / (1.0 + s) * PI * 0.5)
                    .cos()
                    .powi(2);
                (ft / f0).clamp(0.0, 1.0)
            })
            .collect();
        // Derive betas from alpha_bar differences: β_t = 1 - ᾱ_t / ᾱ_{t-1}
        let mut betas = Vec::with_capacity(t_steps);
        for t in 0..t_steps {
            let ab_prev = alpha_bar[t];
            let ab_curr = alpha_bar[t + 1];
            let beta = (1.0 - ab_curr / ab_prev.max(1e-12)).clamp(0.0, 0.999);
            betas.push(beta);
        }
        Self::from_betas(betas, t_steps)
    }

    fn from_betas(betas: Vec<f64>, t_steps: usize) -> Self {
        let alphas: Vec<f64> = betas.iter().map(|&b| 1.0 - b).collect();
        let mut alpha_bar = Vec::with_capacity(t_steps);
        let mut cum = 1.0;
        for &a in &alphas {
            cum *= a;
            alpha_bar.push(cum);
        }
        Self {
            betas,
            alphas,
            alpha_bar,
            T: t_steps,
        }
    }

    /// Add noise to `x0` at diffusion step `t`.
    ///
    /// Returns `(x_t, noise)` where `x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε`.
    pub fn add_noise(&self, x0: &[f64], t: usize, seed: &mut u64) -> (Vec<f64>, Vec<f64>) {
        let t_idx = t.min(self.T - 1);
        let ab = self.alpha_bar[t_idx];
        let sqrt_ab = ab.sqrt();
        let sqrt_one_minus_ab = (1.0 - ab).max(0.0).sqrt();

        let noise: Vec<f64> = (0..x0.len()).map(|_| ag_randn(seed)).collect();
        let x_t: Vec<f64> = x0
            .iter()
            .zip(&noise)
            .map(|(&x, &e)| sqrt_ab * x + sqrt_one_minus_ab * e)
            .collect();
        (x_t, noise)
    }

    /// DDPM reverse step: given `x_t` and predicted noise `eps_theta`, compute `x_{t-1}`.
    ///
    /// Uses the DDPM formula with σ_t = √β_t (variance-preserving).
    pub fn denoise_step(&self, x_t: &[f64], predicted_noise: &[f64], t: usize) -> Vec<f64> {
        if t == 0 {
            // At t=0 just return the clean signal estimate
            let t_idx = 0;
            let ab = self.alpha_bar[t_idx];
            let sqrt_ab = ab.sqrt().max(1e-12);
            let sqrt_one_minus_ab = (1.0 - ab).max(0.0).sqrt();
            return x_t
                .iter()
                .zip(predicted_noise)
                .map(|(&xt, &eps)| (xt - sqrt_one_minus_ab * eps) / sqrt_ab)
                .collect();
        }
        let t_idx = t.min(self.T - 1);
        let beta_t = self.betas[t_idx];
        let alpha_t = self.alphas[t_idx];
        let ab_t = self.alpha_bar[t_idx];
        let ab_prev = if t_idx > 0 {
            self.alpha_bar[t_idx - 1]
        } else {
            1.0
        };

        let sqrt_alpha_t = alpha_t.sqrt().max(1e-12);
        let coeff = beta_t / (1.0 - ab_t).max(1e-12).sqrt();

        // σ_t = sqrt( β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t) )
        let sigma_t = (beta_t * (1.0 - ab_prev) / (1.0 - ab_t).max(1e-12))
            .max(0.0)
            .sqrt();

        x_t.iter()
            .zip(predicted_noise)
            .map(|(&xt, &eps)| {
                let mean = (xt - coeff * eps) / sqrt_alpha_t;
                // For simplicity, use zero noise at inference (DDIM-like deterministic)
                mean + sigma_t * 0.0 // deterministic path
            })
            .collect()
    }

    /// Signal-to-noise ratio at step `t`: ᾱ_t / (1 − ᾱ_t).
    pub fn snr(&self, t: usize) -> f64 {
        let t_idx = t.min(self.T - 1);
        let ab = self.alpha_bar[t_idx];
        ab / (1.0 - ab).max(1e-40)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  AgDiffWave
// ─────────────────────────────────────────────────────────────────────────────

/// A single residual layer in the DiffWave score network.
#[derive(Clone, Debug)]
pub struct AgDiffWaveLayer {
    /// Residual dilated conv (filter path).
    pub residual_conv: AgDilatedCausalConv1D,
    /// Gate dilated conv.
    pub gate_conv: AgDilatedCausalConv1D,
    /// Conditioning projection: `[channels × cond_dim]`.
    pub cond_w: Vec<Vec<f64>>,
    /// Diffusion-step embedding projection: `[channels × embed_dim]`.
    pub diffusion_w: Vec<Vec<f64>>,
    /// Skip 1×1: `[skip_ch × channels]`.
    pub skip_w: Vec<Vec<f64>>,
    /// Residual channel count.
    pub channels: usize,
    /// Skip channel count.
    pub skip_ch: usize,
}

impl AgDiffWaveLayer {
    fn new(
        channels: usize,
        skip_ch: usize,
        cond_dim: usize,
        embed_dim: usize,
        dilation: usize,
        seed: &mut u64,
    ) -> Self {
        Self {
            residual_conv: AgDilatedCausalConv1D::new(channels, channels, 3, dilation, seed),
            gate_conv: AgDilatedCausalConv1D::new(channels, channels, 3, dilation, seed),
            cond_w: glorot_uniform(channels, cond_dim, seed),
            diffusion_w: glorot_uniform(channels, embed_dim, seed),
            skip_w: glorot_uniform(skip_ch, channels, seed),
            channels,
            skip_ch,
        }
    }
}

/// DiffWave score network (Kong et al. 2021).
///
/// Predicts the noise `ε_θ(x_t, t, c)` given a noisy waveform `x_t`, diffusion step `t`,
/// and optional conditioning `c`.
#[derive(Clone, Debug)]
pub struct AgDiffWave {
    /// Score-network residual layers.
    pub layers: Vec<AgDiffWaveLayer>,
    /// Input projection: `[channels × 1]` (mono audio).
    pub input_conv_w: Vec<Vec<f64>>,
    /// Output conv 1: `[skip_ch × skip_ch]`.
    pub output_conv1: Vec<Vec<f64>>,
    /// Output conv 2: `[1 × skip_ch]`.
    pub output_conv2: Vec<Vec<f64>>,
    /// Diffusion noise schedule.
    pub schedule: AgNoiseSchedule,
    /// Residual channel count.
    pub channels: usize,
    /// Conditioning dimension (0 = unconditional).
    pub cond_dim: usize,
}

impl AgDiffWave {
    const EMBED_DIM: usize = 64;

    /// Build a new DiffWave model.
    pub fn new(
        n_layers: usize,
        channels: usize,
        skip_ch: usize,
        cond_dim: usize,
        t_steps: usize,
        seed: &mut u64,
    ) -> Self {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let dilation = 1 << (i % 10); // wrap dilations at 512
            layers.push(AgDiffWaveLayer::new(
                channels,
                skip_ch,
                cond_dim.max(1),
                Self::EMBED_DIM,
                dilation,
                seed,
            ));
        }

        Self {
            layers,
            input_conv_w: glorot_uniform(channels, 1, seed),
            output_conv1: glorot_uniform(skip_ch, skip_ch, seed),
            output_conv2: glorot_uniform(1, skip_ch, seed),
            schedule: AgNoiseSchedule::linear(t_steps, 1e-4, 0.02),
            channels,
            cond_dim,
        }
    }

    /// Sinusoidal timestep embedding (same as Transformer positional encoding).
    fn timestep_embed(t: usize, dim: usize) -> Vec<f64> {
        let mut emb = vec![0.0_f64; dim];
        for i in 0..dim / 2 {
            let freq = 10000.0_f64.powf(2.0 * i as f64 / dim as f64);
            emb[2 * i] = (t as f64 / freq).sin();
            emb[2 * i + 1] = (t as f64 / freq).cos();
        }
        // Handle odd dim
        if dim % 2 == 1 {
            emb[dim - 1] = (t as f64).sin();
        }
        emb
    }

    /// Predict noise for a noisy waveform at diffusion step `t`.
    ///
    /// `x_t`: flat audio sample `[length]`
    /// `conditioning`: optional `[cond_dim]`
    /// Returns predicted noise `[length]`.
    pub fn forward(&self, x_t: &[f64], t: usize, conditioning: Option<&[f64]>) -> Vec<f64> {
        let length = x_t.len();
        // Project mono audio to [channels × length]
        let mut hidden: Vec<Vec<f64>> = (0..self.channels)
            .map(|oc| {
                let w = self.input_conv_w[oc][0];
                x_t.iter().map(|&v| w * v).collect()
            })
            .collect();

        // Timestep embedding → [EMBED_DIM]
        let t_emb = Self::timestep_embed(t, Self::EMBED_DIM);

        // Optionally project conditioning
        let cond_proj: Vec<f64> = if let Some(c) = conditioning {
            // simple mean-pool then project — returns [channels]
            let cd = self.cond_dim.max(1);
            let c_padded: Vec<f64> = (0..cd)
                .map(|i| if i < c.len() { c[i] } else { 0.0 })
                .collect();
            // Use the first layer's cond_w as a representative projection
            if !self.layers.is_empty() {
                mat_vec(&self.layers[0].cond_w, &c_padded)
            } else {
                vec![0.0; self.channels]
            }
        } else {
            vec![0.0; self.channels]
        };

        let mut skip_sum: Vec<f64> = vec![0.0; self.channels];

        for layer in &self.layers {
            // Diffusion step bias: project t_emb → [channels], broadcast over time
            let t_bias = mat_vec(&layer.diffusion_w, &t_emb);

            // Add t_bias and cond_proj to hidden (broadcast)
            let biased: Vec<Vec<f64>> = hidden
                .iter()
                .enumerate()
                .map(|(ch, hch)| {
                    let b = t_bias[ch] + cond_proj[ch.min(cond_proj.len() - 1)];
                    hch.iter().map(|&v| v + b).collect()
                })
                .collect();

            // Gated activation
            let f_out = layer.residual_conv.forward(&biased);
            let g_out = layer.gate_conv.forward(&biased);

            let gated: Vec<Vec<f64>> = (0..self.channels)
                .map(|ch| {
                    f_out[ch]
                        .iter()
                        .zip(&g_out[ch])
                        .map(|(&f, &g)| f.tanh() * (1.0 / (1.0 + (-g).exp())))
                        .collect()
                })
                .collect();

            // Accumulate skip
            let skip = mat_vec_cols(&layer.skip_w, &gated);
            for sc in 0..skip_sum.len() {
                let col_sum: f64 = skip[sc].iter().sum::<f64>();
                skip_sum[sc] += col_sum;
            }

            // Residual update: hidden = hidden + gated
            hidden = hidden
                .iter()
                .zip(gated.iter())
                .map(|(h, g)| h.iter().zip(g.iter()).map(|(&hv, &gv)| hv + gv).collect())
                .collect();
        }

        // Output: relu(skip_sum) → conv1 → relu → conv2 → noise prediction
        let after_relu: Vec<f64> = skip_sum.iter().map(|&v| relu(v)).collect();
        let after_conv1: Vec<f64> = mat_vec(&self.output_conv1, &after_relu)
            .into_iter()
            .map(relu)
            .collect();
        let noise_pred_scalar = mat_vec(&self.output_conv2, &after_conv1)[0];

        // Broadcast scalar noise prediction over all samples
        vec![noise_pred_scalar; length]
    }

    /// DDPM ancestral sampling.
    pub fn ddpm_generate(
        &self,
        length: usize,
        conditioning: Option<&[f64]>,
        seed: &mut u64,
    ) -> Vec<f64> {
        // Start from Gaussian noise
        let mut x_t: Vec<f64> = (0..length).map(|_| ag_randn(seed)).collect();

        for t_rev in (0..self.schedule.T).rev() {
            let pred_noise = self.forward(&x_t, t_rev, conditioning);
            x_t = self.schedule.denoise_step(&x_t, &pred_noise, t_rev);
            // Add stochastic noise at intermediate steps
            if t_rev > 0 {
                let beta_t = self.schedule.betas[t_rev];
                let sigma = beta_t.sqrt();
                for v in x_t.iter_mut() {
                    *v += sigma * ag_randn(seed);
                }
            }
        }
        x_t
    }

    /// DDIM deterministic sampling with `n_steps` steps (step skipping).
    pub fn ddim_generate(
        &self,
        length: usize,
        n_steps: usize,
        conditioning: Option<&[f64]>,
        seed: &mut u64,
    ) -> Vec<f64> {
        let full_t = self.schedule.T;
        let step_size = full_t / n_steps.max(1);

        // Collect timesteps in reverse
        let timesteps: Vec<usize> = (0..n_steps)
            .map(|i| (n_steps - 1 - i) * step_size)
            .collect();

        let mut x_t: Vec<f64> = (0..length).map(|_| ag_randn(seed)).collect();

        for &t in &timesteps {
            let pred_noise = self.forward(&x_t, t, conditioning);

            let ab_t = self.schedule.alpha_bar[t.min(self.schedule.T - 1)];
            let ab_prev = if t >= step_size {
                self.schedule.alpha_bar[(t - step_size).min(self.schedule.T - 1)]
            } else {
                1.0
            };

            let sqrt_ab_t = ab_t.sqrt().max(1e-12);
            let sqrt_one_minus_ab_t = (1.0 - ab_t).max(0.0).sqrt();

            // DDIM update (η = 0: deterministic)
            x_t = x_t
                .iter()
                .zip(&pred_noise)
                .map(|(&xt, &eps)| {
                    let x0_pred = (xt - sqrt_one_minus_ab_t * eps) / sqrt_ab_t;
                    let x0_pred = x0_pred.clamp(-1.0, 1.0);
                    ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).max(0.0).sqrt() * eps
                })
                .collect();
        }
        x_t
    }
}

/// Helper: apply matrix `m [out × in_ch]` to each column of `x [in_ch × length]`.
/// Returns `[out × length]`.
fn mat_vec_cols(m: &[Vec<f64>], x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let out_ch = m.len();
    let length = if x.is_empty() { 0 } else { x[0].len() };
    let in_ch = x.len();
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; length]; out_ch];
    for t in 0..length {
        let x_t: Vec<f64> = (0..in_ch).map(|c| x[c][t]).collect();
        let projected = mat_vec(m, &x_t);
        for oc in 0..out_ch {
            result[oc][t] = projected[oc];
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  AgGriffinLim
// ─────────────────────────────────────────────────────────────────────────────

/// Griffin-Lim iterative phase reconstruction vocoder.
///
/// Reconstructs a time-domain signal from a magnitude spectrogram by
/// alternately enforcing the magnitude constraint and the STFT consistency
/// constraint.
#[derive(Clone, Debug)]
pub struct AgGriffinLim {
    /// FFT size.
    pub n_fft: usize,
    /// Hop length (samples between successive STFT frames).
    pub hop_length: usize,
    /// Number of Griffin-Lim iterations.
    pub n_iter: usize,
}

impl AgGriffinLim {
    /// Construct a new Griffin-Lim vocoder.
    pub fn new(n_fft: usize, hop_length: usize, n_iter: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            n_iter,
        }
    }

    /// Short-time Fourier transform with a Hann window.
    ///
    /// Returns `[time_frames × freq_bins]` complex where `freq_bins = n_fft/2 + 1`.
    pub fn stft(&self, signal: &[f64]) -> Vec<Vec<(f64, f64)>> {
        let n = signal.len();
        let n_fft = self.n_fft;
        let hop = self.hop_length;
        let freq_bins = n_fft / 2 + 1;

        // Number of frames (pad signal so every frame has n_fft samples)
        let n_frames = if n <= n_fft { 1 } else { (n - n_fft) / hop + 1 };

        let hann: Vec<f64> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / n_fft as f64).cos()))
            .collect();

        let mut frames: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            // Windowed frame
            let frame: Vec<f64> = (0..n_fft)
                .map(|k| {
                    let sample_idx = start + k;
                    let s = if sample_idx < n {
                        signal[sample_idx]
                    } else {
                        0.0
                    };
                    s * hann[k]
                })
                .collect();

            // DFT for positive frequencies (direct formula; O(n_fft²) — acceptable for small n_fft)
            let mut spectrum: Vec<(f64, f64)> = Vec::with_capacity(freq_bins);
            for bin in 0..freq_bins {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for k in 0..n_fft {
                    let angle = -2.0 * PI * bin as f64 * k as f64 / n_fft as f64;
                    re += frame[k] * angle.cos();
                    im += frame[k] * angle.sin();
                }
                spectrum.push((re, im));
            }
            frames.push(spectrum);
        }
        frames
    }

    /// Inverse STFT via overlap-add.
    ///
    /// `stft`: `[time_frames × freq_bins]` complex.
    /// Returns the reconstructed signal.
    pub fn istft(&self, stft: &[Vec<(f64, f64)>]) -> Vec<f64> {
        let n_fft = self.n_fft;
        let hop = self.hop_length;
        let n_frames = stft.len();
        let output_len = (n_frames - 1) * hop + n_fft;

        let hann: Vec<f64> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / n_fft as f64).cos()))
            .collect();

        let mut signal = vec![0.0_f64; output_len];
        let mut weight = vec![0.0_f64; output_len];

        for (frame_idx, spectrum) in stft.iter().enumerate() {
            let start = frame_idx * hop;

            // Reconstruct the time-domain frame via inverse DFT
            let mut frame = vec![0.0_f64; n_fft];
            for k in 0..n_fft {
                let mut re = 0.0_f64;
                // Sum positive freqs and mirror conjugates
                for (bin, &(sr, si)) in spectrum.iter().enumerate() {
                    let angle = 2.0 * PI * bin as f64 * k as f64 / n_fft as f64;
                    if bin == 0 || bin == n_fft / 2 {
                        re += sr * angle.cos() - si * angle.sin();
                    } else {
                        // Double for conjugate symmetry
                        re += 2.0 * (sr * angle.cos() - si * angle.sin());
                    }
                }
                frame[k] = re / n_fft as f64;
            }

            // Overlap-add with Hann window
            for k in 0..n_fft {
                signal[start + k] += frame[k] * hann[k];
                weight[start + k] += hann[k] * hann[k];
            }
        }

        // Normalize by overlap-add weights
        for (s, &w) in signal.iter_mut().zip(&weight) {
            if w > 1e-12 {
                *s /= w;
            }
        }
        signal
    }

    /// Griffin-Lim reconstruction from a magnitude spectrogram.
    ///
    /// `magnitude`: `[time_frames × freq_bins]` non-negative magnitudes.
    /// Returns reconstructed waveform.
    pub fn reconstruct(&self, magnitude: &[Vec<f64>]) -> Vec<f64> {
        let n_frames = magnitude.len();
        if n_frames == 0 {
            return Vec::new();
        }
        let freq_bins = magnitude[0].len();

        // Initialise with random phase (use deterministic seed for reproducibility)
        let mut seed = 42u64;
        let mut complex_spec: Vec<Vec<(f64, f64)>> = magnitude
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|&mag| {
                        let phase = ag_uniform(&mut seed) * 2.0 * PI;
                        (mag * phase.cos(), mag * phase.sin())
                    })
                    .collect()
            })
            .collect();

        for _ in 0..self.n_iter {
            // Time-domain signal from current estimate
            let signal = self.istft(&complex_spec);
            // Re-analyse to get consistent phases
            let new_spec = self.stft(&signal);

            // Apply magnitude constraint: keep magnitude, update phase from new_spec
            let new_len = new_spec.len().min(n_frames);
            for frame_idx in 0..new_len {
                for bin in 0..freq_bins {
                    if bin < new_spec[frame_idx].len() {
                        let (re, im) = new_spec[frame_idx][bin];
                        let angle = (re * re + im * im).sqrt();
                        let mag = if frame_idx < magnitude.len() && bin < magnitude[frame_idx].len()
                        {
                            magnitude[frame_idx][bin]
                        } else {
                            0.0
                        };
                        if angle > 1e-12 {
                            complex_spec[frame_idx][bin] = (mag * re / angle, mag * im / angle);
                        } else {
                            complex_spec[frame_idx][bin] = (mag, 0.0);
                        }
                    }
                }
            }
        }

        self.istft(&complex_spec)
    }

    /// Power spectrogram `|STFT|²`.
    ///
    /// Returns `[time_frames × freq_bins]` non-negative values.
    pub fn power_spectrogram(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        self.stft(signal)
            .iter()
            .map(|frame| frame.iter().map(|&(re, im)| re * re + im * im).collect())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  AgHifiGanGenerator
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-Receptive-Field Fusion block for HiFi-GAN.
///
/// Sums responses of `n_kernels` convolutional kernels with different sizes,
/// providing multi-scale temporal context.
#[derive(Clone, Debug)]
pub struct AgMrfBlock {
    /// `[n_kernels × out_ch × (out_ch * kernel)]` — one kernel matrix per receptive field size.
    pub convs: Vec<Vec<Vec<f64>>>,
    /// Number of distinct kernel sizes.
    pub n_kernels: usize,
    /// Channel count.
    pub channels: usize,
}

impl AgMrfBlock {
    fn new(channels: usize, n_kernels: usize, kernel_sizes: &[usize], seed: &mut u64) -> Self {
        let convs: Vec<Vec<Vec<f64>>> = kernel_sizes
            .iter()
            .map(|&ks| glorot_uniform(channels, channels * ks, seed))
            .collect();
        Self {
            convs,
            n_kernels,
            channels,
        }
    }
}

/// HiFi-GAN-style upsampling vocoder generator (simplified, pure-Rust).
///
/// Converts mel-spectrogram frames to audio via repeated linear upsampling and
/// multi-receptive-field fusion residual blocks.
#[derive(Clone, Debug)]
pub struct AgHifiGanGenerator {
    /// Input projection: `[hidden_ch × mel_dim]`.
    pub pre_conv: Vec<Vec<f64>>,
    /// Upsampling convolutions: `[n_upsampling × hidden_ch × hidden_ch]`.
    pub upsampling_convs: Vec<Vec<Vec<f64>>>,
    /// MRF blocks, one per upsampling stage.
    pub mrf_blocks: Vec<AgMrfBlock>,
    /// Output projection: `[1 × hidden_ch]`.
    pub post_conv: Vec<Vec<f64>>,
    /// Hidden channel count.
    pub hidden_ch: usize,
    /// Number of upsampling stages.
    pub n_upsampling: usize,
}

impl AgHifiGanGenerator {
    /// Construct with random weights.
    ///
    /// `hidden_ch`: number of hidden channels.
    /// `n_upsampling`: number of ×2 upsampling stages.
    pub fn new(hidden_ch: usize, n_upsampling: usize, seed: &mut u64) -> Self {
        // pre_conv: project from mel (we keep it square here; caller selects mel_dim = hidden_ch)
        let pre_conv = glorot_uniform(hidden_ch, hidden_ch, seed);

        let upsampling_convs: Vec<Vec<Vec<f64>>> = (0..n_upsampling)
            .map(|_| glorot_uniform(hidden_ch, hidden_ch, seed))
            .collect();

        // Kernel sizes 3, 7, 11 for MRF
        let kernel_sizes = vec![3_usize, 7, 11];
        let mrf_blocks: Vec<AgMrfBlock> = (0..n_upsampling)
            .map(|_| AgMrfBlock::new(hidden_ch, kernel_sizes.len(), &kernel_sizes, seed))
            .collect();

        let post_conv = glorot_uniform(1, hidden_ch, seed);

        Self {
            pre_conv,
            upsampling_convs,
            mrf_blocks,
            post_conv,
            hidden_ch,
            n_upsampling,
        }
    }

    /// Linear interpolation upsampling: increase time dimension by `factor`.
    ///
    /// `x`: `[channels × frames]` → `[channels × (frames * factor)]`.
    pub fn upsample(&self, x: &[Vec<f64>], factor: usize) -> Vec<Vec<f64>> {
        if x.is_empty() || factor == 0 {
            return x.to_vec();
        }
        let ch = x.len();
        let frames = x[0].len();
        let new_len = frames * factor;

        (0..ch)
            .map(|c| {
                (0..new_len)
                    .map(|t| {
                        // Linear interpolation between frames
                        let src_f = t as f64 / factor as f64;
                        let lo = src_f.floor() as usize;
                        let hi = (lo + 1).min(frames - 1);
                        let frac = src_f - lo as f64;
                        x[c][lo] * (1.0 - frac) + x[c][hi] * frac
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply MRF block `block_idx` to `x [channels × length]`.
    ///
    /// Returns `[channels × length]` — sum of multi-kernel projections + residual.
    pub fn mrf_forward(&self, x: &[Vec<f64>], block_idx: usize) -> Vec<Vec<f64>> {
        if block_idx >= self.mrf_blocks.len() || x.is_empty() {
            return x.to_vec();
        }
        let block = &self.mrf_blocks[block_idx];
        let ch = x.len();
        let length = x[0].len();

        let mut sum_out: Vec<Vec<f64>> = vec![vec![0.0; length]; ch];

        for (k_idx, kernel) in block.convs.iter().enumerate() {
            let ks = [3_usize, 7, 11][k_idx.min(2)];
            // Apply a simplified 1-D conv (causal, no padding for simplicity)
            for oc in 0..ch.min(kernel.len()) {
                for t in 0..length {
                    // Gather window across channels × time; flattened: [ch * ks]
                    let mut window = Vec::with_capacity(ch * ks);
                    for k in 0..ks {
                        let src_t = (t + k).saturating_sub(ks - 1);
                        for ic in 0..ch {
                            window.push(x[ic][src_t]);
                        }
                    }
                    let row = &kernel[oc];
                    let w_len = row.len().min(window.len());
                    let val: f64 = row[..w_len]
                        .iter()
                        .zip(&window[..w_len])
                        .map(|(a, b)| a * b)
                        .sum();
                    sum_out[oc][t] += relu(val);
                }
            }
        }

        // Residual connection + average over kernels
        let n_k = block.convs.len().max(1) as f64;
        (0..ch)
            .map(|c| (0..length).map(|t| x[c][t] + sum_out[c][t] / n_k).collect())
            .collect()
    }

    /// Generate waveform from mel spectrogram.
    ///
    /// `mel`: `[mel_dim × frames]`
    /// Returns audio `[frames × 2^n_upsampling]`.
    pub fn forward(&self, mel: &[Vec<f64>]) -> Vec<f64> {
        if mel.is_empty() {
            return Vec::new();
        }

        // Project mel → [hidden_ch × frames]
        let frames = mel[0].len();
        let mel_dim = mel.len().min(self.pre_conv[0].len()); // Clamp to weight col count

        let mut hidden: Vec<Vec<f64>> = (0..self.hidden_ch)
            .map(|oc| {
                (0..frames)
                    .map(|t| {
                        let mut acc = 0.0_f64;
                        for ic in 0..mel_dim {
                            if ic < mel.len()
                                && oc < self.pre_conv.len()
                                && ic < self.pre_conv[oc].len()
                            {
                                acc += self.pre_conv[oc][ic] * mel[ic][t];
                            }
                        }
                        relu(acc)
                    })
                    .collect()
            })
            .collect();

        // Upsample + MRF for each stage
        for stage in 0..self.n_upsampling {
            hidden = self.upsample(&hidden, 2);
            hidden = self.mrf_forward(&hidden, stage);
        }

        // Project to mono: post_conv [1 × hidden_ch]
        let out_len = hidden[0].len();
        (0..out_len)
            .map(|t| {
                let h_t: Vec<f64> = hidden.iter().map(|ch| ch[t]).collect();
                let out = mat_vec(&self.post_conv, &h_t)[0];
                out.tanh()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  AgTextToSpeechPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Simple neural TTS pipeline: text tokens → mel spectrogram → waveform.
///
/// Intended as a minimal but complete demonstration of the TTS pipeline pattern.
/// The encoder is a linear embedding + duration-guided frame expansion; the vocoder
/// uses Griffin-Lim.
#[derive(Clone, Debug)]
pub struct AgTextToSpeechPipeline {
    /// Token embedding table: `[embed_dim × vocab_size]`.
    pub text_embed: Vec<Vec<f64>>,
    /// Linear encoder: `[mel_dim × embed_dim]`.
    pub encoder_w: Vec<Vec<f64>>,
    /// Encoder bias: `[mel_dim]`.
    pub encoder_b: Vec<f64>,
    /// Duration predictor weights: `[1 × embed_dim]`.
    pub duration_predictor: Vec<Vec<f64>>,
    /// Griffin-Lim vocoder.
    pub vocoder: AgGriffinLim,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Mel-spectrogram dimension.
    pub mel_dim: usize,
}

impl AgTextToSpeechPipeline {
    /// Build with deterministic weights (seed 1337).
    pub fn new(vocab_size: usize, embed_dim: usize, mel_dim: usize) -> Self {
        let mut seed = 1337u64;
        Self {
            text_embed: glorot_uniform(embed_dim, vocab_size, &mut seed),
            encoder_w: glorot_uniform(mel_dim, embed_dim, &mut seed),
            encoder_b: vec![0.0; mel_dim],
            duration_predictor: glorot_uniform(1, embed_dim, &mut seed),
            vocoder: AgGriffinLim::new(128, 32, 5),
            vocab_size,
            embed_dim,
            mel_dim,
        }
    }

    /// Predict duration (in frames) for a token embedding.
    ///
    /// `duration = max(1, round(|w · emb|))`
    pub fn predict_duration(&self, token_embedding: &[f64]) -> usize {
        let raw = mat_vec(&self.duration_predictor, token_embedding)[0].abs();
        // Scale to a reasonable frame count: clamp to [1, 20]
        let frames = (raw * 10.0 + 1.0).round() as usize;
        frames.clamp(1, 20)
    }

    /// Encode token IDs to a mel spectrogram.
    ///
    /// Each token is embedded, its duration is predicted, and the mel frame is
    /// repeated for `duration` frames.
    ///
    /// Returns `[mel_dim × n_frames]`.
    pub fn encode_text(&self, token_ids: &[usize]) -> Vec<Vec<f64>> {
        if token_ids.is_empty() {
            return vec![Vec::new(); self.mel_dim];
        }

        let mut all_frames: Vec<Vec<f64>> = Vec::new(); // frames accumulated

        for &tok in token_ids {
            let tok_idx = tok % self.vocab_size;
            // Embed: look up column tok_idx of text_embed → [embed_dim]
            let embed: Vec<f64> = self.text_embed.iter().map(|row| row[tok_idx]).collect();

            // Predict duration
            let dur = self.predict_duration(&embed);

            // Mel frame: encoder_w @ embed + encoder_b → [mel_dim]
            let mel_frame: Vec<f64> = mat_vec(&self.encoder_w, &embed)
                .iter()
                .zip(&self.encoder_b)
                .map(|(&v, &b)| v + b)
                .collect();

            // Expand by duration
            for _ in 0..dur {
                all_frames.push(mel_frame.clone());
            }
        }

        if all_frames.is_empty() {
            return vec![Vec::new(); self.mel_dim];
        }

        // Transpose from [n_frames × mel_dim] to [mel_dim × n_frames]
        let n_frames = all_frames.len();
        (0..self.mel_dim)
            .map(|d| all_frames.iter().map(|f| f[d]).collect())
            .collect()
    }

    /// Full TTS synthesis: token IDs → audio waveform.
    pub fn synthesize(&self, token_ids: &[usize]) -> Vec<f64> {
        let mel = self.encode_text(token_ids);
        // Transpose to [time_frames × mel_dim] for Griffin-Lim
        if mel.is_empty() || mel[0].is_empty() {
            return Vec::new();
        }
        let n_frames = mel[0].len();
        let magnitude: Vec<Vec<f64>> = (0..n_frames)
            .map(|t| mel.iter().map(|m| m[t].abs()).collect())
            .collect();
        self.vocoder.reconstruct(&magnitude)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  AgAudioCodec
// ─────────────────────────────────────────────────────────────────────────────

/// SoundStream-inspired neural audio codec.
///
/// Encodes audio frames into discrete codebook indices and decodes them back.
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct AgAudioCodec {
    /// Encoder weight: `[latent_dim × input_dim]`.
    pub encoder_w: Vec<Vec<f64>>,
    /// Encoder bias: `[latent_dim]`.
    pub encoder_b: Vec<f64>,
    /// Vector-quantization codebook: `[K × latent_dim]`.
    pub codebook: Vec<Vec<f64>>,
    /// Decoder weight: `[output_dim × latent_dim]`.
    pub decoder_w: Vec<Vec<f64>>,
    /// Decoder bias: `[output_dim]`.
    pub decoder_b: Vec<f64>,
    /// Codebook size.
    pub K: usize,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Input frame dimension.
    pub input_dim: usize,
    /// Decoded output dimension (= input_dim for reconstruction).
    pub output_dim: usize,
}

impl AgAudioCodec {
    /// Construct with random weights.
    pub fn new(input_dim: usize, latent_dim: usize, k: usize, seed: &mut u64) -> Self {
        let codebook: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..latent_dim).map(|_| ag_randn(seed) * 0.1).collect())
            .collect();
        Self {
            encoder_w: glorot_uniform(latent_dim, input_dim, seed),
            encoder_b: vec![0.0; latent_dim],
            codebook,
            decoder_w: glorot_uniform(input_dim, latent_dim, seed),
            decoder_b: vec![0.0; input_dim],
            K: k,
            latent_dim,
            input_dim,
            output_dim: input_dim,
        }
    }

    /// Encode a single audio frame.
    ///
    /// Returns `(quantized_latent, codebook_index)`.
    pub fn encode(&self, frame: &[f64]) -> (Vec<f64>, usize) {
        // Linear encoder + tanh activation
        let z: Vec<f64> = mat_vec(&self.encoder_w, frame)
            .iter()
            .zip(&self.encoder_b)
            .map(|(&v, &b)| (v + b).tanh())
            .collect();

        // Find nearest codebook entry (L2)
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (idx, entry) in self.codebook.iter().enumerate() {
            let dist: f64 = z
                .iter()
                .zip(entry)
                .map(|(&zi, &ei)| (zi - ei).powi(2))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        (self.codebook[best_idx].clone(), best_idx)
    }

    /// Decode codebook entry `code` to audio frame.
    ///
    /// Returns `[output_dim]` via `sigmoid(W_d @ e_k + b_d)`.
    pub fn decode(&self, code: usize) -> Vec<f64> {
        let entry = &self.codebook[code.min(self.K - 1)];
        mat_vec(&self.decoder_w, entry)
            .iter()
            .zip(&self.decoder_b)
            .map(|(&v, &b)| 1.0 / (1.0 + (-(v + b)).exp()))
            .collect()
    }

    /// Encode a full audio signal in non-overlapping frames.
    ///
    /// Returns the sequence of codebook indices.
    pub fn encode_sequence(&self, audio: &[f64], frame_size: usize) -> Vec<usize> {
        if frame_size == 0 || audio.is_empty() {
            return Vec::new();
        }
        let n_frames = audio.len() / frame_size;
        (0..n_frames)
            .map(|i| {
                let frame = &audio[i * frame_size..(i + 1) * frame_size];
                // Pad or truncate to input_dim
                let mut padded = vec![0.0; self.input_dim];
                for (j, &v) in frame.iter().enumerate().take(self.input_dim) {
                    padded[j] = v;
                }
                let (_, code) = self.encode(&padded);
                code
            })
            .collect()
    }

    /// Decode a code sequence to audio.
    pub fn decode_sequence(&self, codes: &[usize]) -> Vec<f64> {
        codes.iter().flat_map(|&c| self.decode(c)).collect()
    }

    /// Compression ratio = bits-per-sample input / bits-per-sample compressed.
    ///
    /// `input_sr`: input sample rate (Hz).
    /// `frame_size`: audio samples per encoded frame.
    pub fn compression_ratio(&self, input_sr: usize, frame_size: usize) -> f64 {
        // Input: 16-bit PCM = 16 bits/sample
        let bits_per_sample_input = 16.0_f64;
        // Compressed: log2(K) bits per frame, spread over frame_size samples
        let bits_per_sample_compressed = (self.K as f64).log2() / frame_size.max(1) as f64;
        bits_per_sample_input / bits_per_sample_compressed.max(1e-12)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  AgMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Audio quality metrics: SNR, SI-SDR, spectral convergence, log-spectral distance, MOS.
pub struct AgMetrics;

impl AgMetrics {
    /// Signal-to-Noise Ratio (dB).
    ///
    /// `10 * log10(signal_power / noise_power)`
    pub fn snr(original: &[f64], reconstructed: &[f64]) -> f64 {
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return 0.0;
        }
        let signal_power: f64 = original[..n].iter().map(|&v| v * v).sum::<f64>() / n as f64;
        let noise_power: f64 = original[..n]
            .iter()
            .zip(&reconstructed[..n])
            .map(|(&o, &r)| (o - r).powi(2))
            .sum::<f64>()
            / n as f64;
        if noise_power < 1e-40 {
            return 100.0;
        }
        10.0 * (signal_power / noise_power).log10()
    }

    /// Scale-Invariant Signal-to-Distortion Ratio (SI-SDR, dB).
    ///
    /// `alpha = <ŝ, s> / ||s||²; SI-SDR = 10 log10(||α s||² / ||ŝ − α s||²)`
    pub fn si_sdr(reference: &[f64], estimated: &[f64]) -> f64 {
        let n = reference.len().min(estimated.len());
        if n == 0 {
            return 0.0;
        }

        let dot_prod: f64 = reference[..n]
            .iter()
            .zip(&estimated[..n])
            .map(|(&r, &e)| r * e)
            .sum();
        let ref_power: f64 = reference[..n].iter().map(|&r| r * r).sum();

        if ref_power < 1e-40 {
            return -100.0;
        }

        let alpha = dot_prod / ref_power;
        let signal_power: f64 = reference[..n].iter().map(|&r| (alpha * r).powi(2)).sum();
        let noise_power: f64 = reference[..n]
            .iter()
            .zip(&estimated[..n])
            .map(|(&r, &e)| (e - alpha * r).powi(2))
            .sum();

        if noise_power < 1e-40 {
            return 100.0;
        }
        10.0 * (signal_power / noise_power).log10()
    }

    /// Spectral convergence: `||ref_mag − est_mag||_F / ||ref_mag||_F`.
    ///
    /// Value in `[0, ∞)`, ideally close to 0; clipped to `[0, 1]` for reporting.
    pub fn spectral_convergence(reference_spec: &[Vec<f64>], estimated_spec: &[Vec<f64>]) -> f64 {
        let n_frames = reference_spec.len().min(estimated_spec.len());
        if n_frames == 0 {
            return 0.0;
        }

        let mut diff_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;

        for i in 0..n_frames {
            let n_bins = reference_spec[i].len().min(estimated_spec[i].len());
            for j in 0..n_bins {
                let r = reference_spec[i][j];
                let e = estimated_spec[i][j];
                diff_sq += (r - e).powi(2);
                ref_sq += r * r;
            }
        }

        if ref_sq < 1e-40 {
            return 0.0;
        }
        (diff_sq / ref_sq).sqrt().min(1.0)
    }

    /// Log-spectral distance: mean absolute difference of log magnitude spectra.
    pub fn log_spectral_distance(reference_spec: &[Vec<f64>], estimated_spec: &[Vec<f64>]) -> f64 {
        let n_frames = reference_spec.len().min(estimated_spec.len());
        if n_frames == 0 {
            return 0.0;
        }

        let mut total = 0.0_f64;
        let mut count = 0usize;

        for i in 0..n_frames {
            let n_bins = reference_spec[i].len().min(estimated_spec[i].len());
            for j in 0..n_bins {
                let r = reference_spec[i][j].max(1e-10);
                let e = estimated_spec[i][j].max(1e-10);
                total += (r.ln() - e.ln()).abs();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }
        total / count as f64
    }

    /// Rough Mean Opinion Score estimate from SI-SDR.
    ///
    /// Maps SI-SDR to `[1, 5]` via a sigmoid: `1 + 4 / (1 + exp(-0.5 * (si_sdr - 20)))`.
    pub fn mos_estimate(si_sdr: f64) -> f64 {
        1.0 + 4.0 / (1.0 + (-0.5 * (si_sdr - 20.0)).exp())
    }
}
