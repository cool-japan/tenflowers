//! Audio model extensions: vocoder, speaker recognition, and augmentation.

use super::{dot, he_init, inval, l2_norm, layer_norm_rows, linear_fwd, log_sum_exp, relu};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// Griffin-Lim iterative phase reconstruction vocoder.
#[derive(Debug, Clone)]
pub struct GriffinLimVocoder {
    /// FFT size.
    pub n_fft: usize,
    /// Hop length between frames.
    pub hop_length: usize,
}

impl GriffinLimVocoder {
    /// Create a new Griffin-Lim vocoder.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self { n_fft, hop_length }
    }

    /// Reconstruct waveform from magnitude spectrogram via Griffin-Lim.
    ///
    /// `magnitude`: `[T, n_fft/2+1]` flat (linear amplitude).
    /// Returns waveform of length `≈ (T-1)*hop_length + n_fft`.
    pub fn iterate(&self, magnitude: &[f32], n_iter: usize, seed: u64) -> Result<Vec<f32>> {
        let n = self.n_fft;
        let nf = n / 2 + 1;
        if magnitude.is_empty() || magnitude.len() % nf != 0 {
            return Err(inval(
                "GriffinLimVocoder::iterate",
                format!(
                    "magnitude.len() {} not divisible by n_freqs {nf}",
                    magnitude.len()
                ),
            ));
        }
        let tf = magnitude.len() / nf;
        let slen = (tf - 1) * self.hop_length + n;
        let pi2 = 2.0 * std::f32::consts::PI;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut phase: Vec<(f32, f32)> = (0..tf * nf)
            .map(|_| {
                let a = rng.random::<f32>() * pi2;
                (a.cos(), a.sin())
            })
            .collect();
        let hann: Vec<f32> = (0..n)
            .map(|j| 0.5 * (1.0 - (pi2 * j as f32 / (n - 1) as f32).cos()))
            .collect();
        let mut signal = vec![0.0_f32; slen];
        for _iter in 0..n_iter {
            signal = vec![0.0_f32; slen];
            let mut wsum = vec![0.0_f32; slen];
            for ti in 0..tf {
                let mut frame = vec![0.0_f32; n];
                for k in 0..nf {
                    let (re, im) = (
                        magnitude[ti * nf + k] * phase[ti * nf + k].0,
                        magnitude[ti * nf + k] * phase[ti * nf + k].1,
                    );
                    for j in 0..n {
                        let a = pi2 * k as f32 * j as f32 / n as f32;
                        frame[j] += (re * a.cos() - im * a.sin()) / n as f32;
                        if k > 0 && k < nf - 1 {
                            frame[j] += (re * a.cos() + im * a.sin()) / n as f32;
                        }
                    }
                }
                let off = ti * self.hop_length;
                for j in 0..n {
                    if off + j < slen {
                        signal[off + j] += frame[j] * hann[j];
                        wsum[off + j] += hann[j] * hann[j];
                    }
                }
            }
            for i in 0..slen {
                if wsum[i] > 1e-8 {
                    signal[i] /= wsum[i];
                }
            }
            for ti in 0..tf {
                let off = ti * self.hop_length;
                for k in 0..nf {
                    let (mut re, mut im) = (0.0_f32, 0.0_f32);
                    for j in 0..n {
                        if off + j < slen {
                            let a = -pi2 * k as f32 * j as f32 / n as f32;
                            let s = signal[off + j] * hann[j];
                            re += s * a.cos();
                            im += s * a.sin();
                        }
                    }
                    let m = (re * re + im * im).sqrt().max(1e-8);
                    phase[ti * nf + k] = (re / m, im / m);
                }
            }
        }
        Ok(signal)
    }
}

/// Time-Delay Neural Network (TDNN) layer: dilated 1D conv over a context window.
#[derive(Debug, Clone)]
pub struct TdnnLayer {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output feature dimension.
    pub output_dim: usize,
    /// Number of context frames (kernel size).
    pub context: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Convolution weights: `[out_d, in_d, context]`.
    pub weights: Vec<f32>,
    /// Output biases: `[out_d]`.
    pub bias: Vec<f32>,
}

impl TdnnLayer {
    /// Create a new TDNN layer.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        context: usize,
        dilation: usize,
        seed: u64,
    ) -> Self {
        let weights = he_init(output_dim * input_dim * context, input_dim * context, seed);
        let bias = vec![0.0_f32; output_dim];
        Self {
            input_dim,
            output_dim,
            context,
            dilation,
            weights,
            bias,
        }
    }

    /// Forward: `[T, in_d]` → `[T', out_d]` where T' accounts for valid convolution.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let in_d = self.input_dim;
        let out_d = self.output_dim;
        if x.len() % in_d != 0 {
            return Err(inval(
                "TdnnLayer::forward",
                format!("x.len() {} not divisible by input_dim {in_d}", x.len()),
            ));
        }
        let t = x.len() / in_d;
        let ks = self.context;
        let dil = self.dilation;
        let span = (ks - 1) * dil + 1;
        if t < span {
            return Err(inval(
                "TdnnLayer::forward",
                format!("input length {t} < effective span {span}"),
            ));
        }
        let t_out = t - span + 1;
        let mut out = vec![0.0_f32; t_out * out_d];
        for ti in 0..t_out {
            for oc in 0..out_d {
                let mut s = self.bias[oc];
                for ic in 0..in_d {
                    for ki in 0..ks {
                        let src_t = ti + ki * dil;
                        s += self.weights[oc * in_d * ks + ic * ks + ki] * x[src_t * in_d + ic];
                    }
                }
                out[ti * out_d + oc] = relu(s);
            }
        }
        Ok(out)
    }
}

/// X-vector extractor: stack of TDNN layers + statistics pooling + affine.
#[derive(Debug, Clone)]
pub struct XVectorExtractor {
    /// Stack of TDNN layers.
    pub layers: Vec<TdnnLayer>,
    /// Affine weights after statistics pooling: `[embed_dim, 2 * last_out_dim]`.
    pub stats_affine_w: Vec<f32>,
    /// Affine biases after statistics pooling.
    pub stats_affine_b: Vec<f32>,
    /// Embedding (output) dimension.
    pub embed_dim: usize,
    /// Output dimension of the last TDNN layer.
    pub last_out_dim: usize,
}

impl XVectorExtractor {
    /// Create an x-vector extractor with the standard 5-layer architecture.
    /// `hidden_dim` controls the TDNN hidden size (512 in the original paper).
    pub fn new(input_dim: usize, embed_dim: usize, seed: u64) -> Self {
        Self::with_hidden(input_dim, embed_dim, 512, seed)
    }

    /// Create with a custom hidden dimension (useful for lightweight/test configs).
    pub fn with_hidden(input_dim: usize, embed_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let last_dim = hidden_dim * 3; // stats pooling dimension
        let layers = vec![
            TdnnLayer::new(input_dim, hidden_dim, 5, 1, seed),
            TdnnLayer::new(hidden_dim, hidden_dim, 3, 2, seed + 1),
            TdnnLayer::new(hidden_dim, hidden_dim, 3, 3, seed + 2),
            TdnnLayer::new(hidden_dim, hidden_dim, 1, 1, seed + 3),
            TdnnLayer::new(hidden_dim, last_dim, 1, 1, seed + 4),
        ];
        let stats_affine_w = he_init(embed_dim * 2 * last_dim, 2 * last_dim, seed + 5);
        let stats_affine_b = vec![0.0_f32; embed_dim];
        Self {
            layers,
            stats_affine_w,
            stats_affine_b,
            embed_dim,
            last_out_dim: last_dim,
        }
    }

    /// Forward: `[T, input_dim]` → `[embed_dim]` speaker embedding.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            if h.len() < layer.input_dim {
                return Err(inval(
                    "XVectorExtractor::forward",
                    format!("input too short: {} < {}", h.len(), layer.input_dim),
                ));
            }
            h = layer.forward(&h)?;
        }
        let ld = self.last_out_dim;
        let t = h.len().checked_div(ld).unwrap_or(0);
        if t == 0 {
            return Err(inval(
                "XVectorExtractor::forward",
                "time dim is 0 after TDNN layers",
            ));
        }
        let mut mean = vec![0.0_f32; ld];
        for ti in 0..t {
            for j in 0..ld {
                mean[j] += h[ti * ld + j];
            }
        }
        for v in mean.iter_mut() {
            *v /= t as f32;
        }
        let mut std_d = vec![0.0_f32; ld];
        for ti in 0..t {
            for j in 0..ld {
                let d = h[ti * ld + j] - mean[j];
                std_d[j] += d * d;
            }
        }
        for v in std_d.iter_mut() {
            *v = (*v / t as f32).sqrt();
        }
        let pooled: Vec<f32> = mean.iter().chain(std_d.iter()).copied().collect();
        Ok(linear_fwd(
            &pooled,
            &self.stats_affine_w,
            &self.stats_affine_b,
            2 * ld,
            self.embed_dim,
        ))
    }
}

/// Speaker embedding: L2-normalized d-vector from x-vector.
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// Underlying x-vector extractor.
    pub extractor: XVectorExtractor,
}

impl SpeakerEmbedding {
    /// Create a new speaker embedding extractor.
    pub fn new(input_dim: usize, embed_dim: usize, seed: u64) -> Self {
        Self {
            extractor: XVectorExtractor::new(input_dim, embed_dim, seed),
        }
    }

    /// Compute L2-normalized speaker embedding.
    pub fn embed(&self, x: &[f32]) -> Result<Vec<f32>> {
        let raw = self.extractor.forward(x)?;
        let norm = l2_norm(&raw).max(1e-8);
        Ok(raw.iter().map(|v| v / norm).collect())
    }
}

/// GE2E (Generalized End-to-End) speaker similarity loss.
///
/// For N speakers × M utterances, computes centroid-based cosine similarity
/// and cross-entropy over speakers.
#[derive(Debug, Clone)]
pub struct GeSim {
    /// Scale parameter (learnable, initialized to 10).
    pub w: f32,
    /// Bias parameter (learnable, initialized to -5).
    pub b: f32,
}

impl GeSim {
    /// Create a new GE2E similarity module with default parameters.
    pub fn new() -> Self {
        Self { w: 10.0, b: -5.0 }
    }

    /// Compute GE2E loss.
    ///
    /// `embeddings`: `[N*M, D]` flat (speaker major: first M rows = speaker 0, etc.)
    /// Returns scalar loss.
    pub fn loss(&self, embeddings: &[f32], n_speakers: usize, n_utterances: usize) -> Result<f32> {
        let nm = n_speakers * n_utterances;
        if embeddings.is_empty() || embeddings.len() % nm != 0 {
            return Err(inval(
                "GeSim::loss",
                format!(
                    "embeddings.len() {} not compatible with {n_speakers}×{n_utterances}",
                    embeddings.len()
                ),
            ));
        }
        let d = embeddings.len() / nm;
        if d == 0 {
            return Err(inval("GeSim::loss", "embedding dim is 0"));
        }
        let mut centroids = vec![0.0_f32; n_speakers * d];
        for spk in 0..n_speakers {
            for utt in 0..n_utterances {
                let emb =
                    &embeddings[(spk * n_utterances + utt) * d..(spk * n_utterances + utt + 1) * d];
                for j in 0..d {
                    centroids[spk * d + j] += emb[j];
                }
            }
            for j in 0..d {
                centroids[spk * d + j] /= n_utterances as f32;
            }
        }
        let mut total_loss = 0.0_f32;
        for spk in 0..n_speakers {
            for utt in 0..n_utterances {
                let emb =
                    &embeddings[(spk * n_utterances + utt) * d..(spk * n_utterances + utt + 1) * d];
                let emb_norm = l2_norm(emb).max(1e-8);
                let mut logits = vec![0.0_f32; n_speakers];
                for s in 0..n_speakers {
                    let c = &centroids[s * d..(s + 1) * d];
                    let c_norm = l2_norm(c).max(1e-8);
                    let cos_sim = dot(emb, c) / (emb_norm * c_norm);
                    logits[s] = self.w * cos_sim + self.b;
                }
                let lse = log_sum_exp(&logits);
                total_loss += lse - logits[spk];
            }
        }
        Ok(total_loss / nm as f32)
    }
}

impl Default for GeSim {
    fn default() -> Self {
        Self::new()
    }
}

/// Speaker verifier using cosine similarity.
#[derive(Debug, Clone)]
pub struct SpeakerVerifier {
    /// Cosine similarity threshold for acceptance.
    pub threshold: f64,
}

impl SpeakerVerifier {
    /// Create a new speaker verifier with given threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compute cosine similarity score between two L2-normalized embeddings.
    pub fn score(&self, emb1: &[f32], emb2: &[f32]) -> Result<f64> {
        if emb1.len() != emb2.len() || emb1.is_empty() {
            return Err(inval(
                "SpeakerVerifier::score",
                "embeddings must have equal non-zero length",
            ));
        }
        let n1 = l2_norm(emb1).max(1e-8);
        let n2 = l2_norm(emb2).max(1e-8);
        Ok((dot(emb1, emb2) / (n1 * n2)) as f64)
    }

    /// Verify if two embeddings belong to the same speaker.
    pub fn verify(&self, emb1: &[f32], emb2: &[f32]) -> Result<bool> {
        let s = self.score(emb1, emb2)?;
        Ok(s >= self.threshold)
    }
}

/// SpecAugment: time masking + frequency masking on mel spectrogram.
#[derive(Debug, Clone)]
pub struct SpecAugment {
    /// Maximum time mask length (T parameter).
    pub max_time_mask: usize,
    /// Maximum frequency mask length (F parameter).
    pub max_freq_mask: usize,
    /// Number of time masks to apply.
    pub num_time_masks: usize,
    /// Number of frequency masks to apply.
    pub num_freq_masks: usize,
}

impl SpecAugment {
    /// Create a new SpecAugment module.
    pub fn new(
        max_time_mask: usize,
        max_freq_mask: usize,
        num_time_masks: usize,
        num_freq_masks: usize,
    ) -> Self {
        Self {
            max_time_mask,
            max_freq_mask,
            num_time_masks,
            num_freq_masks,
        }
    }

    /// Apply SpecAugment to a `[T, F]` mel spectrogram (in-place clone).
    pub fn apply(
        &self,
        spec: &[f32],
        t_frames: usize,
        f_bins: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        if t_frames == 0 || f_bins == 0 || spec.len() != t_frames * f_bins {
            return Err(inval(
                "SpecAugment::apply",
                format!(
                    "spec.len() {} != t_frames*f_bins {}",
                    spec.len(),
                    t_frames * f_bins
                ),
            ));
        }
        let mut out = spec.to_vec();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..self.num_time_masks {
            if self.max_time_mask == 0 {
                break;
            }
            let max_len = self.max_time_mask.min(t_frames);
            let mask_len = if max_len == 0 {
                0
            } else {
                rng.random_range(1..max_len + 1)
            };
            if mask_len == 0 {
                continue;
            }
            let start = rng.random_range(0..t_frames - mask_len + 1);
            for ti in start..start + mask_len {
                for fi in 0..f_bins {
                    out[ti * f_bins + fi] = 0.0;
                }
            }
        }
        for _ in 0..self.num_freq_masks {
            if self.max_freq_mask == 0 {
                break;
            }
            let max_len = self.max_freq_mask.min(f_bins);
            let mask_len = if max_len == 0 {
                0
            } else {
                rng.random_range(1..max_len + 1)
            };
            if mask_len == 0 {
                continue;
            }
            let start = rng.random_range(0..f_bins - mask_len + 1);
            for ti in 0..t_frames {
                for fi in start..start + mask_len {
                    out[ti * f_bins + fi] = 0.0;
                }
            }
        }
        Ok(out)
    }
}

/// Add Gaussian noise at a specified SNR (dB).
#[derive(Debug, Clone)]
pub struct AddNoise;

impl AddNoise {
    /// Create a new AddNoise augmentation.
    pub fn new() -> Self {
        Self
    }

    /// Add noise to signal at given SNR in dB.
    pub fn add_noise_snr(&self, signal: &[f32], snr_db: f64, seed: u64) -> Vec<f32> {
        if signal.is_empty() {
            return Vec::new();
        }
        let signal_power: f64 =
            signal.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / signal.len() as f64;
        let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);
        let noise_std = noise_power.sqrt().max(0.0);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(signal.len());
        let mut spare: Option<f32> = None;
        for &s in signal.iter() {
            let noise = if let Some(n) = spare.take() {
                n
            } else {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                let r = (-2.0 * u1.ln()).sqrt() * noise_std;
                let theta = 2.0 * std::f64::consts::PI * u2;
                let z0 = (r * theta.cos()) as f32;
                let z1 = (r * theta.sin()) as f32;
                spare = Some(z1);
                z0
            };
            out.push(s + noise);
        }
        out
    }
}

impl Default for AddNoise {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple pitch shift via resampling (linear interpolation).
#[derive(Debug, Clone)]
pub struct PitchShift {
    /// Stretch factor: >1 shifts up, <1 shifts down.
    pub factor: f64,
}

impl PitchShift {
    /// Create a new pitch shift with given factor (must be > 0).
    pub fn new(factor: f64) -> Result<Self> {
        if factor <= 0.0 {
            return Err(inval("PitchShift::new", "factor must be > 0"));
        }
        Ok(Self { factor })
    }

    /// Apply pitch shift to waveform via resampling.
    /// Output length is approximately `signal.len() / factor`.
    pub fn apply(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let out_len = (n as f64 / self.factor).round() as usize;
        if out_len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 * self.factor;
            let idx = src.floor() as usize;
            let frac = (src - src.floor()) as f32;
            let v0 = signal[idx.min(n - 1)];
            let v1 = if idx + 1 < n { signal[idx + 1] } else { v0 };
            out.push(v0 + frac * (v1 - v0));
        }
        out
    }
}

/// WSOLA-style time stretching (nearest-neighbor approximation).
#[derive(Debug, Clone)]
pub struct TimeStretch {
    /// Time stretch factor: >1 stretches (slower), <1 compresses (faster).
    pub factor: f64,
}

impl TimeStretch {
    /// Create a new time stretch with given factor (must be > 0).
    pub fn new(factor: f64) -> Result<Self> {
        if factor <= 0.0 {
            return Err(inval("TimeStretch::new", "factor must be > 0"));
        }
        Ok(Self { factor })
    }

    /// Apply time stretching. Output length ≈ `signal.len() * factor`.
    pub fn apply(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let out_len = (n as f64 * self.factor).round() as usize;
        if out_len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 / self.factor;
            let idx = src.floor() as usize;
            let frac = (src - src.floor()) as f32;
            let v0 = signal[idx.min(n - 1)];
            let v1 = if idx + 1 < n { signal[idx + 1] } else { v0 };
            out.push(v0 + frac * (v1 - v0));
        }
        out
    }
}

/// Augmentation step with probability.
#[derive(Debug, Clone)]
pub enum AudioAugStep {
    /// Apply SpecAugment to a spectrogram.
    SpecAugmentStep {
        /// The SpecAugment configuration.
        aug: SpecAugment,
        /// Number of time frames in the spectrogram.
        t_frames: usize,
        /// Number of frequency bins.
        f_bins: usize,
    },
    /// Add Gaussian noise at a given SNR.
    AddNoiseStep {
        /// The noise augmentation.
        aug: AddNoise,
        /// Signal-to-noise ratio in dB.
        snr_db: f64,
    },
    /// Apply pitch shifting.
    PitchShiftStep {
        /// The pitch shift configuration.
        aug: PitchShift,
    },
    /// Apply time stretching.
    TimeStretchStep {
        /// The time stretch configuration.
        aug: TimeStretch,
    },
}

/// Composable audio augmentation pipeline.
#[derive(Debug, Clone)]
pub struct AudioAugmentPipeline {
    /// List of (augmentation step, application probability) pairs.
    pub steps: Vec<(AudioAugStep, f64)>,
}

impl AudioAugmentPipeline {
    /// Create an empty augmentation pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add an augmentation step with a given application probability.
    pub fn add_step(mut self, step: AudioAugStep, probability: f64) -> Self {
        self.steps.push((step, probability));
        self
    }

    /// Apply pipeline to a 1D waveform signal.
    pub fn apply_waveform(&self, signal: &[f32], seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = signal.to_vec();
        for (i, (step, prob)) in self.steps.iter().enumerate() {
            let r: f64 = rng.random();
            if r > *prob {
                continue;
            }
            let step_seed = seed.wrapping_add(i as u64 * 1000 + 42);
            match step {
                AudioAugStep::AddNoiseStep { aug, snr_db } => {
                    out = aug.add_noise_snr(&out, *snr_db, step_seed);
                }
                AudioAugStep::PitchShiftStep { aug } => {
                    out = aug.apply(&out);
                }
                AudioAugStep::TimeStretchStep { aug } => {
                    out = aug.apply(&out);
                }
                AudioAugStep::SpecAugmentStep { .. } => {}
            }
        }
        out
    }

    /// Apply pipeline to a spectrogram `[T, F]`.
    pub fn apply_spectrogram(
        &self,
        spec: &[f32],
        t_frames: usize,
        f_bins: usize,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = spec.to_vec();
        for (i, (step, prob)) in self.steps.iter().enumerate() {
            let r: f64 = rng.random();
            if r > *prob {
                continue;
            }
            let step_seed = seed.wrapping_add(i as u64 * 1000 + 99);
            if let AudioAugStep::SpecAugmentStep { aug, .. } = step {
                if let Ok(result) = aug.apply(&out, t_frames, f_bins, step_seed) {
                    out = result;
                }
            }
        }
        out
    }
}

impl Default for AudioAugmentPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod ext_tests {
    use super::*;
    use crate::audio_models::{
        ContextNetworkConfig, CtcDecoder, FastSpeech2Block, FastSpeechDuration, FeatureExtractor,
        FeatureExtractorConfig, GriffinLimVocoder, LengthRegulator, MelSpectrogram,
        QuantizerCodebook, TransformerContextNetwork,
    };

    #[test]
    fn test_feature_extractor_shape() {
        let cfg = FeatureExtractorConfig {
            channels: vec![8],
            kernel_sizes: vec![10],
            strides: vec![5],
        };
        let fe = FeatureExtractor::new(cfg, 42).expect("fe init");
        let waveform: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01).sin()).collect();
        let (features, t_prime) = fe.forward(&waveform).expect("fe forward");
        assert_eq!(features.len(), t_prime * fe.output_dim());
        assert!(t_prime > 0, "t_prime should be > 0, got {t_prime}");
        assert!(t_prime < 200, "t_prime should be less than input length");
    }

    #[test]
    fn test_quantizer_codebook_output() {
        let qc = QuantizerCodebook::new(64, 2, 16, 1.0, 42).expect("qc init");
        let z: Vec<f32> = (0..10 * 64).map(|i| (i as f32) * 0.01).collect();
        let out = qc.quantize(&z).expect("quantize");
        assert_eq!(out.indices.len(), 10 * 2, "indices shape: T*G");
        assert_eq!(out.quantized.len(), 10 * 64, "quantized shape: T*D");
    }

    #[test]
    fn test_diversity_loss_bounds() {
        let qc = QuantizerCodebook::new(64, 2, 16, 1.0, 42).expect("qc init");
        let z: Vec<f32> = (0..10 * 64).map(|i| (i as f32) * 0.01).collect();
        let out = qc.quantize(&z).expect("quantize");
        assert!(
            out.diversity_loss.is_finite(),
            "diversity_loss is not finite"
        );
    }

    #[test]
    fn test_contrastive_loss_positive() {
        use crate::audio_models::ContrastiveWav2VecLoss;
        let loss_fn = ContrastiveWav2VecLoss::new(5, 0.1);
        let t = 20;
        let d = 32;
        let context: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let quantized: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.1).sin() + 0.01).collect();
        let mask: Vec<bool> = (0..t).map(|i| i % 4 == 0).collect();
        let loss = loss_fn
            .compute(&context, &quantized, &mask, 42)
            .expect("loss compute");
        assert!(
            loss >= 0.0,
            "contrastive loss should be non-negative, got {loss}"
        );
        assert!(loss.is_finite(), "contrastive loss should be finite");
    }

    #[test]
    fn test_wav2vec2_construction() {
        let cfg = FeatureExtractorConfig {
            channels: vec![8],
            kernel_sizes: vec![3],
            strides: vec![1],
        };
        let feature_extractor = FeatureExtractor::new(cfg, 42).expect("fe init");
        let feat_dim = feature_extractor.output_dim(); // 8
        let quantizer = QuantizerCodebook::new(feat_dim, 2, 4, 1.0, 42).expect("qc init");
        let ctx_cfg = ContextNetworkConfig {
            hidden_dim: feat_dim,
            num_heads: 2,
            num_layers: 1,
            ffn_dim: 16,
        };
        let context_network = TransformerContextNetwork::new(ctx_cfg, 42);
        assert_eq!(feature_extractor.output_dim(), 8);
        assert_eq!(quantizer.num_groups, 2);
        assert_eq!(context_network.config.num_layers, 1);
    }

    #[test]
    fn test_conv_module_shape() {
        use crate::audio_models::ConvModule;
        let cm = ConvModule::new(64, 31, 42);
        let x: Vec<f32> = (0..10 * 64).map(|i| i as f32 * 0.01).collect();
        let out = cm.forward(&x);
        assert_eq!(out.len(), x.len(), "ConvModule should preserve shape");
    }

    #[test]
    fn test_conformer_block_residual() {
        use crate::audio_models::ConformerBlock;
        let block = ConformerBlock::new(32, 4, 128, 15, 42);
        let x: Vec<f32> = (0..8 * 32).map(|_| 0.1_f32).collect();
        let out = block.forward(&x);
        assert_eq!(out.len(), x.len(), "ConformerBlock should preserve shape");
        let diff: f32 = x.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "output should differ from input");
    }

    #[test]
    fn test_conformer_encoder_shape() {
        use crate::audio_models::ConformerEncoder;
        let enc = ConformerEncoder::new(32, 64, 2, 4, 128, 15, 42);
        let x: Vec<f32> = (0..8 * 32).map(|i| i as f32 * 0.001).collect();
        let out = enc.forward(&x).expect("conformer forward");
        assert_eq!(out.len(), 8 * 64, "Conformer output: T × output_dim");
    }

    #[test]
    fn test_ctc_greedy_decode_repeats() {
        let decoder = CtcDecoder::new(5);
        let mut logits = vec![0.0_f32; 6 * 5];
        for (ti, tok) in [1, 1, 2, 2, 3, 3].iter().enumerate() {
            logits[ti * 5 + tok] = 10.0;
        }
        let decoded = decoder.decode(&logits).expect("ctc decode");
        assert_eq!(
            decoded,
            vec![1, 2, 3],
            "repeats should be collapsed: {decoded:?}"
        );
    }

    #[test]
    fn test_ctc_decode_blank_removal() {
        let decoder = CtcDecoder::new(5);
        let mut logits = vec![0.0_f32; 5 * 5];
        for (ti, tok) in [0, 1, 0, 2, 0].iter().enumerate() {
            logits[ti * 5 + tok] = 10.0;
        }
        let decoded = decoder.decode(&logits).expect("ctc decode blanks");
        assert_eq!(decoded, vec![1, 2], "blanks should be removed: {decoded:?}");
    }

    #[test]
    fn test_duration_predictor_output() {
        let dp = FastSpeechDuration::new(32, 3, 42);
        let x: Vec<f32> = (0..8 * 32).map(|i| i as f32 * 0.01).collect();
        let durations = dp.predict_duration(&x).expect("duration predict");
        assert_eq!(durations.len(), 8, "should return T durations");
        for &d in &durations {
            assert!(d > 0.0, "softplus durations should be positive, got {d}");
        }
    }

    #[test]
    fn test_length_regulator_upsampling() {
        let lr = LengthRegulator::new();
        let d = 16;
        let x: Vec<f32> = (0..4 * d).map(|i| i as f32).collect();
        let durations = vec![2_usize, 3, 1, 4];
        let out = lr.regulate(&x, &durations, d).expect("regulate");
        let expected_t: usize = durations.iter().sum();
        assert_eq!(out.len(), expected_t * d, "upsampled length mismatch");
        for j in 0..d {
            assert_eq!(out[j], x[j], "frame 0 copy 0");
            assert_eq!(out[d + j], x[j], "frame 0 copy 1");
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 80;
        let n_fft = 512;
        let sr = 16000.0_f64;
        let mel = MelSpectrogram::new(n_mels, n_fft, sr).expect("mel init");
        let n_freqs = n_fft / 2 + 1;
        assert_eq!(
            mel.filterbank.len(),
            n_mels * n_freqs,
            "filterbank shape mismatch"
        );
        for m in 0..n_mels {
            let row_sum: f32 = mel.filterbank[m * n_freqs..(m + 1) * n_freqs].iter().sum();
            assert!(row_sum >= 0.0, "mel filter {m} sum negative: {row_sum}");
        }
    }

    #[test]
    fn test_griffin_lim_shape() {
        let vocoder = GriffinLimVocoder::new(256, 128);
        let n_freqs = 256 / 2 + 1;
        let t_frames = 10;
        let magnitude: Vec<f32> = vec![1.0_f32; t_frames * n_freqs];
        let waveform = vocoder.iterate(&magnitude, 3, 42).expect("griffin-lim");
        let expected_len = (t_frames - 1) * 128 + 256;
        assert_eq!(waveform.len(), expected_len, "waveform length mismatch");
    }

    #[test]
    fn test_fastspeech2_block() {
        let block = FastSpeech2Block::new(32, 4, 128, 3, 42);
        let x: Vec<f32> = (0..8 * 32).map(|i| i as f32 * 0.01).collect();
        let out = block.forward(&x);
        assert_eq!(
            out.len(),
            x.len(),
            "FastSpeech2Block should preserve [T, D] shape"
        );
    }

    #[test]
    fn test_tdnn_layer_forward() {
        let layer = TdnnLayer::new(4, 8, 5, 1, 42);
        let x: Vec<f32> = (0..20 * 4).map(|i| i as f32 * 0.01).collect();
        let out = layer.forward(&x).expect("tdnn forward");
        let span = (5 - 1) + 1;
        let t_out = 20 - span + 1;
        assert_eq!(
            out.len(),
            t_out * 8,
            "TDNN output shape mismatch: got {}, expected {}",
            out.len(),
            t_out * 8
        );
    }

    #[test]
    fn test_xvector_stats_pooling() {
        let extractor = XVectorExtractor::with_hidden(4, 16, 8, 42);
        let x: Vec<f32> = (0..60 * 4).map(|i| (i as f32 * 0.05).sin()).collect();
        let emb = extractor.forward(&x).expect("xvector forward");
        assert_eq!(emb.len(), 16, "x-vector embedding size mismatch");
    }

    #[test]
    fn test_speaker_embedding_normalized() {
        let se = SpeakerEmbedding {
            extractor: XVectorExtractor::with_hidden(4, 16, 8, 42),
        };
        let x: Vec<f32> = (0..60 * 4).map(|i| (i as f32 * 0.05).cos()).collect();
        let emb = se.embed(&x).expect("embed");
        assert_eq!(emb.len(), 16);
        let norm = l2_norm(&emb);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "embedding should be L2-normalized, norm={norm}"
        );
    }

    #[test]
    fn test_ge2e_loss_positive() {
        let ge2e = GeSim::new();
        let n_spk = 3;
        let n_utt = 4;
        let d = 32;
        let embeddings: Vec<f32> = (0..n_spk * n_utt * d)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let loss = ge2e.loss(&embeddings, n_spk, n_utt).expect("ge2e loss");
        assert!(loss.is_finite(), "GE2E loss should be finite, got {loss}");
        assert!(loss >= 0.0, "GE2E loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_speaker_verifier_score_range() {
        let verifier = SpeakerVerifier::new(0.7);
        let d = 32;
        let emb1: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
        let mut emb2 = emb1.clone();
        emb2[0] += 0.1;
        let score = verifier.score(&emb1, &emb2).expect("score");
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&score),
            "cosine score out of range: {score}"
        );
        let score_same = verifier.score(&emb1, &emb1).expect("score same");
        assert!(
            (score_same - 1.0).abs() < 1e-4,
            "same embedding score should be ~1.0, got {score_same}"
        );
    }

    #[test]
    fn test_spec_augment_time_mask() {
        let aug = SpecAugment::new(5, 0, 1, 0);
        let t = 20;
        let f = 40;
        let spec: Vec<f32> = vec![1.0_f32; t * f];
        let out = aug.apply(&spec, t, f, 42).expect("spec augment time");
        assert_eq!(out.len(), t * f);
        let zeros: usize = out.iter().filter(|&&x| x == 0.0).count();
        assert!(
            zeros > 0,
            "time masking should zero some elements, got {zeros} zeros"
        );
    }

    #[test]
    fn test_spec_augment_freq_mask() {
        let aug = SpecAugment::new(0, 5, 0, 1);
        let t = 20;
        let f = 40;
        let spec: Vec<f32> = vec![1.0_f32; t * f];
        let out = aug.apply(&spec, t, f, 42).expect("spec augment freq");
        assert_eq!(out.len(), t * f);
        let zeros: usize = out.iter().filter(|&&x| x == 0.0).count();
        assert!(
            zeros > 0,
            "freq masking should zero some elements, got {zeros} zeros"
        );
    }

    #[test]
    fn test_add_noise_snr() {
        let aug = AddNoise::new();
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let noisy = aug.add_noise_snr(&signal, 20.0, 42);
        assert_eq!(noisy.len(), signal.len(), "noisy signal length mismatch");
        let diff: f32 = signal
            .iter()
            .zip(noisy.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "noisy signal should differ from original");
    }

    #[test]
    fn test_pitch_shift() {
        let ps = PitchShift::new(1.2).expect("pitch shift init");
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.02).sin()).collect();
        let shifted = ps.apply(&signal);
        let expected_len = (1000_f64 / 1.2).round() as usize;
        assert_eq!(
            shifted.len(),
            expected_len,
            "pitch shift output length mismatch"
        );
    }

    #[test]
    fn test_augment_pipeline() {
        let noise_step = AudioAugStep::AddNoiseStep {
            aug: AddNoise::new(),
            snr_db: 20.0,
        };
        let pitch_step = AudioAugStep::PitchShiftStep {
            aug: PitchShift::new(1.05).expect("pitch init"),
        };
        let pipeline = AudioAugmentPipeline::new()
            .add_step(noise_step, 1.0)
            .add_step(pitch_step, 1.0);
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = pipeline.apply_waveform(&signal, 42);
        assert!(!out.is_empty(), "pipeline output should be non-empty");
        let original_len = signal.len();
        assert!(
            out.len() < original_len,
            "pitch shift should reduce length slightly"
        );
    }

    #[test]
    fn test_time_stretch() {
        let ts = TimeStretch::new(1.3).expect("time stretch init");
        let signal: Vec<f32> = (0..500).map(|i| (i as f32 * 0.05).cos()).collect();
        let stretched = ts.apply(&signal);
        let expected_len = (500_f64 * 1.3).round() as usize;
        assert_eq!(
            stretched.len(),
            expected_len,
            "time stretch output length mismatch"
        );
    }

    #[test]
    fn test_feature_extractor_gelu_activation() {
        let cfg = FeatureExtractorConfig {
            channels: vec![8],
            kernel_sizes: vec![3],
            strides: vec![1],
        };
        let fe = FeatureExtractor::new(cfg, 42).expect("fe init");
        let wav: Vec<f32> = (0..20).map(|i| i as f32 * 0.01).collect();
        let (feats, t) = fe.forward(&wav).expect("fe forward");
        assert_eq!(feats.len(), t * 8);
    }

    #[test]
    fn test_mel_stft_to_mel() {
        let mel = MelSpectrogram::new(40, 256, 16000.0).expect("mel init");
        let n_freqs = 256 / 2 + 1;
        let t = 5;
        let spec: Vec<f32> = vec![0.5_f32; t * n_freqs];
        let mel_out = mel.stft_to_mel(&spec).expect("stft_to_mel");
        assert_eq!(mel_out.len(), t * 40, "mel output shape");
        for &v in &mel_out {
            assert!(v.is_finite(), "log mel should be finite");
        }
    }

    #[test]
    fn test_ctc_decoder_all_blank() {
        let decoder = CtcDecoder::new(4);
        let logits = vec![
            10.0_f32, 0.0, 0.0, 0.0, // blank
            10.0, 0.0, 0.0, 0.0, // blank
            10.0, 0.0, 0.0, 0.0, // blank
        ];
        let decoded = decoder.decode(&logits).expect("ctc all blank");
        assert!(
            decoded.is_empty(),
            "all-blank sequence should decode to empty: {decoded:?}"
        );
    }

    #[test]
    fn test_length_regulator_zero_duration() {
        let lr = LengthRegulator::new();
        let d = 4;
        let x: Vec<f32> = (0..3 * d).map(|i| i as f32).collect();
        let durations = vec![0_usize, 2, 0];
        let out = lr.regulate(&x, &durations, d).expect("regulate zero dur");
        assert_eq!(
            out.len(),
            2 * d,
            "zero-duration frames should produce no output"
        );
    }

    #[test]
    fn test_speaker_verifier_threshold() {
        let verifier_high = SpeakerVerifier::new(0.99);
        let verifier_low = SpeakerVerifier::new(0.0);
        let d = 16;
        let emb1: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
        let emb2: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0).sin()).collect();
        let _ = verifier_high.verify(&emb1, &emb2).expect("verify high");
        let result_low = verifier_low.verify(&emb1, &emb2).expect("verify low");
        assert!(
            result_low,
            "low threshold verifier should accept most pairs"
        );
    }

    #[test]
    fn test_conformer_context_network_forward() {
        let cfg = ContextNetworkConfig {
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 1,
            ffn_dim: 32,
        };
        let net = TransformerContextNetwork::new(cfg, 42);
        let x: Vec<f32> = (0..4 * 16).map(|i| i as f32 * 0.01).collect();
        let out = net.forward(&x).expect("context net forward");
        assert_eq!(
            out.len(),
            4 * 16,
            "context network should preserve [T, D] shape"
        );
    }

    #[test]
    fn test_gesim_default() {
        let ge = GeSim::default();
        assert_eq!(ge.w, 10.0);
        assert_eq!(ge.b, -5.0);
    }

    #[test]
    fn test_length_regulator_default() {
        let lr = LengthRegulator;
        let d = 2;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = lr.regulate(&x, &[1, 1], d).expect("default regulate");
        assert_eq!(out, x);
    }

    #[test]
    fn test_add_noise_snr_zero_signal() {
        let aug = AddNoise;
        let out = aug.add_noise_snr(&[], 20.0, 42);
        assert!(out.is_empty(), "empty signal should return empty");
    }
}
