//! Affective Computing and Emotion Recognition.
//!
//! Implements a full emotion recognition pipeline covering facial Action Units,
//! EEG signals, speech acoustics, multi-modal fusion, continuous tracking, and
//! evaluation metrics.  All computation is in pure Rust using `Vec<Vec<f32>>`.
//!
//! ## Components
//! - [`BasicEmotion`] / [`ValenceArousal`] / [`EmotionLabel`] — taxonomy
//! - [`FacialAuEncoder`] / [`AuFacsRules`] — facial-AU→emotion mapping
//! - [`EegEmotionNet`] — EEG band-power feature extraction + MLP
//! - [`SpeechEmotionRecognizer`] — MFCC / pitch / energy → valence-arousal
//! - [`MultiModalEmotionFuser`] — average / weighted / attention fusion
//! - \[`ContinuousEmotionTracker`\] / [`KalmanEmotionFilter`] / [`EmotionTransitionModel`]
//! - [`EmotionMetrics`] — accuracy, confusion matrix, F1-macro, RMSE, CCC

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §1  Emotion taxonomy
// ─────────────────────────────────────────────────────────────────────────────

/// The seven basic emotions (Ekman, 1971).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasicEmotion {
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Neutral,
}

impl BasicEmotion {
    /// Integer index (0-based) used for one-hot / argmax operations.
    pub fn index(self) -> usize {
        match self {
            BasicEmotion::Happy => 0,
            BasicEmotion::Sad => 1,
            BasicEmotion::Angry => 2,
            BasicEmotion::Fearful => 3,
            BasicEmotion::Disgusted => 4,
            BasicEmotion::Surprised => 5,
            BasicEmotion::Neutral => 6,
        }
    }

    /// Construct from index; returns `Neutral` for out-of-range indices.
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => BasicEmotion::Happy,
            1 => BasicEmotion::Sad,
            2 => BasicEmotion::Angry,
            3 => BasicEmotion::Fearful,
            4 => BasicEmotion::Disgusted,
            5 => BasicEmotion::Surprised,
            _ => BasicEmotion::Neutral,
        }
    }

    /// All seven variants, in index order.
    pub fn all() -> [BasicEmotion; 7] {
        [
            BasicEmotion::Happy,
            BasicEmotion::Sad,
            BasicEmotion::Angry,
            BasicEmotion::Fearful,
            BasicEmotion::Disgusted,
            BasicEmotion::Surprised,
            BasicEmotion::Neutral,
        ]
    }
}

/// Russell's circumplex model coordinates: valence and arousal in `[-1, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValenceArousal {
    pub valence: f32,
    pub arousal: f32,
}

impl ValenceArousal {
    pub fn new(valence: f32, arousal: f32) -> Self {
        Self { valence, arousal }
    }
}

/// Unified emotion label that can be either discrete or dimensional.
#[derive(Debug, Clone, PartialEq)]
pub enum EmotionLabel {
    Discrete(BasicEmotion),
    Dimensional(ValenceArousal),
}

/// Map a valence-arousal coordinate to a discrete emotion using quadrant rules.
///
/// Quadrant logic (Russell, 1980):
/// * Q1 (V+, A+) — Happy / Surprised
/// * Q2 (V-, A+) — Angry / Fearful
/// * Q3 (V-, A-) — Sad / Disgusted
/// * Q4 (V+, A-) — Neutral (calm, content)
///
/// Near-zero magnitude → Neutral.
pub fn va_to_discrete(va: ValenceArousal) -> BasicEmotion {
    let v = va.valence;
    let a = va.arousal;
    let mag = (v * v + a * a).sqrt();
    if mag < 0.15 {
        return BasicEmotion::Neutral;
    }
    match (v >= 0.0, a >= 0.0) {
        (true, true) => {
            // Happy vs Surprised: high arousal → surprised
            if a > 0.5 {
                BasicEmotion::Surprised
            } else {
                BasicEmotion::Happy
            }
        }
        (false, true) => {
            // Angry vs Fearful: very negative valence → angry
            if v < -0.4 {
                BasicEmotion::Angry
            } else {
                BasicEmotion::Fearful
            }
        }
        (false, false) => {
            // Sad vs Disgusted: strongly negative valence → disgusted
            if v < -0.5 {
                BasicEmotion::Disgusted
            } else {
                BasicEmotion::Sad
            }
        }
        (true, false) => BasicEmotion::Neutral,
    }
}

/// Canonical VA coordinates for each basic emotion (Warriner et al.).
pub fn discrete_to_va(e: BasicEmotion) -> ValenceArousal {
    match e {
        BasicEmotion::Happy => ValenceArousal::new(0.76, 0.48),
        BasicEmotion::Sad => ValenceArousal::new(-0.63, -0.27),
        BasicEmotion::Angry => ValenceArousal::new(-0.65, 0.60),
        BasicEmotion::Fearful => ValenceArousal::new(-0.52, 0.60),
        BasicEmotion::Disgusted => ValenceArousal::new(-0.60, 0.35),
        BasicEmotion::Surprised => ValenceArousal::new(0.40, 0.67),
        BasicEmotion::Neutral => ValenceArousal::new(0.0, 0.0),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Facial Action Unit encoder
// ─────────────────────────────────────────────────────────────────────────────

/// FACS Action Units relevant to basic emotion recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionUnit {
    /// Inner brow raise.
    Au1,
    /// Outer brow raise.
    Au2,
    /// Brow lowerer.
    Au4,
    /// Cheek raiser.
    Au6,
    /// Lid tightener.
    Au7,
    /// Lip corner puller (smile).
    Au12,
    /// Lip corner depressor.
    Au15,
    /// Chin raiser.
    Au17,
    /// Lip tightener.
    Au23,
    /// Lips part.
    Au25,
    /// Jaw drop.
    Au26,
}

impl ActionUnit {
    /// Integer index for weight-matrix lookups.
    pub fn index(self) -> usize {
        match self {
            ActionUnit::Au1 => 0,
            ActionUnit::Au2 => 1,
            ActionUnit::Au4 => 2,
            ActionUnit::Au6 => 3,
            ActionUnit::Au7 => 4,
            ActionUnit::Au12 => 5,
            ActionUnit::Au15 => 6,
            ActionUnit::Au17 => 7,
            ActionUnit::Au23 => 8,
            ActionUnit::Au25 => 9,
            ActionUnit::Au26 => 10,
        }
    }
}

/// A set of AU intensity values (0–5 FACS scale).
#[derive(Debug, Clone)]
pub struct AuProfile {
    /// Pairs of (ActionUnit, intensity ∈ [0, 5]).
    pub au_intensities: Vec<(ActionUnit, f32)>,
}

impl AuProfile {
    pub fn new(au_intensities: Vec<(ActionUnit, f32)>) -> Self {
        Self { au_intensities }
    }

    /// Look up the intensity for a given AU (returns 0.0 if absent).
    pub fn get(&self, au: ActionUnit) -> f32 {
        self.au_intensities
            .iter()
            .find(|(a, _)| *a == au)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }

    /// Convert to a dense feature vector of length `n` (AU index → intensity).
    pub fn to_dense(&self, n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n];
        for (au, intensity) in &self.au_intensities {
            let idx = au.index();
            if idx < n {
                out[idx] = *intensity;
            }
        }
        out
    }
}

/// Linear encoder: AU intensities → emotion logits (length 7).
#[derive(Debug, Clone)]
pub struct FacialAuEncoder {
    /// Weight matrix: shape [n_emotions × n_aus].
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: length n_emotions.
    pub bias: Vec<f32>,
}

impl FacialAuEncoder {
    /// Construct with Xavier-uniform initialisation.
    pub fn new(n_aus: usize, n_emotions: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0_f32 / (n_aus + n_emotions) as f32).sqrt();
        let weights = (0..n_emotions)
            .map(|_| {
                (0..n_aus)
                    .map(|_| rng.random_range(-limit..limit))
                    .collect::<Vec<_>>()
            })
            .collect();
        let bias = vec![0.0_f32; n_emotions];
        Self { weights, bias }
    }

    /// Forward pass: AuProfile → raw logits (length == n_emotions).
    pub fn encode(&self, profile: &AuProfile) -> Vec<f32> {
        let n_aus = self.weights.first().map(|r| r.len()).unwrap_or(0);
        let feats = profile.to_dense(n_aus);
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, b)| {
                let dot: f32 = row.iter().zip(feats.iter()).map(|(w, x)| w * x).sum();
                dot + b
            })
            .collect()
    }

    /// Return the `BasicEmotion` with the highest logit.
    pub fn predict(&self, profile: &AuProfile) -> BasicEmotion {
        let logits = self.encode(profile);
        argmax_emotion(&logits)
    }
}

/// Rule-based FACS → emotion classifier (Ekman's FACS coding manual).
pub struct AuFacsRules;

impl AuFacsRules {
    /// Classify emotion using Ekman's prototypical AU combinations.
    ///
    /// Priority order (most distinctive AU checked first).
    pub fn facs_classify(profile: &AuProfile) -> BasicEmotion {
        let au12 = profile.get(ActionUnit::Au12);
        let au6 = profile.get(ActionUnit::Au6);
        let au15 = profile.get(ActionUnit::Au15);
        let au17 = profile.get(ActionUnit::Au17);
        let au4 = profile.get(ActionUnit::Au4);
        let au7 = profile.get(ActionUnit::Au7);
        let au23 = profile.get(ActionUnit::Au23);
        let au1 = profile.get(ActionUnit::Au1);
        let au2 = profile.get(ActionUnit::Au2);
        let au25 = profile.get(ActionUnit::Au25);
        let au26 = profile.get(ActionUnit::Au26);

        // Happy: AU12 + AU6
        if au12 >= 1.5 && au6 >= 1.0 {
            return BasicEmotion::Happy;
        }
        // Angry: AU4 + AU7 + AU23
        if au4 >= 1.5 && au7 >= 1.0 && au23 >= 1.0 {
            return BasicEmotion::Angry;
        }
        // Disgusted: AU9 (mapped to AU4 proxy) + AU17 — use AU4+AU17 combo
        if au4 >= 1.0 && au17 >= 1.5 {
            return BasicEmotion::Disgusted;
        }
        // Fearful: AU1+AU2+AU4+AU7+AU25+AU26
        if au1 >= 1.0 && au2 >= 1.0 && au4 >= 1.0 && au25 >= 1.0 {
            return BasicEmotion::Fearful;
        }
        // Sad: AU1+AU15+AU17
        if au15 >= 1.5 && au17 >= 1.0 {
            return BasicEmotion::Sad;
        }
        // Surprised: AU1+AU2+AU25+AU26
        if au1 >= 1.0 && au2 >= 1.0 && au26 >= 1.5 {
            return BasicEmotion::Surprised;
        }
        // Lonely happy (AU12 only)
        if au12 >= 2.0 {
            return BasicEmotion::Happy;
        }
        // Lonely sad (AU15 only)
        if au15 >= 2.0 {
            return BasicEmotion::Sad;
        }
        BasicEmotion::Neutral
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  EEG emotion network
// ─────────────────────────────────────────────────────────────────────────────

/// Standard EEG frequency bands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EegBand {
    /// 0.5 – 4 Hz
    Delta,
    /// 4 – 8 Hz
    Theta,
    /// 8 – 13 Hz
    Alpha,
    /// 13 – 30 Hz
    Beta,
    /// 30 – 100 Hz
    Gamma,
}

impl EegBand {
    /// (low_hz, high_hz) frequency limits.
    pub fn limits(self) -> (f32, f32) {
        match self {
            EegBand::Delta => (0.5, 4.0),
            EegBand::Theta => (4.0, 8.0),
            EegBand::Alpha => (8.0, 13.0),
            EegBand::Beta => (13.0, 30.0),
            EegBand::Gamma => (30.0, 100.0),
        }
    }

    /// All five bands in order.
    pub fn all() -> [EegBand; 5] {
        [
            EegBand::Delta,
            EegBand::Theta,
            EegBand::Alpha,
            EegBand::Beta,
            EegBand::Gamma,
        ]
    }
}

/// Feature container for one EEG epoch.
#[derive(Debug, Clone)]
pub struct EegFeatures {
    /// Band power per (band × channel): length = n_bands * n_channels.
    pub band_power: Vec<f32>,
    /// Differential entropy per (band × channel): length = n_bands * n_channels.
    pub differential_entropy: Vec<f32>,
    /// Frontal asymmetry (left − right) per band: length = n_bands.
    pub asymmetry: Vec<f32>,
}

/// Compute power in a given frequency band using a simple rectangular window DFT.
///
/// Sums the squared DFT magnitudes whose bin frequency falls within `[band.low, band.high)`.
pub fn extract_band_power(signal: &[f32], sample_rate: f32, band: EegBand) -> f32 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let (low, high) = band.limits();
    let freq_res = sample_rate / n as f32;
    let mut power = 0.0_f32;
    // Compute real-valued DFT squared magnitudes for bins in [low, high).
    for k in 0..=(n / 2) {
        let freq = k as f32 * freq_res;
        if freq < low || freq >= high {
            continue;
        }
        // DFT bin k: X_k = Σ x_n * exp(-2πi kn/N)
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        let two_pi_k_over_n = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
        for (t, &x) in signal.iter().enumerate() {
            let angle = two_pi_k_over_n * t as f32;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        let mag_sq = (re * re + im * im) / n as f32;
        // Double-count inner bins (conjugate symmetry).
        let weight = if k == 0 || (n % 2 == 0 && k == n / 2) {
            1.0
        } else {
            2.0
        };
        power += weight * mag_sq;
    }
    power
}

/// Estimate differential entropy: `0.5 * ln(2πe * σ²)`.
pub fn differential_entropy(signal: &[f32]) -> f32 {
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f32 = signal.iter().sum::<f32>() / n as f32;
    let var: f32 = signal.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    let var_pos = var.max(1e-12);
    0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E * var_pos).ln()
}

/// Extract a full feature set from `n_channels` EEG signals.
///
/// Returns an `EegFeatures` with:
/// - `band_power` length = 5 * n_channels
/// - `differential_entropy` length = 5 * n_channels
/// - `asymmetry` length = 5 (one per band, left = channel 0, right = last channel)
pub fn extract_features(signals: &[Vec<f32>], sample_rate: f32) -> EegFeatures {
    let n_ch = signals.len();
    let bands = EegBand::all();
    let n_bands = bands.len();

    let mut band_power = Vec::with_capacity(n_bands * n_ch);
    let mut de_feats = Vec::with_capacity(n_bands * n_ch);
    let mut asymmetry = Vec::with_capacity(n_bands);

    for &band in &bands {
        let mut bp_row = Vec::with_capacity(n_ch);
        let mut de_row = Vec::with_capacity(n_ch);
        for sig in signals {
            let bp = extract_band_power(sig, sample_rate, band);
            let de = differential_entropy(sig);
            bp_row.push(bp);
            de_row.push(de);
        }
        // Frontal asymmetry: left (ch 0) minus right (last ch).
        let left_bp = bp_row.first().copied().unwrap_or(0.0);
        let right_bp = bp_row.last().copied().unwrap_or(0.0);
        asymmetry.push(left_bp - right_bp);
        band_power.extend(bp_row);
        de_feats.extend(de_row);
    }
    EegFeatures {
        band_power,
        differential_entropy: de_feats,
        asymmetry,
    }
}

/// Multi-layer perceptron for EEG-based emotion classification.
#[derive(Debug, Clone)]
pub struct EegEmotionNet {
    /// Each layer is (weight_matrix [out × in], bias \[out\]).
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    pub n_features: usize,
}

impl EegEmotionNet {
    /// Build a two-hidden-layer MLP: `n_features → hidden → hidden/2 → n_emotions`.
    pub fn new(n_features: usize, hidden: usize, n_emotions: usize, rng: &mut impl Rng) -> Self {
        let arch = [n_features, hidden, hidden / 2 + 1, n_emotions];
        let layers = arch
            .windows(2)
            .map(|w| {
                let (inp, out) = (w[0], w[1]);
                let limit = (2.0_f32 / inp as f32).sqrt(); // He init
                let weights: Vec<Vec<f32>> = (0..out)
                    .map(|_| (0..inp).map(|_| rng.random_range(-limit..limit)).collect())
                    .collect();
                let bias = vec![0.0_f32; out];
                (weights, bias)
            })
            .collect();
        Self { layers, n_features }
    }

    /// Forward pass: EegFeatures → softmax probability vector (length n_emotions).
    pub fn forward(&self, features: &EegFeatures) -> Vec<f32> {
        // Concatenate all feature sub-vectors.
        let mut x: Vec<f32> = features
            .band_power
            .iter()
            .chain(features.differential_entropy.iter())
            .chain(features.asymmetry.iter())
            .copied()
            .collect();
        // Truncate or zero-pad to n_features.
        x.resize(self.n_features, 0.0);

        for (i, (weights, bias)) in self.layers.iter().enumerate() {
            let is_last = i == self.layers.len() - 1;
            x = mlp_layer_forward(&x, weights, bias, !is_last);
        }
        softmax_vec(&x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Speech Emotion Recognizer
// ─────────────────────────────────────────────────────────────────────────────

/// Acoustic features extracted from a speech signal.
#[derive(Debug, Clone)]
pub struct AcousticFeatures {
    /// 13 MFCC coefficients.
    pub mfcc: Vec<f32>,
    /// Pitch statistics: [mean, std, min, max].
    pub pitch_stats: Vec<f32>,
    /// Energy statistics: [mean, std, rms].
    pub energy_stats: Vec<f32>,
    /// Zero crossing rate.
    pub zcr: f32,
}

/// Simple DCT-based MFCC extraction (no pre-emphasis, triangular filter bank).
///
/// Steps:
/// 1. Frame into a single window (or use the whole signal).
/// 2. Compute squared DFT magnitudes → log filterbank energies.
/// 3. Apply DCT-II to get cepstral coefficients.
pub fn extract_mfcc(signal: &[f32], sample_rate: f32, n_mfcc: usize) -> Vec<f32> {
    if signal.is_empty() {
        return vec![0.0; n_mfcc];
    }
    let n = signal.len();
    let n_fft = n.next_power_of_two();
    let n_filters = 26_usize;

    // Windowed power spectrum (Hann window).
    let mut power_spec: Vec<f32> = (0..=n_fft / 2)
        .map(|k| {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let two_pi_k = 2.0 * std::f32::consts::PI * k as f32 / n_fft as f32;
            for (t, &x) in signal.iter().enumerate().take(n_fft) {
                let hann = 0.5
                    * (1.0 - (2.0 * std::f32::consts::PI * t as f32 / (n - 1).max(1) as f32).cos());
                let windowed = x * hann;
                let angle = two_pi_k * t as f32;
                re += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            (re * re + im * im) / n_fft as f32
        })
        .collect();

    // Mel filterbank (triangular filters evenly spaced on mel scale).
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);
    let mel_points: Vec<f32> = (0..=n_filters + 1)
        .map(|i| mel_to_hz(mel_low + (mel_high - mel_low) * i as f32 / (n_filters + 1) as f32))
        .collect();

    let bin_of_freq = |hz: f32| -> usize {
        let bin = (hz / (sample_rate / n_fft as f32)).round() as usize;
        bin.min(n_fft / 2)
    };

    let mut filter_energies = vec![0.0_f32; n_filters];
    for (m, fe) in filter_energies.iter_mut().enumerate() {
        let f_start = mel_points[m];
        let f_mid = mel_points[m + 1];
        let f_end = mel_points[m + 2];
        let b_start = bin_of_freq(f_start);
        let b_mid = bin_of_freq(f_mid);
        let b_end = bin_of_freq(f_end);
        let mut energy = 0.0_f32;
        for b in b_start..=b_end {
            if b >= power_spec.len() {
                break;
            }
            let hz = b as f32 * sample_rate / n_fft as f32;
            let weight = if hz <= f_mid && (f_mid - f_start).abs() > 1e-6 {
                (hz - f_start) / (f_mid - f_start)
            } else if hz > f_mid && (f_end - f_mid).abs() > 1e-6 {
                (f_end - hz) / (f_end - f_mid)
            } else {
                0.0
            };
            energy += weight * power_spec[b];
        }
        *fe = (energy.max(1e-12)).ln();
    }

    // DCT-II to get cepstral coefficients.
    let n_coeff = n_mfcc.min(n_filters);
    let mut mfcc = Vec::with_capacity(n_mfcc);
    for k in 0..n_coeff {
        let val: f32 = filter_energies
            .iter()
            .enumerate()
            .map(|(n, &e)| {
                e * (std::f32::consts::PI * k as f32 * (2 * n + 1) as f32 / (2 * n_filters) as f32)
                    .cos()
            })
            .sum::<f32>()
            * (2.0 / n_filters as f32).sqrt();
        mfcc.push(val);
    }
    // Pad if fewer coefficients available.
    while mfcc.len() < n_mfcc {
        mfcc.push(0.0);
    }
    mfcc
}

#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Autocorrelation-based fundamental frequency (F0) estimation.
///
/// Computes the normalised autocorrelation and picks the peak in the lag range
/// corresponding to 60 – 500 Hz. Returns a pitch track (one value per frame).
pub fn extract_pitch(signal: &[f32], sample_rate: f32) -> Vec<f32> {
    if signal.len() < 2 {
        return vec![0.0];
    }
    let min_period = (sample_rate / 500.0).round() as usize;
    let max_period = (sample_rate / 60.0).round() as usize;

    let frame_size = 512_usize;
    let hop_size = 256_usize;
    let mut pitch_track = Vec::new();

    let mut start = 0;
    while start + frame_size <= signal.len() {
        let frame = &signal[start..start + frame_size];
        let energy: f32 = frame.iter().map(|x| x * x).sum();
        if energy < 1e-10 {
            pitch_track.push(0.0);
            start += hop_size;
            continue;
        }
        // Normalised autocorrelation.
        let mut best_lag = min_period;
        let mut best_corr = f32::NEG_INFINITY;
        let n = frame.len();
        for lag in min_period..=max_period.min(n / 2) {
            let mut corr = 0.0_f32;
            let mut denom = 0.0_f32;
            for t in 0..n - lag {
                corr += frame[t] * frame[t + lag];
                denom += frame[t] * frame[t];
            }
            let norm_corr = if denom > 1e-12 { corr / denom } else { 0.0 };
            if norm_corr > best_corr {
                best_corr = norm_corr;
                best_lag = lag;
            }
        }
        let f0 = if best_corr > 0.3 {
            sample_rate / best_lag as f32
        } else {
            0.0
        };
        pitch_track.push(f0);
        start += hop_size;
    }
    if pitch_track.is_empty() {
        pitch_track.push(0.0);
    }
    pitch_track
}

/// Extract all acoustic features from a raw speech signal.
pub fn extract_acoustic_features(signal: &[f32], sample_rate: f32) -> AcousticFeatures {
    let mfcc = extract_mfcc(signal, sample_rate, 13);
    let pitch = extract_pitch(signal, sample_rate);

    // Pitch statistics.
    let voiced: Vec<f32> = pitch.iter().copied().filter(|&x| x > 0.0).collect();
    let pitch_stats = if voiced.is_empty() {
        vec![0.0_f32; 4]
    } else {
        let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
        let std =
            (voiced.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / voiced.len() as f32).sqrt();
        let min = voiced.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = voiced.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        vec![mean, std, min, max]
    };

    // Energy statistics.
    let n = signal.len().max(1);
    let energy_per_sample: Vec<f32> = signal.iter().map(|x| x * x).collect();
    let energy_mean = energy_per_sample.iter().sum::<f32>() / n as f32;
    let energy_std = (energy_per_sample
        .iter()
        .map(|e| (e - energy_mean).powi(2))
        .sum::<f32>()
        / n as f32)
        .sqrt();
    let rms = energy_mean.sqrt();
    let energy_stats = vec![energy_mean, energy_std, rms];

    // Zero crossing rate.
    let crossings = signal.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    let zcr = crossings as f32 / (n - 1).max(1) as f32;

    AcousticFeatures {
        mfcc,
        pitch_stats,
        energy_stats,
        zcr,
    }
}

/// MLP-based Speech Emotion Recognizer that outputs a `ValenceArousal` pair.
#[derive(Debug, Clone)]
pub struct SpeechEmotionRecognizer {
    /// Hidden layers (weight [out × in], bias \[out\]).
    pub encoder: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl SpeechEmotionRecognizer {
    /// Build: `n_features → hidden → hidden/2 → 2` (valence, arousal).
    pub fn new(n_features: usize, hidden: usize, rng: &mut impl Rng) -> Self {
        let arch = [n_features, hidden, hidden / 2 + 1, 2];
        let encoder = arch
            .windows(2)
            .map(|w| {
                let (inp, out) = (w[0], w[1]);
                let limit = (6.0_f32 / (inp + out) as f32).sqrt();
                let weights: Vec<Vec<f32>> = (0..out)
                    .map(|_| (0..inp).map(|_| rng.random_range(-limit..limit)).collect())
                    .collect();
                let bias = vec![0.0_f32; out];
                (weights, bias)
            })
            .collect();
        Self { encoder }
    }

    /// Predict valence and arousal from `AcousticFeatures`.
    pub fn predict(&self, features: &AcousticFeatures) -> ValenceArousal {
        let mut x: Vec<f32> = features
            .mfcc
            .iter()
            .chain(features.pitch_stats.iter())
            .chain(features.energy_stats.iter())
            .chain(std::iter::once(&features.zcr))
            .copied()
            .collect();

        for (i, (weights, bias)) in self.encoder.iter().enumerate() {
            let n_in = weights.first().map(|r| r.len()).unwrap_or(0);
            x.resize(n_in, 0.0);
            let is_last = i == self.encoder.len() - 1;
            x = mlp_layer_forward(&x, weights, bias, !is_last);
        }
        // Squeeze outputs to [-1, 1] with tanh.
        let valence = x.first().copied().unwrap_or(0.0).tanh();
        let arousal = x.get(1).copied().unwrap_or(0.0).tanh();
        ValenceArousal::new(valence, arousal)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Multi-Modal Emotion Fuser
// ─────────────────────────────────────────────────────────────────────────────

/// A single modality's prediction represented as raw logits + confidence.
#[derive(Debug, Clone)]
pub struct ModalityPrediction {
    pub modality: String,
    pub logits: Vec<f32>,
    pub confidence: f32,
}

impl ModalityPrediction {
    pub fn new(modality: impl Into<String>, logits: Vec<f32>, confidence: f32) -> Self {
        Self {
            modality: modality.into(),
            logits,
            confidence,
        }
    }
}

/// Strategy for fusing multiple modality predictions.
#[derive(Debug, Clone)]
pub enum ErFusionStrategy {
    /// Simple arithmetic mean of logits.
    AverageFusion,
    /// Weighted sum (weights must sum to 1 or will be normalised).
    WeightedFusion(Vec<f32>),
    /// Soft-attention over predictions using the mean as query.
    AttentionFusion,
}

/// Fuses predictions from multiple modalities into a single emotion estimate.
#[derive(Debug, Clone)]
pub struct MultiModalEmotionFuser {
    pub strategy: ErFusionStrategy,
    pub n_emotions: usize,
}

impl MultiModalEmotionFuser {
    pub fn new(strategy: ErFusionStrategy, n_emotions: usize) -> Self {
        Self {
            strategy,
            n_emotions,
        }
    }

    /// Fuse modality predictions into a single logit vector (length n_emotions).
    pub fn fuse(&self, predictions: &[ModalityPrediction]) -> Vec<f32> {
        if predictions.is_empty() {
            return vec![0.0; self.n_emotions];
        }
        match &self.strategy {
            ErFusionStrategy::AverageFusion => {
                let n = predictions.len() as f32;
                let mut out = vec![0.0_f32; self.n_emotions];
                for pred in predictions {
                    for (o, l) in out.iter_mut().zip(pred.logits.iter()) {
                        *o += l / n;
                    }
                }
                out
            }
            ErFusionStrategy::WeightedFusion(weights) => {
                let wsum: f32 = weights.iter().sum::<f32>();
                let wsum = if wsum.abs() < 1e-12 { 1.0 } else { wsum };
                let mut out = vec![0.0_f32; self.n_emotions];
                for (pred, &w) in predictions.iter().zip(weights.iter()) {
                    let w_norm = w / wsum;
                    for (o, l) in out.iter_mut().zip(pred.logits.iter()) {
                        *o += w_norm * l;
                    }
                }
                out
            }
            ErFusionStrategy::AttentionFusion => {
                // Query = element-wise mean of all logits.
                let n = predictions.len();
                let mut query = vec![0.0_f32; self.n_emotions];
                for pred in predictions {
                    for (q, l) in query.iter_mut().zip(pred.logits.iter()) {
                        *q += l / n as f32;
                    }
                }
                // Attention score = dot(query, pred_logits) / sqrt(n_emotions).
                let scale = (self.n_emotions as f32).sqrt().max(1.0);
                let mut scores: Vec<f32> = predictions
                    .iter()
                    .map(|pred| {
                        query
                            .iter()
                            .zip(pred.logits.iter())
                            .map(|(q, l)| q * l)
                            .sum::<f32>()
                            / scale
                    })
                    .collect();
                // Softmax over scores.
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp() / exp_sum.max(1e-12);
                }
                // Weighted sum of logit vectors.
                let mut out = vec![0.0_f32; self.n_emotions];
                for (pred, &attn) in predictions.iter().zip(scores.iter()) {
                    for (o, l) in out.iter_mut().zip(pred.logits.iter()) {
                        *o += attn * l;
                    }
                }
                out
            }
        }
    }

    /// Return the most likely `BasicEmotion` from fused logits.
    pub fn predict_emotion(&self, predictions: &[ModalityPrediction]) -> BasicEmotion {
        let logits = self.fuse(predictions);
        argmax_emotion(&logits)
    }

    /// Return `ValenceArousal` by weighting discrete_to_va by softmax probabilities.
    pub fn predict_va(&self, predictions: &[ModalityPrediction]) -> ValenceArousal {
        let logits = self.fuse(predictions);
        let probs = softmax_vec(&logits);
        let mut v = 0.0_f32;
        let mut a = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            let emotion = BasicEmotion::from_index(i);
            let va = discrete_to_va(emotion);
            v += p * va.valence;
            a += p * va.arousal;
        }
        ValenceArousal::new(v, a)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Continuous Emotion Tracker
// ─────────────────────────────────────────────────────────────────────────────

/// A single timestamped emotion observation.
#[derive(Debug, Clone)]
pub struct EmotionState {
    pub emotion: ValenceArousal,
    pub timestamp: f32,
    pub intensity: f32,
}

impl EmotionState {
    pub fn new(emotion: ValenceArousal, timestamp: f32, intensity: f32) -> Self {
        Self {
            emotion,
            timestamp,
            intensity,
        }
    }
}

/// Ordered sequence of emotion states.
#[derive(Debug, Clone, Default)]
pub struct EmotionTrajectory {
    pub states: Vec<EmotionState>,
}

impl EmotionTrajectory {
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    pub fn push(&mut self, state: EmotionState) {
        self.states.push(state);
    }
}

/// Independent 1-D Kalman filter applied separately to valence and arousal.
///
/// State model: `x_{t+1} = x_t + w` (random walk).
/// Observation: `z_t = x_t + v`.
#[derive(Debug, Clone)]
pub struct KalmanEmotionFilter {
    /// Current state estimate.
    pub state: ValenceArousal,
    /// 2×2 error covariance (diagonal: [cov_valence, cov_arousal]).
    pub covariance: Vec<Vec<f32>>,
    /// Process noise variance Q.
    pub process_noise: f32,
    /// Measurement noise variance R.
    pub measurement_noise: f32,
}

impl KalmanEmotionFilter {
    pub fn new(initial: ValenceArousal, process_noise: f32, measurement_noise: f32) -> Self {
        Self {
            state: initial,
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            process_noise,
            measurement_noise,
        }
    }

    /// Perform a Kalman predict+update step and return the filtered state.
    pub fn update(&mut self, measurement: ValenceArousal) -> ValenceArousal {
        // Predict.
        let p_v = self.covariance[0][0] + self.process_noise;
        let p_a = self.covariance[1][1] + self.process_noise;
        // Update (Kalman gain per dimension).
        let k_v = p_v / (p_v + self.measurement_noise).max(1e-12);
        let k_a = p_a / (p_a + self.measurement_noise).max(1e-12);
        self.state.valence += k_v * (measurement.valence - self.state.valence);
        self.state.arousal += k_a * (measurement.arousal - self.state.arousal);
        self.covariance[0][0] = (1.0 - k_v) * p_v;
        self.covariance[1][1] = (1.0 - k_a) * p_a;
        self.state
    }
}

/// First-order Markov model over the seven basic emotions.
#[derive(Debug, Clone)]
pub struct EmotionTransitionModel {
    /// Row-stochastic 7×7 matrix: `T[i][j]` = P(next = j | current = i).
    pub transition_matrix: Vec<Vec<f32>>,
}

impl EmotionTransitionModel {
    /// Create a model with a near-identity prior (high self-transition probability).
    pub fn new() -> Self {
        let n = 7_usize;
        let stay = 0.70_f32;
        let spread = (1.0 - stay) / (n - 1) as f32;
        let matrix: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { stay } else { spread }).collect())
            .collect();
        Self {
            transition_matrix: matrix,
        }
    }

    /// Return the most likely next emotion given the current state.
    pub fn most_likely_next(&self, current: BasicEmotion) -> BasicEmotion {
        let row = &self.transition_matrix[current.index()];
        argmax_emotion(row)
    }

    /// Greedy Viterbi decoding given per-step observation likelihoods.
    ///
    /// `observations`: each entry is a length-7 probability vector over emotions.
    /// Returns the most probable state sequence of length `n_steps`.
    pub fn viterbi(&self, observations: &[Vec<f32>], n_steps: usize) -> Vec<BasicEmotion> {
        let n_states = 7;
        let steps = n_steps.min(observations.len());
        if steps == 0 {
            return Vec::new();
        }

        let mut delta: Vec<f32> = observations[0]
            .iter()
            .copied()
            .take(n_states)
            .chain(std::iter::repeat(0.0))
            .take(n_states)
            .collect();
        let mut psi: Vec<Vec<usize>> = vec![vec![0_usize; n_states]; steps];

        for t in 1..steps {
            let obs = &observations[t];
            let mut new_delta = vec![0.0_f32; n_states];
            for j in 0..n_states {
                let obs_j = obs.get(j).copied().unwrap_or(0.0);
                let (best_prev, best_val) = (0..n_states)
                    .map(|i| (i, delta[i] * self.transition_matrix[i][j]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, 0.0));
                new_delta[j] = best_val * obs_j;
                psi[t][j] = best_prev;
            }
            // Normalise to prevent underflow.
            let norm: f32 = new_delta.iter().sum::<f32>().max(1e-30);
            for d in new_delta.iter_mut() {
                *d /= norm;
            }
            delta = new_delta;
        }

        // Backtrack.
        let mut path = vec![0_usize; steps];
        path[steps - 1] = delta
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for t in (1..steps).rev() {
            path[t - 1] = psi[t][path[t]];
        }
        path.iter().map(|&i| BasicEmotion::from_index(i)).collect()
    }
}

impl Default for EmotionTransitionModel {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of emotion recognition evaluation functions.
pub struct EmotionMetrics;

impl EmotionMetrics {
    /// Classification accuracy (fraction of correct predictions).
    pub fn accuracy(predictions: &[BasicEmotion], targets: &[BasicEmotion]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| p == t)
            .count();
        correct as f32 / predictions.len() as f32
    }

    /// 7×7 confusion matrix (rows = true, columns = predicted).
    pub fn confusion_matrix(
        predictions: &[BasicEmotion],
        targets: &[BasicEmotion],
    ) -> Vec<Vec<usize>> {
        let mut cm = vec![vec![0_usize; 7]; 7];
        for (p, t) in predictions.iter().zip(targets.iter()) {
            cm[t.index()][p.index()] += 1;
        }
        cm
    }

    /// Macro-averaged F1 score over all 7 classes.
    pub fn f1_macro(predictions: &[BasicEmotion], targets: &[BasicEmotion]) -> f32 {
        let cm = Self::confusion_matrix(predictions, targets);
        let f1s: Vec<f32> = (0..7)
            .map(|c| {
                let tp = cm[c][c] as f32;
                let fp: f32 = (0..7)
                    .map(|r| if r != c { cm[r][c] as f32 } else { 0.0 })
                    .sum();
                let fn_: f32 = (0..7)
                    .map(|p| if p != c { cm[c][p] as f32 } else { 0.0 })
                    .sum();
                let precision = tp / (tp + fp).max(1e-12);
                let recall = tp / (tp + fn_).max(1e-12);
                if precision + recall < 1e-12 {
                    0.0
                } else {
                    2.0 * precision * recall / (precision + recall)
                }
            })
            .collect();
        f1s.iter().sum::<f32>() / 7.0
    }

    /// Root mean squared error on the valence dimension.
    pub fn valence_rmse(predictions: &[ValenceArousal], targets: &[ValenceArousal]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let n = predictions.len() as f32;
        let mse: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p.valence - t.valence).powi(2))
            .sum::<f32>()
            / n;
        mse.sqrt()
    }

    /// Root mean squared error on the arousal dimension.
    pub fn arousal_rmse(predictions: &[ValenceArousal], targets: &[ValenceArousal]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let n = predictions.len() as f32;
        let mse: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p.arousal - t.arousal).powi(2))
            .sum::<f32>()
            / n;
        mse.sqrt()
    }

    /// Concordance Correlation Coefficient between two sequences.
    ///
    /// CCC = 2σ_{xy} / (σ²_x + σ²_y + (μ_x - μ_y)²)
    pub fn concordance_correlation_coefficient(pred: &[f32], target: &[f32]) -> f32 {
        let n = pred.len().min(target.len());
        if n == 0 {
            return 0.0;
        }
        let n_f = n as f32;
        let mean_p = pred[..n].iter().sum::<f32>() / n_f;
        let mean_t = target[..n].iter().sum::<f32>() / n_f;
        let var_p = pred[..n].iter().map(|x| (x - mean_p).powi(2)).sum::<f32>() / n_f;
        let var_t = target[..n]
            .iter()
            .map(|x| (x - mean_t).powi(2))
            .sum::<f32>()
            / n_f;
        let cov: f32 = pred[..n]
            .iter()
            .zip(target[..n].iter())
            .map(|(p, t)| (p - mean_p) * (t - mean_t))
            .sum::<f32>()
            / n_f;
        let denom = var_p + var_t + (mean_p - mean_t).powi(2);
        if denom.abs() < 1e-12 {
            1.0
        } else {
            2.0 * cov / denom
        }
    }
}

/// Structured evaluation report.
#[derive(Debug, Clone)]
pub struct EmotionEvalReport {
    pub accuracy: f32,
    pub f1_macro: f32,
    pub valence_rmse: f32,
    pub arousal_rmse: f32,
    pub ccc_valence: f32,
    pub ccc_arousal: f32,
}

/// Evaluate discrete (BasicEmotion) predictions.
pub fn evaluate_discrete(pred: &[BasicEmotion], target: &[BasicEmotion]) -> EmotionEvalReport {
    let acc = EmotionMetrics::accuracy(pred, target);
    let f1 = EmotionMetrics::f1_macro(pred, target);
    let pred_va: Vec<ValenceArousal> = pred.iter().map(|&e| discrete_to_va(e)).collect();
    let tgt_va: Vec<ValenceArousal> = target.iter().map(|&e| discrete_to_va(e)).collect();
    let v_rmse = EmotionMetrics::valence_rmse(&pred_va, &tgt_va);
    let a_rmse = EmotionMetrics::arousal_rmse(&pred_va, &tgt_va);
    let pred_v: Vec<f32> = pred_va.iter().map(|va| va.valence).collect();
    let tgt_v: Vec<f32> = tgt_va.iter().map(|va| va.valence).collect();
    let pred_a: Vec<f32> = pred_va.iter().map(|va| va.arousal).collect();
    let tgt_a: Vec<f32> = tgt_va.iter().map(|va| va.arousal).collect();
    let ccc_v = EmotionMetrics::concordance_correlation_coefficient(&pred_v, &tgt_v);
    let ccc_a = EmotionMetrics::concordance_correlation_coefficient(&pred_a, &tgt_a);
    EmotionEvalReport {
        accuracy: acc,
        f1_macro: f1,
        valence_rmse: v_rmse,
        arousal_rmse: a_rmse,
        ccc_valence: ccc_v,
        ccc_arousal: ccc_a,
    }
}

/// Evaluate continuous (ValenceArousal) predictions.
pub fn evaluate_continuous(
    pred: &[ValenceArousal],
    target: &[ValenceArousal],
) -> EmotionEvalReport {
    let v_rmse = EmotionMetrics::valence_rmse(pred, target);
    let a_rmse = EmotionMetrics::arousal_rmse(pred, target);
    let pred_v: Vec<f32> = pred.iter().map(|va| va.valence).collect();
    let tgt_v: Vec<f32> = target.iter().map(|va| va.valence).collect();
    let pred_a: Vec<f32> = pred.iter().map(|va| va.arousal).collect();
    let tgt_a: Vec<f32> = target.iter().map(|va| va.arousal).collect();
    let ccc_v = EmotionMetrics::concordance_correlation_coefficient(&pred_v, &tgt_v);
    let ccc_a = EmotionMetrics::concordance_correlation_coefficient(&pred_a, &tgt_a);
    // Discrete metrics: convert predictions via va_to_discrete.
    let pred_disc: Vec<BasicEmotion> = pred.iter().map(|&va| va_to_discrete(va)).collect();
    let tgt_disc: Vec<BasicEmotion> = target.iter().map(|&va| va_to_discrete(va)).collect();
    let acc = EmotionMetrics::accuracy(&pred_disc, &tgt_disc);
    let f1 = EmotionMetrics::f1_macro(&pred_disc, &tgt_disc);
    EmotionEvalReport {
        accuracy: acc,
        f1_macro: f1,
        valence_rmse: v_rmse,
        arousal_rmse: a_rmse,
        ccc_valence: ccc_v,
        ccc_arousal: ccc_a,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Argmax over a slice returning a `BasicEmotion`.
fn argmax_emotion(logits: &[f32]) -> BasicEmotion {
    let idx = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(6);
    BasicEmotion::from_index(idx)
}

/// ReLU activation.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Softmax over a slice; returns a new `Vec<f32>`.
fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|e| e / sum).collect()
}

/// Dense layer forward: y = ReLU(Wx + b) (or linear if `activate == false`).
fn mlp_layer_forward(x: &[f32], weights: &[Vec<f32>], bias: &[f32], activate: bool) -> Vec<f32> {
    weights
        .iter()
        .zip(bias.iter())
        .map(|(row, b)| {
            let dot: f32 = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            let y = dot + b;
            if activate {
                relu(y)
            } else {
                y
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // ── taxonomy ──────────────────────────────────────────────────────────────

    #[test]
    fn test_va_to_discrete_happy_quadrant() {
        let va = ValenceArousal::new(0.7, 0.3);
        let e = va_to_discrete(va);
        assert!(e == BasicEmotion::Happy || e == BasicEmotion::Surprised);
    }

    #[test]
    fn test_va_to_discrete_sad_quadrant() {
        let va = ValenceArousal::new(-0.6, -0.5);
        let e = va_to_discrete(va);
        assert!(
            e == BasicEmotion::Sad || e == BasicEmotion::Disgusted,
            "expected Sad or Disgusted, got {:?}",
            e
        );
    }

    #[test]
    fn test_discrete_to_va_happy_positive_valence() {
        let va = discrete_to_va(BasicEmotion::Happy);
        assert!(va.valence > 0.0, "Happy valence should be positive");
    }

    #[test]
    fn test_au_profile_creation() {
        let profile = AuProfile::new(vec![(ActionUnit::Au12, 3.0), (ActionUnit::Au6, 2.0)]);
        assert_eq!(profile.au_intensities.len(), 2);
        assert!((profile.get(ActionUnit::Au12) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_facs_happy_au12() {
        let profile = AuProfile::new(vec![(ActionUnit::Au12, 3.0), (ActionUnit::Au6, 2.5)]);
        assert_eq!(AuFacsRules::facs_classify(&profile), BasicEmotion::Happy);
    }

    #[test]
    fn test_facs_sad_au15() {
        let profile = AuProfile::new(vec![(ActionUnit::Au15, 2.5), (ActionUnit::Au17, 2.0)]);
        assert_eq!(AuFacsRules::facs_classify(&profile), BasicEmotion::Sad);
    }

    #[test]
    fn test_facs_rules_angry_au23() {
        let profile = AuProfile::new(vec![
            (ActionUnit::Au4, 2.0),
            (ActionUnit::Au7, 1.5),
            (ActionUnit::Au23, 1.5),
        ]);
        assert_eq!(AuFacsRules::facs_classify(&profile), BasicEmotion::Angry);
    }

    #[test]
    fn test_facial_au_encoder_forward_shape() {
        let mut rng = make_rng();
        let enc = FacialAuEncoder::new(11, 7, &mut rng);
        let profile = AuProfile::new(vec![(ActionUnit::Au12, 3.0)]);
        let logits = enc.encode(&profile);
        assert_eq!(logits.len(), 7, "should have 7 logits");
    }

    #[test]
    fn test_facial_au_encoder_predict_returns_emotion() {
        let mut rng = make_rng();
        let enc = FacialAuEncoder::new(11, 7, &mut rng);
        let profile = AuProfile::new(vec![(ActionUnit::Au12, 3.0)]);
        let _emotion = enc.predict(&profile); // just must not panic
    }

    // ── EEG ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_eeg_band_power_sine_wave() {
        let sample_rate = 256.0_f32;
        let n = 512_usize;
        // Pure 10 Hz sine → should land in Alpha (8-13 Hz).
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 10.0 * i as f32 / sample_rate).sin())
            .collect();
        let alpha_power = extract_band_power(&signal, sample_rate, EegBand::Alpha);
        let delta_power = extract_band_power(&signal, sample_rate, EegBand::Delta);
        assert!(alpha_power > delta_power,
            "10 Hz sine should have higher alpha than delta power: alpha={alpha_power:.4}, delta={delta_power:.4}");
    }

    #[test]
    fn test_eeg_band_delta_range() {
        let (low, high) = EegBand::Delta.limits();
        assert!((low - 0.5).abs() < 1e-5 && (high - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_eeg_band_gamma_range() {
        let (low, high) = EegBand::Gamma.limits();
        assert!((low - 30.0).abs() < 1e-5 && (high - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_eeg_differential_entropy_positive() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let de = differential_entropy(&signal);
        // Should be a finite real number (can be negative for very small variance).
        assert!(de.is_finite());
    }

    #[test]
    fn test_eeg_feature_extraction_shape() {
        let n_ch = 4_usize;
        let signals: Vec<Vec<f32>> = (0..n_ch)
            .map(|_| (0..256).map(|i| (i as f32 * 0.1).sin()).collect())
            .collect();
        let feats = extract_features(&signals, 256.0);
        assert_eq!(feats.band_power.len(), 5 * n_ch);
        assert_eq!(feats.differential_entropy.len(), 5 * n_ch);
        assert_eq!(feats.asymmetry.len(), 5);
    }

    #[test]
    fn test_eeg_emotion_net_forward_shape() {
        let mut rng = make_rng();
        let n_feat = 5 * 4 * 2 + 5; // 4 channels, 5 bands, 2 feature types + asymmetry
        let net = EegEmotionNet::new(n_feat, 32, 7, &mut rng);
        let feats = EegFeatures {
            band_power: vec![0.1; 20],
            differential_entropy: vec![0.2; 20],
            asymmetry: vec![0.05; 5],
        };
        let probs = net.forward(&feats);
        assert_eq!(probs.len(), 7);
    }

    #[test]
    fn test_eeg_emotion_net_forward_sum_to_one() {
        let mut rng = make_rng();
        let net = EegEmotionNet::new(45, 32, 7, &mut rng);
        let feats = EegFeatures {
            band_power: vec![0.1; 20],
            differential_entropy: vec![0.2; 20],
            asymmetry: vec![0.05; 5],
        };
        let probs = net.forward(&feats);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax should sum to 1, got {sum}"
        );
    }

    // ── Speech ────────────────────────────────────────────────────────────────

    #[test]
    fn test_speech_extract_mfcc_shape() {
        let signal: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mfcc = extract_mfcc(&signal, 16000.0, 13);
        assert_eq!(mfcc.len(), 13, "should produce 13 MFCC coefficients");
    }

    #[test]
    fn test_speech_extract_pitch_nonempty() {
        let signal: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();
        let pitch = extract_pitch(&signal, 16000.0);
        assert!(!pitch.is_empty());
    }

    #[test]
    fn test_speech_acoustic_features_extraction() {
        let signal: Vec<f32> = (0..4000)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / 8000.0).sin() * 0.5)
            .collect();
        let feats = extract_acoustic_features(&signal, 8000.0);
        assert_eq!(feats.mfcc.len(), 13);
        assert_eq!(feats.pitch_stats.len(), 4);
        assert_eq!(feats.energy_stats.len(), 3);
        assert!(feats.zcr >= 0.0);
    }

    #[test]
    fn test_ser_predict_returns_va() {
        let mut rng = make_rng();
        let ser = SpeechEmotionRecognizer::new(21, 32, &mut rng);
        let feats = extract_acoustic_features(
            &(0..4000)
                .map(|i| (i as f32 * 0.01).sin())
                .collect::<Vec<_>>(),
            16000.0,
        );
        let _va = ser.predict(&feats); // must not panic
    }

    #[test]
    fn test_ser_va_in_range() {
        let mut rng = make_rng();
        let ser = SpeechEmotionRecognizer::new(21, 32, &mut rng);
        let feats = AcousticFeatures {
            mfcc: vec![0.0; 13],
            pitch_stats: vec![150.0, 10.0, 100.0, 200.0],
            energy_stats: vec![0.01, 0.001, 0.1],
            zcr: 0.05,
        };
        let va = ser.predict(&feats);
        assert!(
            va.valence >= -1.0 && va.valence <= 1.0,
            "valence out of range: {}",
            va.valence
        );
        assert!(
            va.arousal >= -1.0 && va.arousal <= 1.0,
            "arousal out of range: {}",
            va.arousal
        );
    }

    // ── Multi-Modal Fusion ────────────────────────────────────────────────────

    #[test]
    fn test_modal_prediction_creation() {
        let mp = ModalityPrediction::new("face", vec![0.7, 0.1, 0.2], 0.85);
        assert_eq!(mp.modality, "face");
        assert!((mp.confidence - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_average_fusion_equal_weights() {
        let fuser = MultiModalEmotionFuser::new(ErFusionStrategy::AverageFusion, 7);
        let preds = vec![
            ModalityPrediction::new("a", vec![1.0; 7], 1.0),
            ModalityPrediction::new("b", vec![1.0; 7], 1.0),
        ];
        let fused = fuser.fuse(&preds);
        assert_eq!(fused.len(), 7);
        assert!((fused[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_weighted_fusion_shapes() {
        let fuser =
            MultiModalEmotionFuser::new(ErFusionStrategy::WeightedFusion(vec![0.6, 0.4]), 7);
        let preds = vec![
            ModalityPrediction::new("a", vec![2.0; 7], 0.9),
            ModalityPrediction::new("b", vec![1.0; 7], 0.5),
        ];
        let fused = fuser.fuse(&preds);
        assert_eq!(fused.len(), 7);
        // Expected: 0.6*2 + 0.4*1 = 1.6
        assert!((fused[0] - 1.6).abs() < 1e-4, "fused={}", fused[0]);
    }

    #[test]
    fn test_attention_fusion_shape() {
        let fuser = MultiModalEmotionFuser::new(ErFusionStrategy::AttentionFusion, 7);
        let preds = vec![
            ModalityPrediction::new("a", vec![0.5; 7], 0.8),
            ModalityPrediction::new("b", vec![0.3; 7], 0.6),
        ];
        let fused = fuser.fuse(&preds);
        assert_eq!(fused.len(), 7);
    }

    #[test]
    fn test_multi_modal_predict_emotion() {
        let fuser = MultiModalEmotionFuser::new(ErFusionStrategy::AverageFusion, 7);
        let mut logits = vec![0.0; 7];
        logits[BasicEmotion::Happy.index()] = 5.0;
        let preds = vec![ModalityPrediction::new("face", logits, 0.9)];
        let e = fuser.predict_emotion(&preds);
        assert_eq!(e, BasicEmotion::Happy);
    }

    #[test]
    fn test_multi_modal_predict_va_range() {
        let fuser = MultiModalEmotionFuser::new(ErFusionStrategy::AverageFusion, 7);
        let preds = vec![ModalityPrediction::new("face", vec![1.0; 7], 0.8)];
        let va = fuser.predict_va(&preds);
        assert!(va.valence.abs() <= 1.5, "valence={}", va.valence);
        assert!(va.arousal.abs() <= 1.5, "arousal={}", va.arousal);
    }

    // ── Continuous tracker ────────────────────────────────────────────────────

    #[test]
    fn test_emotion_state_creation() {
        let va = ValenceArousal::new(0.5, -0.2);
        let state = EmotionState::new(va, 1.5, 0.8);
        assert!((state.timestamp - 1.5).abs() < 1e-5);
        assert!((state.intensity - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_emotion_trajectory_creation() {
        let mut traj = EmotionTrajectory::new();
        traj.push(EmotionState::new(ValenceArousal::new(0.3, 0.1), 0.0, 0.5));
        traj.push(EmotionState::new(ValenceArousal::new(0.4, 0.2), 1.0, 0.6));
        assert_eq!(traj.states.len(), 2);
    }

    #[test]
    fn test_kalman_filter_update() {
        let init = ValenceArousal::new(0.0, 0.0);
        let mut kf = KalmanEmotionFilter::new(init, 0.01, 0.1);
        let meas = ValenceArousal::new(0.8, 0.6);
        let filtered = kf.update(meas);
        // Should move toward measurement.
        assert!(filtered.valence > 0.0, "valence should move toward 0.8");
        assert!(filtered.arousal > 0.0, "arousal should move toward 0.6");
    }

    #[test]
    fn test_kalman_filter_smoothing() {
        let init = ValenceArousal::new(0.0, 0.0);
        let mut kf = KalmanEmotionFilter::new(init, 0.01, 0.1);
        let meas = ValenceArousal::new(1.0, 1.0);
        for _ in 0..20 {
            kf.update(meas);
        }
        // After many updates should converge close to measurement.
        assert!(
            kf.state.valence > 0.8,
            "valence should converge, got {}",
            kf.state.valence
        );
        assert!(
            kf.state.arousal > 0.8,
            "arousal should converge, got {}",
            kf.state.arousal
        );
    }

    #[test]
    fn test_emotion_transition_matrix_row_sum() {
        let model = EmotionTransitionModel::new();
        for (i, row) in model.transition_matrix.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {i} sums to {sum}");
        }
    }

    #[test]
    fn test_most_likely_next_valid_emotion() {
        let model = EmotionTransitionModel::new();
        let next = model.most_likely_next(BasicEmotion::Happy);
        // The self-transition is the highest entry, so next should be Happy.
        assert_eq!(next, BasicEmotion::Happy);
    }

    #[test]
    fn test_viterbi_output_length() {
        let model = EmotionTransitionModel::new();
        let observations: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0 / 7.0; 7]).collect();
        let path = model.viterbi(&observations, 5);
        assert_eq!(path.len(), 5);
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    #[test]
    fn test_emotion_accuracy_perfect() {
        let labels = vec![BasicEmotion::Happy, BasicEmotion::Sad, BasicEmotion::Angry];
        let acc = EmotionMetrics::accuracy(&labels, &labels);
        assert!((acc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_emotion_accuracy_zero() {
        let pred = vec![BasicEmotion::Happy, BasicEmotion::Happy];
        let tgt = vec![BasicEmotion::Sad, BasicEmotion::Angry];
        let acc = EmotionMetrics::accuracy(&pred, &tgt);
        assert!(acc < 1e-5);
    }

    #[test]
    fn test_confusion_matrix_shape() {
        let pred = vec![BasicEmotion::Happy, BasicEmotion::Sad];
        let tgt = vec![BasicEmotion::Happy, BasicEmotion::Sad];
        let cm = EmotionMetrics::confusion_matrix(&pred, &tgt);
        assert_eq!(cm.len(), 7);
        assert_eq!(cm[0].len(), 7);
    }

    #[test]
    fn test_confusion_matrix_diagonal() {
        let labels = BasicEmotion::all().to_vec();
        let cm = EmotionMetrics::confusion_matrix(&labels, &labels);
        for i in 0..7 {
            assert_eq!(cm[i][i], 1, "diagonal should be 1 for class {i}");
        }
    }

    #[test]
    fn test_f1_macro_perfect() {
        let labels = vec![
            BasicEmotion::Happy,
            BasicEmotion::Sad,
            BasicEmotion::Angry,
            BasicEmotion::Fearful,
            BasicEmotion::Disgusted,
            BasicEmotion::Surprised,
            BasicEmotion::Neutral,
        ];
        let f1 = EmotionMetrics::f1_macro(&labels, &labels);
        assert!(
            (f1 - 1.0).abs() < 1e-4,
            "perfect F1 should be 1.0, got {f1}"
        );
    }

    #[test]
    fn test_valence_rmse_zero() {
        let va = vec![
            ValenceArousal::new(0.5, 0.3),
            ValenceArousal::new(-0.2, 0.1),
        ];
        let rmse = EmotionMetrics::valence_rmse(&va, &va);
        assert!(rmse < 1e-5);
    }

    #[test]
    fn test_arousal_rmse_positive() {
        let pred = vec![ValenceArousal::new(0.5, 0.8)];
        let tgt = vec![ValenceArousal::new(0.5, -0.2)];
        let rmse = EmotionMetrics::arousal_rmse(&pred, &tgt);
        assert!(rmse > 0.0);
    }

    #[test]
    fn test_ccc_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ccc = EmotionMetrics::concordance_correlation_coefficient(&x, &x);
        assert!(
            (ccc - 1.0).abs() < 1e-4,
            "perfect CCC should be 1.0, got {ccc}"
        );
    }

    #[test]
    fn test_ccc_range() {
        let pred = vec![0.2, -0.5, 0.8, 0.0, -0.3];
        let tgt = vec![-0.1, 0.4, 0.2, -0.7, 0.9];
        let ccc = EmotionMetrics::concordance_correlation_coefficient(&pred, &tgt);
        assert!((-1.0..=1.0).contains(&ccc), "CCC out of range: {ccc}");
    }

    #[test]
    fn test_eval_report_discrete_fields() {
        let pred = vec![BasicEmotion::Happy, BasicEmotion::Sad];
        let tgt = vec![BasicEmotion::Happy, BasicEmotion::Angry];
        let report = evaluate_discrete(&pred, &tgt);
        assert!(report.accuracy >= 0.0 && report.accuracy <= 1.0);
        assert!(report.f1_macro >= 0.0 && report.f1_macro <= 1.0);
    }

    #[test]
    fn test_eval_report_continuous_fields() {
        let pred = vec![
            ValenceArousal::new(0.5, 0.3),
            ValenceArousal::new(-0.2, 0.1),
        ];
        let tgt = vec![
            ValenceArousal::new(0.4, 0.2),
            ValenceArousal::new(-0.3, 0.2),
        ];
        let report = evaluate_continuous(&pred, &tgt);
        assert!(report.valence_rmse >= 0.0);
        assert!(report.arousal_rmse >= 0.0);
    }
}
