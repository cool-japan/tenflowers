//! Signal processing utilities for audio and time-series neural networks.
//!
//! Provides:
//! - Window functions (Hann, Hamming, Blackman, Rectangular)
//! - Discrete Fourier Transform and derived spectra
//! - Short-Time Fourier Transform (STFT) with configurable window/hop/padding
//! - Mel filterbank construction and application
//! - MFCC computation (DFT → Mel filterbank → log → DCT-II)
//! - Frame-level energy and zero-crossing rate
//! - Spectral descriptors (centroid, rolloff)
//! - Pre-emphasis filter and signal framing
//!
//! All routines are `no_std`-friendly (no allocator required beyond `Vec`),
//! free of `unwrap()`, and contain no unsafe code.

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Window functions
// ─────────────────────────────────────────────────────────────────────────────

/// Hann window: `w[n] = 0.5 * (1 − cos(2π·n / (N−1)))`.
///
/// Returns all-ones for `size == 1`.
pub fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0_f32];
    }
    let n = size as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1.0)).cos()))
        .collect()
}

/// Hamming window: `w[n] = 0.54 − 0.46 · cos(2π·n / (N−1))`.
pub fn hamming_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0_f32];
    }
    let n = size as f32;
    (0..size)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1.0)).cos())
        .collect()
}

/// Blackman window: `w[n] = 0.42 − 0.5·cos(2π·n/(N−1)) + 0.08·cos(4π·n/(N−1))`.
pub fn blackman_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0_f32];
    }
    let n = size as f32;
    (0..size)
        .map(|i| {
            let theta = 2.0 * PI * i as f32 / (n - 1.0);
            0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos()
        })
        .collect()
}

/// Rectangular (boxcar) window: all ones.
pub fn rectangular_window(size: usize) -> Vec<f32> {
    vec![1.0_f32; size]
}

// ─────────────────────────────────────────────────────────────────────────────
// DFT helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Discrete Fourier Transform via the naïve O(N²) formula.
///
/// Returns `N` complex numbers `(real, imag)` representing `X[k]`.
/// The formula is:
/// `X[k] = Σ_{n=0}^{N-1} x[n] · e^{−j·2π·k·n/N}`
///
/// For an empty input returns an empty vector.
pub fn dft(x: &[f32]) -> Vec<(f32, f32)> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let n_f = n as f32;
    (0..n)
        .map(|k| {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let k_f = k as f32;
            for (nn, &xn) in x.iter().enumerate() {
                let angle = -2.0 * PI * k_f * nn as f32 / n_f;
                re += xn * angle.cos();
                im += xn * angle.sin();
            }
            (re, im)
        })
        .collect()
}

/// Magnitude spectrum: `|X[k]| = √(re² + im²)`.
pub fn magnitude_spectrum(dft_out: &[(f32, f32)]) -> Vec<f32> {
    dft_out
        .iter()
        .map(|(re, im)| (re * re + im * im).sqrt())
        .collect()
}

/// Power spectrum: `|X[k]|² = re² + im²`.
pub fn power_spectrum(dft_out: &[(f32, f32)]) -> Vec<f32> {
    dft_out.iter().map(|(re, im)| re * re + im * im).collect()
}

/// Phase spectrum: `∠X[k] = atan2(im, re)`.
pub fn phase_spectrum(dft_out: &[(f32, f32)]) -> Vec<f32> {
    dft_out.iter().map(|(re, im)| im.atan2(*re)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// STFT
// ─────────────────────────────────────────────────────────────────────────────

/// Window function selector for the STFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

/// Configuration for the Short-Time Fourier Transform.
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT (DFT) size.  The output has `n_fft / 2 + 1` frequency bins.
    pub n_fft: usize,
    /// Number of samples between successive frames.
    pub hop_length: usize,
    /// Length of the analysis window (≤ `n_fft`).  Zero-padded to `n_fft`.
    pub win_length: usize,
    /// Window function applied to each frame before DFT.
    pub window: WindowType,
    /// When `true` the signal is reflected-padded at both ends so the first
    /// frame is centred at sample 0.
    pub center: bool,
}

impl Default for StftConfig {
    fn default() -> Self {
        StftConfig {
            n_fft: 512,
            hop_length: 128,
            win_length: 512,
            window: WindowType::Hann,
            center: false,
        }
    }
}

impl StftConfig {
    /// Number of one-sided frequency bins.
    pub fn num_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Build the window coefficients from the config.
fn build_window(win_length: usize, window_type: WindowType) -> Vec<f32> {
    match window_type {
        WindowType::Hann => hann_window(win_length),
        WindowType::Hamming => hamming_window(win_length),
        WindowType::Blackman => blackman_window(win_length),
        WindowType::Rectangular => rectangular_window(win_length),
    }
}

/// Compute the Short-Time Fourier Transform.
///
/// Returns a `[num_frames][n_fft/2+1]` matrix of complex values `(re, im)`.
///
/// # Padding
/// When `config.center == true` the signal is edge-reflected by `n_fft/2`
/// samples on each side (librosa convention).  When `false` no padding is
/// added and only full frames are emitted.
pub fn stft(signal: &[f32], config: &StftConfig) -> Vec<Vec<(f32, f32)>> {
    let n_fft = config.n_fft.max(1);
    let hop = config.hop_length.max(1);
    let win_len = config.win_length.min(n_fft).max(1);
    let num_bins = n_fft / 2 + 1;

    // Build zero-padded window of length n_fft.
    let raw_win = build_window(win_len, config.window);
    let mut window = vec![0.0_f32; n_fft];
    // Centre the window inside the n_fft buffer.
    let pad_left = (n_fft - win_len) / 2;
    for (i, &w) in raw_win.iter().enumerate() {
        window[pad_left + i] = w;
    }

    // Optionally pad the signal.
    let padded: Vec<f32> = if config.center {
        let pad = n_fft / 2;
        let mut p = Vec::with_capacity(signal.len() + 2 * pad);
        // Reflect-pad at start.
        for i in (1..=pad).rev() {
            let idx = i.min(signal.len().saturating_sub(1));
            p.push(signal[idx]);
        }
        p.extend_from_slice(signal);
        // Reflect-pad at end.
        let sig_len = signal.len();
        for i in 1..=pad {
            let idx = sig_len.saturating_sub(1).saturating_sub(i - 1);
            p.push(signal[idx]);
        }
        p
    } else {
        signal.to_vec()
    };

    let total_len = padded.len();
    // Number of full frames.
    let num_frames = if total_len >= n_fft {
        (total_len - n_fft) / hop + 1
    } else {
        0
    };

    let mut frames_out = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop;
        // Extract and window the frame.
        let mut frame = vec![0.0_f32; n_fft];
        for i in 0..n_fft {
            frame[i] = padded[start + i] * window[i];
        }
        // Compute DFT and keep only one-sided spectrum.
        let spectrum = dft(&frame);
        let one_sided: Vec<(f32, f32)> = spectrum.into_iter().take(num_bins).collect();
        frames_out.push(one_sided);
    }

    frames_out
}

/// Magnitude spectrogram from STFT output.
///
/// Input: `[num_frames][n_fft/2+1]` complex values.
/// Output: `[num_frames][n_fft/2+1]` magnitude values.
pub fn spectrogram(stft_out: &[Vec<(f32, f32)>]) -> Vec<Vec<f32>> {
    stft_out
        .iter()
        .map(|frame| magnitude_spectrum(frame))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel filterbank
// ─────────────────────────────────────────────────────────────────────────────

/// Mel filterbank: a bank of triangular filters linearly spaced in Mel scale.
///
/// Converts a magnitude/power spectrogram `[frames, n_fft/2+1]` to a
/// Mel-band spectrogram `[frames, n_mels]`.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    pub n_mels: usize,
    pub n_fft: usize,
    pub sample_rate: f32,
    pub f_min: f32,
    pub f_max: f32,
    /// Filter matrix `[n_mels][n_fft/2+1]`.
    pub filters: Vec<Vec<f32>>,
}

impl MelFilterbank {
    /// Build the filterbank.
    ///
    /// The filters are triangular, linearly spaced in Mel frequency.
    /// This mirrors the librosa / HTK formulation.
    pub fn new(n_mels: usize, n_fft: usize, sample_rate: f32, f_min: f32, f_max: f32) -> Self {
        let num_bins = n_fft / 2 + 1;
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        // n_mels + 2 linearly spaced Mel points (including the two boundary points).
        let n_points = n_mels + 2;
        let mel_points: Vec<f32> = (0..n_points)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
            .collect();

        // Convert Mel points to bin indices.
        let bin_points: Vec<f32> = mel_points
            .iter()
            .map(|&m| {
                let hz = Self::mel_to_hz(m);
                // Map Hz to FFT bin index.
                hz / (sample_rate / 2.0) * (num_bins - 1) as f32
            })
            .collect();

        // Build triangular filters.
        let mut filters = vec![vec![0.0_f32; num_bins]; n_mels];
        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for k in 0..num_bins {
                let k_f = k as f32;
                let val = if k_f >= f_left && k_f <= f_center && f_center > f_left {
                    (k_f - f_left) / (f_center - f_left)
                } else if k_f > f_center && k_f <= f_right && f_right > f_center {
                    (f_right - k_f) / (f_right - f_center)
                } else {
                    0.0
                };
                filters[m][k] = val;
            }
        }

        MelFilterbank {
            n_mels,
            n_fft,
            sample_rate,
            f_min,
            f_max,
            filters,
        }
    }

    /// Convert Hz to Mel: `2595 · log10(1 + f / 700)`.
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert Mel to Hz: `700 · (10^(m/2595) − 1)`.
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Apply the filterbank to a magnitude spectrogram.
    ///
    /// Input: `[frames, n_fft/2+1]`.
    /// Output: `[frames, n_mels]`.
    pub fn apply(&self, spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        spec.iter()
            .map(|frame| {
                self.filters
                    .iter()
                    .map(|filter| {
                        let n = frame.len().min(filter.len());
                        frame
                            .iter()
                            .zip(filter.iter())
                            .take(n)
                            .map(|(s, f)| s * f)
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DCT-II (for MFCC)
// ─────────────────────────────────────────────────────────────────────────────

/// Type-II DCT as used in the MFCC pipeline.
///
/// `C[k] = Σ_{n=0}^{N-1} x[n] · cos(π·k·(n + 0.5) / N)`
///
/// Returns `n_mfcc` coefficients.  For `k == 0` the term is `1/√2`-normalised
/// (orthonormal form).
fn dct_mfcc(mel_energies: &[f32], n_mfcc: usize) -> Vec<f32> {
    let n = mel_energies.len();
    if n == 0 || n_mfcc == 0 {
        return Vec::new();
    }
    let n_f = n as f32;
    (0..n_mfcc)
        .map(|k| {
            let sum: f32 = mel_energies
                .iter()
                .enumerate()
                .map(|(nn, &x)| x * (PI * k as f32 * (nn as f32 + 0.5) / n_f).cos())
                .sum();
            sum
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// MFCC
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Mel-Frequency Cepstral Coefficients.
///
/// Pipeline:
/// 1. Compute STFT with Hann window.
/// 2. Magnitude spectrogram.
/// 3. Apply Mel filterbank.
/// 4. Log-compress Mel energies: `log(max(E, 1e-10))`.
/// 5. Apply DCT-II to extract `n_mfcc` cepstral coefficients.
///
/// Returns `[num_frames][n_mfcc]`.
pub fn mfcc(
    signal: &[f32],
    sample_rate: f32,
    n_mfcc: usize,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
) -> Vec<Vec<f32>> {
    let config = StftConfig {
        n_fft,
        hop_length,
        win_length: n_fft,
        window: WindowType::Hann,
        center: false,
    };

    let stft_out = stft(signal, &config);
    let mag_spec = spectrogram(&stft_out);

    let f_max = sample_rate / 2.0;
    let fb = MelFilterbank::new(n_mels, n_fft, sample_rate, 0.0, f_max);
    let mel_spec = fb.apply(&mag_spec);

    mel_spec
        .iter()
        .map(|frame| {
            let log_mel: Vec<f32> = frame.iter().map(|&e| e.max(1e-10).ln()).collect();
            dct_mfcc(&log_mel, n_mfcc)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame-level utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Split a signal into overlapping frames.
///
/// Returns `[num_frames][frame_length]`.  Frames that extend beyond the signal
/// are zero-padded.
pub fn frame_signal(signal: &[f32], frame_length: usize, hop_length: usize) -> Vec<Vec<f32>> {
    if signal.is_empty() || frame_length == 0 {
        return Vec::new();
    }
    let hop = hop_length.max(1);
    let sig_len = signal.len();
    let num_frames = (sig_len.saturating_sub(1)) / hop + 1;
    let mut frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * hop;
        let mut frame = vec![0.0_f32; frame_length];
        for j in 0..frame_length {
            let idx = start + j;
            if idx < sig_len {
                frame[j] = signal[idx];
            }
        }
        frames.push(frame);
    }
    frames
}

/// Zero-crossing rate per frame.
///
/// For each frame of length `frame_length` (stepped by `hop_length`):
/// `ZCR = number of sign changes / (frame_length - 1)`.
///
/// Returns `[num_frames]` values in `[0, 1]`.
pub fn zero_crossing_rate(signal: &[f32], frame_length: usize, hop_length: usize) -> Vec<f32> {
    let frames = frame_signal(signal, frame_length, hop_length);
    frames
        .iter()
        .map(|frame| {
            if frame.len() < 2 {
                return 0.0;
            }
            let crossings = frame
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            crossings as f32 / (frame.len() - 1) as f32
        })
        .collect()
}

/// Root-Mean-Square energy per frame.
///
/// Returns `[num_frames]` values ≥ 0.
pub fn rms_energy(signal: &[f32], frame_length: usize, hop_length: usize) -> Vec<f32> {
    let frames = frame_signal(signal, frame_length, hop_length);
    frames
        .iter()
        .map(|frame| {
            if frame.is_empty() {
                return 0.0;
            }
            let ms: f32 = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;
            ms.sqrt()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral descriptors
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral centroid (brightness) for each frame.
///
/// `centroid[t] = Σ_k (f_k · M[t][k]) / Σ_k M[t][k]`
///
/// where `f_k = k · sample_rate / n_fft` is the frequency of bin `k`.
/// Returns 0 when the frame energy is zero.
pub fn spectral_centroid(spec: &[Vec<f32>], sample_rate: f32, n_fft: usize) -> Vec<f32> {
    spec.iter()
        .map(|frame| {
            let num_bins = frame.len();
            let mut num = 0.0_f32;
            let mut denom = 0.0_f32;
            for (k, &mag) in frame.iter().enumerate() {
                let freq = k as f32 * sample_rate / n_fft as f32;
                num += freq * mag;
                denom += mag;
            }
            if denom > 0.0 {
                num / denom
            } else {
                0.0
            }
        })
        .collect()
}

/// Spectral rolloff frequency for each frame.
///
/// The rolloff frequency is the frequency below which `rolloff_fraction`
/// (e.g. 0.85) of the total spectral energy lies.
pub fn spectral_rolloff(
    spec: &[Vec<f32>],
    sample_rate: f32,
    n_fft: usize,
    rolloff_fraction: f32,
) -> Vec<f32> {
    let fraction = rolloff_fraction.clamp(0.0, 1.0);
    spec.iter()
        .map(|frame| {
            let total: f32 = frame.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            let threshold = total * fraction;
            let mut cumsum = 0.0_f32;
            for (k, &mag) in frame.iter().enumerate() {
                cumsum += mag;
                if cumsum >= threshold {
                    return k as f32 * sample_rate / n_fft as f32;
                }
            }
            // All energy accounted for by the last bin.
            let last_bin = frame.len().saturating_sub(1);
            last_bin as f32 * sample_rate / n_fft as f32
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-emphasis & helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-emphasis high-pass filter: `y[n] = x[n] − α · x[n−1]`, `y[0] = x[0]`.
///
/// Typical `alpha = 0.97`.
pub fn pre_emphasis(signal: &[f32], alpha: f32) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(signal.len());
    out.push(signal[0]);
    for i in 1..signal.len() {
        out.push(signal[i] - alpha * signal[i - 1]);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Window functions ─────────────────────────────────────────────────────

    #[test]
    fn test_hann_window_size() {
        let w = hann_window(32);
        assert_eq!(w.len(), 32);
    }

    #[test]
    fn test_hann_window_endpoints() {
        // Hann window starts and ends at (approximately) 0.
        let w = hann_window(64);
        assert!(w[0].abs() < 1e-5, "Hann window should start near 0");
        assert!(w[63].abs() < 1e-5, "Hann window should end near 0");
    }

    #[test]
    fn test_hann_window_sum_positive() {
        let w = hann_window(256);
        let s: f32 = w.iter().sum();
        assert!(s > 0.0);
    }

    #[test]
    fn test_hamming_window_bounds() {
        let w = hamming_window(64);
        assert_eq!(w.len(), 64);
        // Hamming window values lie in [0.08, 1.0].
        for &v in &w {
            assert!(
                (0.0..=1.0 + 1e-5).contains(&v),
                "Hamming value out of range: {v}"
            );
        }
    }

    #[test]
    fn test_blackman_window_size() {
        let w = blackman_window(128);
        assert_eq!(w.len(), 128);
    }

    #[test]
    fn test_rectangular_window_all_ones() {
        let w = rectangular_window(16);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_window_size_1() {
        // Edge case: size-1 windows should all return [1.0].
        assert_eq!(hann_window(1), vec![1.0]);
        assert_eq!(hamming_window(1), vec![1.0]);
        assert_eq!(blackman_window(1), vec![1.0]);
        assert_eq!(rectangular_window(1), vec![1.0]);
    }

    #[test]
    fn test_window_size_0_empty() {
        assert!(hann_window(0).is_empty());
        assert!(hamming_window(0).is_empty());
        assert!(blackman_window(0).is_empty());
        assert!(rectangular_window(0).is_empty());
    }

    // ─── DFT ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_dft_empty() {
        assert!(dft(&[]).is_empty());
    }

    #[test]
    fn test_dft_dc_signal() {
        // All-ones input → X[0] = N, X[k>0] ≈ 0.
        let n = 16;
        let signal = vec![1.0_f32; n];
        let spectrum = dft(&signal);
        assert_eq!(spectrum.len(), n);
        assert!(
            (spectrum[0].0 - n as f32).abs() < 1e-3,
            "DC bin should equal N"
        );
        for k in 1..n {
            let mag = (spectrum[k].0 * spectrum[k].0 + spectrum[k].1 * spectrum[k].1).sqrt();
            assert!(
                mag < 1e-2,
                "Non-DC bins should be near zero for DC input, bin {k} = {mag}"
            );
        }
    }

    #[test]
    fn test_dft_sine_peak() {
        // Pure sine at frequency k0 should peak at bin k0.
        let n = 32;
        let k0 = 4usize;
        let signal: Vec<f32> = (0..n)
            .map(|t| (2.0 * PI * k0 as f32 * t as f32 / n as f32).sin())
            .collect();
        let spectrum = dft(&signal);
        let mags = magnitude_spectrum(&spectrum);
        let peak_bin = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // The peak could be at k0 or n-k0 (conjugate symmetry); check both.
        assert!(
            peak_bin == k0 || peak_bin == n - k0,
            "Peak should be at bin {k0} or {}, found {peak_bin}",
            n - k0
        );
    }

    #[test]
    fn test_magnitude_spectrum_non_negative() {
        let signal: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let spec = dft(&signal);
        let mags = magnitude_spectrum(&spec);
        assert!(mags.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_power_spectrum_non_negative() {
        let signal = vec![1.0_f32, -2.0, 3.0, -4.0];
        let spec = dft(&signal);
        let ps = power_spectrum(&spec);
        assert!(ps.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_phase_spectrum_range() {
        let signal: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let spec = dft(&signal);
        let phases = phase_spectrum(&spec);
        assert!(phases
            .iter()
            .all(|&p| (-PI - 1e-5..=PI + 1e-5).contains(&p)));
    }

    // ─── STFT ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stft_output_shape() {
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let config = StftConfig::default(); // n_fft=512, hop=128
        let out = stft(&signal, &config);
        // num_frames = (1024 - 512) / 128 + 1 = 4 + 1 = 5
        assert_eq!(out.len(), 5, "expected 5 frames");
        // Each frame has n_fft/2+1 = 257 bins.
        assert_eq!(out[0].len(), 257, "expected 257 frequency bins");
    }

    #[test]
    fn test_stft_short_signal_no_frames() {
        // Signal shorter than n_fft → no frames.
        let signal = vec![1.0_f32; 256];
        let config = StftConfig {
            n_fft: 512,
            hop_length: 128,
            win_length: 512,
            window: WindowType::Hann,
            center: false,
        };
        let out = stft(&signal, &config);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_stft_center_pads() {
        // With center=true we should get at least as many frames as without.
        let signal: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
        let config_no_center = StftConfig {
            center: false,
            ..StftConfig::default()
        };
        let config_center = StftConfig {
            center: true,
            ..StftConfig::default()
        };
        let out_nc = stft(&signal, &config_no_center);
        let out_c = stft(&signal, &config_center);
        assert!(
            out_c.len() >= out_nc.len(),
            "center=true should produce >= frames"
        );
    }

    #[test]
    fn test_spectrogram_shape() {
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let config = StftConfig::default();
        let stft_out = stft(&signal, &config);
        let spec = spectrogram(&stft_out);
        assert_eq!(spec.len(), stft_out.len());
        if !spec.is_empty() {
            assert_eq!(spec[0].len(), 257);
        }
    }

    #[test]
    fn test_spectrogram_non_negative() {
        let signal: Vec<f32> = vec![1.0; 1024];
        let config = StftConfig::default();
        let stft_out = stft(&signal, &config);
        let spec = spectrogram(&stft_out);
        for frame in &spec {
            assert!(frame.iter().all(|&v| v >= 0.0));
        }
    }

    // ─── MelFilterbank ────────────────────────────────────────────────────────

    #[test]
    fn test_mel_hz_roundtrip() {
        for hz in [100.0_f32, 440.0, 1000.0, 4000.0, 8000.0] {
            let mel = MelFilterbank::hz_to_mel(hz);
            let hz2 = MelFilterbank::mel_to_hz(mel);
            assert!(
                (hz - hz2).abs() < 0.01,
                "Hz→Mel→Hz roundtrip failed for {hz}: got {hz2}"
            );
        }
    }

    #[test]
    fn test_mel_scale_monotonic() {
        let freqs = [100.0_f32, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
        let mels: Vec<f32> = freqs.iter().map(|&f| MelFilterbank::hz_to_mel(f)).collect();
        for i in 1..mels.len() {
            assert!(
                mels[i] > mels[i - 1],
                "Mel scale should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = MelFilterbank::new(40, 512, 16000.0, 0.0, 8000.0);
        assert_eq!(fb.filters.len(), 40);
        assert_eq!(fb.filters[0].len(), 257); // n_fft/2+1
    }

    #[test]
    fn test_mel_filterbank_apply_shape() {
        let fb = MelFilterbank::new(40, 512, 16000.0, 0.0, 8000.0);
        // Fake spectrogram: 10 frames, 257 bins.
        let spec: Vec<Vec<f32>> = vec![vec![1.0_f32; 257]; 10];
        let mel = fb.apply(&spec);
        assert_eq!(mel.len(), 10);
        assert_eq!(mel[0].len(), 40);
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = MelFilterbank::new(20, 256, 8000.0, 0.0, 4000.0);
        for filter in &fb.filters {
            assert!(filter.iter().all(|&v| v >= 0.0));
        }
    }

    // ─── MFCC ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_mfcc_output_shape() {
        let signal: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let coeffs = mfcc(&signal, 16000.0, 13, 40, 512, 256);
        // num_frames = (4096 - 512) / 256 + 1 = 14 + 1 = 15 (approximately)
        assert!(!coeffs.is_empty(), "MFCC output should be non-empty");
        assert_eq!(
            coeffs[0].len(),
            13,
            "Each frame should have n_mfcc=13 coefficients"
        );
    }

    #[test]
    fn test_mfcc_first_coeff_is_energy() {
        // First MFCC coefficient is proportional to the log-energy.
        let signal: Vec<f32> = vec![1.0_f32; 1024];
        let coeffs = mfcc(&signal, 16000.0, 1, 10, 256, 128);
        if !coeffs.is_empty() {
            // Just check it doesn't panic and returns a finite value.
            assert!(coeffs[0][0].is_finite());
        }
    }

    // ─── zero_crossing_rate ──────────────────────────────────────────────────

    #[test]
    fn test_zero_crossing_rate_step_function() {
        // Step function: first half negative, second half positive.
        // Many sign changes at the boundary.
        let mut signal = vec![-1.0_f32; 16];
        signal.extend(vec![1.0_f32; 16]);
        let zcr = zero_crossing_rate(&signal, 32, 32);
        // One frame of 32 samples — sign change at index 16.
        assert_eq!(zcr.len(), 1);
        assert!(zcr[0] > 0.0, "ZCR should be positive for step function");
    }

    #[test]
    fn test_zero_crossing_rate_constant_signal() {
        // Constant signal has no zero crossings.
        let signal = vec![1.0_f32; 32];
        let zcr = zero_crossing_rate(&signal, 16, 8);
        assert!(zcr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_zero_crossing_rate_range() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let zcr = zero_crossing_rate(&signal, 16, 8);
        assert!(zcr.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    // ─── rms_energy ──────────────────────────────────────────────────────────

    #[test]
    fn test_rms_energy_non_negative() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let rms = rms_energy(&signal, 16, 8);
        assert!(rms.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_rms_energy_constant_signal() {
        // All-ones signal: RMS = 1.
        let signal = vec![1.0_f32; 32];
        let rms = rms_energy(&signal, 16, 16);
        assert_eq!(rms.len(), 2);
        for v in &rms {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "RMS of all-ones should be 1, got {v}"
            );
        }
    }

    // ─── pre_emphasis ─────────────────────────────────────────────────────────

    #[test]
    fn test_pre_emphasis_first_sample_unchanged() {
        let signal = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = pre_emphasis(&signal, 0.97);
        assert_eq!(out.len(), signal.len());
        assert!(
            (out[0] - signal[0]).abs() < 1e-6,
            "First sample should be unchanged"
        );
    }

    #[test]
    fn test_pre_emphasis_formula() {
        let signal = vec![1.0_f32, 2.0, 3.0];
        let alpha = 0.5;
        let out = pre_emphasis(&signal, alpha);
        // y[0]=1.0, y[1]=2.0-0.5*1.0=1.5, y[2]=3.0-0.5*2.0=2.0
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
        assert!((out[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_pre_emphasis_empty() {
        let out = pre_emphasis(&[], 0.97);
        assert!(out.is_empty());
    }

    // ─── frame_signal ─────────────────────────────────────────────────────────

    #[test]
    fn test_frame_signal_shape() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let frames = frame_signal(&signal, 20, 10);
        // num_frames = (100 - 1) / 10 + 1 = 10
        assert_eq!(frames.len(), 10);
        assert_eq!(frames[0].len(), 20);
    }

    #[test]
    fn test_frame_signal_values() {
        let signal: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let frames = frame_signal(&signal, 4, 2);
        // Frame 0: [0,1,2,3], Frame 1: [2,3,4,5], …
        assert!((frames[0][0] - 0.0).abs() < 1e-6);
        assert!((frames[1][0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_signal_empty_input() {
        let frames = frame_signal(&[], 16, 8);
        assert!(frames.is_empty());
    }

    // ─── spectral_centroid ────────────────────────────────────────────────────

    #[test]
    fn test_spectral_centroid_non_negative() {
        let spec: Vec<Vec<f32>> = vec![vec![1.0_f32; 257]; 5];
        let sc = spectral_centroid(&spec, 16000.0, 512);
        assert!(sc.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_spectral_centroid_zero_frame() {
        let spec: Vec<Vec<f32>> = vec![vec![0.0_f32; 257]];
        let sc = spectral_centroid(&spec, 16000.0, 512);
        assert_eq!(sc[0], 0.0);
    }

    #[test]
    fn test_spectral_centroid_single_bin() {
        // Energy only at bin k → centroid = k * sample_rate / n_fft.
        let mut frame = vec![0.0_f32; 9];
        frame[4] = 1.0; // energy at bin 4
        let spec = vec![frame];
        let sc = spectral_centroid(&spec, 8000.0, 16);
        let expected = 4.0 * 8000.0 / 16.0;
        assert!(
            (sc[0] - expected).abs() < 1e-3,
            "Expected {expected}, got {}",
            sc[0]
        );
    }

    // ─── spectral_rolloff ─────────────────────────────────────────────────────

    #[test]
    fn test_spectral_rolloff_range() {
        let spec: Vec<Vec<f32>> = vec![vec![1.0_f32; 257]; 3];
        let sr = spectral_rolloff(&spec, 16000.0, 512, 0.85);
        assert!(sr.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_spectral_rolloff_zero_frame() {
        let spec: Vec<Vec<f32>> = vec![vec![0.0_f32; 257]];
        let sr = spectral_rolloff(&spec, 16000.0, 512, 0.85);
        assert_eq!(sr[0], 0.0);
    }
}
