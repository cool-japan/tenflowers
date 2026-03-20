//! Temporal fusion & forecasting: NHitsLayer, PatchTsT, TimeMixer,
//! FrequencyDomainForecaster, ForecastMetrics, ForecastScores.

use super::order_book::{layer_norm_vec, relu, softmax_vec, FinResult, Linear};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// N-HiTS Layer
// ─────────────────────────────────────────────────────────────────────────────

/// N-HiTS layer: multi-rate hierarchical interpolation layer.
///
/// Reference: Challu et al. (2023) — N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.
#[derive(Debug, Clone)]
pub struct NHitsLayer {
    /// Input (lookback) length.
    pub lookback: usize,
    /// Forecast horizon.
    pub horizon: usize,
    /// Backcast basis linear layer.
    backcast_basis: Linear,
    /// Forecast basis linear layer.
    forecast_basis: Linear,
    /// MLP layers.
    mlp1: Linear,
    mlp2: Linear,
}

impl NHitsLayer {
    pub fn new(
        lookback: usize,
        horizon: usize,
        n_basis: usize,
        hidden: usize,
        seed: u64,
    ) -> FinResult<Self> {
        if lookback == 0 || horizon == 0 || n_basis == 0 || hidden == 0 {
            return Err("NHitsLayer: dimensions must be > 0".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            lookback,
            horizon,
            mlp1: Linear::new(lookback, hidden, &mut rng),
            mlp2: Linear::new(hidden, n_basis, &mut rng),
            backcast_basis: Linear::new(n_basis, lookback, &mut rng),
            forecast_basis: Linear::new(n_basis, horizon, &mut rng),
        })
    }

    /// Forward pass.
    ///
    /// * `x`             – input sequence of length `lookback`
    /// * `sampling_rate` – sub-sampling factor (decimation ratio ≥ 1)
    ///
    /// Returns `(backcast, forecast)`.
    pub fn forward(&self, x: &[f32], sampling_rate: usize) -> FinResult<(Vec<f32>, Vec<f32>)> {
        if x.len() < self.lookback {
            return Err(format!("NHitsLayer: need {} inputs", self.lookback));
        }
        // Sub-sample input
        let rate = sampling_rate.max(1);
        let sampled: Vec<f32> = x[..self.lookback].iter().step_by(rate).cloned().collect();
        // Pad/truncate to lookback
        let mut inp = vec![0.0_f32; self.lookback];
        let copy_len = sampled.len().min(self.lookback);
        inp[..copy_len].copy_from_slice(&sampled[..copy_len]);

        let h1: Vec<f32> = self.mlp1.forward(&inp).into_iter().map(relu).collect();
        let basis = self.mlp2.forward(&h1);
        let backcast = self.backcast_basis.forward(&basis);
        let forecast = self.forecast_basis.forward(&basis);
        Ok((backcast, forecast))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PatchTsT
// ─────────────────────────────────────────────────────────────────────────────

/// Patch Time Series Transformer (PatchTST).
///
/// Reference: Nie et al. (2023).
#[derive(Debug, Clone)]
pub struct PatchTsT {
    pub seq_len: usize,
    pub patch_len: usize,
    pub d_model: usize,
    pub horizon: usize,
    patch_embed: Linear,
    attn_q: Linear,
    attn_k: Linear,
    attn_v: Linear,
    attn_out: Linear,
    ffn1: Linear,
    ffn2: Linear,
    head: Linear,
}

impl PatchTsT {
    pub fn new(
        seq_len: usize,
        patch_len: usize,
        d_model: usize,
        horizon: usize,
        seed: u64,
    ) -> FinResult<Self> {
        if seq_len == 0 || patch_len == 0 || d_model == 0 || horizon == 0 {
            return Err("PatchTsT: dimensions must be > 0".to_string());
        }
        let n_patches = (seq_len + patch_len - 1) / patch_len;
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            seq_len,
            patch_len,
            d_model,
            horizon,
            patch_embed: Linear::new(patch_len, d_model, &mut rng),
            attn_q: Linear::new(d_model, d_model, &mut rng),
            attn_k: Linear::new(d_model, d_model, &mut rng),
            attn_v: Linear::new(d_model, d_model, &mut rng),
            attn_out: Linear::new(d_model, d_model, &mut rng),
            ffn1: Linear::new(d_model, d_model * 4, &mut rng),
            ffn2: Linear::new(d_model * 4, d_model, &mut rng),
            head: Linear::new(n_patches * d_model, horizon, &mut rng),
        })
    }

    /// Forward pass over a univariate time series.
    pub fn forward(&self, x: &[f32]) -> FinResult<Vec<f32>> {
        if x.len() < self.seq_len {
            return Err(format!("PatchTsT: need seq_len={} inputs", self.seq_len));
        }
        let patch_len = self.patch_len;
        let n_patches = (self.seq_len + patch_len - 1) / patch_len;

        // Segment into patches
        let mut patches: Vec<Vec<f32>> = Vec::with_capacity(n_patches);
        for p in 0..n_patches {
            let start = p * patch_len;
            let end = (start + patch_len).min(self.seq_len);
            let mut patch = x[start..end].to_vec();
            patch.resize(patch_len, 0.0);
            patches.push(patch);
        }

        // Embed patches
        let mut emb: Vec<Vec<f32>> = patches
            .iter()
            .map(|p| self.patch_embed.forward(p))
            .collect();

        // Single self-attention block
        let scale = (self.d_model as f32).sqrt();
        let q: Vec<Vec<f32>> = emb.iter().map(|e| self.attn_q.forward(e)).collect();
        let k: Vec<Vec<f32>> = emb.iter().map(|e| self.attn_k.forward(e)).collect();
        let v: Vec<Vec<f32>> = emb.iter().map(|e| self.attn_v.forward(e)).collect();

        let mut attn_out: Vec<Vec<f32>> = Vec::with_capacity(n_patches);
        for i in 0..n_patches {
            // Compute attention weights
            let scores: Vec<f32> = (0..n_patches)
                .map(|j| {
                    let dot: f32 = q[i].iter().zip(k[j].iter()).map(|(&a, &b)| a * b).sum();
                    dot / scale
                })
                .collect();
            let weights = softmax_vec(&scores);
            // Weighted sum of values
            let mut out = vec![0.0_f32; self.d_model];
            for j in 0..n_patches {
                for d in 0..self.d_model {
                    out[d] += weights[j] * v[j][d];
                }
            }
            attn_out.push(self.attn_out.forward(&out));
        }

        // Residual + FFN
        for i in 0..n_patches {
            let res: Vec<f32> = emb[i]
                .iter()
                .zip(attn_out[i].iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let normed = layer_norm_vec(&res, 1e-5);
            let ff: Vec<f32> = self.ffn1.forward(&normed).into_iter().map(relu).collect();
            let ff2 = self.ffn2.forward(&ff);
            emb[i] = normed
                .iter()
                .zip(ff2.iter())
                .map(|(&a, &b)| a + b)
                .collect();
        }

        // Flatten and project
        let flat: Vec<f32> = emb.into_iter().flatten().collect();
        Ok(self.head.forward(&flat))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimeMixer
// ─────────────────────────────────────────────────────────────────────────────

/// TSMixer-style mixer: inter-leaved time-mixing and channel-mixing MLPs.
#[derive(Debug, Clone)]
pub struct TimeMixer {
    pub seq_len: usize,
    pub n_channels: usize,
    pub hidden: usize,
    /// Time-mixing MLP (seq_len → seq_len per channel).
    time_mix1: Linear,
    time_mix2: Linear,
    /// Channel-mixing MLP (n_channels → n_channels per time step).
    chan_mix1: Linear,
    chan_mix2: Linear,
    /// Output projection.
    out_proj: Linear,
    pub horizon: usize,
}

impl TimeMixer {
    pub fn new(
        seq_len: usize,
        n_channels: usize,
        hidden: usize,
        horizon: usize,
        seed: u64,
    ) -> FinResult<Self> {
        if seq_len == 0 || n_channels == 0 || hidden == 0 || horizon == 0 {
            return Err("TimeMixer: dimensions must be > 0".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            seq_len,
            n_channels,
            hidden,
            time_mix1: Linear::new(seq_len, hidden, &mut rng),
            time_mix2: Linear::new(hidden, seq_len, &mut rng),
            chan_mix1: Linear::new(n_channels, hidden, &mut rng),
            chan_mix2: Linear::new(hidden, n_channels, &mut rng),
            out_proj: Linear::new(seq_len * n_channels, horizon, &mut rng),
            horizon,
        })
    }

    /// Forward.
    ///
    /// Input: flattened `[seq_len × n_channels]` (channel-last order).
    /// Returns forecast of length `horizon`.
    pub fn forward(&self, x: &[f32]) -> FinResult<Vec<f32>> {
        let expected = self.seq_len * self.n_channels;
        if x.len() < expected {
            return Err(format!(
                "TimeMixer: expected {} inputs, got {}",
                expected,
                x.len()
            ));
        }
        // x shape: [seq_len, n_channels]
        let t = self.seq_len;
        let c = self.n_channels;

        // Time-mixing: for each channel, apply MLP over time
        let mut mixed = x[..expected].to_vec();
        for ch in 0..c {
            let time_slice: Vec<f32> = (0..t).map(|i| mixed[i * c + ch]).collect();
            let normed = layer_norm_vec(&time_slice, 1e-5);
            let h: Vec<f32> = self
                .time_mix1
                .forward(&normed)
                .into_iter()
                .map(relu)
                .collect();
            let out = self.time_mix2.forward(&h);
            // Residual
            for i in 0..t {
                mixed[i * c + ch] = time_slice[i] + out[i];
            }
        }

        // Channel-mixing: for each time step, apply MLP over channels
        for i in 0..t {
            let chan_slice: Vec<f32> = (0..c).map(|ch| mixed[i * c + ch]).collect();
            let normed = layer_norm_vec(&chan_slice, 1e-5);
            let h: Vec<f32> = self
                .chan_mix1
                .forward(&normed)
                .into_iter()
                .map(relu)
                .collect();
            let out = self.chan_mix2.forward(&h);
            for ch in 0..c {
                mixed[i * c + ch] = chan_slice[ch] + out[ch];
            }
        }

        Ok(self.out_proj.forward(&mixed))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frequency Domain Forecaster
// ─────────────────────────────────────────────────────────────────────────────

/// Frequency-domain MLP forecaster (FreTS-style).
///
/// Applies element-wise complex multiplication in the frequency domain,
/// then inverse-transforms back.
#[derive(Debug, Clone)]
pub struct FrequencyDomainForecaster {
    pub seq_len: usize,
    pub horizon: usize,
    /// Real and imaginary weights for frequency-domain MLP.
    freq_w_real: Vec<f32>,
    freq_w_imag: Vec<f32>,
    /// Output projection.
    out_proj: Linear,
}

impl FrequencyDomainForecaster {
    pub fn new(seq_len: usize, horizon: usize, seed: u64) -> FinResult<Self> {
        if seq_len == 0 || horizon == 0 {
            return Err("FrequencyDomainForecaster: dimensions must be > 0".to_string());
        }
        let n_freq = seq_len / 2 + 1;
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0_f32 / n_freq as f32).sqrt();
        let freq_w_real: Vec<f32> = (0..n_freq)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let freq_w_imag: Vec<f32> = (0..n_freq)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let mut rng2 = StdRng::seed_from_u64(seed + 1);
        Ok(Self {
            seq_len,
            horizon,
            freq_w_real,
            freq_w_imag,
            out_proj: Linear::new(seq_len, horizon, &mut rng2),
        })
    }

    /// Forward: apply frequency-domain element-wise MLP + IFFT → output projection.
    pub fn forward(&self, x: &[f32]) -> FinResult<Vec<f32>> {
        if x.len() < self.seq_len {
            return Err(format!(
                "FreqDomainForecaster: need {} inputs",
                self.seq_len
            ));
        }
        let n = self.seq_len;
        // DFT (real input)
        let (re, im) = dft_real(&x[..n]);
        // Element-wise complex multiplication in frequency domain
        let n_freq = n / 2 + 1;
        let mut out_re = vec![0.0_f32; n_freq];
        let mut out_im = vec![0.0_f32; n_freq];
        for k in 0..n_freq {
            // (re + im*i) * (wr + wi*i) = (re*wr - im*wi) + (re*wi + im*wr)*i
            out_re[k] = re[k] * self.freq_w_real[k] - im[k] * self.freq_w_imag[k];
            out_im[k] = re[k] * self.freq_w_imag[k] + im[k] * self.freq_w_real[k];
        }
        // IDFT
        let reconstructed = idft_real(&out_re, &out_im, n);
        // Output projection
        Ok(self.out_proj.forward(&reconstructed))
    }
}

/// Naive O(N²) DFT (real input → complex coefficients up to N/2+1).
fn dft_real(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let n_freq = n / 2 + 1;
    let mut re = vec![0.0_f32; n_freq];
    let mut im = vec![0.0_f32; n_freq];
    for k in 0..n_freq {
        for t in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            re[k] += x[t] * angle.cos();
            im[k] += x[t] * angle.sin();
        }
    }
    (re, im)
}

/// IDFT from one-sided spectrum back to real signal of length n.
fn idft_real(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    let n_freq = n / 2 + 1;
    let mut out = vec![0.0_f32; n];
    for t in 0..n {
        let mut val = 0.0_f32;
        for k in 0..n_freq {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            let mult = if k == 0 || (n % 2 == 0 && k == n_freq - 1) {
                1.0
            } else {
                2.0
            };
            val += mult * (re[k] * angle.cos() - im[k] * angle.sin());
        }
        out[t] = val / n as f32;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Forecast Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated forecast metric scores.
#[derive(Debug, Clone)]
pub struct ForecastScores {
    pub mase: f64,
    pub smape: f64,
    pub mape: f64,
    pub rmse: f64,
    pub mae: f64,
}

/// Forecast evaluation metrics.
#[derive(Debug, Clone)]
pub struct ForecastMetrics;

impl ForecastMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Compute all metrics at once.
    ///
    /// * `y_true`      – ground-truth values
    /// * `y_pred`      – predicted values
    /// * `seasonality` – period used for MASE denominator
    pub fn compute_all(
        &self,
        y_true: &[f64],
        y_pred: &[f64],
        seasonality: usize,
    ) -> FinResult<ForecastScores> {
        let n = y_true.len();
        if n == 0 || y_pred.len() != n {
            return Err("ForecastMetrics: length mismatch or empty".to_string());
        }
        let mae = self.mae(y_true, y_pred);
        let rmse = self.rmse(y_true, y_pred);
        let mape = self.mape(y_true, y_pred);
        let smape = self.smape(y_true, y_pred);
        let mase = self.mase(y_true, y_pred, seasonality);
        Ok(ForecastScores {
            mase,
            smape,
            mape,
            rmse,
            mae,
        })
    }

    pub fn mae(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        y_true
            .iter()
            .zip(y_pred)
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / n as f64
    }

    pub fn rmse(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        let mse = y_true
            .iter()
            .zip(y_pred)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        mse.sqrt()
    }

    pub fn mape(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        let sum = y_true
            .iter()
            .zip(y_pred)
            .filter(|(&a, _)| a.abs() > 1e-8)
            .map(|(&a, &b)| (a - b).abs() / a.abs())
            .sum::<f64>();
        sum / n as f64 * 100.0
    }

    pub fn smape(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        let sum = y_true
            .iter()
            .zip(y_pred)
            .map(|(&a, &b)| {
                let denom = (a.abs() + b.abs()) / 2.0;
                if denom < 1e-8 {
                    0.0
                } else {
                    (a - b).abs() / denom
                }
            })
            .sum::<f64>();
        sum / n as f64 * 100.0
    }

    pub fn mase(&self, y_true: &[f64], y_pred: &[f64], seasonality: usize) -> f64 {
        let n = y_true.len();
        let mae_val = self.mae(y_true, y_pred);
        let m = seasonality.max(1);
        let naive_mae = if n > m {
            let naive_errors: f64 = (m..n).map(|i| (y_true[i] - y_true[i - m]).abs()).sum();
            naive_errors / (n - m) as f64
        } else {
            1.0
        };
        mae_val / naive_mae.max(1e-8)
    }
}

impl Default for ForecastMetrics {
    fn default() -> Self {
        Self::new()
    }
}
