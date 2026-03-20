//! Helper types, proxy evaluation, and metrics for architecture distillation.
//!
//! Extracted from `mod.rs` to comply with the 2000-line refactoring policy.

use super::AdArchEncoding;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// S9: ProxyTask — Cheap architecture evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a proxy evaluation.
#[derive(Debug, Clone)]
pub struct AdProxyResult {
    /// Proxy score (e.g., accuracy on reduced task).
    pub score: f32,
    /// Number of training epochs used.
    pub epochs_used: usize,
    /// Fraction of data used.
    pub data_fraction: f32,
    /// Estimated FLOPs for training.
    pub training_flops: f64,
}

/// Cheap proxy task for architecture evaluation.
///
/// Instead of fully training each candidate, uses reduced training
/// (fewer epochs, less data) to estimate performance.
#[derive(Debug, Clone)]
pub struct ProxyTask {
    /// Default number of proxy training epochs.
    pub default_epochs: usize,
    /// Default fraction of training data to use.
    pub default_data_fraction: f32,
    /// Channel reduction factor (e.g., 0.5 = half channels).
    pub channel_factor: f32,
    /// Layer reduction factor.
    pub layer_factor: f32,
}

impl ProxyTask {
    /// Create a new proxy task configuration.
    pub fn new(
        default_epochs: usize,
        default_data_fraction: f32,
        channel_factor: f32,
        layer_factor: f32,
    ) -> Result<Self> {
        if default_epochs == 0 {
            return Err(TensorError::invalid_argument(
                "ProxyTask: default_epochs must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&default_data_fraction) {
            return Err(TensorError::invalid_argument(format!(
                "ProxyTask: default_data_fraction {} must be in [0, 1]",
                default_data_fraction
            )));
        }
        if channel_factor <= 0.0 || channel_factor > 1.0 {
            return Err(TensorError::invalid_argument(
                "ProxyTask: channel_factor must be in (0, 1]".into(),
            ));
        }
        if layer_factor <= 0.0 || layer_factor > 1.0 {
            return Err(TensorError::invalid_argument(
                "ProxyTask: layer_factor must be in (0, 1]".into(),
            ));
        }

        Ok(Self {
            default_epochs,
            default_data_fraction,
            channel_factor,
            layer_factor,
        })
    }

    /// Evaluate an architecture using proxy task (reduced training).
    ///
    /// `arch` is the architecture encoding, `train_data` is (inputs, targets),
    /// `n_epochs` overrides default_epochs, `data_fraction` overrides default.
    ///
    /// Returns the proxy result with estimated accuracy.
    pub fn evaluate_proxy(
        &self,
        arch: &AdArchEncoding,
        train_data: &[(Vec<f32>, Vec<f32>)],
        n_epochs: Option<usize>,
        data_fraction: Option<f32>,
    ) -> Result<AdProxyResult> {
        if train_data.is_empty() {
            return Err(TensorError::invalid_argument(
                "ProxyTask::evaluate_proxy: train_data must be non-empty".into(),
            ));
        }

        let epochs = n_epochs.unwrap_or(self.default_epochs);
        let frac = data_fraction.unwrap_or(self.default_data_fraction);
        let _n_samples = ((train_data.len() as f32 * frac).ceil() as usize).max(1);

        // Create reduced architecture
        let _reduced_channels =
            ((arch.channels as f32 * self.channel_factor).ceil() as usize).max(1);
        let _reduced_cells = ((arch.n_cells as f32 * self.layer_factor).ceil() as usize).max(1);

        // Simplified training simulation
        let flops_per_sample = arch.estimate_flops(32) as f64; // 32x32 resolution proxy
        let training_flops = flops_per_sample * _n_samples as f64 * epochs as f64;

        // Proxy score based on architecture properties (heuristic)
        // More params generally = more capacity = higher proxy score
        let param_score = (arch.estimate_params() as f32).ln() / 20.0;
        let depth_score = _reduced_cells as f32 / 20.0;
        let width_score = _reduced_channels as f32 / 512.0;

        // Simple proxy: simulate training with diminishing returns
        let mut score = 0.0f32;
        for e in 0..epochs {
            let lr_factor = 1.0 / (1.0 + e as f32 * 0.1);
            let improvement = lr_factor * (param_score + depth_score + width_score) * 0.1;
            score += improvement;
        }
        score = score.clamp(0.0, 1.0);

        // Add some noise based on arch complexity
        let complexity_bonus = (arch.edges.len() as f32 * 0.02).min(0.1);
        score = (score + complexity_bonus).min(1.0);

        Ok(AdProxyResult {
            score,
            epochs_used: epochs,
            data_fraction: frac,
            training_flops,
        })
    }

    /// Compute rank correlation (Kendall tau) between proxy scores and full scores.
    pub fn correlation_analysis(
        proxy_scores: &[f32],
        full_scores: &[f32],
    ) -> Result<AdCorrelation> {
        let n = proxy_scores.len();
        if n != full_scores.len() {
            return Err(TensorError::invalid_argument(
                "ProxyTask::correlation_analysis: score vectors must have same length".into(),
            ));
        }
        if n < 2 {
            return Err(TensorError::invalid_argument(
                "ProxyTask::correlation_analysis: need at least 2 samples".into(),
            ));
        }

        // Kendall tau
        let mut concordant = 0i64;
        let mut discordant = 0i64;
        for i in 0..n {
            for j in (i + 1)..n {
                let proxy_diff = proxy_scores[i] - proxy_scores[j];
                let full_diff = full_scores[i] - full_scores[j];
                let product = proxy_diff * full_diff;
                if product > 0.0 {
                    concordant += 1;
                } else if product < 0.0 {
                    discordant += 1;
                }
                // Ties are ignored
            }
        }

        let total = concordant + discordant;
        let kendall_tau = if total > 0 {
            (concordant - discordant) as f32 / total as f32
        } else {
            0.0
        };

        // Spearman rank correlation
        let proxy_ranks = compute_ranks(proxy_scores);
        let full_ranks = compute_ranks(full_scores);
        let spearman = pearson_correlation(&proxy_ranks, &full_ranks);

        Ok(AdCorrelation {
            kendall_tau,
            spearman,
            n_samples: n,
        })
    }
}

/// Rank correlation results.
#[derive(Debug, Clone)]
pub struct AdCorrelation {
    /// Kendall tau rank correlation.
    pub kendall_tau: f32,
    /// Spearman rank correlation.
    pub spearman: f32,
    /// Number of samples.
    pub n_samples: usize,
}

/// Compute ranks for a slice (1-indexed, average ties).
pub(super) fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j + 1) as f32 / 2.0; // 1-indexed midpoint
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation between two slices.
pub(super) fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S11: AdMetrics — Architecture evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive architecture evaluation metrics.
#[derive(Debug, Clone)]
pub struct AdMetrics;

impl AdMetrics {
    /// Estimate FLOPs (multiply-accumulate operations) for a linear layer.
    pub fn linear_flops(in_dim: usize, out_dim: usize) -> usize {
        2 * in_dim * out_dim
    }

    /// Estimate FLOPs for a conv layer.
    pub fn conv_flops(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        spatial_size: usize,
    ) -> usize {
        2 * in_channels * out_channels * kernel_size * kernel_size * spatial_size * spatial_size
    }

    /// Estimate total parameters for a linear layer (including bias).
    pub fn linear_params(in_dim: usize, out_dim: usize) -> usize {
        in_dim * out_dim + out_dim
    }

    /// Estimate memory footprint in bytes (float32 activations).
    pub fn activation_memory(batch_size: usize, channels: usize, spatial_size: usize) -> usize {
        batch_size * channels * spatial_size * spatial_size * 4 // 4 bytes per f32
    }

    /// Estimate latency as a FLOPs-based proxy (normalized to GFLOPS).
    pub fn latency_proxy(flops: usize, gflops_throughput: f64) -> f64 {
        if gflops_throughput <= 0.0 {
            return 0.0;
        }
        flops as f64 / (gflops_throughput * 1e9)
    }

    /// Compute Kendall tau rank correlation between two score vectors.
    pub fn kendall_tau(scores_a: &[f32], scores_b: &[f32]) -> Result<f32> {
        if scores_a.len() != scores_b.len() {
            return Err(TensorError::invalid_argument(
                "AdMetrics::kendall_tau: score vectors must have same length".into(),
            ));
        }
        let n = scores_a.len();
        if n < 2 {
            return Ok(0.0);
        }

        let mut concordant = 0i64;
        let mut discordant = 0i64;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff_a = scores_a[i] - scores_a[j];
                let diff_b = scores_b[i] - scores_b[j];
                let product = diff_a * diff_b;
                if product > 0.0 {
                    concordant += 1;
                } else if product < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total = concordant + discordant;
        if total == 0 {
            Ok(0.0)
        } else {
            Ok((concordant - discordant) as f32 / total as f32)
        }
    }

    /// Compute architecture efficiency score: accuracy / (FLOPs / budget).
    pub fn efficiency_score(accuracy: f32, flops: usize, flops_budget: usize) -> f32 {
        if flops_budget == 0 || flops == 0 {
            return 0.0;
        }
        accuracy / (flops as f32 / flops_budget as f32)
    }

    /// Compute Pareto optimality mask (accuracy vs FLOPs, both maximize accuracy and minimize FLOPs).
    pub fn pareto_front(accuracies: &[f32], flops: &[usize]) -> Result<Vec<bool>> {
        let n = accuracies.len();
        if n != flops.len() {
            return Err(TensorError::invalid_argument(
                "AdMetrics::pareto_front: accuracies and flops must have same length".into(),
            ));
        }

        let mut is_pareto = vec![true; n];
        for i in 0..n {
            if !is_pareto[i] {
                continue;
            }
            for j in 0..n {
                if i == j || !is_pareto[j] {
                    continue;
                }
                // j dominates i if j has >= accuracy and <= flops with at least one strict
                if accuracies[j] >= accuracies[i]
                    && flops[j] <= flops[i]
                    && (accuracies[j] > accuracies[i] || flops[j] < flops[i])
                {
                    is_pareto[i] = false;
                    break;
                }
            }
        }

        Ok(is_pareto)
    }
}

/// Comprehensive architecture evaluation report.
#[derive(Debug, Clone)]
pub struct AdReport {
    /// Total FLOPs.
    pub flops: usize,
    /// Total parameter count.
    pub params: usize,
    /// Estimated memory footprint in bytes.
    pub memory_bytes: usize,
    /// Estimated latency in seconds.
    pub latency_seconds: f64,
    /// Accuracy (if evaluated).
    pub accuracy: Option<f32>,
    /// Efficiency score (accuracy / normalized FLOPs).
    pub efficiency: Option<f32>,
    /// Architecture description.
    pub description: String,
}

impl AdReport {
    /// Create a report for an architecture encoding.
    pub fn from_arch(arch: &AdArchEncoding, resolution: usize) -> Self {
        let flops = arch.estimate_flops(resolution);
        let params = arch.estimate_params();
        let memory_bytes = AdMetrics::activation_memory(1, arch.channels, resolution);
        let latency_seconds = AdMetrics::latency_proxy(flops, 10.0); // Assume 10 GFLOPS

        Self {
            flops,
            params,
            memory_bytes,
            latency_seconds,
            accuracy: None,
            efficiency: None,
            description: arch.decode(),
        }
    }

    /// Update with accuracy measurement.
    pub fn with_accuracy(mut self, accuracy: f32, flops_budget: usize) -> Self {
        self.accuracy = Some(accuracy);
        self.efficiency = Some(AdMetrics::efficiency_score(
            accuracy,
            self.flops,
            flops_budget,
        ));
        self
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "FLOPs: {:.2}M, Params: {:.2}K, Memory: {:.2}MB, Latency: {:.4}s",
            self.flops as f64 / 1e6,
            self.params as f64 / 1e3,
            self.memory_bytes as f64 / (1024.0 * 1024.0),
            self.latency_seconds,
        );
        if let Some(acc) = self.accuracy {
            s.push_str(&format!(", Accuracy: {:.4}", acc));
        }
        if let Some(eff) = self.efficiency {
            s.push_str(&format!(", Efficiency: {:.4}", eff));
        }
        s
    }
}
