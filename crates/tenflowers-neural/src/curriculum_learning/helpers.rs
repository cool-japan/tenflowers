//! Helper types and metrics for curriculum learning.
//!
//! Extracted from `mod.rs` to comply with the 2000-line refactoring policy.

// ─────────────────────────────────────────────────────────────────────────────
// ClReport — Curriculum learning evaluation report
// ─────────────────────────────────────────────────────────────────────────────

/// Curriculum learning evaluation report.
#[derive(Debug, Clone)]
pub struct ClReport {
    /// Strategy name.
    pub strategy: String,
    /// Final loss.
    pub final_loss: f64,
    /// Best loss achieved.
    pub best_loss: f64,
    /// Epoch at which best loss was achieved.
    pub best_epoch: usize,
    /// Convergence speed: epochs to reach within 10% of best loss.
    pub convergence_speed: usize,
    /// Average sample utilization rate.
    pub avg_utilization: f64,
    /// Gini coefficient of sample selection (0 = uniform, 1 = concentrated).
    pub selection_gini: f64,
    /// Learning curve (loss per epoch).
    pub learning_curve: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumMetrics — Evaluation metrics and reporting
// ─────────────────────────────────────────────────────────────────────────────

/// Curriculum evaluation metrics and reporting.
#[derive(Debug, Clone)]
pub struct CurriculumMetrics;

impl CurriculumMetrics {
    /// Compute convergence speed: number of epochs to reach within `threshold_fraction`
    /// of the best (minimum) loss.
    pub fn convergence_speed(loss_history: &[f64], threshold_fraction: f64) -> usize {
        if loss_history.is_empty() {
            return 0;
        }
        let best_loss = loss_history.iter().copied().fold(f64::INFINITY, f64::min);
        let target = best_loss * (1.0 + threshold_fraction);

        for (epoch, &loss) in loss_history.iter().enumerate() {
            if loss <= target {
                return epoch;
            }
        }
        loss_history.len()
    }

    /// Compute the Gini coefficient of sample selection counts.
    ///
    /// Returns a value in [0, 1]: 0 = perfectly uniform, 1 = maximally concentrated.
    pub fn selection_gini(selection_counts: &[usize]) -> f64 {
        let n = selection_counts.len();
        if n == 0 {
            return 0.0;
        }

        let mut sorted: Vec<f64> = selection_counts.iter().map(|&c| c as f64).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let total: f64 = sorted.iter().sum();
        if total < 1e-15 {
            return 0.0;
        }

        let mut cumulative = 0.0_f64;
        let mut area_under = 0.0_f64;
        for (i, &val) in sorted.iter().enumerate() {
            cumulative += val;
            area_under += cumulative / total;
            let _ = i;
        }

        let n_f = n as f64;
        // Gini = 1 - 2 * (area_under / n)
        // area_under / n is the average cumulative fraction
        let lorenz_area = area_under / n_f;
        let gini = 1.0 - 2.0 * lorenz_area + 1.0 / n_f;
        gini.clamp(0.0, 1.0)
    }

    /// Compute average sample utilization rate across epochs.
    pub fn avg_utilization(utilization_history: &[f64]) -> f64 {
        if utilization_history.is_empty() {
            return 0.0;
        }
        utilization_history.iter().sum::<f64>() / utilization_history.len() as f64
    }

    /// Compute difficulty distribution per bin.
    ///
    /// # Arguments
    /// * `difficulties` - Normalized difficulty scores.
    /// * `n_bins` - Number of histogram bins.
    ///
    /// Returns bin counts.
    pub fn difficulty_histogram(difficulties: &[f64], n_bins: usize) -> Vec<usize> {
        if n_bins == 0 || difficulties.is_empty() {
            return Vec::new();
        }
        let mut counts = vec![0usize; n_bins];
        let step = 1.0 / n_bins as f64;
        for &d in difficulties {
            let bin = ((d / step) as usize).min(n_bins - 1);
            counts[bin] += 1;
        }
        counts
    }

    /// Compare learning curves: ratio of area under the curve.
    ///
    /// Returns AUC(method) / AUC(baseline). Lower is better for loss curves.
    pub fn learning_curve_ratio(method_losses: &[f64], baseline_losses: &[f64]) -> f64 {
        let method_auc = Self::auc_trapezoid(method_losses);
        let baseline_auc = Self::auc_trapezoid(baseline_losses);
        if baseline_auc.abs() < 1e-15 {
            return 1.0;
        }
        method_auc / baseline_auc
    }

    /// Trapezoidal AUC for a sequence of values.
    fn auc_trapezoid(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return values.first().copied().unwrap_or(0.0);
        }
        let mut auc = 0.0;
        for i in 1..values.len() {
            auc += 0.5 * (values[i - 1] + values[i]);
        }
        auc
    }

    /// Generate a full curriculum report from training history.
    pub fn generate_report(
        strategy_name: &str,
        loss_history: &[f64],
        utilization_history: &[f64],
        selection_counts: &[usize],
    ) -> ClReport {
        let best_loss = loss_history.iter().copied().fold(f64::INFINITY, f64::min);
        let best_epoch = loss_history
            .iter()
            .position(|&l| (l - best_loss).abs() < 1e-15)
            .unwrap_or(0);

        ClReport {
            strategy: strategy_name.to_string(),
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            best_loss,
            best_epoch,
            convergence_speed: Self::convergence_speed(loss_history, 0.1),
            avg_utilization: Self::avg_utilization(utilization_history),
            selection_gini: Self::selection_gini(selection_counts),
            learning_curve: loss_history.to_vec(),
        }
    }

    /// Compare two reports and return the better strategy name.
    pub fn compare_reports(a: &ClReport, b: &ClReport) -> String {
        // Lower best_loss wins; tiebreak by convergence speed
        if a.best_loss < b.best_loss - 1e-10 {
            a.strategy.clone()
        } else if b.best_loss < a.best_loss - 1e-10 {
            b.strategy.clone()
        } else if a.convergence_speed < b.convergence_speed {
            a.strategy.clone()
        } else {
            b.strategy.clone()
        }
    }

    /// Compute the effective dataset size: number of unique samples selected
    /// at least once.
    pub fn effective_dataset_size(selection_counts: &[usize]) -> usize {
        selection_counts.iter().filter(|&&c| c > 0).count()
    }

    /// Compute sample efficiency: performance improvement per sample used.
    pub fn sample_efficiency(loss_history: &[f64], n_samples_per_epoch: &[usize]) -> f64 {
        if loss_history.len() < 2 || n_samples_per_epoch.is_empty() {
            return 0.0;
        }
        let improvement = loss_history[0] - loss_history.last().copied().unwrap_or(loss_history[0]);
        let total_samples: usize = n_samples_per_epoch.iter().sum();
        if total_samples == 0 {
            return 0.0;
        }
        improvement / total_samples as f64
    }
}
