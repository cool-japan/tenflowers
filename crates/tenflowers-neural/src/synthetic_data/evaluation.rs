//! Evaluation & quality metrics for synthetic data.

use super::math::euclidean;

/// Compute correlation matrix for `dim` columns of `data`.
fn correlation_matrix(data: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let n = data.len() as f64;
    let means: Vec<f64> = (0..dim)
        .map(|d| data.iter().map(|r| r[d]).sum::<f64>() / n)
        .collect();
    let stds: Vec<f64> = (0..dim)
        .map(|d| {
            let m = means[d];
            (data.iter().map(|r| (r[d] - m).powi(2)).sum::<f64>() / n)
                .sqrt()
                .max(1e-12)
        })
        .collect();
    let mut corr = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        corr[i * dim + i] = 1.0;
        for j in (i + 1)..dim {
            let c = data
                .iter()
                .map(|r| (r[i] - means[i]) * (r[j] - means[j]))
                .sum::<f64>()
                / (n * stds[i] * stds[j]);
            corr[i * dim + j] = c;
            corr[j * dim + i] = c;
        }
    }
    corr
}

/// Fidelity metrics for synthetic data quality.
pub struct FidelityMetrics;

impl FidelityMetrics {
    /// Marginal coverage: mean `(1 - KS)` across dimensions. Higher = better match.
    pub fn marginal_coverage(real: &[Vec<f64>], synthetic: &[Vec<f64>], n_bins: usize) -> f64 {
        if real.is_empty() || synthetic.is_empty() {
            return 0.0;
        }
        let dim = real[0].len().min(synthetic[0].len());
        if dim == 0 {
            return 0.0;
        }
        let mut total_ks = 0.0_f64;
        for d in 0..dim {
            let mut real_col: Vec<f64> = real.iter().map(|r| r[d]).collect();
            let mut syn_col: Vec<f64> = synthetic.iter().map(|r| r[d]).collect();
            real_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            syn_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min = real_col[0].min(syn_col[0]);
            let max = real_col[real_col.len() - 1].max(syn_col[syn_col.len() - 1]);
            let step = (max - min) / n_bins as f64;
            let ks = if step < 1e-12 {
                0.0
            } else {
                (0..=n_bins)
                    .map(|b| {
                        let t = min + b as f64 * step;
                        let ecdf_r =
                            real_col.partition_point(|&x| x <= t) as f64 / real_col.len() as f64;
                        let ecdf_s =
                            syn_col.partition_point(|&x| x <= t) as f64 / syn_col.len() as f64;
                        (ecdf_r - ecdf_s).abs()
                    })
                    .fold(0.0_f64, f64::max)
            };
            total_ks += ks;
        }
        1.0 - total_ks / dim as f64
    }

    /// Correlation similarity: `1 - mean |Δcorr|`. Higher = better.
    pub fn correlation_similarity(real: &[Vec<f64>], synthetic: &[Vec<f64>]) -> f64 {
        if real.is_empty() || synthetic.is_empty() {
            return 0.0;
        }
        let dim = real[0].len().min(synthetic[0].len());
        if dim < 2 {
            return 1.0;
        }
        let corr_real = correlation_matrix(real, dim);
        let corr_syn = correlation_matrix(synthetic, dim);
        let n_pairs = dim * (dim - 1) / 2;
        if n_pairs == 0 {
            return 1.0;
        }
        let mut total_diff = 0.0_f64;
        for i in 0..dim {
            for j in (i + 1)..dim {
                total_diff += (corr_real[i * dim + j] - corr_syn[i * dim + j]).abs();
            }
        }
        (1.0 - total_diff / n_pairs as f64).clamp(0.0, 1.0)
    }
}

/// Privacy metrics for synthetic data.
pub struct PrivacyMetrics;

impl PrivacyMetrics {
    /// Distance-to-Closest-Record (DCR) ratio.
    ///
    /// Ratio of median synthetic-to-real distance over median within-real distance.
    /// Values near 1.0 indicate good privacy (synthetic is not memorized).
    pub fn nearest_neighbor_distance_ratio(real: &[Vec<f64>], synthetic: &[Vec<f64>]) -> f64 {
        if real.len() < 2 || synthetic.is_empty() {
            return 1.0;
        }
        let mut dcr_syn: Vec<f64> = synthetic
            .iter()
            .map(|s| {
                real.iter()
                    .map(|r| euclidean(s, r))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        dcr_syn.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_syn = dcr_syn[dcr_syn.len() / 2];
        let m = real.len().min(50);
        let mut within_real: Vec<f64> = (0..m)
            .flat_map(|i| {
                (i + 1..m)
                    .map(|j| euclidean(&real[i], &real[j]))
                    .collect::<Vec<_>>()
            })
            .collect();
        within_real.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_real = if within_real.is_empty() {
            1.0
        } else {
            within_real[within_real.len() / 2]
        };
        if median_real < 1e-12 {
            return 1.0;
        }
        (median_syn / median_real).clamp(0.0, 10.0)
    }
}

/// TSTR (Train-on-Synthetic, Test-on-Real) utility evaluator via 1-NN.
pub struct UtilityEvaluator;

impl UtilityEvaluator {
    /// Returns classification accuracy (0..1) on real test set.
    pub fn evaluate(
        synthetic_x: &[Vec<f64>],
        synthetic_y: &[usize],
        real_x: &[Vec<f64>],
        real_y: &[usize],
    ) -> f64 {
        if synthetic_x.is_empty() || real_x.is_empty() {
            return 0.0;
        }
        let correct = real_x
            .iter()
            .zip(real_y.iter())
            .filter(|(rx, &ry)| {
                let nn_label = synthetic_x
                    .iter()
                    .zip(synthetic_y.iter())
                    .map(|(sx, &sy)| (euclidean(rx, sx), sy))
                    .fold((f64::INFINITY, 0_usize), |best, cur| {
                        if cur.0 < best.0 {
                            cur
                        } else {
                            best
                        }
                    })
                    .1;
                nn_label == ry
            })
            .count();
        correct as f64 / real_x.len() as f64
    }
}

/// Coverage and density diversity metrics.
pub struct DiversityMetric;

impl DiversityMetric {
    /// Proportion of real samples within `threshold` of at least one synthetic sample.
    pub fn compute_coverage(real: &[Vec<f64>], synthetic: &[Vec<f64>], threshold: f64) -> f64 {
        if real.is_empty() || synthetic.is_empty() {
            return 0.0;
        }
        let covered = real
            .iter()
            .filter(|r| synthetic.iter().any(|s| euclidean(r, s) <= threshold))
            .count();
        covered as f64 / real.len() as f64
    }

    /// Mean number of synthetic samples within the k-th NN ball of each real sample.
    pub fn compute_density(real: &[Vec<f64>], synthetic: &[Vec<f64>], k: usize) -> f64 {
        if real.is_empty() || synthetic.is_empty() || k == 0 {
            return 0.0;
        }
        let m = real.len().min(50);
        let density_sum: f64 = real[..m]
            .iter()
            .map(|r| {
                let mut dists_real: Vec<f64> = real
                    .iter()
                    .filter(|x| !std::ptr::eq(*x, r))
                    .map(|x| euclidean(r, x))
                    .collect();
                dists_real.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let radius = dists_real
                    .get(k.saturating_sub(1))
                    .cloned()
                    .unwrap_or(1.0)
                    .max(1e-12);
                synthetic
                    .iter()
                    .filter(|s| euclidean(r, s) <= radius)
                    .count() as f64
            })
            .sum();
        density_sum / m as f64
    }
}

/// Trait for data generators used in benchmarking.
pub trait DataGenerator: Send + Sync {
    /// Return a descriptive name.
    fn name(&self) -> &str;
    /// Generate `n` samples using real data as reference.
    fn generate(&self, real: &[Vec<f64>], n: usize, seed: u64) -> Vec<Vec<f64>>;
}

/// Per-generator benchmark result.
#[derive(Debug, Clone)]
pub struct BenchmarkScore {
    /// Generator name.
    pub name: String,
    /// Marginal coverage (higher = better).
    pub marginal_coverage: f64,
    /// Correlation similarity (higher = better).
    pub correlation_similarity: f64,
    /// DCR privacy ratio (near 1.0 = better privacy).
    pub dcr_ratio: f64,
    /// Coverage diversity metric.
    pub coverage: f64,
}

/// Benchmark harness comparing multiple generators against real data.
pub struct SyntheticBenchmark;

impl SyntheticBenchmark {
    /// Run all generators and aggregate quality metrics.
    pub fn benchmark(
        generators: &[Box<dyn DataGenerator>],
        real_data: &[Vec<f64>],
        n_synthetic: usize,
        n_bins: usize,
        threshold: f64,
        seed: u64,
    ) -> Vec<BenchmarkScore> {
        generators
            .iter()
            .map(|gen| {
                let synthetic = gen.generate(real_data, n_synthetic, seed);
                BenchmarkScore {
                    name: gen.name().to_string(),
                    marginal_coverage: FidelityMetrics::marginal_coverage(
                        real_data, &synthetic, n_bins,
                    ),
                    correlation_similarity: FidelityMetrics::correlation_similarity(
                        real_data, &synthetic,
                    ),
                    dcr_ratio: PrivacyMetrics::nearest_neighbor_distance_ratio(
                        real_data, &synthetic,
                    ),
                    coverage: DiversityMetric::compute_coverage(real_data, &synthetic, threshold),
                }
            })
            .collect()
    }
}
