//! Meta-Feature Extraction, Algorithm Configuration Space, Neural Architecture Generation,
//! and AutoML Pipeline components.
//!
//! Implements:
//! - `DatasetMetaFeatures`: statistical meta-features from tabular datasets
//! - `LandmarkingFeatures`: fast landmark algorithm accuracy features
//! - `MetaFeatureNormalizer`: standardize meta-features across datasets
//! - `AlgorithmSelector`: kNN-based algorithm recommendation
//! - `ConfigSpace`: hyperparameter search spaces with conditionals
//! - `SmacOptimizer`: Sequential Model-based Algorithm Configuration
//! - `AlgorithmPerformancePredictor`: meta-learning predictor
//! - `PortfolioSelector`: Greedy portfolio construction (Feurer 2015)
//! - `CellBasedNas`: cell-based neural architecture generation
//! - `EfficientNasPredictor`: NASWOT-style training-free architecture scoring
//! - `ArchitectureEnsemble`: top-k architecture ensemble
//! - `TransferNasFeatures`: transfer learning for NAS
//! - `AutoMlPipeline`: end-to-end AutoML pipeline
//! - `PipelineOptimizer`: joint pipeline configuration optimization
//! - `AutoMlReport`: best pipeline results summary

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use super::{
    backward_sub, cholesky, forward_sub, percentile_sorted, standard_normal_cdf,
    standard_normal_pdf,
};

// ── Internal helpers ──────────────────────────────────────────────────────────

fn vec_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn vec_variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = vec_mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

fn vec_std(v: &[f64]) -> f64 {
    vec_variance(v).sqrt()
}

fn vec_skewness(v: &[f64]) -> f64 {
    if v.len() < 3 {
        return 0.0;
    }
    let m = vec_mean(v);
    let s = vec_std(v).max(1e-12);
    let n = v.len() as f64;
    v.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

fn vec_kurtosis(v: &[f64]) -> f64 {
    if v.len() < 4 {
        return 0.0;
    }
    let m = vec_mean(v);
    let s = vec_std(v).max(1e-12);
    let n = v.len() as f64;
    v.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n - 3.0
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let na = a.iter().map(|ai| ai.powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|bi| bi.powi(2)).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let mx = vec_mean(x);
    let my = vec_mean(y);
    let sx = vec_std(x).max(1e-12);
    let sy = vec_std(y).max(1e-12);
    let n = x.len() as f64;
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mx) * (yi - my))
        .sum::<f64>()
        / (n * sx * sy)
}

// Simple power-iteration PCA returning fraction of variance explained by top-k components.
fn pca_variance_explained(data: &[Vec<f64>], k: usize) -> Vec<f64> {
    if data.is_empty() || k == 0 {
        return vec![0.0; k];
    }
    let n = data.len() as f64;
    let d = data[0].len();
    if d == 0 {
        return vec![0.0; k];
    }
    // Center data
    let means: Vec<f64> = (0..d)
        .map(|j| {
            data.iter()
                .map(|r| r.get(j).cloned().unwrap_or(0.0))
                .sum::<f64>()
                / n
        })
        .collect();
    let centered: Vec<Vec<f64>> = data
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| v - means.get(j).cloned().unwrap_or(0.0))
                .collect()
        })
        .collect();
    // Covariance matrix (d×d)
    let mut cov = vec![vec![0.0_f64; d]; d];
    for row in &centered {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] +=
                    row.get(i).cloned().unwrap_or(0.0) * row.get(j).cloned().unwrap_or(0.0) / n;
            }
        }
    }
    // Total variance
    let total_var: f64 = (0..d).map(|i| cov[i][i]).sum::<f64>().max(1e-12);
    // Power iteration for top-k eigenvectors (deflation)
    let mut eigvals = Vec::new();
    let mut deflated = cov.clone();
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..k.min(d) {
        let mut v: Vec<f64> = (0..d).map(|_| rng.random::<f64>() - 0.5).collect();
        for _iter in 0..50 {
            // Av
            let av: Vec<f64> = (0..d)
                .map(|i| {
                    (0..d)
                        .map(|j| deflated[i][j] * v.get(j).cloned().unwrap_or(0.0))
                        .sum::<f64>()
                })
                .collect();
            let norm = av.iter().map(|x| x.powi(2)).sum::<f64>().sqrt().max(1e-12);
            v = av.iter().map(|x| x / norm).collect();
        }
        // Rayleigh quotient
        let av: Vec<f64> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| deflated[i][j] * v.get(j).cloned().unwrap_or(0.0))
                    .sum::<f64>()
            })
            .collect();
        let lambda: f64 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();
        eigvals.push(lambda.max(0.0));
        // Deflation: A = A - λ v v^T
        for i in 0..d {
            for j in 0..d {
                deflated[i][j] -= lambda * v[i] * v[j];
            }
        }
    }
    eigvals.iter().map(|&lam| lam / total_var).collect()
}

// ── Section 1: Meta-Feature Extraction ────────────────────────────────────────

/// Statistical meta-features extracted from a tabular dataset.
///
/// Captures distributional properties useful for algorithm selection
/// and meta-learning (AutoSklearn / SMAC style).
#[derive(Debug, Clone)]
pub struct DatasetMetaFeatures {
    /// Number of samples in the dataset.
    pub n_samples: usize,
    /// Number of features (columns).
    pub n_features: usize,
    /// Class imbalance ratio: min_class / max_class (1.0 = balanced).
    pub class_imbalance: f64,
    /// Mean of per-feature skewness values.
    pub mean_skewness: f64,
    /// Standard deviation of per-feature skewness.
    pub std_skewness: f64,
    /// Mean of per-feature kurtosis values.
    pub mean_kurtosis: f64,
    /// Fraction of variance explained by the top principal component.
    pub pca_var_pc1: f64,
    /// Fraction of variance explained by the second principal component.
    pub pca_var_pc2: f64,
    /// Mean absolute Pearson correlation between feature pairs.
    pub mean_abs_correlation: f64,
    /// Mean fraction of missing values per feature (0.0 if no NaN).
    pub missing_rate: f64,
}

impl DatasetMetaFeatures {
    /// Construct meta-features from a (n_samples × n_features) data matrix and
    /// an optional class label vector (integers encoded as f64).
    pub fn extract(x: &[Vec<f64>], labels: Option<&[f64]>) -> Result<Self> {
        if x.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "DatasetMetaFeatures::extract".to_string(),
                reason: "dataset must not be empty".to_string(),
                context: None,
            });
        }
        let n_samples = x.len();
        let n_features = x[0].len();
        if n_features == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "DatasetMetaFeatures::extract".to_string(),
                reason: "must have at least one feature".to_string(),
                context: None,
            });
        }

        // Per-feature statistics
        let mut skewness_vals = Vec::with_capacity(n_features);
        let mut kurtosis_vals = Vec::with_capacity(n_features);
        let mut missing_rates = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col: Vec<f64> = x
                .iter()
                .filter_map(|r| r.get(j).cloned())
                .filter(|v| v.is_finite())
                .collect();
            let missing = 1.0 - col.len() as f64 / n_samples as f64;
            missing_rates.push(missing);
            skewness_vals.push(vec_skewness(&col));
            kurtosis_vals.push(vec_kurtosis(&col));
        }
        let mean_skewness = vec_mean(&skewness_vals);
        let std_skewness = vec_std(&skewness_vals);
        let mean_kurtosis = vec_mean(&kurtosis_vals);
        let missing_rate = vec_mean(&missing_rates);

        // PCA variance explained (top-2)
        let pca = pca_variance_explained(x, 2);
        let pca_var_pc1 = pca.first().cloned().unwrap_or(0.0);
        let pca_var_pc2 = pca.get(1).cloned().unwrap_or(0.0);

        // Mean absolute correlation among features (sampled to O(d^2) limit)
        let max_pairs = 200_usize;
        let mut corr_vals = Vec::new();
        let mut count = 0;
        'outer: for i in 0..n_features {
            for j in (i + 1)..n_features {
                if count >= max_pairs {
                    break 'outer;
                }
                let ci: Vec<f64> = x.iter().filter_map(|r| r.get(i).cloned()).collect();
                let cj: Vec<f64> = x.iter().filter_map(|r| r.get(j).cloned()).collect();
                corr_vals.push(pearson_correlation(&ci, &cj).abs());
                count += 1;
            }
        }
        let mean_abs_correlation = if corr_vals.is_empty() {
            0.0
        } else {
            vec_mean(&corr_vals)
        };

        // Class imbalance
        let class_imbalance = if let Some(lbl) = labels {
            if lbl.is_empty() {
                1.0
            } else {
                let mut counts = std::collections::HashMap::new();
                for &l in lbl {
                    let key = l.to_bits();
                    *counts.entry(key).or_insert(0usize) += 1;
                }
                let min_c = counts.values().cloned().min().unwrap_or(1) as f64;
                let max_c = counts.values().cloned().max().unwrap_or(1) as f64;
                min_c / max_c.max(1.0)
            }
        } else {
            1.0
        };

        Ok(Self {
            n_samples,
            n_features,
            class_imbalance,
            mean_skewness,
            std_skewness,
            mean_kurtosis,
            pca_var_pc1,
            pca_var_pc2,
            mean_abs_correlation,
            missing_rate,
        })
    }

    /// Convert to a flat feature vector for use in meta-learning models.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.n_samples as f64,
            self.n_features as f64,
            self.class_imbalance,
            self.mean_skewness,
            self.std_skewness,
            self.mean_kurtosis,
            self.pca_var_pc1,
            self.pca_var_pc2,
            self.mean_abs_correlation,
            self.missing_rate,
        ]
    }
}

/// Fast landmark algorithm results used as meta-features.
///
/// Runs three lightweight baseline learners on a dataset and
/// captures their leave-one-out / hold-out accuracies as features.
#[derive(Debug, Clone)]
pub struct LandmarkingFeatures {
    /// Accuracy of a single decision stump (best threshold on one feature).
    pub decision_stump_acc: f64,
    /// Accuracy of a 1-nearest-neighbour classifier.
    pub knn1_acc: f64,
    /// Accuracy of a single random decision tree (depth 3).
    pub random_tree_acc: f64,
}

impl LandmarkingFeatures {
    /// Compute landmarking features from data matrix and integer class labels.
    ///
    /// Uses a simple 70/30 train/test split (deterministic, no shuffle).
    pub fn compute(x: &[Vec<f64>], labels: &[f64], seed: u64) -> Result<Self> {
        if x.is_empty() || labels.is_empty() || x.len() != labels.len() {
            return Err(TensorError::InvalidArgument {
                operation: "LandmarkingFeatures::compute".to_string(),
                reason: "data and labels must be non-empty and same length".to_string(),
                context: None,
            });
        }
        let n = x.len();
        let split = (n * 7 / 10).max(1).min(n - 1);
        let x_train = &x[..split];
        let y_train = &labels[..split];
        let x_test = &x[split..];
        let y_test = &labels[split..];

        let decision_stump_acc = Self::stump_accuracy(x_train, y_train, x_test, y_test, seed);
        let knn1_acc = Self::knn1_accuracy(x_train, y_train, x_test, y_test);
        let random_tree_acc = Self::random_tree_accuracy(x_train, y_train, x_test, y_test, seed);

        Ok(Self {
            decision_stump_acc,
            knn1_acc,
            random_tree_acc,
        })
    }

    fn stump_accuracy(
        x_tr: &[Vec<f64>],
        y_tr: &[f64],
        x_te: &[Vec<f64>],
        y_te: &[f64],
        _seed: u64,
    ) -> f64 {
        if x_tr.is_empty() || x_te.is_empty() {
            return 0.0;
        }
        let d = x_tr[0].len();
        let mut best_acc = 0.0_f64;
        for feat in 0..d {
            // Try thresholds at unique training values
            let mut vals: Vec<f64> = x_tr.iter().filter_map(|r| r.get(feat).cloned()).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
            for &thresh in &vals {
                // Majority class for each side
                let left_y: Vec<f64> = x_tr
                    .iter()
                    .zip(y_tr.iter())
                    .filter(|(r, _)| r.get(feat).cloned().unwrap_or(f64::NAN) <= thresh)
                    .map(|(_, &y)| y)
                    .collect();
                let right_y: Vec<f64> = x_tr
                    .iter()
                    .zip(y_tr.iter())
                    .filter(|(r, _)| r.get(feat).cloned().unwrap_or(f64::NAN) > thresh)
                    .map(|(_, &y)| y)
                    .collect();
                let left_pred = Self::majority(&left_y);
                let right_pred = Self::majority(&right_y);
                let correct = x_te
                    .iter()
                    .zip(y_te.iter())
                    .filter(|(r, &y)| {
                        let v = r.get(feat).cloned().unwrap_or(f64::NAN);
                        let pred = if v <= thresh { left_pred } else { right_pred };
                        (pred - y).abs() < 1e-6
                    })
                    .count();
                let acc = correct as f64 / y_te.len() as f64;
                if acc > best_acc {
                    best_acc = acc;
                }
            }
        }
        best_acc
    }

    fn majority(labels: &[f64]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }
        let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for &l in labels {
            *counts.entry(l.to_bits()).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| f64::from_bits(k))
            .unwrap_or(0.0)
    }

    fn knn1_accuracy(x_tr: &[Vec<f64>], y_tr: &[f64], x_te: &[Vec<f64>], y_te: &[f64]) -> f64 {
        if x_tr.is_empty() || x_te.is_empty() {
            return 0.0;
        }
        let correct = x_te
            .iter()
            .zip(y_te.iter())
            .filter(|(q, &y)| {
                let pred = x_tr
                    .iter()
                    .zip(y_tr.iter())
                    .map(|(tr, &ty)| {
                        let d: f64 = q
                            .iter()
                            .zip(tr.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();
                        (d, ty)
                    })
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(_, ty)| ty)
                    .unwrap_or(0.0);
                (pred - y).abs() < 1e-6
            })
            .count();
        correct as f64 / y_te.len().max(1) as f64
    }

    fn random_tree_accuracy(
        x_tr: &[Vec<f64>],
        y_tr: &[f64],
        x_te: &[Vec<f64>],
        y_te: &[f64],
        seed: u64,
    ) -> f64 {
        if x_tr.is_empty() || x_te.is_empty() || x_tr[0].is_empty() {
            return 0.0;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let d = x_tr[0].len();
        // Build a depth-3 random tree as a flat array of (feat, thresh, left_pred, right_pred)
        // We use a simple recursive structure stored as a vector of nodes.
        // Each node: (feature, threshold, left_class, right_class) -- leaves store majority class.
        let tree = Self::build_tree(x_tr, y_tr, &mut rng, d, 3);
        let correct = x_te
            .iter()
            .zip(y_te.iter())
            .filter(|(r, &y)| {
                let pred = Self::predict_tree(&tree, r);
                (pred - y).abs() < 1e-6
            })
            .count();
        correct as f64 / y_te.len().max(1) as f64
    }

    fn build_tree(x: &[Vec<f64>], y: &[f64], rng: &mut StdRng, d: usize, depth: usize) -> TreeNode {
        if depth == 0 || x.is_empty() {
            return TreeNode::Leaf(Self::majority(y));
        }
        let feat = rng.random_range(0..d);
        let vals: Vec<f64> = x.iter().filter_map(|r| r.get(feat).cloned()).collect();
        if vals.is_empty() {
            return TreeNode::Leaf(Self::majority(y));
        }
        let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let thresh = min_v + rng.random::<f64>() * (max_v - min_v);
        let (left_x, left_y, right_x, right_y): (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) =
            x.iter().zip(y.iter()).fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |(mut lx, mut ly, mut rx, mut ry), (r, &lab)| {
                    let v = r.get(feat).cloned().unwrap_or(f64::NAN);
                    if v <= thresh {
                        lx.push(r.clone());
                        ly.push(lab);
                    } else {
                        rx.push(r.clone());
                        ry.push(lab);
                    }
                    (lx, ly, rx, ry)
                },
            );
        let left = Box::new(Self::build_tree(&left_x, &left_y, rng, d, depth - 1));
        let right = Box::new(Self::build_tree(&right_x, &right_y, rng, d, depth - 1));
        TreeNode::Split {
            feat,
            thresh,
            left,
            right,
        }
    }

    fn predict_tree(node: &TreeNode, x: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf(c) => *c,
            TreeNode::Split {
                feat,
                thresh,
                left,
                right,
            } => {
                let v = x.get(*feat).cloned().unwrap_or(0.0);
                if v <= *thresh {
                    Self::predict_tree(left, x)
                } else {
                    Self::predict_tree(right, x)
                }
            }
        }
    }

    /// Convert to a flat feature vector for use in meta-learning models.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.decision_stump_acc, self.knn1_acc, self.random_tree_acc]
    }
}

/// Internal decision tree node used by `LandmarkingFeatures`.
#[derive(Debug, Clone)]
enum TreeNode {
    Leaf(f64),
    Split {
        feat: usize,
        thresh: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Standardizes meta-feature vectors across a collection of datasets.
///
/// Uses mean/std normalization fitted from a portfolio of prior datasets.
#[derive(Debug, Clone)]
pub struct MetaFeatureNormalizer {
    /// Fitted mean for each meta-feature dimension.
    pub means: Vec<f64>,
    /// Fitted standard deviation for each meta-feature dimension.
    pub stds: Vec<f64>,
}

impl MetaFeatureNormalizer {
    /// Construct an unfitted normalizer.
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
        }
    }

    /// Fit normalizer from a matrix of meta-feature rows.
    pub fn fit(&mut self, meta_matrix: &[Vec<f64>]) -> Result<()> {
        if meta_matrix.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "MetaFeatureNormalizer::fit".to_string(),
                reason: "meta_matrix must not be empty".to_string(),
                context: None,
            });
        }
        let d = meta_matrix[0].len();
        self.means = (0..d)
            .map(|j| {
                let col: Vec<f64> = meta_matrix
                    .iter()
                    .filter_map(|r| r.get(j).cloned())
                    .collect();
                vec_mean(&col)
            })
            .collect();
        self.stds = (0..d)
            .map(|j| {
                let col: Vec<f64> = meta_matrix
                    .iter()
                    .filter_map(|r| r.get(j).cloned())
                    .collect();
                vec_std(&col).max(1e-12)
            })
            .collect();
        Ok(())
    }

    /// Normalize a single meta-feature vector.
    pub fn transform(&self, meta: &[f64]) -> Result<Vec<f64>> {
        if self.means.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "MetaFeatureNormalizer::transform".to_string(),
                reason: "normalizer not fitted".to_string(),
                context: None,
            });
        }
        Ok(meta
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let m = self.means.get(j).cloned().unwrap_or(0.0);
                let s = self.stds.get(j).cloned().unwrap_or(1.0);
                (v - m) / s
            })
            .collect())
    }
}

impl Default for MetaFeatureNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// kNN-based algorithm selector: recommends algorithms based on meta-feature
/// similarity to past dataset evaluations.
///
/// Uses cosine similarity (k=5 by default).
#[derive(Debug, Clone)]
pub struct AlgorithmSelector {
    /// Number of nearest datasets to consider.
    pub k: usize,
    /// Stored (meta_features, algorithm_name, performance) triples.
    pub portfolio: Vec<(Vec<f64>, String, f64)>,
}

impl AlgorithmSelector {
    /// Construct a new selector with the given neighbourhood size.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            portfolio: Vec::new(),
        }
    }

    /// Register a past dataset evaluation.
    pub fn add_entry(&mut self, meta_features: Vec<f64>, algorithm: String, performance: f64) {
        self.portfolio.push((meta_features, algorithm, performance));
    }

    /// Recommend algorithms for a new dataset described by `meta_features`.
    ///
    /// Returns up to `k` (algorithm_name, score) pairs sorted by predicted performance.
    pub fn recommend(&self, meta_features: &[f64]) -> Vec<(String, f64)> {
        if self.portfolio.is_empty() {
            return Vec::new();
        }
        // Compute cosine similarity to each entry
        let mut similarities: Vec<(usize, f64)> = self
            .portfolio
            .iter()
            .enumerate()
            .map(|(i, (mf, _, _))| (i, cosine_similarity(mf, meta_features)))
            .collect();
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Aggregate performance per algorithm using k nearest
        let mut alg_scores: std::collections::HashMap<String, (f64, f64)> =
            std::collections::HashMap::new();
        for (idx, sim) in similarities.iter().take(self.k) {
            let (_, alg, perf) = &self.portfolio[*idx];
            let entry = alg_scores.entry(alg.clone()).or_insert((0.0, 0.0));
            entry.0 += sim * perf;
            entry.1 += sim;
        }
        let mut ranked: Vec<(String, f64)> = alg_scores
            .into_iter()
            .map(|(alg, (weighted, weight))| {
                let score = if weight < 1e-12 {
                    0.0
                } else {
                    weighted / weight
                };
                (alg, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ── Section 2: Algorithm Configuration Space ──────────────────────────────────

/// A single hyperparameter dimension with its type and bounds.
#[derive(Debug, Clone)]
pub enum MfHpType {
    /// Continuous parameter in [lo, hi].
    Continuous { lo: f64, hi: f64 },
    /// Log-scale continuous in [lo, hi] (sampled in log space).
    LogContinuous { lo: f64, hi: f64 },
    /// Integer in [lo, hi] inclusive.
    Integer { lo: i64, hi: i64 },
    /// Categorical choice from a discrete set.
    Categorical { choices: Vec<f64> },
}

/// Defines the hyperparameter search space for an algorithm.
#[derive(Debug, Clone)]
pub struct ConfigSpace {
    /// List of (name, type) hyperparameter definitions.
    pub params: Vec<(String, MfHpType)>,
    /// Forbidden combinations: list of (param_idx, value) pairs that must NOT co-occur.
    pub forbidden: Vec<Vec<(usize, f64)>>,
}

impl ConfigSpace {
    /// Construct an empty configuration space.
    pub fn new() -> Self {
        Self {
            params: Vec::new(),
            forbidden: Vec::new(),
        }
    }

    /// Add a hyperparameter definition.
    pub fn add_param(&mut self, name: impl Into<String>, hp_type: MfHpType) {
        self.params.push((name.into(), hp_type));
    }

    /// Add a forbidden clause: no configuration may have all listed (param_idx, value) pairs.
    pub fn add_forbidden(&mut self, clause: Vec<(usize, f64)>) {
        self.forbidden.push(clause);
    }

    /// Sample a random configuration, rejecting forbidden combinations (max 100 attempts).
    pub fn sample(&self, rng: &mut StdRng) -> Vec<f64> {
        for _attempt in 0..100 {
            let config: Vec<f64> = self
                .params
                .iter()
                .map(|(_, hp)| match hp {
                    MfHpType::Continuous { lo, hi } => lo + rng.random::<f64>() * (hi - lo),
                    MfHpType::LogContinuous { lo, hi } => {
                        let log_lo = lo.max(1e-10).ln();
                        let log_hi = hi.max(1e-10).ln();
                        (log_lo + rng.random::<f64>() * (log_hi - log_lo)).exp()
                    }
                    MfHpType::Integer { lo, hi } => {
                        let range = (hi - lo + 1).max(1) as usize;
                        (*lo + rng.random_range(0..range) as i64) as f64
                    }
                    MfHpType::Categorical { choices } => {
                        if choices.is_empty() {
                            0.0
                        } else {
                            choices[rng.random_range(0..choices.len())]
                        }
                    }
                })
                .collect();
            if !self.is_forbidden(&config) {
                return config;
            }
        }
        // Fallback: return default (mid-range) config
        self.params
            .iter()
            .map(|(_, hp)| match hp {
                MfHpType::Continuous { lo, hi } => (lo + hi) / 2.0,
                MfHpType::LogContinuous { lo, hi } => (lo * hi).sqrt(),
                MfHpType::Integer { lo, hi } => ((lo + hi) / 2) as f64,
                MfHpType::Categorical { choices } => choices.first().cloned().unwrap_or(0.0),
            })
            .collect()
    }

    /// Check if a configuration violates any forbidden clause.
    pub fn is_forbidden(&self, config: &[f64]) -> bool {
        for clause in &self.forbidden {
            if clause.iter().all(|(idx, val)| {
                config
                    .get(*idx)
                    .map(|v| (v - val).abs() < 1e-8)
                    .unwrap_or(false)
            }) {
                return true;
            }
        }
        false
    }

    /// Return the dimensionality of the configuration space.
    pub fn dim(&self) -> usize {
        self.params.len()
    }
}

impl Default for ConfigSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// SMAC-style optimizer: Random Forest surrogate model + EI acquisition.
///
/// Simplified implementation using a forest of decision stumps
/// as the surrogate, with EI-based selection and random restarts.
#[derive(Debug)]
pub struct SmacOptimizer {
    /// Configuration space being optimized.
    pub space: ConfigSpace,
    /// History of (config, performance) evaluations.
    pub history: Vec<(Vec<f64>, f64)>,
    n_trees: usize,
    n_candidates: usize,
    rng: StdRng,
}

impl Clone for SmacOptimizer {
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            history: self.history.clone(),
            n_trees: self.n_trees,
            n_candidates: self.n_candidates,
            rng: StdRng::seed_from_u64(0x5AAC_0001),
        }
    }
}

impl SmacOptimizer {
    /// Create a new SMAC optimizer for the given space.
    pub fn new(space: ConfigSpace, seed: u64) -> Self {
        Self {
            space,
            history: Vec::new(),
            n_trees: 10,
            n_candidates: 128,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Register an observed (config, performance) pair.
    pub fn observe(&mut self, config: Vec<f64>, performance: f64) {
        self.history.push((config, performance));
    }

    /// Suggest the next configuration to evaluate.
    pub fn suggest(&mut self) -> Result<Vec<f64>> {
        if self.space.dim() == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "SmacOptimizer::suggest".to_string(),
                reason: "configuration space is empty".to_string(),
                context: None,
            });
        }
        if self.history.len() < 5 {
            // Warm-up: pure random exploration
            return Ok(self
                .space
                .sample(&mut StdRng::seed_from_u64(self.rng.random::<u64>())));
        }

        let best_perf = self
            .history
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::NEG_INFINITY, f64::max);

        // Random forest surrogate: ensemble of random stumps
        let forest = self.build_forest();

        let mut best_config = self
            .space
            .sample(&mut StdRng::seed_from_u64(self.rng.random::<u64>()));
        let mut best_ei_val = f64::NEG_INFINITY;

        for _ in 0..self.n_candidates {
            let cfg = self
                .space
                .sample(&mut StdRng::seed_from_u64(self.rng.random::<u64>()));
            let (mean, var) = self.forest_predict(&forest, &cfg);
            let std = var.sqrt().max(1e-10);
            let z = (mean - best_perf) / std;
            let ei = (mean - best_perf) * standard_normal_cdf(z) + std * standard_normal_pdf(z);
            if ei > best_ei_val {
                best_ei_val = ei;
                best_config = cfg;
            }
        }
        Ok(best_config)
    }

    // Build a forest of random decision stumps as surrogate.
    fn build_forest(&mut self) -> Vec<(usize, f64, f64, f64)> {
        let d = self.space.dim();
        let mut forest = Vec::with_capacity(self.n_trees);
        for t in 0..self.n_trees {
            let seed = self.history.len() as u64 * 97 + t as u64 * 31;
            let mut rng = StdRng::seed_from_u64(seed);
            let feat = rng.random_range(0..d.max(1));
            let vals: Vec<f64> = self
                .history
                .iter()
                .filter_map(|(c, _)| c.get(feat).cloned())
                .collect();
            if vals.is_empty() {
                forest.push((feat, 0.0, 0.0, 0.0));
                continue;
            }
            let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let thresh = min_v + rng.random::<f64>() * (max_v - min_v + 1e-12);
            let left_y: Vec<f64> = self
                .history
                .iter()
                .filter(|(c, _)| c.get(feat).cloned().unwrap_or(f64::NAN) <= thresh)
                .map(|(_, p)| *p)
                .collect();
            let right_y: Vec<f64> = self
                .history
                .iter()
                .filter(|(c, _)| c.get(feat).cloned().unwrap_or(f64::NAN) > thresh)
                .map(|(_, p)| *p)
                .collect();
            let lp = vec_mean(&left_y);
            let rp = vec_mean(&right_y);
            forest.push((feat, thresh, lp, rp));
        }
        forest
    }

    fn forest_predict(&self, forest: &[(usize, f64, f64, f64)], cfg: &[f64]) -> (f64, f64) {
        if forest.is_empty() {
            return (0.0, 1.0);
        }
        let preds: Vec<f64> = forest
            .iter()
            .map(|(feat, thresh, lp, rp)| {
                let v = cfg.get(*feat).cloned().unwrap_or(0.0);
                if v <= *thresh {
                    *lp
                } else {
                    *rp
                }
            })
            .collect();
        let mean = vec_mean(&preds);
        let var = vec_variance(&preds).max(1e-6);
        (mean, var)
    }
}

/// Meta-learning predictor for algorithm performance given meta-features + config.
///
/// Uses a two-layer MLP trained on (meta_features ++ config → performance) data.
#[derive(Debug, Clone)]
pub struct AlgorithmPerformancePredictor {
    input_dim: usize,
    hidden_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    /// Training history: (meta_features ++ config, performance).
    pub training_data: Vec<(Vec<f64>, f64)>,
}

impl AlgorithmPerformancePredictor {
    /// Create a new predictor for given input dimension.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "AlgorithmPerformancePredictor::new".to_string(),
                reason: "dims must be > 0".to_string(),
                context: None,
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s1 = (2.0 / input_dim as f64).sqrt();
        let s2 = (2.0 / hidden_dim as f64).sqrt();
        let w1 = (0..hidden_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2 = (0..hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s2)
            .collect();
        Ok(Self {
            input_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2: 0.0,
            training_data: Vec::new(),
        })
    }

    /// Add a training example (meta_features ++ config, performance).
    pub fn add_example(&mut self, features: Vec<f64>, performance: f64) {
        self.training_data.push((features, performance));
    }

    /// Forward pass: predict performance for a given input vector.
    pub fn predict(&self, x: &[f64]) -> Result<f64> {
        if x.len() < self.input_dim {
            return Err(TensorError::ShapeMismatch {
                operation: "AlgorithmPerformancePredictor::predict".to_string(),
                expected: format!("{}", self.input_dim),
                got: format!("{}", x.len()),
                context: None,
            });
        }
        let inp = &x[..self.input_dim];
        let z1: Vec<f64> = self
            .w1
            .iter()
            .zip(self.b1.iter())
            .map(|(row, b)| {
                row.iter()
                    .zip(inp.iter())
                    .map(|(w, xi)| w * xi)
                    .sum::<f64>()
                    + b
            })
            .collect();
        let a1: Vec<f64> = z1.iter().map(|v| v.max(0.0)).collect();
        let out: f64 = self
            .w2
            .iter()
            .zip(a1.iter())
            .map(|(w, a)| w * a)
            .sum::<f64>()
            + self.b2;
        Ok(out)
    }

    /// Simple one-step gradient update (SGD) on a mini-batch from training_data.
    pub fn train_step(&mut self, lr: f64, batch_size: usize, seed: u64) -> Result<f64> {
        if self.training_data.is_empty() {
            return Ok(0.0);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let n = self.training_data.len();
        let bs = batch_size.min(n).max(1);
        let mut loss = 0.0_f64;
        for _ in 0..bs {
            let idx = rng.random_range(0..n);
            let (x, y) = &self.training_data[idx];
            let pred = self.predict(x)?;
            let err = pred - y;
            loss += err.powi(2);
            // Backprop through 2-layer MLP
            let inp = &x[..self.input_dim.min(x.len())];
            let z1: Vec<f64> = self
                .w1
                .iter()
                .zip(self.b1.iter())
                .map(|(row, b)| {
                    row.iter()
                        .zip(inp.iter())
                        .map(|(w, xi)| w * xi)
                        .sum::<f64>()
                        + b
                })
                .collect();
            let a1: Vec<f64> = z1.iter().map(|v| v.max(0.0)).collect();
            // dL/dout = 2*err
            let d_out = 2.0 * err;
            // Update w2, b2
            for (k, w) in self.w2.iter_mut().enumerate() {
                *w -= lr * d_out * a1.get(k).cloned().unwrap_or(0.0);
            }
            self.b2 -= lr * d_out;
            // dL/da1
            let da1: Vec<f64> = self.w2.iter().map(|w| d_out * w).collect();
            // dL/dz1 (ReLU gate)
            let dz1: Vec<f64> = da1
                .iter()
                .zip(z1.iter())
                .map(|(g, z)| if *z > 0.0 { *g } else { 0.0 })
                .collect();
            // Update w1, b1
            for (i, row) in self.w1.iter_mut().enumerate() {
                let dz = dz1.get(i).cloned().unwrap_or(0.0);
                for (j, w) in row.iter_mut().enumerate() {
                    *w -= lr * dz * inp.get(j).cloned().unwrap_or(0.0);
                }
            }
            for (i, b) in self.b1.iter_mut().enumerate() {
                *b -= lr * dz1.get(i).cloned().unwrap_or(0.0);
            }
        }
        Ok(loss / bs as f64)
    }
}

/// Greedy portfolio construction: select algorithms that maximize marginal coverage.
///
/// Feurer et al. (2015): builds a portfolio that covers the most datasets
/// using a greedy set cover approach.
#[derive(Debug, Clone)]
pub struct PortfolioSelector {
    /// Maximum portfolio size.
    pub max_size: usize,
    /// Selected algorithm names.
    pub portfolio: Vec<String>,
}

impl PortfolioSelector {
    /// Create a new portfolio selector with the given size limit.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size: max_size.max(1),
            portfolio: Vec::new(),
        }
    }

    /// Build portfolio from a performance matrix.
    ///
    /// `perf_matrix[i][j]` = performance of algorithm `j` on dataset `i`.
    /// `algorithm_names` must have length == n_algorithms (columns).
    pub fn build(&mut self, perf_matrix: &[Vec<f64>], algorithm_names: &[String]) -> Result<()> {
        if perf_matrix.is_empty() || algorithm_names.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "PortfolioSelector::build".to_string(),
                reason: "perf_matrix and algorithm_names must not be empty".to_string(),
                context: None,
            });
        }
        let n_datasets = perf_matrix.len();
        let n_algs = algorithm_names.len();
        // Current best performance per dataset
        let mut best_perf: Vec<f64> = vec![f64::NEG_INFINITY; n_datasets];
        let mut remaining: std::collections::HashSet<usize> = (0..n_algs).collect();
        self.portfolio.clear();

        for _ in 0..self.max_size {
            if remaining.is_empty() {
                break;
            }
            // Find the algorithm with maximum marginal gain
            let mut best_alg = None;
            let mut best_gain = f64::NEG_INFINITY;
            for &alg_idx in &remaining {
                let gain: f64 = perf_matrix
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        let p = row.get(alg_idx).cloned().unwrap_or(0.0);
                        (p - best_perf[i]).max(0.0)
                    })
                    .sum();
                if gain > best_gain {
                    best_gain = gain;
                    best_alg = Some(alg_idx);
                }
            }
            if let Some(alg_idx) = best_alg {
                // Update best_perf
                for (i, row) in perf_matrix.iter().enumerate() {
                    let p = row.get(alg_idx).cloned().unwrap_or(0.0);
                    if p > best_perf[i] {
                        best_perf[i] = p;
                    }
                }
                self.portfolio.push(
                    algorithm_names
                        .get(alg_idx)
                        .cloned()
                        .unwrap_or_else(|| format!("alg_{}", alg_idx)),
                );
                remaining.remove(&alg_idx);
            } else {
                break;
            }
        }
        Ok(())
    }
}

// ── Section 3: Neural Architecture Generation ─────────────────────────────────

/// Cell operation grammar for cell-based NAS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellOp {
    /// 3×3 separable convolution (simulated as projection).
    SepConv3x3,
    /// 5×5 separable convolution.
    SepConv5x5,
    /// 3×3 dilated convolution.
    DilConv3x3,
    /// Max pooling (simulated as identity with scale).
    MaxPool3x3,
    /// Average pooling.
    AvgPool3x3,
    /// Skip connection.
    Skip,
    /// No operation (zero tensor).
    Zero,
}

impl CellOp {
    /// Number of distinct cell operation types.
    pub fn n_ops() -> usize {
        7
    }

    /// Map integer index to CellOp.
    pub fn from_idx(idx: usize) -> Self {
        match idx % 7 {
            0 => Self::SepConv3x3,
            1 => Self::SepConv5x5,
            2 => Self::DilConv3x3,
            3 => Self::MaxPool3x3,
            4 => Self::AvgPool3x3,
            5 => Self::Skip,
            _ => Self::Zero,
        }
    }
}

/// Cell-based NAS: generates neural architectures as sequences of cells,
/// each cell being a directed acyclic graph over a set of operations.
#[derive(Debug)]
pub struct CellBasedNas {
    /// Number of nodes per cell.
    pub n_nodes: usize,
    /// Number of cells in the architecture.
    pub n_cells: usize,
    rng: StdRng,
}

impl Clone for CellBasedNas {
    fn clone(&self) -> Self {
        Self {
            n_nodes: self.n_nodes,
            n_cells: self.n_cells,
            rng: StdRng::seed_from_u64(0xCE11_BA5E),
        }
    }
}

impl CellBasedNas {
    /// Construct a new cell-based NAS generator.
    pub fn new(n_nodes: usize, n_cells: usize, seed: u64) -> Self {
        Self {
            n_nodes: n_nodes.max(2),
            n_cells: n_cells.max(1),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample a random architecture as a list of (op, input_node) pairs per cell.
    ///
    /// Each cell has `n_nodes` intermediate nodes; each node has 2 input edges.
    /// Returns a flat vector encoding: for each cell, for each node, 2 × (op_idx, src_node).
    pub fn sample_architecture(&mut self) -> Vec<(CellOp, usize)> {
        let mut arch = Vec::new();
        for _ in 0..self.n_cells {
            for node in 2..self.n_nodes {
                // Two input edges per intermediate node
                for _ in 0..2 {
                    let src = self.rng.random_range(0..node);
                    let op = CellOp::from_idx(self.rng.random_range(0..CellOp::n_ops()));
                    arch.push((op, src));
                }
            }
        }
        arch
    }

    /// Encode an architecture as a flat f64 vector for use in predictors.
    pub fn encode_architecture(arch: &[(CellOp, usize)], n_nodes: usize) -> Vec<f64> {
        let mut enc = Vec::with_capacity(arch.len() * (CellOp::n_ops() + n_nodes));
        for (op, src) in arch {
            // One-hot op
            let op_idx = match op {
                CellOp::SepConv3x3 => 0,
                CellOp::SepConv5x5 => 1,
                CellOp::DilConv3x3 => 2,
                CellOp::MaxPool3x3 => 3,
                CellOp::AvgPool3x3 => 4,
                CellOp::Skip => 5,
                CellOp::Zero => 6,
            };
            for i in 0..CellOp::n_ops() {
                enc.push(if i == op_idx { 1.0 } else { 0.0 });
            }
            // One-hot src (capped at n_nodes)
            for i in 0..n_nodes {
                enc.push(if i == *src { 1.0 } else { 0.0 });
            }
        }
        enc
    }
}

/// Training-free NAS predictor using the NASWOT principle (Mellor et al. 2020).
///
/// Scores architectures by the rank of the activation kernel matrix
/// computed over a mini-batch of random inputs — no training required.
#[derive(Debug)]
pub struct EfficientNasPredictor {
    /// Input dimensionality for the simulated activations.
    pub input_dim: usize,
    rng: StdRng,
}

impl Clone for EfficientNasPredictor {
    fn clone(&self) -> Self {
        Self {
            input_dim: self.input_dim,
            rng: StdRng::seed_from_u64(0xEFF1_C1E7),
        }
    }
}

impl EfficientNasPredictor {
    /// Construct a new predictor.
    pub fn new(input_dim: usize, seed: u64) -> Self {
        Self {
            input_dim: input_dim.max(1),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Score an architecture encoding via a simulated activation kernel.
    ///
    /// Higher score indicates more expressive architecture (higher kernel rank).
    /// The encoding is used to seed a linear map that produces binary activation patterns.
    pub fn score(&mut self, arch_encoding: &[f64], n_samples: usize) -> f64 {
        let n = n_samples.clamp(4, 64);
        let d = self.input_dim;
        // Generate random mini-batch
        let batch: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| self.rng.random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        // Simulate activations: project through a weight matrix derived from arch_encoding
        let arch_seed: u64 = arch_encoding.iter().enumerate().fold(0u64, |acc, (i, v)| {
            acc.wrapping_add((v.to_bits()).wrapping_mul(i as u64 + 1))
        });
        let mut wrng = StdRng::seed_from_u64(arch_seed);
        let hidden = d.clamp(8, 32);
        let w: Vec<Vec<f64>> = (0..hidden)
            .map(|_| (0..d).map(|_| wrng.random::<f64>() * 2.0 - 1.0).collect())
            .collect();
        // Binary activation patterns via ReLU sign
        let patterns: Vec<Vec<bool>> = batch
            .iter()
            .map(|x| {
                w.iter()
                    .map(|row| {
                        let z: f64 = row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
                        z > 0.0
                    })
                    .collect()
            })
            .collect();
        // Kernel matrix: normalized Hamming kernel
        let n_p = patterns.len();
        let dh = patterns.first().map(|p| p.len()).unwrap_or(1).max(1);
        let mut k: Vec<Vec<f64>> = vec![vec![0.0; n_p]; n_p];
        for i in 0..n_p {
            for j in 0..n_p {
                let ham = patterns[i]
                    .iter()
                    .zip(patterns[j].iter())
                    .filter(|(a, b)| a != b)
                    .count() as f64;
                k[i][j] = (dh as f64 - ham) / dh as f64;
                if i == j {
                    k[i][j] += 1e-6;
                }
            }
        }
        // Log-determinant via Cholesky
        let l = cholesky(&k, n_p);
        l.iter()
            .enumerate()
            .map(|(i, row)| row[i].max(1e-12).ln())
            .sum::<f64>()
    }
}

/// Ensemble of top-k architectures weighted by validation performance.
#[derive(Debug, Clone)]
pub struct ArchitectureEnsemble {
    /// Maximum number of architectures in the ensemble.
    pub max_k: usize,
    /// Stored (encoding, weight) pairs.
    pub members: Vec<(Vec<f64>, f64)>,
}

impl ArchitectureEnsemble {
    /// Construct a new ensemble with the given capacity.
    pub fn new(max_k: usize) -> Self {
        Self {
            max_k: max_k.max(1),
            members: Vec::new(),
        }
    }

    /// Add an architecture encoding with its validation performance.
    pub fn add(&mut self, encoding: Vec<f64>, val_performance: f64) {
        self.members.push((encoding, val_performance));
        // Keep only top-k by performance
        self.members
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.members.truncate(self.max_k);
    }

    /// Compute ensemble prediction for a given input via softmax-weighted average.
    ///
    /// `predict_fn` maps (encoding, input) → scalar prediction.
    pub fn predict<F>(&self, input: &[f64], predict_fn: F) -> Result<f64>
    where
        F: Fn(&[f64], &[f64]) -> Result<f64>,
    {
        if self.members.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "ArchitectureEnsemble::predict".to_string(),
                reason: "ensemble is empty".to_string(),
                context: None,
            });
        }
        let perfs: Vec<f64> = self.members.iter().map(|(_, p)| *p).collect();
        let max_p = perfs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = perfs.iter().map(|p| (p - max_p).exp()).collect();
        let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-12);
        let weights: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

        let mut out = 0.0_f64;
        for (idx, (enc, _)) in self.members.iter().enumerate() {
            let pred = predict_fn(enc, input)?;
            out += weights[idx] * pred;
        }
        Ok(out)
    }
}

/// Transfer learning for NAS: reuse evaluation history from similar tasks.
///
/// Maintains a registry of (task_meta_features, arch_encoding, performance) triples
/// and extrapolates performance on new tasks via nearest-task weighting.
#[derive(Debug, Clone)]
pub struct TransferNasFeatures {
    /// Registry: (task_meta_features, arch_encoding, performance).
    pub registry: Vec<(Vec<f64>, Vec<f64>, f64)>,
}

impl TransferNasFeatures {
    /// Construct an empty transfer registry.
    pub fn new() -> Self {
        Self {
            registry: Vec::new(),
        }
    }

    /// Register an evaluation from a past task.
    pub fn register(&mut self, task_meta: Vec<f64>, arch_encoding: Vec<f64>, performance: f64) {
        self.registry.push((task_meta, arch_encoding, performance));
    }

    /// Predict performance of an architecture on a new task.
    ///
    /// Uses cosine similarity over task meta-features to weight past evaluations.
    pub fn predict_transfer(&self, task_meta: &[f64], arch_encoding: &[f64]) -> Result<f64> {
        if self.registry.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "TransferNasFeatures::predict_transfer".to_string(),
                reason: "registry is empty".to_string(),
                context: None,
            });
        }
        // Weight by task similarity × architecture similarity
        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for (tmeta, arch_enc, perf) in &self.registry {
            let task_sim = cosine_similarity(tmeta, task_meta).max(0.0);
            let arch_sim = cosine_similarity(arch_enc, arch_encoding).max(0.0);
            let w = task_sim * arch_sim;
            weighted_sum += w * perf;
            weight_sum += w;
        }
        if weight_sum < 1e-12 {
            // Fallback: unweighted mean
            let mean =
                self.registry.iter().map(|(_, _, p)| *p).sum::<f64>() / self.registry.len() as f64;
            return Ok(mean);
        }
        Ok(weighted_sum / weight_sum)
    }
}

impl Default for TransferNasFeatures {
    fn default() -> Self {
        Self::new()
    }
}

// ── Section 4: AutoML Pipeline ────────────────────────────────────────────────

/// Step in an AutoML preprocessing pipeline.
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// Feature selection retaining top-k features.
    FeatureSelection { k: usize },
    /// Standard z-score normalization.
    Normalization,
    /// Log1p transform for positive features.
    LogTransform,
    /// Polynomial degree-2 feature expansion.
    PolynomialExpansion,
}

/// End-to-end AutoML pipeline: preprocessing → algorithm selection → HPO → ensemble.
#[derive(Debug)]
pub struct AutoMlPipeline {
    /// Preprocessing steps applied in order.
    pub steps: Vec<PipelineStep>,
    /// Algorithm to use (name).
    pub selected_algorithm: Option<String>,
    /// Best configuration found.
    pub best_config: Option<Vec<f64>>,
    /// Best cross-validation score achieved.
    pub best_cv_score: f64,
    /// Total number of configurations evaluated.
    pub n_evaluations: usize,
    rng: StdRng,
}

impl Clone for AutoMlPipeline {
    fn clone(&self) -> Self {
        Self {
            steps: self.steps.clone(),
            selected_algorithm: self.selected_algorithm.clone(),
            best_config: self.best_config.clone(),
            best_cv_score: self.best_cv_score,
            n_evaluations: self.n_evaluations,
            rng: StdRng::seed_from_u64(0xA070_F1FE),
        }
    }
}

impl AutoMlPipeline {
    /// Construct a new AutoML pipeline.
    pub fn new(seed: u64) -> Self {
        Self {
            steps: Vec::new(),
            selected_algorithm: None,
            best_config: None,
            best_cv_score: f64::NEG_INFINITY,
            n_evaluations: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Add a preprocessing step to the pipeline.
    pub fn add_step(&mut self, step: PipelineStep) {
        self.steps.push(step);
    }

    /// Apply preprocessing pipeline to a dataset.
    pub fn preprocess(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mut data = x.to_vec();
        for step in &self.steps {
            data = match step {
                PipelineStep::FeatureSelection { k } => {
                    if data.is_empty() || *k == 0 {
                        data
                    } else {
                        let d = data[0].len();
                        let keep = (*k).min(d);
                        data.iter().map(|r| r[..keep].to_vec()).collect()
                    }
                }
                PipelineStep::Normalization => {
                    if data.is_empty() {
                        data
                    } else {
                        let d = data[0].len();
                        let means: Vec<f64> = (0..d)
                            .map(|j| {
                                vec_mean(
                                    &data
                                        .iter()
                                        .filter_map(|r| r.get(j).cloned())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect();
                        let stds: Vec<f64> = (0..d)
                            .map(|j| {
                                vec_std(
                                    &data
                                        .iter()
                                        .filter_map(|r| r.get(j).cloned())
                                        .collect::<Vec<_>>(),
                                )
                                .max(1e-12)
                            })
                            .collect();
                        data.iter()
                            .map(|r| {
                                r.iter()
                                    .enumerate()
                                    .map(|(j, v)| {
                                        (v - means.get(j).cloned().unwrap_or(0.0))
                                            / stds.get(j).cloned().unwrap_or(1.0)
                                    })
                                    .collect()
                            })
                            .collect()
                    }
                }
                PipelineStep::LogTransform => data
                    .iter()
                    .map(|r| r.iter().map(|v| (v.max(0.0) + 1.0).ln()).collect())
                    .collect(),
                PipelineStep::PolynomialExpansion => {
                    if data.is_empty() {
                        data
                    } else {
                        let d = data[0].len();
                        data.iter()
                            .map(|r| {
                                let mut out = r.clone();
                                for i in 0..d {
                                    for j in i..d {
                                        out.push(
                                            r.get(i).cloned().unwrap_or(0.0)
                                                * r.get(j).cloned().unwrap_or(0.0),
                                        );
                                    }
                                }
                                out
                            })
                            .collect()
                    }
                }
            };
        }
        Ok(data)
    }

    /// Simulate a configuration evaluation (returns a mock CV score).
    ///
    /// In a real system this would run k-fold cross-validation.
    /// Here we use the config values to deterministically generate a score
    /// for testing purposes.
    pub fn evaluate_config(&mut self, config: &[f64]) -> f64 {
        self.n_evaluations += 1;
        let base: f64 = config
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, v)| acc + (v * (i + 1) as f64).sin());
        (0.5 + 0.4 * (base / config.len().max(1) as f64).tanh()).clamp(0.0, 1.0)
    }

    /// Run a simple random search HPO loop for the given space and budget.
    pub fn run_random_search(&mut self, space: &ConfigSpace, n_trials: usize) -> Result<()> {
        if space.dim() == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "AutoMlPipeline::run_random_search".to_string(),
                reason: "configuration space is empty".to_string(),
                context: None,
            });
        }
        let mut local_rng = StdRng::seed_from_u64(self.rng.random::<u64>());
        for _ in 0..n_trials {
            let config = space.sample(&mut local_rng);
            let score = self.evaluate_config(&config);
            if score > self.best_cv_score {
                self.best_cv_score = score;
                self.best_config = Some(config);
            }
        }
        Ok(())
    }
}

/// Optimizer that jointly tunes the full pipeline configuration.
#[derive(Debug)]
pub struct PipelineOptimizer {
    /// Configuration space covering pipeline + algorithm hyperparameters.
    pub space: ConfigSpace,
    /// Best pipeline config found.
    pub best_config: Option<Vec<f64>>,
    /// Best score achieved.
    pub best_score: f64,
    /// All evaluated (config, score) pairs.
    pub history: Vec<(Vec<f64>, f64)>,
    rng: StdRng,
}

impl Clone for PipelineOptimizer {
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            best_config: self.best_config.clone(),
            best_score: self.best_score,
            history: self.history.clone(),
            rng: StdRng::seed_from_u64(0xF1FE_0F71),
        }
    }
}

impl PipelineOptimizer {
    /// Construct a pipeline optimizer for the given joint config space.
    pub fn new(space: ConfigSpace, seed: u64) -> Self {
        Self {
            space,
            best_config: None,
            best_score: f64::NEG_INFINITY,
            history: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Optimize the pipeline over `n_trials` random configurations.
    ///
    /// `score_fn` takes a config vector and returns a scalar score (higher = better).
    pub fn optimize<F>(&mut self, n_trials: usize, score_fn: F) -> Result<()>
    where
        F: Fn(&[f64]) -> f64,
    {
        if self.space.dim() == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "PipelineOptimizer::optimize".to_string(),
                reason: "space must have at least one parameter".to_string(),
                context: None,
            });
        }
        // Use SMAC-style GP-EI after warm-up
        let warmup = (n_trials / 4).max(3).min(n_trials);
        let mut local_rng = StdRng::seed_from_u64(self.rng.random::<u64>());

        // Warm-up phase
        for _ in 0..warmup {
            let cfg = self.space.sample(&mut local_rng);
            let score = score_fn(&cfg);
            if score > self.best_score {
                self.best_score = score;
                self.best_config = Some(cfg.clone());
            }
            self.history.push((cfg, score));
        }

        // GP-EI phase (simplified)
        for trial in warmup..n_trials {
            let cfg = if self.history.len() < 5 || trial % 4 == 0 {
                self.space.sample(&mut local_rng)
            } else {
                self.gp_suggest(&mut local_rng)
            };
            let score = score_fn(&cfg);
            if score > self.best_score {
                self.best_score = score;
                self.best_config = Some(cfg.clone());
            }
            self.history.push((cfg, score));
        }
        Ok(())
    }

    fn gp_suggest(&self, rng: &mut StdRng) -> Vec<f64> {
        // GP with RBF kernel on the history
        let n = self.history.len().min(20);
        let xs: Vec<Vec<f64>> = self
            .history
            .iter()
            .rev()
            .take(n)
            .map(|(c, _)| c.clone())
            .collect();
        let ys: Vec<f64> = self.history.iter().rev().take(n).map(|(_, s)| *s).collect();
        let best_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let length_scale = 1.0_f64;
        let noise = 0.01_f64;
        // Build kernel matrix
        let k: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let sq: f64 = xs[i]
                            .iter()
                            .zip(xs[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        let v = (-sq / (2.0 * length_scale * length_scale)).exp();
                        if i == j {
                            v + noise
                        } else {
                            v
                        }
                    })
                    .collect()
            })
            .collect();
        let chol = cholesky(&k, n);
        let ly = forward_sub(&chol, &ys);
        let alpha = backward_sub(&chol, &ly);

        let mut best_cfg = self.space.sample(rng);
        let mut best_ei = f64::NEG_INFINITY;
        for _ in 0..64 {
            let cfg = self.space.sample(rng);
            let ks: Vec<f64> = xs
                .iter()
                .map(|xi| {
                    let sq: f64 = xi
                        .iter()
                        .zip(cfg.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (-sq / (2.0 * length_scale * length_scale)).exp()
                })
                .collect();
            let mean: f64 = ks.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();
            let v = forward_sub(&chol, &ks);
            let var = (1.0 - v.iter().map(|vi| vi.powi(2)).sum::<f64>()).max(0.0);
            let std = var.sqrt().max(1e-10);
            let z = (mean - best_y) / std;
            let ei = (mean - best_y) * standard_normal_cdf(z) + std * standard_normal_pdf(z);
            if ei > best_ei {
                best_ei = ei;
                best_cfg = cfg;
            }
        }
        best_cfg
    }
}

/// Summary report from an AutoML run.
#[derive(Debug, Clone)]
pub struct AutoMlReport {
    /// Name of the best-performing algorithm/pipeline.
    pub best_pipeline: String,
    /// Best hyperparameter configuration found.
    pub best_config: Vec<f64>,
    /// Cross-validation score of the best pipeline.
    pub cv_score: f64,
    /// Wall-clock time in seconds (mock; set by caller).
    pub total_time_secs: f64,
    /// Total number of pipeline evaluations performed.
    pub n_evaluations: usize,
}

impl AutoMlReport {
    /// Construct a new AutoML report.
    pub fn new(
        best_pipeline: impl Into<String>,
        best_config: Vec<f64>,
        cv_score: f64,
        total_time_secs: f64,
        n_evaluations: usize,
    ) -> Self {
        Self {
            best_pipeline: best_pipeline.into(),
            best_config,
            cv_score,
            total_time_secs,
            n_evaluations,
        }
    }

    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "AutoML Report\n  Pipeline: {}\n  CV Score: {:.4}\n  Evaluations: {}\n  Time: {:.1}s",
            self.best_pipeline, self.cv_score, self.n_evaluations, self.total_time_secs
        )
    }
}
