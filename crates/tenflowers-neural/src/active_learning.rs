//! Active Learning query strategies for labeling-budget-efficient model training.
//!
//! This module implements several state-of-the-art active learning algorithms:
//!
//! * **Uncertainty Sampling** — select samples whose predictions are least certain
//!   (max-entropy, least-confident, margin, BALD).
//! * **Core-Set Selection** (Sener & Savarese 2018) — greedy k-center in feature
//!   space to ensure diversity of the labeled set.
//! * **Query-By-Committee** — measure disagreement among an ensemble of models.
//! * **Expected Gradient Length** — proxy for expected parameter update magnitude.
//! * **Max-Cover Sampling** — diverse subset with minimum pairwise distance.
//! * **`ActiveLearningPool`** — bookkeeping utility for labeled / unlabeled indices.

use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that may arise from active-learning operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ActiveLearningError {
    /// The pool (labeled or unlabeled) is empty.
    EmptyPool,
    /// The requested budget exceeds the number of available samples.
    BudgetExceedsPool { budget: usize, available: usize },
    /// A feature vector has the wrong dimensionality.
    DimensionMismatch { expected: usize, found: usize },
    /// A prediction object was constructed with invalid data.
    InvalidPrediction { msg: String },
    /// A committee ensemble has no members.
    EmptyCommittee,
}

impl fmt::Display for ActiveLearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPool => write!(f, "active-learning pool is empty"),
            Self::BudgetExceedsPool { budget, available } => write!(
                f,
                "budget ({budget}) exceeds available unlabeled samples ({available})"
            ),
            Self::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            Self::InvalidPrediction { msg } => write!(f, "invalid prediction: {msg}"),
            Self::EmptyCommittee => write!(f, "committee ensemble has no members"),
        }
    }
}

impl std::error::Error for ActiveLearningError {}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// A single-sample prediction from a classifier.
///
/// `probabilities` must sum to approximately 1.0 and contain at least one
/// element; use [`Prediction::new`] to validate this invariant.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Softmax probabilities `[num_classes]`.
    pub probabilities: Vec<f32>,
    /// Index of the predicted class (argmax of probabilities).
    pub class_idx: usize,
    /// Maximum probability value (`p_max`).
    pub confidence: f32,
}

impl Prediction {
    /// Construct a `Prediction` from a probability vector.
    ///
    /// Returns [`ActiveLearningError::InvalidPrediction`] if the vector is
    /// empty or contains negative values.
    pub fn new(probabilities: Vec<f32>) -> Result<Self, ActiveLearningError> {
        if probabilities.is_empty() {
            return Err(ActiveLearningError::InvalidPrediction {
                msg: "probability vector must not be empty".to_string(),
            });
        }
        for &p in &probabilities {
            if p < 0.0 {
                return Err(ActiveLearningError::InvalidPrediction {
                    msg: format!("probability value {p} is negative"),
                });
            }
        }

        let class_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let confidence = probabilities[class_idx];

        Ok(Self {
            probabilities,
            class_idx,
            confidence,
        })
    }

    /// Shannon entropy: `-Σ p * log2(p + ε)`.
    ///
    /// Higher values indicate greater uncertainty.
    pub fn entropy(&self) -> f32 {
        const EPS: f32 = 1e-10;
        -self
            .probabilities
            .iter()
            .map(|&p| p * (p + EPS).log2())
            .sum::<f32>()
    }

    /// Margin between the two most likely classes: `p_max - p_second_max`.
    ///
    /// Smaller margin → more uncertain. Returns `p_max` if there is only one
    /// class.
    pub fn margin(&self) -> f32 {
        if self.probabilities.len() < 2 {
            return self.confidence;
        }
        let mut sorted = self.probabilities.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted[0] - sorted[1]
    }

    /// Returns the maximum softmax probability (`p_max`).
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Least-confident score: `1 - p_max`.
    ///
    /// Higher values indicate greater uncertainty.
    pub fn least_confident(&self) -> f32 {
        1.0 - self.confidence
    }
}

// ---------------------------------------------------------------------------
// Uncertainty Sampling
// ---------------------------------------------------------------------------

/// Strategy used by [`UncertaintySampler`] to rank unlabeled samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyStrategy {
    /// Highest-entropy first (works for any number of classes).
    MaxEntropy,
    /// Lowest maximum probability first.
    LeastConfident,
    /// Smallest margin between top-2 classes first.
    MarginSampling,
    /// Bayesian Active Learning by Disagreement (requires MC-dropout passes).
    BALD,
}

/// Selects the most uncertain unlabeled samples using a configurable strategy.
#[derive(Debug, Clone)]
pub struct UncertaintySampler {
    /// The uncertainty measure applied to each prediction.
    pub strategy: UncertaintyStrategy,
}

impl UncertaintySampler {
    /// Create a new `UncertaintySampler` with the given `strategy`.
    pub fn new(strategy: UncertaintyStrategy) -> Self {
        Self { strategy }
    }

    /// Compute an uncertainty score per sample (higher = more uncertain).
    ///
    /// For `BALD` this returns all zeros; use [`Self::bald_scores`] instead
    /// when MC-dropout predictions are available.
    pub fn score(&self, predictions: &[Prediction]) -> Vec<f32> {
        predictions
            .iter()
            .map(|p| match self.strategy {
                UncertaintyStrategy::MaxEntropy => p.entropy(),
                UncertaintyStrategy::LeastConfident => p.least_confident(),
                UncertaintyStrategy::MarginSampling => 1.0 - p.margin(),
                UncertaintyStrategy::BALD => 0.0, // needs MC passes
            })
            .collect()
    }

    /// Select the `budget` most uncertain sample indices (0-based into
    /// `predictions`).
    pub fn query(&self, predictions: &[Prediction], budget: usize) -> Vec<usize> {
        let scores = self.score(predictions);
        top_k_indices_desc(&scores, budget)
    }

    /// Compute BALD scores from multiple MC-dropout passes.
    ///
    /// `bald_predictions[i][m][c]` = class-`c` probability for sample `i` in
    /// MC pass `m`.
    ///
    /// BALD = H[y | x, D_train] − E_θ[ H[y | x, θ] ]
    ///
    /// where the first term is the entropy of the mean predictive distribution
    /// and the second is the mean entropy across MC samples.
    pub fn bald_scores(&self, bald_predictions: &[Vec<Vec<f32>>]) -> Vec<f32> {
        bald_predictions
            .iter()
            .map(|mc_passes| {
                let num_passes = mc_passes.len();
                if num_passes == 0 {
                    return 0.0;
                }
                let num_classes = mc_passes[0].len();
                if num_classes == 0 {
                    return 0.0;
                }
                // Mean predictive distribution
                let mut mean_probs = vec![0.0_f32; num_classes];
                for pass in mc_passes {
                    for (c, &p) in pass.iter().enumerate() {
                        if c < num_classes {
                            mean_probs[c] += p;
                        }
                    }
                }
                let inv_n = 1.0 / num_passes as f32;
                for p in mean_probs.iter_mut() {
                    *p *= inv_n;
                }

                // H[ mean ] — entropy of mean distribution
                const EPS: f32 = 1e-10;
                let h_mean: f32 = -mean_probs
                    .iter()
                    .map(|&p| p * (p + EPS).log2())
                    .sum::<f32>();

                // E[ H[pass] ] — expected entropy across passes
                let e_h: f32 = mc_passes
                    .iter()
                    .map(|pass| -pass.iter().map(|&p| p * (p + EPS).log2()).sum::<f32>())
                    .sum::<f32>()
                    * inv_n;

                (h_mean - e_h).max(0.0)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Core-Set Selection
// ---------------------------------------------------------------------------

/// Distance metric used by [`CoreSetSampler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine dissimilarity: `1 − cos(a, b)`.
    Cosine,
    /// Manhattan (L1) distance.
    L1,
}

/// Greedy k-center core-set selection (Sener & Savarese 2018).
///
/// Iteratively adds the unlabeled point that is *farthest* from the current
/// labeled set, minimising the maximum distance any unlabeled point has to
/// its nearest labeled neighbor.
#[derive(Debug, Clone)]
pub struct CoreSetSampler {
    /// Distance metric used when comparing feature vectors.
    pub metric: DistanceMetric,
}

impl CoreSetSampler {
    /// Create a new `CoreSetSampler` with the given `metric`.
    pub fn new(metric: DistanceMetric) -> Self {
        Self { metric }
    }

    /// Greedy k-center query.
    ///
    /// Returns indices into `unlabeled_features` of the `budget` points to
    /// label next.
    ///
    /// Errors if `unlabeled_features` is empty or `budget > unlabeled.len()`.
    pub fn query(
        &self,
        labeled_features: &[Vec<f32>],
        unlabeled_features: &[Vec<f32>],
        budget: usize,
    ) -> Result<Vec<usize>, ActiveLearningError> {
        if unlabeled_features.is_empty() {
            return Err(ActiveLearningError::EmptyPool);
        }
        if budget == 0 {
            return Ok(Vec::new());
        }
        if budget > unlabeled_features.len() {
            return Err(ActiveLearningError::BudgetExceedsPool {
                budget,
                available: unlabeled_features.len(),
            });
        }

        // Validate feature dimensions
        if !labeled_features.is_empty() && !unlabeled_features.is_empty() {
            let d_l = labeled_features[0].len();
            let d_u = unlabeled_features[0].len();
            if d_l != d_u {
                return Err(ActiveLearningError::DimensionMismatch {
                    expected: d_l,
                    found: d_u,
                });
            }
        }

        // Compute min-distance from each unlabeled point to the labeled set.
        // If labeled_features is empty every point starts at +infinity.
        let mut min_dist: Vec<f32> = if labeled_features.is_empty() {
            vec![f32::INFINITY; unlabeled_features.len()]
        } else {
            self.min_distances_to_labeled(labeled_features, unlabeled_features)
        };

        let mut selected: Vec<usize> = Vec::with_capacity(budget);

        for _ in 0..budget {
            // Pick the unlabeled point with maximum min-distance
            let best = min_dist
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected.contains(i))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let best = match best {
                Some(b) => b,
                None => break,
            };

            selected.push(best);

            // Update min_dist with distances to the newly selected point
            let new_center = &unlabeled_features[best];
            for (i, d) in min_dist.iter_mut().enumerate() {
                if selected.contains(&i) {
                    continue;
                }
                let dist = self.dist(new_center, &unlabeled_features[i]);
                if dist < *d {
                    *d = dist;
                }
            }
        }

        Ok(selected)
    }

    /// Pairwise distances matrix `[n_a, n_b]`.
    pub fn pairwise_distances(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
        a.iter()
            .map(|row_a| b.iter().map(|row_b| self.dist(row_a, row_b)).collect())
            .collect()
    }

    /// For each unlabeled point, the minimum distance to any labeled point.
    pub fn min_distances_to_labeled(
        &self,
        labeled: &[Vec<f32>],
        unlabeled: &[Vec<f32>],
    ) -> Vec<f32> {
        unlabeled
            .iter()
            .map(|u| {
                labeled
                    .iter()
                    .map(|l| self.dist(l, u))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect()
    }

    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        match self.metric {
            DistanceMetric::Euclidean => {
                let sq_sum: f32 = (0..len).map(|i| (a[i] - b[i]).powi(2)).sum();
                sq_sum.sqrt()
            }
            DistanceMetric::Cosine => {
                let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
                let norm_a: f32 = (0..len).map(|i| a[i] * a[i]).sum::<f32>().sqrt();
                let norm_b: f32 = (0..len).map(|i| b[i] * b[i]).sum::<f32>().sqrt();
                if norm_a < 1e-12 || norm_b < 1e-12 {
                    1.0
                } else {
                    (1.0 - dot / (norm_a * norm_b)).max(0.0)
                }
            }
            DistanceMetric::L1 => (0..len).map(|i| (a[i] - b[i]).abs()).sum(),
        }
    }
}

// ---------------------------------------------------------------------------
// Query-By-Committee
// ---------------------------------------------------------------------------

/// Measure of disagreement among committee members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisagreementMeasure {
    /// Entropy of committee votes (each member votes for its argmax class).
    VoteEntropy,
    /// Average KL divergence from the committee mean prediction.
    AverageKLDiv,
    /// Maximum pairwise KL divergence among committee members.
    MaxDisagreement,
}

/// Selects samples on which a committee of models most disagrees.
#[derive(Debug, Clone)]
pub struct QueryByCommittee {
    /// The disagreement measure to apply.
    pub disagreement: DisagreementMeasure,
}

impl QueryByCommittee {
    /// Create a new `QueryByCommittee` with the given `disagreement` measure.
    pub fn new(disagreement: DisagreementMeasure) -> Self {
        Self { disagreement }
    }

    /// Compute disagreement scores for each sample.
    ///
    /// `committee_predictions[m][i][c]` = probability for class `c` of sample
    /// `i` according to committee member `m`.
    pub fn score(&self, committee_predictions: &[Vec<Vec<f32>>]) -> Vec<f32> {
        if committee_predictions.is_empty() {
            return Vec::new();
        }
        let num_samples = committee_predictions[0].len();
        let num_members = committee_predictions.len();
        if num_samples == 0 || num_members == 0 {
            return Vec::new();
        }

        match self.disagreement {
            DisagreementMeasure::VoteEntropy => {
                vote_entropy_scores(committee_predictions, num_samples, num_members)
            }
            DisagreementMeasure::AverageKLDiv => {
                average_kl_div_scores(committee_predictions, num_samples, num_members)
            }
            DisagreementMeasure::MaxDisagreement => {
                max_pairwise_kl_scores(committee_predictions, num_samples, num_members)
            }
        }
    }

    /// Select the `budget` most-disagreed-upon sample indices.
    pub fn query(&self, committee_predictions: &[Vec<Vec<f32>>], budget: usize) -> Vec<usize> {
        let scores = self.score(committee_predictions);
        top_k_indices_desc(&scores, budget)
    }
}

fn vote_entropy_scores(
    committee_predictions: &[Vec<Vec<f32>>],
    num_samples: usize,
    num_members: usize,
) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            // Collect argmax votes
            let mut vote_counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for member in committee_predictions {
                if i < member.len() {
                    let vote = argmax_f32(&member[i]);
                    *vote_counts.entry(vote).or_insert(0) += 1;
                }
            }
            // Entropy over vote distribution
            const EPS: f32 = 1e-10;
            let total = num_members as f32;
            -vote_counts
                .values()
                .map(|&c| {
                    let p = c as f32 / total;
                    p * (p + EPS).log2()
                })
                .sum::<f32>()
        })
        .collect()
}

fn average_kl_div_scores(
    committee_predictions: &[Vec<Vec<f32>>],
    num_samples: usize,
    num_members: usize,
) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            // Mean prediction
            let num_classes = committee_predictions[0][i].len();
            let mut mean_pred = vec![0.0_f32; num_classes];
            for member in committee_predictions {
                if i < member.len() {
                    for (c, &p) in member[i].iter().enumerate() {
                        if c < num_classes {
                            mean_pred[c] += p;
                        }
                    }
                }
            }
            let inv_m = 1.0 / num_members as f32;
            for p in mean_pred.iter_mut() {
                *p *= inv_m;
            }
            // Average KL(member || mean)
            const EPS: f32 = 1e-10;
            let total_kl: f32 = committee_predictions
                .iter()
                .map(|member| {
                    if i >= member.len() {
                        return 0.0;
                    }
                    member[i]
                        .iter()
                        .enumerate()
                        .map(|(c, &p)| {
                            let q = if c < mean_pred.len() {
                                mean_pred[c]
                            } else {
                                EPS
                            };
                            if p < EPS {
                                0.0
                            } else {
                                p * (p / (q + EPS)).ln()
                            }
                        })
                        .sum::<f32>()
                })
                .sum();
            (total_kl * inv_m).max(0.0)
        })
        .collect()
}

fn max_pairwise_kl_scores(
    committee_predictions: &[Vec<Vec<f32>>],
    num_samples: usize,
    num_members: usize,
) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let mut max_kl = 0.0_f32;
            for m1 in 0..num_members {
                for m2 in (m1 + 1)..num_members {
                    let p = if i < committee_predictions[m1].len() {
                        &committee_predictions[m1][i]
                    } else {
                        continue;
                    };
                    let q = if i < committee_predictions[m2].len() {
                        &committee_predictions[m2][i]
                    } else {
                        continue;
                    };
                    const EPS: f32 = 1e-10;
                    let kl: f32 = p
                        .iter()
                        .zip(q.iter())
                        .map(|(&pi, &qi)| {
                            if pi < EPS {
                                0.0
                            } else {
                                pi * (pi / (qi + EPS)).ln()
                            }
                        })
                        .sum();
                    if kl > max_kl {
                        max_kl = kl;
                    }
                }
            }
            max_kl
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Expected Gradient Length
// ---------------------------------------------------------------------------

/// Proxy score combining predicted probability with estimated gradient norm.
///
/// `EGL_i = Σ_c  p_c * ||∇L_c||`
///
/// In practice `gradient_norms[i]` is a single pre-computed estimate of
/// `||∇L||` for sample `i`, so we weight it by `(1 - p_max)` to approximate
/// how much the loss gradient would change if the predicted label were wrong.
pub fn expected_gradient_length(predictions: &[Prediction], gradient_norms: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(gradient_norms.iter())
        .map(|(pred, &g_norm)| {
            // weight by expected loss magnitude
            let uncertainty = pred.entropy();
            uncertainty * g_norm
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Diversity-based sampling (Max-Cover)
// ---------------------------------------------------------------------------

/// Select a diverse subset by ensuring all selected points are at least
/// `min_distance` apart (greedy sequential max-cover).
///
/// Returns up to `budget` indices into `unlabeled_features`.  The first
/// selected point is the one with the largest L2 norm (acts as a
/// deterministic seed).
pub fn max_cover_sample(
    unlabeled_features: &[Vec<f32>],
    budget: usize,
    min_distance: f32,
) -> Vec<usize> {
    if unlabeled_features.is_empty() || budget == 0 {
        return Vec::new();
    }

    let euclidean_dist = |a: &[f32], b: &[f32]| -> f32 {
        let len = a.len().min(b.len());
        (0..len).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>().sqrt()
    };

    // Seed: point with largest L2 norm (reproducible, no randomness needed)
    let seed = unlabeled_features
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut selected = vec![seed];

    for i in 0..unlabeled_features.len() {
        if selected.len() >= budget {
            break;
        }
        if selected.contains(&i) {
            continue;
        }
        // Check that this point is at least min_distance from all selected points
        let far_enough = selected.iter().all(|&s| {
            euclidean_dist(&unlabeled_features[i], &unlabeled_features[s]) >= min_distance
        });
        if far_enough {
            selected.push(i);
        }
    }

    selected
}

// ---------------------------------------------------------------------------
// Active Learning Pool
// ---------------------------------------------------------------------------

/// Tracks which samples are labeled vs. unlabeled during an active learning
/// loop.
#[derive(Debug, Clone)]
pub struct ActiveLearningPool {
    /// Indices (into the full dataset) that are currently labeled.
    pub labeled_indices: Vec<usize>,
    /// Indices that are currently unlabeled.
    pub unlabeled_indices: Vec<usize>,
    /// Total number of samples in the dataset.
    pub total_size: usize,
}

impl ActiveLearningPool {
    /// Create a pool for a dataset of `total_size` samples where
    /// `initial_labeled` are already labeled.
    ///
    /// Any index in `initial_labeled` that is `>= total_size` is ignored.
    pub fn new(total_size: usize, initial_labeled: Vec<usize>) -> Self {
        let labeled_set: std::collections::HashSet<usize> = initial_labeled
            .into_iter()
            .filter(|&i| i < total_size)
            .collect();
        let labeled_indices: Vec<usize> = {
            let mut v: Vec<usize> = labeled_set.iter().copied().collect();
            v.sort_unstable();
            v
        };
        let unlabeled_indices: Vec<usize> = (0..total_size)
            .filter(|i| !labeled_set.contains(i))
            .collect();

        Self {
            labeled_indices,
            unlabeled_indices,
            total_size,
        }
    }

    /// Move `new_indices` (indices into the unlabeled pool) into the labeled
    /// pool.
    ///
    /// Indices that are already labeled or out of range are silently skipped.
    pub fn add_labeled(&mut self, new_indices: Vec<usize>) {
        // Convert from "index into unlabeled_indices" to actual dataset indices
        // (support both interpretations; here we treat them as actual dataset indices)
        let mut to_label: std::collections::HashSet<usize> = new_indices.into_iter().collect();

        // Retain only those that are actually in unlabeled_indices
        let (to_move, remaining): (Vec<usize>, Vec<usize>) = self
            .unlabeled_indices
            .drain(..)
            .partition(|i| to_label.remove(i));

        self.labeled_indices.extend(to_move);
        self.labeled_indices.sort_unstable();
        self.labeled_indices.dedup();
        self.unlabeled_indices = remaining;
    }

    /// Number of currently labeled samples.
    pub fn labeled_count(&self) -> usize {
        self.labeled_indices.len()
    }

    /// Number of currently unlabeled samples.
    pub fn unlabeled_count(&self) -> usize {
        self.unlabeled_indices.len()
    }

    /// Fraction of the dataset that is labeled: `labeled / total`.
    pub fn labeling_ratio(&self) -> f32 {
        if self.total_size == 0 {
            return 0.0;
        }
        self.labeled_indices.len() as f32 / self.total_size as f32
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return the indices of the top-`k` largest values in `scores` (desc order).
fn top_k_indices_desc(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Return the argmax index of a float slice.
fn argmax_f32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Prediction tests ---

    #[test]
    fn test_prediction_entropy_uniform() {
        // Uniform distribution over 4 classes should have maximum entropy = log2(4) = 2
        let pred = Prediction::new(vec![0.25, 0.25, 0.25, 0.25]).expect("prediction failed");
        let ent = pred.entropy();
        assert!((ent - 2.0).abs() < 0.01, "expected ~2.0, got {ent}");
    }

    #[test]
    fn test_prediction_entropy_certain() {
        // Certain prediction should have near-zero entropy
        let pred = Prediction::new(vec![1.0, 0.0, 0.0]).expect("prediction failed");
        assert!(pred.entropy() < 0.01);
    }

    #[test]
    fn test_prediction_margin_binary() {
        let pred = Prediction::new(vec![0.6, 0.4]).expect("prediction failed");
        let m = pred.margin();
        assert!((m - 0.2).abs() < 1e-5, "expected 0.2, got {m}");
    }

    #[test]
    fn test_prediction_margin_single_class() {
        let pred = Prediction::new(vec![1.0]).expect("prediction failed");
        let m = pred.margin();
        assert!((m - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prediction_confidence_and_class_idx() {
        let pred = Prediction::new(vec![0.1, 0.7, 0.2]).expect("prediction failed");
        assert_eq!(pred.class_idx, 1);
        assert!((pred.confidence() - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_prediction_least_confident() {
        let pred = Prediction::new(vec![0.1, 0.7, 0.2]).expect("prediction failed");
        assert!((pred.least_confident() - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_prediction_invalid_empty() {
        let res = Prediction::new(vec![]);
        assert!(matches!(
            res,
            Err(ActiveLearningError::InvalidPrediction { .. })
        ));
    }

    #[test]
    fn test_prediction_invalid_negative() {
        let res = Prediction::new(vec![0.5, -0.1, 0.6]);
        assert!(matches!(
            res,
            Err(ActiveLearningError::InvalidPrediction { .. })
        ));
    }

    // --- UncertaintySampler tests ---

    fn make_preds() -> Vec<Prediction> {
        vec![
            Prediction::new(vec![0.9, 0.1]).expect("prediction failed"), // low entropy, high confidence
            Prediction::new(vec![0.5, 0.5]).expect("prediction failed"), // max entropy
            Prediction::new(vec![0.8, 0.2]).expect("prediction failed"), // medium
        ]
    }

    #[test]
    fn test_uncertainty_sampler_max_entropy_selects_highest() {
        let sampler = UncertaintySampler::new(UncertaintyStrategy::MaxEntropy);
        let preds = make_preds();
        let selected = sampler.query(&preds, 1);
        assert_eq!(selected, vec![1], "should select the most uncertain sample");
    }

    #[test]
    fn test_uncertainty_sampler_least_confident() {
        let sampler = UncertaintySampler::new(UncertaintyStrategy::LeastConfident);
        let preds = make_preds();
        let selected = sampler.query(&preds, 1);
        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn test_uncertainty_sampler_margin_selects_smallest_margin() {
        let sampler = UncertaintySampler::new(UncertaintyStrategy::MarginSampling);
        let preds = make_preds();
        let selected = sampler.query(&preds, 1);
        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn test_uncertainty_sampler_query_budget_count() {
        let sampler = UncertaintySampler::new(UncertaintyStrategy::MaxEntropy);
        let preds = make_preds();
        let selected = sampler.query(&preds, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_uncertainty_sampler_budget_exceeds_returns_all() {
        let sampler = UncertaintySampler::new(UncertaintyStrategy::MaxEntropy);
        let preds = make_preds();
        let selected = sampler.query(&preds, 10);
        assert_eq!(selected.len(), preds.len());
    }

    #[test]
    fn test_bald_score_variance() {
        let sampler = UncertaintySampler::new(UncertaintyStrategy::BALD);
        // Sample 0: disagreeing MC passes → high BALD
        // Sample 1: agreeing MC passes → low BALD
        let mc = vec![
            // Sample 0: disagreeing
            vec![vec![0.9_f32, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            // Sample 1: all agree
            vec![vec![0.95_f32, 0.05], vec![0.95, 0.05], vec![0.95, 0.05]],
        ];
        let scores = sampler.bald_scores(&mc);
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "disagreeing sample should have higher BALD: {} vs {}",
            scores[0],
            scores[1]
        );
    }

    // --- CoreSetSampler tests ---

    #[test]
    fn test_coreset_query_selects_diverse_points() {
        let sampler = CoreSetSampler::new(DistanceMetric::Euclidean);
        // Three unlabeled points; one labeled at origin
        let labeled = vec![vec![0.0_f32, 0.0]];
        let unlabeled = vec![
            vec![1.0_f32, 0.0], // near labeled
            vec![10.0, 0.0],    // far from labeled
            vec![10.0, 10.0],   // even farther
        ];
        let selected = sampler
            .query(&labeled, &unlabeled, 2)
            .expect("query failed");
        assert_eq!(selected.len(), 2);
        // First pick must be one of the far points
        assert!(selected.contains(&1) || selected.contains(&2));
    }

    #[test]
    fn test_coreset_pairwise_distances_shape() {
        let sampler = CoreSetSampler::new(DistanceMetric::Euclidean);
        let a = vec![vec![0.0_f32; 4]; 3];
        let b = vec![vec![1.0_f32; 4]; 5];
        let dist = sampler.pairwise_distances(&a, &b);
        assert_eq!(dist.len(), 3);
        assert_eq!(dist[0].len(), 5);
    }

    #[test]
    fn test_coreset_euclidean_known_distance() {
        let sampler = CoreSetSampler::new(DistanceMetric::Euclidean);
        let a = vec![vec![0.0_f32, 0.0]];
        let b = vec![vec![3.0_f32, 4.0]];
        let dist = sampler.pairwise_distances(&a, &b);
        assert!((dist[0][0] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_coreset_cosine_orthogonal() {
        let sampler = CoreSetSampler::new(DistanceMetric::Cosine);
        let a = vec![vec![1.0_f32, 0.0]];
        let b = vec![vec![0.0_f32, 1.0]];
        let dist = sampler.pairwise_distances(&a, &b);
        assert!(
            (dist[0][0] - 1.0).abs() < 1e-4,
            "orthogonal cosine dist = 1"
        );
    }

    #[test]
    fn test_coreset_l1_distance() {
        let sampler = CoreSetSampler::new(DistanceMetric::L1);
        let a = vec![vec![0.0_f32, 0.0]];
        let b = vec![vec![1.0_f32, 2.0]];
        let dist = sampler.pairwise_distances(&a, &b);
        assert!((dist[0][0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_coreset_budget_exceeds_unlabeled_error() {
        let sampler = CoreSetSampler::new(DistanceMetric::Euclidean);
        let labeled = vec![vec![0.0_f32; 2]];
        let unlabeled = vec![vec![1.0_f32; 2]];
        let res = sampler.query(&labeled, &unlabeled, 5);
        assert!(matches!(
            res,
            Err(ActiveLearningError::BudgetExceedsPool { .. })
        ));
    }

    #[test]
    fn test_coreset_empty_unlabeled_error() {
        let sampler = CoreSetSampler::new(DistanceMetric::Euclidean);
        let res = sampler.query(&[], &[], 1);
        assert!(matches!(res, Err(ActiveLearningError::EmptyPool)));
    }

    // --- QueryByCommittee tests ---

    #[test]
    fn test_qbc_vote_entropy_score() {
        let qbc = QueryByCommittee::new(DisagreementMeasure::VoteEntropy);
        // Two members; sample 0 both vote class 0; sample 1 one votes 0 one votes 1
        let committee = vec![
            vec![vec![0.9_f32, 0.1], vec![0.6, 0.4]],
            vec![vec![0.8_f32, 0.2], vec![0.3, 0.7]],
        ];
        let scores = qbc.score(&committee);
        assert_eq!(scores.len(), 2);
        // sample 1 has split vote → higher entropy
        assert!(
            scores[1] > scores[0],
            "split vote should yield higher entropy: {} vs {}",
            scores[1],
            scores[0]
        );
    }

    #[test]
    fn test_qbc_average_kl_div_score() {
        let qbc = QueryByCommittee::new(DisagreementMeasure::AverageKLDiv);
        // Unanimous → near-zero KL; split → positive KL
        let committee = vec![
            vec![vec![0.9_f32, 0.1], vec![0.9, 0.1]],
            vec![vec![0.9_f32, 0.1], vec![0.1, 0.9]],
        ];
        let scores = qbc.score(&committee);
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] < scores[1],
            "unanimous < split: {} vs {}",
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn test_qbc_query_budget() {
        let qbc = QueryByCommittee::new(DisagreementMeasure::VoteEntropy);
        let committee = vec![
            vec![vec![0.9_f32, 0.1], vec![0.5, 0.5], vec![0.8, 0.2]],
            vec![vec![0.8_f32, 0.2], vec![0.4, 0.6], vec![0.9, 0.1]],
        ];
        let selected = qbc.query(&committee, 2);
        assert_eq!(selected.len(), 2);
    }

    // --- max_cover_sample tests ---

    #[test]
    fn test_max_cover_minimum_distance() {
        let features = vec![
            vec![0.0_f32, 0.0],
            vec![0.1_f32, 0.0],  // too close to [0,0]
            vec![5.0_f32, 0.0],  // far enough
            vec![5.1_f32, 0.0],  // too close to [5,0]
            vec![10.0_f32, 0.0], // far enough
        ];
        let selected = max_cover_sample(&features, 10, 1.0);
        // Check pairwise minimum distances
        for i in 0..selected.len() {
            for j in (i + 1)..selected.len() {
                let a = &features[selected[i]];
                let b = &features[selected[j]];
                let d: f32 = (0..a.len())
                    .map(|k| (a[k] - b[k]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    d >= 1.0 - 1e-5,
                    "pair ({},{}) distance {d} < min_distance 1.0",
                    selected[i],
                    selected[j]
                );
            }
        }
    }

    #[test]
    fn test_max_cover_empty_input() {
        let selected = max_cover_sample(&[], 5, 1.0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_max_cover_zero_budget() {
        let features = vec![vec![0.0_f32; 2]; 5];
        let selected = max_cover_sample(&features, 0, 1.0);
        assert!(selected.is_empty());
    }

    // --- ActiveLearningPool tests ---

    #[test]
    fn test_pool_initial_state() {
        let pool = ActiveLearningPool::new(10, vec![0, 1, 2]);
        assert_eq!(pool.labeled_count(), 3);
        assert_eq!(pool.unlabeled_count(), 7);
        assert_eq!(pool.total_size, 10);
    }

    #[test]
    fn test_pool_labeling_ratio() {
        let pool = ActiveLearningPool::new(10, vec![0, 1, 2, 3, 4]);
        let ratio = pool.labeling_ratio();
        assert!((ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_pool_add_labeled() {
        let mut pool = ActiveLearningPool::new(5, vec![0]);
        pool.add_labeled(vec![1, 2]);
        assert_eq!(pool.labeled_count(), 3);
        assert_eq!(pool.unlabeled_count(), 2);
    }

    #[test]
    fn test_pool_out_of_range_indices_ignored() {
        let pool = ActiveLearningPool::new(5, vec![0, 99, 100]);
        assert_eq!(pool.labeled_count(), 1); // only 0 is valid
        assert_eq!(pool.unlabeled_count(), 4);
    }

    #[test]
    fn test_pool_labeling_ratio_zero_total() {
        let pool = ActiveLearningPool::new(0, vec![]);
        assert_eq!(pool.labeling_ratio(), 0.0);
    }

    // --- error display tests ---

    #[test]
    fn test_error_display_empty_pool() {
        let e = ActiveLearningError::EmptyPool;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_error_display_budget_exceeds_pool() {
        let e = ActiveLearningError::BudgetExceedsPool {
            budget: 10,
            available: 3,
        };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("3"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = ActiveLearningError::DimensionMismatch {
            expected: 128,
            found: 64,
        };
        let s = e.to_string();
        assert!(s.contains("128") && s.contains("64"));
    }

    #[test]
    fn test_error_display_invalid_prediction() {
        let e = ActiveLearningError::InvalidPrediction {
            msg: "negative value".to_string(),
        };
        assert!(e.to_string().contains("negative value"));
    }

    #[test]
    fn test_error_display_empty_committee() {
        let e = ActiveLearningError::EmptyCommittee;
        assert!(e.to_string().contains("committee"));
    }

    // --- EGL test ---

    #[test]
    fn test_expected_gradient_length() {
        let preds = vec![
            Prediction::new(vec![0.9, 0.1]).expect("prediction failed"), // low entropy
            Prediction::new(vec![0.5, 0.5]).expect("prediction failed"), // high entropy
        ];
        let grad_norms = vec![1.0_f32, 1.0];
        let egl = expected_gradient_length(&preds, &grad_norms);
        assert_eq!(egl.len(), 2);
        assert!(egl[1] > egl[0], "high entropy should produce larger EGL");
    }

    // --- min_distances test ---

    #[test]
    fn test_min_distances_to_labeled() {
        let sampler = CoreSetSampler::new(DistanceMetric::Euclidean);
        let labeled = vec![vec![0.0_f32, 0.0], vec![10.0, 0.0]];
        let unlabeled = vec![vec![1.0_f32, 0.0], vec![9.0, 0.0]];
        let dists = sampler.min_distances_to_labeled(&labeled, &unlabeled);
        assert!((dists[0] - 1.0).abs() < 1e-4);
        assert!((dists[1] - 1.0).abs() < 1e-4);
    }
}
