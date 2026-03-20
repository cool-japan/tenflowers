//! Evaluation utilities: metrics, similarity measures, negative sampling.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::{dot, l2_norm_f32, RecResult, RecSysError};

// ─────────────────────────────────────────────────────────────────────────────
// RecMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics produced by [`RecommendationEvaluator`].
#[derive(Debug, Clone, Default)]
pub struct RecMetrics {
    /// Hit Rate @ K (fraction of queries where ≥1 ground-truth item is in top-K).
    pub hit_rate_at_k: f32,
    /// Normalised Discounted Cumulative Gain @ K.
    pub ndcg_at_k: f32,
    /// Mean Reciprocal Rank (MRR).
    pub mrr: f32,
    /// Recall @ K.
    pub recall_at_k: f32,
    /// Precision @ K.
    pub precision_at_k: f32,
    /// K used for evaluation.
    pub k: usize,
    /// Number of queries evaluated.
    pub n_queries: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// RecommendationEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Offline evaluation utility for recommendation systems.
#[derive(Debug, Clone, Default)]
pub struct RecommendationEvaluator;

impl RecommendationEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate ranked predictions against ground-truth item sets.
    ///
    /// - `predictions[i]` — ranked list of item IDs for query i (best first).
    /// - `ground_truth[i]` — set of relevant item IDs for query i.
    /// - `k` — cutoff rank.
    pub fn evaluate(
        &self,
        predictions: &[Vec<usize>],
        ground_truth: &[Vec<usize>],
        k: usize,
    ) -> RecResult<RecMetrics> {
        if predictions.len() != ground_truth.len() {
            return Err(RecSysError::DimensionMismatch {
                expected: predictions.len(),
                got: ground_truth.len(),
            });
        }
        if predictions.is_empty() {
            return Ok(RecMetrics::default());
        }
        let n = predictions.len();
        let mut total_hit = 0.0_f32;
        let mut total_ndcg = 0.0_f32;
        let mut total_mrr = 0.0_f32;
        let mut total_recall = 0.0_f32;
        let mut total_precision = 0.0_f32;

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            let truth_set: std::collections::HashSet<usize> = truth.iter().cloned().collect();
            let top_k: Vec<usize> = pred.iter().take(k).cloned().collect();

            // Hit
            let hits: usize = top_k.iter().filter(|i| truth_set.contains(i)).count();
            total_hit += if hits > 0 { 1.0 } else { 0.0 };

            // NDCG
            let dcg: f32 = top_k
                .iter()
                .enumerate()
                .filter(|(_, i)| truth_set.contains(i))
                .map(|(rank, _)| 1.0 / (rank as f32 + 2.0).log2())
                .sum();
            let ideal_k = truth_set.len().min(k);
            let idcg: f32 = (0..ideal_k)
                .map(|rank| 1.0 / (rank as f32 + 2.0).log2())
                .sum();
            total_ndcg += if idcg > 0.0 { dcg / idcg } else { 0.0 };

            // MRR — first hit position
            let rr: f32 = top_k
                .iter()
                .enumerate()
                .find(|(_, i)| truth_set.contains(i))
                .map(|(rank, _)| 1.0 / (rank as f32 + 1.0))
                .unwrap_or(0.0);
            total_mrr += rr;

            // Recall
            let recall = if truth_set.is_empty() {
                0.0
            } else {
                hits as f32 / truth_set.len() as f32
            };
            total_recall += recall;

            // Precision
            let precision = hits as f32 / k.max(1) as f32;
            total_precision += precision;
        }

        let n_f = n as f32;
        Ok(RecMetrics {
            hit_rate_at_k: total_hit / n_f,
            ndcg_at_k: total_ndcg / n_f,
            mrr: total_mrr / n_f,
            recall_at_k: total_recall / n_f,
            precision_at_k: total_precision / n_f,
            k,
            n_queries: n,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DotProductSimilarity
// ─────────────────────────────────────────────────────────────────────────────

/// Dot-product similarity scorer.
///
/// `score(a, b) = a · b`
#[derive(Debug, Clone, Default)]
pub struct DotProductSimilarity;

impl DotProductSimilarity {
    /// Create a new DotProductSimilarity.
    pub fn new() -> Self {
        Self
    }

    /// Compute the dot-product similarity between two embedding vectors.
    pub fn score(&self, a: &[f32], b: &[f32]) -> RecResult<f32> {
        if a.len() != b.len() {
            return Err(RecSysError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        Ok(dot(a, b))
    }

    /// Rank all items in `candidates` against a query embedding.
    ///
    /// Returns item indices sorted by descending dot-product score.
    pub fn rank(&self, query: &[f32], candidates: &[Vec<f32>]) -> RecResult<Vec<(usize, f32)>> {
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let s = self.score(query, c).unwrap_or(f32::NEG_INFINITY);
                (i, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CosineSimilarity
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity scorer.
///
/// `score(a, b) = (a · b) / (‖a‖ ‖b‖)`
#[derive(Debug, Clone, Default)]
pub struct CosineSimilarity;

impl CosineSimilarity {
    /// Create a new CosineSimilarity.
    pub fn new() -> Self {
        Self
    }

    /// Compute the cosine similarity between two embedding vectors.
    pub fn score(&self, a: &[f32], b: &[f32]) -> RecResult<f32> {
        if a.len() != b.len() {
            return Err(RecSysError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        let na = l2_norm_f32(a);
        let nb = l2_norm_f32(b);
        if na < 1e-8 || nb < 1e-8 {
            return Ok(0.0);
        }
        Ok(dot(a, b) / (na * nb))
    }

    /// Rank all candidates against a query by cosine similarity (descending).
    pub fn rank(&self, query: &[f32], candidates: &[Vec<f32>]) -> RecResult<Vec<(usize, f32)>> {
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let s = self.score(query, c).unwrap_or(0.0);
                (i, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NegativeSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for negative item sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeSamplingStrategy {
    /// Uniform sampling over all items.
    Uniform,
    /// Popularity-weighted sampling: probability ∝ item interaction count.
    PopularityWeighted,
}

/// Negative sampler for recommendation model training.
///
/// Supports both uniform and popularity-weighted sampling.
/// Avoids sampling items that the user has already interacted with.
#[derive(Debug)]
pub struct NegativeSampler {
    /// Total number of items.
    pub n_items: usize,
    /// Sampling strategy.
    pub strategy: NegativeSamplingStrategy,
    /// Popularity counts per item (used by `PopularityWeighted`).
    popularity: Vec<f32>,
    /// Cumulative distribution for popularity sampling.
    cdf: Vec<f32>,
    rng: StdRng,
}

impl Clone for NegativeSampler {
    fn clone(&self) -> Self {
        Self {
            n_items: self.n_items,
            strategy: self.strategy.clone(),
            popularity: self.popularity.clone(),
            cdf: self.cdf.clone(),
            rng: StdRng::seed_from_u64(42),
        }
    }
}

impl NegativeSampler {
    /// Create a new sampler with uniform strategy.
    pub fn new_uniform(n_items: usize, seed: u64) -> Self {
        let popularity = vec![1.0_f32; n_items];
        let cdf = uniform_cdf(n_items);
        Self {
            n_items,
            strategy: NegativeSamplingStrategy::Uniform,
            popularity,
            cdf,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create a popularity-weighted sampler from interaction counts.
    ///
    /// `item_counts[i]` is the number of observed interactions for item `i`.
    pub fn new_popularity(item_counts: &[usize], seed: u64) -> Self {
        let n_items = item_counts.len();
        // Smooth with alpha=0.75 as in word2vec negative sampling
        let popularity: Vec<f32> = item_counts
            .iter()
            .map(|&c| (c as f32 + 1.0).powf(0.75))
            .collect();
        let total: f32 = popularity.iter().sum();
        let mut cdf = Vec::with_capacity(n_items);
        let mut acc = 0.0_f32;
        for &p in &popularity {
            acc += p / total;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self {
            n_items,
            strategy: NegativeSamplingStrategy::PopularityWeighted,
            popularity,
            cdf,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Update popularity counts (e.g., after receiving new interactions).
    pub fn update_popularity(&mut self, item_counts: &[usize]) {
        let n = item_counts.len().min(self.n_items);
        let popularity: Vec<f32> = item_counts[..n]
            .iter()
            .map(|&c| (c as f32 + 1.0).powf(0.75))
            .collect();
        let total: f32 = popularity.iter().sum();
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0_f32;
        for &p in &popularity {
            acc += p / total;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        self.popularity = popularity;
        self.cdf = cdf;
    }

    /// Sample `n` negative items for a user, avoiding the user's positive items.
    ///
    /// `positives` is the set of items the user has interacted with.
    /// Returns at most `n` unique negative item IDs.
    pub fn sample(&mut self, positives: &[usize], n: usize) -> Vec<usize> {
        let pos_set: std::collections::HashSet<usize> = positives.iter().cloned().collect();
        let mut negatives = Vec::with_capacity(n);
        let mut attempts = 0;
        let max_attempts = n * 20 + 100;
        while negatives.len() < n && attempts < max_attempts {
            let item = self.sample_one();
            if !pos_set.contains(&item) && !negatives.contains(&item) {
                negatives.push(item);
            }
            attempts += 1;
        }
        negatives
    }

    /// Sample a single item according to the configured strategy.
    pub fn sample_one(&mut self) -> usize {
        match self.strategy {
            NegativeSamplingStrategy::Uniform => {
                let u: f32 = self.rng.random::<f32>();
                ((u * self.n_items as f32) as usize).min(self.n_items - 1)
            }
            NegativeSamplingStrategy::PopularityWeighted => {
                let u: f32 = self.rng.random::<f32>();
                self.cdf.partition_point(|&c| c < u).min(self.n_items - 1)
            }
        }
    }
}

fn uniform_cdf(n: usize) -> Vec<f32> {
    (1..=n).map(|i| i as f32 / n as f32).collect()
}
