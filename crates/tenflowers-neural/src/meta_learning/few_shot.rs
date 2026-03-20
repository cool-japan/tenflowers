//! Few-shot episodic sampling and Matching Networks utilities (Vinyals et al., 2016).

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

use super::prototypical::cosine_similarity;
use super::types::{softmax, Episode, MetaLearningError};

/// Samples N-way K-shot episodes from a dataset for few-shot evaluation.
#[derive(Debug, Clone)]
pub struct EpisodeSampler {
    /// Number of classes per episode.
    pub n_way: usize,
    /// Number of support examples per class.
    pub k_shot: usize,
    /// Number of query examples per class.
    pub n_query: usize,
}

impl EpisodeSampler {
    /// Create a new `EpisodeSampler`.
    pub fn new(n_way: usize, k_shot: usize, n_query: usize) -> Self {
        Self {
            n_way,
            k_shot,
            n_query,
        }
    }

    /// Sample one N-way K-shot episode from `dataset`.
    ///
    /// 1. Groups examples by class.
    /// 2. Randomly selects `n_way` classes.
    /// 3. For each selected class, samples `k_shot` support and `n_query` query
    ///    examples without replacement.
    ///
    /// # Arguments
    ///
    /// * `dataset` – slice of `(feature_vector, class_label)` pairs.
    /// * `num_classes` – total number of distinct classes in the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`MetaLearningError::InvalidNWay`] if `n_way > num_classes`.
    /// Returns [`MetaLearningError::InsufficientData`] if any selected class
    /// has fewer than `k_shot + n_query` examples.
    pub fn sample_episode(
        &self,
        dataset: &[(Vec<f32>, usize)],
        num_classes: usize,
    ) -> Result<Episode, MetaLearningError> {
        if self.n_way > num_classes {
            return Err(MetaLearningError::InvalidNWay {
                n_way: self.n_way,
                available_classes: num_classes,
            });
        }

        // Group indices by class.
        let mut by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, (_, lbl)) in dataset.iter().enumerate() {
            by_class.entry(*lbl).or_default().push(idx);
        }

        // Collect classes that have enough examples.
        let needed_per_class = self.k_shot + self.n_query;
        let eligible: Vec<usize> = by_class
            .keys()
            .filter(|&&c| by_class[&c].len() >= needed_per_class)
            .copied()
            .collect();

        if eligible.len() < self.n_way {
            return Err(MetaLearningError::InsufficientData {
                needed: self.n_way,
                found: eligible.len(),
            });
        }

        // Use a deterministic-ish seed based on dataset size for reproducibility
        // in tests; production callers should reseed as needed.
        let seed =
            dataset.len() as u64 ^ (self.n_way as u64).wrapping_mul(6_364_136_223_846_793_005_u64);
        let mut rng = StdRng::seed_from_u64(seed);

        // Shuffle eligible classes and take n_way of them.
        let mut eligible_shuffled = eligible.clone();
        fisher_yates_shuffle(&mut eligible_shuffled, &mut rng);
        let chosen_classes: Vec<usize> = eligible_shuffled[..self.n_way].to_vec();

        let mut support_features = Vec::new();
        let mut support_labels = Vec::new();
        let mut support_relabeled = Vec::new();
        let mut query_features = Vec::new();
        let mut query_labels = Vec::new();

        for (episode_label, &orig_class) in chosen_classes.iter().enumerate() {
            let indices = &by_class[&orig_class];
            let mut indices_shuffled = indices.clone();
            fisher_yates_shuffle(&mut indices_shuffled, &mut rng);

            let support_indices = &indices_shuffled[..self.k_shot];
            let query_indices = &indices_shuffled[self.k_shot..self.k_shot + self.n_query];

            for &i in support_indices {
                support_features.push(dataset[i].0.clone());
                support_labels.push(orig_class);
                support_relabeled.push(episode_label);
            }
            for &i in query_indices {
                query_features.push(dataset[i].0.clone());
                query_labels.push(episode_label);
            }
        }

        Ok(Episode {
            support_features,
            support_labels,
            support_relabeled,
            query_features,
            query_labels,
            n_way: self.n_way,
            k_shot: self.k_shot,
        })
    }
}

/// In-place Fisher-Yates shuffle using the given RNG.
pub(super) fn fisher_yates_shuffle<T, R: Rng>(v: &mut [T], rng: &mut R) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

/// Compute softmax cosine-attention weights between a query and support set.
///
/// Each weight `a_i` is:
/// ```text
/// a_i = softmax( cosine_similarity(query, support_i) )_i
/// ```
///
/// # Arguments
///
/// * `query` – a single query embedding vector.
/// * `support` – the full support set, one embedding per example.
///
/// # Returns
///
/// A vector of `support.len()` non-negative weights summing to 1.
pub fn attention_kernel(query: &[f32], support: &[Vec<f32>]) -> Vec<f32> {
    if support.is_empty() {
        return Vec::new();
    }

    let similarities: Vec<f32> = support
        .iter()
        .map(|s| cosine_similarity(query, s))
        .collect();

    softmax(&similarities)
}

/// Classify a single query using Matching Networks attention.
///
/// The predicted class is the *mode* of a distribution over `n_way` classes
/// formed by summing the attention weights of all support examples whose label
/// matches each class.
///
/// # Arguments
///
/// * `query` – query embedding.
/// * `support_features` – support-set embeddings `[n_support, dim]`.
/// * `support_labels` – relabelled support labels in `0..n_way`.
/// * `n_way` – number of classes.
///
/// # Returns
///
/// Predicted class index in `0..n_way`.
pub fn matching_networks_predict(
    query: &[f32],
    support_features: &[Vec<f32>],
    support_labels: &[usize],
    n_way: usize,
) -> usize {
    if support_features.is_empty() || n_way == 0 {
        return 0;
    }

    let weights = attention_kernel(query, support_features);
    let mut class_scores = vec![0.0_f32; n_way];

    for (w, &lbl) in weights.iter().zip(support_labels.iter()) {
        if lbl < n_way {
            class_scores[lbl] += w;
        }
    }

    class_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
