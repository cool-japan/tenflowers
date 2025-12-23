//! Active Learning module for intelligent data selection in machine learning pipelines
//!
//! This module provides uncertainty sampling and diversity sampling strategies for active learning,
//! enabling efficient selection of the most informative samples for training.

use crate::Dataset;
use tenflowers_core::{Result, Tensor, TensorError};

/// Uncertainty sampling strategy for active learning
#[derive(Debug, Clone)]
pub enum UncertaintyStrategy {
    /// Select samples with highest entropy
    Entropy,
    /// Select samples with smallest margin between top predictions
    Margin,
    /// Select samples with lowest confidence (highest uncertainty)
    LeastConfident,
    /// Query by committee - select samples where committee disagrees most
    QueryByCommittee,
}

/// Diversity sampling strategy for active learning
#[derive(Debug, Clone)]
pub enum DiversityStrategy {
    /// Select samples using k-means clustering to maximize diversity
    KMeansClustering,
    /// Select representative samples from feature space
    Representative,
    /// Hybrid approach combining uncertainty and diversity
    Hybrid {
        uncertainty_weight: f32,
        diversity_weight: f32,
    },
}

/// Active learning sampler that selects most informative samples
pub struct ActiveLearningSampler {
    uncertainty_strategy: UncertaintyStrategy,
    diversity_strategy: Option<DiversityStrategy>,
    batch_size: usize,
}

impl ActiveLearningSampler {
    /// Create a new active learning sampler
    pub fn new(uncertainty_strategy: UncertaintyStrategy, batch_size: usize) -> Self {
        Self {
            uncertainty_strategy,
            diversity_strategy: None,
            batch_size,
        }
    }

    /// Add diversity sampling to the active learning strategy
    pub fn with_diversity(mut self, diversity_strategy: DiversityStrategy) -> Self {
        self.diversity_strategy = Some(diversity_strategy);
        self
    }

    /// Select the most informative samples for active learning
    pub fn select_samples<T, D: Dataset<T>>(
        &self,
        dataset: &D,
        predictions: &[Vec<f32>], // Model predictions for uncertainty estimation
        features: Option<&[Vec<f32>]>, // Feature vectors for diversity sampling
    ) -> Result<Vec<usize>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        if predictions.len() != dataset.len() {
            return Err(TensorError::invalid_argument(
                "Number of predictions must match dataset size".to_string(),
            ));
        }

        // Calculate uncertainty scores
        let uncertainty_scores = self.calculate_uncertainty_scores(predictions)?;

        // Calculate diversity scores if diversity strategy is enabled
        let diversity_scores = if let Some(ref diversity_strategy) = self.diversity_strategy {
            if let Some(features) = features {
                self.calculate_diversity_scores(features, diversity_strategy)?
            } else {
                return Err(TensorError::invalid_argument(
                    "Features required for diversity sampling".to_string(),
                ));
            }
        } else {
            vec![0.0; dataset.len()]
        };

        // Combine uncertainty and diversity scores
        let combined_scores = self.combine_scores(&uncertainty_scores, &diversity_scores)?;

        // Select top samples based on combined scores
        let mut indexed_scores: Vec<(usize, f32)> =
            combined_scores.into_iter().enumerate().collect();

        // Sort by score in descending order (higher score = more informative)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top batch_size indices
        Ok(indexed_scores
            .into_iter()
            .take(self.batch_size)
            .map(|(idx, _)| idx)
            .collect())
    }

    /// Calculate uncertainty scores based on the selected strategy
    fn calculate_uncertainty_scores(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(predictions.len());

        for pred in predictions {
            let score = match self.uncertainty_strategy {
                UncertaintyStrategy::Entropy => self.calculate_entropy(pred)?,
                UncertaintyStrategy::Margin => self.calculate_margin(pred)?,
                UncertaintyStrategy::LeastConfident => self.calculate_least_confident(pred)?,
                UncertaintyStrategy::QueryByCommittee => {
                    // For QBC, we need multiple predictions - using entropy as fallback
                    self.calculate_entropy(pred)?
                }
            };
            scores.push(score);
        }

        Ok(scores)
    }

    /// Calculate entropy of prediction distribution
    fn calculate_entropy(&self, predictions: &[f32]) -> Result<f32> {
        let mut entropy = 0.0;
        let sum: f32 = predictions.iter().sum();

        if sum == 0.0 {
            return Ok(0.0);
        }

        for &p in predictions {
            let normalized_p = p / sum;
            if normalized_p > 0.0 {
                entropy -= normalized_p * normalized_p.ln();
            }
        }

        Ok(entropy)
    }

    /// Calculate margin between top two predictions
    fn calculate_margin(&self, predictions: &[f32]) -> Result<f32> {
        if predictions.len() < 2 {
            return Ok(0.0);
        }

        let mut sorted_preds = predictions.to_vec();
        sorted_preds.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Return negative margin (smaller margin = higher uncertainty)
        Ok(-(sorted_preds[0] - sorted_preds[1]))
    }

    /// Calculate least confident score
    fn calculate_least_confident(&self, predictions: &[f32]) -> Result<f32> {
        let max_pred = predictions.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
        match max_pred {
            Some(max_val) => Ok(1.0 - max_val), // Higher uncertainty = lower confidence
            None => Ok(0.0),
        }
    }

    /// Calculate diversity scores based on the selected strategy
    fn calculate_diversity_scores(
        &self,
        features: &[Vec<f32>],
        strategy: &DiversityStrategy,
    ) -> Result<Vec<f32>> {
        match strategy {
            DiversityStrategy::KMeansClustering => self.calculate_kmeans_diversity_scores(features),
            DiversityStrategy::Representative => self.calculate_representative_scores(features),
            DiversityStrategy::Hybrid { .. } => {
                // For hybrid, we'll use k-means as the base diversity measure
                self.calculate_kmeans_diversity_scores(features)
            }
        }
    }

    /// Calculate diversity scores using k-means clustering
    fn calculate_kmeans_diversity_scores(&self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        // Simplified k-means diversity: distance from cluster centers
        let k = ((features.len() as f32).sqrt() as usize).max(2);
        let centroids = self.simple_kmeans(features, k)?;

        let mut scores = Vec::with_capacity(features.len());
        for feature in features {
            // Find distance to nearest centroid
            let min_distance = centroids
                .iter()
                .map(|centroid| self.euclidean_distance(feature, centroid))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            scores.push(min_distance);
        }

        Ok(scores)
    }

    /// Calculate representative scores (distance from dataset center)
    fn calculate_representative_scores(&self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        if features.is_empty() {
            return Ok(vec![]);
        }

        let feature_dim = features[0].len();
        let mut centroid = vec![0.0; feature_dim];

        // Calculate dataset centroid
        for feature in features {
            for (i, &val) in feature.iter().enumerate() {
                centroid[i] += val;
            }
        }

        let n = features.len() as f32;
        for val in centroid.iter_mut() {
            *val /= n;
        }

        // Calculate distances from centroid
        let mut scores = Vec::with_capacity(features.len());
        for feature in features {
            let distance = self.euclidean_distance(feature, &centroid);
            scores.push(distance);
        }

        Ok(scores)
    }

    /// Simple k-means clustering implementation
    fn simple_kmeans(&self, features: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        if features.is_empty() || k == 0 {
            return Ok(vec![]);
        }

        let feature_dim = features[0].len();
        let mut centroids = Vec::with_capacity(k);

        // Initialize centroids randomly from data points
        use scirs2_core::random::rand_prelude::*;
        let mut rng = scirs2_core::random::rng();
        for _ in 0..k {
            let random_val: f64 = rng.random();
            let idx = (random_val * features.len() as f64) as usize;
            let idx = idx.min(features.len() - 1);
            centroids.push(features[idx].clone());
        }

        // Simple k-means iterations (simplified for efficiency)
        for _ in 0..10 {
            let mut new_centroids = vec![vec![0.0; feature_dim]; k];
            let mut counts = vec![0; k];

            // Assign points to nearest centroid
            for feature in features {
                let nearest_idx = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let dist_a = self.euclidean_distance(feature, a);
                        let dist_b = self.euclidean_distance(feature, b);
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                counts[nearest_idx] += 1;
                for (i, &val) in feature.iter().enumerate() {
                    new_centroids[nearest_idx][i] += val;
                }
            }

            // Update centroids
            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        Ok(centroids)
    }

    /// Calculate Euclidean distance between two feature vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Combine uncertainty and diversity scores
    fn combine_scores(
        &self,
        uncertainty_scores: &[f32],
        diversity_scores: &[f32],
    ) -> Result<Vec<f32>> {
        if uncertainty_scores.len() != diversity_scores.len() {
            return Err(TensorError::invalid_argument(
                "Uncertainty and diversity scores must have same length".to_string(),
            ));
        }

        let mut combined_scores = Vec::with_capacity(uncertainty_scores.len());

        match &self.diversity_strategy {
            Some(DiversityStrategy::Hybrid {
                uncertainty_weight,
                diversity_weight,
            }) => {
                // Normalize scores to [0, 1] range
                let max_uncertainty = uncertainty_scores
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&1.0);
                let max_diversity = diversity_scores
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&1.0);

                for (u_score, d_score) in uncertainty_scores.iter().zip(diversity_scores.iter()) {
                    let normalized_u = u_score / max_uncertainty;
                    let normalized_d = d_score / max_diversity;
                    let combined =
                        uncertainty_weight * normalized_u + diversity_weight * normalized_d;
                    combined_scores.push(combined);
                }
            }
            Some(_) => {
                // Equal weighting for other diversity strategies
                for (u_score, d_score) in uncertainty_scores.iter().zip(diversity_scores.iter()) {
                    combined_scores.push(u_score + d_score);
                }
            }
            None => {
                // Use only uncertainty scores
                combined_scores.extend_from_slice(uncertainty_scores);
            }
        }

        Ok(combined_scores)
    }
}

/// Active learning dataset wrapper that maintains labeled/unlabeled pools
pub struct ActiveLearningDataset<T, D: Dataset<T>> {
    dataset: D,
    labeled_indices: Vec<usize>,
    unlabeled_indices: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D: Dataset<T>> ActiveLearningDataset<T, D> {
    /// Create a new active learning dataset with initial labeled samples
    pub fn new(dataset: D, initial_labeled_indices: Vec<usize>) -> Self {
        let total_len = dataset.len();
        let labeled_set: std::collections::HashSet<usize> =
            initial_labeled_indices.iter().cloned().collect();
        let unlabeled_indices: Vec<usize> = (0..total_len)
            .filter(|i| !labeled_set.contains(i))
            .collect();

        Self {
            dataset,
            labeled_indices: initial_labeled_indices,
            unlabeled_indices,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add samples to the labeled pool
    pub fn add_labeled_samples(&mut self, indices: Vec<usize>) {
        let indices_set: std::collections::HashSet<usize> = indices.iter().cloned().collect();

        // Add to labeled pool
        self.labeled_indices.extend(indices);

        // Remove from unlabeled pool
        self.unlabeled_indices
            .retain(|&i| !indices_set.contains(&i));
    }

    /// Get labeled dataset
    pub fn get_labeled_dataset(&self) -> LabeledSubset<'_, T, D>
    where
        D: Clone,
    {
        LabeledSubset {
            dataset: self.dataset.clone(),
            indices: &self.labeled_indices,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get unlabeled dataset
    pub fn get_unlabeled_dataset(&self) -> UnlabeledSubset<'_, T, D>
    where
        D: Clone,
    {
        UnlabeledSubset {
            dataset: self.dataset.clone(),
            indices: &self.unlabeled_indices,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get labeled indices
    pub fn labeled_indices(&self) -> &[usize] {
        &self.labeled_indices
    }

    /// Get unlabeled indices
    pub fn unlabeled_indices(&self) -> &[usize] {
        &self.unlabeled_indices
    }
}

/// Labeled subset for active learning
pub struct LabeledSubset<'a, T, D: Dataset<T>> {
    dataset: D,
    indices: &'a [usize],
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, D: Dataset<T>> Dataset<T> for LabeledSubset<'a, T, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.indices.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for labeled subset of length {}",
                index,
                self.indices.len()
            )));
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

/// Unlabeled subset for active learning
pub struct UnlabeledSubset<'a, T, D: Dataset<T>> {
    dataset: D,
    indices: &'a [usize],
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, D: Dataset<T>> Dataset<T> for UnlabeledSubset<'a, T, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.indices.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for unlabeled subset of length {}",
                index,
                self.indices.len()
            )));
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_uncertainty_sampling() {
        let sampler = ActiveLearningSampler::new(UncertaintyStrategy::Entropy, 2);

        // Create mock predictions (higher entropy = more uncertain)
        let predictions = vec![
            vec![0.9, 0.1], // Low entropy (confident)
            vec![0.5, 0.5], // High entropy (uncertain)
            vec![0.8, 0.2], // Medium entropy
            vec![0.6, 0.4], // Medium-high entropy
        ];

        let scores = sampler.calculate_uncertainty_scores(&predictions).unwrap();

        // Higher entropy should have higher score
        assert!(scores[1] > scores[0]); // 0.5,0.5 > 0.9,0.1
        assert!(scores[3] > scores[2]); // 0.6,0.4 > 0.8,0.2
    }

    #[test]
    fn test_active_learning_dataset() {
        // Create test dataset
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Create active learning dataset with initial labeled samples
        let mut al_dataset = ActiveLearningDataset::new(dataset, vec![0, 1]);

        assert_eq!(al_dataset.labeled_indices().len(), 2);
        assert_eq!(al_dataset.unlabeled_indices().len(), 2);

        // Add more labeled samples
        al_dataset.add_labeled_samples(vec![2]);

        assert_eq!(al_dataset.labeled_indices().len(), 3);
        assert_eq!(al_dataset.unlabeled_indices().len(), 1);

        // Test labeled subset
        let labeled_subset = al_dataset.get_labeled_dataset();
        assert_eq!(labeled_subset.len(), 3);

        // Test unlabeled subset
        let unlabeled_subset = al_dataset.get_unlabeled_dataset();
        assert_eq!(unlabeled_subset.len(), 1);
    }

    #[test]
    fn test_diversity_sampling() {
        let sampler = ActiveLearningSampler::new(UncertaintyStrategy::Entropy, 2)
            .with_diversity(DiversityStrategy::Representative);

        // Create mock features with clear distance relationships
        let features = vec![
            vec![0.0, 0.0], // Close to center
            vec![2.0, 2.0], // Far from center
            vec![0.1, 0.1], // Very close to center
            vec![1.5, 1.5], // Medium distance from center
        ];

        let scores = sampler
            .calculate_diversity_scores(&features, &DiversityStrategy::Representative)
            .unwrap();

        // Points further from center should have higher diversity scores
        assert!(scores[1] > scores[2]); // (2,2) is further from center than (0.1,0.1)
        assert!(scores[1] > scores[0]); // (2,2) is further from center than (0,0)
                                        // Check that we have reasonable diversity scores
        assert!(scores.len() == 4);
        assert!(scores.iter().all(|&s| s >= 0.0)); // All scores should be non-negative
    }

    #[test]
    fn test_margin_uncertainty() {
        let sampler = ActiveLearningSampler::new(UncertaintyStrategy::Margin, 2);

        let predictions = vec![
            vec![0.9, 0.1],   // Large margin (confident)
            vec![0.51, 0.49], // Small margin (uncertain)
            vec![0.8, 0.2],   // Medium margin
        ];

        let scores = sampler.calculate_uncertainty_scores(&predictions).unwrap();

        // Smaller margin should have higher uncertainty score (negative margin)
        assert!(scores[1] > scores[0]); // 0.51,0.49 > 0.9,0.1
        assert!(scores[2] > scores[0]); // 0.8,0.2 > 0.9,0.1
    }
}
