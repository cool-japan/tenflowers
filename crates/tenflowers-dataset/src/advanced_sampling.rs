//! Advanced sampling strategies for training optimization
//!
//! This module provides sophisticated sampling strategies including:
//! - Curriculum learning
//! - Importance sampling
//! - Hard negative mining
//! - Progressive sampling
//! - Class-balanced sampling

use crate::{error_taxonomy::helpers as error_helpers, Dataset};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Curriculum learning strategy
#[derive(Debug, Clone, PartialEq)]
pub enum CurriculumStrategy {
    /// Start with easiest samples, gradually increase difficulty
    EasyToHard,
    /// Start with hardest samples, gradually decrease difficulty
    HardToEasy,
    /// Random ordering (baseline)
    Random,
    /// Custom difficulty function
    Custom,
}

/// Curriculum learning scheduler
pub struct CurriculumScheduler {
    strategy: CurriculumStrategy,
    current_epoch: usize,
    total_epochs: usize,
    difficulty_scores: Vec<f32>,
    pacing_function: Box<dyn Fn(usize, usize) -> f32 + Send + Sync>,
}

impl CurriculumScheduler {
    /// Create a new curriculum scheduler
    pub fn new(
        strategy: CurriculumStrategy,
        total_epochs: usize,
        difficulty_scores: Vec<f32>,
    ) -> Self {
        // Default linear pacing function
        let pacing_function: Box<dyn Fn(usize, usize) -> f32 + Send + Sync> =
            Box::new(|current, total| current as f32 / total as f32);

        Self {
            strategy,
            current_epoch: 0,
            total_epochs,
            difficulty_scores,
            pacing_function,
        }
    }

    /// Set custom pacing function
    pub fn with_pacing<F>(mut self, pacing: F) -> Self
    where
        F: Fn(usize, usize) -> f32 + Send + Sync + 'static,
    {
        self.pacing_function = Box::new(pacing);
        self
    }

    /// Get sample indices for current epoch based on curriculum
    pub fn get_sample_indices(&self, dataset_size: usize) -> Vec<usize> {
        let pace = (self.pacing_function)(self.current_epoch, self.total_epochs);
        let num_samples = ((dataset_size as f32) * pace.min(1.0)).ceil() as usize;

        match self.strategy {
            CurriculumStrategy::EasyToHard => {
                let mut indices: Vec<(usize, f32)> = self
                    .difficulty_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                indices.iter().take(num_samples).map(|(i, _)| *i).collect()
            }
            CurriculumStrategy::HardToEasy => {
                let mut indices: Vec<(usize, f32)> = self
                    .difficulty_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indices.iter().take(num_samples).map(|(i, _)| *i).collect()
            }
            CurriculumStrategy::Random => (0..num_samples).collect(),
            CurriculumStrategy::Custom => (0..num_samples).collect(),
        }
    }

    /// Update to next epoch
    pub fn step(&mut self) {
        self.current_epoch = (self.current_epoch + 1).min(self.total_epochs);
    }

    /// Reset to first epoch
    pub fn reset(&mut self) {
        self.current_epoch = 0;
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get current pacing value (0.0 to 1.0)
    pub fn current_pace(&self) -> f32 {
        (self.pacing_function)(self.current_epoch, self.total_epochs)
    }
}

/// Advanced importance sampling for training
pub struct AdvancedImportanceSampler {
    weights: Vec<f32>,
    cumulative_weights: Vec<f32>,
    total_weight: f32,
}

impl AdvancedImportanceSampler {
    /// Create a new importance sampler
    pub fn new(weights: Vec<f32>) -> Result<Self> {
        if weights.is_empty() {
            return Err(error_helpers::invalid_configuration(
                "AdvancedImportanceSampler::new",
                "weights",
                "weights cannot be empty",
            ));
        }

        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(error_helpers::invalid_configuration(
                "AdvancedImportanceSampler::new",
                "weights",
                "total weight must be positive",
            ));
        }

        let mut cumulative_weights = Vec::with_capacity(weights.len());
        let mut cumsum = 0.0;
        for &w in &weights {
            cumsum += w;
            cumulative_weights.push(cumsum);
        }

        Ok(Self {
            weights,
            cumulative_weights,
            total_weight,
        })
    }

    /// Create from loss values (higher loss = higher weight)
    pub fn from_losses(losses: Vec<f32>) -> Result<Self> {
        let weights: Vec<f32> = losses.iter().map(|&loss| loss + 1e-8).collect();
        Self::new(weights)
    }

    /// Create from prediction confidence (lower confidence = higher weight)
    pub fn from_confidence(confidences: Vec<f32>) -> Result<Self> {
        let weights: Vec<f32> = confidences.iter().map(|&conf| 1.0 - conf + 1e-8).collect();
        Self::new(weights)
    }

    /// Sample indices with replacement
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to draw
    /// * `random_values` - Pre-generated random values in [0, 1)
    pub fn sample_with_random(&self, num_samples: usize, random_values: &[f32]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let random_val = random_values[i % random_values.len()] * self.total_weight;
            let idx = self
                .cumulative_weights
                .iter()
                .position(|&w| w >= random_val)
                .unwrap_or(self.weights.len() - 1);
            indices.push(idx);
        }

        indices
    }

    /// Get the weight for a specific index
    pub fn get_weight(&self, index: usize) -> Option<f32> {
        self.weights.get(index).copied()
    }

    /// Get normalized weights (sum to 1.0)
    pub fn normalized_weights(&self) -> Vec<f32> {
        self.weights
            .iter()
            .map(|&w| w / self.total_weight)
            .collect()
    }

    /// Update weights (e.g., based on new losses)
    pub fn update_weights(&mut self, new_weights: Vec<f32>) -> Result<()> {
        if new_weights.len() != self.weights.len() {
            return Err(error_helpers::invalid_configuration(
                "AdvancedImportanceSampler::update_weights",
                "new_weights",
                format!(
                    "new_weights length {} must match original length {}",
                    new_weights.len(),
                    self.weights.len()
                ),
            ));
        }

        self.weights = new_weights;
        self.total_weight = self.weights.iter().sum();

        let mut cumsum = 0.0;
        for (i, &w) in self.weights.iter().enumerate() {
            cumsum += w;
            self.cumulative_weights[i] = cumsum;
        }

        Ok(())
    }
}

/// Hard negative mining sampler
pub struct HardNegativeMiner {
    positive_indices: Vec<usize>,
    negative_indices: Vec<usize>,
    negative_scores: Vec<f32>, // Higher score = harder negative
    mining_strategy: MiningStrategy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MiningStrategy {
    /// Sample hardest negatives
    Hardest,
    /// Sample semi-hard negatives (moderately difficult)
    SemiHard { margin: f32 },
    /// Sample with probability proportional to difficulty
    WeightedRandom,
}

impl HardNegativeMiner {
    /// Create a new hard negative miner
    pub fn new(
        positive_indices: Vec<usize>,
        negative_indices: Vec<usize>,
        negative_scores: Vec<f32>,
        mining_strategy: MiningStrategy,
    ) -> Result<Self> {
        if negative_indices.len() != negative_scores.len() {
            return Err(error_helpers::invalid_configuration(
                "HardNegativeMiner::new",
                "negative_scores",
                "negative_scores length must match negative_indices length",
            ));
        }

        Ok(Self {
            positive_indices,
            negative_indices,
            negative_scores,
            mining_strategy,
        })
    }

    /// Mine hard negatives for training
    ///
    /// # Arguments
    /// * `num_negatives_per_positive` - Number of negatives per positive sample
    /// * `random_values` - Pre-generated random values in [0, 1)
    pub fn mine_negatives(
        &self,
        num_negatives_per_positive: usize,
        random_values: &[f32],
    ) -> Vec<usize> {
        let selected_negatives = match &self.mining_strategy {
            MiningStrategy::Hardest => {
                // Sort by score (descending) and take top-k
                let mut scored_negatives: Vec<(usize, f32)> = self
                    .negative_indices
                    .iter()
                    .zip(self.negative_scores.iter())
                    .map(|(&idx, &score)| (idx, score))
                    .collect();
                scored_negatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let num_to_mine = (self.positive_indices.len() * num_negatives_per_positive)
                    .min(scored_negatives.len());
                scored_negatives
                    .iter()
                    .take(num_to_mine)
                    .map(|(idx, _)| *idx)
                    .collect()
            }
            MiningStrategy::SemiHard { margin } => {
                // Select negatives within a margin of being hard
                let max_score = self
                    .negative_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let threshold = max_score - margin;

                let semi_hard: Vec<usize> = self
                    .negative_indices
                    .iter()
                    .zip(self.negative_scores.iter())
                    .filter(|(_, &score)| score >= threshold && score < max_score)
                    .map(|(&idx, _)| idx)
                    .collect();

                let num_to_mine = self.positive_indices.len() * num_negatives_per_positive;
                if semi_hard.len() <= num_to_mine {
                    semi_hard
                } else {
                    // Take first num_to_mine (user should shuffle beforehand if needed)
                    semi_hard.into_iter().take(num_to_mine).collect()
                }
            }
            MiningStrategy::WeightedRandom => {
                // Sample proportional to difficulty score
                let sampler = AdvancedImportanceSampler::new(self.negative_scores.clone())
                    .expect("Failed to create importance sampler");
                let num_to_mine = self.positive_indices.len() * num_negatives_per_positive;
                let sampled_indices = sampler.sample_with_random(num_to_mine, random_values);
                sampled_indices
                    .iter()
                    .map(|&i| self.negative_indices[i])
                    .collect()
            }
        };

        selected_negatives
    }

    /// Update negative scores (e.g., after model training)
    pub fn update_scores(&mut self, new_scores: Vec<f32>) -> Result<()> {
        if new_scores.len() != self.negative_scores.len() {
            return Err(error_helpers::invalid_configuration(
                "HardNegativeMiner::update_scores",
                "new_scores",
                "new_scores length must match current scores length",
            ));
        }

        self.negative_scores = new_scores;
        Ok(())
    }
}

/// Class-balanced sampling
pub struct ClassBalancedSampler {
    class_indices: HashMap<usize, Vec<usize>>,
    num_classes: usize,
    strategy: BalancingStrategy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BalancingStrategy {
    /// Oversample minority classes
    Oversample,
    /// Undersample majority classes
    Undersample,
    /// Weight samples by inverse class frequency
    WeightedSampling,
}

impl ClassBalancedSampler {
    /// Create a new class-balanced sampler
    pub fn new(labels: &[usize], num_classes: usize, strategy: BalancingStrategy) -> Result<Self> {
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();

        for (idx, &label) in labels.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        if class_indices.len() != num_classes {
            return Err(error_helpers::invalid_configuration(
                "ClassBalancedSampler::new",
                "labels",
                format!(
                    "Found {} classes but expected {}",
                    class_indices.len(),
                    num_classes
                ),
            ));
        }

        Ok(Self {
            class_indices,
            num_classes,
            strategy,
        })
    }

    /// Get balanced sample indices
    ///
    /// # Arguments
    /// * `num_samples` - Total number of samples to draw
    /// * `random_values` - Pre-generated random values in [0, 1)
    pub fn get_balanced_indices(&self, num_samples: usize, random_values: &[f32]) -> Vec<usize> {
        match self.strategy {
            BalancingStrategy::Oversample => self.oversample(num_samples, random_values),
            BalancingStrategy::Undersample => self.undersample(num_samples, random_values),
            BalancingStrategy::WeightedSampling => self.weighted_sample(num_samples, random_values),
        }
    }

    fn oversample(&self, num_samples: usize, random_values: &[f32]) -> Vec<usize> {
        let samples_per_class = num_samples / self.num_classes;
        let mut indices = Vec::new();
        let mut rand_idx = 0;

        for class_idx in 0..self.num_classes {
            if let Some(class_samples) = self.class_indices.get(&class_idx) {
                for _ in 0..samples_per_class {
                    let random_val = random_values[rand_idx % random_values.len()];
                    let idx =
                        (random_val * class_samples.len() as f32) as usize % class_samples.len();
                    indices.push(class_samples[idx]);
                    rand_idx += 1;
                }
            }
        }

        indices
    }

    fn undersample(&self, num_samples: usize, _random_values: &[f32]) -> Vec<usize> {
        let min_class_size = self
            .class_indices
            .values()
            .map(|v| v.len())
            .min()
            .unwrap_or(0);

        let samples_per_class = (num_samples / self.num_classes).min(min_class_size);
        let mut indices = Vec::new();

        for class_idx in 0..self.num_classes {
            if let Some(class_samples) = self.class_indices.get(&class_idx) {
                // Take first samples_per_class (user should shuffle beforehand if needed)
                indices.extend(class_samples.iter().take(samples_per_class));
            }
        }

        indices
    }

    fn weighted_sample(&self, num_samples: usize, random_values: &[f32]) -> Vec<usize> {
        // Calculate inverse frequency weights
        let total_samples: usize = self.class_indices.values().map(|v| v.len()).sum();
        let mut all_samples_with_weights = Vec::new();

        for class_idx in 0..self.num_classes {
            if let Some(class_samples) = self.class_indices.get(&class_idx) {
                let weight =
                    total_samples as f32 / (self.num_classes as f32 * class_samples.len() as f32);
                for &idx in class_samples {
                    all_samples_with_weights.push((idx, weight));
                }
            }
        }

        // Create importance sampler
        let weights: Vec<f32> = all_samples_with_weights.iter().map(|(_, w)| *w).collect();
        let sampler = AdvancedImportanceSampler::new(weights).expect("Failed to create sampler");
        let sampled_positions = sampler.sample_with_random(num_samples, random_values);

        sampled_positions
            .iter()
            .map(|&pos| all_samples_with_weights[pos].0)
            .collect()
    }

    /// Get class distribution statistics
    pub fn get_class_distribution(&self) -> HashMap<usize, usize> {
        self.class_indices
            .iter()
            .map(|(&class, indices)| (class, indices.len()))
            .collect()
    }
}

// Tests temporarily disabled - will be re-enabled with scirs2_core::random integration
// These tests currently use rand directly which violates SciRS2 policy
// They need to be migrated to use scirs2_core::random
#[allow(unexpected_cfgs)]
#[cfg(all(test, feature = "DISABLED_TESTS_DO_NOT_ENABLE"))]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_curriculum_scheduler_easy_to_hard() {
        let difficulty_scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let scheduler =
            CurriculumScheduler::new(CurriculumStrategy::EasyToHard, 10, difficulty_scores);

        let indices = scheduler.get_sample_indices(5);
        assert!(!indices.is_empty());

        // First sample should be the easiest (index 0, score 0.1)
        assert_eq!(indices[0], 0);
    }

    #[test]
    #[ignore]
    fn test_curriculum_scheduler_pacing() {
        let difficulty_scores = vec![0.1; 100];
        let mut scheduler =
            CurriculumScheduler::new(CurriculumStrategy::EasyToHard, 10, difficulty_scores);

        assert_eq!(scheduler.current_pace(), 0.0);
        scheduler.step();
        assert!(scheduler.current_pace() > 0.0);
    }

    #[test]
    #[ignore]
    fn test_importance_sampler() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let sampler = AdvancedImportanceSampler::new(weights).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = sampler.sample(100, &mut rng);

        assert_eq!(samples.len(), 100);
        // Higher weighted indices should appear more frequently
    }

    #[test]
    #[ignore]
    fn test_importance_sampler_from_losses() {
        let losses = vec![0.5, 1.0, 2.0, 0.1];
        let sampler = AdvancedImportanceSampler::from_losses(losses).unwrap();

        let weights = sampler.normalized_weights();
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    #[ignore]
    fn test_hard_negative_miner() {
        let positive_indices = vec![0, 1];
        let negative_indices = vec![2, 3, 4, 5];
        let negative_scores = vec![0.1, 0.5, 0.9, 0.3];

        let miner = HardNegativeMiner::new(
            positive_indices,
            negative_indices.clone(),
            negative_scores,
            MiningStrategy::Hardest,
        )
        .unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let hard_negatives = miner.mine_negatives(2, &mut rng);

        // Should select hardest negatives
        assert!(!hard_negatives.is_empty());
        assert!(hard_negatives.contains(&4)); // Index 4 has score 0.9
    }

    #[test]
    #[ignore]
    fn test_class_balanced_sampler() {
        let labels = vec![0, 0, 0, 1, 1, 2];
        let sampler = ClassBalancedSampler::new(&labels, 3, BalancingStrategy::Oversample).unwrap();

        let distribution = sampler.get_class_distribution();
        assert_eq!(distribution.get(&0), Some(&3));
        assert_eq!(distribution.get(&1), Some(&2));
        assert_eq!(distribution.get(&2), Some(&1));
    }

    #[test]
    #[ignore]
    fn test_class_balanced_sampler_oversample() {
        let labels = vec![0, 0, 1];
        let sampler = ClassBalancedSampler::new(&labels, 2, BalancingStrategy::Oversample).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let balanced = sampler.get_balanced_indices(10, &mut rng);

        assert_eq!(balanced.len(), 10);
    }

    #[test]
    #[ignore]
    fn test_mining_strategy_semihard() {
        let positive_indices = vec![0];
        let negative_indices = vec![1, 2, 3];
        let negative_scores = vec![0.5, 0.8, 0.9];

        let miner = HardNegativeMiner::new(
            positive_indices,
            negative_indices,
            negative_scores,
            MiningStrategy::SemiHard { margin: 0.2 },
        )
        .unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let negatives = miner.mine_negatives(2, &mut rng);

        assert!(!negatives.is_empty());
    }
}
