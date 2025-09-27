//! Modern ML Dataset Generation Patterns
//!
//! This module provides synthetic dataset generation for modern machine learning paradigms
//! including few-shot learning, meta-learning, self-supervised learning, and contrastive learning.

use crate::{Dataset, DatasetUtilsExt};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for modern ML synthetic datasets
#[derive(Debug, Clone)]
pub struct ModernMLConfig {
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of dimensions for feature vectors
    pub feature_dim: usize,
    /// Noise level for synthetic data generation
    pub noise_level: f32,
}

impl Default for ModernMLConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            feature_dim: 128,
            noise_level: 0.1,
        }
    }
}

/// Few-Shot Learning Dataset Generator
///
/// Generates episodic datasets for few-shot learning with N-way K-shot structure
pub struct FewShotDataset<T> {
    episodes: Vec<Episode<T>>,
    current_episode: usize,
    _phantom: PhantomData<T>,
}

/// An episode in few-shot learning
#[derive(Debug, Clone)]
pub struct Episode<T> {
    /// Support set (training examples within the episode)
    pub support_set: Vec<(Tensor<T>, usize)>, // (features, class_id)
    /// Query set (test examples within the episode)
    pub query_set: Vec<(Tensor<T>, usize)>, // (features, class_id)
    /// Number of ways (classes) in this episode
    pub n_way: usize,
    /// Number of shots (examples per class) in support set
    pub k_shot: usize,
}

impl<T> FewShotDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new few-shot learning dataset
    ///
    /// # Arguments
    /// * `num_episodes` - Number of episodes to generate
    /// * `n_way` - Number of classes per episode
    /// * `k_shot` - Number of support examples per class
    /// * `query_per_class` - Number of query examples per class
    /// * `config` - Configuration for synthetic data generation
    pub fn new(
        num_episodes: usize,
        n_way: usize,
        k_shot: usize,
        query_per_class: usize,
        config: &ModernMLConfig,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut episodes = Vec::with_capacity(num_episodes);

        for episode_idx in 0..num_episodes {
            let mut support_set = Vec::new();
            let mut query_set = Vec::new();

            // Generate prototypes for each class in this episode
            let mut class_prototypes = Vec::with_capacity(n_way);
            for _ in 0..n_way {
                let prototype = generate_random_vector(&mut rng, config.feature_dim, T::one());
                class_prototypes.push(prototype);
            }

            // Generate support set
            for (class_id, prototype) in class_prototypes.iter().enumerate().take(n_way) {
                for _ in 0..k_shot {
                    let features = add_noise_to_vector(prototype, &mut rng, config.noise_level)?;
                    support_set.push((features, class_id));
                }
            }

            // Generate query set
            for (class_id, prototype) in class_prototypes.iter().enumerate().take(n_way) {
                for _ in 0..query_per_class {
                    let features = add_noise_to_vector(prototype, &mut rng, config.noise_level)?;
                    query_set.push((features, class_id));
                }
            }

            episodes.push(Episode {
                support_set,
                query_set,
                n_way,
                k_shot,
            });
        }

        Ok(Self {
            episodes,
            current_episode: 0,
            _phantom: PhantomData,
        })
    }

    /// Get the next episode
    pub fn next_episode(&mut self) -> Option<&Episode<T>> {
        if self.current_episode < self.episodes.len() {
            let episode = &self.episodes[self.current_episode];
            self.current_episode += 1;
            Some(episode)
        } else {
            None
        }
    }

    /// Reset episode iterator
    pub fn reset(&mut self) {
        self.current_episode = 0;
    }

    /// Get total number of episodes
    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }
}

/// Contrastive Learning Dataset Generator
///
/// Generates pairs of similar and dissimilar examples for contrastive learning
pub struct ContrastiveLearningDataset<T> {
    positive_pairs: Vec<(Tensor<T>, Tensor<T>)>,
    negative_pairs: Vec<(Tensor<T>, Tensor<T>)>,
    _phantom: PhantomData<T>,
}

impl<T> ContrastiveLearningDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new contrastive learning dataset
    ///
    /// # Arguments
    /// * `num_positive_pairs` - Number of similar (positive) pairs to generate
    /// * `num_negative_pairs` - Number of dissimilar (negative) pairs to generate
    /// * `config` - Configuration for synthetic data generation
    pub fn new(
        num_positive_pairs: usize,
        num_negative_pairs: usize,
        config: &ModernMLConfig,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut positive_pairs = Vec::with_capacity(num_positive_pairs);
        let mut negative_pairs = Vec::with_capacity(num_negative_pairs);

        // Generate positive pairs (similar examples)
        for _ in 0..num_positive_pairs {
            let anchor = generate_random_vector(&mut rng, config.feature_dim, T::one());
            let positive = add_noise_to_vector(&anchor, &mut rng, config.noise_level)?;
            positive_pairs.push((anchor, positive));
        }

        // Generate negative pairs (dissimilar examples)
        for _ in 0..num_negative_pairs {
            let anchor = generate_random_vector(&mut rng, config.feature_dim, T::one());
            let negative = generate_random_vector(&mut rng, config.feature_dim, T::one());
            negative_pairs.push((anchor, negative));
        }

        Ok(Self {
            positive_pairs,
            negative_pairs,
            _phantom: PhantomData,
        })
    }

    /// Get all positive pairs
    pub fn positive_pairs(&self) -> &[(Tensor<T>, Tensor<T>)] {
        &self.positive_pairs
    }

    /// Get all negative pairs
    pub fn negative_pairs(&self) -> &[(Tensor<T>, Tensor<T>)] {
        &self.negative_pairs
    }

    /// Get a specific positive pair
    pub fn get_positive_pair(&self, index: usize) -> Option<&(Tensor<T>, Tensor<T>)> {
        self.positive_pairs.get(index)
    }

    /// Get a specific negative pair
    pub fn get_negative_pair(&self, index: usize) -> Option<&(Tensor<T>, Tensor<T>)> {
        self.negative_pairs.get(index)
    }
}

/// Self-Supervised Learning Dataset Generator
///
/// Generates augmented views of data for self-supervised learning tasks
pub struct SelfSupervisedDataset<T> {
    original_data: Vec<Tensor<T>>,
    augmented_data: Vec<Vec<Tensor<T>>>, // Multiple augmentations per original
    _phantom: PhantomData<T>,
}

impl<T> SelfSupervisedDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new self-supervised learning dataset
    ///
    /// # Arguments
    /// * `num_samples` - Number of original samples to generate
    /// * `num_augmentations` - Number of augmentations per sample
    /// * `config` - Configuration for synthetic data generation
    pub fn new(
        num_samples: usize,
        num_augmentations: usize,
        config: &ModernMLConfig,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut original_data = Vec::with_capacity(num_samples);
        let mut augmented_data = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Generate original sample
            let original = generate_random_vector(&mut rng, config.feature_dim, T::one());
            original_data.push(original.clone());

            // Generate multiple augmentations
            let mut augmentations = Vec::with_capacity(num_augmentations);
            for _ in 0..num_augmentations {
                let augmentation = apply_augmentation(&original, &mut rng, config.noise_level)?;
                augmentations.push(augmentation);
            }
            augmented_data.push(augmentations);
        }

        Ok(Self {
            original_data,
            augmented_data,
            _phantom: PhantomData,
        })
    }

    /// Get original sample at index
    pub fn get_original(&self, index: usize) -> Option<&Tensor<T>> {
        self.original_data.get(index)
    }

    /// Get augmentations for sample at index
    pub fn get_augmentations(&self, index: usize) -> Option<&[Tensor<T>]> {
        self.augmented_data.get(index).map(|augs| augs.as_slice())
    }

    /// Get a specific augmentation for a sample
    pub fn get_augmentation(&self, sample_index: usize, aug_index: usize) -> Option<&Tensor<T>> {
        self.augmented_data
            .get(sample_index)
            .and_then(|augs| augs.get(aug_index))
    }

    /// Number of original samples
    pub fn len(&self) -> usize {
        self.original_data.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.original_data.is_empty()
    }
}

/// Meta-Learning Dataset Generator
///
/// Generates datasets for meta-learning scenarios with multiple tasks
pub struct MetaLearningDataset<T> {
    tasks: Vec<TaskDataset<T>>,
    _phantom: PhantomData<T>,
}

/// A single task in meta-learning
#[derive(Debug, Clone)]
pub struct TaskDataset<T> {
    /// Training data for this task
    pub train_data: Vec<(Tensor<T>, Tensor<T>)>,
    /// Test data for this task
    pub test_data: Vec<(Tensor<T>, Tensor<T>)>,
    /// Task identifier
    pub task_id: usize,
}

impl<T> MetaLearningDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new meta-learning dataset
    ///
    /// # Arguments
    /// * `num_tasks` - Number of different tasks to generate
    /// * `samples_per_task` - Number of samples per task
    /// * `test_ratio` - Ratio of test samples (0.0 to 1.0)
    /// * `config` - Configuration for synthetic data generation
    pub fn new(
        num_tasks: usize,
        samples_per_task: usize,
        test_ratio: f64,
        config: &ModernMLConfig,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut tasks = Vec::with_capacity(num_tasks);

        let num_test = (samples_per_task as f64 * test_ratio) as usize;
        let num_train = samples_per_task - num_test;

        for task_id in 0..num_tasks {
            let mut train_data = Vec::with_capacity(num_train);
            let mut test_data = Vec::with_capacity(num_test);

            // Generate task-specific parameters
            let task_weight = generate_random_vector(&mut rng, config.feature_dim, T::one());
            let task_bias = rng.random::<f32>() * 2.0 - 1.0; // Random bias in [-1, 1]

            // Generate training data
            for _ in 0..num_train {
                let features = generate_random_vector(&mut rng, config.feature_dim, T::one());
                let label = compute_synthetic_label(&features, &task_weight, task_bias)?;
                train_data.push((features, label));
            }

            // Generate test data
            for _ in 0..num_test {
                let features = generate_random_vector(&mut rng, config.feature_dim, T::one());
                let label = compute_synthetic_label(&features, &task_weight, task_bias)?;
                test_data.push((features, label));
            }

            tasks.push(TaskDataset {
                train_data,
                test_data,
                task_id,
            });
        }

        Ok(Self {
            tasks,
            _phantom: PhantomData,
        })
    }

    /// Get all tasks
    pub fn tasks(&self) -> &[TaskDataset<T>] {
        &self.tasks
    }

    /// Get a specific task
    pub fn get_task(&self, index: usize) -> Option<&TaskDataset<T>> {
        self.tasks.get(index)
    }

    /// Number of tasks
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }
}

// Helper functions

/// Generate a random vector with specified dimensions
fn generate_random_vector<T, R>(rng: &mut R, dim: usize, scale: T) -> Tensor<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    R: Rng,
{
    let data: Vec<T> = (0..dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0; // Range [-1, 1]
            T::from(val).unwrap_or(T::zero()) * scale
        })
        .collect();

    Tensor::from_vec(data, &[dim]).unwrap()
}

/// Add noise to a vector
fn add_noise_to_vector<T, R>(vector: &Tensor<T>, rng: &mut R, noise_level: f32) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    R: Rng,
{
    let data = vector.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access tensor data".to_string())
    })?;

    let noisy_data: Vec<T> = data
        .iter()
        .map(|&val| {
            let noise = rng.random::<f32>() * noise_level * 2.0 - noise_level;
            val + T::from(noise).unwrap_or(T::zero())
        })
        .collect();

    Tensor::from_vec(noisy_data, vector.shape().dims())
}

/// Apply augmentation (rotation, scaling, noise)
fn apply_augmentation<T, R>(vector: &Tensor<T>, rng: &mut R, aug_strength: f32) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    R: Rng,
{
    let data = vector.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access tensor data".to_string())
    })?;

    // Apply random scaling and rotation-like transformations
    let scale = 1.0 + (rng.random::<f32>() - 0.5) * aug_strength;
    let shift = (rng.random::<f32>() - 0.5) * aug_strength;

    let augmented_data: Vec<T> = data
        .iter()
        .map(|&val| {
            let scaled = val * T::from(scale).unwrap_or(T::one());
            let shifted = scaled + T::from(shift).unwrap_or(T::zero());
            shifted
        })
        .collect();

    Tensor::from_vec(augmented_data, vector.shape().dims())
}

/// Compute synthetic label using task-specific parameters
fn compute_synthetic_label<T>(
    features: &Tensor<T>,
    weights: &Tensor<T>,
    bias: f32,
) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
{
    let feat_data = features.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access features tensor data".to_string())
    })?;
    let weight_data = weights.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access weights tensor data".to_string())
    })?;

    if feat_data.len() != weight_data.len() {
        return Err(TensorError::invalid_shape_simple(format!(
            "Feature and weight dimensions mismatch: {} vs {}",
            feat_data.len(),
            weight_data.len()
        )));
    }

    // Compute dot product
    let dot_product: T = feat_data
        .iter()
        .zip(weight_data.iter())
        .map(|(&f, &w)| f * w)
        .fold(T::zero(), |acc, val| acc + val);

    // Add bias and apply activation (tanh)
    let linear_output = dot_product + T::from(bias).unwrap_or(T::zero());
    let activated = linear_output; // For simplicity, not applying tanh here

    Tensor::from_vec(vec![activated], &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_few_shot_dataset_creation() {
        let config = ModernMLConfig::default();
        let dataset = FewShotDataset::<f32>::new(5, 3, 2, 1, &config).unwrap();

        assert_eq!(dataset.num_episodes(), 5);

        // Check first episode
        if let Some(episode) = dataset.episodes.first() {
            assert_eq!(episode.n_way, 3);
            assert_eq!(episode.k_shot, 2);
            assert_eq!(episode.support_set.len(), 6); // 3 classes * 2 shots
            assert_eq!(episode.query_set.len(), 3); // 3 classes * 1 query each
        }
    }

    #[test]
    fn test_contrastive_learning_dataset() {
        let config = ModernMLConfig::default();
        let dataset = ContrastiveLearningDataset::<f32>::new(10, 15, &config).unwrap();

        assert_eq!(dataset.positive_pairs().len(), 10);
        assert_eq!(dataset.negative_pairs().len(), 15);
    }

    #[test]
    fn test_self_supervised_dataset() {
        let config = ModernMLConfig::default();
        let dataset = SelfSupervisedDataset::<f32>::new(5, 3, &config).unwrap();

        assert_eq!(dataset.len(), 5);
        assert!(!dataset.is_empty());

        // Check that each sample has the correct number of augmentations
        for i in 0..dataset.len() {
            assert_eq!(dataset.get_augmentations(i).unwrap().len(), 3);
        }
    }

    #[test]
    fn test_meta_learning_dataset() {
        let config = ModernMLConfig::default();
        let dataset = MetaLearningDataset::<f32>::new(3, 20, 0.2, &config).unwrap();

        assert_eq!(dataset.num_tasks(), 3);

        // Check each task has correct train/test split
        for task in dataset.tasks() {
            assert_eq!(task.train_data.len(), 16); // 80% of 20
            assert_eq!(task.test_data.len(), 4); // 20% of 20
        }
    }

    #[test]
    fn test_vector_generation() {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let vector = generate_random_vector(&mut rng, 10, 1.0f32);

        assert_eq!(vector.shape().dims(), &[10]);
    }

    #[test]
    fn test_noise_addition() {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};

        let original = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let noisy = add_noise_to_vector(&original, &mut rng, 0.1).unwrap();

        assert_eq!(noisy.shape().dims(), &[3]);

        // Check that the noisy version is different but similar
        let orig_data = original.as_slice().unwrap();
        let noisy_data = noisy.as_slice().unwrap();

        for (orig, noise) in orig_data.iter().zip(noisy_data.iter()) {
            assert!((orig - noise).abs() <= 0.2); // Should be within noise range
        }
    }
}
