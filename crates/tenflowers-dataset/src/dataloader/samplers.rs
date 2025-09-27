//! Sampling Strategies and Implementations
//!
//! This module provides various sampling strategies for data loading including
//! sequential, random, distributed, stratified, and importance-based sampling.

use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

/// Trait for sampling strategies
pub trait Sampler: Send + Sync {
    /// Generate an iterator over sample indices
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send>;

    /// Check if this sampler produces indices in random order
    fn is_random(&self) -> bool;

    /// Set random seed if applicable
    fn set_seed(&mut self, _seed: Option<u64>) {}
}

/// Sequential sampler that iterates through indices in order
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    start: usize,
    end: Option<usize>,
}

impl SequentialSampler {
    pub fn new() -> Self {
        Self {
            start: 0,
            end: None,
        }
    }

    pub fn with_range(start: usize, end: usize) -> Self {
        Self {
            start,
            end: Some(end),
        }
    }
}

impl Default for SequentialSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SequentialSampler {
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send> {
        let end = self.end.unwrap_or(len).min(len);
        Box::new(self.start..end)
    }

    fn is_random(&self) -> bool {
        false
    }
}

/// Random sampler that generates random indices
#[derive(Debug, Clone)]
pub struct RandomSampler {
    replacement: bool,
    seed: Option<u64>,
}

impl RandomSampler {
    pub fn new() -> Self {
        Self {
            replacement: false,
            seed: None,
        }
    }

    pub fn with_replacement() -> Self {
        Self {
            replacement: true,
            seed: None,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            replacement: false,
            seed: Some(seed),
        }
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomSampler {
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send> {
        // Simple random implementation using system time as seed
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        if self.replacement {
            // With replacement - can repeat indices
            let mut indices = Vec::with_capacity(len);
            let mut state = seed;
            for _ in 0..len {
                // Simple LCG random number generator
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                indices.push((state as usize) % len);
            }
            Box::new(indices.into_iter())
        } else {
            // Without replacement - shuffle indices
            let mut indices: Vec<usize> = (0..len).collect();
            let mut state = seed;

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (state as usize) % (i + 1);
                indices.swap(i, j);
            }

            Box::new(indices.into_iter())
        }
    }

    fn is_random(&self) -> bool {
        true
    }

    fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }
}

/// Distributed sampler for multi-node training
#[derive(Debug, Clone)]
pub struct DistributedSampler {
    /// Number of distributed processes (world size)
    num_replicas: usize,
    /// Rank of current process
    rank: usize,
    /// Number of epochs (for shuffling consistency)
    epoch: usize,
    /// Whether to shuffle the data
    shuffle: bool,
    /// Random seed for shuffling
    seed: Option<u64>,
    /// Whether to drop last incomplete batch
    drop_last: bool,
}

impl DistributedSampler {
    /// Create a new distributed sampler
    pub fn new(num_replicas: usize, rank: usize) -> Result<Self> {
        if rank >= num_replicas {
            return Err(TensorError::invalid_argument(format!(
                "Rank {rank} must be less than num_replicas {num_replicas}"
            )));
        }

        Ok(Self {
            num_replicas,
            rank,
            epoch: 0,
            shuffle: true,
            seed: None,
            drop_last: false,
        })
    }

    /// Set whether to shuffle the data (default: true)
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed for shuffling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set whether to drop the last incomplete batch
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set the current epoch (affects shuffling for deterministic behavior)
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    /// Get the current epoch
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Get the rank of this sampler
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the number of replicas
    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    /// Calculate the number of samples per replica
    fn samples_per_replica(&self, total_size: usize) -> usize {
        if self.drop_last {
            total_size / self.num_replicas
        } else {
            (total_size + self.num_replicas - 1) / self.num_replicas
        }
    }

    /// Calculate the total size after padding (if needed)
    fn padded_size(&self, total_size: usize) -> usize {
        if self.drop_last {
            (total_size / self.num_replicas) * self.num_replicas
        } else {
            self.samples_per_replica(total_size) * self.num_replicas
        }
    }
}

impl Sampler for DistributedSampler {
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send> {
        let mut indices: Vec<usize> = (0..len).collect();

        // Shuffle indices if requested
        if self.shuffle {
            let seed = self.seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            });

            // Use epoch in seed for deterministic shuffling across epochs
            let effective_seed = seed.wrapping_add(self.epoch as u64);
            let mut state = effective_seed;

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (state as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        let samples_per_replica = self.samples_per_replica(len);
        let padded_size = self.padded_size(len);

        // Pad indices if necessary (replicate indices to ensure even distribution)
        if !self.drop_last && padded_size > len {
            let padding_needed = padded_size - len;
            for i in 0..padding_needed {
                indices.push(indices[i % len]);
            }
        }

        // Subsample for this rank
        let start_idx = self.rank * samples_per_replica;
        let end_idx = ((self.rank + 1) * samples_per_replica).min(indices.len());

        let rank_indices = if start_idx < indices.len() {
            indices[start_idx..end_idx].to_vec()
        } else {
            Vec::new()
        };

        Box::new(rank_indices.into_iter())
    }

    fn is_random(&self) -> bool {
        self.shuffle
    }

    fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }
}

/// Stratified sampler that maintains class balance during sampling
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    /// Class labels for each sample in the dataset
    class_labels: Vec<usize>,
    /// Number of samples to draw from each class
    samples_per_class: Option<usize>,
    /// Whether to sample with replacement
    replacement: bool,
    /// Random seed for reproducible sampling
    seed: Option<u64>,
    /// Whether to shuffle within each class
    shuffle: bool,
}

impl StratifiedSampler {
    /// Create a new stratified sampler
    pub fn new(class_labels: Vec<usize>) -> Self {
        Self {
            class_labels,
            samples_per_class: None,
            replacement: false,
            seed: None,
            shuffle: true,
        }
    }

    /// Set the number of samples to draw from each class
    pub fn with_samples_per_class(mut self, samples_per_class: usize) -> Self {
        self.samples_per_class = Some(samples_per_class);
        self
    }

    /// Enable sampling with replacement
    pub fn with_replacement(mut self) -> Self {
        self.replacement = true;
        self
    }

    /// Set random seed for reproducible sampling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set whether to shuffle samples within each class
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Get class distribution
    pub fn class_distribution(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &label in &self.class_labels {
            *counts.entry(label).or_insert(0) += 1;
        }
        counts
    }

    /// Get unique classes
    pub fn num_classes(&self) -> usize {
        self.class_distribution().len()
    }
}

impl Sampler for StratifiedSampler {
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send> {
        if self.class_labels.len() != len {
            // If class labels don't match dataset length, fall back to sequential
            return Box::new((0..len).collect::<Vec<_>>().into_iter());
        }

        // Group indices by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &class_label) in self.class_labels.iter().enumerate() {
            class_indices.entry(class_label).or_default().push(idx);
        }

        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        let mut result_indices = Vec::new();
        let mut rng_state = seed;

        // Determine samples per class
        let samples_per_class = if let Some(spc) = self.samples_per_class {
            spc
        } else {
            // Use minimum class size to ensure balanced sampling
            class_indices
                .values()
                .map(|indices| indices.len())
                .min()
                .unwrap_or(0)
        };

        // Sample from each class
        for (_, mut indices) in class_indices {
            // Shuffle indices within class if requested
            if self.shuffle {
                // Fisher-Yates shuffle
                for i in (1..indices.len()).rev() {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let j = (rng_state as usize) % (i + 1);
                    indices.swap(i, j);
                }
            }

            if self.replacement {
                // Sample with replacement
                for _ in 0..samples_per_class {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let idx = (rng_state as usize) % indices.len();
                    result_indices.push(indices[idx]);
                }
            } else {
                // Sample without replacement
                let sample_count = samples_per_class.min(indices.len());
                result_indices.extend_from_slice(&indices[..sample_count]);
            }
        }

        // Final shuffle of all selected indices
        if self.shuffle {
            for i in (1..result_indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (rng_state as usize) % (i + 1);
                result_indices.swap(i, j);
            }
        }

        Box::new(result_indices.into_iter())
    }

    fn is_random(&self) -> bool {
        self.shuffle
    }

    fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }
}

/// Importance sampler for priority-based sampling with dynamic weight updates
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    /// Importance weights for each sample
    weights: Vec<f64>,
    /// Whether to normalize weights to probabilities
    normalize: bool,
    /// Random seed for reproducible sampling
    seed: Option<u64>,
    /// Temperature for weight softening (higher = more uniform)
    temperature: f64,
}

impl ImportanceSampler {
    /// Create a new importance sampler with uniform weights
    pub fn new(dataset_size: usize) -> Self {
        Self {
            weights: vec![1.0; dataset_size],
            normalize: true,
            seed: None,
            temperature: 1.0,
        }
    }

    /// Create importance sampler with initial weights
    pub fn with_weights(weights: Vec<f64>) -> Self {
        Self {
            weights,
            normalize: true,
            seed: None,
            temperature: 1.0,
        }
    }

    /// Set whether to normalize weights to probabilities
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set random seed for reproducible sampling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set temperature for weight softening
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Update importance weight for a specific sample
    pub fn update_weight(&mut self, index: usize, weight: f64) {
        if index < self.weights.len() {
            self.weights[index] = weight;
        }
    }

    /// Update multiple weights at once
    pub fn update_weights(&mut self, updates: &[(usize, f64)]) {
        for &(index, weight) in updates {
            self.update_weight(index, weight);
        }
    }

    /// Get current weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Compute effective probabilities after temperature scaling
    fn compute_probabilities(&self) -> Vec<f64> {
        let mut probs: Vec<f64> = self
            .weights
            .iter()
            .map(|&w| (w / self.temperature).exp())
            .collect();

        if self.normalize {
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            } else {
                // All weights are zero, use uniform distribution
                let uniform_prob = 1.0 / probs.len() as f64;
                probs.fill(uniform_prob);
            }
        }

        probs
    }
}

impl Sampler for ImportanceSampler {
    fn sample_indices(&self, len: usize) -> Box<dyn Iterator<Item = usize> + Send> {
        if self.weights.len() != len {
            // If weights don't match dataset length, fall back to sequential
            return Box::new((0..len).collect::<Vec<_>>().into_iter());
        }

        let probabilities = self.compute_probabilities();

        // Convert probabilities to cumulative distribution
        let mut cumulative = Vec::with_capacity(probabilities.len());
        let mut sum = 0.0;
        for &prob in &probabilities {
            sum += prob;
            cumulative.push(sum);
        }

        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        let mut indices = Vec::with_capacity(len);
        let mut rng_state = seed;

        // Sample according to importance weights
        for _ in 0..len {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = (rng_state as f64) / (u64::MAX as f64);

            // Binary search to find the index
            let index = cumulative
                .binary_search_by(|&x| {
                    if x < random_val {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
                .unwrap_or_else(|i| i);

            indices.push(index.min(len - 1));
        }

        Box::new(indices.into_iter())
    }

    fn is_random(&self) -> bool {
        true
    }

    fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let sampler = SequentialSampler::new();
        let indices: Vec<usize> = sampler.sample_indices(5).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        assert!(!sampler.is_random());
    }

    #[test]
    fn test_sequential_sampler_with_range() {
        let sampler = SequentialSampler::with_range(2, 5);
        let indices: Vec<usize> = sampler.sample_indices(10).collect();
        assert_eq!(indices, vec![2, 3, 4]);
    }

    #[test]
    fn test_random_sampler() {
        let sampler = RandomSampler::with_seed(42);
        let indices: Vec<usize> = sampler.sample_indices(5).collect();
        assert_eq!(indices.len(), 5);
        assert!(sampler.is_random());

        // Test reproducibility with same seed
        let sampler2 = RandomSampler::with_seed(42);
        let indices2: Vec<usize> = sampler2.sample_indices(5).collect();
        assert_eq!(indices, indices2);
    }

    #[test]
    fn test_random_sampler_with_replacement() {
        let sampler = RandomSampler::with_replacement();
        let indices: Vec<usize> = sampler.sample_indices(3).collect();
        assert_eq!(indices.len(), 3);
        // With replacement, indices can repeat
    }

    #[test]
    fn test_distributed_sampler() {
        let sampler = DistributedSampler::new(2, 0).unwrap();
        let indices: Vec<usize> = sampler.sample_indices(10).collect();
        // Should get roughly half the indices for rank 0
        assert!(indices.len() >= 4 && indices.len() <= 6);
    }

    #[test]
    fn test_distributed_sampler_invalid_rank() {
        let result = DistributedSampler::new(2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_sampler() {
        let class_labels = vec![0, 0, 1, 1, 2, 2];
        let sampler = StratifiedSampler::new(class_labels.clone());

        assert_eq!(sampler.num_classes(), 3);

        let distribution = sampler.class_distribution();
        assert_eq!(distribution[&0], 2);
        assert_eq!(distribution[&1], 2);
        assert_eq!(distribution[&2], 2);
    }

    #[test]
    fn test_stratified_sampler_with_samples_per_class() {
        let class_labels = vec![0, 0, 0, 1, 1, 1];
        let sampler = StratifiedSampler::new(class_labels)
            .with_samples_per_class(1)
            .with_seed(42);

        let indices: Vec<usize> = sampler.sample_indices(6).collect();
        // Should get 1 sample from each class = 2 total samples
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_importance_sampler() {
        let sampler = ImportanceSampler::new(5);
        assert_eq!(sampler.weights().len(), 5);
        assert!(sampler.weights().iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_importance_sampler_with_weights() {
        let weights = vec![1.0, 2.0, 3.0];
        let sampler = ImportanceSampler::with_weights(weights.clone());
        assert_eq!(sampler.weights(), &weights);
    }

    #[test]
    fn test_importance_sampler_update_weight() {
        let mut sampler = ImportanceSampler::new(3);
        sampler.update_weight(1, 5.0);
        assert_eq!(sampler.weights()[1], 5.0);
    }

    #[test]
    fn test_importance_sampler_update_weights() {
        let mut sampler = ImportanceSampler::new(3);
        sampler.update_weights(&[(0, 2.0), (2, 4.0)]);
        assert_eq!(sampler.weights()[0], 2.0);
        assert_eq!(sampler.weights()[2], 4.0);
    }
}
