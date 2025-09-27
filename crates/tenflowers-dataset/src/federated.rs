//! Federated learning dataset utilities for privacy-preserving distributed ML
//!
//! This module provides basic infrastructure for federated learning scenarios,
//! including client data partitioning, differential privacy, and heterogeneous
//! data distribution management.

use crate::{Dataset, Result};
use scirs2_core::random::distributions::{Distribution, Normal};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Tensor, TensorError};

/// Unique identifier for federated learning clients
pub type ClientId = String;

/// Federated learning client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Client identifier
    pub client_id: ClientId,
    /// Data distribution type for this client
    pub distribution_type: DataDistribution,
    /// Privacy settings
    pub privacy_config: PrivacyConfig,
    /// Client-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Data distribution types for federated clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataDistribution {
    /// Independent and identically distributed (IID)
    Iid,
    /// Non-IID with class imbalance
    NonIidClassImbalance { class_weights: Vec<f64> },
    /// Non-IID with feature shift
    NonIidFeatureShift { shift_factor: f64 },
    /// Non-IID with both class and feature shifts
    NonIidMixed {
        class_weights: Vec<f64>,
        shift_factor: f64,
    },
    /// Custom distribution strategy
    Custom {
        strategy_name: String,
        parameters: HashMap<String, f64>,
    },
}

/// Privacy configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub enable_dp: bool,
    /// Epsilon value for differential privacy (smaller = more private)
    pub epsilon: f64,
    /// Delta value for differential privacy
    pub delta: f64,
    /// Noise mechanism
    pub noise_mechanism: NoiseMechanism,
    /// Privacy budget tracking
    pub privacy_budget: f64,
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Laplace mechanism
    Laplace { sensitivity: f64 },
    /// Gaussian mechanism
    Gaussian { sensitivity: f64 },
    /// Exponential mechanism
    Exponential { sensitivity: f64 },
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enable_dp: false,
            epsilon: 1.0,
            delta: 1e-5,
            noise_mechanism: NoiseMechanism::Gaussian { sensitivity: 1.0 },
            privacy_budget: 10.0,
        }
    }
}

/// Client dataset for federated learning
#[derive(Debug)]
pub struct FederatedClientDataset<T, D> {
    /// Client configuration
    config: ClientConfig,
    /// Local dataset
    dataset: D,
    /// Privacy manager for differential privacy
    privacy_manager: Arc<Mutex<PrivacyManager>>,
    /// Client statistics
    stats: ClientStats,
    _phantom: std::marker::PhantomData<T>,
}

/// Privacy manager for differential privacy operations
#[derive(Debug)]
pub struct PrivacyManager {
    /// Remaining privacy budget
    remaining_budget: f64,
    /// Noise generation RNG
    rng: StdRng,
    /// Noise scale cache
    noise_scale_cache: HashMap<String, f64>,
}

/// Client-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    /// Number of samples
    pub sample_count: usize,
    /// Class distribution
    pub class_distribution: HashMap<String, usize>,
    /// Feature statistics
    pub feature_stats: FederatedFeatureStats,
    /// Data quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Feature statistics for federated analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedFeatureStats {
    /// Feature means
    pub means: Vec<f64>,
    /// Feature standard deviations
    pub stds: Vec<f64>,
    /// Feature ranges
    pub ranges: Vec<(f64, f64)>,
}

/// Data quality metrics for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Missing value percentage
    pub missing_percentage: f64,
    /// Outlier percentage
    pub outlier_percentage: f64,
    /// Data consistency score (0.0 to 1.0)
    pub consistency_score: f64,
}

impl PrivacyManager {
    /// Create a new privacy manager
    pub fn new(config: &PrivacyConfig, seed: u64) -> Self {
        Self {
            remaining_budget: config.privacy_budget,
            rng: StdRng::seed_from_u64(seed),
            noise_scale_cache: HashMap::new(),
        }
    }

    /// Add differential privacy noise to a value
    pub fn add_noise(
        &mut self,
        value: f64,
        config: &PrivacyConfig,
        query_sensitivity: f64,
    ) -> Result<f64> {
        if !config.enable_dp {
            return Ok(value);
        }

        if self.remaining_budget <= 0.0 {
            return Err(TensorError::invalid_argument(
                "Privacy budget exhausted".to_string(),
            ));
        }

        let noise_scale = self.calculate_noise_scale(config, query_sensitivity);
        let noise = match &config.noise_mechanism {
            NoiseMechanism::Laplace { .. } => {
                // Laplace noise: scale = sensitivity / epsilon
                let scale = noise_scale;
                self.sample_laplace(scale)
            }
            NoiseMechanism::Gaussian { .. } => {
                // Gaussian noise: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
                let sigma = noise_scale;
                self.sample_gaussian(sigma)
            }
            NoiseMechanism::Exponential { .. } => {
                // Simplified exponential mechanism (not full implementation)
                let scale = noise_scale;
                self.sample_laplace(scale)
            }
        };

        // Consume privacy budget
        self.remaining_budget -= config.epsilon;

        Ok(value + noise)
    }

    /// Add noise to a tensor with differential privacy
    pub fn add_noise_tensor<T>(
        &mut self,
        tensor: &Tensor<T>,
        config: &PrivacyConfig,
        sensitivity: f64,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
        T: From<f64> + Into<f64>,
    {
        if !config.enable_dp {
            return Ok(tensor.clone());
        }

        let shape = tensor.shape().dims().to_vec();
        let mut noisy_data = Vec::new();

        if let Some(slice) = tensor.as_slice() {
            for value in slice {
                let original_value: f64 = value.clone().into();
                let noisy_value = self.add_noise(original_value, config, sensitivity)?;
                noisy_data.push(T::from(noisy_value));
            }
        } else {
            // Handle scalar tensor
            let value: f64 = tensor.get(&[]).unwrap_or_default().into();
            let noisy_value = self.add_noise(value, config, sensitivity)?;
            noisy_data.push(T::from(noisy_value));
        }

        Tensor::from_vec(noisy_data, &shape)
    }

    fn calculate_noise_scale(&mut self, config: &PrivacyConfig, sensitivity: f64) -> f64 {
        let cache_key = format!("{}_{}", config.epsilon, sensitivity);

        if let Some(&cached_scale) = self.noise_scale_cache.get(&cache_key) {
            return cached_scale;
        }

        let scale = match &config.noise_mechanism {
            NoiseMechanism::Laplace { .. } => sensitivity / config.epsilon,
            NoiseMechanism::Gaussian { .. } => {
                let factor = (2.0 * (1.25 / config.delta).ln()).sqrt();
                factor * sensitivity / config.epsilon
            }
            NoiseMechanism::Exponential { .. } => sensitivity / config.epsilon,
        };

        self.noise_scale_cache.insert(cache_key, scale);
        scale
    }

    fn sample_laplace(&mut self, scale: f64) -> f64 {
        // Box-Muller transform for Laplace distribution approximation
        let u1: f64 = self.rng.random();
        let u2: f64 = self.rng.random();

        let sign = if u1 < 0.5 { -1.0 } else { 1.0 };
        sign * scale * (1.0_f64 - 2.0_f64 * u2.abs()).max(1e-10_f64).ln()
    }

    fn sample_gaussian(&mut self, sigma: f64) -> f64 {
        let normal = Normal::new(0.0, sigma).unwrap();
        normal.sample(&mut self.rng)
    }

    /// Check if privacy budget allows the operation
    pub fn can_spend_budget(&self, epsilon: f64) -> bool {
        self.remaining_budget >= epsilon
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        self.remaining_budget
    }
}

impl<T, D> FederatedClientDataset<T, D>
where
    D: Dataset<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new federated client dataset
    pub fn new(dataset: D, config: ClientConfig) -> Self {
        let stats = Self::compute_basic_stats(&dataset);
        let privacy_manager = Arc::new(Mutex::new(PrivacyManager::new(&config.privacy_config, 42)));

        Self {
            config,
            dataset,
            privacy_manager,
            stats,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get client configuration
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Get client statistics
    pub fn stats(&self) -> &ClientStats {
        &self.stats
    }

    /// Get a private sample (with differential privacy if enabled)
    pub fn get_private(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: From<f64> + Into<f64>,
    {
        let (features, labels) = self.dataset.get(index)?;

        if !self.config.privacy_config.enable_dp {
            return Ok((features, labels));
        }

        let mut privacy_manager = self.privacy_manager.lock().unwrap();
        let noisy_features =
            privacy_manager.add_noise_tensor(&features, &self.config.privacy_config, 1.0)?;

        Ok((noisy_features, labels))
    }

    /// Compute aggregated statistics with differential privacy
    pub fn compute_private_statistics(&self) -> Result<PrivateStats>
    where
        T: From<f64> + Into<f64>,
    {
        let mut feature_sums = Vec::new();
        let mut feature_counts = Vec::new();
        let sample_count = self.dataset.len();

        if sample_count == 0 {
            return Ok(PrivateStats {
                sample_count: 0,
                feature_means: Vec::new(),
                class_counts: HashMap::new(),
            });
        }

        // Get first sample to determine dimensions
        let (first_features, _) = self.dataset.get(0)?;
        let feature_dim = if let Some(slice) = first_features.as_slice() {
            slice.len()
        } else {
            1
        };

        feature_sums.resize(feature_dim, 0.0);
        feature_counts.resize(feature_dim, 0);

        // Aggregate features
        for i in 0..sample_count {
            let (features, _) = self.dataset.get(i)?;

            if let Some(slice) = features.as_slice() {
                for (j, value) in slice.iter().enumerate() {
                    feature_sums[j] += value.clone().into();
                    feature_counts[j] += 1;
                }
            } else {
                let value: f64 = features.get(&[]).unwrap_or(T::default()).into();
                feature_sums[0] += value;
                feature_counts[0] += 1;
            }
        }

        // Compute means with differential privacy
        let mut private_means = Vec::new();
        let mut privacy_manager = self.privacy_manager.lock().unwrap();

        for i in 0..feature_dim {
            let mean = if feature_counts[i] > 0 {
                feature_sums[i] / feature_counts[i] as f64
            } else {
                0.0
            };

            let private_mean = privacy_manager.add_noise(mean, &self.config.privacy_config, 1.0)?;
            private_means.push(private_mean);
        }

        // Add noise to sample count
        let private_sample_count =
            privacy_manager.add_noise(sample_count as f64, &self.config.privacy_config, 1.0)?
                as usize;

        Ok(PrivateStats {
            sample_count: private_sample_count,
            feature_means: private_means,
            class_counts: HashMap::new(), // Simplified for basic implementation
        })
    }

    fn compute_basic_stats(dataset: &D) -> ClientStats {
        let sample_count = dataset.len();

        ClientStats {
            sample_count,
            class_distribution: HashMap::new(), // Simplified
            feature_stats: FederatedFeatureStats {
                means: Vec::new(),
                stds: Vec::new(),
                ranges: Vec::new(),
            },
            quality_metrics: QualityMetrics {
                missing_percentage: 0.0,
                outlier_percentage: 0.0,
                consistency_score: 1.0,
            },
        }
    }
}

impl<T, D> Dataset<T> for FederatedClientDataset<T, D>
where
    D: Dataset<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        self.dataset.get(index)
    }
}

/// Statistics computed with differential privacy
#[derive(Debug, Clone)]
pub struct PrivateStats {
    /// Sample count (with noise)
    pub sample_count: usize,
    /// Feature means (with noise)
    pub feature_means: Vec<f64>,
    /// Class counts (with noise)
    pub class_counts: HashMap<String, usize>,
}

/// Federated dataset partitioner for distributing data across clients
#[derive(Debug)]
pub struct FederatedPartitioner {
    /// Total number of clients
    num_clients: usize,
    /// Partitioning strategy
    strategy: PartitioningStrategy,
    /// Random number generator for partitioning
    rng: StdRng,
}

/// Partitioning strategies for federated learning
#[derive(Debug, Clone)]
pub enum PartitioningStrategy {
    /// Uniform random distribution
    Uniform,
    /// Dirichlet distribution for non-IID data
    Dirichlet { alpha: f64 },
    /// Class-based partitioning
    ClassBased { classes_per_client: usize },
    /// Quantity-based partitioning (different dataset sizes)
    QuantityBased { size_variance: f64 },
}

impl FederatedPartitioner {
    /// Create a new federated partitioner
    pub fn new(num_clients: usize, strategy: PartitioningStrategy, seed: u64) -> Self {
        Self {
            num_clients,
            strategy,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Partition a dataset across multiple clients
    pub fn partition<T, D>(
        &mut self,
        dataset: D,
    ) -> Result<Vec<FederatedClientDataset<T, ClientIndexedDataset<T, D>>>>
    where
        D: Dataset<T> + Clone,
        T: Clone + Default + Send + Sync + 'static,
    {
        let total_samples = dataset.len();
        let client_assignments = self.generate_client_assignments(total_samples)?;

        let mut client_datasets = Vec::new();

        for (client_idx, indices) in client_assignments.into_iter().enumerate() {
            let client_id = format!("client_{client_idx}");
            let client_dataset = ClientIndexedDataset::new(dataset.clone(), indices);

            let config = ClientConfig {
                client_id: client_id.clone(),
                distribution_type: self.get_distribution_type_for_client(client_idx),
                privacy_config: PrivacyConfig::default(),
                metadata: HashMap::new(),
            };

            let federated_client = FederatedClientDataset::new(client_dataset, config);
            client_datasets.push(federated_client);
        }

        Ok(client_datasets)
    }

    fn generate_client_assignments(&mut self, total_samples: usize) -> Result<Vec<Vec<usize>>> {
        match &self.strategy {
            PartitioningStrategy::Uniform => self.uniform_partition(total_samples),
            PartitioningStrategy::Dirichlet { alpha } => {
                self.dirichlet_partition(total_samples, *alpha)
            }
            PartitioningStrategy::ClassBased {
                classes_per_client: _,
            } => {
                // Simplified class-based partitioning
                self.uniform_partition(total_samples)
            }
            PartitioningStrategy::QuantityBased { size_variance } => {
                self.quantity_based_partition(total_samples, *size_variance)
            }
        }
    }

    fn uniform_partition(&mut self, total_samples: usize) -> Result<Vec<Vec<usize>>> {
        let mut indices: Vec<usize> = (0..total_samples).collect();

        // Shuffle indices
        for i in (1..indices.len()).rev() {
            let j = self.rng.random_range(0..i);
            indices.swap(i, j);
        }

        let base_size = total_samples / self.num_clients;
        let remainder = total_samples % self.num_clients;

        let mut client_assignments = Vec::new();
        let mut start_idx = 0;

        for i in 0..self.num_clients {
            let client_size = base_size + if i < remainder { 1 } else { 0 };
            let end_idx = start_idx + client_size;

            client_assignments.push(indices[start_idx..end_idx].to_vec());
            start_idx = end_idx;
        }

        Ok(client_assignments)
    }

    fn dirichlet_partition(&mut self, total_samples: usize, alpha: f64) -> Result<Vec<Vec<usize>>> {
        // Simplified Dirichlet partitioning (basic implementation)
        // In a full implementation, this would use proper Dirichlet distribution
        let mut proportions = Vec::new();
        let mut sum = 0.0;

        for _ in 0..self.num_clients {
            let prop = self.rng.random::<f64>() * alpha + 0.1; // Simple approximation
            proportions.push(prop);
            sum += prop;
        }

        // Normalize proportions
        for prop in &mut proportions {
            *prop /= sum;
        }

        let mut client_assignments = Vec::new();
        let mut assigned_samples = 0;

        for (i, &proportion) in proportions.iter().enumerate() {
            let client_samples = if i == self.num_clients - 1 {
                // Last client gets remaining samples
                total_samples - assigned_samples
            } else {
                (total_samples as f64 * proportion) as usize
            };

            let indices: Vec<usize> =
                (assigned_samples..assigned_samples + client_samples).collect();
            client_assignments.push(indices);
            assigned_samples += client_samples;
        }

        Ok(client_assignments)
    }

    fn quantity_based_partition(
        &mut self,
        total_samples: usize,
        size_variance: f64,
    ) -> Result<Vec<Vec<usize>>> {
        let base_size = total_samples as f64 / self.num_clients as f64;
        let mut client_sizes = Vec::new();
        let mut total_assigned = 0;

        for i in 0..self.num_clients {
            let variance_factor = 1.0 + (self.rng.random::<f64>() - 0.5) * 2.0 * size_variance;
            let client_size = if i == self.num_clients - 1 {
                // Last client gets remaining samples
                total_samples - total_assigned
            } else {
                ((base_size * variance_factor) as usize).min(total_samples - total_assigned)
            };

            client_sizes.push(client_size);
            total_assigned += client_size;

            if total_assigned >= total_samples {
                break;
            }
        }

        let mut client_assignments = Vec::new();
        let mut start_idx = 0;

        for &size in &client_sizes {
            let end_idx = (start_idx + size).min(total_samples);
            let indices: Vec<usize> = (start_idx..end_idx).collect();
            client_assignments.push(indices);
            start_idx = end_idx;
        }

        Ok(client_assignments)
    }

    fn get_distribution_type_for_client(&self, _client_idx: usize) -> DataDistribution {
        match &self.strategy {
            PartitioningStrategy::Uniform => DataDistribution::Iid,
            PartitioningStrategy::Dirichlet { alpha } => DataDistribution::NonIidClassImbalance {
                class_weights: vec![*alpha, 1.0 - alpha],
            },
            PartitioningStrategy::ClassBased { .. } => DataDistribution::NonIidClassImbalance {
                class_weights: vec![0.8, 0.2],
            },
            PartitioningStrategy::QuantityBased { .. } => DataDistribution::Iid,
        }
    }
}

/// Dataset wrapper that provides access to a subset of indices
#[derive(Debug, Clone)]
pub struct ClientIndexedDataset<T, D> {
    dataset: D,
    indices: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D> ClientIndexedDataset<T, D>
where
    D: Dataset<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new client indexed dataset
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self {
            dataset,
            indices,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the underlying dataset
    pub fn inner(&self) -> &D {
        &self.dataset
    }

    /// Get the client indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

impl<T, D> Dataset<T> for ClientIndexedDataset<T, D>
where
    D: Dataset<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.indices.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for client dataset of length {}",
                index,
                self.indices.len()
            )));
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

/// Federated aggregator for combining results from multiple clients
#[derive(Debug)]
pub struct FederatedAggregator {
    /// Aggregation strategy
    strategy: AggregationStrategy,
    /// Client weights for weighted aggregation
    client_weights: HashMap<ClientId, f64>,
}

/// Aggregation strategies for federated learning
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Simple averaging
    Average,
    /// Weighted averaging by dataset size
    WeightedBySize,
    /// Weighted averaging by data quality
    WeightedByQuality,
    /// Median aggregation
    Median,
    /// Trimmed mean (excluding outliers)
    TrimmedMean { trim_fraction: f64 },
}

impl FederatedAggregator {
    /// Create a new federated aggregator
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            strategy,
            client_weights: HashMap::new(),
        }
    }

    /// Set client weight for weighted aggregation
    pub fn set_client_weight(&mut self, client_id: ClientId, weight: f64) {
        self.client_weights.insert(client_id, weight);
    }

    /// Aggregate statistics from multiple clients
    pub fn aggregate_statistics(
        &self,
        client_stats: Vec<(ClientId, PrivateStats)>,
    ) -> Result<PrivateStats> {
        if client_stats.is_empty() {
            return Err(TensorError::invalid_argument(
                "No client statistics provided".to_string(),
            ));
        }

        match &self.strategy {
            AggregationStrategy::Average => self.average_statistics(client_stats),
            AggregationStrategy::WeightedBySize => {
                self.weighted_statistics(client_stats, |stats| stats.sample_count as f64)
            }
            AggregationStrategy::WeightedByQuality => {
                self.weighted_statistics(client_stats, |_| 1.0)
            } // Simplified
            AggregationStrategy::Median => self.median_statistics(client_stats),
            AggregationStrategy::TrimmedMean { trim_fraction } => {
                self.trimmed_mean_statistics(client_stats, *trim_fraction)
            }
        }
    }

    fn average_statistics(
        &self,
        client_stats: Vec<(ClientId, PrivateStats)>,
    ) -> Result<PrivateStats> {
        let num_clients = client_stats.len() as f64;
        let mut total_samples = 0;
        let mut aggregated_means = Vec::new();
        let mut aggregated_class_counts = HashMap::new();

        // Initialize aggregated means with zeros
        if let Some((_, first_stats)) = client_stats.first() {
            aggregated_means.resize(first_stats.feature_means.len(), 0.0);
        }

        for (_, stats) in &client_stats {
            total_samples += stats.sample_count;

            // Aggregate feature means
            for (i, &mean) in stats.feature_means.iter().enumerate() {
                if i < aggregated_means.len() {
                    aggregated_means[i] += mean / num_clients;
                }
            }

            // Aggregate class counts
            for (class, &count) in &stats.class_counts {
                *aggregated_class_counts.entry(class.clone()).or_insert(0) += count;
            }
        }

        Ok(PrivateStats {
            sample_count: total_samples,
            feature_means: aggregated_means,
            class_counts: aggregated_class_counts,
        })
    }

    fn weighted_statistics<F>(
        &self,
        client_stats: Vec<(ClientId, PrivateStats)>,
        weight_fn: F,
    ) -> Result<PrivateStats>
    where
        F: Fn(&PrivateStats) -> f64,
    {
        #[allow(unused_assignments)]
        let mut total_weight = 0.0;
        let mut total_samples = 0;
        let mut aggregated_means = Vec::new();
        let mut aggregated_class_counts = HashMap::new();

        // Calculate weights and initialize
        let weights: Vec<f64> = client_stats
            .iter()
            .map(|(_, stats)| weight_fn(stats))
            .collect();

        total_weight = weights.iter().sum();

        if let Some((_, first_stats)) = client_stats.first() {
            aggregated_means.resize(first_stats.feature_means.len(), 0.0);
        }

        for ((_, stats), weight) in client_stats.iter().zip(weights.iter()) {
            let normalized_weight = weight / total_weight;
            total_samples += stats.sample_count;

            // Aggregate feature means
            for (i, &mean) in stats.feature_means.iter().enumerate() {
                if i < aggregated_means.len() {
                    aggregated_means[i] += mean * normalized_weight;
                }
            }

            // Aggregate class counts
            for (class, &count) in &stats.class_counts {
                let weighted_count = (count as f64 * normalized_weight) as usize;
                *aggregated_class_counts.entry(class.clone()).or_insert(0) += weighted_count;
            }
        }

        Ok(PrivateStats {
            sample_count: total_samples,
            feature_means: aggregated_means,
            class_counts: aggregated_class_counts,
        })
    }

    fn median_statistics(
        &self,
        client_stats: Vec<(ClientId, PrivateStats)>,
    ) -> Result<PrivateStats> {
        // Simplified median implementation
        self.average_statistics(client_stats)
    }

    fn trimmed_mean_statistics(
        &self,
        client_stats: Vec<(ClientId, PrivateStats)>,
        _trim_fraction: f64,
    ) -> Result<PrivateStats> {
        // Simplified trimmed mean implementation
        self.average_statistics(client_stats)
    }
}

/// Extension trait for federated dataset operations
pub trait FederatedDatasetExt<T>: Dataset<T> + Sized
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a federated client dataset
    fn federated_client(self, config: ClientConfig) -> FederatedClientDataset<T, Self> {
        FederatedClientDataset::new(self, config)
    }

    /// Partition dataset for federated learning
    fn partition_federated(
        self,
        num_clients: usize,
        strategy: PartitioningStrategy,
        seed: u64,
    ) -> Result<Vec<FederatedClientDataset<T, ClientIndexedDataset<T, Self>>>>
    where
        Self: Clone,
    {
        let mut partitioner = FederatedPartitioner::new(num_clients, strategy, seed);
        partitioner.partition(self)
    }
}

impl<T, D: Dataset<T>> FederatedDatasetExt<T> for D where T: Clone + Default + Send + Sync + 'static {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;

    #[test]
    fn test_privacy_manager() {
        let config = PrivacyConfig {
            enable_dp: true,
            epsilon: 1.0,
            delta: 1e-5,
            noise_mechanism: NoiseMechanism::Gaussian { sensitivity: 1.0 },
            privacy_budget: 10.0,
        };

        let mut privacy_manager = PrivacyManager::new(&config, 42);

        assert_eq!(privacy_manager.remaining_budget(), 10.0);
        assert!(privacy_manager.can_spend_budget(1.0));

        let noisy_value = privacy_manager.add_noise(5.0, &config, 1.0).unwrap();
        assert!(privacy_manager.remaining_budget() < 10.0);
        assert_ne!(noisy_value, 5.0); // Should have noise added
    }

    #[test]
    fn test_federated_client_dataset() {
        // Create test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let labels_data = vec![0.0, 1.0, 0.0];
        let features = Tensor::from_vec(features_data, &[3, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[3]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ClientConfig {
            client_id: "test_client".to_string(),
            distribution_type: DataDistribution::Iid,
            privacy_config: PrivacyConfig::default(),
            metadata: HashMap::new(),
        };

        let federated_dataset = FederatedClientDataset::new(dataset, config);

        assert_eq!(federated_dataset.len(), 3);
        assert_eq!(federated_dataset.config().client_id, "test_client");
        assert_eq!(federated_dataset.stats().sample_count, 3);
    }

    #[test]
    fn test_federated_partitioner() {
        // Create test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let labels_data = vec![0.0, 1.0, 0.0, 1.0];
        let features = Tensor::from_vec(features_data, &[4, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[4]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let mut partitioner = FederatedPartitioner::new(2, PartitioningStrategy::Uniform, 42);
        let client_datasets = partitioner.partition(dataset).unwrap();

        assert_eq!(client_datasets.len(), 2);

        let total_samples: usize = client_datasets.iter().map(|d| d.len()).sum();
        assert_eq!(total_samples, 4);
    }

    #[test]
    fn test_client_indexed_dataset() {
        // Create test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let labels_data = vec![0.0, 1.0, 0.0];
        let features = Tensor::from_vec(features_data, &[3, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[3]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let indices = vec![0, 2]; // Skip index 1
        let client_dataset = ClientIndexedDataset::new(dataset, indices);

        assert_eq!(client_dataset.len(), 2);
        assert_eq!(client_dataset.indices(), &[0, 2]);

        let (features, labels) = client_dataset.get(0).unwrap();
        let features_slice = features.as_slice().unwrap();
        assert_eq!(features_slice, &[1.0, 2.0]); // First sample
        assert_eq!(labels.get(&[]).unwrap(), 0.0);

        let (features, labels) = client_dataset.get(1).unwrap();
        let features_slice = features.as_slice().unwrap();
        assert_eq!(features_slice, &[5.0, 6.0]); // Third sample (index 2)
        assert_eq!(labels.get(&[]).unwrap(), 0.0);
    }

    #[test]
    fn test_federated_aggregator() {
        let aggregator = FederatedAggregator::new(AggregationStrategy::Average);

        let client_stats = vec![
            (
                "client1".to_string(),
                PrivateStats {
                    sample_count: 100,
                    feature_means: vec![1.0, 2.0],
                    class_counts: HashMap::new(),
                },
            ),
            (
                "client2".to_string(),
                PrivateStats {
                    sample_count: 200,
                    feature_means: vec![3.0, 4.0],
                    class_counts: HashMap::new(),
                },
            ),
        ];

        let aggregated = aggregator.aggregate_statistics(client_stats).unwrap();

        assert_eq!(aggregated.sample_count, 300);
        assert_eq!(aggregated.feature_means, vec![2.0, 3.0]); // Average of [1,2] and [3,4]
    }

    #[test]
    fn test_privacy_config_serialization() {
        let config = PrivacyConfig {
            enable_dp: true,
            epsilon: 1.0,
            delta: 1e-5,
            noise_mechanism: NoiseMechanism::Gaussian { sensitivity: 1.0 },
            privacy_budget: 10.0,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PrivacyConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.enable_dp, true);
        assert_eq!(deserialized.epsilon, 1.0);
    }

    #[test]
    fn test_extension_trait() {
        // Create test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0];
        let labels_data = vec![0.0, 1.0];
        let features = Tensor::from_vec(features_data, &[2, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Test federated client creation
        let config = ClientConfig {
            client_id: "test_client".to_string(),
            distribution_type: DataDistribution::Iid,
            privacy_config: PrivacyConfig::default(),
            metadata: HashMap::new(),
        };

        let federated_dataset = dataset.federated_client(config);
        assert_eq!(federated_dataset.len(), 2);

        // Test partitioning
        let features_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let labels_data = vec![0.0, 1.0, 0.0, 1.0];
        let features = Tensor::from_vec(features_data, &[4, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[4]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let client_datasets = dataset
            .partition_federated(2, PartitioningStrategy::Uniform, 42)
            .unwrap();
        assert_eq!(client_datasets.len(), 2);
    }
}
