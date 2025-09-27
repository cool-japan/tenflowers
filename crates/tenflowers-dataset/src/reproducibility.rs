//! Reproducibility utilities for deterministic dataset operations
//!
//! This module provides tools for deterministic ordering, seed management,
//! and environment capture to ensure reproducible ML experiments.

use crate::{Dataset, Result};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Tensor, TensorError};

/// Global seed manager for reproducible operations
static GLOBAL_SEED_MANAGER: std::sync::OnceLock<Arc<Mutex<SeedManager>>> =
    std::sync::OnceLock::new();

/// Seed management for reproducible randomness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedManager {
    /// Master seed for the experiment
    master_seed: u64,
    /// Component-specific seeds
    component_seeds: HashMap<String, u64>,
    /// Current operation counter
    operation_counter: u64,
}

impl SeedManager {
    /// Create a new seed manager with a master seed
    pub fn new(master_seed: u64) -> Self {
        Self {
            master_seed,
            component_seeds: HashMap::new(),
            operation_counter: 0,
        }
    }

    /// Get the master seed
    pub fn master_seed(&self) -> u64 {
        self.master_seed
    }

    /// Get or generate a seed for a specific component
    pub fn get_component_seed(&mut self, component: &str) -> u64 {
        if let Some(&seed) = self.component_seeds.get(component) {
            seed
        } else {
            // Generate deterministic seed from master seed and component name
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            self.master_seed.hash(&mut hasher);
            component.hash(&mut hasher);
            let seed = hasher.finish();
            self.component_seeds.insert(component.to_string(), seed);
            seed
        }
    }

    /// Get a seed for the next operation
    pub fn next_operation_seed(&mut self) -> u64 {
        self.operation_counter += 1;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        self.master_seed.hash(&mut hasher);
        self.operation_counter.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a seeded RNG for a component
    pub fn create_rng(&mut self, component: &str) -> StdRng {
        let seed = self.get_component_seed(component);
        StdRng::seed_from_u64(seed)
    }

    /// Set the global seed manager
    pub fn set_global(manager: SeedManager) {
        let _ = GLOBAL_SEED_MANAGER.set(Arc::new(Mutex::new(manager)));
    }

    /// Get the global seed manager
    pub fn global() -> Arc<Mutex<SeedManager>> {
        GLOBAL_SEED_MANAGER
            .get_or_init(|| Arc::new(Mutex::new(SeedManager::new(42))))
            .clone()
    }

    /// Reset all component seeds (keeping master seed)
    pub fn reset(&mut self) {
        self.component_seeds.clear();
        self.operation_counter = 0;
    }
}

/// Environment information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Rust version
    pub rust_version: String,
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// Number of CPU cores
    pub num_cpus: usize,
    /// Timestamp when captured
    pub timestamp: u64,
    /// Environment variables (selected)
    pub env_vars: HashMap<String, String>,
    /// Rng seed state
    pub seed_info: SeedInfo,
}

/// Seed information for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedInfo {
    /// Master seed
    pub master_seed: u64,
    /// Component seeds
    pub component_seeds: HashMap<String, u64>,
}

impl EnvironmentInfo {
    /// Capture current environment information
    pub fn capture(seed_manager: &SeedManager) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Capture selected environment variables
        let mut env_vars = HashMap::new();
        for var in ["RUST_LOG", "CARGO_TARGET_DIR", "RUSTFLAGS"] {
            if let Ok(value) = std::env::var(var) {
                env_vars.insert(var.to_string(), value);
            }
        }

        Self {
            rust_version: "unknown".to_string(), // RUSTC_VERSION not available at compile time
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            num_cpus: num_cpus::get(),
            timestamp,
            env_vars,
            seed_info: SeedInfo {
                master_seed: seed_manager.master_seed,
                component_seeds: seed_manager.component_seeds.clone(),
            },
        }
    }
}

/// Deterministic dataset wrapper that ensures reproducible ordering
#[derive(Debug)]
pub struct DeterministicDataset<T, D> {
    dataset: D,
    indices: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D> DeterministicDataset<T, D>
where
    D: Dataset<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a deterministic dataset with a specific seed
    pub fn new(dataset: D, seed: u64) -> Self {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();

        // Create deterministic shuffle
        let mut rng = StdRng::seed_from_u64(seed);
        Self::fisher_yates_shuffle(&mut indices, &mut rng);

        Self {
            dataset,
            indices,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a deterministic dataset with sequential ordering
    pub fn sequential(dataset: D) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();

        Self {
            dataset,
            indices,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a deterministic dataset with reverse ordering
    pub fn reverse(dataset: D) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).rev().collect();

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

    /// Get the index mapping
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Reshuffle with a new seed
    pub fn reshuffle(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::fisher_yates_shuffle(&mut self.indices, &mut rng);
    }

    fn fisher_yates_shuffle<R: Rng>(indices: &mut [usize], rng: &mut R) {
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..i);
            indices.swap(i, j);
        }
    }
}

impl<T, D> Dataset<T> for DeterministicDataset<T, D>
where
    D: Dataset<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.indices.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.indices.len()
            )));
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

/// Reproducible experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment name
    pub name: String,
    /// Master seed for reproducibility
    pub seed: u64,
    /// Dataset configuration
    pub dataset_config: DatasetConfig,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Dataset configuration for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Ordering strategy
    pub ordering: OrderingStrategy,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Transform configuration
    pub transforms: Vec<TransformConfig>,
}

/// Ordering strategy for deterministic datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderingStrategy {
    /// Sequential ordering (0, 1, 2, ...)
    Sequential,
    /// Reverse ordering (n-1, n-2, ..., 0)
    Reverse,
    /// Shuffled with specific seed
    Shuffled { seed: u64 },
    /// Custom ordering with specific indices
    Custom { indices: Vec<usize> },
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling strategy
    pub strategy: String,
    /// Seed for sampling
    pub seed: u64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Transform configuration for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    /// Transform name
    pub name: String,
    /// Seed for random operations
    pub seed: u64,
    /// Transform parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Deterministic ordering utilities
pub struct DeterministicOrdering;

impl DeterministicOrdering {
    /// Create deterministic indices for a dataset
    pub fn create_indices(len: usize, strategy: &OrderingStrategy) -> Vec<usize> {
        match strategy {
            OrderingStrategy::Sequential => (0..len).collect(),
            OrderingStrategy::Reverse => (0..len).rev().collect(),
            OrderingStrategy::Shuffled { seed } => {
                let mut indices: Vec<usize> = (0..len).collect();
                let mut rng = StdRng::seed_from_u64(*seed);
                Self::shuffle_indices(&mut indices, &mut rng);
                indices
            }
            OrderingStrategy::Custom { indices } => {
                // Validate and clamp indices to dataset size
                indices
                    .iter()
                    .map(|&i| i.min(len.saturating_sub(1)))
                    .collect()
            }
        }
    }

    /// Shuffle indices using Fisher-Yates algorithm
    pub fn shuffle_indices<R: Rng>(indices: &mut [usize], rng: &mut R) {
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..i);
            indices.swap(i, j);
        }
    }

    /// Create stratified deterministic ordering (for f32 datasets only)
    pub fn create_stratified_indices_f32(
        dataset: &dyn Dataset<f32>,
        seed: u64,
        num_classes: usize,
    ) -> Result<Vec<usize>> {
        // Group samples by class
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); num_classes];

        for i in 0..dataset.len() {
            let (_, labels) = dataset.get(i)?;

            // Extract class from label tensor (assume f32 type)
            let class = if labels.is_scalar() {
                labels.get(&[]).unwrap_or(0.0) as usize
            } else if let Some(slice) = labels.as_slice() {
                slice.first().copied().unwrap_or(0.0) as usize
            } else {
                0
            };

            if class < num_classes {
                class_indices[class].push(i);
            }
        }

        // Shuffle each class separately with deterministic seeds
        let mut rng = StdRng::seed_from_u64(seed);
        let mut result = Vec::new();

        for class_samples in &mut class_indices {
            Self::shuffle_indices(class_samples, &mut rng);
            result.extend_from_slice(class_samples);
        }

        Ok(result)
    }
}

/// Extension trait for adding reproducibility to datasets
pub trait ReproducibilityExt<T>: Dataset<T> + Sized
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Make the dataset deterministic with a seed
    fn deterministic(self, seed: u64) -> DeterministicDataset<T, Self> {
        DeterministicDataset::new(self, seed)
    }

    /// Make the dataset sequential
    fn sequential(self) -> DeterministicDataset<T, Self> {
        DeterministicDataset::sequential(self)
    }

    /// Make the dataset reverse ordered
    fn reverse(self) -> DeterministicDataset<T, Self> {
        DeterministicDataset::reverse(self)
    }
}

impl<T, D: Dataset<T>> ReproducibilityExt<T> for D where T: Clone + Default + Send + Sync + 'static {}

/// Experiment tracker for reproducibility
#[derive(Debug)]
pub struct ExperimentTracker {
    config: ExperimentConfig,
    start_time: std::time::Instant,
    operations: Vec<OperationRecord>,
}

/// Record of an operation for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRecord {
    /// Operation name
    pub name: String,
    /// Timestamp
    pub timestamp: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Seed used
    pub seed: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ExperimentTracker {
    /// Create a new experiment tracker
    pub fn new(config: ExperimentConfig) -> Self {
        Self {
            config,
            start_time: std::time::Instant::now(),
            operations: Vec::new(),
        }
    }

    /// Record an operation
    pub fn record_operation(
        &mut self,
        name: String,
        duration: std::time::Duration,
        seed: u64,
        metadata: HashMap<String, String>,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let record = OperationRecord {
            name,
            timestamp,
            duration_ms: duration.as_millis() as u64,
            seed,
            metadata,
        };

        self.operations.push(record);
    }

    /// Get the experiment configuration
    pub fn config(&self) -> &ExperimentConfig {
        &self.config
    }

    /// Get all recorded operations
    pub fn operations(&self) -> &[OperationRecord] {
        &self.operations
    }

    /// Save experiment to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let experiment_data = ExperimentData {
            config: self.config.clone(),
            operations: self.operations.clone(),
            total_duration_ms: self.start_time.elapsed().as_millis() as u64,
        };

        let json_data = serde_json::to_string_pretty(&experiment_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize experiment data: {e}"))
        })?;

        std::fs::write(path, json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to write experiment file: {e}"))
        })?;

        Ok(())
    }

    /// Load experiment from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json_data = std::fs::read_to_string(path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read experiment file: {e}"))
        })?;

        let experiment_data: ExperimentData = serde_json::from_str(&json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse experiment JSON: {e}"))
        })?;

        Ok(Self {
            config: experiment_data.config,
            start_time: std::time::Instant::now(), // Reset start time
            operations: experiment_data.operations,
        })
    }
}

/// Serializable experiment data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentData {
    config: ExperimentConfig,
    operations: Vec<OperationRecord>,
    total_duration_ms: u64,
}

/// Helper functions for deterministic operations
pub struct DeterministicOps;

impl DeterministicOps {
    /// Set global seed for reproducibility
    pub fn set_global_seed(seed: u64) {
        SeedManager::set_global(SeedManager::new(seed));
    }

    /// Get a deterministic RNG for a component
    pub fn get_rng(component: &str) -> StdRng {
        let manager = SeedManager::global();
        let mut manager = manager.lock().unwrap();
        manager.create_rng(component)
    }

    /// Get next operation seed
    pub fn next_operation_seed() -> u64 {
        let manager = SeedManager::global();
        let mut manager = manager.lock().unwrap();
        manager.next_operation_seed()
    }

    /// Capture current environment
    pub fn capture_environment() -> EnvironmentInfo {
        let manager = SeedManager::global();
        let manager = manager.lock().unwrap();
        EnvironmentInfo::capture(&manager)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tempfile::TempDir;

    #[test]
    fn test_seed_manager() {
        let mut manager = SeedManager::new(42);

        assert_eq!(manager.master_seed(), 42);

        // Component seeds should be deterministic
        let seed1 = manager.get_component_seed("test");
        let seed2 = manager.get_component_seed("test");
        assert_eq!(seed1, seed2);

        let seed3 = manager.get_component_seed("other");
        assert_ne!(seed1, seed3);

        // Operation seeds should be different each time
        let op1 = manager.next_operation_seed();
        let op2 = manager.next_operation_seed();
        assert_ne!(op1, op2);
    }

    #[test]
    fn test_deterministic_dataset() {
        // Create test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let labels_data = vec![0.0, 1.0, 0.0];
        let features = Tensor::from_vec(features_data, &[3, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[3]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Create deterministic dataset
        let det_dataset = DeterministicDataset::new(dataset, 42);

        assert_eq!(det_dataset.len(), 3);

        // Order should be deterministic with same seed
        let det_dataset2 = DeterministicDataset::new(det_dataset.inner().clone(), 42);
        assert_eq!(det_dataset.indices(), det_dataset2.indices());

        // Different seed should produce different order
        let det_dataset3 = DeterministicDataset::new(det_dataset.inner().clone(), 123);
        assert_ne!(det_dataset.indices(), det_dataset3.indices());
    }

    #[test]
    fn test_ordering_strategies() {
        let len = 5;

        // Sequential
        let seq_indices = DeterministicOrdering::create_indices(len, &OrderingStrategy::Sequential);
        assert_eq!(seq_indices, vec![0, 1, 2, 3, 4]);

        // Reverse
        let rev_indices = DeterministicOrdering::create_indices(len, &OrderingStrategy::Reverse);
        assert_eq!(rev_indices, vec![4, 3, 2, 1, 0]);

        // Shuffled should be deterministic
        let shuffled1 =
            DeterministicOrdering::create_indices(len, &OrderingStrategy::Shuffled { seed: 42 });
        let shuffled2 =
            DeterministicOrdering::create_indices(len, &OrderingStrategy::Shuffled { seed: 42 });
        assert_eq!(shuffled1, shuffled2);

        // Different seeds should produce different results
        let shuffled3 =
            DeterministicOrdering::create_indices(len, &OrderingStrategy::Shuffled { seed: 123 });
        assert_ne!(shuffled1, shuffled3);

        // Custom indices
        let custom_indices = DeterministicOrdering::create_indices(
            len,
            &OrderingStrategy::Custom {
                indices: vec![2, 0, 4, 1, 3],
            },
        );
        assert_eq!(custom_indices, vec![2, 0, 4, 1, 3]);
    }

    #[test]
    fn test_environment_capture() {
        let manager = SeedManager::new(42);
        let env = EnvironmentInfo::capture(&manager);

        assert!(!env.rust_version.is_empty());
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert!(env.num_cpus > 0);
        assert_eq!(env.seed_info.master_seed, 42);
    }

    #[test]
    fn test_experiment_tracker() {
        let config = ExperimentConfig {
            name: "test_experiment".to_string(),
            seed: 42,
            dataset_config: DatasetConfig {
                ordering: OrderingStrategy::Shuffled { seed: 42 },
                sampling: SamplingConfig {
                    strategy: "random".to_string(),
                    seed: 42,
                    parameters: HashMap::new(),
                },
                transforms: Vec::new(),
            },
            environment: EnvironmentInfo::capture(&SeedManager::new(42)),
            metadata: HashMap::new(),
        };

        let mut tracker = ExperimentTracker::new(config);

        // Record an operation
        tracker.record_operation(
            "data_loading".to_string(),
            std::time::Duration::from_millis(100),
            42,
            HashMap::new(),
        );

        assert_eq!(tracker.operations().len(), 1);
        assert_eq!(tracker.operations()[0].name, "data_loading");
        assert_eq!(tracker.operations()[0].duration_ms, 100);

        // Test file save/load
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("experiment.json");

        tracker.save_to_file(&file_path).unwrap();
        let loaded_tracker = ExperimentTracker::load_from_file(&file_path).unwrap();

        assert_eq!(loaded_tracker.config().name, "test_experiment");
        assert_eq!(loaded_tracker.operations().len(), 1);
    }

    #[test]
    fn test_reproducibility_ext() {
        // Create test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0];
        let labels_data = vec![0.0, 1.0];
        let features = Tensor::from_vec(features_data, &[2, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Test extension methods
        let det_dataset = dataset.deterministic(42);
        assert_eq!(det_dataset.len(), 2);

        let seq_dataset = det_dataset.inner().clone().sequential();
        assert_eq!(seq_dataset.indices(), &[0, 1]);

        let rev_dataset = det_dataset.inner().clone().reverse();
        assert_eq!(rev_dataset.indices(), &[1, 0]);
    }

    #[test]
    fn test_deterministic_ops() {
        // Set global seed
        DeterministicOps::set_global_seed(12345);

        // Get RNG for component
        let mut rng1 = DeterministicOps::get_rng("test_component");
        let val1: f64 = rng1.random();

        // Same component should produce same initial value
        let mut rng2 = DeterministicOps::get_rng("test_component");
        let val2: f64 = rng2.random();
        assert_eq!(val1, val2);

        // Different component should produce different value
        let mut rng3 = DeterministicOps::get_rng("other_component");
        let val3: f64 = rng3.random();
        assert_ne!(val1, val3);

        // Operation seeds should be different
        let op1 = DeterministicOps::next_operation_seed();
        let op2 = DeterministicOps::next_operation_seed();
        assert_ne!(op1, op2);

        // Environment capture should work
        let env = DeterministicOps::capture_environment();
        assert_eq!(env.seed_info.master_seed, 12345);
    }
}
