//! AdaLoRA (Adaptive LoRA) implementation for dynamic rank allocation
//!
//! AdaLoRA builds upon LoRA by adaptively allocating the rank of the low-rank
//! decomposition based on the importance of different weight matrices during training.
//! This leads to better parameter utilization and improved performance.
//!
//! Reference: "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (Zhang et al., 2023)

use super::{lora::LoRAConfig, PEFTAdapter, PEFTMethod};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use scirs2_core::random::distributions::{Distribution, Normal};
use scirs2_core::random::rng;
use std::marker::PhantomData;
use tenflowers_core::{ops::matmul, Result, Tensor, TensorError};

/// Configuration for AdaLoRA adapters
#[derive(Debug, Clone)]
pub struct AdaLoRAConfig {
    /// Initial rank (will be adapted during training)
    pub init_rank: usize,
    /// Target rank (final rank after adaptation)
    pub target_rank: usize,
    /// Scaling factor (alpha parameter)
    pub alpha: f64,
    /// Dropout probability for LoRA layers
    pub dropout: f64,
    /// Whether to use bias in LoRA layers
    pub use_bias: bool,
    /// Budget allocation ratio (how aggressively to reallocate rank)
    pub budget_ratio: f64,
    /// Rank allocation update frequency (in training steps)
    pub update_frequency: usize,
    /// Importance metric type for rank allocation
    pub importance_metric: ImportanceMetric,
    /// Pruning threshold for removing unimportant dimensions
    pub pruning_threshold: f64,
}

/// Types of importance metrics for rank allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportanceMetric {
    /// Magnitude-based importance (L2 norm of singular values)
    Magnitude,
    /// Gradient-based importance (accumulated gradient magnitudes)
    Gradient,
    /// Sensitivity-based importance (parameter sensitivity to loss)
    Sensitivity,
    /// Fisher information-based importance
    Fisher,
}

impl AdaLoRAConfig {
    /// Create a new AdaLoRA configuration
    pub fn new(init_rank: usize, target_rank: usize, alpha: f64) -> Self {
        Self {
            init_rank,
            target_rank,
            alpha,
            dropout: 0.0,
            use_bias: false,
            budget_ratio: 0.3,
            update_frequency: 200,
            importance_metric: ImportanceMetric::Magnitude,
            pruning_threshold: 0.01,
        }
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable bias in LoRA layers
    pub fn with_bias(mut self) -> Self {
        self.use_bias = true;
        self
    }

    /// Set budget allocation ratio
    pub fn with_budget_ratio(mut self, ratio: f64) -> Self {
        self.budget_ratio = ratio;
        self
    }

    /// Set rank update frequency
    pub fn with_update_frequency(mut self, frequency: usize) -> Self {
        self.update_frequency = frequency;
        self
    }

    /// Set importance metric type
    pub fn with_importance_metric(mut self, metric: ImportanceMetric) -> Self {
        self.importance_metric = metric;
        self
    }

    /// Set pruning threshold
    pub fn with_pruning_threshold(mut self, threshold: f64) -> Self {
        self.pruning_threshold = threshold;
        self
    }

    /// Get the scaling factor to apply to AdaLoRA output
    pub fn scaling_factor(&self) -> f64 {
        self.alpha / (self.init_rank as f64)
    }

    /// Convert to base LoRA config for compatibility
    pub fn to_lora_config(&self) -> LoRAConfig {
        let mut config = LoRAConfig::new(self.init_rank, self.alpha);
        if self.dropout > 0.0 {
            config = config.with_dropout(self.dropout);
        }
        if self.use_bias {
            config = config.with_bias();
        }
        config
    }
}

impl Default for AdaLoRAConfig {
    fn default() -> Self {
        Self::new(16, 8, 16.0) // Start with rank 16, adapt to 8
    }
}

/// Importance scores for each rank dimension
#[derive(Clone)]
struct ImportanceScores<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Importance scores for A matrix dimensions
    a_scores: Vec<T>,
    /// Importance scores for B matrix dimensions  
    b_scores: Vec<T>,
    /// Combined importance scores (A ⊗ B)
    combined_scores: Vec<T>,
    /// Accumulated gradient magnitudes for gradient-based importance
    grad_accumulator: Option<Vec<T>>,
    /// Update counter for tracking when to recompute importance
    update_count: usize,
}

impl<T> ImportanceScores<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create new importance tracking
    fn new(rank: usize) -> Self {
        Self {
            a_scores: vec![T::one(); rank],
            b_scores: vec![T::one(); rank],
            combined_scores: vec![T::one(); rank],
            grad_accumulator: Some(vec![T::zero(); rank]),
            update_count: 0,
        }
    }

    /// Update importance scores based on current parameters
    fn update_importance(
        &mut self,
        a_matrix: &Tensor<T>,
        b_matrix: &Tensor<T>,
        metric: &ImportanceMetric,
    ) -> Result<()> {
        match metric {
            ImportanceMetric::Magnitude => self.update_magnitude_importance(a_matrix, b_matrix),
            ImportanceMetric::Gradient => self.update_gradient_importance(a_matrix, b_matrix),
            ImportanceMetric::Sensitivity => self.update_sensitivity_importance(a_matrix, b_matrix),
            ImportanceMetric::Fisher => self.update_fisher_importance(a_matrix, b_matrix),
        }
    }

    /// Magnitude-based importance using L2 norms
    fn update_magnitude_importance(
        &mut self,
        a_matrix: &Tensor<T>,
        b_matrix: &Tensor<T>,
    ) -> Result<()> {
        let a_data = a_matrix.to_vec()?;
        let b_data = b_matrix.to_vec()?;
        let a_shape = a_matrix.shape().dims();
        let b_shape = b_matrix.shape().dims();

        let rank = a_shape[1];

        // Compute L2 norm of each rank dimension in A matrix
        for r in 0..rank {
            let mut norm_sq = T::zero();
            for i in 0..a_shape[0] {
                let idx = i * rank + r;
                if idx < a_data.len() {
                    let val = a_data[idx];
                    norm_sq = norm_sq + val * val;
                }
            }
            self.a_scores[r] = norm_sq.sqrt();
        }

        // Compute L2 norm of each rank dimension in B matrix
        for r in 0..rank {
            let mut norm_sq = T::zero();
            for j in 0..b_shape[1] {
                let idx = r * b_shape[1] + j;
                if idx < b_data.len() {
                    let val = b_data[idx];
                    norm_sq = norm_sq + val * val;
                }
            }
            self.b_scores[r] = norm_sq.sqrt();
        }

        // Combined importance is the product of A and B importance
        for r in 0..rank {
            self.combined_scores[r] = self.a_scores[r] * self.b_scores[r];
        }

        Ok(())
    }

    /// Gradient-based importance (simplified - would need actual gradients in real implementation)
    fn update_gradient_importance(
        &mut self,
        a_matrix: &Tensor<T>,
        b_matrix: &Tensor<T>,
    ) -> Result<()> {
        // For now, fall back to magnitude-based importance
        // In a real implementation, this would use accumulated gradients
        self.update_magnitude_importance(a_matrix, b_matrix)?;

        // Simulate gradient accumulation decay
        if let Some(ref mut accumulator) = self.grad_accumulator {
            for score in accumulator.iter_mut() {
                *score = *score * T::from_f64(0.9).unwrap_or_else(|| T::one());
            }
        }

        Ok(())
    }

    /// Sensitivity-based importance (simplified)
    fn update_sensitivity_importance(
        &mut self,
        a_matrix: &Tensor<T>,
        b_matrix: &Tensor<T>,
    ) -> Result<()> {
        // For now, use magnitude as a proxy for sensitivity
        self.update_magnitude_importance(a_matrix, b_matrix)
    }

    /// Fisher information-based importance (simplified)
    fn update_fisher_importance(
        &mut self,
        a_matrix: &Tensor<T>,
        b_matrix: &Tensor<T>,
    ) -> Result<()> {
        // For now, use magnitude as a proxy for Fisher information
        self.update_magnitude_importance(a_matrix, b_matrix)
    }

    /// Get indices of the most important rank dimensions
    fn get_top_k_indices(&self, k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.combined_scores.len()).collect();

        // Sort by importance (descending)
        indices.sort_by(|&a, &b| {
            self.combined_scores[b]
                .partial_cmp(&self.combined_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        indices.into_iter().take(k).collect()
    }

    /// Check if importance scores should be updated
    fn should_update(&self, frequency: usize) -> bool {
        self.update_count % frequency == 0
    }

    /// Increment update counter
    fn increment_update(&mut self) {
        self.update_count += 1;
    }
}

/// AdaLoRA adapter with dynamic rank allocation
#[derive(Clone)]
pub struct AdaLoRAAdapter<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Low-rank matrix A (input_dim x current_rank)
    a_matrix: Tensor<T>,
    /// Low-rank matrix B (current_rank x output_dim)
    b_matrix: Tensor<T>,
    /// Optional bias for the LoRA path
    bias: Option<Tensor<T>>,
    /// Current effective rank (may be less than matrix dimensions)
    current_rank: usize,
    /// Mask for active rank dimensions
    rank_mask: Vec<bool>,
    /// Importance tracking
    importance: ImportanceScores<T>,
    /// Configuration
    config: AdaLoRAConfig,
    /// Training mode
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> AdaLoRAAdapter<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new AdaLoRA adapter
    pub fn new(input_dim: usize, output_dim: usize, config: AdaLoRAConfig) -> Result<Self> {
        // Initialize with maximum rank
        let init_rank = config.init_rank;
        let a_matrix = Self::create_random_matrix(&[input_dim, init_rank], init_rank)?;
        let b_matrix = Tensor::zeros(&[init_rank, output_dim]);

        let bias = if config.use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        let rank_mask = vec![true; init_rank];
        let importance = ImportanceScores::new(init_rank);

        Ok(Self {
            a_matrix,
            b_matrix,
            bias,
            current_rank: init_rank,
            rank_mask,
            importance,
            config,
            training: false,
            _phantom: PhantomData,
        })
    }

    /// Adapt the rank based on current importance scores
    pub fn adapt_rank(&mut self) -> Result<RankAdaptationStats> {
        if !self.training || !self.importance.should_update(self.config.update_frequency) {
            return Ok(RankAdaptationStats::no_change(self.current_rank));
        }

        // Update importance scores
        self.importance.update_importance(
            &self.a_matrix,
            &self.b_matrix,
            &self.config.importance_metric,
        )?;
        self.importance.increment_update();

        let old_rank = self.current_rank;

        // Determine new rank based on budget and importance
        let target_rank = self.config.target_rank;
        let budget_factor = if old_rank > target_rank {
            // Gradually reduce rank
            1.0 - self.config.budget_ratio
        } else {
            // Can increase rank if needed
            1.0 + self.config.budget_ratio
        };

        let new_rank = ((old_rank as f64 * budget_factor) as usize)
            .clamp(1, self.config.init_rank)
            .min(target_rank);

        if new_rank != old_rank {
            // Get indices of most important dimensions
            let important_indices = self.importance.get_top_k_indices(new_rank);

            // Update rank mask
            self.rank_mask.fill(false);
            for &idx in &important_indices {
                if idx < self.rank_mask.len() {
                    self.rank_mask[idx] = true;
                }
            }

            self.current_rank = new_rank;

            // Prune unimportant dimensions by zeroing them out
            self.apply_rank_mask()?;

            Ok(RankAdaptationStats {
                old_rank,
                new_rank,
                pruned_dimensions: old_rank - new_rank,
                adaptation_occurred: true,
            })
        } else {
            Ok(RankAdaptationStats::no_change(old_rank))
        }
    }

    /// Apply rank mask to zero out pruned dimensions
    fn apply_rank_mask(&mut self) -> Result<()> {
        // Zero out masked dimensions in A matrix
        let mut a_data = self.a_matrix.to_vec()?;
        let a_shape = self.a_matrix.shape().dims();
        let input_dim = a_shape[0];
        let rank = a_shape[1];

        for r in 0..rank {
            if !self.rank_mask[r] {
                // Zero out this rank dimension
                for i in 0..input_dim {
                    let idx = i * rank + r;
                    if idx < a_data.len() {
                        a_data[idx] = T::zero();
                    }
                }
            }
        }
        self.a_matrix = Tensor::from_vec(a_data, a_shape)?;

        // Zero out masked dimensions in B matrix
        let mut b_data = self.b_matrix.to_vec()?;
        let b_shape = self.b_matrix.shape().dims();
        let output_dim = b_shape[1];

        for r in 0..rank {
            if !self.rank_mask[r] {
                // Zero out this rank dimension
                for j in 0..output_dim {
                    let idx = r * output_dim + j;
                    if idx < b_data.len() {
                        b_data[idx] = T::zero();
                    }
                }
            }
        }
        self.b_matrix = Tensor::from_vec(b_data, b_shape)?;

        Ok(())
    }

    /// Get current adaptation statistics
    pub fn adaptation_stats(&self) -> AdaLoRAStats {
        let effective_rank = self.rank_mask.iter().filter(|&&mask| mask).count();
        let sparsity = 1.0 - (effective_rank as f64 / self.config.init_rank as f64);

        AdaLoRAStats {
            initial_rank: self.config.init_rank,
            current_rank: self.current_rank,
            effective_rank,
            target_rank: self.config.target_rank,
            sparsity_ratio: sparsity,
            importance_metric: self.config.importance_metric.clone(),
        }
    }

    /// Helper method to create random matrix with proper initialization
    fn create_random_matrix(shape: &[usize], rank: usize) -> Result<Tensor<T>> {
        let total_elements = shape.iter().product::<usize>();
        let std_dev = 1.0 / (rank as f64).sqrt();

        let mut rng = rng();
        let normal = Normal::new(0.0, std_dev).map_err(|_| {
            TensorError::invalid_argument("Invalid normal distribution parameters".to_string())
        })?;

        let mut data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let random_val = normal.sample(&mut rng);
            let tensor_val = T::from_f64(random_val).unwrap_or_else(|| T::zero());
            data.push(tensor_val);
        }

        Tensor::from_vec(data, shape)
    }

    /// Compute AdaLoRA adaptation with rank masking
    fn compute_adalora_output(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Forward: input @ A @ B (only through active rank dimensions)
        let temp = matmul(input, &self.a_matrix)?;
        let lora_output = matmul(&temp, &self.b_matrix)?;

        // Apply scaling factor
        let scaling = T::from(self.config.scaling_factor()).unwrap_or_else(|| T::one());
        let scaled_output = lora_output.scalar_mul(scaling)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            scaled_output.add(bias)
        } else {
            Ok(scaled_output)
        }
    }
}

impl<T> PEFTAdapter<T> for AdaLoRAAdapter<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>, base_output: &Tensor<T>) -> Result<Tensor<T>> {
        // Compute AdaLoRA adaptation
        let adalora_output = self.compute_adalora_output(input)?;

        // Add to base output
        base_output.add(&adalora_output)
    }

    fn trainable_parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.a_matrix, &self.b_matrix];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.a_matrix, &mut self.b_matrix];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn num_trainable_parameters(&self) -> usize {
        // Only count parameters for active rank dimensions
        let a_params = self.a_matrix.shape().dims()[0] * self.current_rank;
        let b_params = self.current_rank * self.b_matrix.shape().dims()[1];
        let bias_params = self
            .bias
            .as_ref()
            .map(|b| b.shape().dims().iter().product::<usize>())
            .unwrap_or(0);

        a_params + b_params + bias_params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn method_type(&self) -> PEFTMethod {
        PEFTMethod::AdaLoRA
    }
}

/// Statistics about rank adaptation
#[derive(Debug, Clone)]
pub struct RankAdaptationStats {
    pub old_rank: usize,
    pub new_rank: usize,
    pub pruned_dimensions: usize,
    pub adaptation_occurred: bool,
}

impl RankAdaptationStats {
    fn no_change(rank: usize) -> Self {
        Self {
            old_rank: rank,
            new_rank: rank,
            pruned_dimensions: 0,
            adaptation_occurred: false,
        }
    }
}

/// Overall AdaLoRA statistics
#[derive(Debug, Clone)]
pub struct AdaLoRAStats {
    pub initial_rank: usize,
    pub current_rank: usize,
    pub effective_rank: usize,
    pub target_rank: usize,
    pub sparsity_ratio: f64,
    pub importance_metric: ImportanceMetric,
}

impl AdaLoRAStats {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "AdaLoRA: {}/{} rank ({:.1}% sparse), target: {}, metric: {:?}",
            self.effective_rank,
            self.initial_rank,
            self.sparsity_ratio * 100.0,
            self.target_rank,
            self.importance_metric
        )
    }
}

/// Predefined AdaLoRA configurations
impl AdaLoRAConfig {
    /// Configuration for aggressive adaptation with high sparsity
    pub fn for_efficiency() -> Self {
        Self::new(32, 8, 32.0)
            .with_budget_ratio(0.5)
            .with_update_frequency(100)
            .with_importance_metric(ImportanceMetric::Magnitude)
            .with_pruning_threshold(0.05)
    }

    /// Configuration for large language models
    pub fn for_llm() -> Self {
        Self::new(64, 16, 64.0)
            .with_budget_ratio(0.3)
            .with_update_frequency(200)
            .with_importance_metric(ImportanceMetric::Gradient)
            .with_dropout(0.1)
    }

    /// Configuration for vision transformers
    pub fn for_vision() -> Self {
        Self::new(32, 16, 32.0)
            .with_budget_ratio(0.4)
            .with_update_frequency(150)
            .with_importance_metric(ImportanceMetric::Magnitude)
    }

    /// Configuration for research with conservative adaptation
    pub fn for_research() -> Self {
        Self::new(128, 32, 128.0)
            .with_budget_ratio(0.2)
            .with_update_frequency(500)
            .with_importance_metric(ImportanceMetric::Fisher)
            .with_bias()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Dense, Layer};

    #[test]
    fn test_adalora_config_creation() {
        let config = AdaLoRAConfig::new(16, 8, 16.0);
        assert_eq!(config.init_rank, 16);
        assert_eq!(config.target_rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling_factor(), 1.0); // 16.0 / 16.0
    }

    #[test]
    fn test_adalora_config_presets() {
        let efficiency_config = AdaLoRAConfig::for_efficiency();
        assert_eq!(efficiency_config.init_rank, 32);
        assert_eq!(efficiency_config.target_rank, 8);
        assert_eq!(
            efficiency_config.importance_metric,
            ImportanceMetric::Magnitude
        );

        let llm_config = AdaLoRAConfig::for_llm();
        assert_eq!(llm_config.init_rank, 64);
        assert_eq!(llm_config.target_rank, 16);
        assert_eq!(llm_config.importance_metric, ImportanceMetric::Gradient);
    }

    #[test]
    fn test_adalora_adapter_creation() {
        let config = AdaLoRAConfig::new(8, 4, 16.0);
        let adapter: AdaLoRAAdapter<f32> = AdaLoRAAdapter::new(100, 50, config).unwrap();

        // Check initial state
        assert_eq!(adapter.current_rank, 8);
        assert_eq!(adapter.rank_mask.len(), 8);
        assert!(adapter.rank_mask.iter().all(|&mask| mask)); // All dimensions active initially

        // Check matrix shapes
        assert_eq!(adapter.a_matrix.shape().dims(), &[100, 8]);
        assert_eq!(adapter.b_matrix.shape().dims(), &[8, 50]);
    }

    #[test]
    fn test_importance_scores() {
        let mut importance: ImportanceScores<f32> = ImportanceScores::new(4);

        // Test initial state
        assert_eq!(importance.a_scores.len(), 4);
        assert_eq!(importance.b_scores.len(), 4);
        assert_eq!(importance.combined_scores.len(), 4);

        // Test top-k selection
        importance.combined_scores = vec![0.1, 0.8, 0.3, 0.6];
        let top_2 = importance.get_top_k_indices(2);
        assert_eq!(top_2, vec![1, 3]); // Indices of 0.8 and 0.6
    }

    #[test]
    fn test_rank_adaptation() {
        let config = AdaLoRAConfig::new(8, 4, 16.0).with_update_frequency(1); // Update every step for testing
        let mut adapter: AdaLoRAAdapter<f32> = AdaLoRAAdapter::new(10, 5, config).unwrap();
        adapter.set_training(true);

        // Force some importance pattern (manually set for testing)
        adapter.importance.combined_scores = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];

        let stats = adapter.adapt_rank().unwrap();

        // Should have adapted the rank
        if stats.adaptation_occurred {
            assert!(stats.new_rank <= stats.old_rank);
            assert!(adapter.current_rank <= 8);
        }
    }

    #[test]
    fn test_adalora_forward_pass() {
        let config = AdaLoRAConfig::new(4, 2, 8.0);
        let adapter: AdaLoRAAdapter<f32> = AdaLoRAAdapter::new(10, 5, config).unwrap();

        let input = Tensor::ones(&[2, 10]); // Batch of 2, input dim 10
        let base_output = Tensor::zeros(&[2, 5]); // Batch of 2, output dim 5

        let result = adapter.forward(&input, &base_output);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_parameter_counting() {
        let config = AdaLoRAConfig::new(8, 4, 16.0);
        let adapter: AdaLoRAAdapter<f32> = AdaLoRAAdapter::new(100, 50, config).unwrap();

        // Initial parameter count: 100*8 + 8*50 = 1200
        assert_eq!(adapter.num_trainable_parameters(), 1200);

        let stats = adapter.adaptation_stats();
        assert_eq!(stats.initial_rank, 8);
        assert_eq!(stats.current_rank, 8);
        assert_eq!(stats.effective_rank, 8);
    }

    #[test]
    fn test_lora_config_conversion() {
        let adalora_config = AdaLoRAConfig::new(16, 8, 32.0)
            .with_dropout(0.1)
            .with_bias();

        let lora_config = adalora_config.to_lora_config();
        assert_eq!(lora_config.rank, 16);
        assert_eq!(lora_config.alpha, 32.0);
        assert_eq!(lora_config.dropout, 0.1);
        assert!(lora_config.use_bias);
    }
}
