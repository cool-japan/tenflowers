//! LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning
//!
//! LoRA introduces trainable rank decomposition matrices into existing layers,
//! allowing efficient adaptation with minimal additional parameters.
//!
//! Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

use super::{PEFTAdapter, PEFTMethod};
use crate::layers::Dense;
use num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::distributions::{Distribution, Normal};
use scirs2_core::random::rng;
use std::marker::PhantomData;
use tenflowers_core::{ops::matmul, Result, Tensor};

/// Configuration for LoRA adapters
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Rank of the adaptation matrices (r in the paper)
    pub rank: usize,
    /// Scaling factor (alpha in the paper)
    pub alpha: f64,
    /// Dropout probability for LoRA layers
    pub dropout: f64,
    /// Whether to use bias in LoRA layers
    pub use_bias: bool,
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            use_bias: false,
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

    /// Get the scaling factor to apply to LoRA output
    pub fn scaling_factor(&self) -> f64 {
        self.alpha / (self.rank as f64)
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self::new(8, 16.0) // Common defaults from the literature
    }
}

/// Generic LoRA adapter that can be applied to any layer with weight matrices
#[derive(Clone)]
pub struct LoRAAdapter<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Low-rank matrix A (input_dim x rank)
    a_matrix: Tensor<T>,
    /// Low-rank matrix B (rank x output_dim)  
    b_matrix: Tensor<T>,
    /// Optional bias for the LoRA path
    bias: Option<Tensor<T>>,
    /// Configuration
    config: LoRAConfig,
    /// Training mode
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> LoRAAdapter<T>
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
    /// Create a new LoRA adapter
    pub fn new(input_dim: usize, output_dim: usize, config: LoRAConfig) -> Result<Self> {
        // Initialize A with small random values, B with zeros (as in the paper)
        let a_matrix = Self::create_random_matrix(&[input_dim, config.rank], config.rank)?;
        let b_matrix = Tensor::zeros(&[config.rank, output_dim]);

        let bias = if config.use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Ok(Self {
            a_matrix,
            b_matrix,
            bias,
            config,
            training: false,
            _phantom: PhantomData,
        })
    }

    /// Get reference to A matrix
    pub fn a_matrix(&self) -> &Tensor<T> {
        &self.a_matrix
    }

    /// Get reference to B matrix
    pub fn b_matrix(&self) -> &Tensor<T> {
        &self.b_matrix
    }

    /// Get reference to bias (if any)
    pub fn bias(&self) -> Option<&Tensor<T>> {
        self.bias.as_ref()
    }

    /// Get the LoRA configuration
    pub fn config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Helper method to create random matrix with proper LoRA initialization
    /// Initializes with Gaussian distribution with std = 1/sqrt(rank) as per LoRA paper
    fn create_random_matrix(shape: &[usize], rank: usize) -> Result<Tensor<T>> {
        let total_elements = shape.iter().product::<usize>();
        let std_dev = 1.0 / (rank as f64).sqrt();

        let mut rng = rng();
        let normal = Normal::new(0.0, std_dev).map_err(|_| {
            tenflowers_core::TensorError::invalid_argument(
                "Invalid normal distribution parameters".to_string(),
            )
        })?;

        let mut data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let random_val = normal.sample(&mut rng);
            let tensor_val = T::from_f64(random_val).unwrap_or_else(|| T::zero());
            data.push(tensor_val);
        }

        Tensor::from_vec(data, shape)
    }

    /// Compute LoRA adaptation: scaling_factor * (input @ A @ B)
    fn compute_lora_output(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Forward: input @ A @ B
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

impl<T> PEFTAdapter<T> for LoRAAdapter<T>
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
    fn forward(&self, input: &Tensor<T>, base_output: &Tensor<T>) -> Result<Tensor<T>> {
        // Compute LoRA adaptation
        let lora_output = self.compute_lora_output(input)?;

        // Add to base output: base_output + LoRA_adaptation
        base_output.add(&lora_output)
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
        let a_params = self.a_matrix.shape().dims().iter().product::<usize>();
        let b_params = self.b_matrix.shape().dims().iter().product::<usize>();
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
        PEFTMethod::LoRA
    }
}

/// Specialized LoRA layer that wraps any layer type
pub type LoRALayer<T, L> = super::PEFTLayer<T, L, LoRAAdapter<T>>;

/// Convenience type for LoRA-adapted Dense layers
pub type LoRADense<T> = LoRALayer<T, Dense<T>>;

impl<T> LoRADense<T>
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
    /// Create a new LoRA-adapted Dense layer
    pub fn new_lora_dense(
        input_dim: usize,
        output_dim: usize,
        config: LoRAConfig,
        freeze_base: bool,
    ) -> Result<Self> {
        let base_layer = Dense::new(input_dim, output_dim, true);
        let adapter = LoRAAdapter::new(input_dim, output_dim, config)?;

        Ok(super::PEFTLayer::new(base_layer, adapter, freeze_base))
    }

    /// Create with pre-trained base layer
    pub fn from_dense(
        dense_layer: Dense<T>,
        config: LoRAConfig,
        freeze_base: bool,
    ) -> Result<Self> {
        let weight_shape = dense_layer.weight().shape();
        let input_dim = weight_shape.dims()[0];
        let output_dim = weight_shape.dims()[1];

        let adapter = LoRAAdapter::new(input_dim, output_dim, config)?;

        Ok(super::PEFTLayer::new(dense_layer, adapter, freeze_base))
    }
}

/// Helper functions for creating LoRA configurations
impl LoRAConfig {
    /// Configuration optimized for large language models
    pub fn for_llm() -> Self {
        Self::new(16, 32.0).with_dropout(0.1)
    }

    /// Configuration for vision transformers
    pub fn for_vision() -> Self {
        Self::new(8, 16.0).with_dropout(0.0)
    }

    /// Configuration for small models or limited compute
    pub fn for_efficiency() -> Self {
        Self::new(4, 8.0).with_dropout(0.0)
    }

    /// Configuration for maximum adaptation capacity
    pub fn for_high_rank() -> Self {
        Self::new(64, 64.0).with_dropout(0.1).with_bias()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Dense, Layer};

    #[test]
    fn test_lora_config_creation() {
        let config = LoRAConfig::new(8, 16.0);
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling_factor(), 2.0); // 16.0 / 8.0
    }

    #[test]
    fn test_lora_config_presets() {
        let llm_config = LoRAConfig::for_llm();
        assert_eq!(llm_config.rank, 16);
        assert_eq!(llm_config.alpha, 32.0);
        assert_eq!(llm_config.dropout, 0.1);

        let efficiency_config = LoRAConfig::for_efficiency();
        assert_eq!(efficiency_config.rank, 4);
        assert_eq!(efficiency_config.alpha, 8.0);
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoRAConfig::new(4, 8.0);
        let adapter: LoRAAdapter<f32> = LoRAAdapter::new(100, 50, config).unwrap();

        // Check matrix shapes
        assert_eq!(adapter.a_matrix().shape().dims(), &[100, 4]);
        assert_eq!(adapter.b_matrix().shape().dims(), &[4, 50]);

        // Check parameter count
        assert_eq!(adapter.num_trainable_parameters(), 100 * 4 + 4 * 50); // A + B matrices
    }

    #[test]
    fn test_lora_dense_creation() {
        let config = LoRAConfig::new(8, 16.0);
        let lora_dense: LoRADense<f32> = LoRADense::new_lora_dense(128, 64, config, true).unwrap();

        let stats = lora_dense.parameter_efficiency_stats();

        // Base Dense layer: 128*64 + 64 (bias) = 8256 parameters
        // LoRA adapter: 128*8 + 8*64 = 1536 parameters
        // Efficiency ratio: 1536 / (8256 + 1536) ≈ 0.157
        assert!(stats.efficiency_ratio > 0.15 && stats.efficiency_ratio < 0.16);
        assert_eq!(stats.adapter_parameters, 1536);
        assert_eq!(stats.base_parameters, 8256);
    }

    #[test]
    fn test_lora_forward_pass() {
        let config = LoRAConfig::new(4, 8.0);
        let lora_dense: LoRADense<f32> = LoRADense::new_lora_dense(10, 5, config, false).unwrap();

        let input = Tensor::ones(&[2, 10]); // Batch of 2, input dim 10
        let output = lora_dense.forward(&input);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]); // Batch of 2, output dim 5
    }

    #[test]
    fn test_parameter_efficiency() {
        let config = LoRAConfig::new(8, 16.0);
        let lora_dense: LoRADense<f32> =
            LoRADense::new_lora_dense(1000, 1000, config, true).unwrap();

        let stats = lora_dense.parameter_efficiency_stats();
        let summary = stats.summary();

        // With freezing, only adapter parameters are trainable
        assert_eq!(stats.trainable_parameters, stats.adapter_parameters);
        assert!(summary.contains("%"));
        assert!(summary.contains("parameter reduction"));
    }
}
