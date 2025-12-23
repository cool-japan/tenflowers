//! IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) implementation
//!
//! IA³ introduces trainable scaling vectors that modulate activations through
//! element-wise multiplication, requiring significantly fewer parameters than LoRA
//! while achieving competitive performance on many tasks.
//!
//! Reference: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (Liu et al., 2022)

use super::{PEFTAdapter, PEFTMethod};
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Random;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for IA³ adapters
#[derive(Debug, Clone)]
pub struct IA3Config {
    /// Whether to use learnable scaling for intermediate activations
    pub scale_activations: bool,
    /// Whether to use learnable scaling for attention weights
    pub scale_attention: bool,
    /// Whether to use learnable scaling for feed-forward activations
    pub scale_feedforward: bool,
    /// Initialization strategy for scaling factors
    pub init_strategy: IA3InitStrategy,
    /// Initial scaling value (used for constant initialization)
    pub init_value: f64,
    /// Dropout probability for scaling factors (applied during training)
    pub dropout: f64,
    /// Whether to use separate scaling for each head in multi-head attention
    pub per_head_scaling: bool,
    /// Learning rate multiplier for IA³ parameters (often set higher than base model)
    pub learning_rate_multiplier: f64,
}

/// Initialization strategies for IA³ scaling factors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IA3InitStrategy {
    /// Initialize all scaling factors to 1.0 (identity)
    Ones,
    /// Initialize all scaling factors to a constant value
    Constant,
    /// Initialize with small random values around 1.0
    RandomNear1,
    /// Initialize with Xavier/Glorot normal distribution
    Xavier,
    /// Initialize with task-specific values (0.1 for attention, 1.0 for FFN)
    TaskSpecific,
}

impl IA3Config {
    /// Create a new IA³ configuration
    pub fn new() -> Self {
        Self {
            scale_activations: true,
            scale_attention: true,
            scale_feedforward: true,
            init_strategy: IA3InitStrategy::Ones,
            init_value: 1.0,
            dropout: 0.0,
            per_head_scaling: false,
            learning_rate_multiplier: 10.0, // IA³ typically needs higher LR
        }
    }

    /// Enable/disable activation scaling
    pub fn with_activation_scaling(mut self, enable: bool) -> Self {
        self.scale_activations = enable;
        self
    }

    /// Enable/disable attention scaling
    pub fn with_attention_scaling(mut self, enable: bool) -> Self {
        self.scale_attention = enable;
        self
    }

    /// Enable/disable feedforward scaling
    pub fn with_feedforward_scaling(mut self, enable: bool) -> Self {
        self.scale_feedforward = enable;
        self
    }

    /// Set initialization strategy
    pub fn with_init_strategy(mut self, strategy: IA3InitStrategy) -> Self {
        self.init_strategy = strategy;
        self
    }

    /// Set initial constant value
    pub fn with_init_value(mut self, value: f64) -> Self {
        self.init_value = value;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable per-head scaling for attention
    pub fn with_per_head_scaling(mut self) -> Self {
        self.per_head_scaling = true;
        self
    }

    /// Set learning rate multiplier
    pub fn with_learning_rate_multiplier(mut self, multiplier: f64) -> Self {
        self.learning_rate_multiplier = multiplier;
        self
    }
}

impl Default for IA3Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of scaling that can be applied
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IA3ScalingType {
    /// Scale intermediate activations (e.g., after linear layers)
    Activation,
    /// Scale attention weights or values
    Attention,
    /// Scale feedforward network activations
    FeedForward,
    /// Custom scaling type
    Custom(String),
}

/// IA³ adapter that applies learnable scaling to activations
#[derive(Clone)]
pub struct IA3Adapter<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Scaling vector for activations
    scaling_vector: Tensor<T>,
    /// Type of scaling this adapter applies
    scaling_type: IA3ScalingType,
    /// Configuration
    config: IA3Config,
    /// Training mode
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> IA3Adapter<T>
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
    /// Create a new IA³ adapter
    pub fn new(dimension: usize, scaling_type: IA3ScalingType, config: IA3Config) -> Result<Self> {
        let scaling_vector = Self::initialize_scaling_vector(dimension, &scaling_type, &config)?;

        Ok(Self {
            scaling_vector,
            scaling_type,
            config,
            training: false,
            _phantom: PhantomData,
        })
    }

    /// Create IA³ adapter for attention layers
    pub fn for_attention(dimension: usize, config: IA3Config) -> Result<Self> {
        Self::new(dimension, IA3ScalingType::Attention, config)
    }

    /// Create IA³ adapter for feedforward layers
    pub fn for_feedforward(dimension: usize, config: IA3Config) -> Result<Self> {
        Self::new(dimension, IA3ScalingType::FeedForward, config)
    }

    /// Create IA³ adapter for general activations
    pub fn for_activation(dimension: usize, config: IA3Config) -> Result<Self> {
        Self::new(dimension, IA3ScalingType::Activation, config)
    }

    /// Initialize scaling vector based on configuration
    fn initialize_scaling_vector(
        dimension: usize,
        scaling_type: &IA3ScalingType,
        config: &IA3Config,
    ) -> Result<Tensor<T>> {
        let data = match config.init_strategy {
            IA3InitStrategy::Ones => {
                vec![T::one(); dimension]
            }
            IA3InitStrategy::Constant => {
                let value = T::from(config.init_value).unwrap_or_else(|| T::one());
                vec![value; dimension]
            }
            IA3InitStrategy::RandomNear1 => {
                let mut rng = Random::default();
                let mean = 1.0;
                let std_dev = 0.02;

                let mut data = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    // Box-Muller transform for normal distribution
                    let u1 = rng.random_f64();
                    let u2 = rng.random_f64();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let random_val = mean + std_dev * z;
                    let tensor_val = T::from_f64(random_val).unwrap_or_else(|| T::one());
                    data.push(tensor_val);
                }
                data
            }
            IA3InitStrategy::Xavier => {
                let std_dev = (2.0 / dimension as f64).sqrt();
                let mut rng = Random::default();
                let mean = 1.0;

                let mut data = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    // Box-Muller transform for normal distribution
                    let u1 = rng.random_f64();
                    let u2 = rng.random_f64();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let random_val = mean + std_dev * z;
                    let tensor_val = T::from_f64(random_val).unwrap_or_else(|| T::one());
                    data.push(tensor_val);
                }
                data
            }
            IA3InitStrategy::TaskSpecific => {
                let init_value = match scaling_type {
                    IA3ScalingType::Attention => 0.1,   // Small values for attention
                    IA3ScalingType::FeedForward => 1.0, // Identity for feedforward
                    IA3ScalingType::Activation => 1.0,  // Identity for general activations
                    IA3ScalingType::Custom(_) => config.init_value,
                };
                let value = T::from(init_value).unwrap_or_else(|| T::one());
                vec![value; dimension]
            }
        };

        Tensor::from_vec(data, &[dimension])
    }

    /// Apply scaling to input tensor
    fn apply_scaling(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // IA³ applies element-wise multiplication: output = input ⊙ scaling_vector
        // The scaling vector is broadcasted to match the input shape

        let input_shape = input.shape().dims();
        let scaling_shape = self.scaling_vector.shape().dims();

        // Check if scaling vector can be broadcasted to input shape
        if scaling_shape.len() != 1 {
            return Err(TensorError::invalid_argument(
                "IA³ scaling vector must be 1-dimensional".to_string(),
            ));
        }

        let scaling_dim = scaling_shape[0];
        let last_input_dim = input_shape[input_shape.len() - 1];

        if scaling_dim != last_input_dim {
            return Err(TensorError::invalid_argument(
                format!("IA³ scaling dimension {scaling_dim} doesn't match input last dimension {last_input_dim}")
            ));
        }

        // Perform element-wise multiplication with broadcasting
        self.element_wise_multiply(input, &self.scaling_vector)
    }

    /// Element-wise multiplication with broadcasting
    fn element_wise_multiply(&self, input: &Tensor<T>, scaling: &Tensor<T>) -> Result<Tensor<T>> {
        let input_data = input.to_vec()?;
        let scaling_data = scaling.to_vec()?;
        let input_shape = input.shape().dims();

        let total_elements = input_data.len();
        let scaling_dim = scaling_data.len();
        let last_dim = input_shape[input_shape.len() - 1];

        if scaling_dim != last_dim {
            return Err(TensorError::invalid_argument(
                "Scaling vector dimension mismatch".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(total_elements);

        for (i, &input_val) in input_data.iter().enumerate() {
            let scaling_idx = i % scaling_dim;
            let scaled_val = input_val * scaling_data[scaling_idx];
            result.push(scaled_val);
        }

        Tensor::from_vec(result, input_shape)
    }

    /// Get the scaling vector
    pub fn scaling_vector(&self) -> &Tensor<T> {
        &self.scaling_vector
    }

    /// Get the scaling type
    pub fn scaling_type(&self) -> &IA3ScalingType {
        &self.scaling_type
    }

    /// Get configuration
    pub fn config(&self) -> &IA3Config {
        &self.config
    }

    /// Get IA³ statistics
    pub fn stats(&self) -> IA3Stats {
        let scaling_data = self.scaling_vector.to_vec().unwrap_or_default();

        let mean = if !scaling_data.is_empty() {
            let sum: T = scaling_data
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            sum.to_f64().unwrap_or(0.0) / scaling_data.len() as f64
        } else {
            0.0
        };

        let variance = if scaling_data.len() > 1 {
            let mean_t = T::from(mean).unwrap_or_else(|| T::zero());
            let var_sum: T = scaling_data
                .iter()
                .map(|&x| {
                    let diff = x - mean_t;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);
            var_sum.to_f64().unwrap_or(0.0) / scaling_data.len() as f64
        } else {
            0.0
        };

        let min_val = scaling_data
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .fold(f64::INFINITY, f64::min);
        let max_val = scaling_data
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);

        IA3Stats {
            dimension: scaling_data.len(),
            scaling_type: self.scaling_type.clone(),
            mean_scaling: mean,
            scaling_variance: variance,
            min_scaling: if min_val.is_infinite() { 0.0 } else { min_val },
            max_scaling: if max_val.is_infinite() { 0.0 } else { max_val },
        }
    }
}

impl<T> PEFTAdapter<T> for IA3Adapter<T>
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
    fn forward(&self, input: &Tensor<T>, _base_output: &Tensor<T>) -> Result<Tensor<T>> {
        // IA³ applies scaling to the input directly, not to the base output
        // This is because IA³ modulates activations rather than adding residual connections
        self.apply_scaling(input)
    }

    fn trainable_parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.scaling_vector]
    }

    fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.scaling_vector]
    }

    fn num_trainable_parameters(&self) -> usize {
        self.scaling_vector.shape().dims().iter().product::<usize>()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn method_type(&self) -> PEFTMethod {
        PEFTMethod::IA3
    }
}

/// IA³ adaptation statistics
#[derive(Debug, Clone)]
pub struct IA3Stats {
    pub dimension: usize,
    pub scaling_type: IA3ScalingType,
    pub mean_scaling: f64,
    pub scaling_variance: f64,
    pub min_scaling: f64,
    pub max_scaling: f64,
}

impl IA3Stats {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "IA³ {:?}: {} params, mean={:.3}, var={:.3}, range=[{:.3}, {:.3}]",
            self.scaling_type,
            self.dimension,
            self.mean_scaling,
            self.scaling_variance,
            self.min_scaling,
            self.max_scaling
        )
    }
}

/// Multi-layer IA³ adapter that can handle different scaling types
#[derive(Clone)]
pub struct MultiIA3Adapter<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Individual IA³ adapters for different components
    adapters: Vec<IA3Adapter<T>>,
    /// Names/identifiers for each adapter
    adapter_names: Vec<String>,
    /// Overall configuration
    config: IA3Config,
}

impl<T> MultiIA3Adapter<T>
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
    /// Create a new multi-layer IA³ adapter
    pub fn new(config: IA3Config) -> Self {
        Self {
            adapters: Vec::new(),
            adapter_names: Vec::new(),
            config,
        }
    }

    /// Add an attention scaling adapter
    pub fn add_attention_adapter(&mut self, dimension: usize, name: String) -> Result<()> {
        if self.config.scale_attention {
            let adapter = IA3Adapter::for_attention(dimension, self.config.clone())?;
            self.adapters.push(adapter);
            self.adapter_names.push(name);
        }
        Ok(())
    }

    /// Add a feedforward scaling adapter
    pub fn add_feedforward_adapter(&mut self, dimension: usize, name: String) -> Result<()> {
        if self.config.scale_feedforward {
            let adapter = IA3Adapter::for_feedforward(dimension, self.config.clone())?;
            self.adapters.push(adapter);
            self.adapter_names.push(name);
        }
        Ok(())
    }

    /// Add a general activation scaling adapter
    pub fn add_activation_adapter(&mut self, dimension: usize, name: String) -> Result<()> {
        if self.config.scale_activations {
            let adapter = IA3Adapter::for_activation(dimension, self.config.clone())?;
            self.adapters.push(adapter);
            self.adapter_names.push(name);
        }
        Ok(())
    }

    /// Get adapter by name
    pub fn get_adapter(&self, name: &str) -> Option<&IA3Adapter<T>> {
        self.adapter_names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.adapters.get(idx))
    }

    /// Get mutable adapter by name
    pub fn get_adapter_mut(&mut self, name: &str) -> Option<&mut IA3Adapter<T>> {
        if let Some(idx) = self.adapter_names.iter().position(|n| n == name) {
            self.adapters.get_mut(idx)
        } else {
            None
        }
    }

    /// Get all adapters
    pub fn adapters(&self) -> &[IA3Adapter<T>] {
        &self.adapters
    }

    /// Get all adapter names
    pub fn adapter_names(&self) -> &[String] {
        &self.adapter_names
    }

    /// Get total number of trainable parameters across all adapters
    pub fn total_trainable_parameters(&self) -> usize {
        self.adapters
            .iter()
            .map(|adapter| adapter.num_trainable_parameters())
            .sum()
    }

    /// Get comprehensive statistics
    pub fn comprehensive_stats(&self) -> MultiIA3Stats {
        let individual_stats: Vec<_> = self
            .adapters
            .iter()
            .zip(self.adapter_names.iter())
            .map(|(adapter, name)| (name.clone(), adapter.stats()))
            .collect();

        let total_params = self.total_trainable_parameters();
        let num_adapters = self.adapters.len();

        MultiIA3Stats {
            total_parameters: total_params,
            num_adapters,
            individual_stats,
        }
    }
}

/// Statistics for multi-layer IA³ adapter
#[derive(Debug, Clone)]
pub struct MultiIA3Stats {
    pub total_parameters: usize,
    pub num_adapters: usize,
    pub individual_stats: Vec<(String, IA3Stats)>,
}

impl MultiIA3Stats {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Multi-IA³: {} adapters, {} total params\n",
            self.num_adapters, self.total_parameters
        );

        for (name, stats) in &self.individual_stats {
            summary.push_str(&format!("  {}: {}\n", name, stats.summary()));
        }

        summary
    }
}

/// Predefined IA³ configurations
impl IA3Config {
    /// Configuration for language model attention adaptation
    pub fn for_language_attention() -> Self {
        Self::new()
            .with_attention_scaling(true)
            .with_feedforward_scaling(false)
            .with_activation_scaling(false)
            .with_init_strategy(IA3InitStrategy::TaskSpecific)
            .with_learning_rate_multiplier(16.0)
    }

    /// Configuration for vision transformer adaptation
    pub fn for_vision() -> Self {
        Self::new()
            .with_attention_scaling(true)
            .with_feedforward_scaling(true)
            .with_activation_scaling(false)
            .with_init_strategy(IA3InitStrategy::Ones)
            .with_learning_rate_multiplier(8.0)
    }

    /// Configuration for maximum efficiency (minimal parameters)
    pub fn for_efficiency() -> Self {
        Self::new()
            .with_attention_scaling(true)
            .with_feedforward_scaling(false)
            .with_activation_scaling(false)
            .with_init_strategy(IA3InitStrategy::Ones)
            .with_learning_rate_multiplier(32.0)
    }

    /// Configuration for comprehensive adaptation
    pub fn for_comprehensive() -> Self {
        Self::new()
            .with_attention_scaling(true)
            .with_feedforward_scaling(true)
            .with_activation_scaling(true)
            .with_init_strategy(IA3InitStrategy::RandomNear1)
            .with_dropout(0.1)
            .with_per_head_scaling()
            .with_learning_rate_multiplier(12.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ia3_config_creation() {
        let config = IA3Config::new();
        assert!(config.scale_activations);
        assert!(config.scale_attention);
        assert!(config.scale_feedforward);
        assert_eq!(config.init_strategy, IA3InitStrategy::Ones);
        assert_eq!(config.learning_rate_multiplier, 10.0);
    }

    #[test]
    fn test_ia3_config_presets() {
        let attention_config = IA3Config::for_language_attention();
        assert!(attention_config.scale_attention);
        assert!(!attention_config.scale_feedforward);
        assert_eq!(
            attention_config.init_strategy,
            IA3InitStrategy::TaskSpecific
        );

        let efficiency_config = IA3Config::for_efficiency();
        assert!(efficiency_config.scale_attention);
        assert!(!efficiency_config.scale_feedforward);
        assert!(!efficiency_config.scale_activations);
    }

    #[test]
    fn test_ia3_adapter_creation() {
        let config = IA3Config::new();
        let adapter: IA3Adapter<f32> =
            IA3Adapter::new(100, IA3ScalingType::Attention, config).unwrap();

        // Check scaling vector shape
        assert_eq!(adapter.scaling_vector().shape().dims(), &[100]);

        // Check parameter count (should be very small - just the scaling vector)
        assert_eq!(adapter.num_trainable_parameters(), 100);

        // Check scaling type
        assert_eq!(adapter.scaling_type(), &IA3ScalingType::Attention);
    }

    #[test]
    fn test_initialization_strategies() {
        let config_ones = IA3Config::new().with_init_strategy(IA3InitStrategy::Ones);
        let adapter_ones: IA3Adapter<f32> =
            IA3Adapter::new(5, IA3ScalingType::Activation, config_ones).unwrap();

        let scaling_data = adapter_ones.scaling_vector().to_vec().unwrap();
        assert!(scaling_data.iter().all(|&x| (x - 1.0).abs() < 1e-6));

        let config_const = IA3Config::new()
            .with_init_strategy(IA3InitStrategy::Constant)
            .with_init_value(0.5);
        let adapter_const: IA3Adapter<f32> =
            IA3Adapter::new(5, IA3ScalingType::Activation, config_const).unwrap();

        let scaling_data_const = adapter_const.scaling_vector().to_vec().unwrap();
        assert!(scaling_data_const.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_ia3_forward_pass() {
        let config = IA3Config::new()
            .with_init_strategy(IA3InitStrategy::Constant)
            .with_init_value(2.0);
        let adapter: IA3Adapter<f32> =
            IA3Adapter::new(5, IA3ScalingType::Activation, config).unwrap();

        let input = Tensor::ones(&[2, 5]); // Batch of 2, dimension 5
        let base_output = Tensor::zeros(&[2, 5]); // Not used in IA³

        let result = adapter.forward(&input, &base_output);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);

        // With scaling factor 2.0, output should be 2.0 * input = 2.0 * 1.0 = 2.0
        let output_data = output.to_vec().unwrap();
        assert!(output_data.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_parameter_efficiency() {
        let config = IA3Config::new();
        let adapter: IA3Adapter<f32> =
            IA3Adapter::new(1000, IA3ScalingType::Attention, config).unwrap();

        // IA³ should have very few parameters (just the scaling vector)
        assert_eq!(adapter.num_trainable_parameters(), 1000);

        // Compare to equivalent LoRA: 1000*8 + 8*1000 = 16000 parameters (assuming rank 8)
        // IA³ uses 16x fewer parameters!
        let lora_equivalent_params = 16000;
        let ia3_params = adapter.num_trainable_parameters();
        let efficiency_ratio = ia3_params as f64 / lora_equivalent_params as f64;

        assert!(
            efficiency_ratio < 0.1,
            "IA³ should be much more parameter-efficient than LoRA"
        );
    }

    #[test]
    fn test_ia3_statistics() {
        let config = IA3Config::new().with_init_strategy(IA3InitStrategy::RandomNear1);
        let adapter: IA3Adapter<f32> =
            IA3Adapter::new(100, IA3ScalingType::Attention, config).unwrap();

        let stats = adapter.stats();
        assert_eq!(stats.dimension, 100);
        assert_eq!(stats.scaling_type, IA3ScalingType::Attention);

        // For RandomNear1, mean should be close to 1.0
        assert!((stats.mean_scaling - 1.0).abs() < 0.1);
        assert!(stats.scaling_variance > 0.0); // Should have some variance

        let summary = stats.summary();
        assert!(summary.contains("IA³"));
        assert!(summary.contains("100 params"));
    }

    #[test]
    fn test_multi_ia3_adapter() {
        let config = IA3Config::for_comprehensive();
        let mut multi_adapter: MultiIA3Adapter<f32> = MultiIA3Adapter::new(config);

        // Add different types of adapters
        multi_adapter
            .add_attention_adapter(512, "attn_qkv".to_string())
            .unwrap();
        multi_adapter
            .add_feedforward_adapter(2048, "ffn_intermediate".to_string())
            .unwrap();
        multi_adapter
            .add_activation_adapter(512, "output_proj".to_string())
            .unwrap();

        assert_eq!(multi_adapter.adapters().len(), 3);
        assert_eq!(multi_adapter.adapter_names().len(), 3);

        // Check total parameters
        let total_params = multi_adapter.total_trainable_parameters();
        assert_eq!(total_params, 512 + 2048 + 512); // Sum of all dimensions

        // Check adapter retrieval
        assert!(multi_adapter.get_adapter("attn_qkv").is_some());
        assert!(multi_adapter.get_adapter("nonexistent").is_none());

        let stats = multi_adapter.comprehensive_stats();
        assert_eq!(stats.num_adapters, 3);
        assert_eq!(stats.total_parameters, total_params);
        assert_eq!(stats.individual_stats.len(), 3);
    }
}
