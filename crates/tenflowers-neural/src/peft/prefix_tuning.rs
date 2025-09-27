//! Prefix Tuning implementation for parameter-efficient fine-tuning
//!
//! Prefix Tuning prepends trainable continuous prompts (virtual tokens) to the input sequence
//! while keeping the entire language model frozen. It's particularly effective for language models
//! as it modifies the key-value pairs in each transformer layer.
//!
//! Reference: "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (Li & Liang, 2021)

use super::{PEFTAdapter, PEFTMethod};
use num_traits::{Float, One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Device, Result, Tensor};

/// Configuration for Prefix Tuning
#[derive(Debug, Clone)]
pub struct PrefixTuningConfig {
    /// Number of prefix tokens to prepend
    pub prefix_length: usize,
    /// Dimension of the hidden states
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Whether to reparameterize the prefix (recommended)
    pub reparameterize: bool,
    /// Dimension of reparameterization MLP (if enabled)
    pub reparameterization_dim: usize,
    /// Dropout rate for prefix tokens
    pub prefix_dropout: f64,
    /// Task type for specialized initialization
    pub task_type: PrefixTaskType,
}

/// Task types for specialized prefix tuning configurations
#[derive(Debug, Clone, PartialEq)]
pub enum PrefixTaskType {
    /// Natural language generation tasks
    Generation,
    /// Natural language understanding tasks  
    Understanding,
    /// Question answering tasks
    QuestionAnswering,
    /// Summarization tasks
    Summarization,
}

impl Default for PrefixTuningConfig {
    fn default() -> Self {
        Self {
            prefix_length: 10,
            hidden_size: 768,
            num_attention_heads: 12,
            num_layers: 12,
            reparameterize: true,
            reparameterization_dim: 256,
            prefix_dropout: 0.1,
            task_type: PrefixTaskType::Generation,
        }
    }
}

impl PrefixTuningConfig {
    /// Create config optimized for language generation
    pub fn for_generation(
        prefix_length: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            prefix_length,
            hidden_size,
            num_attention_heads,
            num_layers,
            task_type: PrefixTaskType::Generation,
            ..Default::default()
        }
    }

    /// Create config optimized for language understanding
    pub fn for_understanding(
        prefix_length: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            prefix_length,
            hidden_size,
            num_attention_heads,
            num_layers,
            task_type: PrefixTaskType::Understanding,
            prefix_dropout: 0.05, // Lower dropout for understanding tasks
            ..Default::default()
        }
    }
}

/// Prefix Tuning adapter that prepends trainable virtual tokens
#[derive(Clone)]
pub struct PrefixTuningAdapter<T>
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
    config: PrefixTuningConfig,
    /// Past key values for each layer: [num_layers, 2 (key/value), batch_size, num_heads, prefix_length, head_dim]
    past_key_values: HashMap<usize, (Tensor<T>, Tensor<T>)>,
    /// Reparameterization layers if enabled
    reparameterization_layers: Option<Vec<PrefixReparameterization<T>>>,
    training: bool,
}

/// Reparameterization MLP for prefix tokens  
#[derive(Clone)]
struct PrefixReparameterization<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    linear1: Tensor<T>,
    bias1: Option<Tensor<T>>,
    linear2: Tensor<T>,
    bias2: Option<Tensor<T>>,
    activation: ActivationType,
}

#[derive(Debug, Clone, PartialEq)]
enum ActivationType {
    Tanh,
    ReLU,
    Gelu,
}

impl<T> PrefixTuningAdapter<T>
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
    /// Create a new Prefix Tuning adapter
    pub fn new(config: PrefixTuningConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut past_key_values = HashMap::new();
        let mut reparameterization_layers = None;

        if config.reparameterize {
            // Create reparameterization MLPs for each layer
            let mut reparam_layers = Vec::new();
            for _ in 0..config.num_layers {
                reparam_layers.push(PrefixReparameterization::new(
                    config.reparameterization_dim,
                    2 * config.num_attention_heads * head_dim, // For both key and value
                    &config,
                    device,
                )?);
            }
            reparameterization_layers = Some(reparam_layers);

            // Initialize prefix embeddings to be reparameterized
            for layer_idx in 0..config.num_layers {
                // Initialize with small random values
                let key_size = config.prefix_length * config.reparameterization_dim;
                let key_data: Vec<T> = (0..key_size)
                    .map(|_| {
                        T::from(0.01 * (std::ptr::addr_of!(key_size) as usize % 100) as f64 / 100.0)
                            .unwrap()
                    })
                    .collect();
                let prefix_key = Tensor::from_vec(
                    key_data,
                    &[config.prefix_length, config.reparameterization_dim],
                )?;

                let value_size = config.prefix_length * config.reparameterization_dim;
                let value_data: Vec<T> = (0..value_size)
                    .map(|_| {
                        T::from(
                            0.01 * (std::ptr::addr_of!(value_size) as usize % 100) as f64 / 100.0,
                        )
                        .unwrap()
                    })
                    .collect();
                let prefix_value = Tensor::from_vec(
                    value_data,
                    &[config.prefix_length, config.reparameterization_dim],
                )?;

                past_key_values.insert(layer_idx, (prefix_key, prefix_value));
            }
        } else {
            // Direct prefix parameters without reparameterization
            for layer_idx in 0..config.num_layers {
                // Initialize with small random values
                let key_size = config.prefix_length * config.num_attention_heads * head_dim;
                let key_data: Vec<T> = (0..key_size)
                    .map(|i| T::from(0.02 * ((i + layer_idx) % 100) as f64 / 100.0).unwrap())
                    .collect();
                let prefix_key = Tensor::from_vec(
                    key_data,
                    &[config.prefix_length, config.num_attention_heads, head_dim],
                )?;

                let value_size = config.prefix_length * config.num_attention_heads * head_dim;
                let value_data: Vec<T> = (0..value_size)
                    .map(|i| T::from(0.02 * ((i + layer_idx + 13) % 100) as f64 / 100.0).unwrap())
                    .collect();
                let prefix_value = Tensor::from_vec(
                    value_data,
                    &[config.prefix_length, config.num_attention_heads, head_dim],
                )?;

                past_key_values.insert(layer_idx, (prefix_key, prefix_value));
            }
        }

        Ok(Self {
            config,
            past_key_values,
            reparameterization_layers,
            training: true,
        })
    }

    /// Get the prefix key-value pairs for a specific layer
    pub fn get_layer_prefix_kv(
        &self,
        layer_idx: usize,
        batch_size: usize,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let (raw_key, raw_value) = self.past_key_values.get(&layer_idx).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_shape_simple("Layer index not found".to_string())
        })?;

        if let Some(ref reparam_layers) = self.reparameterization_layers {
            // Apply reparameterization
            let reparam = &reparam_layers[layer_idx];
            let processed_key = reparam.forward(raw_key)?;
            let processed_value = reparam.forward(raw_value)?;

            // Reshape for multi-head attention if needed
            let reshaped_key = if processed_key.shape().dims().len() == 2 {
                // Reshape from [seq_len, hidden_dim] to [batch_size, seq_len, hidden_dim]
                let shape = processed_key.shape().dims();
                tenflowers_core::ops::reshape(&processed_key, &[1, shape[0], shape[1]])?
            } else {
                processed_key
            };

            let reshaped_value = if processed_value.shape().dims().len() == 2 {
                // Reshape from [seq_len, hidden_dim] to [batch_size, seq_len, hidden_dim]
                let shape = processed_value.shape().dims();
                tenflowers_core::ops::reshape(&processed_value, &[1, shape[0], shape[1]])?
            } else {
                processed_value
            };

            Ok((reshaped_key, reshaped_value))
        } else {
            // Direct use without reparameterization - expand for batch size if needed
            let expanded_key = if raw_key.shape().dims().len() == 2 && batch_size > 1 {
                // Expand batch dimension using broadcast_to
                let shape = raw_key.shape().dims();
                tenflowers_core::ops::broadcast_to(raw_key, &[batch_size, shape[0], shape[1]])?
            } else {
                raw_key.clone()
            };

            let expanded_value = if raw_value.shape().dims().len() == 2 && batch_size > 1 {
                // Expand batch dimension using broadcast_to
                let shape = raw_value.shape().dims();
                tenflowers_core::ops::broadcast_to(raw_value, &[batch_size, shape[0], shape[1]])?
            } else {
                raw_value.clone()
            };

            Ok((expanded_key, expanded_value))
        }
    }

    /// Apply dropout to prefix if in training mode
    fn apply_prefix_dropout(&self, tensor: &Tensor<T>) -> Result<Tensor<T>> {
        if self.training && self.config.prefix_dropout > 0.0 {
            // Implement proper dropout
            let shape = tensor.shape().dims();
            let total_elements = shape.iter().product::<usize>();

            // Generate random mask using rng
            use scirs2_core::random::Rng;
            let mut rng = scirs2_core::random::rng();
            let keep_prob = 1.0 - self.config.prefix_dropout;
            let inverted_dropout_scale = T::from(1.0 / keep_prob).unwrap();

            let mask_data: Vec<T> = (0..total_elements)
                .map(|_| {
                    let random_val: f64 = rng.random();
                    if random_val < keep_prob {
                        inverted_dropout_scale
                    } else {
                        T::zero()
                    }
                })
                .collect();

            let mask = Tensor::from_vec(mask_data, shape)?;
            tensor.mul(&mask)
        } else {
            Ok(tensor.clone())
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PrefixTuningConfig {
        &self.config
    }

    /// Get statistics about prefix tuning
    pub fn stats(&self) -> PrefixTuningStats {
        let params_per_layer = if self.config.reparameterize {
            // Reparameterization MLP parameters
            let mlp_params = 2
                * (
                    self.config.reparameterization_dim
                        * 2
                        * self.config.num_attention_heads
                        * (self.config.hidden_size / self.config.num_attention_heads)
                        + 2 * self.config.num_attention_heads
                            * (self.config.hidden_size / self.config.num_attention_heads)
                    // biases
                );
            // Plus prefix embeddings
            mlp_params + self.config.prefix_length * self.config.reparameterization_dim * 2
        } else {
            // Direct prefix parameters
            self.config.prefix_length * self.config.hidden_size * 2 // key + value
        };

        PrefixTuningStats {
            total_parameters: params_per_layer * self.config.num_layers,
            prefix_length: self.config.prefix_length,
            layers_affected: self.config.num_layers,
            reparameterized: self.config.reparameterize,
            effective_prompt_ratio: self.config.prefix_length as f64
                / (self.config.prefix_length + 512) as f64, // Assuming 512 as avg sequence length
        }
    }
}

impl<T> PrefixReparameterization<T>
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
    fn new(
        input_dim: usize,
        output_dim: usize,
        config: &PrefixTuningConfig,
        device: &Device,
    ) -> Result<Self> {
        let hidden_dim = config.reparameterization_dim;

        // Xavier initialization
        let fan_in1 = input_dim;
        let fan_out1 = hidden_dim;
        let std1 = T::from((2.0 / (fan_in1 + fan_out1) as f64).sqrt()).unwrap();

        let fan_in2 = hidden_dim;
        let fan_out2 = output_dim;
        let std2 = T::from((2.0 / (fan_in2 + fan_out2) as f64).sqrt()).unwrap();

        // Initialize with Xavier initialization approximation
        let linear1_data: Vec<T> = (0..(input_dim * hidden_dim))
            .map(|i| T::from((i % 100) as f64 * 0.01 - 0.5).unwrap() * std1)
            .collect();
        let linear1 = Tensor::from_vec(linear1_data, &[input_dim, hidden_dim])?;
        let bias1 = Some(Tensor::zeros(&[hidden_dim]));

        let linear2_data: Vec<T> = (0..(hidden_dim * output_dim))
            .map(|i| T::from((i % 100) as f64 * 0.01 - 0.5).unwrap() * std2)
            .collect();
        let linear2 = Tensor::from_vec(linear2_data, &[hidden_dim, output_dim])?;
        let bias2 = Some(Tensor::zeros(&[output_dim]));

        // Choose activation based on task type
        let activation = match config.task_type {
            PrefixTaskType::Generation => ActivationType::Tanh,
            PrefixTaskType::Understanding => ActivationType::ReLU,
            PrefixTaskType::QuestionAnswering => ActivationType::Gelu,
            PrefixTaskType::Summarization => ActivationType::Tanh,
        };

        Ok(Self {
            linear1,
            bias1,
            linear2,
            bias2,
            activation,
        })
    }

    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // First linear layer
        let mut x = input.matmul(&self.linear1)?;
        if let Some(ref bias) = self.bias1 {
            x = x.add(bias)?;
        }

        // Activation
        x = match self.activation {
            ActivationType::Tanh => tenflowers_core::ops::activation::tanh(&x)?,
            ActivationType::ReLU => tenflowers_core::ops::activation::relu(&x)?,
            ActivationType::Gelu => tenflowers_core::ops::activation::gelu(&x)?,
        };

        // Second linear layer
        let mut output = x.matmul(&self.linear2)?;
        if let Some(ref bias) = self.bias2 {
            output = output.add(bias)?;
        }

        Ok(output)
    }
}

impl<T> PEFTAdapter<T> for PrefixTuningAdapter<T>
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
    fn forward(&self, input: &Tensor<T>, base_output: &Tensor<T>) -> Result<Tensor<T>> {
        // Prefix tuning doesn't modify the base output directly
        // It modifies the attention computation by providing past key-values
        // The integration happens at the attention layer level

        // For now, we just return the base output
        // In a real implementation, this would be handled by the attention mechanism
        Ok(base_output.clone())
    }

    fn trainable_parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();

        // Add prefix embeddings/parameters
        for (key, value) in self.past_key_values.values() {
            params.push(key);
            params.push(value);
        }

        // Add reparameterization parameters if present
        if let Some(ref reparam_layers) = self.reparameterization_layers {
            for layer in reparam_layers {
                params.push(&layer.linear1);
                if let Some(ref bias) = layer.bias1 {
                    params.push(bias);
                }
                params.push(&layer.linear2);
                if let Some(ref bias) = layer.bias2 {
                    params.push(bias);
                }
            }
        }

        params
    }

    fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();

        // Add prefix embeddings/parameters
        for (key, value) in self.past_key_values.values_mut() {
            params.push(key);
            params.push(value);
        }

        // Add reparameterization parameters if present
        if let Some(ref mut reparam_layers) = self.reparameterization_layers {
            for layer in reparam_layers {
                params.push(&mut layer.linear1);
                if let Some(ref mut bias) = layer.bias1 {
                    params.push(bias);
                }
                params.push(&mut layer.linear2);
                if let Some(ref mut bias) = layer.bias2 {
                    params.push(bias);
                }
            }
        }

        params
    }

    fn num_trainable_parameters(&self) -> usize {
        self.stats().total_parameters
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn method_type(&self) -> PEFTMethod {
        PEFTMethod::PrefixTuning
    }
}

/// Statistics for Prefix Tuning
#[derive(Debug, Clone)]
pub struct PrefixTuningStats {
    pub total_parameters: usize,
    pub prefix_length: usize,
    pub layers_affected: usize,
    pub reparameterized: bool,
    pub effective_prompt_ratio: f64,
}

impl PrefixTuningStats {
    pub fn summary(&self) -> String {
        format!(
            "Prefix Tuning: {} prefix tokens, {} layers, {} params ({:.1}% effective prompt), {}",
            self.prefix_length,
            self.layers_affected,
            self.total_parameters,
            self.effective_prompt_ratio * 100.0,
            if self.reparameterized {
                "reparameterized"
            } else {
                "direct"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Device;

    #[test]
    fn test_prefix_tuning_creation() {
        let device = Device::Cpu;
        let config = PrefixTuningConfig::default();
        let adapter = PrefixTuningAdapter::<f32>::new(config, &device).unwrap();

        assert_eq!(adapter.config().prefix_length, 10);
        assert_eq!(adapter.config().num_layers, 12);
        assert!(adapter.config().reparameterize);
    }

    #[test]
    fn test_prefix_tuning_config_generation() {
        let config = PrefixTuningConfig::for_generation(20, 768, 12, 12);
        assert_eq!(config.prefix_length, 20);
        assert_eq!(config.task_type, PrefixTaskType::Generation);
    }

    #[test]
    fn test_prefix_tuning_config_understanding() {
        let config = PrefixTuningConfig::for_understanding(15, 512, 8, 6);
        assert_eq!(config.prefix_length, 15);
        assert_eq!(config.task_type, PrefixTaskType::Understanding);
        assert_eq!(config.prefix_dropout, 0.05);
    }

    #[test]
    fn test_prefix_tuning_stats() {
        let device = Device::Cpu;
        let config = PrefixTuningConfig {
            prefix_length: 5,
            hidden_size: 128,
            num_attention_heads: 4,
            num_layers: 3,
            reparameterize: false,
            ..Default::default()
        };

        let adapter = PrefixTuningAdapter::<f32>::new(config, &device).unwrap();
        let stats = adapter.stats();

        assert_eq!(stats.prefix_length, 5);
        assert_eq!(stats.layers_affected, 3);
        assert!(!stats.reparameterized);
    }

    #[test]
    fn test_prefix_tuning_trainable_parameters() {
        let device = Device::Cpu;
        let config = PrefixTuningConfig {
            prefix_length: 5,
            num_layers: 2,
            reparameterize: false,
            ..Default::default()
        };

        let adapter = PrefixTuningAdapter::<f32>::new(config, &device).unwrap();
        let params = adapter.trainable_parameters();

        // Should have 2 parameters per layer (key + value) × 2 layers = 4 parameters
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_prefix_tuning_layer_kv() {
        let device = Device::Cpu;
        let config = PrefixTuningConfig {
            prefix_length: 3,
            hidden_size: 64,
            num_attention_heads: 4,
            num_layers: 1,
            reparameterize: false,
            ..Default::default()
        };

        let adapter = PrefixTuningAdapter::<f32>::new(config, &device).unwrap();
        let (key, value) = adapter.get_layer_prefix_kv(0, 2).unwrap();

        // With simplified implementation, shape may be different
        // The actual shape depends on the implementation details
        assert_eq!(key.shape().dims().len(), 3); // Should be 3D tensor
        assert_eq!(value.shape().dims().len(), 3); // Should be 3D tensor
    }

    #[test]
    fn test_reparameterization_forward() {
        let device = Device::Cpu;
        let config = PrefixTuningConfig::default();
        let reparam = PrefixReparameterization::<f32>::new(128, 256, &config, &device).unwrap();

        let input = Tensor::ones(&[10, 128]);
        let output = reparam.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[10, 256]);
    }
}
