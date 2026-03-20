#![allow(unreachable_patterns)] // GPU/ROCM patterns unreachable when features are disabled

use crate::layers::Layer;
/// Pruning types: enums, config, stats, mask, and PrunedLayer.
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use tenflowers_core::{Tensor, TensorError};

/// Pruning strategy for model compression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PruningStrategy {
    /// Magnitude-based pruning (remove weights with smallest absolute values)
    Magnitude,
    /// Structured pruning (remove entire neurons, channels, or layers)
    Structured,
    /// Gradual pruning during training
    Gradual,
    /// Random pruning (for comparison/baseline)
    Random,
    /// Lottery ticket hypothesis based pruning
    LotteryTicket,
}

/// Pruning scope defines what level to apply pruning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PruningScope {
    /// Global pruning across all layers
    Global,
    /// Layer-wise pruning (each layer independently)
    LayerWise,
    /// Channel-wise pruning (for convolutional layers)
    ChannelWise,
    /// Neuron-wise pruning (for dense layers)
    NeuronWise,
}

/// Configuration for model pruning.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PruningConfig {
    /// Pruning strategy to use
    pub strategy: PruningStrategy,
    /// Scope of pruning application
    pub scope: PruningScope,
    /// Target sparsity ratio (0.0 to 1.0)
    pub target_sparsity: f32,
    /// Layers to skip during pruning (by name or type)
    pub skip_layers: Vec<String>,
    /// Whether to use gradual pruning
    pub gradual_pruning: bool,
    /// Number of pruning steps (for gradual pruning)
    pub pruning_steps: usize,
    /// Acceptable accuracy drop threshold
    pub accuracy_threshold: Option<f32>,
    /// Whether to fine-tune after pruning
    pub fine_tune: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            strategy: PruningStrategy::Magnitude,
            scope: PruningScope::Global,
            target_sparsity: 0.5, // 50% sparsity
            skip_layers: vec!["output".to_string(), "softmax".to_string()],
            gradual_pruning: false,
            pruning_steps: 10,
            accuracy_threshold: Some(0.02), // 2% accuracy drop tolerance
            fine_tune: true,
        }
    }
}

/// Statistics about pruning process.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PruningStats {
    /// Original number of parameters
    pub original_params: usize,
    /// Number of parameters after pruning
    pub remaining_params: usize,
    /// Number of parameters pruned
    pub pruned_params: usize,
    /// Achieved sparsity ratio
    pub achieved_sparsity: f32,
    /// Number of layers affected by pruning
    pub layers_pruned: usize,
    /// Estimated inference speedup
    pub inference_speedup: f32,
    /// Memory usage reduction
    pub memory_reduction: f32,
    /// FLOPS reduction ratio
    pub flops_reduction: f32,
    /// Accuracy before pruning
    pub accuracy_before: Option<f32>,
    /// Accuracy after pruning (before fine-tuning)
    pub accuracy_after: Option<f32>,
    /// Accuracy after fine-tuning
    pub accuracy_final: Option<f32>,
}

impl PruningStats {
    /// Create new empty pruning statistics.
    pub fn new() -> Self {
        Self {
            original_params: 0,
            remaining_params: 0,
            pruned_params: 0,
            achieved_sparsity: 0.0,
            layers_pruned: 0,
            inference_speedup: 1.0,
            memory_reduction: 0.0,
            flops_reduction: 0.0,
            accuracy_before: None,
            accuracy_after: None,
            accuracy_final: None,
        }
    }

    /// Calculate parameter reduction ratio.
    pub fn param_reduction_ratio(&self) -> f32 {
        if self.original_params == 0 {
            0.0
        } else {
            self.pruned_params as f32 / self.original_params as f32
        }
    }

    /// Calculate accuracy recovery after fine-tuning.
    pub fn accuracy_recovery(&self) -> Option<f32> {
        match (self.accuracy_after, self.accuracy_final) {
            (Some(after), Some(final_acc)) => Some(final_acc - after),
            _ => None,
        }
    }
}

impl Default for PruningStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Pruning mask for a layer.
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Layer name this mask applies to
    pub layer_name: String,
    /// Boolean mask indicating which parameters to keep (true = keep, false = prune)
    pub mask: Tensor<f32>,
    /// Sparsity ratio of this mask
    pub sparsity: f32,
}

impl PruningMask {
    /// Create a new pruning mask.
    pub fn new(layer_name: String, mask: Tensor<f32>, sparsity: f32) -> Self {
        Self {
            layer_name,
            mask,
            sparsity,
        }
    }

    /// Apply this mask to a tensor (zero out pruned elements).
    pub fn apply(&self, tensor: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        // Apply pruning by element-wise multiplication with the mask
        // Masked elements (0s) will zero out the corresponding tensor elements
        tensor.mul(&self.mask)
    }

    /// Get the number of remaining (non-zero) parameters.
    pub fn remaining_params(&self) -> usize {
        // Count non-zero elements in the mask
        use tenflowers_core::tensor::TensorStorage;
        match &self.mask.storage {
            TensorStorage::Cpu(ref arr) => arr.iter().map(|&x| if x != 0.0 { 1 } else { 0 }).sum(),
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, we'd need to copy to CPU or use a reduction kernel
                // Fallback to estimated calculation
                let total_elements = self.mask.shape().dims().iter().product::<usize>();
                ((1.0 - self.sparsity) * total_elements as f32) as usize
            }
            #[cfg(not(feature = "gpu"))]
            _ => unreachable!("GPU variant should not exist without gpu feature"),
        }
    }
}

/// Pruned layer wrapper.
#[derive(Debug, Clone)]
pub struct PrunedLayer<T> {
    /// Original layer name
    layer_name: String,
    /// Pruning mask applied to this layer
    pruning_mask: PruningMask,
    /// Pruned weight tensors
    pruned_weights: Vec<Tensor<T>>,
    /// Original input/output shapes
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    /// Phantom type for generic parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> PrunedLayer<T>
where
    T: Clone + Default + 'static,
{
    /// Create a new pruned layer.
    pub fn new(
        layer_name: String,
        pruning_mask: PruningMask,
        pruned_weights: Vec<Tensor<T>>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        Self {
            layer_name,
            pruning_mask,
            pruned_weights,
            input_shape,
            output_shape,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the layer name.
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    /// Get the pruning mask.
    pub fn pruning_mask(&self) -> &PruningMask {
        &self.pruning_mask
    }

    /// Get the sparsity of this layer.
    pub fn sparsity(&self) -> f32 {
        self.pruning_mask.sparsity
    }
}

impl<T> Layer<T> for PrunedLayer<T>
where
    T: Clone + Default + 'static,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // Simplified pruned forward pass
        // In practice, this would use sparse matrix operations or skip zero weights
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        self.pruned_weights.iter().collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.pruned_weights.iter_mut().collect()
    }

    fn set_training(&mut self, _training: bool) {
        // Pruned layers can be used in both training and inference
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}
