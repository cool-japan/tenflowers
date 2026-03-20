//! Parameter-Efficient Fine-Tuning (PEFT) methods for TenfloweRS
//!
//! This module implements modern parameter-efficient fine-tuning techniques
//! that allow efficient adaptation of large pre-trained models to new tasks
//! with minimal additional parameters.

pub mod adalora;
pub mod config;
pub mod ia3;
pub mod lora;
pub mod prefix_tuning;
pub mod ptuning_v2;
pub mod qlora;

pub use adalora::{
    AdaLoRAAdapter, AdaLoRAConfig, AdaLoRAStats, ImportanceMetric, RankAdaptationStats,
};
pub use config::{PEFTConfig, PEFTMethod};
pub use ia3::{
    IA3Adapter, IA3Config, IA3InitStrategy, IA3ScalingType, IA3Stats, MultiIA3Adapter,
    MultiIA3Stats,
};
pub use lora::{LoRAAdapter, LoRAConfig, LoRADense, LoRALayer};
pub use prefix_tuning::{
    PrefixTaskType, PrefixTuningAdapter, PrefixTuningConfig, PrefixTuningStats,
};
pub use ptuning_v2::{
    PTuningTaskType, PTuningV2Adapter, PTuningV2Config, PTuningV2Stats, PromptLayerConfig,
    TokenPosition,
};
pub use qlora::{QLoRAAdapter, QLoRAConfig, QLoRAMemoryStats, QuantizationType};

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Trait for parameter-efficient fine-tuning methods
pub trait PEFTAdapter<T>: Send + Sync + Clone
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Apply the PEFT adaptation to the layer output
    fn forward(&self, input: &Tensor<T>, base_output: &Tensor<T>) -> Result<Tensor<T>>;

    /// Get the trainable parameters introduced by this PEFT method
    fn trainable_parameters(&self) -> Vec<&Tensor<T>>;

    /// Get mutable references to trainable parameters
    fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor<T>>;

    /// Get the number of trainable parameters
    fn num_trainable_parameters(&self) -> usize;

    /// Set training mode
    fn set_training(&mut self, training: bool);

    /// Get the PEFT method type
    fn method_type(&self) -> PEFTMethod;
}

/// A layer wrapper that applies PEFT methods to existing layers
#[derive(Clone)]
pub struct PEFTLayer<T, L, A>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
    L: Layer<T> + Clone,
    A: PEFTAdapter<T>,
{
    base_layer: L,
    adapter: A,
    freeze_base: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, L, A> PEFTLayer<T, L, A>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
    L: Layer<T> + Clone,
    A: PEFTAdapter<T>,
{
    /// Create a new PEFT layer wrapping a base layer
    pub fn new(base_layer: L, adapter: A, freeze_base: bool) -> Self {
        Self {
            base_layer,
            adapter,
            freeze_base,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get reference to the base layer
    pub fn base_layer(&self) -> &L {
        &self.base_layer
    }

    /// Get mutable reference to the base layer
    pub fn base_layer_mut(&mut self) -> &mut L {
        &mut self.base_layer
    }

    /// Get reference to the adapter
    pub fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Get mutable reference to the adapter
    pub fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    /// Check if base layer is frozen
    pub fn is_base_frozen(&self) -> bool {
        self.freeze_base
    }

    /// Set whether to freeze the base layer parameters
    pub fn set_freeze_base(&mut self, freeze: bool) {
        self.freeze_base = freeze;
    }

    /// Get only the trainable parameters (adapter parameters if base is frozen)
    pub fn trainable_parameters(&self) -> Vec<&Tensor<T>> {
        if self.freeze_base {
            self.adapter.trainable_parameters()
        } else {
            let mut params = self.base_layer.parameters();
            params.extend(self.adapter.trainable_parameters());
            params
        }
    }

    /// Get mutable references to only the trainable parameters
    pub fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        if self.freeze_base {
            self.adapter.trainable_parameters_mut()
        } else {
            let mut params = self.base_layer.parameters_mut();
            params.extend(self.adapter.trainable_parameters_mut());
            params
        }
    }

    /// Get statistics about parameter efficiency
    pub fn parameter_efficiency_stats(&self) -> PEFTStats {
        // Calculate actual number of scalar parameters in base layer
        let base_params = self
            .base_layer
            .parameters()
            .iter()
            .map(|p| p.shape().dims().iter().product::<usize>())
            .sum::<usize>();
        let adapter_params = self.adapter.num_trainable_parameters();
        let total_params = base_params + adapter_params;

        PEFTStats {
            base_parameters: base_params,
            adapter_parameters: adapter_params,
            total_parameters: total_params,
            trainable_parameters: if self.freeze_base {
                adapter_params
            } else {
                total_params
            },
            efficiency_ratio: if total_params > 0 {
                adapter_params as f64 / total_params as f64
            } else {
                0.0
            },
        }
    }
}

impl<T, L, A> Layer<T> for PEFTLayer<T, L, A>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
    L: Layer<T> + Clone + 'static,
    A: PEFTAdapter<T> + 'static,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Forward through base layer
        let base_output = self.base_layer.forward(input)?;

        // Apply PEFT adaptation
        self.adapter.forward(input, &base_output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = self.base_layer.parameters();
        params.extend(self.adapter.trainable_parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = self.base_layer.parameters_mut();
        params.extend(self.adapter.trainable_parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.base_layer.set_training(training);
        self.adapter.set_training(training);
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        // For now, we'll implement this using the Clone trait requirement
        // This works because both L and A are required to implement Clone
        Box::new(self.clone())
    }

    fn layer_type(&self) -> crate::layers::LayerType {
        self.base_layer.layer_type()
    }
}

/// Statistics about parameter efficiency for PEFT methods
#[derive(Debug, Clone)]
pub struct PEFTStats {
    pub base_parameters: usize,
    pub adapter_parameters: usize,
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub efficiency_ratio: f64,
}

impl PEFTStats {
    /// Get a human-readable summary of the PEFT efficiency
    pub fn summary(&self) -> String {
        format!(
            "PEFT Efficiency: {:.2}% trainable ({}/{} params), {:.1}x parameter reduction",
            self.efficiency_ratio * 100.0,
            self.trainable_parameters,
            self.total_parameters,
            if self.adapter_parameters > 0 {
                self.base_parameters as f64 / self.adapter_parameters as f64
            } else {
                0.0
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;

    #[test]
    fn test_peft_stats_calculation() {
        let stats = PEFTStats {
            base_parameters: 1000,
            adapter_parameters: 100,
            total_parameters: 1100,
            trainable_parameters: 100,
            efficiency_ratio: 0.1,
        };

        let summary = stats.summary();
        assert!(summary.contains("10.00%"));
        assert!(summary.contains("100/1100"));
    }
}
