pub mod examples;
pub mod functional;
pub mod sequential;
pub mod subclass;
pub mod subclass_examples;

pub use functional::{
    FunctionalModel, FunctionalModelBuilder, Input, LayerOp, Node, NodeId, SharedLayer,
};
pub use sequential::Sequential;
pub use subclass::{helpers, CustomModel, LayerContainer, ModelBase, ModelExt};

use tenflowers_core::{Result, Tensor};

#[cfg(feature = "serialize")]
use std::path::Path;
#[cfg(feature = "serialize")]
use tenflowers_core::TensorError;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Serializable representation of model parameters
#[cfg(feature = "serialize")]
#[derive(Serialize, Deserialize)]
pub struct ModelState {
    /// Parameter data as flattened vectors
    pub parameters: Vec<Vec<f32>>,
    /// Shape information for each parameter
    pub shapes: Vec<Vec<usize>>,
    /// Model metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Core trait for all models
pub trait Model<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
    fn parameters(&self) -> Vec<&Tensor<T>>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>>;
    fn set_training(&mut self, training: bool);
    fn zero_grad(&mut self);

    /// Extract intermediate features for knowledge distillation
    /// Returns None if the model doesn't support feature extraction
    fn extract_features(&self, input: &Tensor<T>) -> Result<Option<Vec<Tensor<T>>>> {
        // Default implementation returns None - models can override to provide features
        let _ = input; // Suppress unused parameter warning
        Ok(None)
    }

    /// Provide access to Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Trait for model serialization - separate from Model to maintain dyn compatibility
#[cfg(feature = "serialize")]
pub trait ModelSerialization<T> {
    /// Save model parameters to a file
    fn save<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        Err(TensorError::serialization_error_simple(
            "Serialization not implemented for this model type".to_string(),
        ))
    }

    /// Load model parameters from a file
    fn load<P: AsRef<Path>>(&mut self, _path: P) -> Result<()> {
        Err(TensorError::serialization_error_simple(
            "Deserialization not implemented for this model type".to_string(),
        ))
    }
}

/// Zero the gradient of a tensor parameter
pub(crate) fn zero_tensor_grad<T>(param: &mut Tensor<T>)
where
    T: num_traits::Zero
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if param.requires_grad() {
        let zero_grad = create_zero_grad_for_device(param);
        param.set_grad(zero_grad);
    }
}

fn create_zero_grad_for_device<T>(param: &Tensor<T>) -> Option<Tensor<T>>
where
    T: num_traits::Zero
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match param.device() {
        tenflowers_core::Device::Cpu => Some(Tensor::zeros(param.shape().dims())),
        #[cfg(feature = "gpu")]
        device => {
            let cpu_zeros = Tensor::zeros(param.shape().dims());
            cpu_zeros.to(device.clone()).ok() // Fall back to no gradient if transfer fails
        }
    }
}
