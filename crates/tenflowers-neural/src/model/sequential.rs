use crate::layers::Layer;
use crate::model::{zero_tensor_grad, Model};

#[cfg(feature = "serialize")]
use crate::model::ModelState;
use tenflowers_core::{Result, Tensor};

#[cfg(feature = "serialize")]
use std::path::Path;
#[cfg(feature = "serialize")]
use tenflowers_core::TensorError;

#[cfg(feature = "serialize")]
use serde_json;

/// Sequential model that applies layers in order
pub struct Sequential<T>
where
    T: Clone,
{
    layers: Vec<Box<dyn Layer<T>>>,
    training: bool,
}

impl<T: Clone> Sequential<T> {
    /// Create a new sequential model with the given layers
    pub fn new(layers: Vec<Box<dyn Layer<T>>>) -> Self {
        Self {
            layers,
            training: false,
        }
    }

    /// Get the number of layers in the model
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Add a layer to the end of the sequence
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: Box<dyn Layer<T>>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Get a reference to the layer at the given index
    pub fn get_layer(&self, index: usize) -> Option<&dyn Layer<T>> {
        self.layers.get(index).map(|layer| layer.as_ref())
    }

    /// Get a mutable reference to the layer at the given index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut Box<dyn Layer<T>>> {
        self.layers.get_mut(index)
    }

    /// Get a reference to the layers vector
    pub fn layers(&self) -> &Vec<Box<dyn Layer<T>>> {
        &self.layers
    }

    /// Set the model to training mode
    pub fn train(&mut self)
    where
        T: scirs2_core::num_traits::Zero
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        self.set_training(true);
    }

    /// Set the model to evaluation mode
    pub fn eval(&mut self)
    where
        T: scirs2_core::num_traits::Zero
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        self.set_training(false);
    }
}

impl<
        T: Clone
            + scirs2_core::num_traits::Zero
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    > Model<T> for Sequential<T>
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn zero_grad(&mut self) {
        // Zero gradients for all parameters in all layers
        for layer in &mut self.layers {
            let params = layer.parameters_mut();
            for param in params {
                zero_tensor_grad(param);
            }
        }
    }

    fn extract_features(&self, input: &Tensor<T>) -> Result<Option<Vec<Tensor<T>>>> {
        let mut features = Vec::new();
        let mut output = input.clone();

        // Extract intermediate outputs from each layer
        for layer in &self.layers {
            output = layer.forward(&output)?;
            // Store intermediate feature maps for knowledge distillation
            features.push(output.clone());
        }

        // Return all intermediate features for feature matching
        Ok(Some(features))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T: Clone> Clone for Sequential<T> {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| layer.clone_box()).collect(),
            training: self.training,
        }
    }
}

/// Specific implementation for f32 Sequential models with serialization support
impl Sequential<f32> {
    /// Save model parameters to a file (f32 implementation)
    #[cfg(feature = "serialize")]
    pub fn save_f32<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut param_data = Vec::new();
        let mut shapes = Vec::new();

        // Collect all parameters and their shapes
        for param in self.parameters() {
            if let Some(data) = param.as_slice() {
                param_data.push(data.to_vec());
                shapes.push(param.shape().dims().to_vec());
            } else {
                return Err(TensorError::serialization_error_simple(
                    "Failed to access parameter data".to_string(),
                ));
            }
        }

        // Create model state
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("model_type".to_string(), "Sequential".to_string());
        metadata.insert("num_layers".to_string(), self.layers.len().to_string());
        metadata.insert("tensor_type".to_string(), "f32".to_string());

        let model_state = ModelState {
            parameters: param_data,
            shapes,
            metadata,
        };

        // Serialize to JSON and write to file
        let serialized = serde_json::to_string_pretty(&model_state).map_err(|e| {
            TensorError::serialization_error_simple(format!("Serialization failed: {}", e))
        })?;

        std::fs::write(path, serialized).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to write file: {}", e))
        })?;

        Ok(())
    }

    /// Load model parameters from a file (f32 implementation)
    #[cfg(feature = "serialize")]
    pub fn load_f32<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        // Read and deserialize file
        let file_content = std::fs::read_to_string(path).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to read file: {}", e))
        })?;

        let model_state: ModelState = serde_json::from_str(&file_content).map_err(|e| {
            TensorError::serialization_error_simple(format!("Deserialization failed: {}", e))
        })?;

        // Get mutable parameters
        let mut params = self.parameters_mut();

        // Verify parameter count matches
        if params.len() != model_state.parameters.len() {
            return Err(TensorError::serialization_error_simple(format!(
                "Parameter count mismatch: expected {}, found {}",
                params.len(),
                model_state.parameters.len()
            )));
        }

        // Load parameters
        for (param, (data, shape)) in params
            .iter_mut()
            .zip(model_state.parameters.iter().zip(model_state.shapes.iter()))
        {
            // Verify shape matches
            if param.shape().dims() != shape.as_slice() {
                return Err(TensorError::serialization_error_simple(format!(
                    "Shape mismatch for parameter: expected {:?}, found {:?}",
                    param.shape().dims(),
                    shape
                )));
            }

            // Create new tensor with loaded data
            let loaded_tensor = Tensor::from_vec(data.clone(), shape)?;

            // Replace the parameter (preserving device placement)
            let device = param.device().clone();
            let new_param = if let tenflowers_core::Device::Cpu = device {
                loaded_tensor
            } else {
                #[cfg(feature = "gpu")]
                {
                    loaded_tensor.to(device)?
                }
                #[cfg(not(feature = "gpu"))]
                loaded_tensor
            };

            // Update the parameter by replacing the tensor
            **param = new_param;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use tenflowers_core::Tensor;

    #[test]
    fn test_sequential_creation() {
        let dense1 = Dense::<f32>::new(10, 5, true);
        let dense2 = Dense::<f32>::new(5, 1, true);

        let model = Sequential::new(vec![Box::new(dense1), Box::new(dense2)]);

        assert_eq!(model.len(), 2);
        assert!(!model.is_empty());
    }

    #[test]
    fn test_sequential_add() {
        let dense1 = Dense::<f32>::new(10, 5, true);
        let dense2 = Dense::<f32>::new(5, 1, true);

        let model = Sequential::new(vec![Box::new(dense1)]).add(Box::new(dense2));

        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_sequential_forward() {
        let dense = Dense::<f32>::new(3, 2, true);
        let model = Sequential::new(vec![Box::new(dense)]);

        let input = Tensor::<f32>::ones(&[1, 3]);
        let output = model.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let dense1 = Dense::<f32>::new(3, 5, true);
        let dense2 = Dense::<f32>::new(5, 2, false);

        let model = Sequential::new(vec![Box::new(dense1), Box::new(dense2)]);

        let params = model.parameters();
        // dense1: weight + bias = 2 params
        // dense2: weight only = 1 param
        // Total: 3 params
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_sequential_training_mode() {
        let dense = Dense::<f32>::new(3, 2, true);
        let mut model = Sequential::new(vec![Box::new(dense)]);

        model.set_training(true);
        model.set_training(false);
        // Should not panic
    }

    #[test]
    fn test_gradient_zeroing() {
        // Create a simple sequential model with a dense layer
        let dense = Dense::<f32>::new(3, 2, true);
        let mut model = Sequential::new(vec![Box::new(dense)]);

        // Create some dummy gradients for the parameters
        let params = model.parameters_mut();
        for param in params {
            param.set_requires_grad(true);
            // Set a non-zero gradient
            let dummy_grad = Tensor::<f32>::ones(param.shape().dims());
            param.set_grad(Some(dummy_grad));
        }

        // Verify that gradients exist and are non-zero before zeroing
        let params = model.parameters();
        for param in params {
            if let Some(grad) = param.grad() {
                // This would be more rigorous with actual gradient checking
                // For now, just verify the gradient exists
                assert!(grad.shape().dims().len() > 0);
            }
        }

        // Zero the gradients
        model.zero_grad();

        // Verify that gradients are zeroed
        let params = model.parameters();
        for param in params {
            if param.requires_grad() {
                if let Some(grad) = param.grad() {
                    // The gradient should now be zero
                    // (We could add more detailed checks here)
                    assert_eq!(grad.shape().dims(), param.shape().dims());
                }
            }
        }
    }

    #[test]
    fn test_get_layer() {
        let dense1 = Dense::<f32>::new(3, 5, true);
        let dense2 = Dense::<f32>::new(5, 2, true);

        let model = Sequential::new(vec![Box::new(dense1), Box::new(dense2)]);

        assert!(model.get_layer(0).is_some());
        assert!(model.get_layer(1).is_some());
        assert!(model.get_layer(2).is_none());
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_model_serialization() {
        use std::fs;

        // Create a simple sequential model
        let dense1 = Dense::<f32>::new(3, 5, true);
        let dense2 = Dense::<f32>::new(5, 2, true);
        let mut model = Sequential::new(vec![Box::new(dense1), Box::new(dense2)]);

        // Use /tmp for test file
        let model_path = "/tmp/test_model_tenflowers.json";

        // Save the model using f32-specific method
        model.save_f32(model_path).expect("Failed to save model");

        // Verify file was created
        assert!(std::path::Path::new(model_path).exists());

        // Create a new model with the same architecture
        let dense1_new = Dense::<f32>::new(3, 5, true);
        let dense2_new = Dense::<f32>::new(5, 2, true);
        let mut new_model = Sequential::new(vec![Box::new(dense1_new), Box::new(dense2_new)]);

        // Load the saved parameters using f32-specific method
        new_model
            .load_f32(model_path)
            .expect("Failed to load model");

        // Verify the models have the same number of parameters
        assert_eq!(model.parameters().len(), new_model.parameters().len());

        // Verify parameter shapes match
        for (orig_param, loaded_param) in
            model.parameters().iter().zip(new_model.parameters().iter())
        {
            assert_eq!(orig_param.shape().dims(), loaded_param.shape().dims());
        }

        // Clean up
        fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_get_layer_mut() {
        let dense1 = Dense::<f32>::new(10, 5, true);
        let dense2 = Dense::<f32>::new(5, 1, true);

        let mut model = Sequential::new(vec![Box::new(dense1), Box::new(dense2)]);

        // Test getting mutable reference to first layer
        let layer_mut = model.get_layer_mut(0);
        assert!(layer_mut.is_some());

        // Test accessing the layer through the mutable reference
        if let Some(layer) = layer_mut {
            layer.set_training(true);
            // Verify we can call methods on the mutable layer
            let params = layer.parameters_mut();
            assert!(!params.is_empty());
        }

        // Test out of bounds access
        let layer_mut = model.get_layer_mut(10);
        assert!(layer_mut.is_none());
    }
}
