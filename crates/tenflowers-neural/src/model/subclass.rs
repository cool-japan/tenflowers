use crate::layers::Layer;
use crate::model::{zero_tensor_grad, Model};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

#[cfg(feature = "serialize")]
use std::path::Path;

/// Base structure for custom models providing common functionality
pub struct ModelBase<T> {
    /// Training mode flag
    pub training: bool,
    /// Model name
    pub name: Option<String>,
    /// Metadata for the model
    pub metadata: HashMap<String, String>,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ModelBase<T> {
    /// Create a new model base
    pub fn new() -> Self {
        Self {
            training: true,
            name: None,
            metadata: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new model base with a name
    pub fn new_named(name: String) -> Self {
        Self {
            training: true,
            name: Some(name),
            metadata: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the model name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Get the model name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Zero gradients for a collection of parameters
    pub fn zero_gradients(&self, params: &mut [&mut Tensor<T>])
    where
        T: scirs2_core::num_traits::Zero
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        for param in params {
            zero_tensor_grad(param);
        }
    }
}

impl<T> Default for ModelBase<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A container for model layers with automatic parameter collection
pub struct LayerContainer<T> {
    layers: Vec<Box<dyn Layer<T>>>,
    names: Vec<Option<String>>,
}

impl<T> LayerContainer<T> {
    /// Create a new layer container
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Add a layer to the container
    pub fn add_layer(&mut self, layer: Box<dyn Layer<T>>) -> usize {
        let index = self.layers.len();
        self.layers.push(layer);
        self.names.push(None);
        index
    }

    /// Add a named layer to the container
    pub fn add_named_layer(&mut self, layer: Box<dyn Layer<T>>, name: String) -> usize {
        let index = self.layers.len();
        self.layers.push(layer);
        self.names.push(Some(name));
        index
    }

    /// Get a layer by index
    pub fn get_layer(&self, index: usize) -> Option<&dyn Layer<T>> {
        self.layers.get(index).map(|layer| layer.as_ref())
    }

    /// Get a mutable layer by index  
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut Box<dyn Layer<T>>> {
        self.layers.get_mut(index)
    }

    /// Get layer by name
    pub fn get_layer_by_name(&self, name: &str) -> Option<&dyn Layer<T>> {
        for (i, layer_name) in self.names.iter().enumerate() {
            if let Some(layer_name) = layer_name {
                if layer_name == name {
                    return self.get_layer(i);
                }
            }
        }
        None
    }

    /// Get mutable layer by name
    pub fn get_layer_by_name_mut(&mut self, name: &str) -> Option<&mut Box<dyn Layer<T>>> {
        for (i, layer_name) in self.names.iter().enumerate() {
            if let Some(layer_name) = layer_name {
                if layer_name == name {
                    return self.get_layer_mut(i);
                }
            }
        }
        None
    }

    /// Get all parameters from all layers
    pub fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Get all mutable parameters from all layers
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    /// Set training mode for all layers
    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    /// Get the number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if container is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get layer names
    pub fn layer_names(&self) -> Vec<Option<&str>> {
        self.names.iter().map(|name| name.as_deref()).collect()
    }
}

impl<T> Default for LayerContainer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait extension for models with additional utilities
pub trait ModelExt<T>: Model<T> {
    /// Forward pass with automatic training mode handling
    fn call(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.forward(input)
    }

    /// Predict (forward pass in evaluation mode)
    fn predict(&mut self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        Self: Sized,
    {
        let was_training = self.is_training_mode();
        self.set_training(false);
        let result = self.forward(input);
        self.set_training(was_training);
        result
    }

    /// Check if model is in training mode
    fn is_training_mode(&self) -> bool;

    /// Get model summary as string
    fn summary(&self) -> String {
        let param_count = self.parameters().len();
        format!(
            "Model Summary:\n  Parameters: {}\n  Training: {}",
            param_count,
            self.is_training_mode()
        )
    }

    /// Count total number of parameters  
    fn parameter_count(&self) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        self.parameters().iter().map(|p| p.size()).sum()
    }

    /// Count trainable parameters
    fn trainable_parameter_count(&self) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        self.parameters()
            .iter()
            .filter(|p| p.requires_grad())
            .map(|p| p.size())
            .sum()
    }
}

/// Macro to implement the Model trait for custom models with common patterns
#[macro_export]
macro_rules! impl_model {
    // Basic implementation with ModelBase and LayerContainer
    ($model_type:ty, $element_type:ty, $base_field:ident, $layers_field:ident) => {
        impl Model<$element_type> for $model_type {
            fn forward(&self, input: &Tensor<$element_type>) -> Result<Tensor<$element_type>> {
                self.call_impl(input)
            }

            fn parameters(&self) -> Vec<&Tensor<$element_type>> {
                self.$layers_field.parameters()
            }

            fn parameters_mut(&mut self) -> Vec<&mut Tensor<$element_type>> {
                self.$layers_field.parameters_mut()
            }

            fn set_training(&mut self, training: bool) {
                self.$base_field.set_training(training);
                self.$layers_field.set_training(training);
            }

            fn zero_grad(&mut self) {
                let mut params = self.parameters_mut();
                self.$base_field.zero_gradients(&mut params);
            }
        }

        impl ModelExt<$element_type> for $model_type {
            fn is_training_mode(&self) -> bool {
                self.$base_field.is_training()
            }
        }
    };

    // Implementation with custom parameter handling
    ($model_type:ty, $element_type:ty, $base_field:ident, $param_method:ident, $param_mut_method:ident) => {
        impl Model<$element_type> for $model_type {
            fn forward(&self, input: &Tensor<$element_type>) -> Result<Tensor<$element_type>> {
                self.call_impl(input)
            }

            fn parameters(&self) -> Vec<&Tensor<$element_type>> {
                self.$param_method()
            }

            fn parameters_mut(&mut self) -> Vec<&mut Tensor<$element_type>> {
                self.$param_mut_method()
            }

            fn set_training(&mut self, training: bool) {
                self.$base_field.set_training(training);
                self.set_training_impl(training);
            }

            fn zero_grad(&mut self) {
                let mut params = self.parameters_mut();
                self.$base_field.zero_gradients(&mut params);
            }
        }

        impl ModelExt<$element_type> for $model_type {
            fn is_training_mode(&self) -> bool {
                self.$base_field.is_training()
            }
        }
    };
}

/// A simple custom model template that users can extend
pub struct CustomModel<T> {
    pub base: ModelBase<T>,
    pub layers: LayerContainer<T>,
}

impl<T> CustomModel<T> {
    /// Create a new custom model
    pub fn new() -> Self {
        Self {
            base: ModelBase::new(),
            layers: LayerContainer::new(),
        }
    }

    /// Create a new custom model with name
    pub fn new_named(name: String) -> Self {
        Self {
            base: ModelBase::new_named(name),
            layers: LayerContainer::new(),
        }
    }

    /// Add a layer to the model
    pub fn add_layer(&mut self, layer: Box<dyn Layer<T>>) -> usize {
        self.layers.add_layer(layer)
    }

    /// Add a named layer to the model
    pub fn add_named_layer(&mut self, layer: Box<dyn Layer<T>>, name: String) -> usize {
        self.layers.add_named_layer(layer, name)
    }

    /// Default implementation for forward pass (sequential through all layers)
    pub fn call_impl(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let mut output = input.clone();
        for i in 0..self.layers.len() {
            if let Some(layer) = self.layers.get_layer(i) {
                output = layer.forward(&output)?;
            }
        }
        Ok(output)
    }
}

impl<T> Default for CustomModel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement the Model trait for CustomModel manually since we need generic implementation
impl<T> Model<T> for CustomModel<T>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.call_impl(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        self.layers.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.layers.parameters_mut()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        self.layers.set_training(training);
    }

    fn zero_grad(&mut self) {
        for param in self.parameters_mut() {
            zero_tensor_grad(param);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> ModelExt<T> for CustomModel<T>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn is_training_mode(&self) -> bool {
        self.base.is_training()
    }
}

/// Helper functions for common model operations
pub mod helpers {
    use super::*;
    // use tenflowers_core::ops::random::random_normal_f32; // Unused for now

    /// Calculate fan_in and fan_out for a given tensor shape
    /// This determines the number of input and output connections for initialization
    fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            1 => {
                // 1D tensor (bias): fan_in = fan_out = size
                (shape[0], shape[0])
            }
            2 => {
                // 2D tensor (dense layer): [input_size, output_size]
                (shape[0], shape[1])
            }
            3 => {
                // 3D tensor (1D conv): [out_channels, in_channels, kernel_size]
                let receptive_field_size = shape[2];
                (
                    shape[1] * receptive_field_size,
                    shape[0] * receptive_field_size,
                )
            }
            4 => {
                // 4D tensor (2D conv): [out_channels, in_channels, kernel_h, kernel_w]
                let receptive_field_size = shape[2] * shape[3];
                (
                    shape[1] * receptive_field_size,
                    shape[0] * receptive_field_size,
                )
            }
            5 => {
                // 5D tensor (3D conv): [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
                let receptive_field_size = shape[2] * shape[3] * shape[4];
                (
                    shape[1] * receptive_field_size,
                    shape[0] * receptive_field_size,
                )
            }
            _ => {
                // Default fallback: treat as 2D
                let total_size = shape.iter().product::<usize>();
                let sqrt_size = (total_size as f64).sqrt() as usize;
                (sqrt_size, sqrt_size)
            }
        }
    }

    /// Create a tensor with random normal distribution
    fn create_random_normal_tensor<T>(shape: &[usize], mean: T, std_dev: T) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::Float
            + Copy
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        // For now, create a simple implementation using available random operations
        // In practice, this would use proper random tensor generation

        let total_elements = shape.iter().product::<usize>();

        // For f32, we can use the available random_normal_f32 function
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Create random f32 data and convert
            let random_data = (0..total_elements)
                .map(|_| {
                    // Simple Box-Muller transform for normal distribution
                    let u1: f32 = scirs2_core::random::quick::random_f32().max(1e-10);
                    let u2: f32 = scirs2_core::random::quick::random_f32();
                    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    let std_dev_f32 = std_dev.to_f32().unwrap_or(0.01);
                    let mean_f32 = mean.to_f32().unwrap_or(0.0);
                    T::from_f32(mean_f32 + std_dev_f32 * z0).unwrap_or(T::zero())
                })
                .collect::<Vec<T>>();

            Tensor::from_vec(random_data, shape)
        } else {
            // Fallback: create tensor with small constant values
            let small_value = std_dev * T::from_f32(0.1).unwrap_or(T::one());
            Ok(Tensor::from_scalar(small_value))
        }
    }

    /// Initialize model parameters with Xavier/Glorot initialization
    /// Initializes weights with variance = 2/(fan_in + fan_out)
    pub fn xavier_init<T>(model: &mut dyn Model<T>)
    where
        T: scirs2_core::num_traits::Float
            + Copy
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        for param in model.parameters_mut() {
            let shape = param.shape().dims();

            // Calculate fan_in and fan_out based on tensor shape
            let (fan_in, fan_out) = calculate_fan_in_fan_out(shape);

            // Xavier initialization: variance = 2 / (fan_in + fan_out)
            let variance = T::from_f32(2.0).unwrap_or(T::from_usize(2).unwrap())
                / T::from_usize(fan_in + fan_out).unwrap_or(T::from_usize(1).unwrap());
            let std_dev = variance.sqrt();

            // Initialize with random normal distribution
            if let Ok(random_tensor) = create_random_normal_tensor::<T>(shape, T::zero(), std_dev) {
                *param = random_tensor;
            }
        }
    }

    /// Initialize model parameters with He initialization
    /// Initializes weights with variance = 2/fan_in (designed for ReLU activations)
    pub fn he_init<T>(model: &mut dyn Model<T>)
    where
        T: scirs2_core::num_traits::Float
            + Copy
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        for param in model.parameters_mut() {
            let shape = param.shape().dims();

            // Calculate fan_in and fan_out based on tensor shape
            let (fan_in, _fan_out) = calculate_fan_in_fan_out(shape);

            // He initialization: variance = 2 / fan_in
            let variance = T::from_f32(2.0).unwrap_or(T::from_usize(2).unwrap())
                / T::from_usize(fan_in).unwrap_or(T::from_usize(1).unwrap());
            let std_dev = variance.sqrt();

            // Initialize with random normal distribution
            if let Ok(random_tensor) = create_random_normal_tensor::<T>(shape, T::zero(), std_dev) {
                *param = random_tensor;
            }
        }
    }

    /// Set all model parameters to require gradients
    pub fn enable_gradients<T>(model: &mut dyn Model<T>) {
        for param in model.parameters_mut() {
            param.set_requires_grad(true);
        }
    }

    /// Freeze all model parameters (disable gradients)
    pub fn freeze_model<T>(model: &mut dyn Model<T>) {
        for param in model.parameters_mut() {
            param.set_requires_grad(false);
        }
    }

    /// Get model info as a structured format
    pub fn model_info<T>(model: &dyn Model<T>) -> HashMap<String, String>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        let mut info = HashMap::new();
        let params = model.parameters();

        info.insert("total_parameters".to_string(), params.len().to_string());
        info.insert(
            "trainable_parameters".to_string(),
            params
                .iter()
                .filter(|p| p.requires_grad())
                .count()
                .to_string(),
        );
        info.insert(
            "parameter_size".to_string(),
            params.iter().map(|p| p.size()).sum::<usize>().to_string(),
        );

        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use tenflowers_core::Tensor;

    // Example custom model for testing
    struct MyCustomModel {
        base: ModelBase<f32>,
        layers: LayerContainer<f32>,
    }

    impl MyCustomModel {
        fn new() -> Self {
            let mut model = Self {
                base: ModelBase::new_named("MyCustomModel".to_string()),
                layers: LayerContainer::new(),
            };

            // Add some layers
            model.layers.add_named_layer(
                Box::new(Dense::<f32>::new(64, 32, true)),
                "dense1".to_string(),
            );
            model.layers.add_named_layer(
                Box::new(Dense::<f32>::new(32, 10, true)),
                "dense2".to_string(),
            );

            model
        }

        fn call_impl(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            let x = self
                .layers
                .get_layer_by_name("dense1")
                .unwrap()
                .forward(input)?;
            let output = self
                .layers
                .get_layer_by_name("dense2")
                .unwrap()
                .forward(&x)?;
            Ok(output)
        }
    }

    // Implement Model trait manually
    impl Model<f32> for MyCustomModel {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            self.call_impl(input)
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            self.layers.parameters()
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            self.layers.parameters_mut()
        }

        fn set_training(&mut self, training: bool) {
            self.base.set_training(training);
            self.layers.set_training(training);
        }

        fn zero_grad(&mut self) {
            for param in self.parameters_mut() {
                zero_tensor_grad(param);
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl ModelExt<f32> for MyCustomModel {
        fn is_training_mode(&self) -> bool {
            self.base.is_training()
        }
    }

    #[test]
    fn test_custom_model() {
        let model = MyCustomModel::new();
        assert_eq!(model.base.name(), Some("MyCustomModel"));
        assert_eq!(model.layers.len(), 2);
        assert!(model.layers.get_layer_by_name("dense1").is_some());
        assert!(model.layers.get_layer_by_name("dense2").is_some());
    }

    #[test]
    fn test_model_base() {
        let mut base = ModelBase::<f32>::new_named("TestModel".to_string());
        assert_eq!(base.name(), Some("TestModel"));
        assert!(base.is_training());

        base.set_training(false);
        assert!(!base.is_training());

        base.add_metadata("version".to_string(), "1.0".to_string());
        assert_eq!(base.get_metadata("version"), Some("1.0"));
    }

    #[test]
    fn test_layer_container() {
        let mut container = LayerContainer::<f32>::new();
        assert!(container.is_empty());

        let dense1 = Dense::<f32>::new(64, 32, true);
        let dense2 = Dense::<f32>::new(32, 10, true);

        container.add_named_layer(Box::new(dense1), "first".to_string());
        container.add_named_layer(Box::new(dense2), "second".to_string());

        assert_eq!(container.len(), 2);
        assert!(container.get_layer_by_name("first").is_some());
        assert!(container.get_layer_by_name("second").is_some());
        assert!(container.get_layer_by_name("third").is_none());
    }

    #[test]
    fn test_custom_model_template() {
        let mut model = CustomModel::<f32>::new_named("Template".to_string());

        let dense = Dense::<f32>::new(64, 32, true);
        model.add_named_layer(Box::new(dense), "dense".to_string());

        assert_eq!(model.base.name(), Some("Template"));
        assert_eq!(model.layers.len(), 1);
    }

    #[test]
    fn test_model_ext_trait() {
        let model = MyCustomModel::new();
        assert!(model.is_training_mode());

        let summary = model.summary();
        assert!(summary.contains("Model Summary"));
        assert!(summary.contains("Parameters"));
        assert!(summary.contains("Training"));
    }
}
