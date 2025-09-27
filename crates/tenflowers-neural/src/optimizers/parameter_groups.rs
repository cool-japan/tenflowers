use crate::model::Model;
use crate::optimizers::Optimizer;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for a parameter group
#[derive(Debug)]
pub struct ParameterGroupConfig {
    /// Learning rate for this group
    pub learning_rate: f32,
    /// Weight decay for this group
    pub weight_decay: f32,
    /// Custom hyperparameters for specific optimizers
    pub custom_params: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl Clone for ParameterGroupConfig {
    fn clone(&self) -> Self {
        Self {
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            // Note: custom_params cannot be cloned due to trait object limitations
            // For a full implementation, you'd need a custom trait for cloneable parameters
            custom_params: HashMap::new(),
        }
    }
}

impl ParameterGroupConfig {
    /// Create a new parameter group with default settings
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            weight_decay: 0.0,
            custom_params: HashMap::new(),
        }
    }

    /// Set weight decay for this group
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Add a custom parameter for this group
    pub fn with_custom_param<T: 'static + Send + Sync>(mut self, key: String, value: T) -> Self {
        self.custom_params.insert(key, Box::new(value));
        self
    }

    /// Get a custom parameter for this group
    pub fn get_custom_param<T: 'static>(&self, key: &str) -> Option<&T> {
        self.custom_params.get(key)?.downcast_ref()
    }
}

impl Default for ParameterGroupConfig {
    fn default() -> Self {
        Self::new(0.001)
    }
}

/// A parameter group associates a set of parameters with specific optimization settings
pub struct ParameterGroup<T> {
    /// Configuration for this group
    pub config: ParameterGroupConfig,
    /// Parameters belonging to this group (identified by their memory addresses)
    pub parameters: HashSet<*const Tensor<T>>,
    /// Name/identifier for this group
    pub name: String,
}

impl<T> ParameterGroup<T> {
    /// Create a new parameter group
    pub fn new(name: String, config: ParameterGroupConfig) -> Self {
        Self {
            config,
            parameters: HashSet::new(),
            name,
        }
    }

    /// Add a parameter to this group
    pub fn add_parameter(&mut self, param: &Tensor<T>) {
        self.parameters.insert(param as *const Tensor<T>);
    }

    /// Remove a parameter from this group
    pub fn remove_parameter(&mut self, param: &Tensor<T>) {
        self.parameters.remove(&(param as *const Tensor<T>));
    }

    /// Check if a parameter belongs to this group
    pub fn contains_parameter(&self, param: &Tensor<T>) -> bool {
        self.parameters.contains(&(param as *const Tensor<T>))
    }

    /// Get the number of parameters in this group
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
}

/// An optimizer that supports parameter groups with different hyperparameters
pub struct ParameterGroupOptimizer<T, O: Optimizer<T>> {
    /// Base optimizer type (e.g., Adam, SGD, etc.)
    base_optimizer_factory: Box<dyn Fn(&ParameterGroupConfig) -> O + Send + Sync>,
    /// Parameter groups
    groups: Vec<ParameterGroup<T>>,
    /// Separate optimizer instances for each group
    group_optimizers: Vec<O>,
    /// Default configuration for new groups
    default_config: ParameterGroupConfig,
}

impl<T, O: Optimizer<T>> ParameterGroupOptimizer<T, O> {
    /// Create a new parameter group optimizer
    pub fn new<F>(factory: F, default_config: ParameterGroupConfig) -> Self
    where
        F: Fn(&ParameterGroupConfig) -> O + Send + Sync + 'static,
    {
        Self {
            base_optimizer_factory: Box::new(factory),
            groups: Vec::new(),
            group_optimizers: Vec::new(),
            default_config,
        }
    }

    /// Add a new parameter group
    pub fn add_group(&mut self, name: String, config: ParameterGroupConfig) -> Result<usize> {
        let group = ParameterGroup::new(name, config.clone());
        let optimizer = (self.base_optimizer_factory)(&config);

        self.groups.push(group);
        self.group_optimizers.push(optimizer);

        Ok(self.groups.len() - 1)
    }

    /// Add parameters to a specific group
    pub fn add_parameters_to_group(
        &mut self,
        group_index: usize,
        parameters: &[&Tensor<T>],
    ) -> Result<()> {
        if group_index >= self.groups.len() {
            return Err(TensorError::invalid_shape_simple(format!(
                "Group index {} out of bounds (max: {})",
                group_index,
                self.groups.len()
            )));
        }

        for param in parameters {
            self.groups[group_index].add_parameter(param);
        }

        Ok(())
    }

    /// Add parameters to a group by name
    pub fn add_parameters_to_group_by_name(
        &mut self,
        group_name: &str,
        parameters: &[&Tensor<T>],
    ) -> Result<()> {
        let group_index = self
            .groups
            .iter()
            .position(|g| g.name == group_name)
            .ok_or_else(|| {
                TensorError::invalid_shape_simple(format!("Group '{group_name}' not found"))
            })?;

        self.add_parameters_to_group(group_index, parameters)
    }

    /// Get a group by name
    pub fn get_group(&self, name: &str) -> Option<&ParameterGroup<T>> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Get a mutable group by name
    pub fn get_group_mut(&mut self, name: &str) -> Option<&mut ParameterGroup<T>> {
        self.groups.iter_mut().find(|g| g.name == name)
    }

    /// Get all group names
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.iter().map(|g| g.name.as_str()).collect()
    }

    /// Find which group a parameter belongs to
    pub fn find_parameter_group(&self, param: &Tensor<T>) -> Option<usize> {
        self.groups
            .iter()
            .position(|group| group.contains_parameter(param))
    }

    /// Set learning rate for a specific group
    pub fn set_group_learning_rate(
        &mut self,
        group_index: usize,
        learning_rate: f32,
    ) -> Result<()> {
        if group_index >= self.groups.len() {
            return Err(TensorError::invalid_shape_simple(format!(
                "Group index {group_index} out of bounds"
            )));
        }

        self.groups[group_index].config.learning_rate = learning_rate;
        self.group_optimizers[group_index].set_learning_rate(learning_rate);
        Ok(())
    }

    /// Set learning rate for a group by name
    pub fn set_group_learning_rate_by_name(
        &mut self,
        group_name: &str,
        learning_rate: f32,
    ) -> Result<()> {
        let group_index = self
            .groups
            .iter()
            .position(|g| g.name == group_name)
            .ok_or_else(|| {
                TensorError::invalid_shape_simple(format!("Group '{group_name}' not found"))
            })?;

        self.set_group_learning_rate(group_index, learning_rate)
    }

    /// Get learning rate for a specific group
    pub fn get_group_learning_rate(&self, group_index: usize) -> Option<f32> {
        self.group_optimizers
            .get(group_index)
            .map(|opt| opt.get_learning_rate())
    }

    /// Create default groups for common use cases
    pub fn with_pretrained_and_new_layers(
        factory: Box<dyn Fn(&ParameterGroupConfig) -> O + Send + Sync>,
        pretrained_lr: f32,
        new_layer_lr: f32,
    ) -> Self {
        let mut optimizer = Self {
            base_optimizer_factory: factory,
            groups: Vec::new(),
            group_optimizers: Vec::new(),
            default_config: ParameterGroupConfig::new(new_layer_lr),
        };

        // Add pretrained layers group with lower learning rate
        let pretrained_config = ParameterGroupConfig::new(pretrained_lr);
        optimizer
            .add_group("pretrained".to_string(), pretrained_config)
            .unwrap();

        // Add new layers group with higher learning rate
        let new_layers_config = ParameterGroupConfig::new(new_layer_lr);
        optimizer
            .add_group("new_layers".to_string(), new_layers_config)
            .unwrap();

        optimizer
    }
}

/// Wrapper for creating a single-model parameter group optimizer that only optimizes specified parameters
struct FilteredModel<'a, T> {
    model: &'a mut dyn Model<T>,
    allowed_parameters: &'a HashSet<*const Tensor<T>>,
}

impl<'a, T> Model<T> for FilteredModel<'a, T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.model.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        self.model
            .parameters()
            .into_iter()
            .filter(|param| {
                self.allowed_parameters
                    .contains(&(*param as *const Tensor<T>))
            })
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.model
            .parameters_mut()
            .into_iter()
            .filter(|param| {
                self.allowed_parameters
                    .contains(&(*param as *const Tensor<T>))
            })
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        self.model.set_training(training);
    }

    fn zero_grad(&mut self) {
        // Zero grad for all parameters, not just filtered ones
        self.model.zero_grad();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        &() as &dyn std::any::Any // Return a dummy value that satisfies 'static
    }
}

impl<T, O: Optimizer<T>> Optimizer<T> for ParameterGroupOptimizer<T, O> {
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        // Apply each group optimizer to its respective parameters
        for (group_index, group) in self.groups.iter().enumerate() {
            if !group.parameters.is_empty() {
                let mut filtered_model = FilteredModel {
                    model,
                    allowed_parameters: &group.parameters,
                };
                self.group_optimizers[group_index].step(&mut filtered_model)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        // Zero gradients for all parameters
        model.zero_grad();
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        // Set learning rate for all groups
        for (group, optimizer) in self.groups.iter_mut().zip(self.group_optimizers.iter_mut()) {
            group.config.learning_rate = learning_rate;
            optimizer.set_learning_rate(learning_rate);
        }
        self.default_config.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> f32 {
        // Return the default learning rate
        self.default_config.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Dense, Layer};
    use crate::model::Sequential;
    use crate::optimizers::SGD;

    #[test]
    fn test_parameter_group_creation() {
        let config = ParameterGroupConfig::new(0.01)
            .with_weight_decay(0.001)
            .with_custom_param("momentum".to_string(), 0.9f32);

        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.weight_decay, 0.001);
        assert_eq!(config.get_custom_param::<f32>("momentum"), Some(&0.9f32));
    }

    #[test]
    fn test_parameter_group_optimizer() {
        let factory = |config: &ParameterGroupConfig| SGD::<f32>::new(config.learning_rate);
        let default_config = ParameterGroupConfig::new(0.001);
        let mut optimizer = ParameterGroupOptimizer::<f32, SGD<f32>>::new(factory, default_config);

        // Add groups
        let group1_config = ParameterGroupConfig::new(0.01);
        let group2_config = ParameterGroupConfig::new(0.001);

        optimizer
            .add_group("backbone".to_string(), group1_config)
            .unwrap();
        optimizer
            .add_group("classifier".to_string(), group2_config)
            .unwrap();

        assert_eq!(optimizer.group_names(), vec!["backbone", "classifier"]);
        assert_eq!(optimizer.get_group_learning_rate(0), Some(0.01));
        assert_eq!(optimizer.get_group_learning_rate(1), Some(0.001));
    }

    #[test]
    fn test_pretrained_and_new_layers_optimizer() {
        let factory =
            Box::new(|config: &ParameterGroupConfig| SGD::<f32>::new(config.learning_rate));
        let optimizer = ParameterGroupOptimizer::<f32, SGD<f32>>::with_pretrained_and_new_layers(
            factory, 0.0001, 0.01,
        );

        assert_eq!(optimizer.group_names(), vec!["pretrained", "new_layers"]);
        assert_eq!(optimizer.get_group_learning_rate(0), Some(0.0001)); // pretrained
        assert_eq!(optimizer.get_group_learning_rate(1), Some(0.01)); // new_layers
    }

    #[test]
    fn test_group_learning_rate_updates() {
        let factory = |config: &ParameterGroupConfig| SGD::<f32>::new(config.learning_rate);
        let default_config = ParameterGroupConfig::new(0.001);
        let mut optimizer = ParameterGroupOptimizer::<f32, SGD<f32>>::new(factory, default_config);

        optimizer
            .add_group("test".to_string(), ParameterGroupConfig::new(0.01))
            .unwrap();

        // Update learning rate by name
        optimizer
            .set_group_learning_rate_by_name("test", 0.005)
            .unwrap();
        assert_eq!(optimizer.get_group_learning_rate(0), Some(0.005));

        // Update learning rate by index
        optimizer.set_group_learning_rate(0, 0.002).unwrap();
        assert_eq!(optimizer.get_group_learning_rate(0), Some(0.002));
    }

    #[test]
    fn test_parameter_assignment() {
        let factory = |config: &ParameterGroupConfig| SGD::<f32>::new(config.learning_rate);
        let default_config = ParameterGroupConfig::new(0.001);
        let mut optimizer = ParameterGroupOptimizer::<f32, SGD<f32>>::new(factory, default_config);

        optimizer
            .add_group("test".to_string(), ParameterGroupConfig::new(0.01))
            .unwrap();

        // Create some dummy parameters
        let param1 = Tensor::zeros(&[10, 10]);
        let param2 = Tensor::zeros(&[5, 5]);

        // Add parameters to group
        optimizer
            .add_parameters_to_group(0, &[&param1, &param2])
            .unwrap();

        // Check if parameters are in the group
        assert_eq!(optimizer.find_parameter_group(&param1), Some(0));
        assert_eq!(optimizer.find_parameter_group(&param2), Some(0));

        // Check parameter count
        assert_eq!(optimizer.groups[0].parameter_count(), 2);
    }
}
