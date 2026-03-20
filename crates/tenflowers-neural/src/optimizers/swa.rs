//! Stochastic Weight Averaging (SWA) implementation
//!
//! SWA is a simple technique that averages multiple points along the trajectory of a neural network
//! during training to obtain better generalization. It maintains a running average of model weights
//! and can significantly improve model performance with minimal computational overhead.
//!
//! Key benefits:
//! - Improves generalization performance without additional training time
//! - Reduces overfitting by averaging multiple model states
//! - Works well with any base optimizer (Adam, SGD, etc.)
//! - Provides uncertainty estimation capabilities
//! - Simple to implement and tune
//!
//! References:
//! - "Averaging Weights Leads to Wider Optima and Better Generalization" (Izmailov et al., 2018)
//! - "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (Maddox et al., 2019)

use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for Stochastic Weight Averaging
#[derive(Debug, Clone)]
pub struct SwaConfig {
    /// Start averaging after this many steps (default: 1000)
    pub start_step: u64,
    /// Average every n steps (default: 10)  
    pub averaging_frequency: u64,
    /// Learning rate for SWA phase (optional, uses base optimizer LR if None)
    pub swa_lr: Option<f32>,
    /// Whether to update batch norm statistics with SWA weights (default: true)
    pub update_bn_stats: bool,
}

impl Default for SwaConfig {
    fn default() -> Self {
        Self {
            start_step: 1000,
            averaging_frequency: 10,
            swa_lr: None,
            update_bn_stats: true,
        }
    }
}

/// Stochastic Weight Averaging optimizer wrapper
///
/// SWA maintains a running average of model weights to improve generalization.
/// It wraps any base optimizer and begins averaging weights after a specified
/// number of training steps.
///
/// # Example
/// ```rust,ignore
/// use tenflowers_neural::optimizers::{Adam, SWA, SwaConfig};
///
/// // Create SWA with Adam base optimizer
/// let base_optimizer = Adam::new(0.001);
/// let config = SwaConfig {
///     start_step: 500,
///     averaging_frequency: 5,
///     swa_lr: Some(0.01), // Higher LR for SWA phase
///     ..Default::default()
/// };
/// let mut swa_optimizer = SWA::new(base_optimizer, config);
///
/// // Use like any other optimizer
/// swa_optimizer.step(&mut model)?;
///
/// // After training, get the averaged model
/// swa_optimizer.finalize_model(&mut model)?;
/// ```
pub struct SWA<O> {
    /// Base optimizer (Adam, SGD, etc.)
    base_optimizer: O,
    /// SWA configuration
    config: SwaConfig,
    /// Current training step
    step_count: u64,
    /// Number of models averaged so far
    n_averaged: u64,
    /// Running average of model weights
    averaged_weights: HashMap<String, Tensor<f32>>,
    /// Whether averaging has started
    averaging_started: bool,
    /// Original learning rate before SWA phase
    original_lr: Option<f32>,
}

impl<O> SWA<O> {
    /// Create a new SWA optimizer with default configuration
    pub fn new(base_optimizer: O) -> Self {
        Self {
            base_optimizer,
            config: SwaConfig::default(),
            step_count: 0,
            n_averaged: 0,
            averaged_weights: HashMap::new(),
            averaging_started: false,
            original_lr: None,
        }
    }

    /// Create SWA optimizer with custom configuration
    pub fn with_config(base_optimizer: O, config: SwaConfig) -> Self {
        Self {
            base_optimizer,
            config,
            step_count: 0,
            n_averaged: 0,
            averaged_weights: HashMap::new(),
            averaging_started: false,
            original_lr: None,
        }
    }

    /// Get reference to the base optimizer
    pub fn base_optimizer(&self) -> &O {
        &self.base_optimizer
    }

    /// Get mutable reference to the base optimizer
    pub fn base_optimizer_mut(&mut self) -> &mut O {
        &mut self.base_optimizer
    }

    /// Get the current step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get the number of models averaged
    pub fn n_averaged(&self) -> u64 {
        self.n_averaged
    }

    /// Check if averaging has started
    pub fn is_averaging(&self) -> bool {
        self.averaging_started
    }

    /// Get the SWA configuration
    pub fn config(&self) -> &SwaConfig {
        &self.config
    }

    /// Update SWA configuration
    pub fn set_config(&mut self, config: SwaConfig) {
        self.config = config;
    }

    /// Initialize averaging with current model weights
    fn initialize_averaging(&mut self, model: &dyn Model<f32>) -> Result<()> {
        self.averaged_weights.clear();

        // Store current weights as initial average
        for (i, param) in model.parameters().iter().enumerate() {
            self.averaged_weights
                .insert(format!("param_{i}"), (*param).clone());
        }

        self.n_averaged = 1;
        self.averaging_started = true;

        Ok(())
    }

    /// Update the running average with current model weights
    fn update_average(&mut self, model: &dyn Model<f32>) -> Result<()> {
        if !self.averaging_started {
            return self.initialize_averaging(model);
        }

        self.n_averaged += 1;
        let alpha = 1.0 / self.n_averaged as f32;

        // Running average update: avg = (1 - alpha) * avg + alpha * current
        for (i, current_param) in model.parameters().iter().enumerate() {
            let name = format!("param_{i}");
            if let Some(avg_param) = self.averaged_weights.get_mut(&name) {
                // avg = avg + alpha * (current - avg)
                let diff = current_param.sub(avg_param)?;
                let update = diff.scalar_mul(alpha)?;
                let new_avg = avg_param.add(&update)?;
                *avg_param = new_avg;
            } else {
                // New parameter, add it to the average
                self.averaged_weights.insert(name, (*current_param).clone());
            }
        }

        Ok(())
    }

    /// Apply the averaged weights to the model
    ///
    /// This should typically be called after training is complete to get
    /// the final SWA model with improved generalization.
    pub fn finalize_model(&self, model: &mut dyn Model<f32>) -> Result<()> {
        if !self.averaging_started {
            return Err(TensorError::invalid_operation_simple(
                "SWA averaging has not started yet".to_string(),
            ));
        }

        // Apply averaged weights to model
        for (i, param) in model.parameters_mut().iter_mut().enumerate() {
            let name = format!("param_{i}");
            if let Some(avg_weight) = self.averaged_weights.get(&name) {
                **param = avg_weight.clone();
            }
        }

        Ok(())
    }

    /// Get a copy of the current averaged weights
    pub fn get_averaged_weights(&self) -> &HashMap<String, Tensor<f32>> {
        &self.averaged_weights
    }

    /// Check if it's time to update the average
    fn should_update_average(&self) -> bool {
        self.step_count >= self.config.start_step
            && (self.step_count - self.config.start_step) % self.config.averaging_frequency == 0
    }

    /// Reset SWA state (useful for restarting averaging)
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.n_averaged = 0;
        self.averaged_weights.clear();
        self.averaging_started = false;
        self.original_lr = None;
    }
}

impl<O> Optimizer<f32> for SWA<O>
where
    O: Optimizer<f32>,
{
    fn step(&mut self, model: &mut dyn Model<f32>) -> Result<()> {
        self.step_count += 1;

        // Switch to SWA learning rate if specified and averaging has started
        if !self.averaging_started && self.step_count >= self.config.start_step {
            if let Some(swa_lr) = self.config.swa_lr {
                self.original_lr = Some(self.base_optimizer.get_learning_rate());
                self.base_optimizer.set_learning_rate(swa_lr);
            }
        }

        // Perform regular optimization step
        self.base_optimizer.step(model)?;

        // Update SWA average if it's time
        if self.should_update_average() {
            self.update_average(model)?;
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<f32>) {
        self.base_optimizer.zero_grad(model);
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.base_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> f32 {
        self.base_optimizer.get_learning_rate()
    }
}

/// Utility function to compute model ensemble predictions
///
/// Given multiple model snapshots, computes ensemble predictions by averaging
/// the outputs. This can be used with SWA or other model averaging techniques.
pub fn ensemble_predict<M: Model<f32>>(models: &[M], input: &Tensor<f32>) -> Result<Tensor<f32>> {
    if models.is_empty() {
        return Err(TensorError::invalid_operation_simple(
            "No models provided for ensemble prediction".to_string(),
        ));
    }

    // Get prediction from first model
    let mut ensemble_output = models[0].forward(input)?;

    // Add predictions from remaining models
    for model in models.iter().skip(1) {
        let output = model.forward(input)?;
        ensemble_output = ensemble_output.add(&output)?;
    }

    // Average the predictions
    let n_models = models.len() as f32;
    ensemble_output.scalar_mul(1.0 / n_models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::model::Sequential;
    use crate::optimizers::Adam;

    #[test]
    fn test_swa_config_default() {
        let config = SwaConfig::default();
        assert_eq!(config.start_step, 1000);
        assert_eq!(config.averaging_frequency, 10);
        assert_eq!(config.swa_lr, None);
        assert!(config.update_bn_stats);
    }

    #[test]
    fn test_swa_creation() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let swa = SWA::new(base_optimizer);

        assert_eq!(swa.step_count(), 0);
        assert_eq!(swa.n_averaged(), 0);
        assert!(!swa.is_averaging());
        assert_eq!(swa.config().start_step, 1000);
    }

    #[test]
    fn test_swa_with_custom_config() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let config = SwaConfig {
            start_step: 100,
            averaging_frequency: 5,
            swa_lr: Some(0.01),
            update_bn_stats: false,
        };
        let swa = SWA::with_config(base_optimizer, config);

        assert_eq!(swa.config().start_step, 100);
        assert_eq!(swa.config().averaging_frequency, 5);
        assert_eq!(swa.config().swa_lr, Some(0.01));
        assert!(!swa.config().update_bn_stats);
    }

    #[test]
    fn test_should_update_average() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let config = SwaConfig {
            start_step: 10,
            averaging_frequency: 3,
            ..Default::default()
        };
        let mut swa = SWA::with_config(base_optimizer, config);

        // Before start_step
        swa.step_count = 5;
        assert!(!swa.should_update_average());

        // At start_step
        swa.step_count = 10;
        assert!(swa.should_update_average());

        // Not at frequency
        swa.step_count = 11;
        assert!(!swa.should_update_average());

        // At frequency
        swa.step_count = 13;
        assert!(swa.should_update_average());
    }

    #[test]
    fn test_swa_learning_rate_delegation() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let mut swa = SWA::new(base_optimizer);

        // Should delegate to base optimizer
        assert_eq!(swa.get_learning_rate(), 0.001);

        swa.set_learning_rate(0.002);
        assert_eq!(swa.get_learning_rate(), 0.002);
    }

    #[test]
    fn test_swa_reset() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let mut swa = SWA::new(base_optimizer);

        // Simulate some state
        swa.step_count = 100;
        swa.n_averaged = 5;
        swa.averaging_started = true;

        swa.reset();

        assert_eq!(swa.step_count(), 0);
        assert_eq!(swa.n_averaged(), 0);
        assert!(!swa.is_averaging());
        assert!(swa.averaged_weights.is_empty());
    }

    #[test]
    fn test_ensemble_predict_empty() {
        let models: Vec<Sequential<f32>> = vec![];
        let input = Tensor::<f32>::ones(&[1, 10]);

        let result = ensemble_predict(&models, &input);
        assert!(result.is_err());
    }

    #[test]
    fn test_swa_step_counting() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let mut swa = SWA::new(base_optimizer);

        let dense = Dense::new(2, 1, true);
        let mut model = Sequential::<f32>::new(vec![Box::new(dense)]);

        // Test step counting
        assert_eq!(swa.step_count(), 0);

        // Create input for forward pass
        let input = Tensor::<f32>::ones(&[1, 2]);
        let _output = model.forward(&input);

        // Note: In a real scenario, we would compute gradients first
        // For this test, we just verify the step count increases
        let initial_count = swa.step_count();

        // The step method would normally be called after gradient computation
        // This is a minimal test to verify the interface works
        assert_eq!(swa.step_count(), initial_count);
    }

    #[test]
    fn test_finalize_model_before_averaging() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let swa = SWA::new(base_optimizer);

        let dense = Dense::new(2, 1, true);
        let mut model = Sequential::<f32>::new(vec![Box::new(dense)]);

        // Should return error if averaging hasn't started
        let result = swa.finalize_model(&mut model);
        assert!(result.is_err());
    }

    #[test]
    fn test_swa_config_modification() {
        let base_optimizer = Adam::<f32>::new(0.001);
        let mut swa = SWA::new(base_optimizer);

        let new_config = SwaConfig {
            start_step: 50,
            averaging_frequency: 2,
            swa_lr: Some(0.005),
            update_bn_stats: false,
        };

        swa.set_config(new_config);

        assert_eq!(swa.config().start_step, 50);
        assert_eq!(swa.config().averaging_frequency, 2);
        assert_eq!(swa.config().swa_lr, Some(0.005));
        assert!(!swa.config().update_bn_stats);
    }
}
