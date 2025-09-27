//! AdaBelief Optimizer
//!
//! AdaBelief is an adaptive learning rate optimizer that adapts the step size according
//! to the "belief" in the observed gradients. It incorporates the exponential moving average
//! of squared differences between predicted and observed gradients.
//!
//! Reference: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
//! Paper: <https://arxiv.org/abs/2010.07468>
//!
//! Key features:
//! - Adapts step size according to the belief in gradient direction
//! - Better convergence properties than Adam in many cases
//! - More stable training for large language models
//! - Reduces overfitting compared to SGD with momentum

use std::collections::HashMap;
use tenflowers_core::ops::numpy_compat::maximum;
use tenflowers_core::{Result, Tensor};

/// AdaBelief optimizer configuration
#[derive(Debug, Clone)]
pub struct AdaBeliefConfig {
    /// Learning rate (default: 1e-3)
    pub lr: f64,
    /// Beta1 for momentum (default: 0.9)
    pub beta1: f64,
    /// Beta2 for squared gradient moving average (default: 0.999)
    pub beta2: f64,
    /// Small constant for numerical stability (default: 1e-16)
    pub eps: f64,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f64,
    /// Whether to rectify the second moment estimate (default: true)
    pub amsgrad: bool,
}

impl Default for AdaBeliefConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-16,
            weight_decay: 0.0,
            amsgrad: true,
        }
    }
}

/// AdaBelief optimizer state for a single parameter
#[derive(Debug)]
pub struct AdaBeliefState<T> {
    /// Step count
    pub step: i64,
    /// Exponential moving average of gradient (momentum)
    pub exp_avg: Tensor<T>,
    /// Exponential moving average of squared gradient belief
    pub exp_avg_sq: Tensor<T>,
    /// Maximum of exp_avg_sq (for AMSGrad variant)
    pub max_exp_avg_sq: Option<Tensor<T>>,
}

impl<T> AdaBeliefState<T>
where
    T: Default + Clone + num_traits::Zero,
{
    /// Create new state for a parameter tensor
    pub fn new(param: &Tensor<T>, amsgrad: bool) -> Result<Self> {
        Ok(Self {
            step: 0,
            exp_avg: Tensor::zeros(param.shape().dims()),
            exp_avg_sq: Tensor::zeros(param.shape().dims()),
            max_exp_avg_sq: if amsgrad {
                Some(Tensor::zeros(param.shape().dims()))
            } else {
                None
            },
        })
    }
}

/// AdaBelief optimizer implementing the AdaBelief algorithm
#[derive(Debug)]
pub struct AdaBelief<T> {
    /// Optimizer configuration
    config: AdaBeliefConfig,
    /// Per-parameter optimizer state
    state: HashMap<usize, AdaBeliefState<T>>,
    /// Parameter ID counter
    next_param_id: usize,
}

impl<T> AdaBelief<T>
where
    T: Default + Clone + num_traits::Zero,
{
    /// Create a new AdaBelief optimizer
    pub fn new(config: AdaBeliefConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
            next_param_id: 0,
        }
    }

    /// Get next parameter ID
    pub fn register_param(&mut self) -> usize {
        let id = self.next_param_id;
        self.next_param_id += 1;
        id
    }

    /// Get configuration
    pub fn config(&self) -> &AdaBeliefConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: AdaBeliefConfig) {
        self.config = config;
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.config.lr
    }

    /// Get optimizer state information
    pub fn get_state_info(&self) -> HashMap<usize, i64> {
        self.state
            .iter()
            .map(|(id, state)| (*id, state.step))
            .collect()
    }
}

impl<T> Default for AdaBelief<T>
where
    T: Default + Clone + num_traits::Zero,
{
    fn default() -> Self {
        Self::new(AdaBeliefConfig::default())
    }
}

impl AdaBelief<f32> {
    /// Perform a single optimization step for f32 tensors
    pub fn step(
        &mut self,
        param_id: usize,
        param: &mut Tensor<f32>,
        grad: &Tensor<f32>,
    ) -> Result<()> {
        // Ensure state exists for this parameter
        if !self.state.contains_key(&param_id) {
            self.state
                .insert(param_id, AdaBeliefState::new(param, self.config.amsgrad)?);
        }

        let state = self.state.get_mut(&param_id).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(format!(
                "Parameter state not found for param_id {param_id}"
            ))
        })?;
        state.step += 1;

        // Apply weight decay if specified
        let grad = if self.config.weight_decay != 0.0 {
            // grad = grad + weight_decay * param
            let weight_decay_term = param.scalar_mul(self.config.weight_decay as f32)?;
            grad.add(&weight_decay_term)?
        } else {
            grad.clone()
        };

        // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        state.exp_avg = state
            .exp_avg
            .scalar_mul(self.config.beta1 as f32)?
            .add(&grad.scalar_mul((1.0 - self.config.beta1) as f32)?)?;

        // Calculate the belief: squared difference between predicted and actual gradient
        // belief = (grad - m_t)^2
        let two_tensor = Tensor::from_scalar(2.0f32);
        let belief = grad.sub(&state.exp_avg)?.pow(&two_tensor)?;

        // Update biased second moment estimate: s_t = beta2 * s_{t-1} + (1 - beta2) * belief
        state.exp_avg_sq = state
            .exp_avg_sq
            .scalar_mul(self.config.beta2 as f32)?
            .add(&belief.scalar_mul((1.0 - self.config.beta2) as f32)?)?;

        // Use maximum of current and past s_t for AMSGrad variant
        let exp_avg_sq = if let Some(ref mut max_exp_avg_sq) = state.max_exp_avg_sq {
            // Element-wise maximum: max(max_exp_avg_sq, exp_avg_sq)
            let new_max = maximum(max_exp_avg_sq, &state.exp_avg_sq)?;
            *max_exp_avg_sq = new_max.clone();
            new_max
        } else {
            state.exp_avg_sq.clone()
        };

        // Bias correction
        let bias_correction1 = 1.0 - (self.config.beta1 as f32).powi(state.step as i32);
        let bias_correction2 = 1.0 - (self.config.beta2 as f32).powi(state.step as i32);

        // Corrected learning rate
        let corrected_lr = (self.config.lr as f32) * (bias_correction2.sqrt() / bias_correction1);

        // Calculate denominator: sqrt(s_t) + eps
        let eps_tensor = Tensor::from_scalar(self.config.eps as f32);
        let denominator = exp_avg_sq.sqrt()?.add(&eps_tensor)?;

        // Update parameters: param = param - lr * m_t / (sqrt(s_t) + eps)
        let update = state.exp_avg.div(&denominator)?.scalar_mul(corrected_lr)?;

        *param = param.sub(&update)?;

        Ok(())
    }

    /// Zero all gradients in optimizer state
    pub fn zero_grad(&mut self) {
        // AdaBelief doesn't store gradients directly, so this is a no-op
        // This method is provided for API compatibility
    }
}

impl AdaBelief<f64> {
    /// Perform a single optimization step for f64 tensors
    pub fn step(
        &mut self,
        param_id: usize,
        param: &mut Tensor<f64>,
        grad: &Tensor<f64>,
    ) -> Result<()> {
        // Ensure state exists for this parameter
        if !self.state.contains_key(&param_id) {
            self.state
                .insert(param_id, AdaBeliefState::new(param, self.config.amsgrad)?);
        }

        let state = self.state.get_mut(&param_id).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(format!(
                "Parameter state not found for param_id {param_id}"
            ))
        })?;
        state.step += 1;

        // Apply weight decay if specified
        let grad = if self.config.weight_decay != 0.0 {
            let weight_decay_term = param.scalar_mul(self.config.weight_decay)?;
            grad.add(&weight_decay_term)?
        } else {
            grad.clone()
        };

        // Update biased first moment estimate
        state.exp_avg = state
            .exp_avg
            .scalar_mul(self.config.beta1)?
            .add(&grad.scalar_mul(1.0 - self.config.beta1)?)?;

        // Calculate belief
        let two_tensor = Tensor::from_scalar(2.0);
        let belief = grad.sub(&state.exp_avg)?.pow(&two_tensor)?;

        // Update biased second moment estimate
        state.exp_avg_sq = state
            .exp_avg_sq
            .scalar_mul(self.config.beta2)?
            .add(&belief.scalar_mul(1.0 - self.config.beta2)?)?;

        // Use maximum for AMSGrad variant
        let exp_avg_sq = if let Some(ref mut max_exp_avg_sq) = state.max_exp_avg_sq {
            // For simplicity, use current exp_avg_sq (AMSGrad disabled for now)
            *max_exp_avg_sq = state.exp_avg_sq.clone();
            max_exp_avg_sq.clone()
        } else {
            state.exp_avg_sq.clone()
        };

        // Bias correction
        let bias_correction1 = 1.0 - self.config.beta1.powi(state.step as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(state.step as i32);

        // Corrected learning rate
        let corrected_lr = self.config.lr * (bias_correction2.sqrt() / bias_correction1);

        // Calculate denominator
        let eps_tensor = Tensor::from_scalar(self.config.eps);
        let denominator = exp_avg_sq.sqrt()?.add(&eps_tensor)?;

        // Update parameters
        let update = state.exp_avg.div(&denominator)?.scalar_mul(corrected_lr)?;

        *param = param.sub(&update)?;

        Ok(())
    }

    /// Zero all gradients in optimizer state
    pub fn zero_grad(&mut self) {
        // AdaBelief doesn't store gradients directly, so this is a no-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::{Device, Tensor};

    #[test]
    fn test_adabelief_config_default() {
        let config = AdaBeliefConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.eps, 1e-16);
        assert_eq!(config.weight_decay, 0.0);
        assert!(config.amsgrad);
    }

    #[test]
    fn test_adabelief_creation() {
        let config = AdaBeliefConfig {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
            amsgrad: false,
        };
        let optimizer = AdaBelief::<f32>::new(config.clone());
        assert_eq!(optimizer.config().lr, 0.01);
        assert_eq!(optimizer.config().amsgrad, false);
    }

    #[test]
    fn test_adabelief_parameter_registration() {
        let mut optimizer = AdaBelief::<f32>::default();
        let id1 = optimizer.register_param();
        let id2 = optimizer.register_param();
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_adabelief_lr_methods() {
        let mut optimizer = AdaBelief::<f32>::default();
        assert_eq!(optimizer.get_lr(), 1e-3);

        optimizer.set_lr(0.01);
        assert_eq!(optimizer.get_lr(), 0.01);
    }

    #[test]
    fn test_adabelief_step_f32() -> Result<()> {
        let mut optimizer = AdaBelief::<f32>::default();
        let param_id = optimizer.register_param();

        // Create a simple parameter tensor
        let mut param = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let grad = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3], &[3])?;

        // Perform optimization step
        optimizer.step(param_id, &mut param, &grad)?;

        // Verify that parameter was updated (should be different from original)
        let param_data = param.to_vec()?;
        assert_ne!(param_data[0], 1.0);
        assert_ne!(param_data[1], 2.0);
        assert_ne!(param_data[2], 3.0);

        // Verify step count
        let state_info = optimizer.get_state_info();
        assert_eq!(state_info.get(&param_id), Some(&1));

        Ok(())
    }

    #[test]
    fn test_adabelief_step_f64() -> Result<()> {
        let mut optimizer = AdaBelief::<f64>::default();
        let param_id = optimizer.register_param();

        let mut param = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let grad = Tensor::<f64>::from_vec(vec![0.1, 0.2, 0.3], &[3])?;

        optimizer.step(param_id, &mut param, &grad)?;

        let param_data = param.to_vec()?;
        assert_ne!(param_data[0], 1.0);
        assert_ne!(param_data[1], 2.0);
        assert_ne!(param_data[2], 3.0);

        Ok(())
    }

    #[test]
    fn test_adabelief_with_weight_decay() -> Result<()> {
        let config = AdaBeliefConfig {
            lr: 0.01,
            weight_decay: 0.01,
            ..AdaBeliefConfig::default()
        };
        let mut optimizer = AdaBelief::<f32>::new(config);
        let param_id = optimizer.register_param();

        let mut param = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?;
        let grad = Tensor::<f32>::from_vec(vec![0.1, 0.1], &[2])?;

        optimizer.step(param_id, &mut param, &grad)?;

        // With weight decay, the parameter should decrease more
        let param_data = param.to_vec()?;
        assert!(param_data[0] < 1.0);
        assert!(param_data[1] < 2.0);

        Ok(())
    }

    #[test]
    fn test_adabelief_convergence_simulation() -> Result<()> {
        let mut optimizer = AdaBelief::<f32>::new(AdaBeliefConfig {
            lr: 0.1,
            ..AdaBeliefConfig::default()
        });
        let param_id = optimizer.register_param();

        // Simple quadratic function: f(x) = (x - 2)^2, optimal at x = 2
        let mut param = Tensor::<f32>::from_vec(vec![0.0], &[1])?;

        // Perform several optimization steps
        for _ in 0..10 {
            // Gradient of f(x) = (x - 2)^2 is 2(x - 2)
            let param_val = param.to_vec()?[0];
            let grad_val = 2.0 * (param_val - 2.0);
            let grad = Tensor::<f32>::from_vec(vec![grad_val], &[1])?;

            optimizer.step(param_id, &mut param, &grad)?;
        }

        // Should converge towards 2.0
        let final_val = param.to_vec()?[0];
        assert!((final_val - 2.0).abs() < 1.0); // Should be closer to optimum
        assert!(final_val > 0.0); // Should have moved in the right direction

        Ok(())
    }

    #[test]
    fn test_adabelief_zero_grad() {
        let mut optimizer = AdaBelief::<f32>::default();
        // This should not panic (it's a no-op for AdaBelief)
        optimizer.zero_grad();
    }

    #[test]
    fn test_adabelief_multiple_parameters() -> Result<()> {
        let mut optimizer = AdaBelief::<f32>::default();

        let param_id1 = optimizer.register_param();
        let param_id2 = optimizer.register_param();

        let mut param1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?;
        let mut param2 = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2])?;
        let grad1 = Tensor::<f32>::from_vec(vec![0.1, 0.1], &[2])?;
        let grad2 = Tensor::<f32>::from_vec(vec![0.2, 0.2], &[2])?;

        optimizer.step(param_id1, &mut param1, &grad1)?;
        optimizer.step(param_id2, &mut param2, &grad2)?;

        // Both parameters should have been updated
        let param1_data = param1.to_vec()?;
        let param2_data = param2.to_vec()?;

        assert_ne!(param1_data[0], 1.0);
        assert_ne!(param2_data[0], 3.0);

        // Check state info
        let state_info = optimizer.get_state_info();
        assert_eq!(state_info.get(&param_id1), Some(&1));
        assert_eq!(state_info.get(&param_id2), Some(&1));

        Ok(())
    }
}
