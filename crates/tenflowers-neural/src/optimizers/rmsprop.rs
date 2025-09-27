use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// RMSprop optimizer with optional momentum
pub struct RMSprop {
    pub learning_rate: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub momentum: Option<f32>,
    pub weight_decay: f32,
    squared_avg: HashMap<*const Tensor<f32>, Tensor<f32>>, // Moving average of squared gradients
    momentum_buffer: HashMap<*const Tensor<f32>, Tensor<f32>>, // Momentum buffer
}

impl RMSprop {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            alpha: 0.99,
            epsilon: 1e-8,
            momentum: None,
            weight_decay: 0.0,
            squared_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

impl Default for RMSprop {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Optimizer<f32> for RMSprop {
    fn step(&mut self, model: &mut dyn Model<f32>) -> Result<()> {
        let alpha_t = self.alpha;
        let lr_t = self.learning_rate;
        let eps_t = self.epsilon;
        let wd_t = self.weight_decay;

        // Update each parameter
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                let param_ptr = param as *const Tensor<f32>;

                // Apply weight decay to gradient if specified
                let grad_with_decay = if wd_t > 0.0 {
                    let weight_decay_term = param.mul(&Tensor::from_scalar(wd_t))?;
                    grad.add(&weight_decay_term)?
                } else {
                    grad.clone()
                };

                // Initialize or get squared average
                let squared_avg = self
                    .squared_avg
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Update squared average: squared_avg = alpha * squared_avg + (1 - alpha) * grad^2
                let one_minus_alpha = Tensor::from_scalar(1.0 - alpha_t);
                let alpha_tensor = Tensor::from_scalar(alpha_t);
                let squared_avg_scaled = squared_avg.mul(&alpha_tensor)?;
                let grad_squared = grad_with_decay.mul(&grad_with_decay)?;
                let grad_squared_scaled = grad_squared.mul(&one_minus_alpha)?;
                *squared_avg = squared_avg_scaled.add(&grad_squared_scaled)?;

                // Compute RMS: sqrt(squared_avg + epsilon)
                let eps_tensor = Tensor::from_scalar(eps_t);
                let rms = squared_avg.add(&eps_tensor)?.sqrt()?;

                // Compute update: grad / rms
                let update = grad_with_decay.div(&rms)?;

                if let Some(momentum_val) = self.momentum {
                    // Apply momentum
                    let momentum_buffer = self
                        .momentum_buffer
                        .entry(param_ptr)
                        .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                    // momentum_buffer = momentum * momentum_buffer + update
                    let momentum_tensor = Tensor::from_scalar(momentum_val);
                    let momentum_term = momentum_buffer.mul(&momentum_tensor)?;
                    *momentum_buffer = momentum_term.add(&update)?;

                    // param = param - lr * momentum_buffer
                    let lr_update = momentum_buffer.mul(&Tensor::from_scalar(lr_t))?;
                    let new_param = param.sub(&lr_update)?;
                    *param = new_param;
                } else {
                    // param = param - lr * update
                    let lr_update = update.mul(&Tensor::from_scalar(lr_t))?;
                    let new_param = param.sub(&lr_update)?;
                    *param = new_param;
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<f32>) {
        model.zero_grad();
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

impl Optimizer<f64> for RMSprop {
    fn step(&mut self, model: &mut dyn Model<f64>) -> Result<()> {
        // Simple gradient descent for f64 version
        let lr_t = self.learning_rate as f64;
        let wd_t = self.weight_decay as f64;

        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                // Apply weight decay if specified
                let grad_with_decay = if wd_t > 0.0 {
                    let weight_decay_term = param.mul(&Tensor::from_scalar(wd_t))?;
                    grad.add(&weight_decay_term)?
                } else {
                    grad.clone()
                };

                // Simple gradient descent
                let lr_tensor = Tensor::from_scalar(lr_t);
                let lr_grad = grad_with_decay.mul(&lr_tensor)?;
                let new_param = param.sub(&lr_grad)?;
                *param = new_param;
            }
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<f64>) {
        model.zero_grad();
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
