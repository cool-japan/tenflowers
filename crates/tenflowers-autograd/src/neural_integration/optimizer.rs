//! Optimizer integration for autograd neural networks

use crate::gradient_accumulation::GradientAccumulator;
use crate::tape::{GradientTape, TrackedTensor};
use scirs2_core::numeric::{Float, One, Zero};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::ops::scalar_add;
use tenflowers_core::{Result, Tensor};

/// Autograd-integrated optimizer for neural networks
pub struct AutogradOptimizer<T> {
    /// Gradient tape for automatic differentiation
    tape: Arc<Mutex<GradientTape>>,
    /// Gradient accumulator for large batch training
    accumulator: Option<GradientAccumulator>,
    /// Learning rate for optimization
    learning_rate: T,
    /// Optimizer type
    optimizer_type: OptimizerType,
    /// Optimizer state (for momentum, Adam, etc.)
    optimizer_state: HashMap<String, Tensor<T>>,
}

/// Supported optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    SGD {
        momentum: Option<f32>,
    },
    Adam {
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    RMSprop {
        alpha: f32,
        epsilon: f32,
    },
    Adagrad {
        epsilon: f32,
    },
}

impl<T> AutogradOptimizer<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new SGD optimizer
    pub fn sgd(tape: Arc<Mutex<GradientTape>>, learning_rate: T) -> Self {
        Self {
            tape,
            accumulator: None,
            learning_rate,
            optimizer_type: OptimizerType::SGD { momentum: None },
            optimizer_state: HashMap::new(),
        }
    }

    /// Create a new Adam optimizer
    pub fn adam(tape: Arc<Mutex<GradientTape>>, learning_rate: T) -> Self {
        Self {
            tape,
            accumulator: None,
            learning_rate,
            optimizer_type: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            optimizer_state: HashMap::new(),
        }
    }

    /// Create a new RMSprop optimizer
    pub fn rmsprop(tape: Arc<Mutex<GradientTape>>, learning_rate: T, alpha: f32) -> Self {
        Self {
            tape,
            accumulator: None,
            learning_rate,
            optimizer_type: OptimizerType::RMSprop {
                alpha,
                epsilon: 1e-8,
            },
            optimizer_state: HashMap::new(),
        }
    }

    /// Create a new Adagrad optimizer
    pub fn adagrad(tape: Arc<Mutex<GradientTape>>, learning_rate: T) -> Self {
        Self {
            tape,
            accumulator: None,
            learning_rate,
            optimizer_type: OptimizerType::Adagrad { epsilon: 1e-8 },
            optimizer_state: HashMap::new(),
        }
    }

    /// Add gradient accumulation for large batch training
    pub fn with_accumulation(mut self, accumulator: GradientAccumulator) -> Self {
        self.accumulator = Some(accumulator);
        self
    }

    /// Compute gradients for the given parameters with respect to the loss
    pub fn compute_gradients(
        &self,
        loss: &TrackedTensor<T>,
        parameters: &[&TrackedTensor<T>],
    ) -> Result<Vec<Tensor<T>>> {
        if let Some(ref accumulator) = self.accumulator {
            // Use gradient accumulation
            accumulator.accumulate(&self.tape.lock().unwrap(), loss, parameters)?;

            let mut gradients = Vec::new();
            for param in parameters {
                if let Some(grad) = accumulator.get_gradient(param)? {
                    gradients.push(grad);
                } else {
                    gradients.push(Tensor::zeros(param.tensor.shape().dims()));
                }
            }
            Ok(gradients)
        } else {
            // Direct gradient computation
            let tape_guard = self.tape.lock().unwrap();
            let targets = vec![loss.clone()];
            let sources: Vec<TrackedTensor<T>> = parameters.iter().map(|&p| p.clone()).collect();
            let computed_grads = tape_guard.gradient(&targets, &sources)?;
            let gradients: Vec<Tensor<T>> = computed_grads
                .into_iter()
                .map(|g| g.unwrap_or_else(|| Tensor::zeros(parameters[0].tensor.shape().dims())))
                .collect();

            Ok(gradients)
        }
    }

    /// Apply optimizer update to parameters
    pub fn apply_gradients(
        &mut self,
        parameters: &mut [TrackedTensor<T>],
        gradients: &[Tensor<T>],
    ) -> Result<()> {
        let optimizer_type = self.optimizer_type.clone();
        match &optimizer_type {
            OptimizerType::SGD { momentum } => {
                self.apply_sgd_update(parameters, gradients, momentum)?;
            }
            OptimizerType::Adam {
                beta1,
                beta2,
                epsilon,
            } => {
                self.apply_adam_update(parameters, gradients, *beta1, *beta2, *epsilon)?;
            }
            OptimizerType::RMSprop { alpha, epsilon } => {
                self.apply_rmsprop_update(parameters, gradients, *alpha, *epsilon)?;
            }
            OptimizerType::Adagrad { epsilon } => {
                self.apply_adagrad_update(parameters, gradients, *epsilon)?;
            }
        }
        Ok(())
    }

    /// Apply SGD update
    fn apply_sgd_update(
        &mut self,
        parameters: &mut [TrackedTensor<T>],
        gradients: &[Tensor<T>],
        momentum: &Option<f32>,
    ) -> Result<()> {
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            if let Some(mom) = momentum {
                // Apply momentum
                let momentum_key = format!("momentum_{}", i);
                if let Some(momentum_tensor) = self.optimizer_state.get(&momentum_key) {
                    let momentum_scalar = T::from_f32(*mom).unwrap_or_else(T::zero);
                    let new_momentum = momentum_tensor
                        .mul_scalar(momentum_scalar)?
                        .add(&grad.mul_scalar(self.learning_rate)?)?;

                    let new_param = param.tensor.sub(&new_momentum)?;
                    let tape_ref = self.tape.lock().unwrap();
                    *param = tape_ref.watch(new_param);

                    self.optimizer_state.insert(momentum_key, new_momentum);
                } else {
                    let momentum_tensor = grad.mul_scalar(self.learning_rate)?;
                    let new_param = param.tensor.sub(&momentum_tensor)?;
                    let tape_ref = self.tape.lock().unwrap();
                    *param = tape_ref.watch(new_param);

                    self.optimizer_state.insert(momentum_key, momentum_tensor);
                }
            } else {
                // Simple SGD
                let update = grad.mul_scalar(self.learning_rate)?;
                let new_param = param.tensor.sub(&update)?;
                let tape_ref = self.tape.lock().unwrap();
                *param = tape_ref.watch(new_param);
            }
        }
        Ok(())
    }

    /// Apply Adam update
    fn apply_adam_update(
        &mut self,
        parameters: &mut [TrackedTensor<T>],
        gradients: &[Tensor<T>],
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<()> {
        let beta1_t = T::from_f32(beta1).unwrap_or_else(T::zero);
        let beta2_t = T::from_f32(beta2).unwrap_or_else(T::zero);
        let eps_t = T::from_f32(epsilon).unwrap_or_else(T::zero);

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let m_key = format!("adam_m_{}", i);
            let v_key = format!("adam_v_{}", i);

            // Initialize or get momentum and velocity
            let m = self
                .optimizer_state
                .get(&m_key)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros(grad.shape().dims()));
            let v = self
                .optimizer_state
                .get(&v_key)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros(grad.shape().dims()));

            // Update biased first moment estimate
            let new_m = m
                .mul_scalar(beta1_t)?
                .add(&grad.mul_scalar(T::one() - beta1_t)?)?;

            // Update biased second raw moment estimate
            let grad_squared = grad.mul(grad)?;
            let new_v = v
                .mul_scalar(beta2_t)?
                .add(&grad_squared.mul_scalar(T::one() - beta2_t)?)?;

            // Compute bias-corrected first moment estimate (simplified)
            let m_hat = &new_m;

            // Compute bias-corrected second raw moment estimate (simplified)
            let v_hat = &new_v;

            // Apply update
            let v_sqrt = v_hat.sqrt()?;
            let denominator = scalar_add(&v_sqrt, eps_t)?;
            let update = m_hat.div(&denominator)?.mul_scalar(self.learning_rate)?;
            let new_param = param.tensor.sub(&update)?;

            let tape_ref = self.tape.lock().unwrap();
            *param = tape_ref.watch(new_param);

            // Store updated moments
            self.optimizer_state.insert(m_key, new_m);
            self.optimizer_state.insert(v_key, new_v);
        }
        Ok(())
    }

    /// Apply RMSprop update
    fn apply_rmsprop_update(
        &mut self,
        parameters: &mut [TrackedTensor<T>],
        gradients: &[Tensor<T>],
        alpha: f32,
        epsilon: f32,
    ) -> Result<()> {
        let alpha_t = T::from_f32(alpha).unwrap_or_else(T::zero);
        let eps_t = T::from_f32(epsilon).unwrap_or_else(T::zero);

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let v_key = format!("rmsprop_v_{}", i);

            // Initialize or get accumulated gradient squares
            let v = self
                .optimizer_state
                .get(&v_key)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros(grad.shape().dims()));

            // Update accumulated gradient squares
            let grad_squared = grad.mul(grad)?;
            let new_v = v
                .mul_scalar(alpha_t)?
                .add(&grad_squared.mul_scalar(T::one() - alpha_t)?)?;

            // Apply update
            let v_sqrt = new_v.sqrt()?;
            let denominator = scalar_add(&v_sqrt, eps_t)?;
            let update = grad.div(&denominator)?.mul_scalar(self.learning_rate)?;
            let new_param = param.tensor.sub(&update)?;

            let tape_ref = self.tape.lock().unwrap();
            *param = tape_ref.watch(new_param);

            // Store updated accumulator
            self.optimizer_state.insert(v_key, new_v);
        }
        Ok(())
    }

    /// Apply Adagrad update
    fn apply_adagrad_update(
        &mut self,
        parameters: &mut [TrackedTensor<T>],
        gradients: &[Tensor<T>],
        epsilon: f32,
    ) -> Result<()> {
        let eps_t = T::from_f32(epsilon).unwrap_or_else(T::zero);

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let g_key = format!("adagrad_g_{}", i);

            // Initialize or get accumulated gradient squares
            let g = self
                .optimizer_state
                .get(&g_key)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros(grad.shape().dims()));

            // Update accumulated gradient squares
            let grad_squared = grad.mul(grad)?;
            let new_g = g.add(&grad_squared)?;

            // Apply update
            let g_sqrt = new_g.sqrt()?;
            let denominator = scalar_add(&g_sqrt, eps_t)?;
            let update = grad.div(&denominator)?.mul_scalar(self.learning_rate)?;
            let new_param = param.tensor.sub(&update)?;

            let tape_ref = self.tape.lock().unwrap();
            *param = tape_ref.watch(new_param);

            // Store updated accumulator
            self.optimizer_state.insert(g_key, new_g);
        }
        Ok(())
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, learning_rate: T) {
        self.learning_rate = learning_rate;
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> T {
        self.learning_rate
    }

    /// Zero gradients in accumulator
    pub fn zero_grad(&mut self) -> Result<()> {
        if let Some(ref mut accumulator) = self.accumulator {
            accumulator.clear();
        }
        Ok(())
    }
}
