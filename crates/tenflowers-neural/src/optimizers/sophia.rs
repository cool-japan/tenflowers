//! Sophia optimizer implementation
//!
//! Sophia (Second-order Clipped Stochastic Optimization) is a scalable second-order optimizer
//! specifically designed for large language models and transformer architectures.
//!
//! Key features:
//! - Uses diagonal Hessian approximation for second-order information
//! - Clip updates by diagonal Hessian to prevent harmful updates
//! - More efficient than full second-order methods like L-BFGS
//! - Particularly effective for large language model training
//! - Can achieve better convergence than Adam with proper tuning
//!
//! Reference: "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
//! (Liu et al., 2023) - https://arxiv.org/abs/2305.14342

use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Sophia optimizer - Scalable Stochastic Second-order Optimizer
///
/// Sophia approximates the diagonal of the Hessian and uses it to scale updates,
/// providing second-order information while remaining computationally tractable
/// for large-scale training.
///
/// Update rules:
/// 1. Compute gradient g_t
/// 2. Update EMA of gradient: m_t = β₁ * m_{t-1} + (1-β₁) * g_t
/// 3. Estimate diagonal Hessian: h_t = β₂ * h_{t-1} + (1-β₂) * g_t²  (simplified)
/// 4. Compute update: u_t = m_t / (max(γ * I, h_t) + ε)
/// 5. Clip update: u_t = clip(u_t, -ρ, ρ)
/// 6. Apply update: θ_{t+1} = θ_t - α * u_t
///
/// # Example
/// ```rust,ignore
/// use tenflowers_neural::optimizers::Sophia;
///
/// // Create Sophia with default hyperparameters
/// let mut optimizer = Sophia::new(0.0003); // Typical LR for LLM pre-training
///
/// // Create with custom hyperparameters  
/// let mut optimizer = Sophia::new(0.0003)
///     .with_betas(0.965, 0.99)
///     .with_clip_threshold(1.0)
///     .with_weight_decay(0.1);
/// ```
pub struct Sophia<T> {
    /// Learning rate (typically 3e-4 for LLM pre-training)
    learning_rate: f32,
    /// EMA coefficient for gradient momentum (default: 0.965)
    beta1: f32,
    /// EMA coefficient for Hessian diagonal estimate (default: 0.99)
    beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    epsilon: f32,
    /// Clip threshold for updates (default: 1.0)
    clip_threshold: f32,
    /// Weight decay coefficient (default: 0.0)
    weight_decay: f32,
    /// Update frequency for Hessian estimation (default: 10)
    hessian_update_freq: usize,
    /// Current timestep
    t: usize,
    /// Gradient momentum estimates
    m: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Diagonal Hessian estimates  
    h: HashMap<*const Tensor<T>, Tensor<T>>,
}

impl<T> Sophia<T> {
    /// Create new Sophia optimizer with default hyperparameters
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (typically 3e-4 for LLM pre-training)
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.965,
            beta2: 0.99,
            epsilon: 1e-8,
            clip_threshold: 1.0,
            weight_decay: 0.0,
            hessian_update_freq: 10,
            t: 0,
            m: HashMap::new(),
            h: HashMap::new(),
        }
    }

    /// Set momentum coefficients
    ///
    /// # Arguments
    /// * `beta1` - EMA coefficient for gradient momentum (default: 0.965)
    /// * `beta2` - EMA coefficient for Hessian diagonal estimate (default: 0.99)
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set numerical stability epsilon
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set update clipping threshold
    ///
    /// Updates are clipped to [-clip_threshold, clip_threshold] to prevent
    /// harmful updates from poor Hessian estimates.
    pub fn with_clip_threshold(mut self, clip_threshold: f32) -> Self {
        self.clip_threshold = clip_threshold;
        self
    }

    /// Set weight decay coefficient
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set frequency for Hessian diagonal updates
    ///
    /// Higher frequencies provide more accurate Hessian estimates but increase
    /// computational cost. Default is 10 (update every 10 steps).
    pub fn with_hessian_update_freq(mut self, freq: usize) -> Self {
        self.hessian_update_freq = freq;
        self
    }

    /// Get current beta1 parameter
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get current beta2 parameter  
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Get current epsilon parameter
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get current clip threshold
    pub fn clip_threshold(&self) -> f32 {
        self.clip_threshold
    }

    /// Get current weight decay coefficient
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get current timestep
    pub fn timestep(&self) -> usize {
        self.t
    }
}

impl<T> Optimizer<T> for Sophia<T>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Float
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        self.t += 1;

        // Bias correction terms
        let bias_correction1 = T::from(1.0 - self.beta1.powi(self.t as i32)).unwrap();
        let bias_correction2 = T::from(1.0 - self.beta2.powi(self.t as i32)).unwrap();

        // Convert constants to T
        let beta1_t = T::from(self.beta1).unwrap();
        let beta2_t = T::from(self.beta2).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let eps_t = T::from(self.epsilon).unwrap();
        let clip_t = T::from(self.clip_threshold).unwrap();
        let wd_t = T::from(self.weight_decay).unwrap();

        let one_minus_beta1 = T::one() - beta1_t;
        let one_minus_beta2 = T::one() - beta2_t;

        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                // Clone the gradient to avoid borrow issues
                let grad = grad.clone();

                // Apply weight decay if specified
                if self.weight_decay > 0.0 {
                    let weight_decay_factor = T::one() - lr_t * wd_t;
                    let param_decayed = param.mul(&Tensor::from_scalar(weight_decay_factor))?;
                    *param = param_decayed;
                }

                let param_ptr = param as *const Tensor<T>;

                // Initialize or get momentum estimate
                let m = self
                    .m
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Initialize or get Hessian diagonal estimate
                let h = self
                    .h
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Update gradient momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                let beta1_tensor = Tensor::from_scalar(beta1_t);
                let one_minus_beta1_tensor = Tensor::from_scalar(one_minus_beta1);

                let m_scaled = m.mul(&beta1_tensor)?;
                let grad_scaled = grad.mul(&one_minus_beta1_tensor)?;
                let m_updated = m_scaled.add(&grad_scaled)?;
                *m = m_updated;

                // Update Hessian diagonal estimate: h_t = beta2 * h_{t-1} + (1 - beta2) * g_t^2
                if self.t % self.hessian_update_freq == 0 {
                    let grad_squared = grad.mul(&grad)?;
                    let beta2_tensor = Tensor::from_scalar(beta2_t);
                    let one_minus_beta2_tensor = Tensor::from_scalar(one_minus_beta2);

                    let h_scaled = h.mul(&beta2_tensor)?;
                    let grad_sq_scaled = grad_squared.mul(&one_minus_beta2_tensor)?;
                    let h_updated = h_scaled.add(&grad_sq_scaled)?;
                    *h = h_updated;
                }

                // Bias-corrected estimates
                let bias_correction1_tensor = Tensor::from_scalar(bias_correction1);
                let bias_correction2_tensor = Tensor::from_scalar(bias_correction2);

                let m_hat = m.div(&bias_correction1_tensor)?;
                let h_hat = h.div(&bias_correction2_tensor)?;

                // Compute update: u = m_hat / (h_hat + epsilon)
                let eps_tensor = Tensor::from_scalar(eps_t);
                let denominator = h_hat.add(&eps_tensor)?;
                let update = m_hat.div(&denominator)?;

                // Clip update: u = clamp(u, -clip_threshold, clip_threshold)
                let clipped_update = update.clamp(-clip_t, clip_t)?;

                // Apply update: param = param - learning_rate * update
                let lr_tensor = Tensor::from_scalar(lr_t);
                let scaled_update = clipped_update.mul(&lr_tensor)?;
                let param_updated = param.sub(&scaled_update)?;
                *param = param_updated;
            }
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        model.zero_grad();
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleModel {
        weight: Tensor<f32>,
        bias: Tensor<f32>,
    }

    impl SimpleModel {
        fn new() -> Self {
            Self {
                weight: Tensor::ones(&[2, 2]),
                bias: Tensor::zeros(&[2]),
            }
        }
    }

    impl crate::model::Model<f32> for SimpleModel {
        fn forward(&self, input: &Tensor<f32>) -> tenflowers_core::Result<Tensor<f32>> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            vec![&self.weight, &self.bias]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            vec![&mut self.weight, &mut self.bias]
        }

        fn set_training(&mut self, _training: bool) {
            // No-op for testing
        }

        fn zero_grad(&mut self) {
            self.weight.set_grad(None);
            self.bias.set_grad(None);
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_sophia_creation() {
        let optimizer = Sophia::<f32>::new(0.0003);
        assert_eq!(optimizer.get_learning_rate(), 0.0003);
        assert_eq!(optimizer.beta1(), 0.965);
        assert_eq!(optimizer.beta2(), 0.99);
        assert_eq!(optimizer.epsilon(), 1e-8);
        assert_eq!(optimizer.clip_threshold(), 1.0);
        assert_eq!(optimizer.weight_decay(), 0.0);
        assert_eq!(optimizer.timestep(), 0);
    }

    #[test]
    fn test_sophia_with_custom_params() {
        let optimizer = Sophia::<f32>::new(0.001)
            .with_betas(0.9, 0.999)
            .with_epsilon(1e-6)
            .with_clip_threshold(0.5)
            .with_weight_decay(0.01)
            .with_hessian_update_freq(5);

        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
        assert_eq!(optimizer.epsilon(), 1e-6);
        assert_eq!(optimizer.clip_threshold(), 0.5);
        assert_eq!(optimizer.weight_decay(), 0.01);
    }

    #[test]
    fn test_sophia_basic_optimization() -> Result<()> {
        let mut model = SimpleModel::new();
        let mut optimizer = Sophia::new(0.1);

        // Set some gradients
        model.weight.set_grad(Some(Tensor::ones(&[2, 2])));
        model.bias.set_grad(Some(Tensor::ones(&[2])));

        // Take optimization step
        optimizer.step(&mut model)?;

        // Timestep should have incremented
        assert_eq!(optimizer.timestep(), 1);

        Ok(())
    }

    #[test]
    fn test_sophia_zero_grad() {
        let mut model = SimpleModel::new();

        // Set gradients
        model.weight.set_grad(Some(Tensor::ones(&[2, 2])));
        model.bias.set_grad(Some(Tensor::ones(&[2])));

        let optimizer = Sophia::<f32>::new(0.1);
        optimizer.zero_grad(&mut model);

        // All gradients should be None after zero_grad
        assert!(model.weight.grad().is_none());
        assert!(model.bias.grad().is_none());
    }

    #[test]
    fn test_sophia_learning_rate_update() {
        let mut optimizer = Sophia::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);

        optimizer.set_learning_rate(0.0001);
        assert_eq!(optimizer.get_learning_rate(), 0.0001);
    }
}
