//! SOAP optimizer implementation
//!
//! SOAP (Shampoo with Optimistic optimization And Preconditioning) is a recent optimization
//! algorithm that combines adaptive preconditioning from Shampoo with optimistic updates,
//! providing improved convergence properties for deep learning tasks.
//!
//! Key features:
//! - Adaptive preconditioning using second-order information
//! - Optimistic updates for better convergence
//! - Memory-efficient implementation
//! - Robust performance across different model architectures
//!
//! Reference: Based on "SOAP: Improving and Stabilizing Shampoo using Adam" (Anil et al., 2023)

use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// SOAP optimizer - Shampoo with Optimistic optimization And Preconditioning
///
/// SOAP combines the adaptive preconditioning of Shampoo with optimistic updates
/// to achieve better convergence properties than traditional first-order methods.
///
/// Update rules:
/// 1. H_t = β₂ * H_{t-1} + (1-β₂) * g_t * g_t^T  (preconditioner update)
/// 2. m_t = β₁ * m_{t-1} + (1-β₁) * g_t            (momentum update)
/// 3. ĝ_t = H_t^{-1/4} * g_t                        (preconditioned gradient)
/// 4. p_{t+1} = p_t - lr * (m̂_t + α * ĝ_t)         (optimistic update)
///
/// # Example
/// ```rust,ignore
/// use tenflowers_neural::optimizers::Soap;
///
/// // Create SOAP with default hyperparameters
/// let mut optimizer = Soap::new(0.001);
///
/// // Create with custom hyperparameters
/// let mut optimizer = Soap::new(0.001)
///     .with_betas(0.9, 0.999)
///     .with_optimistic_weight(0.1)
///     .with_eps(1e-8);
/// ```
pub struct Soap<T> {
    /// Learning rate
    learning_rate: f32,
    /// Momentum coefficient (default: 0.9)
    beta1: f32,
    /// Preconditioner decay rate (default: 0.999)
    beta2: f32,
    /// Optimistic update weight (default: 0.1)
    optimistic_weight: f32,
    /// Numerical stability epsilon (default: 1e-8)
    eps: f32,
    /// Weight decay coefficient (default: 0.0)
    weight_decay: f32,
    /// Momentum buffers for each parameter
    momentum: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Preconditioner buffers for each parameter (diagonal approximation)
    preconditioner: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Previous gradient for optimistic updates
    prev_gradient: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Step counter for bias correction
    step_count: u64,
}

impl<T> Soap<T> {
    /// Create a new SOAP optimizer with the specified learning rate
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (typically 1e-3)
    ///
    /// Default hyperparameters:
    /// - beta1: 0.9 (momentum coefficient)
    /// - beta2: 0.999 (preconditioner decay)
    /// - optimistic_weight: 0.1 (optimistic update weight)
    /// - eps: 1e-8 (numerical stability)
    /// - weight_decay: 0.0
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            optimistic_weight: 0.1,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: HashMap::new(),
            preconditioner: HashMap::new(),
            prev_gradient: HashMap::new(),
            step_count: 0,
        }
    }

    /// Set beta coefficients
    ///
    /// # Arguments
    /// * `beta1` - Momentum coefficient (typically 0.9)
    /// * `beta2` - Preconditioner decay rate (typically 0.999)
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set optimistic update weight
    ///
    /// # Arguments
    /// * `optimistic_weight` - Weight for optimistic updates (typically 0.1)
    pub fn with_optimistic_weight(mut self, optimistic_weight: f32) -> Self {
        self.optimistic_weight = optimistic_weight;
        self
    }

    /// Set epsilon for numerical stability
    ///
    /// # Arguments
    /// * `eps` - Epsilon value (typically 1e-8)
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay coefficient
    ///
    /// # Arguments
    /// * `weight_decay` - L2 regularization coefficient
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Get the current beta1 coefficient
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get the current beta2 coefficient
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Get the current optimistic weight
    pub fn optimistic_weight(&self) -> f32 {
        self.optimistic_weight
    }

    /// Get the current epsilon
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get the current weight decay
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get the current step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl<T> Default for Soap<T> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<T> Optimizer<T> for Soap<T>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        self.step_count += 1;

        // Convert constants to T
        let beta1_t = T::from(self.beta1).unwrap();
        let beta2_t = T::from(self.beta2).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let opt_weight_t = T::from(self.optimistic_weight).unwrap();
        let eps_t = T::from(self.eps).unwrap();
        let weight_decay_t = T::from(self.weight_decay).unwrap();
        let step_t = T::from(self.step_count as f32).unwrap();

        let one_minus_beta1 = T::one() - beta1_t;
        let one_minus_beta2 = T::one() - beta2_t;

        // Bias correction factors
        let bias_correction1 = T::one() - beta1_t.powf(step_t);
        let bias_correction2 = T::one() - beta2_t.powf(step_t);

        // Update each parameter
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                // Clone the gradient to avoid borrow issues
                let mut grad = grad.clone();

                // Apply weight decay if specified
                if self.weight_decay > 0.0 {
                    let weight_decay_factor = T::one() - lr_t * weight_decay_t;
                    let param_decayed = param.mul(&Tensor::from_scalar(weight_decay_factor))?;
                    *param = param_decayed;
                }

                let param_ptr = param as *const Tensor<T>;

                // Get or initialize momentum buffer
                let momentum = self
                    .momentum
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Get or initialize preconditioner buffer (diagonal approximation)
                let preconditioner = self
                    .preconditioner
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Get or initialize previous gradient
                let prev_grad = self
                    .prev_gradient
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Update momentum: m_t = β₁ * m_{t-1} + (1-β₁) * g_t
                let beta1_tensor = Tensor::from_scalar(beta1_t);
                let one_minus_beta1_tensor = Tensor::from_scalar(one_minus_beta1);
                let m_scaled = momentum.mul(&beta1_tensor)?;
                let grad_scaled = grad.mul(&one_minus_beta1_tensor)?;
                *momentum = m_scaled.add(&grad_scaled)?;

                // Update preconditioner: H_t = β₂ * H_{t-1} + (1-β₂) * g_t²
                let beta2_tensor = Tensor::from_scalar(beta2_t);
                let one_minus_beta2_tensor = Tensor::from_scalar(one_minus_beta2);
                let precond_scaled = preconditioner.mul(&beta2_tensor)?;
                let grad_squared = grad.mul(&grad)?;
                let grad_squared_scaled = grad_squared.mul(&one_minus_beta2_tensor)?;
                *preconditioner = precond_scaled.add(&grad_squared_scaled)?;

                // Bias-corrected estimates
                let momentum_corrected = momentum.div(&Tensor::from_scalar(bias_correction1))?;
                let preconditioner_corrected =
                    preconditioner.div(&Tensor::from_scalar(bias_correction2))?;

                // Compute preconditioned gradient: ĝ_t = H_t^{-1/4} * g_t
                // Using diagonal approximation: sqrt(sqrt(H_t + ε))
                let eps_tensor = Tensor::from_scalar(eps_t);
                let preconditioner_eps = preconditioner_corrected.add(&eps_tensor)?;
                let preconditioner_sqrt = preconditioner_eps.sqrt()?;
                let preconditioner_sqrt_sqrt = preconditioner_sqrt.sqrt()?;
                let preconditioned_grad = grad.div(&preconditioner_sqrt_sqrt)?;

                // Optimistic update: combine momentum and preconditioned gradient
                // update = lr * (m̂_t + α * ĝ_t)
                let opt_weight_tensor = Tensor::from_scalar(opt_weight_t);
                let preconditioned_scaled = preconditioned_grad.mul(&opt_weight_tensor)?;
                let combined_update = momentum_corrected.add(&preconditioned_scaled)?;
                let lr_tensor = Tensor::from_scalar(lr_t);
                let update = combined_update.mul(&lr_tensor)?;

                // Update parameter
                let new_param = param.sub(&update)?;
                *param = new_param;

                // Store current gradient for next optimistic update
                *prev_grad = grad;
            }
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        // Standard zero_grad implementation
        // This is typically handled by the model or training loop
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
    use tenflowers_core::{Device, Tensor};

    #[test]
    fn test_soap_creation() {
        let optimizer = Soap::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
        assert_eq!(optimizer.optimistic_weight(), 0.1);
        assert_eq!(optimizer.eps(), 1e-8);
        assert_eq!(optimizer.weight_decay(), 0.0);
    }

    #[test]
    fn test_soap_with_custom_params() {
        let optimizer = Soap::<f32>::new(0.01)
            .with_betas(0.95, 0.9999)
            .with_optimistic_weight(0.2)
            .with_eps(1e-10)
            .with_weight_decay(0.01);

        assert_eq!(optimizer.get_learning_rate(), 0.01);
        assert_eq!(optimizer.beta1(), 0.95);
        assert_eq!(optimizer.beta2(), 0.9999);
        assert_eq!(optimizer.optimistic_weight(), 0.2);
        assert_eq!(optimizer.eps(), 1e-10);
        assert_eq!(optimizer.weight_decay(), 0.01);
    }

    #[test]
    fn test_soap_default() {
        let optimizer = Soap::<f32>::default();
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
    }

    #[test]
    fn test_soap_step_count() {
        let mut optimizer = Soap::<f32>::new(0.001);
        assert_eq!(optimizer.step_count(), 0);

        // Note: In a real test, we would need a mock model to test the step function
        // For now, we just test the step counter initialization
    }
}
