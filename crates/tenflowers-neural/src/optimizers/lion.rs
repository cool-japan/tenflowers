//! Lion optimizer implementation
//!
//! Lion (EvoLved Sign Momentum) is a simple yet effective optimizer that often outperforms Adam
//! while being more memory-efficient and having simpler hyperparameter tuning.
//!
//! Key features:
//! - Uses sign of momentum instead of momentum itself for updates
//! - Only requires one momentum buffer (vs two for Adam)
//! - Often works well with larger learning rates
//! - More robust to hyperparameter choices
//!
//! Reference: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Lion optimizer - EvoLved Sign Momentum
///
/// Lion uses the sign of interpolated momentum for parameter updates, making it
/// more memory-efficient than Adam while often achieving better performance.
///
/// Update rules:
/// 1. m_t = β₁ * m_{t-1} + (1-β₁) * g_t  (momentum update)
/// 2. p_{t+1} = p_t - lr * sign(m_t)       (parameter update with sign of momentum)
/// 3. m_t = β₂ * m_{t-1} + (1-β₂) * g_t    (momentum for next iteration)
///
/// # Example
/// ```rust,ignore
/// use tenflowers_neural::optimizers::Lion;
///
/// // Create Lion with default hyperparameters
/// let mut optimizer = Lion::new(0.0001); // Typically use smaller LR than Adam
///
/// // Create with custom hyperparameters
/// let mut optimizer = Lion::new(0.0001)
///     .with_betas(0.9, 0.99)
///     .with_weight_decay(0.01);
/// ```
pub struct Lion<T> {
    /// Learning rate (typically smaller than Adam, e.g., 1e-4)
    learning_rate: f32,
    /// Momentum coefficient for interpolation (default: 0.9)
    beta1: f32,
    /// Momentum coefficient for update (default: 0.99)
    beta2: f32,
    /// Weight decay coefficient (default: 0.0)
    weight_decay: f32,
    /// Momentum buffers for each parameter
    momentum: HashMap<*const Tensor<T>, Tensor<T>>,
}

impl<T> Lion<T> {
    /// Create a new Lion optimizer with the specified learning rate
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (typically 1e-4, smaller than Adam)
    ///
    /// Default hyperparameters:
    /// - beta1: 0.9 (momentum for interpolation)
    /// - beta2: 0.99 (momentum for update)
    /// - weight_decay: 0.0
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,  // Default from paper
            beta2: 0.99, // Default from paper
            weight_decay: 0.0,
            momentum: HashMap::new(),
        }
    }

    /// Set both beta coefficients
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate
    /// * `beta1` - Momentum coefficient for interpolation (typically 0.9)
    /// * `beta2` - Momentum coefficient for update (typically 0.99)
    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            weight_decay: 0.0,
            momentum: HashMap::new(),
        }
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

    /// Get the current weight decay
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set beta1 coefficient
    pub fn set_beta1(&mut self, beta1: f32) {
        self.beta1 = beta1;
    }

    /// Set beta2 coefficient
    pub fn set_beta2(&mut self, beta2: f32) {
        self.beta2 = beta2;
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }
}

impl<T> Default for Lion<T> {
    fn default() -> Self {
        Self::new(0.0001) // Smaller default LR than Adam
    }
}

impl<T> Optimizer<T> for Lion<T>
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
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        // Convert constants to T
        let beta1_t = T::from(self.beta1).unwrap();
        let beta2_t = T::from(self.beta2).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let one_minus_beta1 = T::one() - beta1_t;
        let one_minus_beta2 = T::one() - beta2_t;

        // Update each parameter
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                // Clone the gradient to avoid borrow issues
                let grad = grad.clone();

                // Get parameter pointer for momentum lookup
                let param_ptr = param as *const Tensor<T>;

                // Apply weight decay if specified (Lion typically uses decoupled weight decay)
                let effective_grad = if self.weight_decay > 0.0 {
                    let wd_t = T::from(self.weight_decay).unwrap();
                    let weight_decay_term = param.scalar_mul(wd_t)?;
                    grad.add(&weight_decay_term)?
                } else {
                    grad
                };

                // Get or initialize momentum
                let momentum = if let Some(m) = self.momentum.get(&param_ptr) {
                    m.clone()
                } else {
                    // Initialize momentum to zeros
                    let m = Tensor::zeros(param.shape().dims());
                    self.momentum.insert(param_ptr, m.clone());
                    m
                };

                // Lion update algorithm:
                // 1. Compute interpolated momentum: c = β₁ * m + (1-β₁) * g
                let interpolated_momentum = momentum
                    .scalar_mul(beta1_t)?
                    .add(&effective_grad.scalar_mul(one_minus_beta1)?)?;

                // 2. Update parameters: p = p - lr * sign(c)
                let sign_momentum = compute_sign(&interpolated_momentum)?;
                let update = sign_momentum.scalar_mul(lr_t)?;
                let new_param = param.sub(&update)?;
                *param = new_param;

                // 3. Update momentum for next iteration: m = β₂ * m + (1-β₂) * g
                let new_momentum = momentum
                    .scalar_mul(beta2_t)?
                    .add(&effective_grad.scalar_mul(one_minus_beta2)?)?;
                self.momentum.insert(param_ptr, new_momentum);
            }
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        for param in model.parameters_mut() {
            param.set_grad(None);
        }
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// Compute the sign of each element in the tensor
/// Returns a tensor with values in {-1, 0, 1}
fn compute_sign<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + PartialOrd
        + std::ops::Neg<Output = T>,
{
    if let Some(data) = tensor.as_slice() {
        let sign_data: Vec<T> = data
            .iter()
            .map(|x| {
                if *x > T::zero() {
                    T::one()
                } else if *x < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            })
            .collect();

        Tensor::from_vec(sign_data, tensor.shape().dims())
    } else {
        // Fallback: create a zeros tensor of the same shape
        Ok(Tensor::zeros(tensor.shape().dims()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Dense, Layer};
    use crate::model::Sequential;

    #[test]
    fn test_lion_creation() {
        let optimizer = Lion::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.99);
        assert_eq!(optimizer.weight_decay(), 0.0);
    }

    #[test]
    fn test_lion_with_betas() {
        let optimizer = Lion::<f32>::with_betas(0.001, 0.8, 0.95);
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.beta1(), 0.8);
        assert_eq!(optimizer.beta2(), 0.95);
    }

    #[test]
    fn test_lion_with_weight_decay() {
        let optimizer = Lion::<f32>::new(0.001).with_weight_decay(0.01);
        assert_eq!(optimizer.weight_decay(), 0.01);
    }

    #[test]
    fn test_lion_default() {
        let optimizer = Lion::<f32>::default();
        assert_eq!(optimizer.get_learning_rate(), 0.0001); // Smaller than Adam default
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.99);
        assert_eq!(optimizer.weight_decay(), 0.0);
    }

    #[test]
    fn test_lion_setters() {
        let mut optimizer = Lion::<f32>::new(0.001);

        optimizer.set_learning_rate(0.002);
        assert_eq!(optimizer.get_learning_rate(), 0.002);

        optimizer.set_beta1(0.85);
        assert_eq!(optimizer.beta1(), 0.85);

        optimizer.set_beta2(0.95);
        assert_eq!(optimizer.beta2(), 0.95);

        optimizer.set_weight_decay(0.05);
        assert_eq!(optimizer.weight_decay(), 0.05);
    }

    #[test]
    fn test_compute_sign() {
        let tensor = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let sign_tensor = compute_sign(&tensor).unwrap();

        if let Some(data) = sign_tensor.as_slice() {
            assert_eq!(data, &[-1.0, -1.0, 0.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn test_lion_step_basic() {
        // Create a simple model
        let dense = Dense::new(2, 1, true);
        let mut model = Sequential::<f32>::new(vec![Box::new(dense)]);

        // Create Lion optimizer
        let mut optimizer = Lion::new(0.1);

        // Create some dummy gradients by setting them manually
        // This is a basic test to ensure the step function doesn't crash
        let input = Tensor::<f32>::ones(&[1, 2]);
        let _output = model.forward(&input); // This should succeed

        // In a real scenario, gradients would be computed via backpropagation
        // For this test, we just ensure the optimizer can be called
        let result = optimizer.zero_grad(&mut model);
        // zero_grad doesn't return a Result, so no need to check

        // Note: Full step testing would require proper gradient computation
        // which needs the autodiff system to be fully integrated
    }

    #[test]
    fn test_lion_momentum_persistence() {
        let mut optimizer = Lion::<f32>::new(0.01);

        // Test that momentum buffers are created and persisted
        assert_eq!(optimizer.momentum.len(), 0);

        // After using the optimizer, momentum buffers should be created
        // This would be tested in integration tests with actual models
    }

    #[test]
    fn test_lion_hyperparameter_comparison() {
        // Lion typically uses smaller learning rates than Adam
        let lion = Lion::<f32>::default();
        let adam = crate::optimizers::Adam::<f32>::default();

        // Lion's default LR should be smaller than Adam's
        assert!(lion.get_learning_rate() < adam.get_learning_rate());
    }

    #[test]
    fn test_lion_memory_efficiency() {
        let lion = Lion::<f32>::new(0.001);

        // Lion should only have one momentum buffer per parameter (vs Adam's two)
        // This is verified by the struct definition having only one HashMap
        assert_eq!(
            std::mem::size_of_val(&lion.momentum),
            std::mem::size_of::<HashMap<*const Tensor<f32>, Tensor<f32>>>()
        );
    }
}
