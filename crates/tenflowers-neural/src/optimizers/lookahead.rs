use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Lookahead Optimizer
///
/// Lookahead is a meta-optimizer that can be applied on top of any existing optimizer.
/// It maintains "slow weights" that are updated less frequently using an exponential moving
/// average of the "fast weights" that are updated by the base optimizer.
///
/// Reference: "Lookahead Optimizer: k steps forward, 1 step back"
/// https://arxiv.org/abs/1907.08610
///
/// Algorithm:
/// 1. Initialize slow weights φ = fast weights θ
/// 2. For k steps, update fast weights θ using base optimizer
/// 3. Update slow weights: φ = φ + α(θ - φ)
/// 4. Reset fast weights: θ = φ
/// 5. Repeat
pub struct Lookahead<T, O: Optimizer<T>> {
    base_optimizer: O,
    alpha: f32,        // slow weight update rate (typically 0.5)
    k: usize,          // update frequency (typically 5-10)
    step_count: usize, // current step counter
    slow_weights: HashMap<*const Tensor<T>, Tensor<T>>, // slow weights φ
    _phantom: std::marker::PhantomData<T>,
}

impl<T, O: Optimizer<T>> Lookahead<T, O> {
    /// Create a new Lookahead optimizer
    ///
    /// # Arguments
    /// * `base_optimizer` - The base optimizer to wrap (e.g., Adam, SGD)
    /// * `alpha` - Slow weight update rate (default: 0.5)
    /// * `k` - Number of fast weight updates before slow weight update (default: 5)
    pub fn new(base_optimizer: O, alpha: f32, k: usize) -> Self {
        Self {
            base_optimizer,
            alpha,
            k,
            step_count: 0,
            slow_weights: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a Lookahead optimizer with default hyperparameters
    ///
    /// Uses α = 0.5 and k = 5, which work well for most applications
    pub fn with_defaults(base_optimizer: O) -> Self {
        Self::new(base_optimizer, 0.5, 5)
    }

    /// Create a Lookahead optimizer with custom α but default k = 5
    pub fn with_alpha(base_optimizer: O, alpha: f32) -> Self {
        Self::new(base_optimizer, alpha, 5)
    }

    /// Create a Lookahead optimizer with custom k but default α = 0.5
    pub fn with_k(base_optimizer: O, k: usize) -> Self {
        Self::new(base_optimizer, 0.5, k)
    }

    /// Get the wrapped base optimizer
    pub fn base_optimizer(&self) -> &O {
        &self.base_optimizer
    }

    /// Get the wrapped base optimizer mutably
    pub fn base_optimizer_mut(&mut self) -> &mut O {
        &mut self.base_optimizer
    }

    /// Get the current slow weight update rate
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get the update frequency
    pub fn k(&self) -> usize {
        self.k
    }

    /// Initialize slow weights from current model parameters
    fn initialize_slow_weights(&mut self, model: &dyn Model<T>) -> Result<()>
    where
        T: Clone,
    {
        for param in model.parameters() {
            let param_ptr = param as *const Tensor<T>;
            self.slow_weights
                .entry(param_ptr)
                .or_insert_with(|| param.clone());
        }
        Ok(())
    }

    /// Update slow weights using exponential moving average
    fn update_slow_weights(&mut self, model: &mut dyn Model<T>) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::Float
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let alpha_t = T::from(self.alpha).unwrap();

        for param in model.parameters_mut() {
            let param_ptr = param as *const Tensor<T>;

            if let Some(slow_weight) = self.slow_weights.get_mut(&param_ptr) {
                // Update slow weights: φ = φ + α(θ - φ) = (1-α)φ + αθ
                let diff = param.sub(slow_weight)?;
                let alpha_diff = diff.mul(&Tensor::from_scalar(alpha_t))?;
                let updated_slow = slow_weight.add(&alpha_diff)?;
                *slow_weight = updated_slow.clone();

                // Reset fast weights to slow weights: θ = φ
                *param = updated_slow;
            }
        }

        Ok(())
    }
}

impl<T, O: Optimizer<T>> Optimizer<T> for Lookahead<T, O>
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
        // Initialize slow weights on first step
        if self.slow_weights.is_empty() {
            self.initialize_slow_weights(model)?;
        }

        // Update fast weights using base optimizer
        self.base_optimizer.step(model)?;

        // Increment step counter
        self.step_count += 1;

        // Update slow weights every k steps
        if self.step_count % self.k == 0 {
            self.update_slow_weights(model)?;
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        self.base_optimizer.zero_grad(model);
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.base_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> f32 {
        self.base_optimizer.get_learning_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;
    use tenflowers_core::Tensor;

    #[test]
    fn test_lookahead_creation() {
        let sgd: SGD<f32> = SGD::new(0.01);
        let lookahead = Lookahead::new(sgd, 0.5, 5);

        assert_eq!(lookahead.alpha(), 0.5);
        assert_eq!(lookahead.k(), 5);
        assert_eq!(lookahead.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_lookahead_with_defaults() {
        let sgd: SGD<f32> = SGD::new(0.01);
        let lookahead = Lookahead::with_defaults(sgd);

        assert_eq!(lookahead.alpha(), 0.5);
        assert_eq!(lookahead.k(), 5);
        assert_eq!(lookahead.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_lookahead_with_alpha() {
        let sgd: SGD<f32> = SGD::new(0.01);
        let lookahead = Lookahead::with_alpha(sgd, 0.8);

        assert_eq!(lookahead.alpha(), 0.8);
        assert_eq!(lookahead.k(), 5);
    }

    #[test]
    fn test_lookahead_with_k() {
        let sgd: SGD<f32> = SGD::new(0.01);
        let lookahead = Lookahead::with_k(sgd, 10);

        assert_eq!(lookahead.alpha(), 0.5);
        assert_eq!(lookahead.k(), 10);
    }

    #[test]
    fn test_lookahead_learning_rate_forwarding() {
        let sgd: SGD<f32> = SGD::new(0.01);
        let mut lookahead = Lookahead::with_defaults(sgd);

        lookahead.set_learning_rate(0.05);
        assert_eq!(lookahead.get_learning_rate(), 0.05);
        assert_eq!(lookahead.base_optimizer().get_learning_rate(), 0.05);
    }
}
