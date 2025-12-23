use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Adagrad optimizer with adaptive learning rates
///
/// The Adagrad algorithm adapts the learning rate to the parameters, performing smaller updates
/// for parameters associated with frequently occurring features, and larger updates for parameters
/// associated with infrequent features.
///
/// Update rule:
/// - G_t = G_{t-1} + g_t^2 (accumulate squared gradients)
/// - θ_t = θ_{t-1} - lr * g_t / (sqrt(G_t) + ε)
pub struct Adagrad<T> {
    learning_rate: f32,
    epsilon: f32,
    accumulated_gradients: HashMap<*const Tensor<T>, Tensor<T>>, // G_t: accumulated squared gradients
}

impl<T> Adagrad<T> {
    /// Create a new Adagrad optimizer with the specified learning rate
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            epsilon: 1e-10, // Default epsilon for numerical stability
            accumulated_gradients: HashMap::new(),
        }
    }

    /// Set the epsilon value for numerical stability
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

impl<T> Default for Adagrad<T> {
    fn default() -> Self {
        Self::new(0.01) // Default learning rate for Adagrad is typically higher than Adam
    }
}

impl<T> Optimizer<T> for Adagrad<T>
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
        // Convert constants to T
        let lr_t = T::from(self.learning_rate).unwrap();
        let eps_t = T::from(self.epsilon).unwrap();

        // Update each parameter
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                let param_ptr = param as *const Tensor<T>;

                // Initialize or get accumulated gradients
                let accumulated = self
                    .accumulated_gradients
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Accumulate squared gradients: G_t = G_{t-1} + g_t^2
                let grad_squared = grad.mul(grad)?;
                *accumulated = accumulated.add(&grad_squared)?;

                // Compute adaptive learning rate: lr / (sqrt(G_t) + ε)
                let accumulated_sqrt = accumulated.sqrt()?;
                let denominator = accumulated_sqrt.add(&Tensor::from_scalar(eps_t))?;
                let adaptive_lr = Tensor::from_scalar(lr_t).div(&denominator)?;

                // Update parameters: θ_t = θ_{t-1} - adaptive_lr * g_t
                let update = grad.mul(&adaptive_lr)?;
                let new_param = param.sub(&update)?;

                *param = new_param;
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
    use tenflowers_core::Tensor;

    // Mock model for testing
    struct MockModel<T> {
        param: Tensor<T>,
    }

    impl MockModel<f32> {
        fn new() -> Self {
            let mut param = Tensor::<f32>::ones(&[2, 2]);
            // Set a mock gradient
            param.set_grad(Some(
                Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).unwrap(),
            ));

            Self { param }
        }
    }

    impl<T> Model<T> for MockModel<T>
    where
        T: Clone + Default + 'static,
    {
        fn forward(&self, _input: &Tensor<T>) -> Result<Tensor<T>> {
            Ok(self.param.clone())
        }

        fn parameters(&self) -> Vec<&Tensor<T>> {
            vec![&self.param]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
            vec![&mut self.param]
        }

        fn zero_grad(&mut self) {
            self.param.set_grad(None);
        }

        fn set_training(&mut self, _training: bool) {}

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_adagrad_creation() {
        let optimizer = Adagrad::<f32>::new(0.01);
        assert_eq!(optimizer.learning_rate, 0.01);
        assert_eq!(optimizer.epsilon, 1e-10);
    }

    #[test]
    fn test_adagrad_with_epsilon() {
        let optimizer = Adagrad::<f32>::new(0.01).with_epsilon(1e-8);
        assert_eq!(optimizer.epsilon, 1e-8);
    }

    #[test]
    fn test_adagrad_default() {
        let optimizer = Adagrad::<f32>::default();
        assert_eq!(optimizer.learning_rate, 0.01);
    }

    #[test]
    fn test_adagrad_step() {
        let mut optimizer = Adagrad::<f32>::new(0.1);
        let mut model = MockModel::<f32>::new();

        // Get initial parameter value
        let initial_param = model.param.clone();

        // Perform optimization step
        optimizer.step(&mut model).unwrap();

        // Parameter should have changed
        assert_ne!(model.param.as_slice(), initial_param.as_slice());

        // Second step should use accumulated gradients
        model.param.set_grad(Some(
            Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).unwrap(),
        ));

        let after_first_step = model.param.clone();
        optimizer.step(&mut model).unwrap();

        // Parameter should change again, but with smaller updates due to accumulated gradients
        assert_ne!(model.param.as_slice(), after_first_step.as_slice());
    }

    #[test]
    fn test_adagrad_zero_grad() {
        let optimizer = Adagrad::<f32>::new(0.01);
        let mut model = MockModel::<f32>::new();

        // Verify gradient exists
        assert!(model.param.grad().is_some());

        // Zero gradients
        optimizer.zero_grad(&mut model);

        // Verify gradient is None
        assert!(model.param.grad().is_none());
    }
}
