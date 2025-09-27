use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Nadam optimizer - Adam with Nesterov momentum
///
/// Nadam combines Adam's adaptive learning rates with Nesterov's accelerated gradient,
/// which can provide better convergence properties, especially for problems with
/// strong curvature.
///
/// The algorithm maintains exponential moving averages of both the gradient (first moment)
/// and the squared gradient (second moment), like Adam, but incorporates Nesterov momentum
/// in the parameter update step.
pub struct Nadam<T> {
    learning_rate: f32,
    beta1: f32,   // Exponential decay rate for first moment estimates
    beta2: f32,   // Exponential decay rate for second moment estimates
    epsilon: f32, // Small constant for numerical stability
    t: usize,     // Timestep counter
    m: HashMap<*const Tensor<T>, Tensor<T>>, // First moment estimates
    v: HashMap<*const Tensor<T>, Tensor<T>>, // Second moment estimates
}

impl<T> Nadam<T> {
    /// Create a new Nadam optimizer with the specified learning rate
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Create a Nadam optimizer with custom beta parameters
    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
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

impl<T> Default for Nadam<T> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<T> Optimizer<T> for Nadam<T>
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        // Increment timestep
        self.t += 1;
        let t = self.t;

        // Convert constants to T
        let beta1_t = T::from(self.beta1).unwrap();
        let beta2_t = T::from(self.beta2).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let eps_t = T::from(self.epsilon).unwrap();

        // Bias correction terms
        let bias_correction1 = T::from(1.0 - self.beta1.powi(t as i32)).unwrap();
        let bias_correction2 = T::from(1.0 - self.beta2.powi(t as i32)).unwrap();

        // Update each parameter
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                let param_ptr = param as *const Tensor<T>;

                // Initialize or get first moment estimate
                let m = self
                    .m
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Initialize or get second moment estimate
                let v = self
                    .v
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Update biased first moment estimate
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                let one_minus_beta1 = Tensor::from_scalar(T::one() - beta1_t);
                let beta1_tensor = Tensor::from_scalar(beta1_t);
                let m_scaled = m.mul(&beta1_tensor)?;
                let grad_scaled = grad.mul(&one_minus_beta1)?;
                *m = m_scaled.add(&grad_scaled)?;

                // Update biased second raw moment estimate
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let one_minus_beta2 = Tensor::from_scalar(T::one() - beta2_t);
                let beta2_tensor = Tensor::from_scalar(beta2_t);
                let v_scaled = v.mul(&beta2_tensor)?;
                let grad_squared = grad.mul(grad)?;
                let grad_squared_scaled = grad_squared.mul(&one_minus_beta2)?;
                *v = v_scaled.add(&grad_squared_scaled)?;

                // Compute bias-corrected second raw moment estimate
                let v_hat = v.div(&Tensor::from_scalar(bias_correction2))?;

                // Compute the Nesterov-style update
                // This is the key difference from Adam:
                // Instead of using m̂_t directly, we use β₁ * m̂_t + (1 - β₁) * g_t / (1 - β₁^t)
                let m_hat = m.div(&Tensor::from_scalar(bias_correction1))?;
                let grad_normalized = grad.div(&Tensor::from_scalar(bias_correction1))?;

                let nesterov_term1 = beta1_tensor.mul(&m_hat)?;
                let nesterov_term2 = one_minus_beta1.mul(&grad_normalized)?;
                let nesterov_momentum = nesterov_term1.add(&nesterov_term2)?;

                // Update parameters
                // param = param - lr * nesterov_momentum / (sqrt(v_hat) + epsilon)
                let v_hat_sqrt = v_hat.sqrt()?;
                let v_hat_sqrt_eps = v_hat_sqrt.add(&Tensor::from_scalar(eps_t))?;
                let update = nesterov_momentum.div(&v_hat_sqrt_eps)?;
                let lr_update = update.mul(&Tensor::from_scalar(lr_t))?;
                let new_param = param.sub(&lr_update)?;

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
    fn test_nadam_creation() {
        let optimizer = Nadam::<f32>::new(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.epsilon, 1e-8);
    }

    #[test]
    fn test_nadam_with_betas() {
        let optimizer = Nadam::<f32>::with_betas(0.001, 0.95, 0.9999);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.beta2, 0.9999);
    }

    #[test]
    fn test_nadam_with_epsilon() {
        let optimizer = Nadam::<f32>::new(0.001).with_epsilon(1e-10);
        assert_eq!(optimizer.epsilon, 1e-10);
    }

    #[test]
    fn test_nadam_default() {
        let optimizer = Nadam::<f32>::default();
        assert_eq!(optimizer.learning_rate, 0.001);
    }

    #[test]
    fn test_nadam_step() {
        let mut optimizer = Nadam::<f32>::new(0.01);
        let mut model = MockModel::<f32>::new();

        // Get initial parameter value
        let initial_param = model.param.clone();

        // Perform optimization step
        optimizer.step(&mut model).unwrap();

        // Parameter should have changed
        assert_ne!(model.param.as_slice(), initial_param.as_slice());

        // Verify timestep was incremented
        assert_eq!(optimizer.t, 1);

        // Second step should use momentum
        model.param.set_grad(Some(
            Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).unwrap(),
        ));

        let after_first_step = model.param.clone();
        optimizer.step(&mut model).unwrap();

        // Parameter should change again
        assert_ne!(model.param.as_slice(), after_first_step.as_slice());
        assert_eq!(optimizer.t, 2);
    }

    #[test]
    fn test_nadam_zero_grad() {
        let optimizer = Nadam::<f32>::new(0.001);
        let mut model = MockModel::<f32>::new();

        // Verify gradient exists
        assert!(model.param.grad().is_some());

        // Zero gradients
        optimizer.zero_grad(&mut model);

        // Verify gradient is None
        assert!(model.param.grad().is_none());
    }

    #[test]
    fn test_nadam_multiple_steps() {
        let mut optimizer = Nadam::<f32>::new(0.01);
        let mut model = MockModel::<f32>::new();

        // Perform multiple steps to test momentum accumulation
        for i in 1..=5 {
            model.param.set_grad(Some(
                Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).unwrap(),
            ));

            optimizer.step(&mut model).unwrap();
            assert_eq!(optimizer.t, i);
        }

        // After 5 steps, the parameters should have moved significantly from initial values
        let final_param = model.param.clone();
        let initial_param = Tensor::<f32>::ones(&[2, 2]);

        // Values should be different from initial
        if let (Some(final_data), Some(initial_data)) =
            (final_param.as_slice(), initial_param.as_slice())
        {
            for (f, i) in final_data.iter().zip(initial_data.iter()) {
                assert_ne!(f, i);
            }
        }
    }
}
