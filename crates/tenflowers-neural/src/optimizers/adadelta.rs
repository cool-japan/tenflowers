use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Adadelta optimizer
///
/// Adadelta is an extension of Adagrad that seeks to reduce its aggressive,
/// monotonically decreasing learning rate. Instead of accumulating all past
/// squared gradients, Adadelta restricts the window of accumulated past gradients
/// to some fixed size.
///
/// Reference: Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method.
pub struct Adadelta<T> {
    learning_rate: f32,
    rho: f32,
    epsilon: f32,
    t: usize,                                  // timestep
    eg2: HashMap<*const Tensor<T>, Tensor<T>>, // accumulated squared gradients
    ed2: HashMap<*const Tensor<T>, Tensor<T>>, // accumulated squared updates
}

impl<T> Adadelta<T> {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            rho: 0.95,
            epsilon: 1e-6,
            t: 0,
            eg2: HashMap::new(),
            ed2: HashMap::new(),
        }
    }

    pub fn with_rho(learning_rate: f32, rho: f32) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon: 1e-6,
            t: 0,
            eg2: HashMap::new(),
            ed2: HashMap::new(),
        }
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

impl<T> Default for Adadelta<T> {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<T> Optimizer<T> for Adadelta<T>
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

        // Convert constants to T
        let rho_t = T::from(self.rho).unwrap();
        let one_minus_rho_t = T::from(1.0 - self.rho).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let eps_t = T::from(self.epsilon).unwrap();

        // Update each parameter
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                let param_ptr = param as *const Tensor<T>;

                // Initialize or get accumulated squared gradients
                let eg2 = self
                    .eg2
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Initialize or get accumulated squared updates
                let ed2 = self
                    .ed2
                    .entry(param_ptr)
                    .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                // Update running average of squared gradients
                // E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g_t²
                let grad_squared = grad.pow(&Tensor::from_scalar(T::from(2.0).unwrap()))?;
                let rho_eg2 = eg2.mul(&Tensor::from_scalar(rho_t))?;
                let one_minus_rho_grad_sq =
                    grad_squared.mul(&Tensor::from_scalar(one_minus_rho_t))?;
                *eg2 = rho_eg2.add(&one_minus_rho_grad_sq)?;

                // Compute RMS of accumulated squared updates
                let rms_ed2 = ed2.add(&Tensor::from_scalar(eps_t))?.sqrt()?;

                // Compute RMS of accumulated squared gradients
                let rms_eg2 = eg2.add(&Tensor::from_scalar(eps_t))?.sqrt()?;

                // Compute update
                // Δp_t = -η * (RMS[Δp]_{t-1} / RMS[g]_t) * g_t
                let update = grad
                    .mul(&rms_ed2)?
                    .div(&rms_eg2)?
                    .mul(&Tensor::from_scalar(-lr_t))?;

                // Update running average of squared updates
                // E[Δp²]_t = ρ * E[Δp²]_{t-1} + (1 - ρ) * Δp_t²
                let update_squared = update.pow(&Tensor::from_scalar(T::from(2.0).unwrap()))?;
                let rho_ed2 = ed2.mul(&Tensor::from_scalar(rho_t))?;
                let one_minus_rho_update_sq =
                    update_squared.mul(&Tensor::from_scalar(one_minus_rho_t))?;
                *ed2 = rho_ed2.add(&one_minus_rho_update_sq)?;

                // Apply update
                *param = param.add(&update)?;
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
    use crate::layers::dense::Dense;
    use crate::model::sequential::Sequential;

    #[test]
    fn test_adadelta_creation() {
        let optimizer = Adadelta::<f32>::new(1.0);
        assert_eq!(optimizer.get_learning_rate(), 1.0);

        let optimizer = Adadelta::<f32>::with_rho(1.0, 0.9);
        assert_eq!(optimizer.rho, 0.9);

        let optimizer = Adadelta::<f32>::default();
        assert_eq!(optimizer.get_learning_rate(), 1.0);
    }

    #[test]
    fn test_adadelta_learning_rate() {
        let mut optimizer = Adadelta::<f32>::new(0.01);
        assert_eq!(optimizer.get_learning_rate(), 0.01);

        optimizer.set_learning_rate(0.1);
        assert_eq!(optimizer.get_learning_rate(), 0.1);
    }
}
