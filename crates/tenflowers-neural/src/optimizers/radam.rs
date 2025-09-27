use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// RAdam (Rectified Adam) optimizer
///
/// RAdam is a variant of Adam that provides an automated, dynamic adjustment
/// to the adaptive learning rate based on the variance. It addresses the
/// problem of bad convergence due to the large variance of the adaptive learning
/// rate in the early stage of training.
///
/// Reference: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu,
/// Jianfeng Gao, Jiawei Han. "On the Variance of the Adaptive Learning Rate and Beyond"
pub struct RAdam<T> {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,                                // timestep
    m: HashMap<*const Tensor<T>, Tensor<T>>, // first moment estimates
    v: HashMap<*const Tensor<T>, Tensor<T>>, // second moment estimates
}

impl<T> RAdam<T> {
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

impl<T> Default for RAdam<T> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<T> Optimizer<T> for RAdam<T>
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
        let t = self.t as f32;

        // Convert constants to T
        let beta1_t = T::from(self.beta1).unwrap();
        let beta2_t = T::from(self.beta2).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let eps_t = T::from(self.epsilon).unwrap();

        // Compute rho_inf (infinite horizon length)
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;

        // Compute rho_t (current step horizon length)
        let rho_t = rho_inf - 2.0 * t * self.beta2.powf(t) / (1.0 - self.beta2.powf(t));

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
                let beta1_m = m.mul(&Tensor::from_scalar(beta1_t))?;
                let one_minus_beta1_g = grad.mul(&one_minus_beta1)?;
                *m = beta1_m.add(&one_minus_beta1_g)?;

                // Update biased second moment estimate
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let one_minus_beta2 = Tensor::from_scalar(T::one() - beta2_t);
                let beta2_v = v.mul(&Tensor::from_scalar(beta2_t))?;
                let grad_squared = grad.pow(&Tensor::from_scalar(T::from(2.0).unwrap()))?;
                let one_minus_beta2_g2 = grad_squared.mul(&one_minus_beta2)?;
                *v = beta2_v.add(&one_minus_beta2_g2)?;

                // Bias correction for first moment
                let bias_correction1 = T::from(1.0 - self.beta1.powf(t)).unwrap();
                let m_hat = m.div(&Tensor::from_scalar(bias_correction1))?;

                // Check if we should use adaptive or SGD-like update
                if rho_t > 4.0 {
                    // Use adaptive learning rate

                    // Bias correction for second moment
                    let bias_correction2 = T::from(1.0 - self.beta2.powf(t)).unwrap();
                    let v_hat = v.div(&Tensor::from_scalar(bias_correction2))?;

                    // Compute variance rectification term
                    let r_t = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                        .sqrt();

                    let r_t_tensor = Tensor::from_scalar(T::from(r_t).unwrap());

                    // Compute adaptive update
                    let sqrt_v_hat = v_hat.sqrt()?;
                    let sqrt_v_hat_plus_eps = sqrt_v_hat.add(&Tensor::from_scalar(eps_t))?;
                    let adaptive_update = m_hat.div(&sqrt_v_hat_plus_eps)?;
                    let rectified_update = adaptive_update.mul(&r_t_tensor)?;

                    // Apply update
                    let lr_tensor = Tensor::from_scalar(lr_t);
                    let update = rectified_update.mul(&lr_tensor)?;
                    *param = param.sub(&update)?;
                } else {
                    // Use SGD-like update (momentum only)
                    let lr_tensor = Tensor::from_scalar(lr_t);
                    let update = m_hat.mul(&lr_tensor)?;
                    *param = param.sub(&update)?;
                }
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
    fn test_radam_creation() {
        let optimizer = RAdam::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);

        let optimizer = RAdam::<f32>::with_betas(0.001, 0.9, 0.999);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);

        let optimizer = RAdam::<f32>::default();
        assert_eq!(optimizer.get_learning_rate(), 0.001);
    }

    #[test]
    fn test_radam_learning_rate() {
        let mut optimizer = RAdam::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);

        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.get_learning_rate(), 0.01);
    }
}
