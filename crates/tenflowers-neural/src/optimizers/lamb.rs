use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer
///
/// LAMB is a layerwise adaptive large batch optimization technique.
/// It changes the effective stepsize on a per-layer basis depending on the
/// ratio of the norm of the layer weights to the norm of the gradients.
///
/// Reference: Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar,
/// Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh.
/// "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
pub struct LAMB<T> {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: usize,                                // timestep
    m: HashMap<*const Tensor<T>, Tensor<T>>, // first moment estimates
    v: HashMap<*const Tensor<T>, Tensor<T>>, // second moment estimates
}

impl<T> LAMB<T> {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.01,
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
            epsilon: 1e-6,
            weight_decay: 0.01,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
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

    // Getter methods for hyperparameters
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    // Setter methods for hyperparameters
    pub fn set_beta1(&mut self, beta1: f32) {
        self.beta1 = beta1;
    }

    pub fn set_beta2(&mut self, beta2: f32) {
        self.beta2 = beta2;
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }
}

impl<T> Default for LAMB<T> {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<T> Optimizer<T> for LAMB<T>
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

        // Bias correction terms
        let bias_correction1 = T::from(1.0 - self.beta1.powi(t as i32)).unwrap();
        let bias_correction2 = T::from(1.0 - self.beta2.powi(t as i32)).unwrap();

        // Convert constants to T
        let beta1_t = T::from(self.beta1).unwrap();
        let beta2_t = T::from(self.beta2).unwrap();
        let lr_t = T::from(self.learning_rate).unwrap();
        let eps_t = T::from(self.epsilon).unwrap();
        let weight_decay_t = T::from(self.weight_decay).unwrap();

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

                // Add weight decay to gradient
                let grad_with_decay =
                    grad.add(&param.mul(&Tensor::from_scalar(weight_decay_t))?)?;

                // Update biased first moment estimate
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                let one_minus_beta1 = Tensor::from_scalar(T::one() - beta1_t);
                let beta1_m = m.mul(&Tensor::from_scalar(beta1_t))?;
                let one_minus_beta1_g = grad_with_decay.mul(&one_minus_beta1)?;
                *m = beta1_m.add(&one_minus_beta1_g)?;

                // Update biased second moment estimate
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let one_minus_beta2 = Tensor::from_scalar(T::one() - beta2_t);
                let beta2_v = v.mul(&Tensor::from_scalar(beta2_t))?;
                let grad_squared =
                    grad_with_decay.pow(&Tensor::from_scalar(T::from(2.0).unwrap()))?;
                let one_minus_beta2_g2 = grad_squared.mul(&one_minus_beta2)?;
                *v = beta2_v.add(&one_minus_beta2_g2)?;

                // Compute bias-corrected first moment estimate
                let m_hat = m.div(&Tensor::from_scalar(bias_correction1))?;

                // Compute bias-corrected second moment estimate
                let v_hat = v.div(&Tensor::from_scalar(bias_correction2))?;

                // Compute update direction
                let sqrt_v_hat = v_hat.sqrt()?;
                let sqrt_v_hat_plus_eps = sqrt_v_hat.add(&Tensor::from_scalar(eps_t))?;
                let r = m_hat.div(&sqrt_v_hat_plus_eps)?;

                // Compute layer-wise adaptation
                // L2 norm = sqrt(sum(x^2))
                let param_squared = param.pow(&Tensor::from_scalar(T::from(2.0).unwrap()))?;
                let param_norm_squared = param_squared.sum(None, false)?;
                let param_norm = param_norm_squared.sqrt()?;

                let r_squared = r.pow(&Tensor::from_scalar(T::from(2.0).unwrap()))?;
                let r_norm_squared = r_squared.sum(None, false)?;
                let r_norm = r_norm_squared.sqrt()?;

                // Compute trust ratio
                // Extract scalar values from 0-dimensional tensors
                let param_norm_scalar = param_norm.as_slice().unwrap()[0];
                let r_norm_scalar = r_norm.as_slice().unwrap()[0];

                let trust_ratio = if r_norm_scalar > T::zero() {
                    param_norm_scalar / r_norm_scalar
                } else {
                    T::one()
                };

                // Apply update with trust ratio
                let trust_ratio_tensor = Tensor::from_scalar(trust_ratio);
                let lr_tensor = Tensor::from_scalar(lr_t);
                let update = r.mul(&trust_ratio_tensor)?.mul(&lr_tensor)?;

                // Update parameter
                *param = param.sub(&update)?;
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
    fn test_lamb_creation() {
        let optimizer = LAMB::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
        assert_eq!(optimizer.epsilon(), 1e-6);
        assert_eq!(optimizer.weight_decay(), 0.01);

        let optimizer = LAMB::<f32>::with_betas(0.001, 0.9, 0.999);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);

        let optimizer = LAMB::<f32>::default();
        assert_eq!(optimizer.get_learning_rate(), 0.001);
    }

    #[test]
    fn test_lamb_learning_rate() {
        let mut optimizer = LAMB::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);

        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_lamb_hyperparameter_setters() {
        let mut optimizer = LAMB::<f32>::new(0.001);

        // Test beta1 setter
        optimizer.set_beta1(0.95);
        assert_eq!(optimizer.beta1(), 0.95);

        // Test beta2 setter
        optimizer.set_beta2(0.9999);
        assert_eq!(optimizer.beta2(), 0.9999);

        // Test epsilon setter
        optimizer.set_epsilon(1e-7);
        assert_eq!(optimizer.epsilon(), 1e-7);

        // Test weight_decay setter
        optimizer.set_weight_decay(0.02);
        assert_eq!(optimizer.weight_decay(), 0.02);
    }
}
