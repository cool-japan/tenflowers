use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// AdamW optimizer with decoupled weight decay
pub struct AdamW<T> {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub epsilon: f32,
    t: usize,                                // timestep
    m: HashMap<*const Tensor<T>, Tensor<T>>, // first moment estimates
    v: HashMap<*const Tensor<T>, Tensor<T>>, // second moment estimates
}

impl<T> AdamW<T> {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_betas(learning_rate: f32, weight_decay: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            weight_decay,
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

impl<T> Default for AdamW<T> {
    fn default() -> Self {
        Self::new(0.001, 0.01)
    }
}

impl<T> Optimizer<T> for AdamW<T>
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
        let wd_t = T::from(self.weight_decay).unwrap();

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

                // Compute bias-corrected first moment estimate
                let m_hat = m.div(&Tensor::from_scalar(bias_correction1))?;

                // Compute bias-corrected second raw moment estimate
                let v_hat = v.div(&Tensor::from_scalar(bias_correction2))?;

                // AdamW weight decay: param = param * (1 - lr * weight_decay)
                let weight_decay_factor = T::one() - lr_t * wd_t;
                let param_decayed = param.mul(&Tensor::from_scalar(weight_decay_factor))?;

                // Update parameters: param = param_decayed - lr * m_hat / (sqrt(v_hat) + epsilon)
                let v_hat_sqrt = v_hat.sqrt()?;
                let v_hat_sqrt_eps = v_hat_sqrt.add(&Tensor::from_scalar(eps_t))?;
                let update = m_hat.div(&v_hat_sqrt_eps)?;
                let lr_update = update.mul(&Tensor::from_scalar(lr_t))?;
                let new_param = param_decayed.sub(&lr_update)?;

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
