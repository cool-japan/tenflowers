use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

#[derive(Debug)]
pub struct SGD<T> {
    learning_rate: f32,
    momentum: Option<f32>,
    weight_decay: f32,
    velocity: HashMap<*const Tensor<T>, Tensor<T>>,
}

impl<T> SGD<T> {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: None,
            weight_decay: 0.0,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        // Get all parameters that need to be updated
        let params = model.parameters_mut();

        for param in params {
            // Check if the parameter has a gradient
            if let Some(grad) = param.grad() {
                // Clone the gradient to avoid borrow issues
                let grad = grad.clone();

                // Apply decoupled weight decay if specified
                if self.weight_decay > 0.0 {
                    let lr_t = T::from(self.learning_rate).unwrap();
                    let wd_t = T::from(self.weight_decay).unwrap();
                    let weight_decay_factor = T::one() - lr_t * wd_t;
                    let param_decayed = param.mul(&Tensor::from_scalar(weight_decay_factor))?;
                    *param = param_decayed;
                }

                // Apply SGD update rule
                if let Some(momentum) = self.momentum {
                    // SGD with momentum
                    let param_ptr = param as *const Tensor<T>;

                    // Initialize or get velocity
                    let velocity = self
                        .velocity
                        .entry(param_ptr)
                        .or_insert_with(|| Tensor::zeros(param.shape().dims()));

                    // Update velocity: v = momentum * v - lr * grad
                    let momentum_t = T::from(momentum).unwrap();
                    let lr_t = T::from(self.learning_rate).unwrap();

                    // v = momentum * v - lr * grad
                    let momentum_term = velocity.mul(&Tensor::from_scalar(momentum_t))?;
                    let lr_grad = grad.mul(&Tensor::from_scalar(lr_t))?;
                    *velocity = momentum_term.sub(&lr_grad)?;

                    // param = param + v
                    let new_param = param.add(velocity)?;
                    // Copy the values back
                    *param = new_param;
                } else {
                    // Standard SGD: param = param - lr * grad
                    let lr_t = T::from(self.learning_rate).unwrap();
                    let lr_grad = grad.mul(&Tensor::from_scalar(lr_t))?;
                    let new_param = param.sub(&lr_grad)?;
                    *param = new_param;
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
