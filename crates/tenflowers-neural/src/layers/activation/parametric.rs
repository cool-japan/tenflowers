//! Parametric activation functions with learnable parameters

use crate::layers::Layer;
use num_traits::{Float, One, Zero};
use tenflowers_core::{ops::manipulation::where_op, Result, Tensor};

/// Parametric ReLU activation function: PReLU(x) = max(0, x) + a * min(0, x)
/// where `a` is a learnable parameter
pub struct PReLU<T> {
    /// The learnable parameter `a` for negative values
    pub alpha: Tensor<T>,
}

impl<T> PReLU<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new PReLU layer
    ///
    /// # Arguments
    /// * `num_parameters` - Number of parameters (typically number of channels)
    /// * `init_value` - Initial value for the alpha parameter (default: 0.25)
    pub fn new(num_parameters: usize, init_value: T) -> Result<Self> {
        let alpha = if num_parameters == 1 {
            Tensor::from_scalar(init_value)
        } else {
            let init_vec = vec![init_value; num_parameters];
            Tensor::from_vec(init_vec, &[num_parameters])?
        };

        Ok(PReLU { alpha })
    }

    /// Create a new PReLU layer with default initialization (0.25)
    pub fn new_default(num_parameters: usize) -> Result<Self> {
        let init_value = T::from(0.25).unwrap();
        Self::new(num_parameters, init_value)
    }
}

impl<T> Layer<T> for PReLU<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + std::cmp::PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let zero = Tensor::from_scalar(T::zero());

        // Positive part: max(0, x)
        let positive_part = where_op(&input.ge(&zero)?, input, &zero)?;

        // Negative part: a * min(0, x)
        let negative_part = where_op(&input.le(&zero)?, input, &zero)?;
        let scaled_negative = self.alpha.mul(&negative_part)?;

        positive_part.add(&scaled_negative)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.alpha]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.alpha]
    }

    fn set_training(&mut self, _training: bool) {
        // PReLU doesn't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

impl<T> Clone for PReLU<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        PReLU {
            alpha: self.alpha.clone(),
        }
    }
}

/// Parametric Softplus activation function: ParametricSoftplus(x) = (1/β) * log(1 + exp(β * x))
/// where `β` is a learnable parameter that controls the sharpness of the activation
pub struct ParametricSoftplus<T> {
    /// The learnable parameter `β` that controls the sharpness
    pub beta: Tensor<T>,
}

impl<T> ParametricSoftplus<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new ParametricSoftplus layer
    ///
    /// # Arguments
    /// * `num_parameters` - Number of parameters (typically number of channels or 1 for shared)
    /// * `init_value` - Initial value for the beta parameter (default: 1.0)
    pub fn new(num_parameters: usize, init_value: T) -> Result<Self> {
        let beta = if num_parameters == 1 {
            Tensor::from_scalar(init_value)
        } else {
            let init_vec = vec![init_value; num_parameters];
            Tensor::from_vec(init_vec, &[num_parameters])?
        };

        Ok(ParametricSoftplus { beta })
    }

    /// Create a new ParametricSoftplus layer with default initialization (β = 1.0)
    pub fn new_default(num_parameters: usize) -> Result<Self> {
        let init_value = T::one();
        Self::new(num_parameters, init_value)
    }
}

impl<T> Layer<T> for ParametricSoftplus<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + std::cmp::PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Sub<Output = T>
        + num_traits::FromPrimitive
        + num_traits::Signed
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // ParametricSoftplus(x) = (1/β) * log(1 + exp(β * x))
        // For numerical stability, we use the identity:
        // log(1 + exp(x)) = max(0, x) + log(1 + exp(-|x|))

        // Compute β * x
        let beta_x = self.beta.mul(input)?;

        // For numerical stability, use log-sum-exp trick
        let zero = Tensor::from_scalar(T::zero());
        let abs_beta_x = beta_x.abs()?;
        let neg_abs_beta_x = zero.sub(&abs_beta_x)?;

        // Compute exp(-|β * x|)
        let exp_neg_abs = neg_abs_beta_x.exp()?;

        // Compute 1 + exp(-|β * x|)
        let one = Tensor::from_scalar(T::one());
        let one_plus_exp_neg_abs = one.add(&exp_neg_abs)?;

        // Compute log(1 + exp(-|β * x|))
        let log_one_plus_exp_neg_abs = one_plus_exp_neg_abs.log()?;

        // Compute max(0, β * x)
        let max_zero_beta_x = tenflowers_core::ops::relu(&beta_x)?;

        // Combine: max(0, β * x) + log(1 + exp(-|β * x|))
        let stable_log_one_plus_exp = max_zero_beta_x.add(&log_one_plus_exp_neg_abs)?;

        // Compute (1/β) * result
        let one_tensor = Tensor::from_scalar(T::one());
        let one_over_beta = one_tensor.div(&self.beta)?;
        one_over_beta.mul(&stable_log_one_plus_exp)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.beta]
    }

    fn set_training(&mut self, _training: bool) {
        // ParametricSoftplus doesn't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

impl<T> Clone for ParametricSoftplus<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        ParametricSoftplus {
            beta: self.beta.clone(),
        }
    }
}
