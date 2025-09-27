//! Adaptive activation functions that learn their parameters during training

use crate::layers::Layer;
use num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Adaptive Swish activation function
///
/// AdaptiveSwish(x) = x * sigmoid(β * x)
/// where `β` is a learnable parameter that controls the shape
pub struct AdaptiveSwish<T> {
    /// The learnable parameter `β` that controls the shape
    pub beta: Tensor<T>,
}

impl<T> AdaptiveSwish<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new AdaptiveSwish layer
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

        Ok(AdaptiveSwish { beta })
    }

    /// Create a new AdaptiveSwish layer with default initialization (β = 1.0)
    pub fn new_default(num_parameters: usize) -> Result<Self> {
        let init_value = T::one();
        Self::new(num_parameters, init_value)
    }
}

impl<T> Clone for AdaptiveSwish<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            beta: self.beta.clone(),
        }
    }
}

impl<T> Layer<T> for AdaptiveSwish<T>
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
        + std::ops::Neg<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // AdaptiveSwish(x) = x * sigmoid(β * x)
        // where sigmoid(x) = 1 / (1 + exp(-x))

        // Compute β * x
        let beta_x = self.beta.mul(input)?;

        // Compute sigmoid(β * x) = 1 / (1 + exp(-β * x))
        let neg_beta_x = beta_x.neg()?;
        let exp_neg_beta_x = neg_beta_x.exp()?;
        let one = Tensor::from_scalar(T::one());
        let one_plus_exp = one.add(&exp_neg_beta_x)?;
        let sigmoid = one.div(&one_plus_exp)?;

        // Return x * sigmoid(β * x)
        input.mul(&sigmoid)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.beta]
    }

    fn set_training(&mut self, _training: bool) {
        // AdaptiveSwish doesn't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Adaptive Piecewise Linear activation function
///
/// Learns a piecewise linear function with configurable breakpoints and slopes.
/// The function has the form: f(x) = max(a_i * x + b_i) for different segments i
pub struct AdaptivePiecewiseLinear<T> {
    /// Slopes for each linear segment
    pub slopes: Tensor<T>,
    /// Biases for each linear segment
    pub biases: Tensor<T>,
    /// Number of linear segments
    num_segments: usize,
}

impl<T> AdaptivePiecewiseLinear<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new AdaptivePiecewiseLinear layer
    ///
    /// # Arguments
    /// * `num_segments` - Number of linear segments (typically 2-5)
    /// * `num_channels` - Number of channels (1 for shared across all inputs)
    pub fn new(num_segments: usize, num_channels: usize) -> Result<Self> {
        if num_segments < 2 {
            return Err(tenflowers_core::TensorError::InvalidArgument {
                operation: "AdaptivePiecewiseLinear::new".to_string(),
                reason: "AdaptivePiecewiseLinear needs at least 2 segments".to_string(),
                context: None,
            });
        }

        // Initialize slopes to different values to break symmetry
        let mut slope_values = Vec::new();
        for i in 0..num_segments {
            for _ in 0..num_channels {
                let slope = T::from(0.5 + i as f64 * 0.3).unwrap_or(T::one());
                slope_values.push(slope);
            }
        }
        let slopes = Tensor::from_vec(slope_values, &[num_segments, num_channels])?;

        // Initialize biases to zero
        let bias_values = vec![T::zero(); num_segments * num_channels];
        let biases = Tensor::from_vec(bias_values, &[num_segments, num_channels])?;

        Ok(AdaptivePiecewiseLinear {
            slopes,
            biases,
            num_segments,
        })
    }

    /// Create a simple 2-segment piecewise linear activation (similar to ReLU but learnable)
    pub fn new_simple(num_channels: usize) -> Result<Self> {
        Self::new(2, num_channels)
    }
}

impl<T> Clone for AdaptivePiecewiseLinear<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            slopes: self.slopes.clone(),
            biases: self.biases.clone(),
            num_segments: self.num_segments,
        }
    }
}

impl<T> Layer<T> for AdaptivePiecewiseLinear<T>
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
        // For each input element, compute all linear segments and take the maximum
        // This is a simplified implementation - in practice you'd want more efficient GPU kernels

        let input_shape = input.shape().dims();
        let input_data =
            input
                .as_slice()
                .ok_or_else(|| tenflowers_core::TensorError::DeviceError {
                    operation: "AdaptivePiecewiseLinear::forward".to_string(),
                    details: "Cannot access input tensor data".to_string(),
                    device: input.device().to_string(),
                    context: None,
                })?;

        let slopes_data =
            self.slopes
                .as_slice()
                .ok_or_else(|| tenflowers_core::TensorError::DeviceError {
                    operation: "AdaptivePiecewiseLinear::forward".to_string(),
                    details: "Cannot access slopes tensor data".to_string(),
                    device: self.slopes.device().to_string(),
                    context: None,
                })?;

        let biases_data =
            self.biases
                .as_slice()
                .ok_or_else(|| tenflowers_core::TensorError::DeviceError {
                    operation: "AdaptivePiecewiseLinear::forward".to_string(),
                    details: "Cannot access biases tensor data".to_string(),
                    device: self.biases.device().to_string(),
                    context: None,
                })?;

        let total_elements = input_data.len();
        let mut output_data = Vec::with_capacity(total_elements);

        // For each input element, compute the piecewise linear function
        for (idx, &x) in input_data.iter().enumerate() {
            let channel_idx = if self.slopes.shape().dims()[1] == 1 {
                0
            } else {
                idx % self.slopes.shape().dims()[1]
            };

            let mut max_value = T::from(-f64::INFINITY).unwrap_or_else(|| T::zero());

            // Evaluate all segments and take the maximum
            for segment in 0..self.num_segments {
                let slope_idx = segment * self.slopes.shape().dims()[1] + channel_idx;
                let bias_idx = segment * self.biases.shape().dims()[1] + channel_idx;

                let segment_value = slopes_data[slope_idx] * x + biases_data[bias_idx];
                if segment_value > max_value
                    || max_value == T::from(-f64::INFINITY).unwrap_or_else(|| T::zero())
                {
                    max_value = segment_value;
                }
            }

            output_data.push(max_value);
        }

        Tensor::from_vec(output_data, input_shape)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.slopes, &self.biases]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.slopes, &mut self.biases]
    }

    fn set_training(&mut self, _training: bool) {
        // AdaptivePiecewiseLinear doesn't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Adaptive Polynomial activation function
///
/// Learns a polynomial activation of the form: f(x) = sum(a_i * x^i) for i from 0 to degree
pub struct AdaptivePolynomial<T> {
    /// Polynomial coefficients [a_0, a_1, a_2, ..., a_degree]
    pub coefficients: Tensor<T>,
    /// Degree of the polynomial
    degree: usize,
}

impl<T> AdaptivePolynomial<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new AdaptivePolynomial layer
    ///
    /// # Arguments
    /// * `degree` - Degree of the polynomial (typically 2-4)
    /// * `num_channels` - Number of channels (1 for shared across all inputs)
    pub fn new(degree: usize, num_channels: usize) -> Result<Self> {
        if degree == 0 {
            return Err(tenflowers_core::TensorError::InvalidArgument {
                operation: "AdaptivePolynomial::new".to_string(),
                reason: "AdaptivePolynomial degree must be > 0".to_string(),
                context: None,
            });
        }

        // Initialize coefficients: a_0=0, a_1=1, others=small random values
        let mut coeff_values = Vec::new();
        for channel in 0..num_channels {
            for deg in 0..=degree {
                let coeff = if deg == 0 {
                    T::zero() // bias term
                } else if deg == 1 {
                    T::one() // linear term (like identity)
                } else {
                    // Higher order terms start small
                    T::from(0.1 / (deg as f64)).unwrap_or(T::zero())
                };
                coeff_values.push(coeff);
            }
        }

        let coefficients = Tensor::from_vec(coeff_values, &[num_channels, degree + 1])?;

        Ok(AdaptivePolynomial {
            coefficients,
            degree,
        })
    }

    /// Create a quadratic adaptive activation (degree 2)
    pub fn new_quadratic(num_channels: usize) -> Result<Self> {
        Self::new(2, num_channels)
    }

    /// Create a cubic adaptive activation (degree 3)
    pub fn new_cubic(num_channels: usize) -> Result<Self> {
        Self::new(3, num_channels)
    }
}

impl<T> Clone for AdaptivePolynomial<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            coefficients: self.coefficients.clone(),
            degree: self.degree,
        }
    }
}

impl<T> Layer<T> for AdaptivePolynomial<T>
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
        // Compute polynomial: f(x) = sum(a_i * x^i)

        let input_shape = input.shape().dims();
        let input_data =
            input
                .as_slice()
                .ok_or_else(|| tenflowers_core::TensorError::DeviceError {
                    operation: "AdaptivePolynomial::forward".to_string(),
                    details: "Cannot access input tensor data".to_string(),
                    device: input.device().to_string(),
                    context: None,
                })?;

        let coeff_data = self.coefficients.as_slice().ok_or_else(|| {
            tenflowers_core::TensorError::DeviceError {
                operation: "AdaptivePolynomial::forward".to_string(),
                details: "Cannot access coefficients tensor data".to_string(),
                device: self.coefficients.device().to_string(),
                context: None,
            }
        })?;

        let total_elements = input_data.len();
        let num_channels = self.coefficients.shape().dims()[0];
        let mut output_data = Vec::with_capacity(total_elements);

        for (idx, &x) in input_data.iter().enumerate() {
            let channel_idx = if num_channels == 1 {
                0
            } else {
                idx % num_channels
            };

            let mut result = T::zero();
            let mut x_power = T::one(); // x^0 = 1

            // Compute polynomial evaluation using Horner's method for stability
            for deg in 0..=self.degree {
                let coeff_idx = channel_idx * (self.degree + 1) + deg;
                result = result + coeff_data[coeff_idx] * x_power;

                if deg < self.degree {
                    x_power = x_power * x; // Update for next iteration
                }
            }

            output_data.push(result);
        }

        Tensor::from_vec(output_data, input_shape)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.coefficients]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.coefficients]
    }

    fn set_training(&mut self, _training: bool) {
        // AdaptivePolynomial doesn't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}
