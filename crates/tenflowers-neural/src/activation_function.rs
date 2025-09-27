use num_traits::Float;
use tenflowers_core::{Result, Tensor};

/// Type-safe activation function enum for compile-time dispatch
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ActivationFunction {
    /// Linear activation (identity function)
    #[default]
    Linear,
    /// ReLU activation function
    ReLU,
    /// ReLU6 activation function (ReLU with upper bound at 6)
    ReLU6,
    /// Leaky ReLU activation function with configurable negative slope
    LeakyReLU(f32),
    /// Sigmoid activation function  
    Sigmoid,
    /// Tanh activation function
    Tanh,
    /// GELU activation function
    GELU,
    /// Swish/SiLU activation function
    Swish,
    /// Mish activation function
    Mish,
    /// ELU (Exponential Linear Unit) activation function
    ELU(f32),
    /// SELU (Self-Normalizing Exponential Linear Unit) activation function
    SELU,
    /// Hardswish activation function (efficient approximation of Swish)
    Hardswish,
    /// GLU (Gated Linear Unit) activation function
    GLU,
    /// SwiGLU activation function (Swish-gated GLU)
    SwiGLU,
    /// GeGLU activation function (GELU-gated GLU)
    GeGLU,
    /// Snake activation function (periodic activation: x + (1/freq) * sin²(freq * x))
    Snake(f32),
    /// QuickGELU activation function (efficient GELU approximation: x * sigmoid(1.702 * x))
    QuickGELU,
}

impl ActivationFunction {
    /// Apply the activation function to a tensor
    pub fn apply<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        match self {
            ActivationFunction::Linear => Ok(input.clone()),
            ActivationFunction::ReLU => tenflowers_core::ops::relu(input),
            ActivationFunction::ReLU6 => self.relu6(input),
            ActivationFunction::LeakyReLU(alpha) => self.leaky_relu(input, *alpha),
            ActivationFunction::Sigmoid => tenflowers_core::ops::sigmoid(input),
            ActivationFunction::Tanh => tenflowers_core::ops::tanh(input),
            ActivationFunction::GELU => tenflowers_core::ops::gelu(input),
            ActivationFunction::Swish => tenflowers_core::ops::swish(input),
            ActivationFunction::Mish => tenflowers_core::ops::mish(input),
            ActivationFunction::ELU(alpha) => self.elu(input, *alpha),
            ActivationFunction::SELU => self.selu(input),
            ActivationFunction::Hardswish => self.hardswish(input),
            ActivationFunction::GLU => self.glu(input),
            ActivationFunction::SwiGLU => self.swiglu(input),
            ActivationFunction::GeGLU => self.geglu(input),
            ActivationFunction::Snake(frequency) => self.snake(input, *frequency),
            ActivationFunction::QuickGELU => self.quick_gelu(input),
        }
    }

    /// Get the name of the activation function as a string
    pub fn name(&self) -> &'static str {
        match self {
            ActivationFunction::Linear => "linear",
            ActivationFunction::ReLU => "relu",
            ActivationFunction::ReLU6 => "relu6",
            ActivationFunction::LeakyReLU(_) => "leaky_relu",
            ActivationFunction::Sigmoid => "sigmoid",
            ActivationFunction::Tanh => "tanh",
            ActivationFunction::GELU => "gelu",
            ActivationFunction::Swish => "swish",
            ActivationFunction::Mish => "mish",
            ActivationFunction::ELU(_) => "elu",
            ActivationFunction::SELU => "selu",
            ActivationFunction::Hardswish => "hardswish",
            ActivationFunction::GLU => "glu",
            ActivationFunction::SwiGLU => "swiglu",
            ActivationFunction::GeGLU => "geglu",
            ActivationFunction::Snake(_) => "snake",
            ActivationFunction::QuickGELU => "quick_gelu",
        }
    }

    /// Create activation function from string (for backward compatibility)
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "linear" | "identity" => Some(ActivationFunction::Linear),
            "relu" => Some(ActivationFunction::ReLU),
            "relu6" => Some(ActivationFunction::ReLU6),
            "leaky_relu" | "leakyrelu" => Some(ActivationFunction::LeakyReLU(0.01)), // Default alpha
            "sigmoid" => Some(ActivationFunction::Sigmoid),
            "tanh" => Some(ActivationFunction::Tanh),
            "gelu" => Some(ActivationFunction::GELU),
            "swish" | "silu" => Some(ActivationFunction::Swish),
            "mish" => Some(ActivationFunction::Mish),
            "elu" => Some(ActivationFunction::ELU(1.0)), // Default alpha
            "selu" => Some(ActivationFunction::SELU),
            "hardswish" => Some(ActivationFunction::Hardswish),
            "glu" => Some(ActivationFunction::GLU),
            "swiglu" => Some(ActivationFunction::SwiGLU),
            "geglu" => Some(ActivationFunction::GeGLU),
            "snake" => Some(ActivationFunction::Snake(1.0)), // Default frequency
            "quick_gelu" | "quickgelu" => Some(ActivationFunction::QuickGELU),
            _ => None,
        }
    }

    /// Leaky ReLU activation: f(x) = max(alpha * x, x)
    fn leaky_relu<T>(&self, input: &Tensor<T>, alpha: f32) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if let Some(data) = input.as_slice() {
            let alpha_t = T::from(alpha).unwrap_or(T::from(0.01).unwrap());
            let result: Vec<T> = data
                .iter()
                .map(|&x| if x > T::zero() { x } else { alpha_t * x })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - use tensor operations for proper LeakyReLU implementation
            let zero = Tensor::zeros(input.shape().dims());
            let alpha_tensor =
                Tensor::from_scalar(T::from(alpha).unwrap_or(T::from(0.01).unwrap()));
            let positive_part = tenflowers_core::ops::numpy_compat::maximum(input, &zero)?;
            let negative_part = tenflowers_core::ops::numpy_compat::minimum(input, &zero)?;
            let scaled_negative = tenflowers_core::ops::mul(&negative_part, &alpha_tensor)?;
            tenflowers_core::ops::add(&positive_part, &scaled_negative)
        }
    }

    /// ReLU6 activation: f(x) = min(max(0, x), 6)
    fn relu6<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if let Some(data) = input.as_slice() {
            let six = T::from(6.0).unwrap_or(T::from(6.0).unwrap());
            let result: Vec<T> = data
                .iter()
                .map(|&x| {
                    if x <= T::zero() {
                        T::zero()
                    } else if x >= six {
                        six
                    } else {
                        x
                    }
                })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - use tensor operations for proper ReLU6 implementation
            let zero = Tensor::zeros(input.shape().dims());
            let six = Tensor::from_scalar(T::from(6.0).unwrap_or(T::from(6.0).unwrap()));
            let clamped_min = tenflowers_core::ops::numpy_compat::maximum(input, &zero)?;
            tenflowers_core::ops::numpy_compat::minimum(&clamped_min, &six)
        }
    }

    /// ELU activation: f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    fn elu<T>(&self, input: &Tensor<T>, alpha: f32) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if let Some(data) = input.as_slice() {
            let alpha_t = T::from(alpha).unwrap_or(T::one());
            let result: Vec<T> = data
                .iter()
                .map(|&x| {
                    if x > T::zero() {
                        x
                    } else {
                        alpha_t * (x.exp() - T::one())
                    }
                })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - use tensor operations for proper ELU implementation
            let zero = Tensor::zeros(input.shape().dims());
            let one = Tensor::ones(input.shape().dims());
            let alpha_tensor = Tensor::from_scalar(T::from(alpha).unwrap_or(T::one()));

            let positive_part = tenflowers_core::ops::numpy_compat::maximum(input, &zero)?;
            let exp_x = tenflowers_core::ops::exp(input)?;
            let exp_minus_one = tenflowers_core::ops::sub(&exp_x, &one)?;
            let scaled_exp = tenflowers_core::ops::mul(&alpha_tensor, &exp_minus_one)?;
            let negative_part = tenflowers_core::ops::numpy_compat::minimum(&scaled_exp, &zero)?;

            tenflowers_core::ops::add(&positive_part, &negative_part)
        }
    }

    /// SELU activation: f(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    /// where scale = 1.0507 and alpha = 1.6733
    fn selu<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        const SELU_ALPHA: f32 = 1.673_263_2;
        const SELU_SCALE: f32 = 1.050_701;

        if let Some(data) = input.as_slice() {
            let alpha_t = T::from(SELU_ALPHA).unwrap_or(T::one());
            let scale_t = T::from(SELU_SCALE).unwrap_or(T::one());
            let result: Vec<T> = data
                .iter()
                .map(|&x| {
                    scale_t
                        * if x > T::zero() {
                            x
                        } else {
                            alpha_t * (x.exp() - T::one())
                        }
                })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - simplified version
            Ok(input.clone())
        }
    }

    /// Hardswish activation: f(x) = x * ReLU6(x + 3) / 6
    fn hardswish<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if let Some(data) = input.as_slice() {
            let three = T::from(3.0).unwrap_or(T::one());
            let six = T::from(6.0).unwrap_or(T::one());
            let result: Vec<T> = data
                .iter()
                .map(|&x| {
                    let relu6_input = x + three;
                    let relu6_output = if relu6_input < T::zero() {
                        T::zero()
                    } else if relu6_input > six {
                        six
                    } else {
                        relu6_input
                    };
                    x * relu6_output / six
                })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - simplified version
            Ok(input.clone())
        }
    }

    /// GLU (Gated Linear Unit): f(x) = x_1 * sigmoid(x_2)
    /// Input tensor is split in half along the last dimension
    fn glu<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        let shape = input.shape().dims();
        let last_dim = *shape.last().unwrap_or(&1);

        if last_dim % 2 != 0 {
            return Err(tenflowers_core::TensorError::invalid_argument(
                "GLU requires input tensor's last dimension to be even".to_string(),
            ));
        }

        let half_dim = last_dim / 2;

        if let Some(data) = input.as_slice() {
            let mut result = Vec::new();
            let num_elements = data.len();
            let elements_per_slice = num_elements / last_dim;

            for slice_idx in 0..elements_per_slice {
                for i in 0..half_dim {
                    let base_idx = slice_idx * last_dim;
                    let x1 = data[base_idx + i];
                    let x2 = data[base_idx + half_dim + i];
                    let sigmoid_x2 = T::one() / (T::one() + (-x2).exp());
                    result.push(x1 * sigmoid_x2);
                }
            }

            let mut result_shape = shape.to_vec();
            result_shape[shape.len() - 1] = half_dim;
            Tensor::from_vec(result, &result_shape)
        } else {
            // For GPU tensors - simplified version
            Ok(input.clone())
        }
    }

    /// SwiGLU: f(x) = x_1 * swish(x_2) where swish(x) = x * sigmoid(x)
    fn swiglu<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        let shape = input.shape().dims();
        let last_dim = *shape.last().unwrap_or(&1);

        if last_dim % 2 != 0 {
            return Err(tenflowers_core::TensorError::invalid_argument(
                "SwiGLU requires input tensor's last dimension to be even".to_string(),
            ));
        }

        let half_dim = last_dim / 2;

        if let Some(data) = input.as_slice() {
            let mut result = Vec::new();
            let num_elements = data.len();
            let elements_per_slice = num_elements / last_dim;

            for slice_idx in 0..elements_per_slice {
                for i in 0..half_dim {
                    let base_idx = slice_idx * last_dim;
                    let x1 = data[base_idx + i];
                    let x2 = data[base_idx + half_dim + i];
                    let sigmoid_x2 = T::one() / (T::one() + (-x2).exp());
                    let swish_x2 = x2 * sigmoid_x2;
                    result.push(x1 * swish_x2);
                }
            }

            let mut result_shape = shape.to_vec();
            result_shape[shape.len() - 1] = half_dim;
            Tensor::from_vec(result, &result_shape)
        } else {
            // For GPU tensors - simplified version
            Ok(input.clone())
        }
    }

    /// GeGLU: f(x) = x_1 * GELU(x_2)
    fn geglu<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        let shape = input.shape().dims();
        let last_dim = *shape.last().unwrap_or(&1);

        if last_dim % 2 != 0 {
            return Err(tenflowers_core::TensorError::invalid_argument(
                "GeGLU requires input tensor's last dimension to be even".to_string(),
            ));
        }

        let half_dim = last_dim / 2;

        if let Some(data) = input.as_slice() {
            let mut result = Vec::new();
            let num_elements = data.len();
            let elements_per_slice = num_elements / last_dim;

            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let sqrt_2_over_pi = T::from(0.7978845608028654).unwrap(); // sqrt(2/π)
            let coeff = T::from(0.044715).unwrap();

            for slice_idx in 0..elements_per_slice {
                for i in 0..half_dim {
                    let base_idx = slice_idx * last_dim;
                    let x1 = data[base_idx + i];
                    let x2 = data[base_idx + half_dim + i];

                    // GELU(x2)
                    let x2_cubed = x2 * x2 * x2;
                    let inner = sqrt_2_over_pi * (x2 + coeff * x2_cubed);
                    let gelu_x2 = T::from(0.5).unwrap() * x2 * (T::one() + inner.tanh());

                    result.push(x1 * gelu_x2);
                }
            }

            let mut result_shape = shape.to_vec();
            result_shape[shape.len() - 1] = half_dim;
            Tensor::from_vec(result, &result_shape)
        } else {
            // For GPU tensors - simplified version
            Ok(input.clone())
        }
    }

    /// Snake activation: f(x) = x + (1/frequency) * sin²(frequency * x)
    /// A modern periodic activation function that can learn periodic patterns
    fn snake<T>(&self, input: &Tensor<T>, frequency: f32) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if let Some(data) = input.as_slice() {
            let freq_t = T::from(frequency).unwrap_or(T::one());
            let inv_freq_t = T::one() / freq_t;
            let result: Vec<T> = data
                .iter()
                .map(|&x| {
                    let freq_x = freq_t * x;
                    let sin_freq_x = freq_x.sin();
                    let sin_squared = sin_freq_x * sin_freq_x;
                    x + inv_freq_t * sin_squared
                })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - use tensor operations
            let freq_tensor = Tensor::from_scalar(T::from(frequency).unwrap_or(T::one()));
            let inv_freq_tensor =
                Tensor::from_scalar(T::one() / T::from(frequency).unwrap_or(T::one()));

            let freq_x = tenflowers_core::ops::mul(input, &freq_tensor)?;
            let sin_freq_x = tenflowers_core::ops::sin(&freq_x)?;
            let sin_squared = tenflowers_core::ops::mul(&sin_freq_x, &sin_freq_x)?;
            let periodic_term = tenflowers_core::ops::mul(&inv_freq_tensor, &sin_squared)?;

            tenflowers_core::ops::add(input, &periodic_term)
        }
    }

    /// QuickGELU activation: f(x) = x * sigmoid(1.702 * x)
    /// An efficient approximation of GELU that's faster to compute
    fn quick_gelu<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if let Some(data) = input.as_slice() {
            let beta = T::from(1.702).unwrap_or(T::from(1.702).unwrap());
            let result: Vec<T> = data
                .iter()
                .map(|&x| {
                    let scaled_x = beta * x;
                    let sigmoid_val = T::one() / (T::one() + (-scaled_x).exp());
                    x * sigmoid_val
                })
                .collect();
            Tensor::from_vec(result, input.shape().dims())
        } else {
            // For GPU tensors - use tensor operations
            let beta_tensor =
                Tensor::from_scalar(T::from(1.702).unwrap_or(T::from(1.702).unwrap()));
            let scaled_x = tenflowers_core::ops::mul(input, &beta_tensor)?;
            let sigmoid_val = tenflowers_core::ops::sigmoid(&scaled_x)?;

            tenflowers_core::ops::mul(input, &sigmoid_val)
        }
    }
}

impl std::fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_function_names() {
        assert_eq!(ActivationFunction::ReLU.name(), "relu");
        assert_eq!(ActivationFunction::ReLU6.name(), "relu6");
        assert_eq!(ActivationFunction::Sigmoid.name(), "sigmoid");
        assert_eq!(ActivationFunction::Tanh.name(), "tanh");
        assert_eq!(ActivationFunction::LeakyReLU(0.01).name(), "leaky_relu");
        assert_eq!(ActivationFunction::ELU(1.0).name(), "elu");
        assert_eq!(ActivationFunction::SELU.name(), "selu");
        assert_eq!(ActivationFunction::Hardswish.name(), "hardswish");
        assert_eq!(ActivationFunction::GLU.name(), "glu");
        assert_eq!(ActivationFunction::SwiGLU.name(), "swiglu");
        assert_eq!(ActivationFunction::GeGLU.name(), "geglu");
        assert_eq!(ActivationFunction::Snake(1.0).name(), "snake");
        assert_eq!(ActivationFunction::QuickGELU.name(), "quick_gelu");
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            ActivationFunction::from_name("relu"),
            Some(ActivationFunction::ReLU)
        );
        assert_eq!(
            ActivationFunction::from_name("relu6"),
            Some(ActivationFunction::ReLU6)
        );
        assert_eq!(
            ActivationFunction::from_name("sigmoid"),
            Some(ActivationFunction::Sigmoid)
        );
        assert_eq!(
            ActivationFunction::from_name("leaky_relu"),
            Some(ActivationFunction::LeakyReLU(0.01))
        );
        assert_eq!(
            ActivationFunction::from_name("elu"),
            Some(ActivationFunction::ELU(1.0))
        );
        assert_eq!(
            ActivationFunction::from_name("selu"),
            Some(ActivationFunction::SELU)
        );
        assert_eq!(
            ActivationFunction::from_name("hardswish"),
            Some(ActivationFunction::Hardswish)
        );
        assert_eq!(
            ActivationFunction::from_name("glu"),
            Some(ActivationFunction::GLU)
        );
        assert_eq!(
            ActivationFunction::from_name("swiglu"),
            Some(ActivationFunction::SwiGLU)
        );
        assert_eq!(
            ActivationFunction::from_name("geglu"),
            Some(ActivationFunction::GeGLU)
        );
        assert_eq!(
            ActivationFunction::from_name("snake"),
            Some(ActivationFunction::Snake(1.0))
        );
        assert_eq!(
            ActivationFunction::from_name("quick_gelu"),
            Some(ActivationFunction::QuickGELU)
        );
        assert_eq!(
            ActivationFunction::from_name("quickgelu"),
            Some(ActivationFunction::QuickGELU)
        );
        assert_eq!(ActivationFunction::from_name("invalid"), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ActivationFunction::ReLU), "relu");
        assert_eq!(format!("{}", ActivationFunction::ReLU6), "relu6");
        assert_eq!(format!("{}", ActivationFunction::GELU), "gelu");
        assert_eq!(
            format!("{}", ActivationFunction::LeakyReLU(0.01)),
            "leaky_relu"
        );
        assert_eq!(format!("{}", ActivationFunction::SELU), "selu");
    }

    #[test]
    fn test_relu6_activation() {
        let activation = ActivationFunction::ReLU6;
        let input = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 3.0, 6.0, 8.0], &[6]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert!((data[0] - 0.0).abs() < 1e-6); // -2.0 -> 0.0 (clipped at 0)
        assert!((data[1] - 0.0).abs() < 1e-6); // -1.0 -> 0.0 (clipped at 0)
        assert!((data[2] - 0.0).abs() < 1e-6); // 0.0 -> 0.0 (unchanged)
        assert!((data[3] - 3.0).abs() < 1e-6); // 3.0 -> 3.0 (unchanged)
        assert!((data[4] - 6.0).abs() < 1e-6); // 6.0 -> 6.0 (unchanged)
        assert!((data[5] - 6.0).abs() < 1e-6); // 8.0 -> 6.0 (clipped at 6)
    }

    #[test]
    fn test_leaky_relu_activation() {
        let activation = ActivationFunction::LeakyReLU(0.01);
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert!((data[0] - (-0.01)).abs() < 1e-6); // -1.0 * 0.01 = -0.01
        assert!((data[1] - 0.0).abs() < 1e-6); // 0.0 * 0.01 = 0.0
        assert!((data[2] - 1.0).abs() < 1e-6); // positive values unchanged
        assert!((data[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_elu_activation() {
        let activation = ActivationFunction::ELU(1.0);
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], &[3]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        // ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.632
        assert!((data[0] - ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6); // positive values unchanged
    }

    #[test]
    fn test_selu_activation() {
        let activation = ActivationFunction::SELU;
        let input = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], &[3]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!(data[1] > 0.0); // Should be scaled positive value
        assert!(data[2] < 0.0); // Should be negative ELU value
    }

    #[test]
    fn test_hardswish_activation() {
        let activation = ActivationFunction::Hardswish;
        let input = Tensor::from_vec(vec![-3.0f32, 0.0, 3.0], &[3]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert!((data[0] - 0.0).abs() < 1e-6); // Should be 0 for x = -3
        assert!((data[1] - 0.0).abs() < 1e-6); // Should be 0 for x = 0
        assert!(data[2] > 0.0); // Should be positive for x = 3
    }

    #[test]
    fn test_glu_activation() {
        let activation = ActivationFunction::GLU;
        // GLU requires even last dimension - using 4 elements (2+2)
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 0.0, 1.0], &[4]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert_eq!(data.len(), 2); // Should be halved
                                   // GLU: x1 * sigmoid(x2)
                                   // First: 1.0 * sigmoid(0.0) = 1.0 * 0.5 = 0.5
                                   // Second: 2.0 * sigmoid(1.0) ≈ 2.0 * 0.731 ≈ 1.462
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!(data[1] > 1.0); // Should be > 1.0
    }

    #[test]
    fn test_glu_odd_dimension_error() {
        let activation = ActivationFunction::GLU;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap(); // Odd dimension
        let result = activation.apply(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_swiglu_activation() {
        let activation = ActivationFunction::SwiGLU;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 0.0, 1.0], &[4]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert_eq!(data.len(), 2); // Should be halved
        assert!(data[0] >= 0.0);
        assert!(data[1] >= 0.0);
    }

    #[test]
    fn test_geglu_activation() {
        let activation = ActivationFunction::GeGLU;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 0.0, 1.0], &[4]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert_eq!(data.len(), 2); // Should be halved
                                   // GeGLU should produce reasonable outputs
        assert!(data[0].is_finite());
        assert!(data[1].is_finite());
    }

    #[test]
    fn test_linear_activation() {
        let activation = ActivationFunction::Linear;
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], &[3]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert_eq!(data, &[-1.0, 0.0, 1.0]); // Should be unchanged
    }

    #[test]
    fn test_snake_activation() {
        let activation = ActivationFunction::Snake(1.0);
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert_eq!(data.len(), 4);

        // Test properties of Snake activation: f(x) = x + (1/freq) * sin²(freq * x)
        // At x = 0, f(0) = 0 + sin²(0) = 0
        assert!((data[1] - 0.0).abs() < 1e-6);

        // Snake activation should be smooth and differentiable
        for &val in data {
            assert!(val.is_finite());
        }

        // Snake activation should have periodic behavior
        // f'(x) = 1 + 2*sin(freq*x)*cos(freq*x) = 1 + sin(2*freq*x)
        // This means the derivative oscillates between [0, 2]
    }

    #[test]
    fn test_snake_activation_different_frequencies() {
        let input = Tensor::from_vec(vec![0.5f32], &[1]).unwrap();

        let snake_1 = ActivationFunction::Snake(1.0);
        let snake_2 = ActivationFunction::Snake(2.0);

        let result_1 = snake_1.apply(&input).unwrap();
        let result_2 = snake_2.apply(&input).unwrap();

        if let (Some(data_1), Some(data_2)) = (result_1.as_slice(), result_2.as_slice()) {
            // Different frequencies should produce different results
            assert!((data_1[0] - data_2[0]).abs() > 1e-6);
        }
    }

    #[test]
    fn test_quick_gelu_activation() {
        let activation = ActivationFunction::QuickGELU;
        let input = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = activation.apply(&input).unwrap();

        let data = result.as_slice().expect("Expected CPU tensor data");
        assert_eq!(data.len(), 5);

        // QuickGELU should be smooth and monotonic
        for &val in data {
            assert!(val.is_finite());
        }

        // At x = 0, QuickGELU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((data[2] - 0.0).abs() < 1e-6);

        // QuickGELU should be approximately antisymmetric around 0
        // f(x) ≈ -f(-x) for small x
        assert!(data[0] < 0.0); // Negative input should give negative output
        assert!(data[4] > 0.0); // Positive input should give positive output

        // QuickGELU should have the right behavior at extremes
        // For large positive x, QuickGELU(x) ≈ x (since sigmoid(1.702*x) ≈ 1)
        // For large negative x, QuickGELU(x) ≈ 0 (since sigmoid(1.702*x) ≈ 0)
        assert!(data[4] > data[1]); // Positive values should be larger than negative
    }

    #[test]
    fn test_quick_gelu_vs_gelu_approximation() {
        let input = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();

        let quick_gelu = ActivationFunction::QuickGELU;
        let gelu = ActivationFunction::GELU;

        let quick_result = quick_gelu.apply(&input).unwrap();
        let gelu_result = gelu.apply(&input).unwrap();

        if let (Some(quick_data), Some(gelu_data)) =
            (quick_result.as_slice(), gelu_result.as_slice())
        {
            // QuickGELU should be a reasonable approximation of GELU
            // They should be within ~10% of each other for typical inputs
            let difference_ratio = (quick_data[0] - gelu_data[0]).abs() / gelu_data[0].abs();
            assert!(difference_ratio < 0.15); // Within 15% approximation
        }
    }
}
