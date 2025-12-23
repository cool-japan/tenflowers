use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

#[derive(Debug)]
pub struct Dense<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    activation: Option<String>,
    training: bool,
}

impl<T> Dense<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        let weight = Tensor::zeros(&[input_dim, output_dim]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            activation: None,
            training: false,
        }
    }

    /// Create Dense layer with Xavier/Glorot initialization (recommended for sigmoid/tanh)
    pub fn new_xavier(input_dim: usize, output_dim: usize, use_bias: bool) -> Self
    where
        T: FromPrimitive,
    {
        let weight = Self::create_xavier_weight(&[input_dim, output_dim]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            activation: None,
            training: false,
        }
    }

    /// Create Dense layer with He initialization (recommended for ReLU activations)
    pub fn new_he(input_dim: usize, output_dim: usize, use_bias: bool) -> Self
    where
        T: FromPrimitive,
    {
        let weight = Self::create_he_weight(&[input_dim, output_dim]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            activation: None,
            training: false,
        }
    }

    /// Create Dense layer with custom normal initialization
    pub fn new_normal(
        input_dim: usize,
        output_dim: usize,
        use_bias: bool,
        mean: f32,
        std: f32,
    ) -> Self
    where
        T: FromPrimitive,
    {
        let weight = Self::create_normal_weight(&[input_dim, output_dim], mean, std);
        let bias = if use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            activation: None,
            training: false,
        }
    }

    pub fn with_activation(mut self, activation: String) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Get a reference to the weight tensor
    pub fn weight(&self) -> &Tensor<T> {
        &self.weight
    }

    /// Get a reference to the bias tensor (if any)
    pub fn bias(&self) -> Option<&Tensor<T>> {
        self.bias.as_ref()
    }

    /// Set the weight tensor
    pub fn set_weight(&mut self, weight: Tensor<T>) {
        self.weight = weight;
    }

    /// Set the bias tensor
    pub fn set_bias(&mut self, bias: Option<Tensor<T>>) {
        self.bias = bias;
    }

    /// Get the activation function name
    pub fn activation(&self) -> Option<&String> {
        self.activation.as_ref()
    }

    /// Get weights tensor by value (for FFI compatibility)
    pub fn weights(&self) -> Option<Tensor<T>> {
        Some(self.weight.clone())
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.weight.shape().dims()[1]
    }

    /// Get activation name
    pub fn activation_name(&self) -> Option<String> {
        self.activation.clone()
    }

    /// Check if the layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Reinitialize weights with Xavier/Glorot initialization
    pub fn reinit_xavier(&mut self)
    where
        T: FromPrimitive,
    {
        let shape = self.weight.shape().dims();
        self.weight = Self::create_xavier_weight(shape);
    }

    /// Reinitialize weights with He initialization
    pub fn reinit_he(&mut self)
    where
        T: FromPrimitive,
    {
        let shape = self.weight.shape().dims();
        self.weight = Self::create_he_weight(shape);
    }

    /// Reinitialize weights with normal distribution
    pub fn reinit_normal(&mut self, mean: f32, std: f32)
    where
        T: FromPrimitive,
    {
        let shape = self.weight.shape().dims();
        self.weight = Self::create_normal_weight(shape, mean, std);
    }

    /// Create a deep copy of this Dense layer
    pub fn clone_layer(&self) -> Self
    where
        T: Clone,
    {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            activation: self.activation.clone(),
            training: self.training,
        }
    }

    /// Create Xavier/Glorot initialized weight tensor
    fn create_xavier_weight(shape: &[usize]) -> Tensor<T>
    where
        T: FromPrimitive,
    {
        let (fan_in, fan_out) = Self::calculate_fan_in_fan_out(shape);
        let variance = 2.0 / (fan_in + fan_out) as f32;
        let std_dev = variance.sqrt();
        Self::create_random_normal_tensor(shape, 0.0, std_dev)
            .unwrap_or_else(|_| Tensor::zeros(shape))
    }

    /// Create He initialized weight tensor
    fn create_he_weight(shape: &[usize]) -> Tensor<T>
    where
        T: FromPrimitive,
    {
        let (fan_in, _) = Self::calculate_fan_in_fan_out(shape);
        let variance = 2.0 / fan_in as f32;
        let std_dev = variance.sqrt();
        Self::create_random_normal_tensor(shape, 0.0, std_dev)
            .unwrap_or_else(|_| Tensor::zeros(shape))
    }

    /// Create normal distribution initialized weight tensor
    fn create_normal_weight(shape: &[usize], mean: f32, std: f32) -> Tensor<T>
    where
        T: FromPrimitive,
    {
        Self::create_random_normal_tensor(shape, mean, std).unwrap_or_else(|_| Tensor::zeros(shape))
    }

    /// Calculate fan-in and fan-out for a tensor shape
    fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            1 => (shape[0], shape[0]), // 1D tensor (bias)
            2 => (shape[0], shape[1]), // 2D tensor (dense layer): [input_dim, output_dim]
            _ => {
                // Default fallback for higher dimensions
                let total_size = shape.iter().product::<usize>();
                let sqrt_size = (total_size as f64).sqrt() as usize;
                (sqrt_size, sqrt_size)
            }
        }
    }

    /// Create a tensor with random normal distribution using Box-Muller transform
    fn create_random_normal_tensor(shape: &[usize], mean: f32, std_dev: f32) -> Result<Tensor<T>>
    where
        T: FromPrimitive,
    {
        let total_elements = shape.iter().product::<usize>();

        let random_data = (0..total_elements)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = scirs2_core::random::quick::random_f32().max(1e-10);
                let u2: f32 = scirs2_core::random::quick::random_f32();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let value = mean + std_dev * z0;
                T::from_f32(value).unwrap_or(T::zero())
            })
            .collect::<Vec<T>>();

        Tensor::from_vec(random_data, shape)
    }
}

impl<T> Clone for Dense<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        self.clone_layer()
    }
}

impl<T> Layer<T> for Dense<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::cmp::PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let output = input.matmul(&self.weight)?;

        let output = if let Some(ref bias) = self.bias {
            output.add(bias)?
        } else {
            output
        };

        match self.activation.as_deref() {
            Some("relu") => tenflowers_core::ops::relu(&output),
            Some("sigmoid") => tenflowers_core::ops::sigmoid(&output),
            Some("tanh") => tenflowers_core::ops::tanh(&output),
            _ => Ok(output),
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> crate::layers::LayerType {
        crate::layers::LayerType::Dense
    }

    fn set_weight(&mut self, weight: Tensor<T>) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    fn set_bias(&mut self, bias: Option<Tensor<T>>) -> Result<()> {
        self.bias = bias;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dense_creation() {
        let dense = Dense::<f32>::new(3, 2, true);
        assert_eq!(dense.weight().shape().dims(), &[3, 2]);
        assert!(dense.bias().is_some());
        assert_eq!(dense.bias().unwrap().shape().dims(), &[2]);
    }

    #[test]
    fn test_dense_xavier_initialization() {
        let dense = Dense::<f32>::new_xavier(4, 3, true);
        assert_eq!(dense.weight().shape().dims(), &[4, 3]);
        assert!(dense.bias().is_some());

        // Check that weights are not all zeros (they should be randomly initialized)
        let weight_data = dense.weight().as_slice().unwrap();
        let all_zeros = weight_data.iter().all(|&x| x == 0.0);
        assert!(
            !all_zeros,
            "Xavier initialized weights should not all be zero"
        );
    }

    #[test]
    fn test_dense_he_initialization() {
        let dense = Dense::<f32>::new_he(5, 2, false);
        assert_eq!(dense.weight().shape().dims(), &[5, 2]);
        assert!(dense.bias().is_none());

        // Check that weights are not all zeros
        let weight_data = dense.weight().as_slice().unwrap();
        let all_zeros = weight_data.iter().all(|&x| x == 0.0);
        assert!(!all_zeros, "He initialized weights should not all be zero");
    }

    #[test]
    fn test_dense_normal_initialization() {
        let dense = Dense::<f32>::new_normal(3, 3, true, 0.5, 0.1);
        assert_eq!(dense.weight().shape().dims(), &[3, 3]);

        // Check that weights are not all zeros and roughly centered around mean
        let weight_data = dense.weight().as_slice().unwrap();
        let all_zeros = weight_data.iter().all(|&x| x == 0.0);
        assert!(
            !all_zeros,
            "Normal initialized weights should not all be zero"
        );

        // Check that the mean is roughly correct (with some tolerance for randomness)
        let mean: f32 = weight_data.iter().sum::<f32>() / weight_data.len() as f32;
        assert_abs_diff_eq!(mean, 0.5, epsilon = 0.2); // Allow some variance due to randomness
    }

    #[test]
    fn test_dense_reinit_xavier() {
        let mut dense = Dense::<f32>::new(2, 2, false);

        // Initially should be zeros
        let initial_weights = dense.weight().as_slice().unwrap();
        assert!(initial_weights.iter().all(|&x| x == 0.0));

        // Reinitialize with Xavier
        dense.reinit_xavier();

        // Should no longer be all zeros
        let new_weights = dense.weight().as_slice().unwrap();
        let all_zeros = new_weights.iter().all(|&x| x == 0.0);
        assert!(
            !all_zeros,
            "Xavier reinitialized weights should not all be zero"
        );
    }

    #[test]
    fn test_dense_reinit_he() {
        let mut dense = Dense::<f32>::new(3, 2, true);

        // Reinitialize with He
        dense.reinit_he();

        // Should not be all zeros
        let weights = dense.weight().as_slice().unwrap();
        let all_zeros = weights.iter().all(|&x| x == 0.0);
        assert!(
            !all_zeros,
            "He reinitialized weights should not all be zero"
        );
    }

    #[test]
    fn test_dense_reinit_normal() {
        let mut dense = Dense::<f32>::new(2, 3, false);

        // Reinitialize with normal distribution
        dense.reinit_normal(1.0, 0.2);

        // Should not be all zeros
        let weights = dense.weight().as_slice().unwrap();
        let all_zeros = weights.iter().all(|&x| x == 0.0);
        assert!(
            !all_zeros,
            "Normal reinitialized weights should not all be zero"
        );

        // Check that the mean is roughly correct
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        assert_abs_diff_eq!(mean, 1.0, epsilon = 0.3); // Allow some variance
    }

    #[test]
    fn test_fan_in_fan_out_calculation() {
        // Test 2D tensor (dense layer)
        let (fan_in, fan_out) = Dense::<f32>::calculate_fan_in_fan_out(&[4, 3]);
        assert_eq!(fan_in, 4);
        assert_eq!(fan_out, 3);

        // Test 1D tensor
        let (fan_in, fan_out) = Dense::<f32>::calculate_fan_in_fan_out(&[5]);
        assert_eq!(fan_in, 5);
        assert_eq!(fan_out, 5);
    }

    #[test]
    fn test_xavier_initialization_variance() {
        // Create multiple Dense layers and check that Xavier initialization
        // produces roughly the expected variance
        let mut variances = Vec::new();

        for _ in 0..10 {
            let dense = Dense::<f32>::new_xavier(10, 5, false);
            let weights = dense.weight().as_slice().unwrap();

            let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
            let variance: f32 =
                weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;

            variances.push(variance);
        }

        let avg_variance: f32 = variances.iter().sum::<f32>() / variances.len() as f32;

        // Expected Xavier variance: 2 / (fan_in + fan_out) = 2 / (10 + 5) = 2/15 ≈ 0.133
        let expected_variance = 2.0 / 15.0;

        // Allow reasonable tolerance for randomness
        assert_abs_diff_eq!(avg_variance, expected_variance, epsilon = 0.1);
    }
}
