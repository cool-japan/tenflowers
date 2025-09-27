use crate::layers::Layer;
use num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Stochastic Depth layer wrapper for regularization
///
/// During training, randomly skips the wrapped layer with probability `drop_prob`.
/// During inference, always applies the layer but scales output by survival probability.
/// This helps prevent overfitting and improves gradient flow in deep networks.
pub struct StochasticDepth<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// The wrapped layer to potentially skip
    layer: Box<dyn Layer<T>>,
    /// Probability of dropping (skipping) the layer during training
    drop_prob: T,
    /// Whether the layer is in training mode
    training: bool,
}

impl<T> StochasticDepth<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new StochasticDepth wrapper
    ///
    /// # Arguments
    /// * `layer` - The layer to wrap
    /// * `drop_prob` - Probability of dropping the layer during training (0.0 to 1.0)
    pub fn new(layer: Box<dyn Layer<T>>, drop_prob: T) -> Self {
        Self {
            layer,
            drop_prob,
            training: true,
        }
    }

    /// Create a new StochasticDepth wrapper with default drop probability (0.1)
    pub fn new_default(layer: Box<dyn Layer<T>>) -> Self {
        let drop_prob = T::from(0.1).unwrap();
        Self::new(layer, drop_prob)
    }

    /// Set the drop probability
    pub fn set_drop_prob(&mut self, drop_prob: T) {
        self.drop_prob = drop_prob;
    }

    /// Get the drop probability
    pub fn get_drop_prob(&self) -> T {
        self.drop_prob
    }

    /// Get the survival probability (1 - drop_prob)
    pub fn get_survival_prob(&self) -> T {
        T::one() - self.drop_prob
    }
}

impl<T> Layer<T> for StochasticDepth<T>
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
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.training {
            // During inference, always apply the layer but scale by survival probability
            let layer_output = self.layer.forward(input)?;
            let survival_prob = self.get_survival_prob();
            let survival_tensor = Tensor::from_scalar(survival_prob);

            // Apply residual connection: input + survival_prob * layer_output
            let scaled_output = layer_output.mul(&survival_tensor)?;
            input.add(&scaled_output)
        } else {
            // During training, randomly skip the layer
            let mut rng = scirs2_core::random::rng();
            let random_val: f64 = rng.gen_range(0.0..1.0);
            let random_t = T::from(random_val).unwrap();

            if random_t < self.drop_prob {
                // Skip the layer (identity function)
                Ok(input.clone())
            } else {
                // Apply the layer with residual connection
                let layer_output = self.layer.forward(input)?;
                let survival_prob = self.get_survival_prob();
                let _survival_tensor = Tensor::from_scalar(survival_prob);

                // Scale by 1/survival_prob for unbiased estimation during training
                let scale_factor = T::one() / survival_prob;
                let scale_tensor = Tensor::from_scalar(scale_factor);
                let scaled_output = layer_output.mul(&scale_tensor)?;

                // Apply residual connection: input + scaled_output
                input.add(&scaled_output)
            }
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        // Return parameters from the wrapped layer
        self.layer.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        // Return mutable parameters from the wrapped layer
        self.layer.parameters_mut()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        // Also set training mode for the wrapped layer
        self.layer.set_training(training);
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(StochasticDepth {
            layer: self.layer.clone_box(),
            drop_prob: self.drop_prob.clone(),
            training: self.training,
        })
    }
}

/// Stochastic Depth layer without residual connection
///
/// This version skips the layer entirely during training instead of using residual connections.
/// Useful for architectures where residual connections are not desired.
pub struct StochasticDepthNoResidual<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// The wrapped layer to potentially skip
    layer: Box<dyn Layer<T>>,
    /// Probability of dropping (skipping) the layer during training
    drop_prob: T,
    /// Whether the layer is in training mode
    training: bool,
}

impl<T> StochasticDepthNoResidual<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new StochasticDepthNoResidual wrapper
    ///
    /// # Arguments
    /// * `layer` - The layer to wrap
    /// * `drop_prob` - Probability of dropping the layer during training (0.0 to 1.0)
    pub fn new(layer: Box<dyn Layer<T>>, drop_prob: T) -> Self {
        Self {
            layer,
            drop_prob,
            training: true,
        }
    }

    /// Create a new StochasticDepthNoResidual wrapper with default drop probability (0.1)
    pub fn new_default(layer: Box<dyn Layer<T>>) -> Self {
        let drop_prob = T::from(0.1).unwrap();
        Self::new(layer, drop_prob)
    }

    /// Set the drop probability
    pub fn set_drop_prob(&mut self, drop_prob: T) {
        self.drop_prob = drop_prob;
    }

    /// Get the drop probability
    pub fn get_drop_prob(&self) -> T {
        self.drop_prob
    }

    /// Get the survival probability (1 - drop_prob)
    pub fn get_survival_prob(&self) -> T {
        T::one() - self.drop_prob
    }
}

impl<T> Layer<T> for StochasticDepthNoResidual<T>
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
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.training {
            // During inference, always apply the layer but scale by survival probability
            let layer_output = self.layer.forward(input)?;
            let survival_prob = self.get_survival_prob();
            let survival_tensor = Tensor::from_scalar(survival_prob);
            layer_output.mul(&survival_tensor)
        } else {
            // During training, randomly skip the layer
            let mut rng = scirs2_core::random::rng();
            let random_val: f64 = rng.gen_range(0.0..1.0);
            let random_t = T::from(random_val).unwrap();

            if random_t < self.drop_prob {
                // Skip the layer - return zeros with same shape as input
                Ok(Tensor::zeros(input.shape().dims()))
            } else {
                // Apply the layer with scaling for unbiased estimation
                let layer_output = self.layer.forward(input)?;
                let survival_prob = self.get_survival_prob();
                let scale_factor = T::one() / survival_prob;
                let scale_tensor = Tensor::from_scalar(scale_factor);
                layer_output.mul(&scale_tensor)
            }
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        // Return parameters from the wrapped layer
        self.layer.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        // Return mutable parameters from the wrapped layer
        self.layer.parameters_mut()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        // Also set training mode for the wrapped layer
        self.layer.set_training(training);
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(StochasticDepthNoResidual {
            layer: self.layer.clone_box(),
            drop_prob: self.drop_prob.clone(),
            training: self.training,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    // Simple test layer that just multiplies input by 2
    #[derive(Clone)]
    struct TestLayer<T> {
        multiplier: T,
    }

    impl<T> TestLayer<T> {
        fn new(multiplier: T) -> Self {
            Self { multiplier }
        }
    }

    impl<T> Layer<T> for TestLayer<T>
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
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
            let multiplier_tensor = Tensor::from_scalar(self.multiplier);
            input.mul(&multiplier_tensor)
        }

        fn parameters(&self) -> Vec<&Tensor<T>> {
            vec![]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
            vec![]
        }

        fn set_training(&mut self, _training: bool) {}

        fn clone_box(&self) -> Box<dyn Layer<T>> {
            Box::new((*self).clone())
        }
    }

    #[test]
    fn test_stochastic_depth_inference_mode() {
        let test_layer = Box::new(TestLayer::new(2.0_f32));
        let mut stochastic_depth = StochasticDepth::new(test_layer, 0.5);
        stochastic_depth.set_training(false);

        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = stochastic_depth.forward(&input).unwrap();

        // In inference mode with survival_prob = 0.5, output should be input + 0.5 * (2 * input)
        // = input + input = 2 * input
        if let Some(output_data) = output.as_slice() {
            assert!((output_data[0] - 2.0).abs() < 1e-5); // 1 + 0.5 * 2 = 2
            assert!((output_data[1] - 4.0).abs() < 1e-5); // 2 + 0.5 * 4 = 4
            assert!((output_data[2] - 6.0).abs() < 1e-5); // 3 + 0.5 * 6 = 6
        }
    }

    #[test]
    fn test_stochastic_depth_training_mode() {
        let test_layer = Box::new(TestLayer::new(2.0_f32));
        let mut stochastic_depth = StochasticDepth::new(test_layer, 0.0); // Never skip
        stochastic_depth.set_training(true);

        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = stochastic_depth.forward(&input).unwrap();

        // With drop_prob = 0.0, layer should always be applied with scale 1/1 = 1
        // Output should be input + 1 * (2 * input) = 3 * input
        if let Some(output_data) = output.as_slice() {
            assert!((output_data[0] - 3.0).abs() < 1e-5); // 1 + 1 * 2 = 3
            assert!((output_data[1] - 6.0).abs() < 1e-5); // 2 + 1 * 4 = 6
            assert!((output_data[2] - 9.0).abs() < 1e-5); // 3 + 1 * 6 = 9
        }
    }

    #[test]
    fn test_stochastic_depth_no_residual_inference() {
        let test_layer = Box::new(TestLayer::new(2.0_f32));
        let mut stochastic_depth = StochasticDepthNoResidual::new(test_layer, 0.5);
        stochastic_depth.set_training(false);

        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = stochastic_depth.forward(&input).unwrap();

        // In inference mode with survival_prob = 0.5, output should be 0.5 * (2 * input)
        if let Some(output_data) = output.as_slice() {
            assert!((output_data[0] - 1.0).abs() < 1e-5); // 0.5 * 2 = 1
            assert!((output_data[1] - 2.0).abs() < 1e-5); // 0.5 * 4 = 2
            assert!((output_data[2] - 3.0).abs() < 1e-5); // 0.5 * 6 = 3
        }
    }

    #[test]
    fn test_stochastic_depth_no_residual_training() {
        let test_layer = Box::new(TestLayer::new(2.0_f32));
        let mut stochastic_depth = StochasticDepthNoResidual::new(test_layer, 0.0); // Never skip
        stochastic_depth.set_training(true);

        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = stochastic_depth.forward(&input).unwrap();

        // With drop_prob = 0.0, layer should always be applied with scale 1/1 = 1
        // Output should be 1 * (2 * input) = 2 * input
        if let Some(output_data) = output.as_slice() {
            assert!((output_data[0] - 2.0).abs() < 1e-5); // 1 * 2 = 2
            assert!((output_data[1] - 4.0).abs() < 1e-5); // 1 * 4 = 4
            assert!((output_data[2] - 6.0).abs() < 1e-5); // 1 * 6 = 6
        }
    }

    #[test]
    fn test_stochastic_depth_parameters() {
        let test_layer = Box::new(TestLayer::new(2.0_f32));
        let stochastic_depth = StochasticDepth::new(test_layer, 0.5);

        // Should have no parameters since TestLayer has no parameters
        assert_eq!(stochastic_depth.parameters().len(), 0);
    }

    #[test]
    fn test_stochastic_depth_probabilities() {
        let test_layer = Box::new(TestLayer::new(2.0_f32));
        let mut stochastic_depth = StochasticDepth::new(test_layer, 0.3);

        assert!((stochastic_depth.get_drop_prob() - 0.3).abs() < 1e-5);
        assert!((stochastic_depth.get_survival_prob() - 0.7).abs() < 1e-5);

        stochastic_depth.set_drop_prob(0.2);
        assert!((stochastic_depth.get_drop_prob() - 0.2).abs() < 1e-5);
        assert!((stochastic_depth.get_survival_prob() - 0.8).abs() < 1e-5);
    }
}
