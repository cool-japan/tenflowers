// Examples demonstrating the Model Subclassing API
// These are not included in the main build but serve as documentation

#[allow(dead_code)]
#[allow(clippy::module_inception)]
pub mod examples {
    use crate::layers::{Dense, Layer};
    use crate::model::{helpers, CustomModel, LayerContainer, Model, ModelBase, ModelExt};
    use tenflowers_core::ops::activation;
    use tenflowers_core::{Result, Tensor};

    /// Example 1: Simple custom model using ModelBase and LayerContainer
    ///
    /// This example shows the easiest way to create a custom model using the
    /// provided infrastructure classes.
    pub struct SimpleCustomModel {
        base: ModelBase<f32>,
        layers: LayerContainer<f32>,
    }

    impl Default for SimpleCustomModel {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SimpleCustomModel {
        pub fn new() -> Self {
            let mut model = Self {
                base: ModelBase::new_named("SimpleCustomModel".to_string()),
                layers: LayerContainer::new(),
            };

            // Add layers to the model
            model.layers.add_named_layer(
                Box::new(Dense::<f32>::new(784, 128, true)),
                "hidden1".to_string(),
            );
            model.layers.add_named_layer(
                Box::new(Dense::<f32>::new(128, 64, true)),
                "hidden2".to_string(),
            );
            model.layers.add_named_layer(
                Box::new(Dense::<f32>::new(64, 10, true)),
                "output".to_string(),
            );

            model
        }

        fn call_impl(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            // Custom forward pass logic
            let h1 = self
                .layers
                .get_layer_by_name("hidden1")
                .unwrap()
                .forward(input)?;
            // Apply ReLU activation
            let h1_relu = activation::relu(&h1)?;

            let h2 = self
                .layers
                .get_layer_by_name("hidden2")
                .unwrap()
                .forward(&h1_relu)?;
            let h2_relu = activation::relu(&h2)?;

            let output = self
                .layers
                .get_layer_by_name("output")
                .unwrap()
                .forward(&h2_relu)?;
            Ok(output)
        }
    }

    // Implement Model trait using manual implementation
    impl Model<f32> for SimpleCustomModel {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            self.call_impl(input)
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            self.layers.parameters()
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            self.layers.parameters_mut()
        }

        fn set_training(&mut self, training: bool) {
            self.base.set_training(training);
            self.layers.set_training(training);
        }

        fn zero_grad(&mut self) {
            for param in self.parameters_mut() {
                crate::model::zero_tensor_grad(param);
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl ModelExt<f32> for SimpleCustomModel {
        fn is_training_mode(&self) -> bool {
            self.base.is_training()
        }
    }

    /// Example 2: Advanced custom model with custom state
    ///
    /// This example shows how to create a more complex model with custom state
    /// and more sophisticated forward pass logic.
    pub struct ResidualBlock {
        base: ModelBase<f32>,
        conv1: Box<dyn Layer<f32>>,
        conv2: Box<dyn Layer<f32>>,
        shortcut: Option<Box<dyn Layer<f32>>>,
    }

    impl ResidualBlock {
        pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
            let mut base = ModelBase::new_named("ResidualBlock".to_string());
            base.add_metadata("in_channels".to_string(), in_channels.to_string());
            base.add_metadata("out_channels".to_string(), out_channels.to_string());
            base.add_metadata("stride".to_string(), stride.to_string());

            // For this example, we'll use Dense layers as placeholders for Conv2D
            let conv1 = Box::new(Dense::<f32>::new(in_channels, out_channels, true));
            let conv2 = Box::new(Dense::<f32>::new(out_channels, out_channels, true));

            // Add shortcut connection if needed
            let shortcut = if in_channels != out_channels || stride != 1 {
                Some(
                    Box::new(Dense::<f32>::new(in_channels, out_channels, false))
                        as Box<dyn Layer<f32>>,
                )
            } else {
                None
            };

            Self {
                base,
                conv1,
                conv2,
                shortcut,
            }
        }

        fn call_impl(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            // First convolution + activation
            let out = self.conv1.forward(input)?;
            let out = activation::relu(&out)?;

            // Second convolution
            let out = self.conv2.forward(&out)?;

            // Add shortcut connection
            let out = if let Some(ref shortcut_layer) = self.shortcut {
                let shortcut = shortcut_layer.forward(input)?;
                out.add(&shortcut)?
            } else {
                out.add(input)?
            };

            // Final activation
            activation::relu(&out)
        }
    }

    impl Model<f32> for ResidualBlock {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            self.call_impl(input)
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            let mut params = Vec::new();
            params.extend(self.conv1.parameters());
            params.extend(self.conv2.parameters());
            if let Some(ref shortcut) = self.shortcut {
                params.extend(shortcut.parameters());
            }
            params
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            let mut params = Vec::new();
            params.extend(self.conv1.parameters_mut());
            params.extend(self.conv2.parameters_mut());
            if let Some(ref mut shortcut) = self.shortcut {
                params.extend(shortcut.parameters_mut());
            }
            params
        }

        fn set_training(&mut self, training: bool) {
            self.base.set_training(training);
            self.conv1.set_training(training);
            self.conv2.set_training(training);
            if let Some(ref mut shortcut) = self.shortcut {
                shortcut.set_training(training);
            }
        }

        fn zero_grad(&mut self) {
            for param in self.parameters_mut() {
                crate::model::zero_tensor_grad(param);
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl ModelExt<f32> for ResidualBlock {
        fn is_training_mode(&self) -> bool {
            self.base.is_training()
        }
    }

    /// Example 3: Using the CustomModel template
    ///
    /// This example shows how to use the pre-built CustomModel template
    /// for simple sequential models.
    pub fn create_template_model() -> CustomModel<f32> {
        let mut model = CustomModel::new_named("TemplateModel".to_string());

        // Add layers sequentially
        model.add_named_layer(
            Box::new(Dense::<f32>::new(784, 256, true)),
            "encoder1".to_string(),
        );
        model.add_named_layer(
            Box::new(Dense::<f32>::new(256, 128, true)),
            "encoder2".to_string(),
        );
        model.add_named_layer(
            Box::new(Dense::<f32>::new(128, 64, true)),
            "bottleneck".to_string(),
        );
        model.add_named_layer(
            Box::new(Dense::<f32>::new(64, 128, true)),
            "decoder1".to_string(),
        );
        model.add_named_layer(
            Box::new(Dense::<f32>::new(128, 784, true)),
            "decoder2".to_string(),
        );

        model
    }

    /// Example 4: Model composition using existing models as layers
    ///
    /// This example shows how to compose larger models from smaller model components.
    pub struct CompositeModel {
        base: ModelBase<f32>,
        encoder: SimpleCustomModel,
        decoder: SimpleCustomModel,
    }

    impl Default for CompositeModel {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CompositeModel {
        pub fn new() -> Self {
            Self {
                base: ModelBase::new_named("CompositeModel".to_string()),
                encoder: SimpleCustomModel::new(),
                decoder: SimpleCustomModel::new(),
            }
        }

        fn call_impl(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            let encoded = self.encoder.forward(input)?;
            self.decoder.forward(&encoded)
        }
    }

    impl Model<f32> for CompositeModel {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            self.call_impl(input)
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            let mut params = Vec::new();
            params.extend(self.encoder.parameters());
            params.extend(self.decoder.parameters());
            params
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            let mut params = Vec::new();
            params.extend(self.encoder.parameters_mut());
            params.extend(self.decoder.parameters_mut());
            params
        }

        fn set_training(&mut self, training: bool) {
            self.base.set_training(training);
            self.encoder.set_training(training);
            self.decoder.set_training(training);
        }

        fn zero_grad(&mut self) {
            for param in self.parameters_mut() {
                crate::model::zero_tensor_grad(param);
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl ModelExt<f32> for CompositeModel {
        fn is_training_mode(&self) -> bool {
            self.base.is_training()
        }
    }

    /// Example usage demonstrating the model subclassing features
    pub fn example_usage() -> Result<()> {
        // Create a simple custom model
        let mut simple_model = SimpleCustomModel::new();

        // Use ModelExt trait features
        println!("Model summary: {}", simple_model.summary());

        // Create sample input
        let input = Tensor::<f32>::ones(&[32, 784]);

        // Forward pass in training mode
        let output_train = simple_model.forward(&input)?;
        println!("Training output shape: {:?}", output_train.shape().dims());

        // Forward pass in evaluation mode
        let output_eval = simple_model.predict(&input)?;
        println!("Evaluation output shape: {:?}", output_eval.shape().dims());

        // Use helper functions
        helpers::enable_gradients(&mut simple_model);
        let model_info = helpers::model_info(&simple_model);
        println!("Model info: {model_info:?}");

        // Create and use template model
        let template_model = create_template_model();
        println!("Template model has {} layers", template_model.layers.len());

        // Create residual block
        let residual = ResidualBlock::new(64, 64, 1);
        let input_64 = Tensor::<f32>::ones(&[32, 64]);
        let residual_output = residual.forward(&input_64)?;
        println!(
            "Residual output shape: {:?}",
            residual_output.shape().dims()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::examples::*;
    use crate::model::{Model, ModelExt};
    use tenflowers_core::Tensor;

    #[test]
    fn test_simple_custom_model() {
        let model = SimpleCustomModel::new();
        // Test that the model works properly
        assert!(model.is_training_mode());
        assert!(!model.parameters().is_empty());
    }

    #[test]
    fn test_residual_block() {
        let block = ResidualBlock::new(64, 64, 1);
        assert!(block.is_training_mode());

        let block_with_shortcut = ResidualBlock::new(32, 64, 2);
        assert!(block_with_shortcut.is_training_mode());
        // Different input/output channels means there should be parameters
        assert!(!block_with_shortcut.parameters().is_empty());
    }

    #[test]
    fn test_template_model() {
        let model = create_template_model();
        // Test that the model has parameters and is in training mode
        assert!(model.is_training_mode());
        assert!(!model.parameters().is_empty());
    }

    #[test]
    fn test_composite_model() {
        let model = CompositeModel::new();
        assert!(model.is_training_mode());
        assert!(!model.parameters().is_empty());
    }

    #[test]
    fn test_model_forward_pass() {
        let model = SimpleCustomModel::new();
        let input = Tensor::<f32>::ones(&[1, 784]);

        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 10]);
    }
}
