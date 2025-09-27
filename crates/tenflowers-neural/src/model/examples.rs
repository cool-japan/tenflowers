// Examples demonstrating the Functional API
// These are not included in the main build but serve as documentation

#[allow(dead_code)]
#[allow(clippy::module_inception)]
pub mod examples {
    use crate::layers::Dense;
    use crate::model::{FunctionalModel, FunctionalModelBuilder, Input, Node, SharedLayer};
    use tenflowers_core::{Result, Tensor};

    /// Example 1: Multi-input model (e.g., for multimodal learning)
    ///
    /// This example shows how to create a model that takes multiple inputs
    /// (like text and image features) and processes them separately before
    /// combining them for a final prediction.
    pub fn create_multi_input_model() -> Result<FunctionalModel<f32>> {
        // Define inputs
        let text_input = Input::<f32>::new_named(vec![32, 128], "text_features".to_string());
        let image_input = Input::<f32>::new_named(vec![32, 256], "image_features".to_string());

        // Create layers
        let text_dense = Dense::<f32>::new(128, 64, true);
        let image_dense = Dense::<f32>::new(256, 64, true);
        let fusion_dense = Dense::<f32>::new(128, 32, true); // 64 + 64 = 128
        let output_dense = Dense::<f32>::new(32, 1, true);

        // Build the model
        let model = FunctionalModelBuilder::new()
            .add_input(text_input.clone())
            .add_input(image_input.clone())
            .add_layer(0, Box::new(text_dense))
            .add_layer(1, Box::new(image_dense))
            .add_layer(2, Box::new(fusion_dense))
            .add_layer(3, Box::new(output_dense))
            .name("MultiInputModel".to_string());

        // Define the computation graph
        let text_processed = Node::<f32>::from_layer(0, vec![32, 64], vec![text_input.id()]);
        let image_processed = Node::<f32>::from_layer(1, vec![32, 64], vec![image_input.id()]);

        // Concatenation would happen via a custom operation
        let concatenated = Node::<f32>::new_named(
            vec![32, 128],
            vec![text_processed.id(), image_processed.id()],
            "concatenated_features".to_string(),
        );

        let fused = Node::<f32>::from_layer(2, vec![32, 32], vec![concatenated.id()]);
        let output = Node::<f32>::from_layer(3, vec![32, 1], vec![fused.id()]);
        let output_id = output.id();

        // Build model with all intermediate nodes, then set actual outputs
        let mut built_model = model.build(vec![
            text_processed,
            image_processed,
            concatenated,
            fused,
            output,
        ])?;

        built_model.set_outputs(vec![output_id]);
        Ok(built_model)
    }

    /// Example 2: Multi-output model (e.g., for multi-task learning)
    ///
    /// This example shows how to create a model with shared feature extraction
    /// and multiple task-specific heads (like classification + regression).
    pub fn create_multi_output_model() -> Result<FunctionalModel<f32>> {
        // Define input
        let input = Input::<f32>::new_named(vec![32, 256], "features".to_string());

        // Create layers
        let shared_dense1 = Dense::<f32>::new(256, 128, true);
        let shared_dense2 = Dense::<f32>::new(128, 64, true);

        // Task-specific heads
        let classification_head = Dense::<f32>::new(64, 10, true); // 10 classes
        let regression_head = Dense::<f32>::new(64, 1, true); // 1 continuous value

        // Build the model
        let model = FunctionalModelBuilder::new()
            .add_input(input.clone())
            .add_layer(0, Box::new(shared_dense1))
            .add_layer(1, Box::new(shared_dense2))
            .add_layer(2, Box::new(classification_head))
            .add_layer(3, Box::new(regression_head))
            .name("MultiOutputModel".to_string());

        // Define the computation graph
        let shared1 = Node::<f32>::from_layer(0, vec![32, 128], vec![input.id()]);
        let shared2 = Node::<f32>::from_layer(1, vec![32, 64], vec![shared1.id()]);

        // Multiple outputs from the same shared features
        let classification_output = Node::<f32>::from_layer(2, vec![32, 10], vec![shared2.id()]);
        let regression_output = Node::<f32>::from_layer(3, vec![32, 1], vec![shared2.id()]);
        let classification_id = classification_output.id();
        let regression_id = regression_output.id();

        // Build model with all intermediate nodes, then set actual outputs
        let mut built_model = model.build(vec![
            shared1,
            shared2,
            classification_output,
            regression_output,
        ])?;

        built_model.set_outputs(vec![classification_id, regression_id]);
        Ok(built_model)
    }

    /// Example 3: Shared layer model (e.g., Siamese network)
    ///
    /// This example shows how to use the same layer multiple times,
    /// useful for architectures like Siamese networks where you want
    /// to process multiple inputs with identical transformations.
    pub fn create_shared_layer_model() -> Result<FunctionalModel<f32>> {
        // Define inputs (e.g., two images for similarity comparison)
        let input1 = Input::<f32>::new_named(vec![32, 784], "image1".to_string());
        let input2 = Input::<f32>::new_named(vec![32, 784], "image2".to_string());

        // Create shared encoder layers
        let encoder_dense1 = Dense::<f32>::new(784, 256, true);
        let encoder_dense2 = Dense::<f32>::new(256, 64, true);

        let shared_encoder1 =
            SharedLayer::new_named(Box::new(encoder_dense1), "shared_encoder1".to_string());
        let shared_encoder2 =
            SharedLayer::new_named(Box::new(encoder_dense2), "shared_encoder2".to_string());

        // Similarity computation layer
        let similarity_dense = Dense::<f32>::new(128, 1, true); // 64 + 64 = 128

        // Get the layer IDs before moving into builder
        let encoder1_id = shared_encoder1.id();
        let encoder2_id = shared_encoder2.id();

        // Build the model
        let model = FunctionalModelBuilder::new()
            .add_input(input1.clone())
            .add_input(input2.clone())
            .add_shared_layer(shared_encoder1)
            .add_shared_layer(shared_encoder2)
            .add_layer(0, Box::new(similarity_dense))
            .name("SiameseModel".to_string());

        // Define the computation graph
        // Process input1 through shared encoders
        let encoded1_1 = Node::<f32>::from_layer(encoder1_id, vec![32, 256], vec![input1.id()]);
        let encoded1_2 = Node::<f32>::from_layer(encoder2_id, vec![32, 64], vec![encoded1_1.id()]);

        // Process input2 through the same shared encoders
        let encoded2_1 = Node::<f32>::from_layer(encoder1_id, vec![32, 256], vec![input2.id()]);
        let encoded2_2 = Node::<f32>::from_layer(encoder2_id, vec![32, 64], vec![encoded2_1.id()]);

        // Combine encodings (concatenation via custom operation)
        let combined = Node::<f32>::new_named(
            vec![32, 128],
            vec![encoded1_2.id(), encoded2_2.id()],
            "combined_encodings".to_string(),
        );

        // Final similarity score
        let similarity = Node::<f32>::from_layer(0, vec![32, 1], vec![combined.id()]);
        let similarity_id = similarity.id();

        // Build model with all nodes in dependency order, then set actual outputs
        let mut built_model = model.build(vec![
            encoded1_1, encoded1_2, encoded2_1, encoded2_2, combined, similarity,
        ])?;

        // Set the actual output to be just the similarity node
        built_model.set_outputs(vec![similarity_id]);
        Ok(built_model)
    }

    /// Example 4: ResNet-style skip connection model
    ///
    /// This example shows how to implement skip connections,
    /// demonstrating more complex graph topologies.
    pub fn create_skip_connection_model() -> Result<FunctionalModel<f32>> {
        // Define input
        let input = Input::<f32>::new_named(vec![32, 64], "input".to_string());

        // Create layers
        let dense1 = Dense::<f32>::new(64, 64, true);
        let dense2 = Dense::<f32>::new(64, 64, true);

        // Build the model with custom add operation for skip connection
        let model = FunctionalModelBuilder::new()
            .add_input(input.clone())
            .add_layer(0, Box::new(dense1))
            .add_layer(1, Box::new(dense2))
            .add_custom_op(
                2,
                Box::new(|inputs: &[&Tensor<f32>]| -> Result<Tensor<f32>> {
                    // Custom addition operation for skip connection
                    if inputs.len() != 2 {
                        return Err(tenflowers_core::TensorError::invalid_argument(
                            "Add operation requires exactly 2 inputs".to_string(),
                        ));
                    }
                    inputs[0].add(inputs[1])
                }),
            )
            .name("SkipConnectionModel".to_string());

        // Define the computation graph with skip connection
        let x1 = Node::<f32>::from_layer(0, vec![32, 64], vec![input.id()]);
        let x2 = Node::<f32>::from_layer(1, vec![32, 64], vec![x1.id()]);

        // Skip connection: add input to the output of the second layer
        let output = Node::<f32>::from_layer(2, vec![32, 64], vec![input.id(), x2.id()]);
        let output_id = output.id();

        // Build model with all intermediate nodes, then set actual outputs
        let mut built_model = model.build(vec![x1, x2, output])?;

        built_model.set_outputs(vec![output_id]);
        Ok(built_model)
    }

    /// Example usage demonstrating forward pass with multiple inputs
    pub fn example_usage() -> Result<()> {
        // Create a multi-input model
        let model = create_multi_input_model()?;

        // Create sample inputs
        let text_features = Tensor::<f32>::ones(&[32, 128]);
        let image_features = Tensor::<f32>::ones(&[32, 256]);

        // Forward pass with multiple inputs
        let outputs = model.forward_multi(&[&text_features, &image_features])?;

        println!(
            "Multi-input model output shape: {:?}",
            outputs[0].shape().dims()
        );

        // Create a multi-output model
        let model = create_multi_output_model()?;

        // Create sample input
        let input = Tensor::<f32>::ones(&[32, 256]);

        // Forward pass (single input, multiple outputs)
        let outputs = model.forward_multi(&[&input])?;

        println!("Multi-output model outputs:");
        println!("  Classification shape: {:?}", outputs[0].shape().dims());
        println!("  Regression shape: {:?}", outputs[1].shape().dims());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::examples::*;

    #[test]
    fn test_multi_input_model_creation() {
        let model = create_multi_input_model().unwrap();
        assert_eq!(model.num_inputs(), 2);
        assert_eq!(model.num_outputs(), 1);
        assert_eq!(model.name(), Some("MultiInputModel"));
    }

    #[test]
    fn test_multi_output_model_creation() {
        let model = create_multi_output_model().unwrap();
        assert_eq!(model.num_inputs(), 1);
        assert_eq!(model.num_outputs(), 2);
        assert_eq!(model.name(), Some("MultiOutputModel"));
    }

    #[test]
    fn test_shared_layer_model_creation() {
        let model = create_shared_layer_model().unwrap();
        assert_eq!(model.num_inputs(), 2);
        assert_eq!(model.num_outputs(), 1);
        assert_eq!(model.name(), Some("SiameseModel"));
    }

    #[test]
    fn test_skip_connection_model_creation() {
        let model = create_skip_connection_model().unwrap();
        assert_eq!(model.num_inputs(), 1);
        assert_eq!(model.num_outputs(), 1);
        assert_eq!(model.name(), Some("SkipConnectionModel"));
    }
}
