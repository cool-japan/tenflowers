//! Metal Performance Shaders Integration for Neural Networks
//!
//! This module provides comprehensive MPS-based neural network operations
//! with optimized training and inference pipelines.

use super::types::ActivationType;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{Result, Tensor, TensorError};
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal;
use std::collections::HashMap;

/// Layer types for neural network operations
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
    Convolution,
    BatchNorm,
    LayerNorm,
    Activation(ActivationType),
}

/// Layer configuration for MPS operations
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub parameters: HashMap<String, Vec<f32>>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// MPS-based neural network operations
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
pub struct MPSNeuralOps {
    device: metal::Device,
    command_queue: metal::CommandQueue,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MPSNeuralOps {
    /// Create a new MPS neural operations instance
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or_else(|| {
            TensorError::device_error_simple("No Metal device available".to_string())
        })?;
        let command_queue = device.new_command_queue();

        Ok(MPSNeuralOps {
            device,
            command_queue,
        })
    }

    /// Execute optimized neural network inference using MPS
    pub fn execute_inference(
        &mut self,
        layers: &[LayerConfig],
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        // Chain MPS operations for optimal inference performance
        let mut current_output = input.clone();

        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();

        for layer in layers.iter() {
            match &layer.layer_type {
                LayerType::Dense => {
                    // Execute dense layer using optimized matrix multiplication
                    if let (Some(weights), Some(bias)) = (
                        layer.parameters.get("weights"),
                        layer.parameters.get("bias"),
                    ) {
                        // Create weight tensor (simplified - assumes proper shape)
                        let weight_shape = vec![
                            weights.len() / current_output.shape()[1],
                            current_output.shape()[1],
                        ];
                        let weight_tensor = Tensor::from_vec(weights.clone(), &weight_shape)?;

                        // Matrix multiplication: output = input * weights^T
                        current_output =
                            self.execute_matrix_multiply(&current_output, &weight_tensor)?;

                        // Add bias if available
                        if !bias.is_empty() {
                            current_output = self.add_bias(&current_output, bias)?;
                        }
                    }
                }
                LayerType::Convolution => {
                    // Execute convolution using MPS-optimized kernels
                    if let (Some(weights), Some(bias)) = (
                        layer.parameters.get("weights"),
                        layer.parameters.get("bias"),
                    ) {
                        // Simplified convolution parameters (in practice would be more sophisticated)
                        let stride = [1, 1];
                        let padding = [0, 0];

                        // Create weight tensor for convolution
                        let weight_shape =
                            self.infer_conv_weight_shape(&current_output, weights.len())?;
                        let weight_tensor = Tensor::from_vec(weights.clone(), &weight_shape)?;

                        let bias_tensor = if !bias.is_empty() {
                            Some(Tensor::from_vec(bias.clone(), &[bias.len()])?)
                        } else {
                            None
                        };

                        current_output = self.execute_convolution(
                            &current_output,
                            &weight_tensor,
                            bias_tensor.as_ref(),
                            stride,
                            padding,
                        )?;
                    }
                }
                LayerType::BatchNorm => {
                    // Execute batch normalization using MPS
                    if let (Some(scale), Some(offset), Some(mean), Some(variance)) = (
                        layer.parameters.get("scale"),
                        layer.parameters.get("offset"),
                        layer.parameters.get("running_mean"),
                        layer.parameters.get("running_var"),
                    ) {
                        current_output = self.execute_batch_norm(
                            &current_output,
                            scale,
                            offset,
                            mean,
                            variance,
                        )?;
                    }
                }
                LayerType::LayerNorm => {
                    // Execute layer normalization
                    if let (Some(gamma), Some(beta)) =
                        (layer.parameters.get("gamma"), layer.parameters.get("beta"))
                    {
                        current_output = self.execute_layer_norm(
                            &current_output,
                            gamma,
                            beta,
                            1e-5, // Default epsilon
                        )?;
                    }
                }
                LayerType::Activation(activation_type) => {
                    // Execute fused activation functions
                    current_output = self.execute_activation(&current_output, *activation_type)?;
                }
            }
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(current_output)
    }

    /// Execute optimized training forward pass
    pub fn execute_training_forward(
        &mut self,
        layers: &[LayerConfig],
        input: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Vec<Tensor<f32>>)> {
        // HONEST ERROR: none of the layer types below have a real Metal MPS
        // training-forward kernel implemented yet. Previously each arm either
        // filled a freshly shaped tensor with `vec![0.0f32; ...]` (Dense,
        // Convolution) or silently passed `current_output` straight through
        // while claiming to have normalized/activated it (BatchNorm, LayerNorm,
        // Activation) -- both are fabrications, since callers receive a tensor
        // labeled as that layer's output which no kernel ever touched. Every
        // layer type fails the same way, so only the first configured layer
        // needs inspecting to report a specific, honest error; an empty
        // `layers` list has nothing to run and is a genuine (non-fabricated)
        // identity result.
        if let Some((layer_idx, layer)) = layers.iter().enumerate().next() {
            return Err(match &layer.layer_type {
                LayerType::Dense => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training forward: dense layer {} GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::Convolution => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training forward: convolution layer {} GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::BatchNorm => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training forward: batch norm layer {} GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::LayerNorm => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training forward: layer norm layer {} GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::Activation(activation_type) => {
                    TensorError::unsupported_operation_simple(format!(
                        "Metal MPS training forward: activation layer {} ({:?}) GPU kernel dispatch not implemented; result would be fabricated",
                        layer_idx, activation_type
                    ))
                }
            });
        }

        // No layers configured: the forward pass is trivially the input
        // itself, a real (non-fabricated) result.
        Ok((input.clone(), vec![input.clone()]))
    }

    /// Execute optimized training backward pass
    pub fn execute_training_backward(
        &mut self,
        layers: &[LayerConfig],
        gradients: &Tensor<f32>,
        activations: &[Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>> {
        // HONEST ERROR: none of the layer types below have a real Metal MPS
        // training-backward (gradient) kernel implemented yet. Previously
        // every gradient (weight/bias/input for Dense, weight/input for
        // Convolution, scale/offset for BatchNorm, gamma/beta for LayerNorm,
        // and the activation gradient for ReLU/GELU/other) was filled with
        // `vec![0.0f32; ...]` (host zeros) instead of being computed from
        // `prev_activation` / `current_gradient` -- a fabrication. Backprop
        // visits layers in reverse, so only the last layer -- the first one
        // backprop would touch, if any -- needs inspecting to report a
        // specific, honest error; an empty `layers` list has no gradients to
        // compute, which is a genuine (non-fabricated) empty result.
        let _ = (gradients, activations);

        if let Some((layer_idx, layer)) = layers.iter().enumerate().next_back() {
            return Err(match &layer.layer_type {
                LayerType::Dense => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training backward: dense layer {} weight/bias/input gradient GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::Convolution => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training backward: convolution layer {} weight/input gradient GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::BatchNorm => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training backward: batch norm layer {} scale/offset gradient GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::LayerNorm => TensorError::unsupported_operation_simple(format!(
                    "Metal MPS training backward: layer norm layer {} gamma/beta gradient GPU kernel dispatch not implemented; result would be fabricated",
                    layer_idx
                )),
                LayerType::Activation(activation_type) => {
                    TensorError::unsupported_operation_simple(format!(
                        "Metal MPS training backward: activation layer {} ({:?}) gradient GPU kernel dispatch not implemented; result would be fabricated",
                        layer_idx, activation_type
                    ))
                }
            });
        }

        // No layers configured: there are no gradients to compute, which is a
        // real (non-fabricated) empty result.
        Ok(Vec::new())
    }

    // Helper methods for MPS operations

    fn execute_matrix_multiply(&mut self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        // HONEST ERROR: this never dispatched a GPU kernel -- it only computed
        // `output_shape` from `a`/`b` and filled it with `vec![0.0f32; ...]`
        // (host zeros), which `execute_inference`'s Dense arm would then hand
        // back to callers as if it were the real `a @ b` result. Fail loudly
        // instead of faking a matmul.
        let _ = (a, b);
        Err(TensorError::unsupported_operation_simple(
            "Metal MPS matrix multiply: GPU kernel dispatch + host readback not implemented; result would be fabricated"
                .to_string(),
        ))
    }

    fn add_bias(&mut self, tensor: &Tensor<f32>, bias: &[f32]) -> Result<Tensor<f32>> {
        // HONEST ERROR: this discarded `tensor`'s real values entirely and
        // returned `vec![0.0f32; tensor.numel()]` -- not even `tensor + bias`,
        // just zeros. Fail loudly instead of silently dropping the matmul
        // output this is meant to add a bias to.
        let _ = (tensor, bias);
        Err(TensorError::unsupported_operation_simple(
            "Metal MPS bias addition: GPU kernel dispatch + host readback not implemented; result would be fabricated"
                .to_string(),
        ))
    }

    fn infer_conv_weight_shape(
        &self,
        input: &Tensor<impl Clone>,
        weight_len: usize,
    ) -> Result<Vec<usize>> {
        // Simplified weight shape inference
        let input_shape = input.shape();
        if input_shape.len() == 4 {
            let out_channels = weight_len / (input_shape[1] * 9); // Assume 3x3 kernel
            Ok(vec![out_channels, input_shape[1], 3, 3])
        } else {
            Err(TensorError::invalid_operation_simple(
                "Invalid input shape for convolution".to_string(),
            ))
        }
    }

    fn execute_convolution(
        &mut self,
        input: &Tensor<f32>,
        weights: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor<f32>> {
        // HONEST ERROR: no kernel was ever dispatched here -- this only
        // derived an output shape from `input`/`weights` and filled it with
        // host zeros, silently ignoring `bias`, `stride`, and `padding`
        // entirely.
        let _ = (input, weights, bias, stride, padding);
        Err(TensorError::unsupported_operation_simple(
            "Metal MPS convolution: GPU kernel dispatch + host readback not implemented; result would be fabricated"
                .to_string(),
        ))
    }

    fn execute_batch_norm(
        &mut self,
        input: &Tensor<f32>,
        scale: &[f32],
        offset: &[f32],
        mean: &[f32],
        variance: &[f32],
    ) -> Result<Tensor<f32>> {
        // HONEST ERROR: `scale`, `offset`, `mean`, and `variance` were
        // accepted but never read -- this just zero-filled a tensor shaped
        // like `input`.
        let _ = (input, scale, offset, mean, variance);
        Err(TensorError::unsupported_operation_simple(
            "Metal MPS batch norm: GPU kernel dispatch + host readback not implemented; result would be fabricated"
                .to_string(),
        ))
    }

    fn execute_layer_norm(
        &mut self,
        input: &Tensor<f32>,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Tensor<f32>> {
        // HONEST ERROR: `gamma`, `beta`, and `eps` were accepted but never
        // read -- this just zero-filled a tensor shaped like `input`.
        let _ = (input, gamma, beta, eps);
        Err(TensorError::unsupported_operation_simple(
            "Metal MPS layer norm: GPU kernel dispatch + host readback not implemented; result would be fabricated"
                .to_string(),
        ))
    }

    fn execute_activation(
        &mut self,
        input: &Tensor<f32>,
        activation_type: ActivationType,
    ) -> Result<Tensor<f32>> {
        // HONEST ERROR: `activation_type` was accepted but never applied --
        // this just zero-filled a tensor shaped like `input`.
        let _ = input;
        Err(TensorError::unsupported_operation_simple(format!(
            "Metal MPS activation ({:?}): GPU kernel dispatch + host readback not implemented; result would be fabricated",
            activation_type
        )))
    }
}

/// Utility functions for MPS integration
#[cfg(all(target_os = "macos", feature = "metal"))]
impl LayerConfig {
    /// Create a new dense layer configuration
    pub fn dense(input_size: usize, output_size: usize) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("weights".to_string(), vec![0.0; input_size * output_size]);
        parameters.insert("bias".to_string(), vec![0.0; output_size]);

        LayerConfig {
            layer_type: LayerType::Dense,
            parameters,
            input_shape: vec![input_size],
            output_shape: vec![output_size],
        }
    }

    /// Create a new convolution layer configuration
    pub fn conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        input_size: (usize, usize),
    ) -> Self {
        let mut parameters = HashMap::new();
        let weight_size = out_channels * in_channels * kernel_size.0 * kernel_size.1;
        parameters.insert("weights".to_string(), vec![0.0; weight_size]);
        parameters.insert("bias".to_string(), vec![0.0; out_channels]);

        LayerConfig {
            layer_type: LayerType::Convolution,
            parameters,
            input_shape: vec![in_channels, input_size.0, input_size.1],
            output_shape: vec![out_channels, input_size.0, input_size.1],
        }
    }

    /// Create a new batch normalization layer configuration
    pub fn batch_norm(num_features: usize) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("scale".to_string(), vec![1.0; num_features]);
        parameters.insert("offset".to_string(), vec![0.0; num_features]);
        parameters.insert("running_mean".to_string(), vec![0.0; num_features]);
        parameters.insert("running_var".to_string(), vec![1.0; num_features]);

        LayerConfig {
            layer_type: LayerType::BatchNorm,
            parameters,
            input_shape: vec![num_features],
            output_shape: vec![num_features],
        }
    }

    /// Create a new layer normalization configuration
    pub fn layer_norm(normalized_shape: Vec<usize>) -> Self {
        let num_elements = normalized_shape.iter().product();
        let mut parameters = HashMap::new();
        parameters.insert("gamma".to_string(), vec![1.0; num_elements]);
        parameters.insert("beta".to_string(), vec![0.0; num_elements]);

        LayerConfig {
            layer_type: LayerType::LayerNorm,
            parameters,
            input_shape: normalized_shape.clone(),
            output_shape: normalized_shape,
        }
    }

    /// Create a new activation layer configuration
    pub fn activation(activation_type: ActivationType, shape: Vec<usize>) -> Self {
        LayerConfig {
            layer_type: LayerType::Activation(activation_type),
            parameters: HashMap::new(),
            input_shape: shape.clone(),
            output_shape: shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_mps_neural_ops_creation() {
        let result = MPSNeuralOps::new();
        // Test should pass on macOS with Metal support
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("No Metal device"));
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_layer_config_creation() {
        let dense_config = LayerConfig::dense(128, 64);
        assert!(matches!(dense_config.layer_type, LayerType::Dense));
        assert_eq!(dense_config.input_shape, vec![128]);
        assert_eq!(dense_config.output_shape, vec![64]);

        let conv_config = LayerConfig::conv2d(3, 64, (3, 3), (224, 224));
        assert!(matches!(conv_config.layer_type, LayerType::Convolution));
        assert_eq!(conv_config.input_shape, vec![3, 224, 224]);
        assert_eq!(conv_config.output_shape, vec![64, 224, 224]);

        let bn_config = LayerConfig::batch_norm(64);
        assert!(matches!(bn_config.layer_type, LayerType::BatchNorm));
        assert_eq!(bn_config.input_shape, vec![64]);
        assert_eq!(bn_config.output_shape, vec![64]);
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn test_mps_not_available() {
        // On non-macOS platforms, MPS integration is not available
        // This test ensures the module compiles correctly on all platforms
        assert!(true);
    }

    // ---------------------------------------------------------------------
    // Honest-error regression tests.
    //
    // Every method below used to silently return a fabricated all-zero
    // `Ok(Tensor)` (or, for BatchNorm/LayerNorm/Activation in the forward
    // pass, silently pass the input through unchanged while claiming to have
    // normalized/activated it). Each test here asserts the method now
    // returns a specific, honest `Err` instead -- never a fabricated `Ok`.
    // ---------------------------------------------------------------------

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_matrix_multiply_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
                .expect("test: 2x2 tensor construction should succeed");
            let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2])
                .expect("test: 2x2 tensor construction should succeed");
            let err = ops.execute_matrix_multiply(&a, &b).expect_err(
                "matrix multiply has no GPU readback; it must honestly error, not fabricate zeros",
            );
            let msg = err.to_string();
            assert!(msg.contains("matrix multiply"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_add_bias_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
                .expect("test: 1D tensor construction should succeed");
            let bias = vec![0.5f32, 0.5, 0.5];
            let err = ops
                .add_bias(&tensor, &bias)
                .expect_err("add_bias must honestly error, not silently zero the matmul output");
            let msg = err.to_string();
            assert!(msg.contains("bias"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_convolution_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![0.0f32; 3 * 8 * 8], &[1, 3, 8, 8])
                .expect("test: NCHW input construction should succeed");
            let weights = Tensor::from_vec(vec![0.0f32; 4 * 3 * 3 * 3], &[4, 3, 3, 3])
                .expect("test: conv weight construction should succeed");
            let err = ops
                .execute_convolution(&input, &weights, None, [1, 1], [0, 0])
                .expect_err("convolution must honestly error, not fabricate zeros");
            let msg = err.to_string();
            assert!(msg.contains("convolution"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_batch_norm_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4])
                .expect("test: 1D tensor construction should succeed");
            let scale = vec![1.0f32; 4];
            let offset = vec![0.0f32; 4];
            let mean = vec![0.0f32; 4];
            let variance = vec![1.0f32; 4];
            let err = ops
                .execute_batch_norm(&input, &scale, &offset, &mean, &variance)
                .expect_err("batch norm must honestly error, not fabricate zeros");
            let msg = err.to_string();
            assert!(msg.contains("batch norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_layer_norm_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4])
                .expect("test: 1D tensor construction should succeed");
            let gamma = vec![1.0f32; 4];
            let beta = vec![0.0f32; 4];
            let err = ops
                .execute_layer_norm(&input, &gamma, &beta, 1e-5)
                .expect_err("layer norm must honestly error, not fabricate zeros");
            let msg = err.to_string();
            assert!(msg.contains("layer norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_activation_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, -1.0, 2.0, -2.0], &[4])
                .expect("test: 1D tensor construction should succeed");
            let err = ops
                .execute_activation(&input, ActivationType::ReLU)
                .expect_err("activation must honestly error, not fabricate zeros");
            let msg = err.to_string();
            assert!(msg.contains("activation"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_forward_dense_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::dense(2, 3)];
            let err = ops.execute_training_forward(&layers, &input).expect_err(
                "training forward must honestly error for dense layers, not fabricate zeros",
            );
            let msg = err.to_string();
            assert!(msg.contains("dense"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_forward_convolution_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![0.0f32; 3 * 8 * 8], &[1, 3, 8, 8])
                .expect("test: NCHW input construction should succeed");
            let layers = vec![LayerConfig::conv2d(3, 4, (3, 3), (8, 8))];
            let err = ops
                .execute_training_forward(&layers, &input)
                .expect_err("training forward must honestly error for convolution layers");
            let msg = err.to_string();
            assert!(msg.contains("convolution"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_forward_batch_norm_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::batch_norm(4)];
            let err = ops
                .execute_training_forward(&layers, &input)
                .expect_err("training forward must honestly error for batch norm layers");
            let msg = err.to_string();
            assert!(msg.contains("batch norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_forward_layer_norm_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::layer_norm(vec![4])];
            let err = ops
                .execute_training_forward(&layers, &input)
                .expect_err("training forward must honestly error for layer norm layers");
            let msg = err.to_string();
            assert!(msg.contains("layer norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_forward_activation_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, -1.0], &[2])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::activation(ActivationType::GELU, vec![2])];
            let err = ops
                .execute_training_forward(&layers, &input)
                .expect_err("training forward must honestly error for activation layers");
            let msg = err.to_string();
            assert!(msg.contains("activation"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_forward_empty_layers_is_identity() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
                .expect("test: input tensor construction should succeed");
            let (output, activations) = ops.execute_training_forward(&[], &input).expect(
                "an empty layer list is a genuine identity pass-through, not a fabrication",
            );
            assert_eq!(output.shape().dims(), input.shape().dims());
            assert_eq!(activations.len(), 1);
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_backward_dense_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let gradients = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], &[1, 3])
                .expect("test: gradient tensor construction should succeed");
            let activation = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2])
                .expect("test: activation tensor construction should succeed");
            let layers = vec![LayerConfig::dense(2, 3)];
            let activations = vec![activation.clone(), activation];
            let err = ops
                .execute_training_backward(&layers, &gradients, &activations)
                .expect_err("training backward must honestly error for dense layers");
            let msg = err.to_string();
            assert!(msg.contains("dense"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_backward_convolution_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let gradients = Tensor::from_vec(vec![0.0f32; 4 * 8 * 8], &[1, 4, 8, 8])
                .expect("test: gradient tensor construction should succeed");
            let activation = Tensor::from_vec(vec![0.0f32; 3 * 8 * 8], &[1, 3, 8, 8])
                .expect("test: activation tensor construction should succeed");
            let layers = vec![LayerConfig::conv2d(3, 4, (3, 3), (8, 8))];
            let activations = vec![activation.clone(), activation];
            let err = ops
                .execute_training_backward(&layers, &gradients, &activations)
                .expect_err("training backward must honestly error for convolution layers");
            let msg = err.to_string();
            assert!(msg.contains("convolution"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_backward_batch_norm_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let gradients = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: gradient tensor construction should succeed");
            let activation = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: activation tensor construction should succeed");
            let layers = vec![LayerConfig::batch_norm(4)];
            let activations = vec![activation.clone(), activation];
            let err = ops
                .execute_training_backward(&layers, &gradients, &activations)
                .expect_err("training backward must honestly error for batch norm layers");
            let msg = err.to_string();
            assert!(msg.contains("batch norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_backward_layer_norm_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let gradients = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: gradient tensor construction should succeed");
            let activation = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: activation tensor construction should succeed");
            let layers = vec![LayerConfig::layer_norm(vec![4])];
            let activations = vec![activation.clone(), activation];
            let err = ops
                .execute_training_backward(&layers, &gradients, &activations)
                .expect_err("training backward must honestly error for layer norm layers");
            let msg = err.to_string();
            assert!(msg.contains("layer norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_backward_activation_errors_honestly() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let gradients = Tensor::from_vec(vec![1.0f32, -1.0], &[2])
                .expect("test: gradient tensor construction should succeed");
            let activation = Tensor::from_vec(vec![1.0f32, -1.0], &[2])
                .expect("test: activation tensor construction should succeed");
            let layers = vec![LayerConfig::activation(ActivationType::ReLU, vec![2])];
            let activations = vec![activation.clone(), activation];
            let err = ops
                .execute_training_backward(&layers, &gradients, &activations)
                .expect_err("training backward must honestly error for activation layers");
            let msg = err.to_string();
            assert!(msg.contains("activation"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_training_backward_empty_layers_is_empty_ok() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let gradients = Tensor::from_vec(vec![1.0f32, 2.0], &[2])
                .expect("test: gradient tensor construction should succeed");
            let activations: Vec<Tensor<f32>> = Vec::new();
            let result = ops
                .execute_training_backward(&[], &gradients, &activations)
                .expect("an empty layer list has no gradients to compute, a real empty result");
            assert!(result.is_empty());
        }
    }

    // ---------------------------------------------------------------------
    // `execute_inference` propagation regression tests.
    //
    // `execute_inference` itself performs no fabrication -- it dispatches to
    // the helper methods above via `?`. These tests confirm that dispatch
    // still propagates the helpers' honest errors instead of swallowing them
    // (e.g. via `.ok()` / `.unwrap_or(...)`) into a fabricated default.
    // ---------------------------------------------------------------------

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_inference_propagates_dense_layer_error() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::dense(2, 3)];
            let err = ops
                .execute_inference(&layers, &input)
                .expect_err("execute_inference must propagate the dense layer's honest error");
            let msg = err.to_string();
            assert!(msg.contains("matrix multiply"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_inference_propagates_convolution_layer_error() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![0.0f32; 3 * 8 * 8], &[1, 3, 8, 8])
                .expect("test: NCHW input construction should succeed");
            let layers = vec![LayerConfig::conv2d(3, 4, (3, 3), (8, 8))];
            let err = ops.execute_inference(&layers, &input).expect_err(
                "execute_inference must propagate the convolution layer's honest error",
            );
            let msg = err.to_string();
            assert!(msg.contains("convolution"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_inference_propagates_batch_norm_layer_error() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::batch_norm(4)];
            let err = ops
                .execute_inference(&layers, &input)
                .expect_err("execute_inference must propagate the batch norm layer's honest error");
            let msg = err.to_string();
            assert!(msg.contains("batch norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_inference_propagates_layer_norm_layer_error() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32; 4], &[4])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::layer_norm(vec![4])];
            let err = ops
                .execute_inference(&layers, &input)
                .expect_err("execute_inference must propagate the layer norm layer's honest error");
            let msg = err.to_string();
            assert!(msg.contains("layer norm"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_execute_inference_propagates_activation_layer_error() {
        if let Ok(mut ops) = MPSNeuralOps::new() {
            let input = Tensor::from_vec(vec![1.0f32, -1.0], &[2])
                .expect("test: input tensor construction should succeed");
            let layers = vec![LayerConfig::activation(ActivationType::ReLU, vec![2])];
            let err = ops
                .execute_inference(&layers, &input)
                .expect_err("execute_inference must propagate the activation layer's honest error");
            let msg = err.to_string();
            assert!(msg.contains("activation"), "unexpected message: {msg}");
            assert!(msg.contains("fabricated"), "unexpected message: {msg}");
        }
    }
}
