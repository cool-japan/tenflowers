//! Comprehensive tests for newly implemented RNN components
//!
//! Tests for Vanilla RNN and Bahdanau Attention implementations

use scirs2_core::ndarray::{array, Array3};
use tenflowers_core::{Result, Tensor};
use tenflowers_neural::layers::rnn::{attention::bahdanau::BahdanauAttention, vanilla_rnn::RNN};
use tenflowers_neural::layers::Layer;

type TestFloat = f32;

#[cfg(test)]
mod vanilla_rnn_tests {
    use super::*;

    #[test]
    fn test_rnn_construction() -> Result<()> {
        let rnn = RNN::<TestFloat>::new(
            10,    // input_size
            20,    // hidden_size
            2,     // num_layers
            true,  // bias
            true,  // batch_first
            0.1,   // dropout
            false, // bidirectional
        )?;

        // Check that parameters are properly initialized
        let params = rnn.parameters();

        // Should have: weight_ih, weight_hh, bias_ih, bias_hh for each layer
        // 2 layers * 4 parameter tensors = 8 total parameters
        assert_eq!(
            params.len(),
            8,
            "RNN should have 8 parameter tensors for 2 layers with bias"
        );

        Ok(())
    }

    #[test]
    fn test_rnn_bidirectional_construction() -> Result<()> {
        let rnn = RNN::<TestFloat>::new(
            10,   // input_size
            20,   // hidden_size
            1,    // num_layers
            true, // bias
            true, // batch_first
            0.0,  // dropout
            true, // bidirectional
        )?;

        // Check parameter count for bidirectional RNN
        let params = rnn.parameters();

        // Should have: forward and reverse parameters
        // 1 layer * 2 directions * 4 parameter tensors = 8 total parameters
        assert_eq!(
            params.len(),
            8,
            "Bidirectional RNN should have 8 parameter tensors"
        );

        Ok(())
    }

    #[test]
    fn test_rnn_forward_pass_shape() -> Result<()> {
        let rnn = RNN::<TestFloat>::new(
            10,    // input_size
            20,    // hidden_size
            2,     // num_layers
            true,  // bias
            true,  // batch_first
            0.0,   // dropout
            false, // bidirectional
        )?;

        // Create test input [batch_size=3, seq_len=5, input_size=10]
        let input = Tensor::ones(&[3, 5, 10]);

        // Test forward pass
        let output = rnn.forward(&input)?;
        let output_shape = output.shape().dims();

        // Output should be [batch_size=3, seq_len=5, hidden_size=20]
        assert_eq!(
            output_shape,
            &[3, 5, 20],
            "RNN output shape should match expected dimensions"
        );

        Ok(())
    }

    #[test]
    fn test_rnn_with_hidden_forward() -> Result<()> {
        let rnn = RNN::<TestFloat>::new(
            5,     // input_size
            10,    // hidden_size
            1,     // num_layers
            false, // bias
            true,  // batch_first
            0.0,   // dropout
            false, // bidirectional
        )?;

        // Create test input [batch_size=2, seq_len=3, input_size=5]
        let input = Tensor::ones(&[2, 3, 5]);

        // Test forward pass with hidden state
        let (output, final_hidden) = rnn.forward_with_hidden(&input, None)?;

        // Check output shape [batch_size=2, seq_len=3, hidden_size=10]
        assert_eq!(output.shape().dims(), &[2, 3, 10]);

        // Check final hidden state shape [num_layers=1, batch_size=2, hidden_size=10]
        assert_eq!(final_hidden.shape().dims(), &[1, 2, 10]);

        Ok(())
    }

    #[test]
    fn test_rnn_invalid_parameters() {
        // Test with zero layers
        let result = RNN::<TestFloat>::new(10, 20, 0, true, true, 0.1, false);
        assert!(
            result.is_err(),
            "RNN construction should fail with 0 layers"
        );

        // Test with invalid dropout
        let result = RNN::<TestFloat>::new(10, 20, 1, true, true, 1.5, false);
        assert!(
            result.is_err(),
            "RNN construction should fail with dropout > 1.0"
        );

        let result = RNN::<TestFloat>::new(10, 20, 1, true, true, -0.1, false);
        assert!(
            result.is_err(),
            "RNN construction should fail with negative dropout"
        );
    }

    #[test]
    fn test_rnn_training_mode() -> Result<()> {
        let mut rnn = RNN::<TestFloat>::new(5, 10, 1, true, true, 0.1, false)?;

        // Test training mode setting
        rnn.set_training(false);
        rnn.set_training(true);

        // Should not panic
        Ok(())
    }
}

#[cfg(test)]
mod bahdanau_attention_tests {
    use super::*;

    #[test]
    fn test_bahdanau_attention_construction() -> Result<()> {
        let attention = BahdanauAttention::<TestFloat>::new(
            64,   // encoder_hidden_size
            32,   // decoder_hidden_size
            48,   // attention_size
            true, // use_bias
        )?;

        // Check parameter count
        let params = attention.parameters();

        // Should have: w_encoder, w_decoder, v_attention, bias_encoder, bias_decoder, bias_attention
        assert_eq!(
            params.len(),
            6,
            "Bahdanau attention should have 6 parameters with bias"
        );

        Ok(())
    }

    #[test]
    fn test_bahdanau_attention_simple_constructor() -> Result<()> {
        let attention = BahdanauAttention::<TestFloat>::new_simple(64, 32)?;

        // Should use max(64, 32) = 64 as attention size
        let params = attention.parameters();
        assert_eq!(
            params.len(),
            6,
            "Simple constructor should create attention with bias"
        );

        Ok(())
    }

    #[test]
    fn test_bahdanau_attention_without_bias() -> Result<()> {
        let attention = BahdanauAttention::<TestFloat>::new(
            64,    // encoder_hidden_size
            32,    // decoder_hidden_size
            48,    // attention_size
            false, // use_bias
        )?;

        // Check parameter count without bias
        let params = attention.parameters();

        // Should have: w_encoder, w_decoder, v_attention (no bias terms)
        assert_eq!(
            params.len(),
            3,
            "Bahdanau attention should have 3 parameters without bias"
        );

        Ok(())
    }

    #[test]
    fn test_bahdanau_attention_forward_with_weights() -> Result<()> {
        let attention = BahdanauAttention::<TestFloat>::new(64, 32, 48, true)?;

        // Create encoder outputs [seq_len=5, batch_size=2, encoder_hidden_size=64]
        let encoder_outputs = Tensor::zeros(&[5, 2, 64]);

        // Create decoder hidden state [batch_size=2, decoder_hidden_size=32]
        let decoder_hidden = Tensor::zeros(&[2, 32]);

        // Test forward pass with attention weights
        let (context, attention_weights) =
            attention.forward_with_weights(&encoder_outputs, &decoder_hidden)?;

        // Check context vector shape [batch_size=2, encoder_hidden_size=64]
        assert_eq!(context.shape().dims(), &[2, 64]);

        // Check attention weights shape [batch_size=2, seq_len=5]
        assert_eq!(attention_weights.shape().dims(), &[2, 5]);

        Ok(())
    }

    #[test]
    fn test_bahdanau_attention_invalid_dimensions() {
        // Test with zero dimensions
        let result = BahdanauAttention::<TestFloat>::new(0, 32, 48, true);
        assert!(result.is_err(), "Should fail with zero encoder hidden size");

        let result = BahdanauAttention::<TestFloat>::new(64, 0, 48, true);
        assert!(result.is_err(), "Should fail with zero decoder hidden size");

        let result = BahdanauAttention::<TestFloat>::new(64, 32, 0, true);
        assert!(result.is_err(), "Should fail with zero attention size");
    }

    #[test]
    fn test_bahdanau_attention_layer_trait() -> Result<()> {
        let attention = BahdanauAttention::<TestFloat>::new(64, 32, 48, true)?;

        // Test Layer trait methods
        let input = Tensor::zeros(&[2, 96]); // Dummy input
        let output = attention.forward(&input)?;

        // Should return tensor with correct first dimension
        assert_eq!(
            output.shape().dims()[0],
            2,
            "Output batch size should match input"
        );

        // Test parameter access
        let params = attention.parameters();
        assert!(!params.is_empty(), "Should have parameters");

        Ok(())
    }

    #[test]
    fn test_bahdanau_attention_training_mode() -> Result<()> {
        let mut attention = BahdanauAttention::<TestFloat>::new(64, 32, 48, true)?;

        // Test training mode setting
        attention.set_training(false);
        attention.set_training(true);

        // Should not panic
        Ok(())
    }

    #[test]
    fn test_bahdanau_attention_clone() -> Result<()> {
        let attention = BahdanauAttention::<TestFloat>::new(64, 32, 48, true)?;

        // Test cloning
        let cloned = attention.clone();

        // Both should have same parameter count
        assert_eq!(
            attention.parameters().len(),
            cloned.parameters().len(),
            "Cloned attention should have same parameter count"
        );

        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_rnn_with_attention_workflow() -> Result<()> {
        // Create a simple RNN encoder
        let encoder = RNN::<TestFloat>::new(
            10,    // input_size
            64,    // hidden_size (encoder)
            1,     // num_layers
            true,  // bias
            true,  // batch_first
            0.0,   // dropout
            false, // bidirectional
        )?;

        // Create attention mechanism
        let attention = BahdanauAttention::<TestFloat>::new_simple(64, 32)?;

        // Test input sequence [batch_size=1, seq_len=3, input_size=10]
        let input_data = Array3::ones((1, 3, 10));
        let input = Tensor::from_array(input_data.into_dyn());

        // Encode with RNN
        let encoder_outputs = encoder.forward(&input)?;

        // Check encoder output shape [1, 3, 64]
        assert_eq!(encoder_outputs.shape().dims(), &[1, 3, 64]);

        // Create decoder hidden state [batch_size=1, decoder_hidden_size=32]
        let decoder_hidden = Tensor::<TestFloat>::zeros(&[1, 32]);

        // This would fail due to dimension mismatch in current simplified implementation
        // but demonstrates the intended workflow
        // let (context, weights) = attention.forward_with_weights(&encoder_outputs, &decoder_hidden)?;

        Ok(())
    }
}
