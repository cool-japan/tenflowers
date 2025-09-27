//! Comprehensive tests for Mixture of Experts implementation
//!
//! Tests for enhanced MoE layer with proper gating, routing, and load balancing

use scirs2_autograd::ndarray::{array, Array3};
use tenflowers_core::{Result, Tensor};
use tenflowers_neural::layers::attention::mixture_of_experts::MixtureOfExperts;
use tenflowers_neural::layers::Layer;

type TestFloat = f32;

#[cfg(test)]
mod mixture_of_experts_tests {
    use super::*;

    #[test]
    fn test_moe_construction() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            8,    // num_experts
            2,    // num_selected (top-k)
            128,  // embed_dim
            512,  // expert_dim
            0.1,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Check parameters - should include gate weights and expert parameters
        let params = moe.parameters();

        // Gate linear weights + 8 experts * (weights + bias for each expert)
        // Exact count depends on FeedForwardNetwork implementation
        assert!(!params.is_empty(), "MoE should have parameters");

        Ok(())
    }

    #[test]
    fn test_moe_forward_pass_shape() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            4,    // num_experts
            2,    // num_selected
            64,   // embed_dim
            256,  // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Create test input [batch_size=2, seq_len=3, embed_dim=64]
        let input_data = Array3::ones((2, 3, 64));
        let input = Tensor::from_array(input_data.into_dyn());

        // Test forward pass
        let output = moe.forward(&input)?;
        let output_shape = output.shape().dims();

        // Output should maintain input dimensions [2, 3, 64]
        assert_eq!(
            output_shape,
            &[2, 3, 64],
            "MoE output should maintain input dimensions"
        );

        Ok(())
    }

    #[test]
    fn test_moe_forward_with_aux_loss() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            4,    // num_experts
            2,    // num_selected
            32,   // embed_dim
            128,  // expert_dim
            0.0,  // dropout_prob
            0.02, // load_balance_loss_coef
        )?;

        // Create test input [batch_size=1, seq_len=2, embed_dim=32]
        let input = Tensor::zeros(&[1, 2, 32]);

        // Test forward pass with auxiliary loss
        let (output, aux_loss) = moe.forward_with_aux_loss(&input)?;

        // Check output shape
        assert_eq!(output.shape().dims(), &[1, 2, 32]);

        // Check auxiliary loss is a scalar
        let aux_loss_shape = aux_loss.shape().dims();
        assert!(
            aux_loss_shape.iter().product::<usize>() == 1,
            "Auxiliary loss should be scalar"
        );

        Ok(())
    }

    #[test]
    fn test_moe_expert_selection_consistency() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            6,    // num_experts
            3,    // num_selected (top-3)
            16,   // embed_dim
            64,   // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Create deterministic input
        let input = Tensor::ones(&[2, 2, 16]);

        // Run forward pass multiple times
        let (output1, _) = moe.forward_with_aux_loss(&input)?;
        let (output2, _) = moe.forward_with_aux_loss(&input)?;

        // With same input, should get consistent outputs (in deterministic mode)
        assert_eq!(
            output1.shape().dims(),
            output2.shape().dims(),
            "Consistent input should produce consistent output shapes"
        );

        Ok(())
    }

    #[test]
    fn test_moe_different_expert_counts() -> Result<()> {
        // Test with various expert configurations
        let configs = vec![
            (2, 1),  // 2 experts, top-1
            (4, 2),  // 4 experts, top-2
            (8, 3),  // 8 experts, top-3
            (16, 4), // 16 experts, top-4
        ];

        for (num_experts, num_selected) in configs {
            let moe = MixtureOfExperts::<TestFloat>::new(
                num_experts,
                num_selected,
                32,   // embed_dim
                128,  // expert_dim
                0.0,  // dropout_prob
                0.01, // load_balance_loss_coef
            )?;

            let input = Tensor::zeros(&[1, 1, 32]);
            let output = moe.forward(&input)?;

            assert_eq!(
                output.shape().dims(),
                &[1, 1, 32],
                "Output shape should be consistent regardless of expert count"
            );
        }

        Ok(())
    }

    #[test]
    fn test_moe_load_balancing_loss_properties() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            4,   // num_experts
            2,   // num_selected
            16,  // embed_dim
            64,  // expert_dim
            0.0, // dropout_prob
            0.1, // load_balance_loss_coef (higher for testing)
        )?;

        // Test with different batch sizes
        let batch_sizes = vec![1, 2, 4];

        for batch_size in batch_sizes {
            let input = Tensor::zeros(&[batch_size, 3, 16]);
            let (_, aux_loss) = moe.forward_with_aux_loss(&input)?;

            // Auxiliary loss should be finite and non-negative
            // (In practice, we'd need to access the actual loss value to check this)
            assert!(
                aux_loss.shape().dims().iter().product::<usize>() == 1,
                "Load balancing loss should be scalar for batch size {}",
                batch_size
            );
        }

        Ok(())
    }

    #[test]
    fn test_moe_parameter_access() -> Result<()> {
        let mut moe = MixtureOfExperts::<TestFloat>::new(
            3,    // num_experts
            1,    // num_selected
            8,    // embed_dim
            32,   // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Test parameter access
        let params_len = moe.parameters().len();
        let params_mut_len = moe.parameters_mut().len();

        assert_eq!(
            params_len, params_mut_len,
            "Mutable and immutable parameter counts should match"
        );

        assert!(!moe.parameters().is_empty(), "MoE should have parameters");

        Ok(())
    }

    #[test]
    fn test_moe_training_mode() -> Result<()> {
        let mut moe = MixtureOfExperts::<TestFloat>::new(
            4,    // num_experts
            2,    // num_selected
            16,   // embed_dim
            64,   // expert_dim
            0.1,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Test training mode changes
        moe.set_training(false);
        moe.set_training(true);

        // Should not panic
        Ok(())
    }

    #[test]
    fn test_moe_clone_functionality() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            4,    // num_experts
            2,    // num_selected
            16,   // embed_dim
            64,   // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Test cloning
        let cloned_moe = moe.clone();

        // Both should have same parameter structure
        assert_eq!(
            moe.parameters().len(),
            cloned_moe.parameters().len(),
            "Cloned MoE should have same parameter count"
        );

        Ok(())
    }

    #[test]
    fn test_moe_large_sequence_handling() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            4,    // num_experts
            2,    // num_selected
            8,    // embed_dim (small for efficiency)
            32,   // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Test with longer sequences
        let input = Tensor::zeros(&[1, 10, 8]); // batch=1, seq_len=10, embed_dim=8

        let (output, aux_loss) = moe.forward_with_aux_loss(&input)?;

        // Check dimensions are preserved
        assert_eq!(
            output.shape().dims(),
            &[1, 10, 8],
            "Long sequence dimensions should be preserved"
        );

        // Auxiliary loss should still be scalar
        assert!(
            aux_loss.shape().dims().iter().product::<usize>() == 1,
            "Auxiliary loss should remain scalar for long sequences"
        );

        Ok(())
    }

    #[test]
    fn test_moe_edge_cases() -> Result<()> {
        // Test edge case: num_selected = num_experts (select all experts)
        let moe = MixtureOfExperts::<TestFloat>::new(
            3,    // num_experts
            3,    // num_selected (all experts)
            8,    // embed_dim
            16,   // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        let input = Tensor::zeros(&[1, 1, 8]);
        let output = moe.forward(&input)?;

        assert_eq!(output.shape().dims(), &[1, 1, 8]);

        // Test edge case: num_selected > num_experts (should be clamped)
        let moe2 = MixtureOfExperts::<TestFloat>::new(
            2,    // num_experts
            5,    // num_selected (more than available)
            8,    // embed_dim
            16,   // expert_dim
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        let output2 = moe2.forward(&input)?;
        assert_eq!(output2.shape().dims(), &[1, 1, 8]);

        Ok(())
    }
}

#[cfg(test)]
mod moe_performance_tests {
    use super::*;

    #[test]
    fn test_moe_with_varying_complexity() -> Result<()> {
        // Test MoE layers with different complexity levels
        let complexities = vec![
            (2, 1, 16, 32),    // Simple
            (8, 2, 64, 256),   // Medium
            (16, 4, 128, 512), // Complex (may be slow in tests)
        ];

        for (num_experts, num_selected, embed_dim, expert_dim) in complexities {
            let moe = MixtureOfExperts::<TestFloat>::new(
                num_experts,
                num_selected,
                embed_dim,
                expert_dim,
                0.0,  // dropout_prob
                0.01, // load_balance_loss_coef
            )?;

            // Small input for performance testing
            let input = Tensor::zeros(&[2, 3, embed_dim]);
            let (output, aux_loss) = moe.forward_with_aux_loss(&input)?;

            assert_eq!(output.shape().dims(), &[2, 3, embed_dim]);
            assert!(aux_loss.shape().dims().iter().product::<usize>() == 1);
        }

        Ok(())
    }

    #[test]
    fn test_moe_expert_utilization_patterns() -> Result<()> {
        let moe = MixtureOfExperts::<TestFloat>::new(
            4,   // num_experts
            1,   // num_selected (top-1 for clearer patterns)
            8,   // embed_dim
            16,  // expert_dim
            0.0, // dropout_prob
            0.1, // load_balance_loss_coef (higher to test loss computation)
        )?;

        // Test with multiple different inputs to see expert utilization
        let inputs = vec![
            Tensor::zeros(&[1, 1, 8]),
            Tensor::ones(&[1, 1, 8]),
            Tensor::zeros(&[2, 1, 8]),
        ];

        for (i, input) in inputs.iter().enumerate() {
            let (output, aux_loss) = moe.forward_with_aux_loss(input)?;

            assert_eq!(
                output.shape().dims()[2],
                8,
                "Output embed_dim should match input for case {}",
                i
            );
            assert!(
                aux_loss.shape().dims().iter().product::<usize>() == 1,
                "Auxiliary loss should be scalar for case {}",
                i
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod moe_integration_tests {
    use super::*;

    #[test]
    fn test_moe_in_transformer_like_workflow() -> Result<()> {
        // Simulate a transformer-like workflow with MoE
        let embed_dim = 32;
        let seq_len = 4;
        let batch_size = 1;

        // Create MoE layer (simulating FFN replacement in transformer)
        let moe = MixtureOfExperts::<TestFloat>::new(
            4, // num_experts
            2, // num_selected
            embed_dim,
            embed_dim * 4, // expert_dim (typical 4x expansion)
            0.1,           // dropout_prob
            0.01,          // load_balance_loss_coef
        )?;

        // Simulate attention output as input to MoE
        let attention_output = Tensor::ones(&[batch_size, seq_len, embed_dim]);

        // Apply MoE
        let (moe_output, load_balance_loss) = moe.forward_with_aux_loss(&attention_output)?;

        // Check output preserves dimensions
        assert_eq!(
            moe_output.shape().dims(),
            &[batch_size, seq_len, embed_dim],
            "MoE should preserve transformer layer dimensions"
        );

        // Load balancing loss should be available for training
        assert!(
            load_balance_loss.shape().dims().iter().product::<usize>() == 1,
            "Load balance loss should be scalar for gradient computation"
        );

        Ok(())
    }

    #[test]
    fn test_moe_memory_efficiency_patterns() -> Result<()> {
        // Test patterns that would be memory efficient in practice
        let moe = MixtureOfExperts::<TestFloat>::new(
            8,    // num_experts (moderate count)
            2,    // num_selected (sparse activation)
            64,   // embed_dim
            128,  // expert_dim (2x expansion, not 4x)
            0.0,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Test with realistic batch processing
        let batch_sizes = vec![1, 4, 8];

        for batch_size in batch_sizes {
            let input = Tensor::zeros(&[batch_size, 16, 64]); // longer sequences
            let (output, _aux_loss) = moe.forward_with_aux_loss(&input)?;

            assert_eq!(
                output.shape().dims(),
                &[batch_size, 16, 64],
                "Memory efficient processing should preserve dimensions for batch size {}",
                batch_size
            );
        }

        Ok(())
    }
}
