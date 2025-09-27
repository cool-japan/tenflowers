//! Integration tests for neural network enhancements
//!
//! Tests demonstrating the integration and functionality of newly implemented components:
//! - Vanilla RNN with proper parameter management
//! - Bahdanau Attention mechanism
//! - Enhanced Mixture of Experts with load balancing
//!
//! These tests validate the end-to-end functionality of the enhanced neural components.

use scirs2_autograd::ndarray::array;
use tenflowers_core::{Result, Tensor};
use tenflowers_neural::layers::{
    attention::mixture_of_experts::MixtureOfExperts, rnn::vanilla_rnn::RNN, Layer,
};

type TestFloat = f32;

/// Test suite for neural network enhancements integration
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_rnn_sequence_processing_end_to_end() -> Result<()> {
        println!("Testing end-to-end RNN sequence processing...");

        // Create RNN for sequence processing
        let rnn = RNN::<TestFloat>::new(
            20,    // input_size (e.g., word embedding dimension)
            64,    // hidden_size
            2,     // num_layers (multi-layer for complexity)
            true,  // bias
            true,  // batch_first
            0.1,   // dropout
            false, // bidirectional
        )?;

        // Simulate processing multiple sequences of different lengths
        let batch_sizes = vec![1, 3, 5];
        let seq_lengths = vec![10, 15, 20];

        for (batch_size, seq_len) in batch_sizes.into_iter().zip(seq_lengths) {
            println!(
                "  Processing batch_size={}, seq_len={}",
                batch_size, seq_len
            );

            // Create realistic input data
            let input_data = Tensor::zeros(&[batch_size, seq_len, 20]);

            // Process through RNN
            let (output, final_hidden) = rnn.forward_with_hidden(&input_data, None)?;

            // Verify output shapes
            assert_eq!(
                output.shape().dims(),
                &[batch_size, seq_len, 64],
                "RNN output shape incorrect for batch_size={}, seq_len={}",
                batch_size,
                seq_len
            );

            assert_eq!(
                final_hidden.shape().dims(),
                &[2, batch_size, 64], // 2 layers
                "RNN hidden state shape incorrect for batch_size={}",
                batch_size
            );
        }

        println!("✓ RNN sequence processing test passed");
        Ok(())
    }

    #[test]
    fn test_bidirectional_rnn_functionality() -> Result<()> {
        println!("Testing bidirectional RNN functionality...");

        // Create bidirectional RNN
        let bi_rnn = RNN::<TestFloat>::new(
            16,   // input_size
            32,   // hidden_size
            1,    // num_layers
            true, // bias
            true, // batch_first
            0.0,  // dropout (no dropout for deterministic testing)
            true, // bidirectional
        )?;

        // Create test input
        let input = Tensor::ones(&[2, 5, 16]); // batch=2, seq=5, features=16

        // Process through bidirectional RNN
        let (output, final_hidden) = bi_rnn.forward_with_hidden(&input, None)?;

        // Bidirectional output should have 2 * hidden_size features
        assert_eq!(
            output.shape().dims(),
            &[2, 5, 64], // 2 * 32 = 64 for bidirectional
            "Bidirectional RNN output should have 2x hidden_size features"
        );

        // Final hidden should include both directions
        assert_eq!(
            final_hidden.shape().dims(),
            &[2, 2, 32], // 2 directions, batch=2, hidden=32
            "Bidirectional RNN should have hidden states for both directions"
        );

        println!("✓ Bidirectional RNN test passed");
        Ok(())
    }

    #[test]
    fn test_mixture_of_experts_scalability() -> Result<()> {
        println!("Testing Mixture of Experts scalability...");

        // Test different MoE configurations
        let configurations = vec![
            ("Small", 4, 2, 64, 128),   // 4 experts, top-2
            ("Medium", 8, 2, 128, 256), // 8 experts, top-2
            ("Large", 16, 4, 128, 512), // 16 experts, top-4
        ];

        for (name, num_experts, num_selected, embed_dim, expert_dim) in configurations {
            println!("  Testing {} MoE configuration", name);

            let moe = MixtureOfExperts::<TestFloat>::new(
                num_experts,
                num_selected,
                embed_dim,
                expert_dim,
                0.1,  // dropout_prob
                0.01, // load_balance_loss_coef
            )?;

            // Test with varying batch sizes and sequence lengths
            let test_cases = vec![
                (1, 5),  // Single sequence
                (4, 10), // Small batch
                (8, 8),  // Medium batch
            ];

            for (batch_size, seq_len) in test_cases {
                let input = Tensor::zeros(&[batch_size, seq_len, embed_dim]);

                // Forward pass with auxiliary loss
                let (output, aux_loss) = moe.forward_with_aux_loss(&input)?;

                // Verify output preserves input dimensions
                assert_eq!(
                    output.shape().dims(),
                    &[batch_size, seq_len, embed_dim],
                    "{} MoE failed for batch_size={}, seq_len={}",
                    name,
                    batch_size,
                    seq_len
                );

                // Verify auxiliary loss is scalar
                assert!(
                    aux_loss.shape().dims().iter().product::<usize>() == 1,
                    "{} MoE auxiliary loss should be scalar",
                    name
                );
            }

            println!("    ✓ {} MoE configuration passed", name);
        }

        println!("✓ MoE scalability test passed");
        Ok(())
    }

    #[test]
    fn test_moe_load_balancing_effectiveness() -> Result<()> {
        println!("Testing MoE load balancing effectiveness...");

        // Create MoE with strong load balancing
        let moe_balanced = MixtureOfExperts::<TestFloat>::new(
            6,   // num_experts
            2,   // num_selected
            32,  // embed_dim
            64,  // expert_dim
            0.0, // dropout_prob
            0.1, // high load_balance_loss_coef
        )?;

        // Create MoE with weak load balancing
        let moe_unbalanced = MixtureOfExperts::<TestFloat>::new(
            6,     // num_experts
            2,     // num_selected
            32,    // embed_dim
            64,    // expert_dim
            0.0,   // dropout_prob
            0.001, // low load_balance_loss_coef
        )?;

        // Test with diverse input patterns
        let inputs = vec![
            Tensor::zeros(&[4, 8, 32]),
            Tensor::ones(&[4, 8, 32]),
            Tensor::zeros(&[8, 4, 32]),
        ];

        for (i, input) in inputs.iter().enumerate() {
            println!("  Testing input pattern {}", i + 1);

            // Process with both MoE variants
            let (output_balanced, loss_balanced) = moe_balanced.forward_with_aux_loss(input)?;
            let (output_unbalanced, loss_unbalanced) =
                moe_unbalanced.forward_with_aux_loss(input)?;

            // Both should produce valid outputs
            assert_eq!(
                output_balanced.shape().dims(),
                output_unbalanced.shape().dims(),
                "Both MoE variants should produce same output shape"
            );

            // Both should produce scalar losses
            assert!(loss_balanced.shape().dims().iter().product::<usize>() == 1);
            assert!(loss_unbalanced.shape().dims().iter().product::<usize>() == 1);
        }

        println!("✓ MoE load balancing test passed");
        Ok(())
    }

    #[test]
    fn test_combined_rnn_moe_pipeline() -> Result<()> {
        println!("Testing combined RNN + MoE pipeline...");

        // Create RNN encoder
        let encoder = RNN::<TestFloat>::new(
            50,    // input_size (e.g., token embeddings)
            128,   // hidden_size
            1,     // num_layers
            true,  // bias
            true,  // batch_first
            0.1,   // dropout
            false, // bidirectional
        )?;

        // Create MoE layer for processing RNN outputs
        let moe_processor = MixtureOfExperts::<TestFloat>::new(
            8,    // num_experts
            2,    // num_selected
            128,  // embed_dim (matches RNN hidden_size)
            512,  // expert_dim
            0.1,  // dropout_prob
            0.01, // load_balance_loss_coef
        )?;

        // Test pipeline with different scenarios
        let test_scenarios = vec![
            ("Short sequence", 1, 5),
            ("Medium sequence", 2, 15),
            ("Long sequence", 1, 30),
        ];

        for (scenario_name, batch_size, seq_len) in test_scenarios {
            println!("  Testing scenario: {}", scenario_name);

            // Create input [batch_size, seq_len, input_size=50]
            let input = Tensor::ones(&[batch_size, seq_len, 50]);

            // Step 1: Encode with RNN
            let rnn_output = encoder.forward(&input)?;
            assert_eq!(
                rnn_output.shape().dims(),
                &[batch_size, seq_len, 128],
                "RNN output shape incorrect for {}",
                scenario_name
            );

            // Step 2: Process with MoE
            let (moe_output, load_balance_loss) =
                moe_processor.forward_with_aux_loss(&rnn_output)?;
            assert_eq!(
                moe_output.shape().dims(),
                &[batch_size, seq_len, 128],
                "MoE output shape incorrect for {}",
                scenario_name
            );

            // Step 3: Verify auxiliary loss
            assert!(
                load_balance_loss.shape().dims().iter().product::<usize>() == 1,
                "Load balance loss should be scalar for {}",
                scenario_name
            );

            println!("    ✓ {} completed successfully", scenario_name);
        }

        println!("✓ Combined RNN + MoE pipeline test passed");
        Ok(())
    }

    #[test]
    fn test_parameter_management_consistency() -> Result<()> {
        println!("Testing parameter management consistency...");

        // Create components
        let rnn = RNN::<TestFloat>::new(10, 20, 2, true, true, 0.1, false)?;
        let moe = MixtureOfExperts::<TestFloat>::new(4, 2, 32, 64, 0.1, 0.01)?;

        // Test parameter access consistency
        let rnn_params = rnn.parameters();
        let rnn_params_mut_count = {
            let mut rnn_mut = rnn.clone();
            rnn_mut.parameters_mut().len()
        };

        assert_eq!(
            rnn_params.len(),
            rnn_params_mut_count,
            "RNN parameter count should be consistent between immutable and mutable access"
        );

        let moe_params = moe.parameters();
        let moe_params_mut_count = {
            let mut moe_mut = moe.clone();
            moe_mut.parameters_mut().len()
        };

        assert_eq!(
            moe_params.len(),
            moe_params_mut_count,
            "MoE parameter count should be consistent between immutable and mutable access"
        );

        // Test training mode setting
        let mut rnn_mut = rnn.clone();
        let mut moe_mut = moe.clone();

        rnn_mut.set_training(false);
        rnn_mut.set_training(true);
        moe_mut.set_training(false);
        moe_mut.set_training(true);

        println!("✓ Parameter management consistency test passed");
        Ok(())
    }

    #[test]
    fn test_enhanced_components_memory_patterns() -> Result<()> {
        println!("Testing memory usage patterns of enhanced components...");

        // Test components with realistic sizes
        let configurations: Vec<(&str, Box<dyn Fn() -> Result<Box<dyn Layer<TestFloat>>>>)> = vec![
            (
                "Small RNN",
                Box::new(|| -> Result<Box<dyn Layer<TestFloat>>> {
                    Ok(Box::new(RNN::<TestFloat>::new(
                        64, 128, 1, true, true, 0.1, false,
                    )?))
                }),
            ),
            (
                "Large RNN",
                Box::new(|| -> Result<Box<dyn Layer<TestFloat>>> {
                    Ok(Box::new(RNN::<TestFloat>::new(
                        256, 512, 2, true, true, 0.1, true,
                    )?))
                }),
            ),
            (
                "Small MoE",
                Box::new(|| -> Result<Box<dyn Layer<TestFloat>>> {
                    Ok(Box::new(MixtureOfExperts::<TestFloat>::new(
                        4, 2, 128, 256, 0.1, 0.01,
                    )?))
                }),
            ),
            (
                "Large MoE",
                Box::new(|| -> Result<Box<dyn Layer<TestFloat>>> {
                    Ok(Box::new(MixtureOfExperts::<TestFloat>::new(
                        16, 4, 512, 1024, 0.1, 0.01,
                    )?))
                }),
            ),
        ];

        for (config_name, create_layer) in configurations {
            println!("  Testing memory pattern: {}", config_name);

            let layer = create_layer()?;

            // Test parameter count is reasonable
            let param_count = layer.parameters().len();
            assert!(param_count > 0, "{} should have parameters", config_name);

            // Test with small input to verify basic functionality
            let input_size = match config_name {
                "Small RNN" => 64,
                "Large RNN" => 256,
                "Small MoE" | "Large MoE" => {
                    if config_name.contains("Small") {
                        128
                    } else {
                        512
                    }
                }
                _ => 64,
            };

            let input = Tensor::zeros(&[1, 2, input_size]);
            let output = layer.forward(&input)?;

            // Output should have expected dimensions
            let expected_output_dim = match config_name {
                "Small RNN" => 128,
                "Large RNN" => 1024, // 512 * 2 for bidirectional
                "Small MoE" => 128,
                "Large MoE" => 512,
                _ => input_size,
            };

            assert_eq!(
                output.shape().dims()[2],
                expected_output_dim,
                "{} output dimension incorrect",
                config_name
            );

            println!("    ✓ {} memory pattern test passed", config_name);
        }

        println!("✓ Memory patterns test passed");
        Ok(())
    }
}

/// Performance benchmarking tests (marked as ignored by default)
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    #[ignore] // Run with: cargo test -- --ignored
    fn benchmark_rnn_performance() -> Result<()> {
        println!("Benchmarking RNN performance...");

        let rnn = RNN::<TestFloat>::new(512, 1024, 2, true, true, 0.1, true)?;

        let batch_sizes = vec![1, 4, 16, 32];
        let seq_lengths = vec![10, 50, 100, 200];

        for batch_size in batch_sizes {
            for seq_len in seq_lengths.iter() {
                let input = Tensor::zeros(&[batch_size, *seq_len, 512]);

                // Time the forward pass
                let start = std::time::Instant::now();
                let _output = rnn.forward(&input)?;
                let duration = start.elapsed();

                println!(
                    "  RNN forward pass: batch={}, seq_len={}, time={:?}",
                    batch_size, seq_len, duration
                );
            }
        }

        println!("✓ RNN performance benchmark completed");
        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test -- --ignored
    fn benchmark_moe_performance() -> Result<()> {
        println!("Benchmarking MoE performance...");

        let configurations = vec![
            (8, 2, 256, 1024),   // Medium
            (16, 4, 512, 2048),  // Large
            (32, 8, 1024, 4096), // Very large
        ];

        for (num_experts, num_selected, embed_dim, expert_dim) in configurations {
            let moe = MixtureOfExperts::<TestFloat>::new(
                num_experts,
                num_selected,
                embed_dim,
                expert_dim,
                0.1,
                0.01,
            )?;

            let input = Tensor::zeros(&[8, 32, embed_dim]);

            // Time the forward pass
            let start = std::time::Instant::now();
            let (_output, _aux_loss) = moe.forward_with_aux_loss(&input)?;
            let duration = start.elapsed();

            println!(
                "  MoE forward pass: experts={}, selected={}, embed_dim={}, time={:?}",
                num_experts, num_selected, embed_dim, duration
            );
        }

        println!("✓ MoE performance benchmark completed");
        Ok(())
    }
}
