#[cfg(test)]
mod tests {
    use super::super::functions::*;
    use crate::Tensor;

    #[test]
    fn test_relu() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = relu(&input).unwrap();

        if let Some(data) = output.as_slice() {
            assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
        }
    }

    #[test]
    fn test_sigmoid() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = sigmoid(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values
            assert!((data[0] - 0.1192).abs() < 0.001); // sigmoid(-2)
            assert!((data[1] - 0.2689).abs() < 0.001); // sigmoid(-1)
            assert!((data[2] - 0.5).abs() < 0.001); // sigmoid(0)
            assert!((data[3] - 0.7311).abs() < 0.001); // sigmoid(1)
            assert!((data[4] - 0.8808).abs() < 0.001); // sigmoid(2)
        }
    }

    #[test]
    fn test_tanh() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = tanh(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values
            assert!((data[0] - (-0.9640)).abs() < 0.001); // tanh(-2)
            assert!((data[1] - (-0.7616)).abs() < 0.001); // tanh(-1)
            assert!((data[2] - 0.0).abs() < 0.001); // tanh(0)
            assert!((data[3] - 0.7616).abs() < 0.001); // tanh(1)
            assert!((data[4] - 0.9640).abs() < 0.001); // tanh(2)
        }
    }

    #[test]
    fn test_softmax_1d() {
        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let output = softmax(&input, None).unwrap();

        if let Some(data) = output.as_slice() {
            // Check that it sums to 1
            let sum: f32 = data.iter().sum();
            assert!((sum - 1.0).abs() < 0.001);

            // Check relative ordering
            assert!(data[0] < data[1]);
            assert!(data[1] < data[2]);
        }
    }

    #[test]
    fn test_softmax_2d() {
        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Softmax along last axis (axis=1)
        let output = softmax(&input, Some(1)).unwrap();

        if let Some(data) = output.as_slice() {
            // Check that each row sums to 1
            let row1_sum: f32 = data[0..3].iter().sum();
            let row2_sum: f32 = data[3..6].iter().sum();
            assert!((row1_sum - 1.0).abs() < 0.001);
            assert!((row2_sum - 1.0).abs() < 0.001);
        }

        // Softmax along first axis (axis=0)
        let output = softmax(&input, Some(0)).unwrap();

        if let Some(data) = output.as_slice() {
            // Check that each column sums to 1
            let col1_sum = data[0] + data[3];
            let col2_sum = data[1] + data[4];
            let col3_sum = data[2] + data[5];
            assert!((col1_sum - 1.0).abs() < 0.001);
            assert!((col2_sum - 1.0).abs() < 0.001);
            assert!((col3_sum - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_relu_2d() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap();

        let output = relu(&input).unwrap();

        if let Some(data) = output.as_slice() {
            assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_gelu() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = gelu(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values for GELU
            assert!((data[0] - (-0.0454)).abs() < 0.01); // gelu(-2)
            assert!((data[1] - (-0.1587)).abs() < 0.01); // gelu(-1)
            assert!((data[2] - 0.0).abs() < 0.001); // gelu(0)
            assert!((data[3] - 0.8413).abs() < 0.01); // gelu(1)
            assert!((data[4] - 1.9545).abs() < 0.01); // gelu(2)
        }
    }

    #[test]
    fn test_swish() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = swish(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values for Swish
            assert!((data[0] - (-0.2384)).abs() < 0.01); // swish(-2)
            assert!((data[1] - (-0.2689)).abs() < 0.01); // swish(-1)
            assert!((data[2] - 0.0).abs() < 0.001); // swish(0)
            assert!((data[3] - 0.7311).abs() < 0.01); // swish(1)
            assert!((data[4] - 1.7616).abs() < 0.01); // swish(2)
        }
    }

    #[test]
    fn test_mish() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = mish(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values for Mish
            // mish(x) = x * tanh(ln(1 + exp(x)))
            assert!((data[0] - (-0.2525)).abs() < 0.01); // mish(-2) ≈ -0.2525
            assert!((data[1] - (-0.3034)).abs() < 0.01); // mish(-1) ≈ -0.3034
            assert!((data[2] - 0.0).abs() < 0.001); // mish(0) = 0
            assert!((data[3] - 0.8651).abs() < 0.01); // mish(1) ≈ 0.8651
            assert!((data[4] - 1.9440).abs() < 0.01); // mish(2) ≈ 1.9440
        }
    }

    #[test]
    fn test_elu() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = elu(&input, 1.0).unwrap();

        if let Some(data) = output.as_slice() {
            // Check values for ELU with alpha=1.0
            assert!((data[0] - (-0.8647)).abs() < 0.01); // elu(-2)
            assert!((data[1] - (-0.6321)).abs() < 0.01); // elu(-1)
            assert!((data[2] - 0.0).abs() < 0.001); // elu(0)
            assert!((data[3] - 1.0).abs() < 0.001); // elu(1)
            assert!((data[4] - 2.0).abs() < 0.001); // elu(2)
        }
    }

    #[test]
    fn test_leaky_relu() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = leaky_relu(&input, 0.01).unwrap();

        if let Some(data) = output.as_slice() {
            // Check values for LeakyReLU with alpha=0.01
            assert_eq!(data[0], -0.02); // leaky_relu(-2)
            assert_eq!(data[1], -0.01); // leaky_relu(-1)
            assert_eq!(data[2], 0.0); // leaky_relu(0)
            assert_eq!(data[3], 1.0); // leaky_relu(1)
            assert_eq!(data[4], 2.0); // leaky_relu(2)
        }
    }

    #[test]
    fn test_hard_swish() {
        let input = Tensor::<f32>::from_vec(vec![-3.0, -1.5, 0.0, 1.5, 3.0], &[5]).unwrap();

        let output = hard_swish(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check values for HardSwish
            // hard_swish(x) = x * relu6(x + 3) / 6
            assert_eq!(data[0], 0.0); // hard_swish(-3) = -3 * relu6(0) / 6 = 0
            assert!((data[1] - (-0.375)).abs() < 0.001); // hard_swish(-1.5) = -1.5 * 1.5 / 6 = -0.375
            assert_eq!(data[2], 0.0); // hard_swish(0) = 0 * 3 / 6 = 0
            assert!((data[3] - 1.125).abs() < 0.001); // hard_swish(1.5) = 1.5 * 4.5 / 6 = 1.125
            assert_eq!(data[4], 3.0); // hard_swish(3) = 3 * 6 / 6 = 3
        }
    }

    #[test]
    fn test_prelu_scalar() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let alpha = Tensor::<f32>::from_vec(vec![0.1], &[1]).unwrap();
        let output = prelu(&input, &alpha).unwrap();

        if let Some(data) = output.as_slice() {
            // Check values for PReLU with alpha=0.1
            assert_eq!(data[0], -0.2); // prelu(-2) = 0.1 * -2
            assert_eq!(data[1], -0.1); // prelu(-1) = 0.1 * -1
            assert_eq!(data[2], 0.0); // prelu(0) = 0
            assert_eq!(data[3], 1.0); // prelu(1) = 1
            assert_eq!(data[4], 2.0); // prelu(2) = 2
        }
    }

    #[test]
    fn test_prelu_channelwise() {
        // Test channel-wise PReLU with 2D input (batch=1, channels=2, spatial=2)
        let input = Tensor::<f32>::from_vec(
            vec![
                -1.0, 1.0, // channel 0
                -2.0, 2.0, // channel 1
            ],
            &[1, 2, 2],
        )
        .unwrap();

        let alpha = Tensor::<f32>::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        let output = prelu(&input, &alpha).unwrap();

        if let Some(data) = output.as_slice() {
            // Check values for channel-wise PReLU
            assert_eq!(data[0], -0.1); // channel 0: prelu(-1) = 0.1 * -1
            assert_eq!(data[1], 1.0); // channel 0: prelu(1) = 1
            assert_eq!(data[2], -0.4); // channel 1: prelu(-2) = 0.2 * -2
            assert_eq!(data[3], 2.0); // channel 1: prelu(2) = 2
        }
    }

    #[test]
    fn test_relu_f32_optimized() {
        let input =
            Tensor::<f32>::from_vec(vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], &[7]).unwrap();

        let output = relu_f32(&input).unwrap();

        if let Some(data) = output.as_slice() {
            assert_eq!(data, &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 5.0]);
        }
    }

    #[test]
    fn test_sigmoid_f32_optimized() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = sigmoid_f32(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values
            assert!((data[0] - 0.1192).abs() < 0.01); // sigmoid(-2)
            assert!((data[1] - 0.2689).abs() < 0.01); // sigmoid(-1)
            assert!((data[2] - 0.5).abs() < 0.01); // sigmoid(0)
            assert!((data[3] - 0.7311).abs() < 0.01); // sigmoid(1)
            assert!((data[4] - 0.8808).abs() < 0.01); // sigmoid(2)
        }
    }

    #[test]
    fn test_gelu_f32_optimized() {
        let input = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let output = gelu_f32(&input).unwrap();

        if let Some(data) = output.as_slice() {
            // Check approximate values for GELU
            assert!((data[0] - (-0.0454)).abs() < 0.1); // gelu(-2)
            assert!((data[1] - (-0.1587)).abs() < 0.1); // gelu(-1)
            assert!((data[2] - 0.0).abs() < 0.01); // gelu(0)
            assert!((data[3] - 0.8413).abs() < 0.1); // gelu(1)
            assert!((data[4] - 1.9545).abs() < 0.1); // gelu(2)
        }
    }

    #[test]
    fn test_large_tensor_performance() {
        // Test with larger tensors to trigger different optimization strategies
        let size = 10000;
        let input_data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) / 1000.0)
            .collect();
        let input = Tensor::<f32>::from_vec(input_data, &[size]).unwrap();

        // Test that optimized functions complete without error on large inputs
        let relu_result = relu_f32(&input).unwrap();
        let sigmoid_result = sigmoid_f32(&input).unwrap();
        let gelu_result = gelu_f32(&input).unwrap();

        // Basic sanity checks
        assert_eq!(relu_result.shape().dims(), &[size]);
        assert_eq!(sigmoid_result.shape().dims(), &[size]);
        assert_eq!(gelu_result.shape().dims(), &[size]);
    }

    #[test]
    fn test_activation_performance_analytics() {
        use super::super::core::{get_activation_performance_report, reset_activation_counters};

        reset_activation_counters();

        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Perform some activations
        let _relu_out = relu_f32(&input).unwrap();
        let _sigmoid_out = sigmoid_f32(&input).unwrap();

        // Get performance report
        let report = get_activation_performance_report();

        // Basic verification that analytics are tracked
        assert!(report.function_counts.len() > 0);
    }

    #[test]
    fn test_edge_cases() {
        // Test with extreme values
        let extreme_input = Tensor::<f32>::from_vec(
            vec![f32::NEG_INFINITY, -1000.0, 0.0, 1000.0, f32::INFINITY],
            &[5],
        )
        .unwrap();

        // These should not panic and should handle edge cases gracefully
        let relu_result = relu(&extreme_input);
        let sigmoid_result = sigmoid(&extreme_input);

        assert!(relu_result.is_ok());
        assert!(sigmoid_result.is_ok());

        if let Ok(sigmoid_output) = sigmoid_result {
            if let Some(data) = sigmoid_output.as_slice() {
                // Sigmoid should be bounded between 0 and 1
                for &val in data {
                    assert!(
                        val >= 0.0 && val <= 1.0,
                        "Sigmoid output out of bounds: {}",
                        val
                    );
                }
            }
        }
    }
}
