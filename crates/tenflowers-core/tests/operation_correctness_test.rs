use scirs2_core::num_traits::{Float, One, Zero};
use std::fmt::Debug;
use tenflowers_core::{ops::*, Result, Tensor};

/// Tolerance for numerical comparisons
const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-8;

/// Tolerance for gradient checking
const GRAD_RTOL: f32 = 1e-3;
const GRAD_ATOL: f32 = 1e-5;

/// Helper function to check if two tensors are approximately equal
fn assert_tensors_close<T>(actual: &Tensor<T>, expected: &[T], rtol: T, atol: T, msg: &str)
where
    T: Float + Debug + Clone,
{
    let actual_slice = actual.as_slice().expect("Failed to get tensor slice");
    assert_eq!(
        actual_slice.len(),
        expected.len(),
        "{}: Shape mismatch",
        msg
    );

    for (i, (&a, &e)) in actual_slice.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tolerance = atol + rtol * e.abs();
        assert!(
            diff <= tolerance,
            "{}: Element {} differs: actual={:?}, expected={:?}, diff={:?}, tolerance={:?}",
            msg,
            i,
            a,
            e,
            diff,
            tolerance
        );
    }
}

/// Generate reference results using simple CPU implementations
mod reference_implementations {
    use super::*;

    /// Reference implementation of matrix multiplication using simple nested loops
    pub fn matmul_reference(
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Vec<f32> {
        assert_eq!(a_shape.len(), 2);
        assert_eq!(b_shape.len(), 2);
        assert_eq!(a_shape[1], b_shape[0]);

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        result
    }

    /// Reference implementation of element-wise addition
    pub fn add_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Reference implementation of element-wise subtraction
    pub fn sub_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }

    /// Reference implementation of element-wise multiplication
    pub fn mul_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Reference implementation of element-wise division
    pub fn div_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
    }

    /// Reference implementation of ReLU activation
    pub fn relu_reference(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
    }

    /// Reference implementation of sigmoid activation
    pub fn sigmoid_reference(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    /// Reference implementation of tanh activation
    pub fn tanh_reference(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v.tanh()).collect()
    }

    /// Reference implementation of sum reduction
    pub fn sum_reference(x: &[f32]) -> f32 {
        x.iter().sum()
    }

    /// Reference implementation of mean reduction
    pub fn mean_reference(x: &[f32]) -> f32 {
        x.iter().sum::<f32>() / x.len() as f32
    }
}

/// Numerical gradient checking utilities
mod gradient_check {
    use super::*;

    /// Compute numerical gradient using finite differences
    pub fn numerical_gradient<F>(f: F, x: &[f32], eps: f32) -> Vec<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let mut grad = vec![0.0; x.len()];

        for i in 0..x.len() {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();

            x_plus[i] += eps;
            x_minus[i] -= eps;

            let f_plus = f(&x_plus);
            let f_minus = f(&x_minus);

            grad[i] = (f_plus - f_minus) / (2.0 * eps);
        }

        grad
    }

    /// Check gradient correctness for a scalar-valued function
    pub fn check_gradient<F>(
        f: F,
        analytical_grad: &[f32],
        x: &[f32],
        eps: f32,
        rtol: f32,
        atol: f32,
        test_name: &str,
    ) where
        F: Fn(&[f32]) -> f32,
    {
        let numerical_grad = numerical_gradient(f, x, eps);

        assert_eq!(
            analytical_grad.len(),
            numerical_grad.len(),
            "{}: Gradient dimension mismatch",
            test_name
        );

        for (i, (&analytical, &numerical)) in analytical_grad
            .iter()
            .zip(numerical_grad.iter())
            .enumerate()
        {
            let diff = (analytical - numerical).abs();
            let tolerance = atol + rtol * numerical.abs();

            assert!(
                diff <= tolerance,
                "{}: Gradient element {} differs: analytical={:?}, numerical={:?}, diff={:?}, tolerance={:?}",
                test_name, i, analytical, numerical, diff, tolerance
            );
        }
    }
}

#[test]
fn test_binary_operations_correctness() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![2.0, 1.0, 4.0, 3.0];

    let a = Tensor::from_vec(a_data.clone(), &[2, 2]).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[2, 2]).unwrap();

    // Test addition
    let add_result = binary::add(&a, &b).unwrap();
    let add_expected = reference_implementations::add_reference(&a_data, &b_data);
    assert_tensors_close(&add_result, &add_expected, RTOL, ATOL, "Addition");

    // Test subtraction
    let sub_result = binary::sub(&a, &b).unwrap();
    let sub_expected = reference_implementations::sub_reference(&a_data, &b_data);
    assert_tensors_close(&sub_result, &sub_expected, RTOL, ATOL, "Subtraction");

    // Test multiplication
    let mul_result = binary::mul(&a, &b).unwrap();
    let mul_expected = reference_implementations::mul_reference(&a_data, &b_data);
    assert_tensors_close(&mul_result, &mul_expected, RTOL, ATOL, "Multiplication");

    // Test division
    let div_result = binary::div(&a, &b).unwrap();
    let div_expected = reference_implementations::div_reference(&a_data, &b_data);
    assert_tensors_close(&div_result, &div_expected, RTOL, ATOL, "Division");
}

#[test]
fn test_matmul_correctness() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = Tensor::from_vec(a_data.clone(), &[2, 3]).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[3, 2]).unwrap();

    let result = matmul::matmul(&a, &b).unwrap();
    let expected = reference_implementations::matmul_reference(&a_data, &[2, 3], &b_data, &[3, 2]);

    assert_eq!(result.shape().dims(), &[2, 2]);
    assert_tensors_close(&result, &expected, RTOL, ATOL, "Matrix multiplication");
}

#[test]
fn test_activation_functions_correctness() {
    let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let input = Tensor::from_vec(input_data.clone(), &[5]).unwrap();

    // Test ReLU
    let relu_result = activation::relu(&input).unwrap();
    let relu_expected = reference_implementations::relu_reference(&input_data);
    assert_tensors_close(&relu_result, &relu_expected, RTOL, ATOL, "ReLU");

    // Test Sigmoid
    let sigmoid_result = activation::sigmoid(&input).unwrap();
    let sigmoid_expected = reference_implementations::sigmoid_reference(&input_data);
    assert_tensors_close(&sigmoid_result, &sigmoid_expected, RTOL, ATOL, "Sigmoid");

    // Test Tanh
    let tanh_result = activation::tanh(&input).unwrap();
    let tanh_expected = reference_implementations::tanh_reference(&input_data);
    assert_tensors_close(&tanh_result, &tanh_expected, RTOL, ATOL, "Tanh");
}

#[test]
fn test_reduction_operations_correctness() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_vec(input_data.clone(), &[2, 3]).unwrap();

    // Test sum reduction
    let sum_result = reduction::sum(&input, None, false).unwrap();
    let sum_expected = reference_implementations::sum_reference(&input_data);
    assert_eq!(sum_result.shape().dims(), &[] as &[usize]);
    assert_tensors_close(&sum_result, &[sum_expected], RTOL, ATOL, "Sum reduction");

    // Test mean reduction
    let mean_result = reduction::mean(&input, None, false).unwrap();
    let mean_expected = reference_implementations::mean_reference(&input_data);
    assert_eq!(mean_result.shape().dims(), &[] as &[usize]);
    assert_tensors_close(&mean_result, &[mean_expected], RTOL, ATOL, "Mean reduction");
}

#[test]
fn test_broadcasting_correctness() {
    // Test broadcasting with different shapes
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();

    let result = binary::add(&a, &b).unwrap();

    // Expected result: [[1+3, 1+4], [2+3, 2+4]] = [[4, 5], [5, 6]]
    let expected = vec![4.0, 5.0, 5.0, 6.0];

    assert_eq!(result.shape().dims(), &[2, 2]);
    assert_tensors_close(&result, &expected, RTOL, ATOL, "Broadcasting addition");
}

#[test]
fn test_edge_cases() {
    // Test with zeros
    let zeros = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
    let ones = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();

    let add_result = binary::add(&zeros, &ones).unwrap();
    assert_tensors_close(
        &add_result,
        &[1.0, 1.0, 1.0, 1.0],
        RTOL,
        ATOL,
        "Add with zeros",
    );

    let mul_result = binary::mul(&zeros, &ones).unwrap();
    assert_tensors_close(
        &mul_result,
        &[0.0, 0.0, 0.0, 0.0],
        RTOL,
        ATOL,
        "Multiply with zeros",
    );

    // Test with negative numbers
    let neg_data = vec![-1.0, -2.0, -3.0, -4.0];
    let neg = Tensor::from_vec(neg_data, &[2, 2]).unwrap();

    let relu_result = activation::relu(&neg).unwrap();
    assert_tensors_close(
        &relu_result,
        &[0.0, 0.0, 0.0, 0.0],
        RTOL,
        ATOL,
        "ReLU with negatives",
    );
}

#[test]
fn test_large_tensor_operations() {
    // Test with larger tensors to ensure scalability
    let size = 100;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let a = Tensor::from_vec(a_data.clone(), &[size]).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[size]).unwrap();

    let add_result = binary::add(&a, &b).unwrap();
    let add_expected = reference_implementations::add_reference(&a_data, &b_data);

    assert_eq!(add_result.shape().dims(), &[size]);
    assert_tensors_close(
        &add_result,
        &add_expected,
        RTOL,
        ATOL,
        "Large tensor addition",
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "GPU shader validation issues - f64 requires FLOAT64 capability"]
fn test_cpu_gpu_parity() {
    use tenflowers_core::device::Device;

    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    // Test on CPU
    let cpu_tensor = Tensor::from_vec(input_data.clone(), &[2, 2]).unwrap();
    let cpu_result = activation::relu(&cpu_tensor).unwrap();

    // Test on GPU
    let gpu_device = Device::try_gpu(0).unwrap();
    let gpu_tensor = cpu_tensor.to_device(gpu_device).unwrap();
    let gpu_result = activation::relu(&gpu_tensor).unwrap();
    let gpu_result_cpu = gpu_result.to_device(Device::Cpu).unwrap();

    // Compare results
    let cpu_slice = cpu_result.as_slice().unwrap();
    let gpu_slice = gpu_result_cpu.as_slice().unwrap();

    assert_eq!(cpu_slice.len(), gpu_slice.len());
    for (i, (&cpu_val, &gpu_val)) in cpu_slice.iter().zip(gpu_slice.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        assert!(
            diff < 1e-6,
            "CPU/GPU mismatch at element {}: CPU={}, GPU={}, diff={}",
            i,
            cpu_val,
            gpu_val,
            diff
        );
    }
}

/// Performance benchmark test
#[test]
fn test_operation_performance() {
    use std::time::Instant;

    // Use smaller size for basic performance test
    let size = 100;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 100) as f32).collect();

    let a = Tensor::from_vec(a_data, &[size, size]).unwrap();
    let b = Tensor::from_vec(b_data, &[size, size]).unwrap();

    // Benchmark matrix multiplication
    let start = Instant::now();
    let _result = matmul::matmul(&a, &b).unwrap();
    let duration = start.elapsed();

    println!(
        "Matrix multiplication ({}x{}) took: {:?}",
        size, size, duration
    );

    // Basic performance threshold - just ensure it completes in reasonable time
    // Note: Current implementation uses simple nested loops, so performance is O(n^3)
    assert!(
        duration.as_secs() < 30,
        "Matrix multiplication took too long: {:?}",
        duration
    );
}
