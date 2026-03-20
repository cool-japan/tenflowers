#![allow(clippy::result_large_err)]

use tenflowers_autograd::grad_ops::{fft3_backward, ifft3_backward};
use tenflowers_core::{Result, Tensor, TensorError};

/// Test 3D FFT backward pass
#[test]
fn test_fft3_backward_basic() -> Result<()> {
    // Create a simple 3D tensor for testing
    let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;

    let grad_output =
        Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 2, 2])?;

    // Compute the gradient
    let grad_input = fft3_backward(&grad_output, &input)?;

    // For now, the implementation returns a clone of grad_output
    // This test ensures the function executes without error and maintains shape
    assert_eq!(grad_input.shape().dims(), input.shape().dims());
    assert_eq!(grad_input.shape().dims(), grad_output.shape().dims());

    Ok(())
}

/// Test 3D IFFT backward pass
#[test]
fn test_ifft3_backward_basic() -> Result<()> {
    // Create a simple 3D tensor for testing
    let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;

    let grad_output =
        Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 2, 2])?;

    // Compute the gradient
    let grad_input = ifft3_backward(&grad_output, &input)?;

    // For now, the implementation returns a clone of grad_output
    // This test ensures the function executes without error and maintains shape
    assert_eq!(grad_input.shape().dims(), input.shape().dims());
    assert_eq!(grad_input.shape().dims(), grad_output.shape().dims());

    Ok(())
}

/// Test that FFT3 backward fails with mismatched shapes
#[test]
fn test_fft3_backward_shape_mismatch() {
    let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let grad_output =
        Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 2, 2]).unwrap();

    // This should return an error due to shape mismatch
    let result = fft3_backward(&grad_output, &input);
    assert!(result.is_err());

    if let Err(TensorError::ShapeMismatch { .. }) = result {
        // Expected error type
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

/// Test that IFFT3 backward fails with mismatched shapes  
#[test]
fn test_ifft3_backward_shape_mismatch() {
    let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let grad_output =
        Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 2, 2]).unwrap();

    // This should return an error due to shape mismatch
    let result = ifft3_backward(&grad_output, &input);
    assert!(result.is_err());

    if let Err(TensorError::ShapeMismatch { .. }) = result {
        // Expected error type
    } else {
        panic!("Expected ShapeMismatch error");
    }
}

/// Test 3D FFT backward with different data types
#[test]
fn test_fft3_backward_f64() -> Result<()> {
    let input = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;

    let grad_output =
        Tensor::<f64>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 2, 2])?;

    let grad_input = fft3_backward(&grad_output, &input)?;

    assert_eq!(grad_input.shape().dims(), input.shape().dims());
    assert_eq!(grad_input.shape().dims(), grad_output.shape().dims());

    Ok(())
}

/// Test 3D IFFT backward with different data types
#[test]
fn test_ifft3_backward_f64() -> Result<()> {
    let input = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;

    let grad_output =
        Tensor::<f64>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 2, 2])?;

    let grad_input = ifft3_backward(&grad_output, &input)?;

    assert_eq!(grad_input.shape().dims(), input.shape().dims());
    assert_eq!(grad_input.shape().dims(), grad_output.shape().dims());

    Ok(())
}

/// Test with larger 3D tensors
#[test]
fn test_fft3_backward_larger_tensor() -> Result<()> {
    // Create a 4x4x4 tensor
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let input = Tensor::<f32>::from_vec(data.clone(), &[4, 4, 4])?;

    let grad_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let grad_output = Tensor::<f32>::from_vec(grad_data, &[4, 4, 4])?;

    let grad_input = fft3_backward(&grad_output, &input)?;

    assert_eq!(grad_input.shape().dims(), &[4, 4, 4]);

    Ok(())
}
