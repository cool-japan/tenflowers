//! Tests for TenfloweRS FFI Python bindings
//!
//! These tests verify that the Python bindings work correctly with the core library.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::{PyDense, PyGradientTape, PyParameter, PySequential};
    use crate::tensor_ops::PyTensor;
    use pyo3::prelude::*;

    /// Test basic tensor creation through Python bindings
    #[test]
    fn test_py_tensor_creation() {
        Python::initialize();

        Python::attach(|py| {
            // Test basic tensor creation with new constructor
            let tensor = PyTensor::new(vec![2, 3]).expect("test: construction should succeed");

            // Verify shape
            let shape = tensor.shape();
            assert_eq!(shape.len(), 2);
            assert_eq!(shape[0], 2);
            assert_eq!(shape[1], 3);
        });
    }

    /// Test tensor basic properties
    #[test]
    fn test_py_tensor_properties() {
        Python::initialize();

        Python::attach(|py| {
            // Create test tensor
            let tensor = PyTensor::new(vec![4, 4]).expect("test: construction should succeed");

            // Test basic properties
            let shape = tensor.shape();
            assert_eq!(shape, vec![4, 4]);

            // Test size calculation
            let size = tensor.size();
            assert_eq!(size, 16);
        });
    }

    /// Test gradient tape functionality
    #[test]
    fn test_py_gradient_tape() {
        Python::initialize();

        Python::attach(|py| {
            // Create gradient tape
            let tape = PyGradientTape::new();

            // Test that tape can be created without errors
            // Basic functionality test - creating a tensor and watching it
            let tensor = PyTensor::new(vec![2, 2]).expect("test: construction should succeed");
            let _tracked = tape.watch(&tensor).expect("test: operation should succeed");

            // Test tape control methods
            tape.stop_recording();
            tape.start_recording();
            tape.reset();
        });
    }

    /// Test dense layer creation
    #[test]
    fn test_py_dense_layer() {
        Python::initialize();

        Python::attach(|py| -> PyResult<()> {
            // Create a dense layer with parameters
            let dense = PyDense::new(
                10,                       // input_dim
                5,                        // output_dim
                Some(true),               // use_bias
                Some("relu".to_string()), // activation
            )?;

            // Test that dense layer was created successfully and test weight and bias access through parameters.
            // parameters() now returns Vec<Py<PyParameter>> (not Vec<PyTensor>), so each element must be
            // `.borrow(py)`-ed into a `PyRef<PyParameter>` before calling inherent PyParameter methods like
            // `.shape()` on it.
            let params = dense.parameters();
            assert!(params.len() >= 1); // Should have at least weight parameter

            // First parameter should be weights
            let weight = params[0].borrow(py);
            assert_eq!(weight.shape(), vec![10, 5]);

            // If bias exists, it should be the second parameter
            if params.len() > 1 {
                let bias = params[1].borrow(py);
                assert_eq!(bias.shape(), vec![5]); // Bias shape should match output dimension
            }

            // Test training mode
            assert!(!dense.is_training()); // Should start in eval mode

            Ok(())
        })
        .expect("test: operation should succeed");
    }

    /// Test sequential model creation
    #[test]
    fn test_py_sequential_model() {
        Python::initialize();

        Python::attach(|py| -> PyResult<()> {
            // Create sequential model
            let mut model = PySequential::new();

            // Test model can be created without immediate layers
            // More complex layer addition would require proper PyAny handling

            // Test model summary functionality
            let summary = model.__str__();
            assert!(summary.contains("Sequential"));

            Ok(())
        })
        .expect("test: operation should succeed");
    }

    /// Test parameter creation
    #[test]
    fn test_py_parameter() {
        Python::initialize();

        Python::attach(|py| {
            // Create a parameter tensor
            let tensor = PyTensor::new(vec![3, 3]).expect("test: construction should succeed");
            let param = PyParameter::new(tensor, Some(true));

            // Verify parameter was created successfully
            let shape = param.shape();
            assert_eq!(shape, vec![3, 3]);

            // Test parameter properties
            assert!(param.requires_grad());
            assert_eq!(param.size(), 9);

            // Test cloning
            let _cloned = param.clone_param();
        });
    }

    /// Test Adam optimizer creation
    // #[test]
    // fn test_py_adam_optimizer() {
    //     Python::initialize();

    //     Python::attach(|py| -> PyResult<()> {
    //         // Create Adam optimizer with learning rate parameter
    //         let _adam = PyAdam::new(Some(0.001)); // Only learning rate supported currently

    //         // Test that Adam optimizer was created successfully
    //         // (internal parameters are private, so we just verify construction)

    //         Ok(())
    //     })
    //     .expect("test: operation should succeed");
    // }

    /// Test tensor size and ndim methods
    #[test]
    fn test_py_tensor_dimensions() {
        Python::initialize();

        Python::attach(|py| {
            // Test different tensor shapes
            let tensor_1d = PyTensor::new(vec![5]).expect("test: construction should succeed");
            let tensor_2d = PyTensor::new(vec![3, 4]).expect("test: construction should succeed");
            let tensor_3d =
                PyTensor::new(vec![2, 3, 4]).expect("test: construction should succeed");

            // Verify dimensions
            assert_eq!(tensor_1d.ndim(), 1);
            assert_eq!(tensor_2d.ndim(), 2);
            assert_eq!(tensor_3d.ndim(), 3);

            // Verify sizes
            assert_eq!(tensor_1d.size(), 5);
            assert_eq!(tensor_2d.size(), 12);
            assert_eq!(tensor_3d.size(), 24);
        });
    }

    /// Test tensor cloning
    #[test]
    fn test_py_tensor_clone() {
        Python::initialize();

        Python::attach(|py| {
            // Create and clone tensor
            let original = PyTensor::new(vec![2, 3]).expect("test: construction should succeed");
            let cloned = original.clone();

            // Verify shapes are equal
            assert_eq!(original.shape(), cloned.shape());
            assert_eq!(original.size(), cloned.size());
        });
    }
}
