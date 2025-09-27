//! Neural network operations module
//!
//! This module provides comprehensive neural network functionality organized into
//! focused sub-modules for better maintainability and clarity.

pub mod functions;
pub mod gradient_tape;
pub mod hooks;
pub mod layers;
pub mod optimizers;

// Re-export main types for backward compatibility
pub use functions::*;
pub use gradient_tape::{PyGradientContext, PyGradientTape, PyTrackedTensor};
pub use hooks::{BackwardHook, ForwardHook, HookManager, PyGlobalHookRegistry, PyHookHandle};
pub use layers::{PyDense, PyParameter, PySequential};
pub use optimizers::PyAdam;

use pyo3::prelude::*;

/// Register neural network functions with Python module
pub fn register_neural_functions(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyHookHandle>()?;
    m.add_class::<PyGradientTape>()?;
    m.add_class::<PyTrackedTensor>()?;
    m.add_class::<PyDense>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PySequential>()?;
    m.add_class::<PyGradientContext>()?;
    m.add_class::<PyGlobalHookRegistry>()?;
    m.add_class::<PyAdam>()?;

    // Register activation functions
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(elu, m)?)?;
    m.add_function(wrap_pyfunction!(swish, m)?)?;
    m.add_function(wrap_pyfunction!(mish, m)?)?;
    m.add_function(wrap_pyfunction!(relu6, m)?)?;
    m.add_function(wrap_pyfunction!(hardswish, m)?)?;

    // Register loss functions
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(binary_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_l1_loss, m)?)?;

    // Register regularization functions
    m.add_function(wrap_pyfunction!(dropout, m)?)?;

    // Register layer functions
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(layers::linear, m)?)?;
    m.add_function(wrap_pyfunction!(layers::relu_linear, m)?)?;
    m.add_function(wrap_pyfunction!(layers::sigmoid_linear, m)?)?;
    m.add_function(wrap_pyfunction!(layers::tanh_linear, m)?)?;

    // Register gradient tape functions
    m.add_function(wrap_pyfunction!(gradient_tape::create_gradient_tape, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_tape::jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_tape::hessian, m)?)?;

    Ok(())
}
