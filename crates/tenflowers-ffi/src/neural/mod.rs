//! Neural network operations module
//!
//! This module provides comprehensive neural network functionality organized into
//! focused sub-modules for better maintainability and clarity.

pub mod activations;
pub mod attention;
pub mod conv_layers;
pub mod embedding;
pub mod extended_optimizers;
pub mod functions;
pub mod gradient_tape;
pub mod gradient_utils;
pub mod hooks;
pub mod layers;
pub mod losses;
pub mod normalization;
pub mod optimizers;
pub mod recurrent;
pub mod regularization;
pub mod schedulers;
pub mod ssm;
pub mod training_utils;
pub mod transformer;

// Re-export main types for backward compatibility
pub use attention::PyMultiheadAttention;
pub use conv_layers::{PyAvgPool2D, PyConv1D, PyConv2D, PyMaxPool2D};
pub use embedding::{PyEmbedding, PyEmbeddingBag};
pub use functions::*;
pub use gradient_tape::{PyGradientContext, PyGradientTape, PyTrackedTensor};
pub use gradient_utils::PyGradientAccumulator;
pub use hooks::{BackwardHook, ForwardHook, HookManager, PyGlobalHookRegistry, PyHookHandle};
pub use layers::{PyDense, PyParameter, PySequential};
// losses module is not re-exported to avoid conflicts, use neural::losses::* instead
pub use extended_optimizers::{PyAdaBelief, PyAdaDelta, PyAdaGrad, PyNadam, PyRAdam};
pub use normalization::{PyBatchNorm1d, PyGroupNorm, PyInstanceNorm1d, PyLayerNorm};
pub use optimizers::{PyAdam, PyAdamW, PyRMSprop, PySGD};
pub use recurrent::{PyGRU, PyGRUCell, PyLSTM, PyLSTMCell, PyRNN};
pub use regularization::{PyAlphaDropout, PyDropout, PyDropout2D, PyFeatureAlphaDropout};
pub use schedulers::{
    PyCosineAnnealingLR, PyCosineAnnealingWarmRestarts, PyExponentialLR, PyLinearLR,
    PyReduceLROnPlateau, PyStepLR,
};
pub use ssm::{PyMamba, PyStateSpaceModel};
pub use training_utils::{PyEarlyStopping, PyLRWarmup, PyMetricsTracker, PyProgressTracker};
pub use transformer::{PyPositionalEncoding, PyTransformerDecoderLayer, PyTransformerEncoderLayer};

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

    // Register optimizers
    m.add_class::<PyAdam>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyRMSprop>()?;
    m.add_class::<PyAdamW>()?;

    // Register extended optimizers
    m.add_class::<PyAdaBelief>()?;
    m.add_class::<PyRAdam>()?;
    m.add_class::<PyNadam>()?;
    m.add_class::<PyAdaGrad>()?;
    m.add_class::<PyAdaDelta>()?;

    // Register normalization layers
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyGroupNorm>()?;
    m.add_class::<PyInstanceNorm1d>()?;

    // Register convolutional layers
    m.add_class::<PyConv1D>()?;
    m.add_class::<PyConv2D>()?;
    m.add_class::<PyMaxPool2D>()?;
    m.add_class::<PyAvgPool2D>()?;

    // Register regularization layers
    m.add_class::<PyDropout>()?;
    m.add_class::<PyDropout2D>()?;
    m.add_class::<PyAlphaDropout>()?;
    m.add_class::<PyFeatureAlphaDropout>()?;

    // Register recurrent layers
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyGRU>()?;
    m.add_class::<PyRNN>()?;
    m.add_class::<PyLSTMCell>()?;
    m.add_class::<PyGRUCell>()?;

    // Register embedding layers
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyEmbeddingBag>()?;

    // Register learning rate schedulers
    m.add_class::<PyStepLR>()?;
    m.add_class::<PyExponentialLR>()?;
    m.add_class::<PyCosineAnnealingLR>()?;
    m.add_class::<PyReduceLROnPlateau>()?;
    m.add_class::<PyCosineAnnealingWarmRestarts>()?;
    m.add_class::<PyLinearLR>()?;

    // Register attention mechanisms
    m.add_class::<PyMultiheadAttention>()?;
    m.add_function(wrap_pyfunction!(
        attention::scaled_dot_product_attention,
        m
    )?)?;

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

    // Register additional activation functions
    m.add_function(wrap_pyfunction!(activations::selu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softplus, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softsign, m)?)?;
    m.add_function(wrap_pyfunction!(activations::silu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::hardtanh, m)?)?;
    m.add_function(wrap_pyfunction!(activations::hardsigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(activations::logsigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(activations::tanhshrink, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softshrink, m)?)?;
    m.add_function(wrap_pyfunction!(activations::hardshrink, m)?)?;
    m.add_function(wrap_pyfunction!(activations::glu, m)?)?;

    // Register loss functions from losses module
    m.add_function(wrap_pyfunction!(losses::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(losses::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(losses::l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::smooth_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::kl_div_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::hinge_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::cosine_embedding_loss, m)?)?;

    // Register regularization functions
    m.add_function(wrap_pyfunction!(dropout, m)?)?;
    m.add_function(wrap_pyfunction!(regularization::l1_regularization, m)?)?;
    m.add_function(wrap_pyfunction!(regularization::l2_regularization, m)?)?;
    m.add_function(wrap_pyfunction!(
        regularization::elastic_net_regularization,
        m
    )?)?;

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

    // Register gradient utility functions
    m.add_class::<PyGradientAccumulator>()?;
    m.add_function(wrap_pyfunction!(gradient_utils::clip_grad_norm, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_utils::clip_grad_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        gradient_utils::detect_gradient_anomaly,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(gradient_utils::gradient_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_utils::normalize_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_utils::scale_gradient, m)?)?;

    // Register training utilities
    m.add_class::<PyEarlyStopping>()?;
    m.add_class::<PyLRWarmup>()?;
    m.add_class::<PyMetricsTracker>()?;
    m.add_class::<PyProgressTracker>()?;

    // Register transformer building blocks
    m.add_class::<PyTransformerEncoderLayer>()?;
    m.add_class::<PyTransformerDecoderLayer>()?;
    m.add_class::<PyPositionalEncoding>()?;
    m.add_function(wrap_pyfunction!(
        transformer::generate_square_subsequent_mask,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transformer::create_padding_mask, m)?)?;

    // Register State Space Model (Mamba/SSM) layers
    m.add_class::<PyMamba>()?;
    m.add_class::<PyStateSpaceModel>()?;

    Ok(())
}
