//! Python bindings for TenfloweRS machine learning framework
#![deny(unsafe_code)]
#![allow(unsafe_code)] // Allow unsafe code for C FFI
#![allow(clippy::too_many_arguments)] // Common in ML APIs
#![allow(clippy::module_name_repetitions)] // Common in FFI bindings
#![allow(unused_variables)] // Some PyO3 method signatures require unused parameters
#![allow(unused_mut)] // PyO3 bindings often require mut
#![allow(dead_code)] // Some functions are exposed only to Python
#![allow(deprecated)] // PyO3 signature deprecations
#![allow(clippy::uninlined_format_args)] // Format string style preference
#![allow(clippy::unnecessary_cast)] // Type casting in PyO3 bindings
#![allow(clippy::redundant_closure)] // PyO3 error handling patterns
#![allow(clippy::new_without_default)] // PyO3 class constructors
#![allow(clippy::manual_map)] // Pattern matching style preference
#![allow(clippy::unnecessary_map_or)] // Option handling style preference

// Module declarations - organize functionality into logical groups
pub mod benchmarks;
pub mod bottleneck_detection;
pub mod dtype_promotion;
pub mod eager_execution_optimizer;
pub mod large_model_support;
pub mod memory_optimizer;
pub mod test_module;

// Refactored modular structures
pub mod neural; // New modular neural operations
pub mod visualization; // New modular visualization

// Core FFI modules
pub mod math_ops;
pub mod tensor_ops;

use pyo3::prelude::*;
use std::sync::{OnceLock, RwLock};
use tenflowers_core::Device;

// Global state management
static DEFAULT_DEVICE: OnceLock<RwLock<Device>> = OnceLock::new();
static MEMORY_PROFILING: OnceLock<RwLock<MemoryProfilingState>> = OnceLock::new();

#[derive(Debug, Clone, Default)]
struct MemoryProfilingState {
    enabled: bool,
    peak_memory: usize,
    current_memory: usize,
}

fn get_default_device_lock() -> &'static RwLock<Device> {
    DEFAULT_DEVICE.get_or_init(|| RwLock::new(Device::Cpu))
}

fn get_memory_profiling_lock() -> &'static RwLock<MemoryProfilingState> {
    MEMORY_PROFILING.get_or_init(|| RwLock::new(MemoryProfilingState::default()))
}

/// Device management functions
#[pyfunction]
fn get_default_device() -> String {
    let device = get_default_device_lock().read().unwrap();
    match *device {
        Device::Cpu => "cpu".to_string(),
        #[cfg(feature = "gpu")]
        Device::Gpu(id) => format!("gpu:{}", id),
    }
}

#[pyfunction]
fn set_default_device(device_str: &str) -> PyResult<()> {
    let device = match device_str {
        "cpu" => Device::Cpu,
        #[cfg(feature = "gpu")]
        s if s.starts_with("gpu:") => {
            let id: usize = s[4..]
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid GPU device ID"))?;
            Device::Gpu(id)
        }
        #[cfg(not(feature = "gpu"))]
        s if s.starts_with("gpu:") => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GPU support not enabled",
            ));
        }
        s if s.starts_with("rocm:") => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ROCm support not available in this build",
            ));
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid device string",
            ))
        }
    };

    *get_default_device_lock().write().unwrap() = device;
    Ok(())
}

/// Memory profiling functions
#[pyfunction]
fn enable_memory_profiling() {
    let mut state = get_memory_profiling_lock().write().unwrap();
    state.enabled = true;
}

#[pyfunction]
fn disable_memory_profiling() {
    let mut state = get_memory_profiling_lock().write().unwrap();
    state.enabled = false;
}

#[pyfunction]
fn get_memory_info() -> PyResult<(usize, usize)> {
    let state = get_memory_profiling_lock().read().unwrap();
    Ok((state.current_memory, state.peak_memory))
}

/// Gradient management functions
#[pyfunction]
fn is_grad_enabled() -> bool {
    tenflowers_autograd::no_grad::is_grad_enabled()
}

#[pyfunction]
fn set_grad_enabled(enabled: bool) {
    tenflowers_autograd::no_grad::set_grad_enabled(enabled);
}

/// Utility functions for working with Python objects
#[pyfunction]
fn tensor_from_numpy(py: Python, array: Bound<'_, PyAny>) -> PyResult<tensor_ops::PyTensor> {
    use numpy::PyReadonlyArrayDyn;

    // Convert numpy array to PyReadonlyArrayDyn<f32>
    let np_array: PyReadonlyArrayDyn<f32> = array.extract()?;
    let array_view = np_array.as_array();

    // Extract shape and data
    let shape: Vec<usize> = array_view.shape().to_vec();
    let data: Vec<f32> = array_view.iter().copied().collect();

    // Create tensor from data
    let tensor = tenflowers_core::Tensor::from_vec(data, &shape).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create tensor: {}", e))
    })?;

    Ok(tensor_ops::PyTensor {
        tensor: std::sync::Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

#[pyfunction]
fn tensor_to_numpy(py: Python, tensor: &tensor_ops::PyTensor) -> PyResult<PyObject> {
    // use numpy::PyArrayDyn; // Unused for now

    // Get tensor data and shape
    let shape = tensor.shape();
    let data = tensor.tensor.to_vec().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e))
    })?;

    // Create multi-dimensional numpy array from tensor data and shape
    use numpy::ndarray::{ArrayD, IxDyn};
    use numpy::{PyArray1, PyArrayDyn};

    // First create an ndarray from the data and shape
    let ndarray = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create ndarray: {}", e))
    })?;

    // Convert ndarray to numpy array
    let np_array = PyArrayDyn::from_array(py, &ndarray);

    Ok(np_array.into_pyobject(py)?.into())
}

/// Setup PyTorch compatibility layer with version detection
fn setup_torch_compatibility(py: Python, torch_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Detect PyTorch version if available
    let pytorch_version = detect_pytorch_version(py);

    match pytorch_version.as_deref() {
        Some(version) => {
            // Parse version to determine compatibility requirements
            let major_version = extract_major_version(version);

            // Add version-specific function mappings
            match major_version {
                1 => setup_pytorch_v1_compatibility(py, torch_module)?,
                2 => setup_pytorch_v2_compatibility(py, torch_module)?,
                _ => setup_default_pytorch_compatibility(py, torch_module)?,
            }

            // Add version info to torch module
            torch_module.setattr("__version__", version)?;
        }
        None => {
            // PyTorch not detected, use default compatibility layer
            setup_default_pytorch_compatibility(py, torch_module)?;
            torch_module.setattr("__version__", "tenflowers-compat")?
        }
    }

    Ok(())
}

/// Detect PyTorch version from Python environment
fn detect_pytorch_version(py: Python) -> Option<String> {
    let code = std::ffi::CString::new(
        "
try:
    import torch
    torch.__version__
except ImportError:
    None
",
    )
    .ok()?;
    py.eval(&code, None, None)
        .ok()
        .and_then(|version_obj| version_obj.extract::<Option<String>>().ok())
        .flatten()
}

/// Extract major version number from version string
fn extract_major_version(version: &str) -> u32 {
    version
        .split('.')
        .next()
        .and_then(|major| major.parse().ok())
        .unwrap_or(1) // Default to version 1 if parsing fails
}

/// Setup compatibility for PyTorch v1.x
fn setup_pytorch_v1_compatibility(py: Python, torch_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // PyTorch 1.x specific mappings
    torch_module.add_function(wrap_pyfunction!(tensor_ops::zeros, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::ones, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::rand, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::randn, py)?)?;

    // Pinned memory tensor creation functions
    torch_module.add_function(wrap_pyfunction!(tensor_ops::zeros_pinned, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::ones_pinned, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::rand_pinned, py)?)?;

    // Activation functions
    torch_module.add_function(wrap_pyfunction!(neural::relu, py)?)?;
    torch_module.add_function(wrap_pyfunction!(neural::sigmoid, py)?)?;
    torch_module.add_function(wrap_pyfunction!(neural::tanh, py)?)?;

    // Mathematical operations
    torch_module.add_function(wrap_pyfunction!(math_ops::sum, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::mean, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::max, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::min, py)?)?;

    // Tensor operations
    torch_module.add_function(wrap_pyfunction!(tensor_ops::add, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::mul, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::matmul, py)?)?;

    Ok(())
}

/// Setup compatibility for PyTorch v2.x  
fn setup_pytorch_v2_compatibility(py: Python, torch_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Include all v1 functions
    setup_pytorch_v1_compatibility(py, torch_module)?;

    // PyTorch 2.x specific additions
    torch_module.add_function(wrap_pyfunction!(neural::gelu, py)?)?;
    torch_module.add_function(wrap_pyfunction!(neural::swish, py)?)?;
    torch_module.add_function(wrap_pyfunction!(neural::mish, py)?)?;

    // Enhanced tensor operations available in 2.x
    torch_module.add_function(wrap_pyfunction!(math_ops::var, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::std, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::clamp, py)?)?;

    // Tensor manipulation operations
    torch_module.add_function(wrap_pyfunction!(math_ops::cat, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::stack, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::squeeze, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::unsqueeze, py)?)?;

    Ok(())
}

/// Setup default PyTorch compatibility (no version detected)
fn setup_default_pytorch_compatibility(
    py: Python,
    torch_module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    // Use conservative compatibility - include core functions that work across versions
    torch_module.add_function(wrap_pyfunction!(tensor_ops::zeros, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::ones, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::rand, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::randn, py)?)?;

    // Basic activation functions
    torch_module.add_function(wrap_pyfunction!(neural::relu, py)?)?;
    torch_module.add_function(wrap_pyfunction!(neural::sigmoid, py)?)?;
    torch_module.add_function(wrap_pyfunction!(neural::tanh, py)?)?;

    // Core mathematical operations
    torch_module.add_function(wrap_pyfunction!(math_ops::sum, py)?)?;
    torch_module.add_function(wrap_pyfunction!(math_ops::mean, py)?)?;

    // Basic tensor operations
    torch_module.add_function(wrap_pyfunction!(tensor_ops::add, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::mul, py)?)?;
    torch_module.add_function(wrap_pyfunction!(tensor_ops::matmul, py)?)?;

    Ok(())
}

/// Main Python module definition
#[pymodule]
fn tenflowers(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version info
    m.setattr("__version__", "0.1.0-alpha.1")?;
    m.setattr("__author__", "TenfloweRS Team")?;

    // Register core tensor class
    m.add_class::<tensor_ops::PyTensor>()?;

    // Register neural network functions using new modular structure
    neural::register_neural_functions(py, m)?;

    // Register tensor creation functions
    m.add_function(wrap_pyfunction!(tensor_ops::zeros, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::ones, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::rand, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::randn, py)?)?;

    // Register tensor operations
    m.add_function(wrap_pyfunction!(tensor_ops::add, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::mul, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::sub, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::div, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::matmul, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::transpose, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::reshape, py)?)?;

    // Neural network functions are now registered via neural::register_neural_functions() above

    // Register mathematical operations
    m.add_function(wrap_pyfunction!(math_ops::exp, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::log, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::sqrt, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::abs, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::neg, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::sin, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::cos, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::tan, py)?)?;

    // Register reduction operations
    m.add_function(wrap_pyfunction!(math_ops::sum, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::mean, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::max, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::min, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::var, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::standard_deviation, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::std, py)?)?;

    // Register utility operations
    m.add_function(wrap_pyfunction!(math_ops::clamp, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::eq, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::ne, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::lt, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::le, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::gt, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::ge, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::argmax, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::argmin, py)?)?;

    // Register tensor manipulation operations
    m.add_function(wrap_pyfunction!(math_ops::cat, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::stack, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::split, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::squeeze, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::unsqueeze, py)?)?;
    m.add_function(wrap_pyfunction!(math_ops::flatten, py)?)?;

    // Register device management functions
    m.add_function(wrap_pyfunction!(get_default_device, py)?)?;
    m.add_function(wrap_pyfunction!(set_default_device, py)?)?;

    // Register memory profiling functions
    m.add_function(wrap_pyfunction!(enable_memory_profiling, py)?)?;
    m.add_function(wrap_pyfunction!(disable_memory_profiling, py)?)?;
    m.add_function(wrap_pyfunction!(get_memory_info, py)?)?;

    // Register gradient management functions
    m.add_function(wrap_pyfunction!(is_grad_enabled, py)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, py)?)?;

    // Register numpy interop functions
    m.add_function(wrap_pyfunction!(tensor_from_numpy, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_numpy, py)?)?;

    // Add submodules for specialized functionality
    let benchmarks_module = PyModule::new(py, "benchmarks")?;
    benchmarks::register_benchmark_functions(py, &benchmarks_module)?;
    m.add_submodule(&benchmarks_module)?;

    let visualization_module = PyModule::new(py, "visualization")?;
    visualization::register_visualization_functions(py, &visualization_module)?;
    m.add_submodule(&visualization_module)?;

    // Add memory optimization submodule
    let memory_module = PyModule::new(py, "memory")?;
    memory_optimizer::register_memory_functions(py, &memory_module)?;
    m.add_submodule(&memory_module)?;

    // Create a torch-compatible namespace for PyTorch users with version handling
    let torch_module = PyModule::new(py, "torch")?;

    // Detect PyTorch version and setup compatibility layer
    setup_torch_compatibility(py, &torch_module)?;

    m.add_submodule(&torch_module)?;

    Ok(())
}

// Export the module creation function for pyo3-build-config
pub fn create_module() -> PyResult<()> {
    Ok(())
}
