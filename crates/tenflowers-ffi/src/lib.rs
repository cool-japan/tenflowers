//! # TenfloweRS Python Bindings (FFI)
//!
//! Python bindings for the TenfloweRS machine learning framework, providing a Pythonic interface
//! to TenfloweRS's high-performance Rust implementation. This crate enables seamless integration
//! between Python ML workflows and TenfloweRS's native performance.
//!
//! ## Features
//!
//! - **Pythonic API**: Natural Python interface following PyTorch/TensorFlow conventions
//! - **Zero-Copy Interop**: Efficient data transfer between Python and Rust
//! - **NumPy Integration**: Direct conversion between NumPy arrays and TenfloweRS tensors
//! - **Complete Bindings**: Access to all TenfloweRS functionality from Python
//! - **Performance**: Near-native Rust performance from Python
//! - **Type Safety**: Strong typing with Python type hints
//! - **Session Profiling**: Operation-level timing and memory accounting via [`PyProfiler`]
//! - **Stable API Catalogue**: Query symbol stability via [`stable_api`] module
//!
//! ## Installation
//!
//! Install from PyPI (when published):
//!
//! ```bash
//! pip install tenflowers
//! ```
//!
//! Or build from source with [maturin](https://github.com/PyO3/maturin):
//!
//! ```bash
//! cd tenflowers/crates/tenflowers-ffi
//! pip install maturin
//! maturin develop --release
//! ```
//!
//! ## Quick Start (Python)
//!
//! ### Tensor Creation
//!
//! ```python
//! import tenflowers as tf
//!
//! # Zero-filled and one-filled tensors
//! z = tf.zeros([3, 4])          # shape (3, 4), dtype float32
//! o = tf.ones([2, 2])
//!
//! # Uniform random  [0, 1)
//! r = tf.rand([10, 10])
//!
//! # Standard-normal random
//! n = tf.randn([10, 10])
//!
//! # Range and linspace helpers (both return a PyTensor, like zeros/ones/rand)
//! x = tf.arange(0.0, 10.0, 1.0)   # shape [10], values [0, 1, ..., 9]
//! y = tf.linspace(0.0, 1.0, 100)  # shape [100], 100 evenly-spaced values
//! ```
//!
//! ### Basic Tensor Operations
//!
//! ```python
//! import tenflowers as tf
//!
//! a = tf.ones([2, 3])
//! b = tf.ones([2, 3])
//!
//! # Element-wise arithmetic
//! c = tf.add(a, b)
//! d = tf.sub(a, b)
//! e = tf.mul(a, b)
//! f = tf.div(a, b)
//!
//! # Linear algebra
//! x = tf.ones([3, 4])
//! y = tf.ones([4, 2])
//! z = tf.matmul(x, y)   # (3, 2)
//!
//! # Shape manipulation
//! t = tf.ones([6])
//! t2d = tf.reshape(t, [2, 3])
//! t_T = tf.transpose(t2d)
//! ```
//!
//! ### PyTensor Methods
//!
//! ```python
//! import tenflowers as tf
//!
//! t = tf.ones([4, 5])
//!
//! # Shape and metadata
//! print(t.shape())          # [4, 5]
//! print(t.ndim())           # 2
//! print(t.size())           # 20
//! print(t.numel())          # 20  (alias)
//! print(t.dtype())          # "float32"
//! print(t.is_matrix())      # True
//!
//! # Gradient tracking
//! t.set_requires_grad(True)
//! print(t.requires_grad())  # True
//!
//! # Python protocols
//! print(len(t))             # 4  (first dimension)
//! for row in t:
//!     print(row.shape())    # [5]
//! ```
//!
//! ### Math Operations
//!
//! ```python
//! import tenflowers as tf
//!
//! t = tf.ones([3, 4])
//!
//! # Unary math
//! tf.exp(t)
//! tf.log(t)
//! tf.sqrt(t)
//! tf.abs(t)
//! tf.neg(t)
//! tf.sin(t); tf.cos(t); tf.tan(t)
//!
//! # Reductions (return scalars as float)
//! s   = tf.sum(t)
//! m   = tf.mean(t)
//! mx  = tf.max(t)
//! mn  = tf.min(t)
//! v   = tf.var(t)
//! sd  = tf.std(t)
//!
//! # Comparison
//! tf.eq(t, t)
//! tf.lt(t, tf.zeros([3, 4]))
//!
//! # Concatenation
//! a = tf.ones([2, 3])
//! b = tf.zeros([2, 3])
//! ab = tf.cat([a, b], dim=0)   # (4, 3)
//! ab2 = tf.stack([a, b], dim=0)  # (2, 2, 3)
//! ```
//!
//! ### Activation Functions
//!
//! ```python
//! import tenflowers as tf
//!
//! x = tf.ones([4, 4])
//!
//! # Basic activations
//! tf.relu(x)
//! tf.sigmoid(x)
//! tf.tanh(x)
//! tf.gelu(x)
//! tf.swish(x)
//! tf.mish(x)
//! tf.softmax(x, dim=-1)
//! tf.log_softmax(x, dim=-1)
//!
//! # Additional activations
//! tf.leaky_relu(x)
//! tf.elu(x)
//! tf.relu6(x)
//! tf.hardswish(x)
//! tf.selu(x)
//! tf.silu(x)
//! ```
//!
//! ### Neural Network Layers
//!
//! ```python
//! import tenflowers as tf
//!
//! # Dense (fully-connected) layer
//! layer = tf.PyDense(in_features=128, out_features=64, use_bias=True, activation='relu')
//! x = tf.ones([8, 128])
//! y = layer.forward(x)  # (8, 64)
//!
//! # Sequential model
//! model = tf.PySequential()
//! model.add(tf.PyDense(128, 256, activation='relu'))
//! model.add(tf.PyDense(256, 10,  activation=None))
//! model.train()          # switch to training mode
//! out = model.forward(tf.ones([4, 128]))
//! print(model.num_parameters())  # 128*256+256 + 256*10+10
//!
//! # Convolutional layers
//! conv = tf.PyConv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1)
//! pool = tf.PyMaxPool2D(kernel_size=2, stride=2)
//!
//! # Normalization
//! bn = tf.PyBatchNorm1d(num_features=128)
//! ln = tf.PyLayerNorm(normalized_shape=[128])
//!
//! # Recurrent layers
//! lstm = tf.PyLSTM(input_size=64, hidden_size=128, num_layers=2)
//! gru  = tf.PyGRU(input_size=64, hidden_size=128)
//!
//! # Attention
//! attn = tf.PyMultiheadAttention(embed_dim=256, num_heads=8)
//!
//! # Transformer building blocks
//! enc = tf.PyTransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)
//! ```
//!
//! ### Optimizers
//!
//! ```python
//! import tenflowers as tf
//!
//! # Standard optimizers
//! adam   = tf.Adam(learning_rate=1e-3)
//! adamw  = tf.AdamW(learning_rate=1e-3)
//! sgd    = tf.SGD(learning_rate=1e-2)
//! rmsprop = tf.RMSprop(learning_rate=1e-3)
//!
//! # Extended optimizers (new in 0.1.2)
//! radam   = tf.RAdam(learning_rate=1e-3)
//! nadam   = tf.Nadam(learning_rate=1e-3)
//! adagrad = tf.AdaGrad(learning_rate=1e-2)
//! adadelta = tf.AdaDelta()
//! adabelief = tf.AdaBelief(learning_rate=1e-3)
//!
//! # Learning rate schedulers
//! step_lr = tf.PyStepLR(step_size=10, gamma=0.1)
//! cosine  = tf.PyCosineAnnealingLR(T_max=100, eta_min=1e-6)
//!
//! # Optimizer state inspection / checkpointing
//! state = adam.state_dict()
//! adam.load_state_dict(state)
//! ```
//!
//! ### Loss Functions
//!
//! ```python
//! import tenflowers as tf
//!
//! y_pred = tf.ones([4, 10])
//! y_true = tf.zeros([4, 10])
//!
//! loss = tf.mse_loss(y_pred, y_true)
//! bce  = tf.binary_cross_entropy(y_pred, y_true)
//! ce   = tf.cross_entropy(y_pred, y_true)
//! l1   = tf.l1_loss(y_pred, y_true)
//! huber = tf.smooth_l1_loss(y_pred, y_true)
//! ```
//!
//! ### Gradient Tape (Automatic Differentiation)
//!
//! ```python
//! import tenflowers as tf
//!
//! x = tf.ones([3, 3])
//! x.set_requires_grad(True)
//!
//! tape = tf.PyGradientTape()
//! tx = tape.watch(x)
//!
//! # Forward computation
//! y = tf.matmul(x, x)   # x²
//! ty = tape.watch(y)
//!
//! grad = tape.gradient(ty, tx)
//! print(grad.shape())   # [3, 3]
//!
//! # Jacobian
//! J = tf.jacobian(ty, [tx])
//!
//! # Context manager style
//! with tf.PyGradientContext() as ctx:
//!     z = tf.relu(x)
//! ```
//!
//! ### Device Management
//!
//! ```python
//! import tenflowers as tf
//!
//! # Query default device
//! dev = tf.get_default_device()   # "cpu"
//!
//! # Switch to GPU
//! tf.set_default_device("gpu:0")
//!
//! # Device objects
//! cpu  = tf.Device.cpu()
//! gpu0 = tf.Device.gpu(0)
//! print(repr(gpu0))               # "Device.gpu(0)"
//! print(gpu0.is_gpu)              # True
//! print(gpu0.device_id)           # 0
//!
//! # Parse from string
//! dev = tf.Device.from_string("rocm:1")
//!
//! # Gradient control
//! tf.set_grad_enabled(False)   # disable autograd globally
//! print(tf.is_grad_enabled())  # False
//! tf.set_grad_enabled(True)
//! ```
//!
//! ### Data Types
//!
//! ```python
//! import tenflowers as tf
//!
//! # Dtype constants exposed at module level
//! print(tf.float32)   # DType.Float32
//! print(tf.int64)     # DType.Int64
//!
//! # DType objects
//! dt = tf.DType.from_string("bfloat16")
//! print(dt.size_bytes())      # 2
//! print(dt.is_floating_point())  # True
//!
//! # Type promotion
//! promoted = tf.promote_types(tf.float32, tf.float64)  # float64
//! result   = tf.result_type(tf.int32, tf.float32)      # float32
//! ok       = tf.is_safe_cast(tf.float32, tf.float64)   # True
//! ```
//!
//! ### NumPy Interoperability
//!
//! ```python
//! import numpy as np
//! import tenflowers as tf
//!
//! # NumPy array -> TenfloweRS tensor
//! arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
//! t = tf.tensor_from_numpy(arr)
//! print(t.shape())   # [2, 2]
//!
//! # TenfloweRS tensor -> NumPy array
//! back = tf.tensor_to_numpy(t)
//! print(type(back))  # <class 'numpy.ndarray'>
//! ```
//!
//! ### Evaluation Metrics
//!
//! ```python
//! import tenflowers as tf
//!
//! logits  = tf.rand([100, 10])   # 100 samples, 10 classes
//! labels  = tf.zeros([100])      # ground-truth class ids
//!
//! acc     = tf.accuracy(logits, labels)
//! mse     = tf.mean_squared_error(logits, logits)
//! mae     = tf.mean_absolute_error(logits, logits)
//! r2      = tf.r2_score(logits, logits)
//! prec, recall, f1 = tf.precision_recall_f1(logits, labels)
//! top5    = tf.top_k_accuracy(logits, labels, k=5)
//! auc     = tf.auc_roc(logits, labels)
//! cm      = tf.confusion_matrix(logits, labels)
//! ```
//!
//! ### Profiling
//!
//! ```python
//! import tenflowers as tf
//! import time
//!
//! profiler = tf.PyProfiler()
//! profiler.start_session()
//!
//! # Instrument operations manually
//! t0 = time.perf_counter_ns()
//! a  = tf.matmul(tf.ones([512, 512]), tf.ones([512, 512]))
//! t1 = time.perf_counter_ns()
//! profiler.record("matmul", t1 - t0, a.memory_usage())
//!
//! t0 = time.perf_counter_ns()
//! b  = tf.relu(a)
//! t1 = time.perf_counter_ns()
//! profiler.record("relu", t1 - t0, b.memory_usage())
//!
//! report = profiler.end_session()
//! print(f"ops={report.total_ops}, total_ns={report.total_time_ns}")
//! for rec in report.top_ops(3):
//!     print(f"  {rec.op_name}: {rec.duration_ns} ns")
//! ```
//!
//! ### Serialization
//!
//! ```python
//! import tenflowers as tf
//!
//! # Save / load a single tensor
//! t = tf.ones([4, 4])
//! tf.save_tensor(t, "/tmp/weights.bin")
//! t2 = tf.load_tensor("/tmp/weights.bin")
//!
//! # Checkpoint manager for full model state
//! ckpt = tf.PyCheckpointManager("/tmp/checkpoints")
//! state = {"layer0.weight": t, "layer0.bias": tf.zeros([4])}
//! tf.save_state_dict(state, "/tmp/model_epoch1.bin")
//! loaded = tf.load_state_dict("/tmp/model_epoch1.bin")
//! ```
//!
//! ### Stable API Catalogue
//!
//! ```python
//! import tenflowers as tf
//!
//! ver = tf.stable_api_version()
//! print(f"API {ver.version_string()} ({ver.stability})")
//! # "API 0.2.1 (Beta)"
//!
//! surface = tf.stable_api_surface()
//! print(f"{surface.count()} entries")
//!
//! for entry in surface.stable_entries():
//!     print(f"  STABLE  {entry.name:30s}  since {entry.since_version}")
//!
//! for entry in surface.entries_since("0.1.2"):
//!     print(f"  NEW     {entry.name}")
//! ```
//!
//! ### Memory Profiling
//!
//! ```python
//! import tenflowers as tf
//!
//! tf.enable_memory_profiling()
//!
//! a = tf.ones([1000, 1000])
//! b = tf.matmul(a, a)
//!
//! current_bytes, peak_bytes = tf.get_memory_info()
//! print(f"current={current_bytes} peak={peak_bytes}")
//!
//! tf.disable_memory_profiling()
//! ```
//!
//! ### Utility Helpers
//!
//! ```python
//! import tenflowers as tf
//!
//! t = tf.ones([4, 5])
//!
//! # Inspection
//! info = tf.tensor_info(t)       # dict with shape/ndim/numel/dtype
//! tf.print_tensor_info(t)        # human-readable summary to stdout
//! tf.tensor_summary(t)           # compact string
//! print(tf.numel(t))             # 20
//! print(tf.tensor_memory_bytes(t))  # 80
//! print(tf.tensor_memory_str(t))    # "80 B"
//! print(tf.format_bytes(1048576))   # "1.00 MB"
//!
//! # Shape predicates
//! tf.is_scalar(t)     # False
//! tf.is_vector(t)     # False
//! tf.is_matrix(t)     # True
//! tf.same_shape(t, t) # True
//!
//! # Broadcasting helpers
//! shape = tf.broadcast_shape([2, 1], [1, 3])  # [2, 3]
//! ok    = tf.is_broadcastable([2, 1], [1, 3]) # True
//!
//! # Device info
//! print(tf.get_device_info())
//! print(tf.is_gpu_available())
//! print(tf.version())
//! ```
//!
//! ## Architecture
//!
//! This crate provides Python bindings through PyO3:
//!
//! - [`tensor_ops`]: Tensor creation, arithmetic, and shape manipulation
//! - [`math_ops`]: Mathematical, trigonometric, statistical, and reduction operations
//! - [`neural`]: Neural network layers, optimizers, loss functions, and autograd
//! - [`metrics`]: Model evaluation metrics (accuracy, MSE, AUC-ROC, etc.)
//! - [`utils`]: Utility functions for tensor inspection and broadcasting
//! - [`serialization`]: Model serialization and checkpointing
//! - [`visualization`]: Training visualization and monitoring
//! - [`profiling`]: Session-based operation profiler ([`PyProfiler`])
//! - [`stable_api`]: API stability catalogue and versioning
//! - [`device`]: Device abstraction (CPU / GPU / ROCm)
//! - [`dtype`]: Data type abstraction (float32 / float16 / bf16 / int* / uint*)
//!
//! ## Error Handling
//!
//! All Python-facing functions return `PyResult<T>`. On error, they raise Python
//! exceptions mapped from Rust error types:
//!
//! ```python
//! import tenflowers as tf
//!
//! try:
//!     a = tf.ones([2, 3])
//!     b = tf.ones([4, 5])
//!     tf.matmul(a, b)  # shape mismatch -> raises RuntimeError
//! except RuntimeError as e:
//!     print(f"Error: {e}")
//! ```
//!
//! ## Integration with Python Ecosystem
//!
//! TenfloweRS integrates with:
//! - **NumPy**: `tensor_from_numpy` / `tensor_to_numpy` for zero-copy conversion
//! - **PyTorch**: `tenflowers.torch` compatibility sub-module
//! - **TensorFlow**: ONNX-based model exchange (planned)
//! - **Pandas**: DataFrame integration for datasets (planned)
//!
//! ## Building the Python Package
//!
//! ```bash
//! # Install maturin
//! pip install maturin
//!
//! # Development build (editable install)
//! maturin develop
//!
//! # Release build
//! maturin build --release
//!
//! # Build wheel for distribution
//! maturin build --release --out dist/
//! ```
//!
//! ## Running Tests
//!
//! ```bash
//! # Rust unit tests
//! cargo test -p tenflowers-ffi
//!
//! # Python integration tests
//! pytest tests/
//! ```

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
pub mod device; // Device abstraction (CPU/GPU/ROCm as Python classes)
pub mod dtype; // Data type abstraction for f16/bf16/etc support
pub mod dtype_promotion;
pub mod eager_execution_optimizer;
pub mod error_mapping; // Error mapping and exception handling
pub mod implicit_autograd; // PyTorch-style eager `.backward()` / `.grad` support
pub mod large_model_support;
pub mod memory_optimizer;
pub mod metrics; // Model evaluation metrics
pub mod serialization; // Model serialization and checkpointing
pub mod test_module;

// Refactored modular structures
pub mod neural; // New modular neural operations
pub mod visualization; // New modular visualization

// Advanced profiling integration
pub mod profiling;

// Stable API surface catalogue
pub mod stable_api;

// Gradient parity validation (pure Rust, no PyO3 dependency)
pub mod gradient_parity;
pub use gradient_parity::{
    gradients_are_close, numeric_jacobian, GradientParityChecker, GradientParityResult,
};

// Core FFI modules
pub mod math_ops;
pub mod tensor_ops;
pub mod utils; // Utility functions for common operations

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

/// Return the current default device as a string, e.g. `"cpu"` or `"gpu:0"`.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// print(tf.get_default_device())  # "cpu"
/// ```
#[pyfunction]
fn get_default_device() -> String {
    let device = get_default_device_lock()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match *device {
        Device::Cpu => "cpu".to_string(),
        #[cfg(feature = "gpu")]
        Device::Gpu(id) => format!("gpu:{}", id),
        #[cfg(feature = "gpu")]
        Device::Rocm(id) => format!("rocm:{}", id),
    }
}

/// Set the global default device used by newly created tensors.
///
/// Accepted strings: `"cpu"`, `"gpu:N"` (requires the `gpu` Cargo feature),
/// or `"rocm:N"` (currently returns an error — ROCm is not yet enabled).
///
/// # Errors
///
/// Raises `ValueError` for unrecognised device strings or when GPU support
/// is not compiled in.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// tf.set_default_device("gpu:0")
/// print(tf.get_default_device())  # "gpu:0"
/// ```
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

    *get_default_device_lock()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = device;
    Ok(())
}

/// Enable memory usage tracking.
///
/// Once enabled, every tensor allocation updates the internal `current_memory`
/// and `peak_memory` counters accessible through [`get_memory_info`].
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// tf.enable_memory_profiling()
/// ```
#[pyfunction]
fn enable_memory_profiling() {
    let mut state = get_memory_profiling_lock()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.enabled = true;
}

/// Disable memory usage tracking.
///
/// Has no effect if profiling was not previously enabled.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// tf.disable_memory_profiling()
/// ```
#[pyfunction]
fn disable_memory_profiling() {
    let mut state = get_memory_profiling_lock()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.enabled = false;
}

/// Return `(current_bytes, peak_bytes)` from the memory profiling state.
///
/// Both values are zero when profiling has never been enabled.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// tf.enable_memory_profiling()
/// current, peak = tf.get_memory_info()
/// print(f"current={current} peak={peak}")
/// ```
#[pyfunction]
fn get_memory_info() -> PyResult<(usize, usize)> {
    let state = get_memory_profiling_lock()
        .read()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("memory profiling lock poisoned"))?;
    Ok((state.current_memory, state.peak_memory))
}

/// Return `True` if autograd is currently enabled (the default).
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// print(tf.is_grad_enabled())   # True
/// tf.set_grad_enabled(False)
/// print(tf.is_grad_enabled())   # False
/// ```
#[pyfunction]
fn is_grad_enabled() -> bool {
    tenflowers_autograd::no_grad::is_grad_enabled()
}

/// Enable or disable the autograd engine globally.
///
/// Equivalent to PyTorch's `torch.set_grad_enabled(mode)`.
/// When disabled, no gradient information is recorded and all gradient-related
/// operations are no-ops.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// tf.set_grad_enabled(False)   # inference-only mode
/// # ... run model forward pass ...
/// tf.set_grad_enabled(True)    # restore for training
/// ```
#[pyfunction]
fn set_grad_enabled(enabled: bool) {
    tenflowers_autograd::no_grad::set_grad_enabled(enabled);
}

/// Convert a NumPy `ndarray` (dtype `float32`) to a [`tensor_ops::PyTensor`].
///
/// The array is read element-by-element (row-major) and the resulting tensor
/// shares the same shape.  Raises `RuntimeError` if the conversion fails.
///
/// # Python Example
///
/// ```python
/// import numpy as np
/// import tenflowers as tf
///
/// arr = np.ones((3, 4), dtype=np.float32)
/// t   = tf.tensor_from_numpy(arr)
/// print(t.shape())  # [3, 4]
/// ```
#[pyfunction]
fn tensor_from_numpy(py: Python, array: Bound<'_, PyAny>) -> PyResult<tensor_ops::PyTensor> {
    use scirs2_numpy::PyReadonlyArrayDyn;

    // Convert numpy array to PyReadonlyArrayDyn<f32>. Only float32 arrays are
    // accepted today; other dtypes (float64, int32, ...) are rejected here with
    // a clear message rather than the confusing generic extraction error.
    let np_array: PyReadonlyArrayDyn<f32> = array.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "tensor_from_numpy expects a numpy array with dtype=float32; \
             cast it first, e.g. `arr.astype(np.float32)`",
        )
    })?;
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

/// Convert a [`tensor_ops::PyTensor`] to a NumPy `ndarray` (dtype `float32`).
///
/// The tensor data is laid out in C-contiguous (row-major) order.  Raises
/// `RuntimeError` if the internal data cannot be extracted.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// import numpy as np
///
/// t   = tf.ones([2, 3])
/// arr = tf.tensor_to_numpy(t)
/// print(arr.shape)        # (2, 3)
/// print(arr.dtype)        # float32
/// ```
#[pyfunction]
fn tensor_to_numpy(py: Python, tensor: &tensor_ops::PyTensor) -> PyResult<Py<PyAny>> {
    // use numpy::PyArrayDyn; // Unused for now

    // Get tensor data and shape
    let shape = tensor.shape();
    let data = tensor.tensor.to_vec().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e))
    })?;

    // Create multi-dimensional numpy array from tensor data and shape
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use scirs2_numpy::{PyArray1, PyArrayDyn};

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
    // Sourced directly from Cargo.toml (workspace version) so it can never drift.
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;
    m.setattr("__author__", "TenfloweRS Team")?;

    // Register custom exceptions
    error_mapping::register_exceptions(py, m)?;

    // Register core tensor class and iterator
    m.add_class::<tensor_ops::PyTensor>()?;
    m.add_class::<tensor_ops::PyTensorIter>()?;

    // Register device abstraction classes
    m.add_class::<device::PyDevice>()?;
    m.add_class::<device::PyDeviceKind>()?;

    // Register dtype system
    m.add_class::<dtype::PyDType>()?;
    m.add_function(wrap_pyfunction!(dtype::is_safe_cast_py, py)?)?;
    m.add_function(wrap_pyfunction!(dtype::result_type, py)?)?;
    m.add_function(wrap_pyfunction!(dtype::promote_types, py)?)?;

    // Add dtype constants for convenience
    m.setattr("float32", dtype::dtypes::FLOAT32)?;
    m.setattr("float64", dtype::dtypes::FLOAT64)?;
    m.setattr("float16", dtype::dtypes::FLOAT16)?;
    m.setattr("bfloat16", dtype::dtypes::BFLOAT16)?;
    m.setattr("int8", dtype::dtypes::INT8)?;
    m.setattr("int16", dtype::dtypes::INT16)?;
    m.setattr("int32", dtype::dtypes::INT32)?;
    m.setattr("int64", dtype::dtypes::INT64)?;
    m.setattr("uint8", dtype::dtypes::UINT8)?;
    m.setattr("uint16", dtype::dtypes::UINT16)?;
    m.setattr("uint32", dtype::dtypes::UINT32)?;
    m.setattr("uint64", dtype::dtypes::UINT64)?;
    m.setattr("bool", dtype::dtypes::BOOL)?;

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
    m.add_function(wrap_pyfunction!(tensor_ops::slice, py)?)?;

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

    // Register utility functions for tensor inspection and manipulation
    m.add_function(wrap_pyfunction!(utils::tensor_info, py)?)?;
    m.add_function(wrap_pyfunction!(utils::same_shape, py)?)?;
    m.add_function(wrap_pyfunction!(utils::is_scalar, py)?)?;
    m.add_function(wrap_pyfunction!(utils::is_vector, py)?)?;
    m.add_function(wrap_pyfunction!(utils::is_matrix, py)?)?;
    m.add_function(wrap_pyfunction!(utils::numel, py)?)?;
    m.add_function(wrap_pyfunction!(utils::validate_shapes, py)?)?;
    m.add_function(wrap_pyfunction!(utils::tensor_summary, py)?)?;
    m.add_function(wrap_pyfunction!(utils::all_same_shape, py)?)?;
    m.add_function(wrap_pyfunction!(utils::broadcast_shape, py)?)?;
    m.add_function(wrap_pyfunction!(utils::is_broadcastable, py)?)?;
    m.add_function(wrap_pyfunction!(utils::tensor_memory_bytes, py)?)?;
    m.add_function(wrap_pyfunction!(utils::tensor_memory_str, py)?)?;
    m.add_function(wrap_pyfunction!(utils::format_bytes, py)?)?;
    m.add_function(wrap_pyfunction!(utils::print_tensor_info, py)?)?;
    m.add_function(wrap_pyfunction!(utils::validate_dimension, py)?)?;
    m.add_function(wrap_pyfunction!(utils::normalize_dimension, py)?)?;
    m.add_function(wrap_pyfunction!(utils::arange, py)?)?;
    m.add_function(wrap_pyfunction!(utils::linspace, py)?)?;
    m.add_function(wrap_pyfunction!(utils::get_device_info, py)?)?;
    m.add_function(wrap_pyfunction!(utils::is_gpu_available, py)?)?;
    m.add_function(wrap_pyfunction!(utils::version, py)?)?;

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

    // Register serialization functions
    m.add_class::<serialization::PyCheckpointManager>()?;
    m.add_function(wrap_pyfunction!(serialization::save_tensor, py)?)?;
    m.add_function(wrap_pyfunction!(serialization::load_tensor, py)?)?;
    m.add_function(wrap_pyfunction!(serialization::save_state_dict, py)?)?;
    m.add_function(wrap_pyfunction!(serialization::load_state_dict, py)?)?;

    // Register profiling classes
    profiling::register_profiling_classes(py, m)?;

    // Register stable API surface
    stable_api::register_stable_api(py, m)?;

    // Register evaluation metrics
    m.add_function(wrap_pyfunction!(metrics::accuracy, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::precision_recall_f1, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::mean_squared_error, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::mean_absolute_error, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::r2_score, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::top_k_accuracy, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::auc_roc, py)?)?;
    m.add_function(wrap_pyfunction!(metrics::confusion_matrix, py)?)?;

    // Register dtype promotion utilities (NumPy/PyTorch-compatible promotion rules)
    dtype_promotion::register_dtype_promotion_functions(py, m)?;

    // Register large-model support utilities (gradient checkpointing, parameter sharding)
    large_model_support::register_large_model_functions(py, m)?;

    // Register performance bottleneck detection utilities
    bottleneck_detection::register_bottleneck_detection_functions(py, m)?;

    // Register eager execution optimizer utilities
    eager_execution_optimizer::register_eager_execution_functions(py, m)?;

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
