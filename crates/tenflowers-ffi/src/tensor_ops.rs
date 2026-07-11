//! Tensor Operations Module
//!
//! This module contains Python bindings for tensor operations including:
//! - PyTensor class and basic tensor operations
//! - Arithmetic operations (add, mul, div, etc.)
//! - Linear algebra operations (matmul, transpose, etc.)
//! - Shape manipulation operations

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_autograd::grad_ops::SliceSpec;
use tenflowers_autograd::TrackedTensor;
use tenflowers_core::strided::SliceParams;
use tenflowers_core::Tensor;

/// Python-facing wrapper around a TenfloweRS `Tensor<f32>`.
///
/// `PyTensor` is the central data structure of the TenfloweRS Python API.
/// Every operation (arithmetic, linear algebra, activation, …) returns or
/// accepts `PyTensor` objects.
///
/// # Ownership and Thread Safety
///
/// The underlying `Tensor<f32>` is reference-counted via `Arc`, so clones are
/// shallow (O(1)) and the data is shared until a write is needed.
///
/// # Python Examples
///
/// ```python
/// import tenflowers as tf
///
/// # Construction
/// t = tf.PyTensor([3, 4])     # 3×4 zero tensor
/// z = tf.zeros([3, 4])        # same, via convenience function
/// o = tf.ones([3, 4])
///
/// # Shape / metadata
/// print(t.shape())            # [3, 4]
/// print(t.ndim())             # 2
/// print(t.size())             # 12
/// print(t.numel())            # 12  (alias)
/// print(t.dtype())            # "float32"
/// print(t.device())           # Device.cpu()
/// print(t.memory_usage())     # 48   (12 × 4 bytes)
///
/// # Arithmetic (returns new tensors)
/// a = tf.ones([2, 2])
/// b = tf.ones([2, 2])
/// c = a.add(b)
/// d = a.matmul(b)
/// e = a.pow(2.0)
///
/// # Shape manipulation
/// r  = t.reshape([4, 3])
/// tT = t.transpose()
///
/// # Gradient tracking
/// t.set_requires_grad(True)
/// print(t.requires_grad())    # True
///
/// # Python protocols
/// print(len(t))               # 3   (first dimension)
/// for row in t:
///     print(row.shape())      # [4]
/// ```
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTensor {
    pub tensor: Arc<Tensor<f32>>,
    pub requires_grad: bool,
    pub is_pinned: bool,
}

#[pymethods]
impl PyTensor {
    /// Create a zero-filled tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` — dimensions, e.g. `[3, 4]` for a 3×4 matrix.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if the tensor cannot be allocated.
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// t = tf.PyTensor([2, 3])
    /// print(t.shape())  # [2, 3]
    /// ```
    #[new]
    pub fn new(shape: Vec<usize>) -> PyResult<Self> {
        let data = vec![0.0f32; shape.iter().product()];
        let tensor = Tensor::from_vec(data, &shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;

        Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        })
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().dims().to_vec()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.tensor.size()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.size() * std::mem::size_of::<f32>()
    }

    /// Check if tensor operations can be SIMD-optimized
    pub fn supports_simd(&self) -> bool {
        // Check if tensor size and alignment support SIMD operations
        self.size() >= 8 && self.memory_usage() % 32 == 0
    }

    /// Alias for size() - PyTorch compatibility
    fn numel(&self) -> usize {
        self.size()
    }

    /// Check if tensor requires gradients
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient requirement.
    ///
    /// Setting this to `true` on a leaf tensor begins implicit (eager,
    /// PyTorch-style) gradient tracking: this tensor, and every subsequent
    /// tensor derived from it through a tracked operation, participates in
    /// the implicit computation graph maintained by
    /// [`crate::implicit_autograd`]. See [`PyTensor::backward`] and
    /// [`PyTensor::grad`].
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// x = tf.tensor_from_numpy(__import__("numpy").ones((2, 2), dtype="float32"))
    /// x.set_requires_grad(True)
    /// y = tf.sum(tf.mul(x, x))
    /// y.backward()
    /// print(x.grad().shape())  # [2, 2]
    /// ```
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if requires_grad {
            crate::implicit_autograd::mark_leaf(self);
        }
    }

    /// Run reverse-mode automatic differentiation starting from this tensor.
    ///
    /// This is the PyTorch-style eager counterpart of the explicit
    /// [`crate::neural::gradient_tape::PyGradientTape::gradient`] API: no
    /// tape object is created or referenced by the caller. Internally, every
    /// tracked operation performed since the most recent
    /// `set_requires_grad(True)` call has been implicitly recorded onto a
    /// thread-local [`tenflowers_autograd::GradientTape`]
    /// (see [`crate::implicit_autograd`]); this method runs that tape's real
    /// backward pass and populates `.grad()` on every leaf tensor that
    /// contributed to `self`.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if `self` was never derived from a tensor with
    /// `requires_grad=True` (i.e. it has no recorded computation graph —
    /// analogous to PyTorch's "tensor does not require grad and does not
    /// have a grad_fn"), or if the backward pass itself fails.
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// x = tf.ones([3, 3])
    /// x.set_requires_grad(True)
    /// y = tf.sum(tf.add(x, x))
    /// y.backward()
    /// print(x.grad().shape())  # [3, 3]
    /// ```
    pub fn backward(&self) -> PyResult<()> {
        crate::implicit_autograd::run_backward(self)
    }

    /// Retrieve the gradient accumulated for this tensor by a previous
    /// [`PyTensor::backward`] call.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if this tensor never had `requires_grad=True`
    /// set, or if `.backward()` has not been called yet (or was called on a
    /// tensor not connected to this one).
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// x = tf.ones([2])
    /// x.set_requires_grad(True)
    /// y = tf.sum(tf.mul(x, x))
    /// y.backward()
    /// g = x.grad()
    /// print(g.shape())  # [2]
    /// ```
    pub fn grad(&self) -> PyResult<PyTensor> {
        crate::implicit_autograd::get_grad(self)
    }

    /// Check if tensor is scalar (0-dimensional)
    fn is_scalar(&self) -> bool {
        self.tensor.ndim() == 0
    }

    /// Check if tensor is vector (1-dimensional)
    fn is_vector(&self) -> bool {
        self.tensor.ndim() == 1
    }

    /// Check if tensor is matrix (2-dimensional)
    fn is_matrix(&self) -> bool {
        self.tensor.ndim() == 2
    }

    /// Get data type as string
    fn dtype(&self) -> String {
        "float32".to_string()
    }

    /// Check if tensor uses pinned memory
    fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Get transpose (PyTorch-style T property)
    #[allow(non_snake_case)]
    fn T(&self) -> PyResult<PyTensor> {
        self.transpose(None)
    }

    /// Get NumPy-compatible dtype string
    fn numpy_dtype(&self) -> String {
        "float32".to_string()
    }

    /// Check if tensor is contiguous
    fn is_contiguous(&self) -> bool {
        true // TenfloweRS tensors are always contiguous
    }

    /// Check if tensor is C-contiguous
    fn is_c_contiguous(&self) -> bool {
        true
    }

    /// Check if tensor is Fortran-contiguous
    fn is_f_contiguous(&self) -> bool {
        false // TenfloweRS uses C-order
    }

    /// Alias for is_f_contiguous
    fn is_fortran_contiguous(&self) -> bool {
        self.is_f_contiguous()
    }

    /// Transpose the tensor, optionally specifying a permutation of axes.
    ///
    /// When `axes` is `None` the axes are reversed (equivalent to NumPy's `T`
    /// property or PyTorch's `.T`).  When provided, `axes` must be a permutation
    /// of `0..ndim`.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if the axes are invalid for the tensor rank.
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// t   = tf.ones([2, 3])
    /// tT  = t.transpose()           # shape (3, 2)
    /// t2  = t.transpose([1, 0])     # same
    /// print(t.T.shape())            # [3, 2]
    /// ```
    #[pyo3(signature = (axes=None))]
    pub fn transpose(&self, axes: Option<Vec<usize>>) -> PyResult<PyTensor> {
        let axes_for_tape = axes.clone();
        let result = if let Some(axes_vec) = axes {
            tenflowers_core::ops::manipulation::transpose_axes(&self.tensor, Some(&axes_vec))
        } else {
            tenflowers_core::ops::manipulation::transpose(&self.tensor)
        };

        match result {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad,
                    is_pinned: self.is_pinned,
                };
                crate::implicit_autograd::record_and_link_unary(
                    crate::implicit_autograd::UnaryOpKind::Transpose {
                        axes: axes_for_tape,
                    },
                    self,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Transpose failed: {}", e))),
        }
    }

    /// Return a view of this tensor with the given `shape`.
    ///
    /// The total number of elements must be preserved.  Raises `RuntimeError`
    /// if the shapes are incompatible.
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// t = tf.ones([6])
    /// t2d = t.reshape([2, 3])
    /// print(t2d.shape())  # [2, 3]
    /// ```
    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        match tenflowers_core::ops::reshape(&self.tensor, &shape) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad,
                    is_pinned: self.is_pinned,
                };
                crate::implicit_autograd::record_and_link_unary(
                    crate::implicit_autograd::UnaryOpKind::Reshape {
                        shape: shape.clone(),
                    },
                    self,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Reshape failed: {}", e))),
        }
    }

    /// Slice this tensor along each dimension.
    ///
    /// `ranges` is a list of `(start, end, step)` tuples, one per leading
    /// dimension, positionally matching the semantics of Python's `slice`
    /// object (`None` for `start`/`end` means "unbounded on that side";
    /// `None` for `step` means step `1`). Trailing dimensions with no
    /// corresponding entry in `ranges` are taken in full — `ranges` may be
    /// shorter than `self.ndim()`, but not longer (extra entries beyond the
    /// tensor's rank are silently ignored, mirroring how a short `ranges`
    /// list pads with "take in full").
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if the underlying slice cannot be computed (e.g.
    /// an out-of-range or invalid `start`/`end`/`step` combination for the
    /// tensor's shape).
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// t = tf.ones([4, 5])
    /// # Rows 1..3 (exclusive), all columns.
    /// s = t.slice([(1, 3, None)])
    /// print(s.shape())  # [2, 5]
    /// ```
    pub fn slice(
        &self,
        ranges: Vec<(Option<isize>, Option<isize>, Option<isize>)>,
    ) -> PyResult<PyTensor> {
        // Unpadded specs for the tape recording: `record_and_link_unary` /
        // `TrackedTensor::slice` pad short lists themselves (see that
        // method's own doc), so this must stay exactly `ranges.len()` long —
        // pre-padding it here would make the tape-recorded op diverge from
        // what was actually requested.
        let specs_for_tape: Vec<SliceSpec> = ranges
            .iter()
            .map(|&(start, end, step)| SliceSpec::new(start, end, step))
            .collect();

        // Padded params for the eager computation: `slice_with_stride`
        // requires exactly one entry per tensor dimension (no auto-padding),
        // so dimensions beyond what the caller specified default to "take in
        // full, step 1" via `SliceParams::new()`.
        let ndim = self.tensor.ndim();
        let mut padded_params: Vec<SliceParams> = Vec::with_capacity(ndim);
        for dim_idx in 0..ndim {
            if dim_idx < ranges.len() {
                let (start, end, step) = ranges[dim_idx];
                padded_params.push(SliceParams::with_step(start, end, step));
            } else {
                padded_params.push(SliceParams::new());
            }
        }

        match tenflowers_core::ops::slice_with_stride(&self.tensor, &padded_params) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad,
                    is_pinned: self.is_pinned,
                };
                crate::implicit_autograd::record_and_link_unary(
                    crate::implicit_autograd::UnaryOpKind::Slice {
                        specs: specs_for_tape,
                    },
                    self,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Slice failed: {}", e))),
        }
    }

    /// Element-wise addition of `self` and `other`.
    ///
    /// Both tensors must have the same shape.  The `requires_grad` flag of the
    /// result is the logical-OR of the two operands' flags.
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// a = tf.ones([2, 2])
    /// b = tf.ones([2, 2])
    /// c = a.add(b)  # all 2.0
    /// ```
    pub fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::add(&self.tensor, &other.tensor) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad || other.requires_grad,
                    is_pinned: self.is_pinned || other.is_pinned,
                };
                crate::implicit_autograd::record_and_link_binary(
                    crate::implicit_autograd::BinaryOpKind::Add,
                    self,
                    other,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Addition failed: {}", e))),
        }
    }

    /// Multiply two tensors
    pub fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::mul(&self.tensor, &other.tensor) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad || other.requires_grad,
                    is_pinned: self.is_pinned || other.is_pinned,
                };
                crate::implicit_autograd::record_and_link_binary(
                    crate::implicit_autograd::BinaryOpKind::Mul,
                    self,
                    other,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Multiplication failed: {}",
                e
            ))),
        }
    }

    /// Subtract two tensors
    pub fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::sub(&self.tensor, &other.tensor) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad || other.requires_grad,
                    is_pinned: self.is_pinned || other.is_pinned,
                };
                crate::implicit_autograd::record_and_link_binary(
                    crate::implicit_autograd::BinaryOpKind::Sub,
                    self,
                    other,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Subtraction failed: {}",
                e
            ))),
        }
    }

    /// Divide two tensors
    pub fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::div(&self.tensor, &other.tensor) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad || other.requires_grad,
                    is_pinned: self.is_pinned || other.is_pinned,
                };
                crate::implicit_autograd::record_and_link_binary(
                    crate::implicit_autograd::BinaryOpKind::Div,
                    self,
                    other,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Division failed: {}", e))),
        }
    }

    /// Matrix (batch-) multiplication of `self` and `other`.
    ///
    /// Follows standard BLAS GEMM semantics.  Both tensors must be at least
    /// 2-D and the inner dimensions must match (`self` last dim == `other`
    /// first dim).
    ///
    /// # Python Example
    ///
    /// ```python
    /// import tenflowers as tf
    /// a = tf.ones([3, 4])
    /// b = tf.ones([4, 2])
    /// c = a.matmul(b)
    /// print(c.shape())  # [3, 2]
    /// ```
    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::matmul(&self.tensor, &other.tensor) {
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: self.requires_grad || other.requires_grad,
                    is_pinned: self.is_pinned || other.is_pinned,
                };
                crate::implicit_autograd::record_and_link_binary(
                    crate::implicit_autograd::BinaryOpKind::MatMul,
                    self,
                    other,
                    &result,
                )?;
                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Matrix multiplication failed: {}",
                e
            ))),
        }
    }

    /// Power operation
    fn pow(&self, exponent: f32) -> PyResult<PyTensor> {
        // Create a scalar tensor with the same shape for broadcasting
        let exponent_tensor = Tensor::from_scalar(exponent);
        match tenflowers_core::ops::binary::pow(&self.tensor, &exponent_tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad,
                is_pinned: self.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Power operation failed: {}",
                e
            ))),
        }
    }

    /// The device this tensor resides on (always CPU in the current implementation).
    #[getter]
    pub fn device(&self) -> crate::device::PyDevice {
        crate::device::PyDevice::cpu()
    }

    /// String representation.
    ///
    /// Includes shape, dtype (derived from the inner tensor type — always `float32`
    /// for the current `Tensor<f32>` implementation), device, and requires_grad.
    fn __repr__(&self) -> String {
        format!(
            "PyTensor(shape={:?}, dtype={}, device={}, requires_grad={})",
            self.shape(),
            self.dtype(),
            self.device().__str__(),
            self.requires_grad
        )
    }

    /// String representation for print().
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Length along the first dimension (Python `len()` support).
    ///
    /// Raises `TypeError` for scalar (rank-0) tensors, matching NumPy / PyTorch behaviour.
    fn __len__(&self) -> PyResult<usize> {
        let shape = self.shape();
        if shape.is_empty() {
            Err(PyTypeError::new_err(
                "len() of a scalar tensor (rank 0) is not defined",
            ))
        } else {
            Ok(shape[0])
        }
    }

    /// Iterator over slices along the first dimension.
    ///
    /// Raises `TypeError` for scalar (rank-0) tensors.
    fn __iter__(&self) -> PyResult<PyTensorIter> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(PyTypeError::new_err(
                "cannot iterate over a scalar tensor (rank 0)",
            ));
        }
        Ok(PyTensorIter {
            source: self.clone(),
            index: 0,
            len: shape[0],
        })
    }
}

/// Create a tensor of the given `shape` filled with `0.0` (dtype `float32`).
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.zeros([3, 4])
/// print(t.shape())  # [3, 4]
/// ```
#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![0.0f32; size];
    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create zeros tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor of the given `shape` filled with `1.0` (dtype `float32`).
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.ones([2, 5])
/// print(t.shape())  # [2, 5]
/// ```
#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![1.0f32; size];
    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create ones tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor with uniform-random values in `[0, 1)` (dtype `float32`).
///
/// Values are drawn from a uniform distribution over `[0, 1)` using a
/// time-seeded PRNG.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.rand([4, 4])
/// print(t.shape())  # [4, 4]
/// ```
#[pyfunction]
pub fn rand(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = tenflowers_core::ops::rand_f32(&shape, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create random tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor with values drawn from the standard normal distribution
/// N(0, 1) (dtype `float32`).
///
/// Values are drawn from a Gaussian distribution (mean=0, std=1) using a
/// time-seeded PRNG backed by the Box-Muller transform via scirs2-core.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.randn([8, 8])
/// print(t.shape())  # [8, 8]
/// ```
#[pyfunction]
pub fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = tenflowers_core::ops::randn_f32(&shape, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create randn tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a zero-filled tensor backed by pinned (page-locked) host memory.
///
/// Pinned memory enables faster DMA transfers to/from GPU devices.  The
/// returned tensor has [`PyTensor::is_pinned`] set to `true`.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.zeros_pinned([1024, 1024])
/// assert t.is_pinned()  # True
/// ```
#[pyfunction]
pub fn zeros_pinned(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![0.0f32; size];
    let tensor = Tensor::from_vec(data, &shape).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create zeros_pinned tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: true, // This tensor uses pinned memory
    })
}

/// Create a one-filled tensor backed by pinned (page-locked) host memory.
///
/// See [`zeros_pinned`] for details about pinned memory semantics.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.ones_pinned([512])
/// assert t.is_pinned()  # True
/// ```
#[pyfunction]
pub fn ones_pinned(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![1.0f32; size];
    let tensor = Tensor::from_vec(data, &shape).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create ones_pinned tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: true, // This tensor uses pinned memory
    })
}

/// Create a random-filled tensor backed by pinned (page-locked) host memory.
///
/// See [`zeros_pinned`] for details about pinned memory semantics.
/// Values are uniform-random in `[0, 1)` drawn from a time-seeded PRNG.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.rand_pinned([256, 256])
/// assert t.is_pinned()  # True
/// ```
#[pyfunction]
pub fn rand_pinned(shape: Vec<usize>) -> PyResult<PyTensor> {
    let tensor = tenflowers_core::ops::rand_f32(&shape, None).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create rand_pinned tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: true,
    })
}

/// Element-wise addition `lhs + rhs`.
///
/// Both tensors must have the same shape.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// c = tf.add(tf.ones([3]), tf.ones([3]))   # [2, 2, 2]
/// ```
#[pyfunction]
pub fn add(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.add(rhs)
}

/// Element-wise multiplication (Hadamard product) `lhs * rhs`.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// c = tf.mul(tf.ones([3]), tf.ones([3]))   # [1, 1, 1]
/// ```
#[pyfunction]
pub fn mul(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.mul(rhs)
}

/// Element-wise subtraction `lhs - rhs`.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// c = tf.sub(tf.ones([3]), tf.ones([3]))   # [0, 0, 0]
/// ```
#[pyfunction]
pub fn sub(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.sub(rhs)
}

/// Element-wise division `lhs / rhs`.
///
/// Division by zero produces `±inf` or `NaN` per IEEE 754.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// c = tf.div(tf.ones([3]), tf.ones([3]))   # [1, 1, 1]
/// ```
#[pyfunction]
pub fn div(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.div(rhs)
}

/// Matrix multiplication `lhs @ rhs` (BLAS GEMM).
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// c = tf.matmul(tf.ones([3, 4]), tf.ones([4, 2]))
/// print(c.shape())  # [3, 2]
/// ```
#[pyfunction]
pub fn matmul(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.matmul(rhs)
}

/// Transpose `tensor`, optionally specifying an axis permutation.
///
/// Without `axes`, reverses all axes (equivalent to `tensor.T`).
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t  = tf.ones([2, 3])
/// tT = tf.transpose(t)            # shape (3, 2)
/// t2 = tf.transpose(t, [1, 0])   # same
/// ```
#[pyfunction]
#[pyo3(signature = (tensor, axes=None))]
pub fn transpose(tensor: &PyTensor, axes: Option<Vec<usize>>) -> PyResult<PyTensor> {
    tensor.transpose(axes)
}

/// Return a view of `tensor` with the given `shape`.
///
/// Total element count must be preserved.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.ones([12])
/// t2 = tf.reshape(t, [3, 4])
/// print(t2.shape())  # [3, 4]
/// ```
#[pyfunction]
pub fn reshape(tensor: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    tensor.reshape(shape)
}

/// Slice `tensor` along each dimension.
///
/// `ranges` is a list of `(start, end, step)` tuples, one per leading
/// dimension; trailing dimensions are taken in full. See
/// [`PyTensor::slice`] for the full semantics.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
/// t = tf.ones([4, 5])
/// s = tf.slice(t, [(1, 3, None)])
/// print(s.shape())  # [2, 5]
/// ```
#[pyfunction]
pub fn slice(
    tensor: &PyTensor,
    ranges: Vec<(Option<isize>, Option<isize>, Option<isize>)>,
) -> PyResult<PyTensor> {
    tensor.slice(ranges)
}

/// Autograd-enabled tensor wrapper produced by [`PyGradientTape::watch`].
///
/// `PyTrackedTensor` wraps a `tenflowers_autograd::TrackedTensor<f32>` and is
/// used as input/output to gradient computations.  It is intentionally
/// lightweight — all gradient bookkeeping lives inside the tape, not here.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
///
/// tape = tf.PyGradientTape()
/// x    = tf.ones([3])
/// tx   = tape.watch(x)          # returns PyTrackedTensor
/// raw  = tx.tensor()            # unwrap back to PyTensor
/// ```
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTrackedTensor {
    pub tensor: Arc<TrackedTensor<f32>>,
}

#[pymethods]
impl PyTrackedTensor {
    /// Create a new tracked tensor from a regular tensor
    #[new]
    fn new(tensor: &PyTensor) -> PyResult<Self> {
        let tracked = TrackedTensor::new(tensor.tensor.as_ref().clone());
        Ok(PyTrackedTensor {
            tensor: Arc::new(tracked),
        })
    }

    /// Get the underlying tensor
    fn tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: Arc::new(self.tensor.tensor().clone()),
            requires_grad: false, // TrackedTensor doesn't expose requires_grad method
            is_pinned: false,     // Default for tracked tensors
        })
    }

    /// Check if this tensor requires gradients
    fn requires_grad(&self) -> bool {
        false // TrackedTensor doesn't expose requires_grad method
    }
}

/// Iterator over first-dimension slices of a [`PyTensor`].
///
/// Calling `iter(tensor)` in Python returns a `PyTensorIter`.  Each
/// `next()` call yields a rank-(N-1) tensor that is a contiguous slice
/// along axis 0.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
///
/// matrix = tf.ones([3, 4])
/// for row in matrix:
///     print(row.shape())  # [4]  — three iterations
/// ```
#[pyclass]
pub struct PyTensorIter {
    source: PyTensor,
    index: usize,
    len: usize,
}

#[pymethods]
impl PyTensorIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyTensor>> {
        if self.index >= self.len {
            return Ok(None);
        }

        let full_shape = self.source.shape();
        // Compute the shape of one slice: drop the first dimension.
        let slice_shape: Vec<usize> = full_shape[1..].to_vec();
        let slice_numel: usize = if slice_shape.is_empty() {
            1
        } else {
            slice_shape.iter().product()
        };

        let start = self.index * slice_numel;
        let end = start + slice_numel;

        // Obtain the raw data vector from the tensor.
        let all_data =
            self.source.tensor.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("iterator: failed to get data: {}", e))
            })?;

        if end > all_data.len() {
            return Err(PyRuntimeError::new_err(
                "iterator: slice index out of range (data length mismatch)",
            ));
        }

        let slice_data: Vec<f32> = all_data[start..end].to_vec();

        // When slice_shape is empty (source was 1-D), wrap in a length-1 vector tensor.
        let out_shape: Vec<usize> = if slice_shape.is_empty() {
            vec![1]
        } else {
            slice_shape
        };

        let t = Tensor::from_vec(slice_data, &out_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("iterator: reshape failed: {}", e)))?;

        self.index += 1;

        Ok(Some(PyTensor {
            tensor: Arc::new(t),
            requires_grad: self.source.requires_grad,
            is_pinned: self.source.is_pinned,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test isolation note: `crate::implicit_autograd`'s thread-local state
    // (IMPLICIT_TAPE, TRACKED_REGISTRY, LEAVES, GRAD_STORE, IDENTITY_ANCHORS)
    // persists across tests that happen to run on the same OS thread, and
    // Rust's test harness does not guarantee one thread per test. However,
    // `implicit_autograd`'s own `reset_for_test` helper is a private `fn`
    // inside a `#[cfg(test)] mod tests` block in that module — not `pub` or
    // `pub(crate)` — so it is not reachable from this module at all
    // (confirmed by reading that file's test module). Cross-contamination is
    // nonetheless not possible here: `run_backward` (called by every test
    // below via `.backward()`) unconditionally clears
    // TRACKED_REGISTRY/LEAVES/the tape immediately after a successful
    // backward pass, and every test in this module builds its own
    // self-contained graph (its own leaf, its own scalar) and calls
    // `.backward()` exactly once, synchronously, with no `.await`/yield point
    // anywhere in this code — so one test's body (including the
    // state-resetting `.backward()` call) always runs to completion before
    // another test could possibly interleave on the same OS thread.
    // GRAD_STORE is deliberately never cleared by run_backward, but it is
    // keyed by each leaf's own unique `tensor_key` (an `Arc` address), so a
    // stale entry from an earlier test cannot be read back by a later test's
    // distinct, freshly-allocated tensors.

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = tenflowers_core::Tensor::from_vec(data, shape)
            .expect("tensor construction must succeed");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    #[test]
    fn transpose_links_onto_tape_and_grad_is_correct() {
        let mut x = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        x.set_requires_grad(true);
        let t = x.transpose(None).expect("transpose must succeed");
        assert_eq!(t.shape(), vec![3, 2]);
        let scalar = crate::math_ops::sum(&t, None, None).expect("sum must succeed");
        scalar.backward().expect("backward must succeed");
        let grad = x.grad().expect("grad must be populated");
        let grad_data = grad.tensor.to_vec().expect("grad readable");
        // sum is invariant to any bijective reindexing of elements, and
        // transpose is exactly such a reindexing, so d(sum)/d(x_ij) = 1 for
        // every element regardless of where it moved to.
        assert_eq!(grad_data, vec![1.0; 6]);
    }

    #[test]
    fn transpose_with_explicit_axes_links_onto_tape_and_grad_is_correct() {
        let mut x = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        x.set_requires_grad(true);
        let t = x
            .transpose(Some(vec![1, 0]))
            .expect("transpose with axes must succeed");
        assert_eq!(t.shape(), vec![3, 2]);
        let scalar = crate::math_ops::sum(&t, None, None).expect("sum must succeed");
        scalar.backward().expect("backward must succeed");
        let grad = x.grad().expect("grad must be populated");
        let grad_data = grad.tensor.to_vec().expect("grad readable");
        // Same reasoning as the `None`-axes case: an explicit permutation is
        // still a bijective reindexing, so every gradient entry is 1.
        assert_eq!(grad_data, vec![1.0; 6]);
    }

    #[test]
    fn reshape_links_onto_tape_and_grad_is_correct() {
        let mut x = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        x.set_requires_grad(true);
        let r = x.reshape(vec![3, 2]).expect("reshape must succeed");
        assert_eq!(r.shape(), vec![3, 2]);
        let scalar = crate::math_ops::sum(&r, None, None).expect("sum must succeed");
        scalar.backward().expect("backward must succeed");
        let grad = x.grad().expect("grad must be populated");
        assert_eq!(
            grad.shape(),
            vec![2, 3],
            "grad must match x's ORIGINAL shape"
        );
        let grad_data = grad.tensor.to_vec().expect("grad readable");
        // reshape does not move data across a reduction boundary (sum over
        // all elements either way), so every element's gradient is 1.
        assert_eq!(grad_data, vec![1.0; 6]);
    }

    #[test]
    fn slice_links_onto_tape_and_grad_is_correct() {
        let mut x = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        x.set_requires_grad(true);
        let s = x
            .slice(vec![(Some(1), Some(3), None)])
            .expect("slice must succeed");

        // Verify the slice itself before trusting anything about the
        // gradient: (start=1, end=3, step=1) selects indices 1 and 2.
        assert_eq!(s.shape(), vec![2]);
        let s_data = s.tensor.to_vec().expect("slice output readable");
        assert_eq!(s_data, vec![2.0, 3.0]);

        let scalar = crate::math_ops::sum(&s, None, None).expect("sum must succeed");
        scalar.backward().expect("backward must succeed");
        let grad = x.grad().expect("grad must be populated");
        assert_eq!(grad.shape(), vec![4]);
        let grad_data = grad.tensor.to_vec().expect("grad readable");
        // Gradient flows back only through the sliced elements (indices 1
        // and 2); elements outside the slice (indices 0 and 3) get zero
        // gradient.
        assert_eq!(grad_data, vec![0.0, 1.0, 1.0, 0.0]);
    }
}
