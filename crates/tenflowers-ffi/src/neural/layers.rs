//! Neural network layer implementations
//!
//! This module provides Python bindings for various neural network layers
//! including dense layers, parameter management, and sequential models.

use super::hooks::{HookManager, PyHookHandle};
use crate::tensor_ops::PyTensor;
use pyo3::prelude::*;
// use std::collections::HashMap; // Unused for now
use std::sync::Arc;
use tenflowers_core::Tensor;
use tenflowers_neural::layers::{Dense, Layer};

/// A trainable parameter tensor with a stable identity across value updates.
///
/// `PyParameter` is the leaf node of the autograd graph — it represents a
/// single learnable weight or bias.  When `requires_grad` is `True` (the
/// default), gradient information is accumulated during the backward pass.
///
/// # Stable identity, mutable value
///
/// Unlike [`PyTensor`] (whose identity *is* its `Arc<Tensor<f32>>` address,
/// and which [`crate::implicit_autograd`] therefore treats as a brand-new
/// tensor every time an operation produces a new `Arc`), a `PyParameter`'s
/// value is designed to be mutated *in place* by an optimizer step (e.g.
/// `w -= lr * grad`) while its identity — used as the autograd tape's lookup
/// key — stays exactly the same across that mutation. This is what a real
/// training loop needs: the same parameter object must remain the same tape
/// leaf from one `forward()`/`backward()` step to the next, even though its
/// numeric contents changed in between.
///
/// This is implemented by separating *identity* from *value*:
///
/// * `data: Arc<RwLock<Tensor<f32>>>` — the value cell. Its **contents** are
///   freely replaceable via [`PyParameter::set_data`], but the `RwLock`
///   allocation itself never moves or is replaced for the parameter's whole
///   lifetime.
/// * `id: usize` — captured *once*, at construction, as the address of the
///   `RwLock` allocation (`Arc::as_ptr(&data) as usize`) — **not** the
///   address of any `Tensor` inside it. Because that address never changes
///   even as `set_data` replaces what is stored inside the lock, `id` is
///   exactly the kind of address-stable, mutation-surviving identity that
///   [`crate::implicit_autograd::mark_leaf_param`] /
///   [`crate::implicit_autograd::get_grad_by_id`] need.
///
/// A plain `Arc<Tensor<f32>>` (what `PyTensor` uses) cannot support this:
/// `Arc::make_mut` would silently deep-copy whenever another clone of the
/// same `Arc` is alive (breaking identity), and replacing the `Arc` itself
/// to get a new value changes the very address used as the tape key.
///
/// # Why `RwLock` and not `RefCell`/`Mutex`
///
/// `#[pyclass]` types must be `Send + Sync` to cross the Python/Rust
/// boundary, which rules out `RefCell`. `RwLock` (rather than `Mutex`) is
/// used purely for API-shape consistency with the rest of this crate (see
/// e.g. [`super::hooks::HookManager`]) — in practice PyO3 holds the GIL for
/// every Python-visible call, so there is no real lock contention here, just
/// a `Send + Sync`-compatible interior-mutability cell. Every lock scope in
/// this `impl` is kept minimal (a single read/write plus an immediate clone)
/// and is always released before returning to any caller, so a lock is never
/// held across a call that could re-enter Python.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
///
/// w = tf.PyParameter(tf.ones([128, 64]))         # requires_grad=True by default
/// b = tf.PyParameter(tf.zeros([64]), requires_grad=True)
///
/// print(w.shape())          # [128, 64]
/// print(w.size())           # 8192
/// print(w.requires_grad())  # True
/// print(w.dtype())          # "f32"
/// print(w.device())         # "cpu"
///
/// w.zero_grad()   # clears any accumulated gradient for this parameter's id
/// ```
#[pyclass]
#[derive(Debug)]
pub struct PyParameter {
    /// Stable data cell: contents are replaceable via `set_data`, but the
    /// `RwLock` allocation itself never moves for the life of the parameter.
    data: Arc<std::sync::RwLock<Tensor<f32>>>,
    /// Identity captured ONCE at construction — the address of the `RwLock`
    /// allocation itself. Never changes even as `*data.write()`'s contents
    /// are replaced, which is exactly what makes this suitable as a stable
    /// autograd tape key across in-place parameter updates (unlike
    /// `PyTensor`'s `Arc<Tensor<f32>>` address, which changes on every new
    /// tensor computation).
    id: usize,
    requires_grad: bool,
}

#[pymethods]
impl PyParameter {
    /// Construct a new parameter from an existing tensor.
    ///
    /// The incoming `tensor`'s contents are cloned into a **fresh**,
    /// independent `RwLock` allocation — the incoming `PyTensor`'s own
    /// `Arc<Tensor<f32>>` is a different allocation by construction and is
    /// never reused as the parameter's `data` cell, since a `PyParameter`
    /// must own an allocation whose address it exclusively controls for the
    /// life of the parameter (see the struct-level "Stable identity" doc).
    ///
    /// # Arguments
    ///
    /// * `tensor`       — initial value (typically from `tf.zeros` / `tf.ones` / `tf.rand`).
    /// * `requires_grad` — whether to accumulate gradients (default: `True`).
    #[new]
    #[pyo3(signature = (tensor, requires_grad=None))]
    pub fn new(tensor: PyTensor, requires_grad: Option<bool>) -> Self {
        let data = Arc::new(std::sync::RwLock::new((*tensor.tensor).clone()));
        let id = Arc::as_ptr(&data) as usize;
        Self {
            data,
            id,
            requires_grad: requires_grad.unwrap_or(true),
        }
    }

    /// Get tensor shape.
    ///
    /// Read-only metadata: a poisoned lock (meaning some other thread
    /// panicked while holding the write lock, e.g. inside `set_data`) does
    /// not invalidate the *data* itself, so this recovers via
    /// `PoisonError::into_inner` rather than propagating a hard error — the
    /// standard idiom for "keep reading known-still-valid data past a
    /// poison" (this is not `.unwrap()`; it never panics).
    pub fn shape(&self) -> Vec<usize> {
        self.data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .shape()
            .dims()
            .to_vec()
    }

    /// Get tensor size. See [`PyParameter::shape`] for the poison-recovery rationale.
    pub fn size(&self) -> usize {
        self.data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .size()
    }

    /// Check if gradients are required
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient requirement
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Create an independent parameter with a value-copy of the current data.
    ///
    /// This is **not** a shallow/aliasing clone: the returned `PyParameter`
    /// gets its own fresh `RwLock` allocation and therefore its own fresh
    /// `id`, distinct from `self`'s. An aliasing clone (two `PyParameter`
    /// Python objects secretly sharing one `data` cell and `id`) would be a
    /// footgun — writes to one would silently appear on the other, and both
    /// would resolve to the same autograd tape leaf — so this always
    /// constructs a genuinely new, independent parameter, matching the
    /// current value and `requires_grad` flag but nothing else.
    pub fn clone_param(&self) -> Self {
        let current = self
            .data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let data = Arc::new(std::sync::RwLock::new(current));
        let id = Arc::as_ptr(&data) as usize;
        Self {
            data,
            id,
            requires_grad: self.requires_grad,
        }
    }

    /// Get data type information
    pub fn dtype(&self) -> String {
        "f32".to_string()
    }

    /// Get device information. See [`PyParameter::shape`] for the
    /// poison-recovery rationale.
    #[allow(unexpected_cfgs)]
    pub fn device(&self) -> String {
        match self
            .data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .device()
        {
            tenflowers_core::Device::Cpu => "cpu".to_string(),
            #[cfg(feature = "gpu")]
            tenflowers_core::Device::Gpu(id) => format!("gpu:{}", id),
            #[cfg(feature = "gpu")]
            tenflowers_core::Device::Rocm(id) => format!("rocm:{}", id),
        }
    }

    /// Retrieve the gradient computed for this parameter by a previous
    /// `.backward()` call that passed through a `forward()` marking this
    /// parameter as a leaf (see
    /// [`crate::implicit_autograd::mark_leaf_param`]).
    ///
    /// Looks the gradient up by this parameter's stable `id` — **not** by
    /// re-deriving a key from a transient `PyTensor` via
    /// [`crate::implicit_autograd::tensor_key`], which would be the address
    /// of some unrelated, freshly allocated `Arc<Tensor<f32>>` and would
    /// never match anything the tape recorded for this parameter.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if no gradient has been computed for this
    /// parameter's `id` yet.
    pub fn grad(&self) -> PyResult<PyTensor> {
        crate::implicit_autograd::get_grad_by_id(self.id)
    }

    /// Clear this parameter's accumulated gradient, if any.
    ///
    /// # Errors
    ///
    /// This method itself cannot currently fail (gradient-store removal is
    /// infallible), but returns `PyResult` to match the rest of this
    /// mutating/gradient-facing API and to leave room for future validation
    /// without a breaking signature change.
    pub fn zero_grad(&self) -> PyResult<()> {
        crate::implicit_autograd::clear_grad_by_id(self.id);
        Ok(())
    }

    /// Snapshot this parameter's current value into a fresh, independent
    /// [`PyTensor`].
    ///
    /// This is the snapshot a layer's `forward()` takes at the start of
    /// every forward pass: read-lock `data`, clone its contents, and wrap
    /// the clone in a **brand new** `Arc<Tensor<f32>>` allocation (never
    /// reusing anything from `data`'s own `Arc<RwLock<..>>` allocation — the
    /// two are deliberately different allocations with different addresses,
    /// which is exactly why the parameter's `id` stays stable regardless of
    /// how many transient snapshots are taken from it over time). The
    /// resulting `PyTensor` is then handed to
    /// [`crate::implicit_autograd::mark_leaf_param`] together with this
    /// parameter's `id` to (re-)register it as a tape leaf for the new
    /// forward pass.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if the internal lock is poisoned. Never panics.
    pub fn to_tensor(&self) -> PyResult<PyTensor> {
        let guard = self.data.read().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("parameter data lock poisoned")
        })?;
        Ok(PyTensor {
            tensor: Arc::new(guard.clone()),
            requires_grad: self.requires_grad,
            is_pinned: false,
        })
    }

    /// This parameter's stable identity, usable as an
    /// [`crate::implicit_autograd`] tape key. Exposed to Python in case a
    /// future Python-side optimizer wants to reference it directly (e.g. for
    /// diagnostics or a custom parameter-group implementation); the
    /// authoritative Rust-side accessor is the inherent `PyParameter::id`
    /// method below, used by `mark_leaf_param`/`get_grad_by_id` call sites
    /// within this crate.
    #[pyo3(name = "id")]
    pub fn id_py(&self) -> usize {
        self.id
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyParameter(shape={:?}, requires_grad={})",
            self.shape(),
            self.requires_grad
        )
    }
}

impl PyParameter {
    /// This parameter's stable identity (see the struct-level "Stable
    /// identity, mutable value" doc). Plain inherent accessor for Rust-side
    /// callers (tests, and a future Rust-side optimizer implementation) that
    /// don't need to go through the `#[pymethods]` Python-facing `id()`
    /// getter above.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Replace this parameter's value in place, preserving its `id`.
    ///
    /// Takes a raw `Tensor<f32>` (not a `PyTensor`) since the intended
    /// caller — a future Rust-side optimizer's `.step()` implementation —
    /// computes a raw tensor result (e.g. `w - lr * grad`) and hands it in
    /// directly; a caller holding a `PyTensor` instead can get to a
    /// `Tensor<f32>` via `(*py_tensor.tensor).clone()`.
    ///
    /// Deliberately **not** a `#[pymethods]` method: `pyo3` requires
    /// Python-visible method arguments to implement `FromPyObject`, which a
    /// bare `Tensor<f32>` (not itself a `#[pyclass]`) does not — only the
    /// `#[pyclass]`-wrapped [`PyTensor`] does. Since this method's whole
    /// purpose is accepting a raw `Tensor<f32>` a Rust-side caller already
    /// has in hand (rather than one freshly unwrapped from a `PyTensor`
    /// argument), it is exposed as a plain Rust inherent method here,
    /// alongside [`PyParameter::id`], instead of forcing a
    /// `PyTensor`-accepting `#[pymethods]` overload that the task this
    /// method exists for (an in-place optimizer update) does not need.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if the internal lock is poisoned. Never panics.
    pub fn set_data(&self, new_tensor: Tensor<f32>) -> PyResult<()> {
        let mut guard = self.data.write().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("parameter data lock poisoned")
        })?;
        *guard = new_tensor;
        Ok(())
    }
}

impl Drop for PyParameter {
    /// Proactively clear every trace of this parameter's `id` from
    /// [`crate::implicit_autograd`]'s thread-local tables — not just
    /// [`crate::implicit_autograd::GRAD_STORE`], but also
    /// [`crate::implicit_autograd::TRACKED_REGISTRY`] and
    /// [`crate::implicit_autograd::LEAVES`] — when the parameter itself is
    /// deallocated. See [`crate::implicit_autograd::unregister_param_id`]
    /// for the full reasoning on why all three (not just `GRAD_STORE`)
    /// must be scrubbed here; this doc focuses on why `Drop` is the right
    /// place to do it at all.
    ///
    /// # Why this is necessary despite `id` needing no `IDENTITY_ANCHORS` entry
    ///
    /// A parameter's `id` is the address of its own `Arc<RwLock<Tensor<f32>>>`
    /// allocation, and [`crate::implicit_autograd`] deliberately does *not*
    /// anchor that address (see `register_tracked_by_key`'s doc) — the
    /// reasoning being that the `PyParameter` object's own lifetime already
    /// keeps the allocation alive for as long as anything could need its
    /// `id` to stay meaningful. That reasoning has an edge this `Drop` impl
    /// exists to close: it only holds *while the object is alive*. Once a
    /// `PyParameter` is fully dropped, its `RwLock` allocation genuinely is
    /// freed, and the allocator is free to reuse that exact address for a
    /// later, completely unrelated `PyParameter`'s own
    /// `Arc::new(RwLock::new(..))` allocation — this is ordinary, expected
    /// behaviour (parameters are dropped all the time: models get freed,
    /// parameters get replaced or pruned, etc.), not a contrived edge case.
    ///
    /// [`crate::implicit_autograd::GRAD_STORE`] is deliberately *never*
    /// cleared by `run_backward` (so `.grad()` keeps working after the graph
    /// is freed, mirroring a real PyTorch leaf's `.grad` surviving after its
    /// graph is freed) — which means a dropped parameter's gradient entry
    /// would otherwise survive, keyed by an address now available for reuse,
    /// and a later, unrelated parameter that happens to land at that same
    /// freed address would spuriously "inherit" a stale gradient that was
    /// never actually computed for it. This was caught during development by
    /// `independent_parameters_do_not_cross_contaminate_even_on_address_reuse`'s
    /// final assertion, exactly mirroring how
    /// `address_reuse_does_not_leak_stale_gradient` caught the analogous
    /// `tensor_key` hazard that motivated `IDENTITY_ANCHORS` in the first
    /// place — the two mechanisms solve the same underlying class of "stale
    /// numeric key" hazard via two different, equally valid strategies: keep
    /// the address from ever being freed (`IDENTITY_ANCHORS`, for
    /// `tensor_key`) vs. proactively scrub the stale table entry the moment
    /// the address genuinely is about to be freed (this `Drop` impl, for
    /// parameter `id`s).
    ///
    /// `TRACKED_REGISTRY`/`LEAVES` have the exact same address-reuse
    /// exposure as `GRAD_STORE` does, via a different symptom: a parameter
    /// that was [`crate::implicit_autograd::mark_leaf_param`]'d but never
    /// backward'd through before being dropped leaves its `id` present in
    /// `TRACKED_REGISTRY`/`LEAVES` (only [`crate::implicit_autograd::run_backward`]
    /// clears those, and it may never run for an abandoned forward pass).
    /// A later, address-colliding parameter's own `mark_leaf_param` call
    /// then finds that stale entry, mistakes it for "this parameter was
    /// already marked earlier this cycle", and silently never watches the
    /// new parameter on the tape at all — confirmed reproducible via
    /// `crate::implicit_autograd::tests::dropped_untracked_backward_param_does_not_poison_tracked_registry_on_address_reuse`.
    ///
    /// This is sound specifically because a `PyParameter`'s `id` is
    /// *exclusively* owned: [`PyParameter`] does not implement `Clone` (see
    /// the struct-level doc — [`PyParameter::clone_param`] always allocates
    /// a fresh, independent `RwLock` cell with its own fresh `id` rather
    /// than aliasing), so exactly one `PyParameter` value ever owns a given
    /// `id` at a time, and this `Drop` firing can never race with or
    /// prematurely invalidate a *different*, still-alive `PyParameter` that
    /// happens to share the same `id` — no such aliasing is possible.
    fn drop(&mut self) {
        crate::implicit_autograd::unregister_param_id(self.id);
    }
}

/// A fully-connected (linear + optional activation) layer.
///
/// `PyDense` wraps a `tenflowers_neural::layers::Dense<f32>` and exposes it
/// to Python with PyTorch-style hooks, training/eval mode, and parameter
/// introspection.
///
/// Weights are initialised with Xavier (Glorot) uniform initialisation.
///
/// # Autograd: identity lives in `PyParameter::id()`, not in an `Arc` address
///
/// The weight and (optional) bias are each held as a [`PyParameter`] —
/// specifically as a `Py<PyParameter>` (a Python-heap-allocated handle,
/// constructed once in [`PyDense::new`]) rather than a bare `PyParameter`
/// value, precisely so that [`PyDense::parameters`] can hand out additional
/// *owning references to that exact same Python object* (via
/// [`Py::clone_ref`]) instead of a value-copy. This is the crucial property
/// an optimizer needs: calling `.set_data(...)` on a `PyParameter` handle
/// returned by `.parameters()` must be visible to this layer's *own* next
/// `.forward()` call, which requires both call sites to ultimately be
/// looking at the same `Arc<RwLock<Tensor<f32>>>` cell (see
/// [`PyParameter`]'s struct-level "Stable identity, mutable value" doc for
/// why that cell's address — its `id()` — is the thing that must stay
/// constant, not the tensor value inside it).
///
/// Note this is a different, and stronger, guarantee than a plain
/// `Arc<Tensor<f32>>` (what the pre-`PyParameter` design, and what
/// [`crate::tensor_ops::PyTensor`] still, uses) can offer: an `Arc` clone
/// shares *value* identity only until something replaces the `Arc` itself to
/// install a new value (e.g. an optimizer step), at which point the "shared"
/// identity silently diverges — exactly the bug [`PyParameter`] exists to
/// close, by separating a stable `id` from a freely-replaceable value cell.
///
/// # Why `forward()` cannot just wrap the parameter's cell directly
///
/// [`crate::implicit_autograd`] keys its tape side-tables by a [`PyTensor`]'s
/// own `Arc<Tensor<f32>>` pointer identity (see
/// `implicit_autograd::tensor_key`) — but a `PyParameter`'s value cell is an
/// `Arc<RwLock<Tensor<f32>>>`, a different type entirely, and is never
/// directly exposed as a `PyTensor`. Every `forward()` call must therefore
/// take a **snapshot**: [`PyParameter::to_tensor`] read-locks the cell,
/// clones its current contents into a *brand-new* `Arc<Tensor<f32>>`
/// allocation, and wraps that in a fresh [`PyTensor`]. Because
/// [`PyParameter::to_tensor`] allocates a new `Arc` on every single call,
/// plain [`crate::implicit_autograd::mark_leaf`] cannot be used on the
/// result — its own idempotency check is keyed by that snapshot's transient
/// `Arc` address, which is different on every forward pass, so it would
/// never recognize two snapshots of the same parameter as "the same leaf"
/// and would silently re-register a fresh tape leaf every step instead of
/// reusing the one whose gradient an optimizer is about to read.
/// [`crate::implicit_autograd::mark_leaf_param`] is the fix: it is keyed
/// directly by the `PyParameter`'s own stable `id()` (passed in explicitly by
/// the caller) rather than by the transient snapshot's incidental address,
/// so repeated forward passes over the *same* parameter correctly collapse
/// onto the *same* tape leaf, and a computed gradient becomes reachable via
/// [`PyParameter::grad`] (itself keyed by `id()`) regardless of how many
/// snapshots were taken along the way. See that function's own doc for the
/// full reasoning, including why the resulting `TrackedTensor` is registered
/// under two different keys at once.
///
/// `forward()` computes the layer's output purely through tape-aware
/// `PyTensor` operations (`PyTensor::matmul`/`PyTensor::add` and the
/// tape-aware activation functions in [`super::functions`]) over these
/// snapshots, instead of delegating to
/// `tenflowers_neural::layers::Dense::forward`, whose raw `Tensor<f32>`
/// arithmetic has no relationship to the implicit tape at all.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
///
/// # Create a 128→64 ReLU layer
/// layer = tf.PyDense(128, 64, use_bias=True, activation='relu')
/// x   = tf.ones([4, 128])
/// out = layer.forward(x)
/// print(out.shape())           # [4, 64]
/// print(layer.input_dim())     # 128
/// print(layer.output_dim())    # 64
/// print(layer.has_bias())      # True
/// print(layer.activation())    # "relu"
/// print(layer.num_parameters()) # 128*64 + 64
///
/// # Training / eval mode
/// layer.set_training(True)
/// layer.set_training(False)
///
/// # Hook registration
/// h = layer.register_forward_hook(lambda inp, out: None)
/// h.remove()
///
/// # End-to-end gradient flow: parameters() returns Parameter handles that
/// # share identity (via PyParameter::id()) with whatever forward() actually
/// # snapshotted and computed with, so .grad() is populated on them after a
/// # .backward() call — and an optimizer's .set_data() on one of these
/// # handles is visible to this SAME layer's next forward() call.
/// out = layer.forward(x)
/// loss = tf.sum(out)
/// loss.backward()
/// for p in layer.parameters():
///     print(p.grad().shape())
///     # p.set_data(p.to_tensor() - lr * p.grad())  # optimizer-style update
/// ```
#[pyclass]
#[derive(Debug)]
pub struct PyDense {
    layer: Dense<f32>,
    /// Stable parameter identity for the weight, shared between every
    /// `forward()` snapshot and every `parameters()` call: `parameters()`
    /// hands out additional owning references to this *exact* Python object
    /// (via [`Py::clone_ref`]), not a value-copy, so writes through any of
    /// those references (e.g. an optimizer's `.set_data()`) are visible the
    /// next time `forward()` reads it. See the "Autograd" section above.
    weight_param: Py<PyParameter>,
    /// Stable parameter identity for the bias (`None` when `use_bias=False`).
    bias_param: Option<Py<PyParameter>>,
    hook_manager: HookManager,
    training: bool,
}

impl Clone for PyDense {
    /// Produce an **independent** layer: a fresh copy of the current weight
    /// and bias values, under **fresh** parameter identities distinct from
    /// `self`'s.
    ///
    /// This is a deliberate choice, not an oversight: [`PyParameter`] itself
    /// has no `#[derive(Clone)]` (see its struct-level doc — the type is
    /// designed so that copying its *value* and aliasing its *identity* are
    /// two different, explicitly-named operations, never conflated), so
    /// there is no "obvious" derived meaning for `#[derive(Clone)]` on
    /// `PyDense` to fall back on; the meaning has to be chosen deliberately.
    ///
    /// [`PyParameter::clone_param`] is exactly the right tool for this,
    /// despite being the *wrong* tool for [`PyDense::parameters`] below —
    /// the two call sites want opposite things:
    ///
    /// * `parameters()` needs the SAME identity handed to an optimizer, so a
    ///   `.set_data()` write is visible to this layer's own next `forward()`
    ///   — `clone_param()`'s fresh, independent identity would silently
    ///   break that (see `parameters()`'s doc for the full failure mode).
    /// * Cloning a whole `PyDense` — e.g. via [`PySequential::get_layer`],
    ///   which returns `self.layers[index].clone()` to hand a *copy* of a
    ///   layer back to Python — is the textbook case `clone_param()`'s
    ///   "independent copy, fresh id" semantics exist for: the caller
    ///   explicitly asked for a new, standalone layer object, and two
    ///   `PyDense`s that secretly shared one parameter's `id` (so that
    ///   mutating one layer's weights silently mutated the "different"
    ///   layer returned earlier) would be a much worse footgun than the
    ///   independent copy this impl actually produces.
    ///
    /// `hook_manager` is reset to [`HookManager::new`] on clone (hooks are
    /// per-object registrations, not layer state, and should not be shared
    /// across independent copies), matching this impl's pre-existing
    /// behaviour.
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let weight_param = self.weight_param.borrow(py).clone_param();
            let bias_param = self
                .bias_param
                .as_ref()
                .map(|bias| bias.borrow(py).clone_param());
            Self {
                layer: self.layer.clone(),
                weight_param: Py::new(py, weight_param).expect(
                    "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                ),
                bias_param: bias_param.map(|bias| {
                    Py::new(py, bias).expect(
                        "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                    )
                }),
                hook_manager: HookManager::new(),
                training: self.training,
            }
        })
    }
}

#[pymethods]
impl PyDense {
    /// Construct a dense (fully-connected) layer.
    ///
    /// # Arguments
    ///
    /// * `input_dim`  — number of input features.
    /// * `output_dim` — number of output features.
    /// * `use_bias`   — whether to include a bias term (default: `True`).
    /// * `activation` — optional activation name, e.g. `"relu"`, `"sigmoid"`, `"tanh"`.
    ///
    /// Weights are Xavier-uniform initialised; biases are zero-initialised.
    ///
    /// # Errors
    ///
    /// Raises whatever error [`Py::new`] itself can raise while allocating
    /// the fresh `PyParameter` Python objects backing the weight/bias (see
    /// [`PyDense`]'s struct-level "Autograd" doc for why they must be
    /// `Py<PyParameter>` from the start rather than bare `PyParameter`
    /// values) — in practice this is Python-interpreter-level allocation
    /// failure, since neither `PyParameter`'s own `#[new]` nor this
    /// construction path performs any additional fallible validation.
    #[new]
    #[pyo3(signature = (input_dim, output_dim, use_bias=None, activation=None))]
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        use_bias: Option<bool>,
        activation: Option<String>,
    ) -> PyResult<Self> {
        let use_bias = use_bias.unwrap_or(true);
        let mut layer = Dense::new_xavier(input_dim, output_dim, use_bias);

        if let Some(act) = activation {
            layer = layer.with_activation(act);
        }

        // Snapshot the freshly initialised weight/bias into their own
        // PyParameter identities exactly once, here at construction time.
        // Every later `forward()`/`parameters()` call reuses these same
        // `Py<PyParameter>` handles (see the "Autograd" section on the
        // struct doc comment) rather than re-deriving them from `layer` via
        // a fresh clone each time.
        Python::attach(|py| {
            let weight_param = Py::new(
                py,
                PyParameter::new(
                    PyTensor {
                        tensor: Arc::new(layer.weight().clone()),
                        requires_grad: true,
                        is_pinned: false,
                    },
                    Some(true),
                ),
            )?;
            let bias_param = layer
                .bias()
                .map(|b| {
                    Py::new(
                        py,
                        PyParameter::new(
                            PyTensor {
                                tensor: Arc::new(b.clone()),
                                requires_grad: true,
                                is_pinned: false,
                            },
                            Some(true),
                        ),
                    )
                })
                .transpose()?;

            Ok(Self {
                layer,
                weight_param,
                bias_param,
                hook_manager: HookManager::new(),
                training: false, // Start in eval mode by default
            })
        })
    }

    /// Forward pass through the layer.
    ///
    /// Computed entirely through tape-aware [`PyTensor`] operations (not
    /// `tenflowers_neural::layers::Dense::forward`) so the result correctly
    /// participates in implicit autograd whenever `input`, the weight, or
    /// the bias requires gradients. See the struct-level "Autograd" doc for
    /// why this snapshots each parameter via [`PyParameter::to_tensor`] and
    /// registers the snapshot via
    /// [`crate::implicit_autograd::mark_leaf_param`] (keyed by the
    /// parameter's stable `id()`) rather than [`crate::implicit_autograd::mark_leaf`].
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Execute pre-forward hooks
        self.hook_manager.execute_forward_hooks(input, input)?;

        // Snapshot the weight parameter's current value and (re-)register it
        // as this forward pass's tape leaf. `mark_leaf_param` is idempotent
        // per parameter `id()` (see its own doc), so calling it on every
        // forward pass is safe and cheap: after the first call for a given
        // `id()` it is just a hash-map lookup that finds the leaf already
        // tracked, regardless of how many *different* transient snapshots
        // `to_tensor()` has produced along the way.
        let (weight_id, weight_snapshot) = Python::attach(|py| -> PyResult<(usize, PyTensor)> {
            let weight_param = self.weight_param.borrow(py);
            Ok((weight_param.id(), weight_param.to_tensor()?))
        })?;
        crate::implicit_autograd::mark_leaf_param(&weight_snapshot, weight_id);

        // output = input @ weight
        let mut output = input.matmul(&weight_snapshot)?;

        // output = output + bias  (only if this layer has a bias)
        if let Some(bias_param) = &self.bias_param {
            let (bias_id, bias_snapshot) = Python::attach(|py| -> PyResult<(usize, PyTensor)> {
                let bias_param = bias_param.borrow(py);
                Ok((bias_param.id(), bias_param.to_tensor()?))
            })?;
            crate::implicit_autograd::mark_leaf_param(&bias_snapshot, bias_id);
            output = output.add(&bias_snapshot)?;
        }

        // Apply the configured activation, if any, through the tape-aware
        // free functions in `super::functions` (which call
        // `record_and_link_unary` internally) rather than
        // `tenflowers_core::ops::*` directly.
        let output = match self.layer.activation_name().as_deref() {
            Some("relu") => super::functions::relu(&output)?,
            Some("sigmoid") => super::functions::sigmoid(&output)?,
            Some("tanh") => super::functions::tanh(&output)?,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "PyDense: activation '{}' is not supported by the implicit-autograd \
                     forward path (only 'relu', 'sigmoid', 'tanh' are currently tape-aware); \
                     construct the layer with one of those, or with activation=None and apply \
                     the activation as a separate call",
                    other
                )));
            }
            None => output,
        };

        // Execute post-forward hooks
        self.hook_manager.execute_forward_hooks(input, &output)?;

        Ok(output)
    }

    /// Get layer parameters.
    ///
    /// # Identity-sharing guarantee
    ///
    /// Returns [`Py<PyParameter>`] handles that are additional **owning
    /// references to the exact same underlying Python objects** this layer
    /// itself holds in `weight_param`/`bias_param` — obtained via
    /// [`Py::clone_ref`], which bumps the Python object's reference count
    /// and hands back a second handle to the *same* object, as opposed to
    /// [`PyParameter::clone_param`], which would allocate a **new, distinct**
    /// `PyParameter` with its own fresh `id()` (see that method's own doc,
    /// and [`PyDense`]'s `impl Clone` above for the one situation where that
    /// *is* the right tool).
    ///
    /// This distinction is the entire point of this method returning
    /// `Vec<Py<PyParameter>>` rather than `Vec<PyTensor>`: every handle
    /// returned here has `id() == self.weight_param.borrow(py).id()` (and
    /// likewise for bias), which is exactly the identity `forward()` passes
    /// to [`crate::implicit_autograd::mark_leaf_param`] on every call. Two
    /// consequences follow directly from that shared identity:
    ///
    /// * `.grad()` on a handle returned here is populated after a
    ///   `.backward()` call that passes through this layer's `forward()`,
    ///   because `mark_leaf_param` recorded the gradient under this exact
    ///   `id()`.
    /// * `.set_data(new_value)` on a handle returned here — e.g. from an
    ///   optimizer's `.step()` — mutates the *same* `Arc<RwLock<Tensor<f32>>>`
    ///   cell this layer's own `forward()` reads from via
    ///   `weight_param.borrow(py).to_tensor()`, so the very next `forward()`
    ///   call on this `PyDense` sees the update. Using `clone_param()` here
    ///   instead would silently break this: an optimizer would update a
    ///   throwaway copy nobody ever reads again, while this layer kept
    ///   forwarding through its original, never-updated value — training
    ///   would appear to run without error while learning nothing.
    pub fn parameters(&self) -> Vec<Py<PyParameter>> {
        Python::attach(|py| {
            let mut params = Vec::new();

            params.push(self.weight_param.clone_ref(py));

            if let Some(bias_param) = &self.bias_param {
                params.push(bias_param.clone_ref(py));
            }

            params
        })
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        if let Some(weights) = self.layer.weights() {
            count += weights.size();
        }
        if let Some(bias) = self.layer.bias() {
            count += bias.size();
        }
        count
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Check if layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.layer.input_dim()
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.layer.output_dim()
    }

    /// Check if layer uses bias
    pub fn has_bias(&self) -> bool {
        self.layer.bias().is_some()
    }

    /// Get activation function name
    pub fn activation(&self) -> Option<String> {
        self.layer.activation_name()
    }

    /// Register a forward hook
    pub fn register_forward_hook(&self, hook: Py<PyAny>) -> PyResult<PyHookHandle> {
        self.hook_manager.register_forward_hook(hook)
    }

    /// Register a backward hook
    pub fn register_backward_hook(&self, hook: Py<PyAny>) -> PyResult<PyHookHandle> {
        self.hook_manager.register_backward_hook(hook)
    }

    /// Remove a hook by handle
    pub fn remove_hook(&self, handle: &PyHookHandle) -> PyResult<()> {
        self.hook_manager.remove_hook(handle)
    }

    /// Clear all hooks
    pub fn clear_hooks(&self) {
        self.hook_manager.clear_hooks();
    }

    /// Get hook count
    pub fn hook_count(&self) -> (usize, usize) {
        self.hook_manager.hook_count()
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyDense(in_features={}, out_features={}, bias={})",
            self.input_dim(),
            self.output_dim(),
            self.has_bias()
        )
    }

    /// String representation
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyDense {
    /// Execute forward hooks (internal method)
    fn execute_forward_hooks(&self, input: &PyTensor, output: &PyTensor) -> PyResult<()> {
        self.hook_manager.execute_forward_hooks(input, output)
    }

    /// Execute backward hooks (internal method)
    fn execute_backward_hooks(&self, grad_input: &PyTensor) -> PyResult<()> {
        self.hook_manager.execute_backward_hooks(grad_input)
    }
}

/// An ordered container of [`PyDense`] layers that are executed sequentially.
///
/// `PySequential` chains multiple dense layers so that the output of layer `i`
/// becomes the input of layer `i+1`.  It supports the same hook API as
/// `PyDense` and exposes aggregate parameter/training utilities.
///
/// # Python Example
///
/// ```python
/// import tenflowers as tf
///
/// model = tf.PySequential()
/// model.add(tf.PyDense(784, 256, activation='relu'))
/// model.add(tf.PyDense(256, 128, activation='relu'))
/// model.add(tf.PyDense(128, 10,  activation=None))
///
/// model.train()          # set all layers to training mode
/// out = model.forward(tf.ones([8, 784]))
/// print(out.shape())     # [8, 10]
/// print(model.num_parameters())
///
/// model.eval()           # switch to inference mode
/// out = model.forward(tf.zeros([1, 784]))
///
/// # Layer access
/// l0 = model.get_layer(0)
/// model.insert(1, tf.PyDense(256, 256, activation='relu'))
/// model.remove(1)
/// model.clear()
///
/// print(model.is_empty())  # True
/// ```
#[pyclass]
#[derive(Debug)]
pub struct PySequential {
    layers: Vec<PyDense>,
    hook_manager: HookManager,
}

impl Clone for PySequential {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            hook_manager: HookManager::new(),
        }
    }
}

#[pymethods]
impl PySequential {
    #[new]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            hook_manager: HookManager::new(),
        }
    }

    /// Add a dense layer to the sequential model
    pub fn add(&mut self, layer: PyDense) {
        self.layers.push(layer);
    }

    /// Forward pass through the model
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Execute pre-forward hooks
        self.hook_manager.execute_forward_hooks(input, input)?;

        let mut current_input = input.clone();

        // Chain through all layers
        for layer in &self.layers {
            current_input = layer.forward(&current_input)?;
        }

        // Execute post-forward hooks
        self.hook_manager
            .execute_forward_hooks(input, &current_input)?;

        Ok(current_input)
    }

    /// Get the number of layers in the model
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get all model parameters, in layer order.
    ///
    /// Each element is a [`Py<PyParameter>`] handle sharing identity with the
    /// owning [`PyDense`] layer's own `weight_param`/`bias_param` — see
    /// [`PyDense::parameters`]'s doc for the full identity-sharing guarantee,
    /// which holds per-layer and therefore also holds here: an optimizer
    /// iterating this method's return value can `.set_data()` any element and
    /// have that update be visible the next time `.forward()` runs on this
    /// same `PySequential`.
    ///
    /// No `Python<'_>` parameter is needed here: like [`PyDense::parameters`],
    /// this internally acquires the GIL via [`Python::attach`] itself (each
    /// per-layer `parameters()` call does so independently), keeping this
    /// method's signature a plain `&self` matching the rest of this file's
    /// aggregate-method conventions (e.g. [`PySequential::num_parameters`],
    /// [`PySequential::is_training`]).
    pub fn parameters(&self) -> Vec<Py<PyParameter>> {
        let mut all_params = Vec::new();
        for layer in &self.layers {
            all_params.extend(layer.parameters());
        }
        all_params
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }

    /// Set training mode for all layers
    #[pyo3(signature = (training=None))]
    pub fn train(&mut self, training: Option<bool>) {
        let training_mode = training.unwrap_or(true);
        for layer in &mut self.layers {
            layer.set_training(training_mode);
        }
    }

    /// Set evaluation mode for all layers
    pub fn eval(&mut self) {
        self.train(Some(false));
    }

    /// Check if model is in training mode
    pub fn is_training(&self) -> bool {
        self.layers
            .first()
            .map(|l| l.is_training())
            .unwrap_or(false)
    }

    /// Get layer at index
    pub fn get_layer(&self, index: usize) -> PyResult<PyDense> {
        if index < self.layers.len() {
            Ok(self.layers[index].clone())
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Layer index out of range",
            ))
        }
    }

    /// Insert layer at index
    pub fn insert(&mut self, index: usize, layer: PyDense) -> PyResult<()> {
        if index <= self.layers.len() {
            self.layers.insert(index, layer);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Insert index out of range",
            ))
        }
    }

    /// Remove layer at index
    pub fn remove(&mut self, index: usize) -> PyResult<PyDense> {
        if index < self.layers.len() {
            Ok(self.layers.remove(index))
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Remove index out of range",
            ))
        }
    }

    /// Clear all layers
    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// String representation of the model
    pub fn __str__(&self) -> String {
        if self.layers.is_empty() {
            "Sequential()".to_string()
        } else {
            let layer_strs: Vec<String> = self
                .layers
                .iter()
                .enumerate()
                .map(|(i, layer)| format!("  ({}): {}", i, layer.__repr__()))
                .collect();
            format!("Sequential(\n{}\n)", layer_strs.join("\n"))
        }
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    /// Register a forward hook
    pub fn register_forward_hook(&self, hook: Py<PyAny>) -> PyResult<PyHookHandle> {
        self.hook_manager.register_forward_hook(hook)
    }

    /// Register a backward hook
    pub fn register_backward_hook(&self, hook: Py<PyAny>) -> PyResult<PyHookHandle> {
        self.hook_manager.register_backward_hook(hook)
    }

    /// Remove a hook by handle
    pub fn remove_hook(&self, handle: &PyHookHandle) -> PyResult<()> {
        self.hook_manager.remove_hook(handle)
    }

    /// Clear all hooks
    pub fn clear_hooks(&self) {
        self.hook_manager.clear_hooks();
    }

    /// Get hook count
    pub fn hook_count(&self) -> (usize, usize) {
        self.hook_manager.hook_count()
    }
}

impl Default for PySequential {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a linear layer (alias for Dense)
#[pyfunction]
#[pyo3(signature = (input_dim, output_dim, bias=None))]
pub fn linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyResult<PyDense> {
    PyDense::new(input_dim, output_dim, bias, None)
}

/// Create a ReLU dense layer
#[pyfunction]
#[pyo3(signature = (input_dim, output_dim, bias=None))]
pub fn relu_linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyResult<PyDense> {
    PyDense::new(input_dim, output_dim, bias, Some("relu".to_string()))
}

/// Create a sigmoid dense layer
#[pyfunction]
#[pyo3(signature = (input_dim, output_dim, bias=None))]
pub fn sigmoid_linear(
    input_dim: usize,
    output_dim: usize,
    bias: Option<bool>,
) -> PyResult<PyDense> {
    PyDense::new(input_dim, output_dim, bias, Some("sigmoid".to_string()))
}

/// Create a tanh dense layer
#[pyfunction]
#[pyo3(signature = (input_dim, output_dim, bias=None))]
pub fn tanh_linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyResult<PyDense> {
    PyDense::new(input_dim, output_dim, bias, Some("tanh".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end proof that [`PyDense`]'s `PyParameter`-based redesign is
    /// genuinely optimizer-compatible: a `Py<PyParameter>` handle obtained
    /// from `.parameters()` shares identity with whatever `.forward()`
    /// actually snapshots and computes with, an optimizer-style
    /// `.set_data()` write through that handle is visible to this SAME
    /// layer's next `.forward()` call, and `.grad()` remains correctly
    /// retrievable through that SAME handle after a second, independent
    /// `.backward()` pass — proving the parameter's `id()` survived the
    /// mutation.
    ///
    /// The layer under test is `PyDense::new(4, 3, Some(true), None)`
    /// (`activation=None` deliberately, so the forward math is exactly
    /// `output = input @ weight + bias` with no nonlinearity, keeping the
    /// expected numeric relationship between the pre- and post-update
    /// outputs exactly predictable rather than merely "different").
    ///
    /// # The math this test verifies numerically
    ///
    /// For a fixed input `x` (shape `[1, 4]`) and loss `L = sum(x @ W + b)`
    /// (shape `[1, 3]` reduced to a scalar):
    ///
    /// * `dL/dW[i, j] = x[0, i]` for every output column `j` (each `W[i, j]`
    ///   affects exactly one output element `output[0, j] = ... + x[0,i] *
    ///   W[i,j] + ...`, and summing over `j` does not change the coefficient
    ///   on `W[i,j]` itself) — i.e. `grad_W` is `x` broadcast across all 3
    ///   output columns, independent of `W`'s actual value.
    /// * `dL/db[j] = 1` for every `j` (each `b[j]` contributes additively to
    ///   exactly one output element, with a fixed coefficient of 1).
    ///
    /// After an SGD step `W' = W - lr * grad_W`, `b' = b - lr * grad_b`, the
    /// *change* in output (before vs. after, same input `x`) is, by
    /// linearity of matmul + add:
    ///
    /// ```text
    /// output' - output = x @ (W' - W) + (b' - b)
    ///                   = x @ (-lr * grad_W) + (-lr * grad_b)
    /// ```
    ///
    /// Substituting `grad_W[i,j] = x[0,i]` (independent of `j`) collapses
    /// `(x @ (-lr * grad_W))[0, j] = -lr * sum_i(x[0,i] * grad_W[i,j]) = -lr
    /// * sum_i(x[0,i]^2)` — the SAME value for every output column `j`,
    /// since neither `grad_W[i,j]` nor the sum depends on `j`. With `x =
    ///   [1, 2, 3, 4]`, `sum_i(x[0,i]^2) = 1 + 4 + 9 + 16 = 30`, and `lr =
    ///   0.1`: `output'[0,j] - output[0,j] = -0.1 * 30 + (-0.1 * 1) = -3.0 -
    ///   0.1 = -3.1` for every `j in {0, 1, 2}` — asserted below with a
    ///   `1e-4` tolerance.
    ///
    /// Because neither `grad_W` nor `grad_b` depends on `W`'s or `b`'s
    /// actual *value* (only on `x`, which is unchanged), the gradient
    /// computed by the SECOND backward pass (after the update) is expected
    /// to be numerically identical to the first — this is also asserted
    /// explicitly, and is exactly the check that would fail if `set_data`
    /// had silently broken the parameter's tape identity.
    #[test]
    fn parameters_share_identity_with_forward_across_an_optimizer_update_cycle() {
        Python::initialize();

        Python::attach(|py| -> PyResult<()> {
            let lr: f32 = 0.1;

            let dense = PyDense::new(4, 3, Some(true), None)
                .expect("PyDense::new should construct with activation=None");

            // Step 1: obtain the layer's Parameter handles. These are the
            // SAME handles used throughout the rest of this test — this is
            // the crux of what is being proven.
            let params = dense.parameters();
            assert_eq!(
                params.len(),
                2,
                "expected exactly [weight, bias] since use_bias=Some(true)"
            );
            let weight_handle = &params[0];
            let bias_handle = &params[1];
            let weight_id = weight_handle.borrow(py).id();
            let bias_id = bias_handle.borrow(py).id();

            // Step 2: build a small, deterministic input and run the first
            // forward + loss + backward cycle.
            let x_tensor =
                tenflowers_core::Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4])
                    .expect("from_vec should succeed for matching data/shape lengths");
            let x = PyTensor {
                tensor: Arc::new(x_tensor),
                requires_grad: false,
                is_pinned: false,
            };

            let output1 = dense
                .forward(&x)
                .expect("first forward pass should succeed");
            assert_eq!(output1.shape(), vec![1, 3]);
            let output1_values = output1.tensor.data().to_vec();

            let loss1 = crate::math_ops::sum(&output1, None, None)
                .expect("sum reduction should succeed and remain tape-linked");
            // `PyTensor::backward` (tensor_ops.rs) is a private wrapper
            // around this exact call; calling it directly here is the
            // established idiom for Rust-side tests in this crate (see
            // e.g. `neural/normalization.rs`, `neural/extended_optimizers/mod.rs`).
            crate::implicit_autograd::run_backward(&loss1)
                .expect("first backward pass should succeed");

            // Step 3: retrieve gradients through the SAME handles from step 1.
            let grad_w1 = weight_handle
                .borrow(py)
                .grad()
                .expect("weight gradient should be populated after backward()");
            let grad_b1 = bias_handle
                .borrow(py)
                .grad()
                .expect("bias gradient should be populated after backward()");

            // Expected: grad_W[i,j] = x[0,i] for every j (broadcast across
            // the 3 output columns); grad_b[j] = 1 for every j.
            let expected_grad_w: Vec<f32> = vec![
                1.0, 1.0, 1.0, // row 0: x[0,0] = 1.0
                2.0, 2.0, 2.0, // row 1: x[0,1] = 2.0
                3.0, 3.0, 3.0, // row 2: x[0,2] = 3.0
                4.0, 4.0, 4.0, // row 3: x[0,3] = 4.0
            ];
            let expected_grad_b: Vec<f32> = vec![1.0, 1.0, 1.0];

            let grad_w1_values = grad_w1.tensor.data().to_vec();
            let grad_b1_values = grad_b1.tensor.data().to_vec();
            assert_eq!(grad_w1_values.len(), expected_grad_w.len());
            for (actual, expected) in grad_w1_values.iter().zip(expected_grad_w.iter()) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "grad_W mismatch: actual={actual}, expected={expected}"
                );
            }
            for (actual, expected) in grad_b1_values.iter().zip(expected_grad_b.iter()) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "grad_b mismatch: actual={actual}, expected={expected}"
                );
            }

            // Step 4/5: manual SGD update `new = old - lr * grad`, applied
            // via `.set_data()` on the SAME Py<PyParameter> handles from
            // step 1.
            let old_weight_tensor = weight_handle
                .borrow(py)
                .to_tensor()
                .expect("to_tensor should succeed")
                .tensor
                .as_ref()
                .clone();
            let old_bias_tensor = bias_handle
                .borrow(py)
                .to_tensor()
                .expect("to_tensor should succeed")
                .tensor
                .as_ref()
                .clone();
            let grad_w1_raw = grad_w1.tensor.as_ref().clone();
            let grad_b1_raw = grad_b1.tensor.as_ref().clone();

            let scaled_grad_w = grad_w1_raw
                .scalar_mul(lr)
                .expect("scalar_mul should succeed");
            let scaled_grad_b = grad_b1_raw
                .scalar_mul(lr)
                .expect("scalar_mul should succeed");
            let new_weight_tensor = old_weight_tensor
                .sub(&scaled_grad_w)
                .expect("sub should succeed for matching shapes");
            let new_bias_tensor = old_bias_tensor
                .sub(&scaled_grad_b)
                .expect("sub should succeed for matching shapes");

            weight_handle
                .borrow(py)
                .set_data(new_weight_tensor)
                .expect("set_data should succeed for the weight parameter");
            bias_handle
                .borrow(py)
                .set_data(new_bias_tensor)
                .expect("set_data should succeed for the bias parameter");

            // Step 6: run a SECOND, independent forward + loss + backward
            // cycle on the SAME PyDense instance, same input.
            let output2 = dense
                .forward(&x)
                .expect("second forward pass (after set_data) should succeed");
            let output2_values = output2.tensor.data().to_vec();

            // Step 7a: the second forward pass must reflect the updated
            // weight/bias values. Expected delta = -lr * sum_i(x[0,i]^2) -
            // lr = -0.1 * 30.0 - 0.1 = -3.1 for every output column.
            let expected_delta: f32 = -lr * 30.0 - lr;
            assert_eq!(output1_values.len(), output2_values.len());
            for (v1, v2) in output1_values.iter().zip(output2_values.iter()) {
                let actual_delta = v2 - v1;
                assert!(
                    (actual_delta - expected_delta).abs() < 1e-4,
                    "output delta mismatch: actual={actual_delta}, expected={expected_delta} \
                     (output1={v1}, output2={v2})"
                );
            }

            // Step 7b: the second backward pass's gradient must still be
            // correctly retrievable via `.grad()` on the SAME handles from
            // step 1 — proving `id()` stability survived `set_data`. Since
            // neither grad_W nor grad_b depends on W's/b's actual value
            // (only on x, which is unchanged), the expected values are
            // identical to the first backward pass.
            let loss2 = crate::math_ops::sum(&output2, None, None)
                .expect("sum reduction should succeed on the second forward pass's output");
            crate::implicit_autograd::run_backward(&loss2)
                .expect("second backward pass should succeed");

            // Confirm identity did not change across the whole cycle.
            assert_eq!(
                weight_handle.borrow(py).id(),
                weight_id,
                "weight parameter id must be stable across set_data()"
            );
            assert_eq!(
                bias_handle.borrow(py).id(),
                bias_id,
                "bias parameter id must be stable across set_data()"
            );

            let grad_w2 = weight_handle
                .borrow(py)
                .grad()
                .expect("weight gradient should be populated after the SECOND backward()");
            let grad_b2 = bias_handle
                .borrow(py)
                .grad()
                .expect("bias gradient should be populated after the SECOND backward()");
            let grad_w2_values = grad_w2.tensor.data().to_vec();
            let grad_b2_values = grad_b2.tensor.data().to_vec();

            for (actual, expected) in grad_w2_values.iter().zip(expected_grad_w.iter()) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "second-pass grad_W mismatch: actual={actual}, expected={expected}"
                );
            }
            for (actual, expected) in grad_b2_values.iter().zip(expected_grad_b.iter()) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "second-pass grad_b mismatch: actual={actual}, expected={expected}"
                );
            }

            Ok(())
        })
        .expect("test: operation should succeed");
    }
}
