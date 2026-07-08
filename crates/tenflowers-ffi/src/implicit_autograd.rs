//! Implicit (PyTorch-style) eager automatic differentiation.
//!
//! # Design
//!
//! TenfloweRS already has a fully working *explicit* tape API
//! ([`tenflowers_autograd::GradientTape`], exposed to Python as
//! [`crate::neural::gradient_tape::PyGradientTape`]): a user calls
//! `tape.watch(x)`, performs operations on the returned
//! [`tenflowers_autograd::TrackedTensor`], and then calls `tape.gradient(y, x)`
//! on the *same* tape object.
//!
//! PyTorch-style eager autograd has no explicit tape in user code at all:
//!
//! ```python
//! x.set_requires_grad(True)
//! y = x * x
//! y.backward()
//! g = x.grad()
//! ```
//!
//! Something has to *implicitly* track the computation graph from the moment
//! `requires_grad=True` starts, without the user ever constructing a
//! `GradientTape` themselves. This module provides that missing piece by
//! layering a thread-local, auto-activating `GradientTape` underneath
//! [`crate::tensor_ops::PyTensor`], and reusing the exact same
//! `TrackedTensor` op-recording + `GradientTape::gradient` backward-pass
//! machinery that the explicit `PyGradientTape` already relies on. No new
//! gradient math is introduced anywhere in this module — it only decides
//! *when* to call into the existing, tested autodiff engine.
//!
//! ## Why a side-table instead of a `PyTensor` field
//!
//! [`crate::tensor_ops::PyTensor`] is constructed via `PyTensor { tensor,
//! requires_grad, is_pinned }` struct literals at ~190 call sites across this
//! crate. Adding a required field to the struct would force updating every
//! one of those literals purely for field-list completeness, which is an
//! enormous, unrelated blast radius for this feature and directly conflicts
//! with the instruction to avoid touching code that isn't part of the task.
//!
//! Instead, every piece of implicit-autograd state is keyed by a
//! [`PyTensor`]'s *stable identity* — the address of its `Arc<Tensor<f32>>`
//! allocation (see [`tensor_key`]) — and stored in thread-local side-tables.
//! `PyTensor` itself is completely unmodified by this module; only a handful
//! of new methods are added to it (in `tensor_ops.rs`) that delegate here.
//!
//! ### The address-reuse hazard, and how [`IDENTITY_ANCHORS`] closes it
//!
//! An `Arc<Tensor<f32>>` address is only a *safe* identity key for as long as
//! something keeps that exact `Arc` allocation alive. Once every strong
//! reference to it is dropped (e.g. the Python tensor object is garbage
//! collected), the allocator is free to reuse that address for a completely
//! unrelated `Arc` allocated later — including another `PyTensor`'s. Without
//! precaution, a side-table keyed purely by address would then confuse the
//! new, unrelated tensor for the old one (this was caught during development
//! via exactly this scenario: a `requires_grad=False` tensor's `.grad()`
//! spuriously returned a stale gradient that belonged to an earlier,
//! already-freed tensor whose `Arc` address had been reused).
//!
//! [`IDENTITY_ANCHORS`] closes this hole: whenever a key is inserted into
//! [`TRACKED_REGISTRY`], [`LEAVES`], or [`GRAD_STORE`], a strong
//! `Arc<Tensor<f32>>` clone is also stashed in `IDENTITY_ANCHORS` under that
//! same key, guaranteeing the address cannot be freed (and therefore cannot
//! be reused for a different tensor) for as long as *any* table still
//! references that key. The anchor for a given key is only ever removed once
//! every table that could reference it no longer does.
//!
//! ## Components
//!
//! * [`IMPLICIT_TAPE`] — one [`GradientTape`] per thread, created lazily and
//!   reused for the life of the thread. It auto-activates: nothing must be
//!   explicitly "started", it simply exists and is always recording.
//! * [`TRACKED_REGISTRY`] — maps a tensor's stable identity to the
//!   [`TrackedTensor<f32>`] that represents it on the implicit tape. Populated
//!   by [`mark_leaf`] (for tensors with `requires_grad=True`) and by
//!   [`record_and_link_binary`] / [`record_and_link_unary`] (for the outputs
//!   of tracked operations).
//! * [`LEAVES`] — the set of tensors that became tracked *directly* via
//!   `set_requires_grad(True)` (as opposed to becoming tracked because an
//!   input was tracked). These are exactly the tensors `.backward()`
//!   differentiates with respect to, mirroring PyTorch's notion of a "leaf".
//! * [`GRAD_STORE`] — the `.grad` storage itself: after `.backward()` runs,
//!   each leaf's computed gradient is stored here keyed by the leaf's stable
//!   identity, so `PyTensor::grad()` can look it up later.
//! * [`IDENTITY_ANCHORS`] — see above; keeps a key's `Arc<Tensor<f32>>`
//!   allocation alive (and therefore its address un-reusable) for exactly as
//!   long as any of the three tables above still references that key.
//!
//! ## Generic operation hook
//!
//! [`record_and_link_binary`] and [`record_and_link_unary`] are dispatched on
//! a small [`BinaryOpKind`] / [`UnaryOpKind`] tag rather than being one
//! function per named op. Any binary op that has a `TrackedTensor` method
//! (`add`, `sub`, `mul`, `div`, `pow`, `matmul`, ...) can be wired in by
//! adding one match arm and one call site — nothing about the hook itself is
//! specific to `add`/`mul`/`matmul` by name.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

use crate::tensor_ops::PyTensor;

/// Stable identity of a [`PyTensor`] for use as a side-table key.
///
/// Two [`PyTensor`] values compare equal under this key exactly when they
/// share the same underlying `Arc<Tensor<f32>>` allocation (e.g. one was
/// cloned from the other). A fresh tensor produced by any operation gets a
/// fresh key because it owns a fresh `Arc` allocation.
pub fn tensor_key(tensor: &PyTensor) -> usize {
    Arc::as_ptr(&tensor.tensor) as usize
}

thread_local! {
    /// The implicit, auto-activating gradient tape for this thread.
    ///
    /// Unlike [`crate::neural::gradient_tape::PyGradientTape`], nothing ever
    /// calls `start_recording`/`stop_recording` on this tape from Python —
    /// it is always live for the lifetime of the thread. `GradientTape` is
    /// cheap to hold (an `Arc<Mutex<..>>` around a growable node list), so
    /// keeping one alive per thread is not a meaningful cost.
    static IMPLICIT_TAPE: GradientTape = GradientTape::new();

    /// Maps a tensor's stable identity to its tracked counterpart on the
    /// implicit tape, if any. Absence means the tensor is not (yet)
    /// participating in implicit autograd.
    static TRACKED_REGISTRY: RefCell<HashMap<usize, Arc<TrackedTensor<f32>>>> =
        RefCell::new(HashMap::new());

    /// Tensors that became tracked directly via `set_requires_grad(True)`,
    /// in the order they were registered, paired with their stable
    /// [`tensor_key`] identity (kept alongside the `TrackedTensor` because
    /// the identity must still be usable to key [`GRAD_STORE`] *after*
    /// [`run_backward`] clears [`TRACKED_REGISTRY`] and this list). `.backward()`
    /// differentiates with respect to exactly this set.
    static LEAVES: RefCell<Vec<(usize, Arc<TrackedTensor<f32>>)>> =
        const { RefCell::new(Vec::new()) };

    /// Populated by `.backward()`: maps a leaf's stable [`tensor_key`]
    /// identity to its computed gradient tensor. Deliberately **not**
    /// cleared when the tape/registry/leaves are reset after a successful
    /// backward pass — `.grad()` must remain readable afterward, exactly
    /// like a real PyTorch leaf tensor's `.grad` attribute survives after
    /// its graph is freed.
    static GRAD_STORE: RefCell<HashMap<usize, Tensor<f32>>> = RefCell::new(HashMap::new());

    /// Keeps a [`tensor_key`] identity's `Arc<Tensor<f32>>` allocation alive
    /// — and therefore its address safe to use as a hash-map key — for
    /// exactly as long as [`TRACKED_REGISTRY`], [`LEAVES`], or [`GRAD_STORE`]
    /// still references that key. See the module-level "address-reuse
    /// hazard" documentation above for why this exists.
    static IDENTITY_ANCHORS: RefCell<HashMap<usize, Arc<Tensor<f32>>>> =
        RefCell::new(HashMap::new());
}

/// Look up the [`TrackedTensor`] a [`PyTensor`] is linked to on the implicit
/// tape, if it is currently participating in implicit autograd.
pub fn lookup_tracked(tensor: &PyTensor) -> Option<Arc<TrackedTensor<f32>>> {
    let key = tensor_key(tensor);
    TRACKED_REGISTRY.with(|registry| registry.borrow().get(&key).cloned())
}

/// Pin `tensor`'s `Arc<Tensor<f32>>` allocation alive under its
/// [`tensor_key`] so that key can safely be used elsewhere (see
/// [`IDENTITY_ANCHORS`]). Idempotent: anchoring an already-anchored key just
/// keeps one clone, since a `HashMap` insert with the same key simply
/// replaces the (identical) previous value.
fn anchor_identity(tensor: &PyTensor) {
    let key = tensor_key(tensor);
    IDENTITY_ANCHORS.with(|anchors| {
        anchors.borrow_mut().insert(key, Arc::clone(&tensor.tensor));
    });
}

/// Register `tracked` as the implicit-tape counterpart of `tensor`.
///
/// Also anchors `tensor`'s identity (see [`anchor_identity`]) so the key
/// just inserted into [`TRACKED_REGISTRY`] cannot be invalidated by the
/// tensor's `Arc` being freed and its address reused for something else
/// while it is still registered.
fn register_tracked(tensor: &PyTensor, tracked: Arc<TrackedTensor<f32>>) {
    let key = tensor_key(tensor);
    anchor_identity(tensor);
    TRACKED_REGISTRY.with(|registry| {
        registry.borrow_mut().insert(key, tracked);
    });
}

/// Mark `tensor` as a leaf that requires gradients: begin watching it on the
/// implicit tape (if not already watched) and record it as a leaf so that
/// `.backward()` knows to differentiate with respect to it.
///
/// Called from [`PyTensor::set_requires_grad`] when the flag transitions to
/// `true`. Calling this multiple times on the same tensor is a harmless
/// no-op after the first call (idempotent), matching the fact that a real
/// tensor's `Arc` identity does not change.
pub fn mark_leaf(tensor: &PyTensor) {
    if lookup_tracked(tensor).is_some() {
        // Already tracked (either already marked as a leaf, or already the
        // output of a tracked operation) — nothing further to do.
        return;
    }

    let tracked = IMPLICIT_TAPE.with(|tape| tape.watch((*tensor.tensor).clone()));
    let tracked = Arc::new(tracked);
    let key = tensor_key(tensor);

    register_tracked(tensor, Arc::clone(&tracked));
    LEAVES.with(|leaves| leaves.borrow_mut().push((key, tracked)));
}

/// The binary tensor operations wired into the implicit-tracking hook.
///
/// Each variant corresponds to one `TrackedTensor<f32>` method that records
/// itself onto whatever tape it is linked to (see
/// `tenflowers_autograd::tape::tracked_tensor`). Adding support for another
/// binary op means adding one variant here and one match arm in
/// [`record_and_link_binary`] — the hook itself has no per-op special
/// casing beyond that dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

/// The unary tensor operations wired into the implicit-tracking hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOpKind {
    Relu,
    Sigmoid,
    Tanh,
    /// `sum(axes, keepdims)` reduction.
    Sum {
        axes: Option<Vec<i32>>,
        keepdims: bool,
    },
    /// `mean(axes, keepdims)` reduction.
    Mean {
        axes: Option<Vec<i32>>,
        keepdims: bool,
    },
}

/// After computing `result = op(lhs, rhs)` eagerly (as `PyTensor` operations
/// already do via `tenflowers_core::ops::*`), call this to *additionally*
/// record the same operation onto the implicit tape — but only if at least
/// one of `lhs`/`rhs` is actually participating in implicit autograd.
///
/// This keeps the non-autograd path (the overwhelming majority of tensor
/// operations, where neither operand has `requires_grad=True`) at zero
/// overhead beyond two hash-map lookups.
///
/// # Errors
///
/// Returns an error only if both operands are tracked but recording the
/// operation on the tape itself fails (e.g. a poisoned lock) — never because
/// gradients are unsupported for `kind`, since only ops with a real
/// `TrackedTensor` backward implementation are wired in here.
pub fn record_and_link_binary(
    kind: BinaryOpKind,
    lhs: &PyTensor,
    rhs: &PyTensor,
    result: &PyTensor,
) -> PyResult<()> {
    let lhs_tracked = lookup_tracked(lhs);
    let rhs_tracked = lookup_tracked(rhs);

    if lhs_tracked.is_none() && rhs_tracked.is_none() {
        return Ok(());
    }

    // Both sides must be tracked tensors to invoke a TrackedTensor method
    // (it needs an `id` on both operands to build the graph edge). A
    // constant (non-tracked) operand is watched on the *same shared implicit
    // tape* so it receives a genuine, globally-unique id from the tape's own
    // counter — it is deliberately **not** pushed onto `LEAVES`, so
    // `.backward()` never differentiates with respect to it.
    //
    // This must not use `TrackedTensor::new` (which always assigns `id: 0`
    // and no tape) for the constant side: the tape's own id counter also
    // starts at 0, so the very first real leaf watched on a fresh tape gets
    // id 0 too, and a `TrackedTensor::new`-synthesized "constant" would
    // silently collide with it in the backward pass's gradient map, causing
    // gradients to be summed across two unrelated tensors.
    let lhs_tt = lhs_tracked
        .unwrap_or_else(|| Arc::new(IMPLICIT_TAPE.with(|tape| tape.watch((*lhs.tensor).clone()))));
    let rhs_tt = rhs_tracked
        .unwrap_or_else(|| Arc::new(IMPLICIT_TAPE.with(|tape| tape.watch((*rhs.tensor).clone()))));

    let recorded = match kind {
        BinaryOpKind::Add => lhs_tt.add(&rhs_tt),
        BinaryOpKind::Sub => lhs_tt.sub(&rhs_tt),
        BinaryOpKind::Mul => lhs_tt.mul(&rhs_tt),
        BinaryOpKind::Div => lhs_tt.div(&rhs_tt),
        BinaryOpKind::MatMul => lhs_tt.matmul(&rhs_tt),
    }
    .map_err(|e| PyRuntimeError::new_err(format!("implicit autograd recording failed: {e}")))?;

    register_tracked(result, Arc::new(recorded));
    Ok(())
}

/// Unary counterpart of [`record_and_link_binary`]: record `op(input) =
/// result` onto the implicit tape if `input` is currently tracked.
pub fn record_and_link_unary(
    kind: UnaryOpKind,
    input: &PyTensor,
    result: &PyTensor,
) -> PyResult<()> {
    let Some(input_tt) = lookup_tracked(input) else {
        return Ok(());
    };

    let recorded = match kind {
        UnaryOpKind::Relu => input_tt.relu(),
        UnaryOpKind::Sigmoid => input_tt.sigmoid(),
        UnaryOpKind::Tanh => input_tt.tanh(),
        UnaryOpKind::Sum { axes, keepdims } => input_tt.sum(axes, keepdims),
        UnaryOpKind::Mean { axes, keepdims } => input_tt.mean(axes, keepdims),
    }
    .map_err(|e| PyRuntimeError::new_err(format!("implicit autograd recording failed: {e}")))?;

    register_tracked(result, Arc::new(recorded));
    Ok(())
}

/// Run the backward pass for `output` (as `output.backward()`), populating
/// `.grad` on every leaf tensor that contributed to it.
///
/// After a successful backward pass, the implicit tape's recorded nodes, the
/// tracked-tensor registry, and the leaf list are all reset (see
/// [`GradientTape::clear`]) — mirroring PyTorch's default `retain_graph=False`
/// behaviour, where the graph is freed once backward has consumed it. Two
/// independent forward passes (e.g. two separate test functions each
/// building their own tiny graph and calling `.backward()` once) must not
/// leak state into each other: without this reset, a *later* `.backward()`
/// call would re-walk *every* operation ever recorded on the thread's
/// implicit tape since the process started, including operations on
/// completely unrelated tensors from earlier computations, and would
/// differentiate with respect to every leaf ever registered rather than just
/// the ones that actually feed the new target.
///
/// `.grad()` remains valid after this reset because [`GRAD_STORE`] is keyed
/// by each leaf's stable [`tensor_key`] (the tensor's own `Arc` address) and
/// [`IDENTITY_ANCHORS`] keeps that exact `Arc` allocation alive for as long
/// as `GRAD_STORE` references its key — the key does not depend on the tape
/// or registry (both of which this function clears) at all.
///
/// # Errors
///
/// * Raises `RuntimeError` if `output` never participated in implicit
///   autograd (no operation recorded it on the implicit tape) — this mirrors
///   PyTorch raising `RuntimeError: element 0 of tensors does not require
///   grad and does not have a grad_fn` rather than silently doing nothing.
/// * Raises `RuntimeError` if the underlying tape's backward pass fails
///   (e.g. an operation without a registered backward rule appears on the
///   graph). The tape is left untouched in this case so the error can be
///   investigated (e.g. via the explicit `PyGradientTape` API) rather than
///   silently discarding the graph on failure.
pub fn run_backward(output: &PyTensor) -> PyResult<()> {
    let Some(target) = lookup_tracked(output) else {
        return Err(PyRuntimeError::new_err(
            "backward() called on a tensor with no recorded computation graph \
             (it was not derived from any tensor with requires_grad=True); \
             nothing to differentiate",
        ));
    };

    let leaves: Vec<(usize, TrackedTensor<f32>)> = LEAVES.with(|leaves| {
        leaves
            .borrow()
            .iter()
            .map(|(key, tt)| (*key, (**tt).clone()))
            .collect()
    });

    if leaves.is_empty() {
        return Err(PyRuntimeError::new_err(
            "backward() found no leaf tensors (nothing had requires_grad=True set); \
             nothing to differentiate",
        ));
    }

    let leaf_tensors: Vec<TrackedTensor<f32>> = leaves.iter().map(|(_, tt)| tt.clone()).collect();
    let targets = [(*target).clone()];
    let grads = IMPLICIT_TAPE
        .with(|tape| tape.gradient(&targets, &leaf_tensors))
        .map_err(|e| PyRuntimeError::new_err(format!("backward() failed: {e}")))?;

    GRAD_STORE.with(|store| {
        let mut store = store.borrow_mut();
        for ((key, _), grad) in leaves.iter().zip(grads) {
            if let Some(grad_tensor) = grad {
                store.insert(*key, grad_tensor);
            }
        }
    });

    // The graph has now been fully consumed by this backward pass. Free it
    // so the next independent forward pass starts from a clean tape, exactly
    // as PyTorch frees the graph by default after `.backward()`.
    IMPLICIT_TAPE.with(GradientTape::clear);
    TRACKED_REGISTRY.with(|registry| registry.borrow_mut().clear());
    LEAVES.with(|leaves| leaves.borrow_mut().clear());

    // Prune identity anchors for every key that is no longer referenced by
    // any table. A key survives the prune exactly when it is currently
    // present in GRAD_STORE — which holds every leaf gradient ever computed
    // by *any* backward() call on this thread, not just this one. Checking
    // membership in the real, persistent GRAD_STORE map (rather than only
    // this call's local `leaves_with_grad` set) is essential: this function
    // runs once per independent backward() call, and an earlier call's
    // gradients (and their anchors) must survive a *later*, unrelated
    // call's cleanup. Getting this wrong (checking only the current call's
    // leaves) was caught by `address_reuse_does_not_leak_stale_gradient`
    // during development: it silently dropped anchors for every
    // previously-computed gradient on every subsequent backward() call,
    // freeing addresses that GRAD_STORE still referenced.
    //
    // Anything NOT in GRAD_STORE (leaves that received no gradient, and
    // every non-leaf intermediate tensor that was only ever in
    // TRACKED_REGISTRY) is now unreachable from all three tables and must
    // not keep its Arc pinned forever, or every implicit tensor operation
    // would leak memory for the life of the process.
    GRAD_STORE.with(|store| {
        let store = store.borrow();
        IDENTITY_ANCHORS.with(|anchors| {
            anchors
                .borrow_mut()
                .retain(|key, _| store.contains_key(key));
        });
    });

    Ok(())
}

/// Retrieve the gradient computed for `tensor` by a previous `.backward()`
/// call, if any.
///
/// Looks `tensor` up in [`GRAD_STORE`] directly by its stable [`tensor_key`]
/// rather than via [`TRACKED_REGISTRY`], because [`run_backward`] clears the
/// registry once the graph has been consumed — `.grad()` must still work
/// afterward.
///
/// # Errors
///
/// Raises `RuntimeError` if no gradient has been computed for `tensor` yet —
/// either `.backward()` was never called, or `tensor` was not a leaf / was
/// not on the path to whatever was differentiated.
pub fn get_grad(tensor: &PyTensor) -> PyResult<PyTensor> {
    let key = tensor_key(tensor);
    GRAD_STORE.with(|store| {
        store
            .borrow()
            .get(&key)
            .map(|grad| PyTensor {
                tensor: Arc::new(grad.clone()),
                requires_grad: false,
                is_pinned: false,
            })
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "no gradient has been computed for this tensor yet; either it never had \
                     requires_grad=True set, or .backward() has not been called yet on a \
                     tensor derived from it",
                )
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reset all thread-local implicit-autograd state. The implicit tape
    /// itself is intentionally *not* resettable (mirrors real usage — a
    /// process does not tear down its autograd engine between calls), but
    /// tests need a clean leaf/registry/grad-store view so they do not
    /// observe leaves registered by other tests on the same thread.
    ///
    /// Test threads in `cargo nextest` are usually one-shot per test binary
    /// invocation, but nextest can reuse OS threads across `#[test]`
    /// functions within one process, so we reset explicitly rather than
    /// relying on thread-per-test isolation.
    fn reset_for_test() {
        TRACKED_REGISTRY.with(|r| r.borrow_mut().clear());
        LEAVES.with(|l| l.borrow_mut().clear());
        GRAD_STORE.with(|g| g.borrow_mut().clear());
        IDENTITY_ANCHORS.with(|a| a.borrow_mut().clear());
    }

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction must succeed");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    #[test]
    fn leaf_without_requires_grad_is_not_tracked() {
        reset_for_test();
        let x = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
        assert!(lookup_tracked(&x).is_none());
    }

    #[test]
    fn mark_leaf_is_idempotent() {
        reset_for_test();
        let x = make_tensor(vec![1.0, 2.0], &[2]);
        mark_leaf(&x);
        let first_id = lookup_tracked(&x).expect("must be tracked").id;
        mark_leaf(&x);
        let second_id = lookup_tracked(&x).expect("must still be tracked").id;
        assert_eq!(
            first_id, second_id,
            "marking a leaf twice must not re-watch it"
        );
        let leaf_count = LEAVES.with(|l| l.borrow().len());
        assert_eq!(
            leaf_count, 1,
            "leaf list must not grow on repeated mark_leaf"
        );
    }

    /// Build `scalar = sum(x + y)` with implicit tracking wired up exactly as
    /// `PyTensor::add` / `math_ops::sum` do, and run `.backward()` on it.
    /// Returns the leaf `x` (with `requires_grad` already set) so the caller
    /// can inspect `get_grad(&x)`.
    ///
    /// `x` must not already be tracked when passed in — this always marks it
    /// as a fresh leaf via `mark_leaf`.
    fn add_then_sum_backward(x: PyTensor, y: &PyTensor) -> PyTensor {
        mark_leaf(&x);

        let raw = tenflowers_core::ops::add(&x.tensor, &y.tensor).expect("add must succeed");
        let sum_result = PyTensor {
            tensor: Arc::new(raw),
            requires_grad: true,
            is_pinned: false,
        };
        record_and_link_binary(BinaryOpKind::Add, &x, y, &sum_result)
            .expect("recording add must succeed");

        // Reduce to scalar via sum so backward() has something to seed with ones.
        let summed =
            tenflowers_core::ops::sum(&sum_result.tensor, None, false).expect("sum must succeed");
        let scalar = PyTensor {
            tensor: Arc::new(summed),
            requires_grad: true,
            is_pinned: false,
        };
        record_and_link_unary(
            UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &sum_result,
            &scalar,
        )
        .expect("recording sum must succeed");

        run_backward(&scalar).expect("backward must succeed");
        x
    }

    #[test]
    fn add_gradient_matches_expected() {
        reset_for_test();
        let x = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
        let y = make_tensor(vec![10.0, 20.0, 30.0], &[3]);
        let x = add_then_sum_backward(x, &y);

        let grad = get_grad(&x).expect("grad must be available");
        let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
        // d(sum(x + y))/dx = 1 for every element.
        assert_eq!(grad_data, vec![1.0, 1.0, 1.0]);
    }

    /// Regression test for the exact bug this design hit during
    /// development: two *independent* forward passes on the same thread
    /// (e.g. two separate Python test functions, each building its own
    /// small graph and calling `.backward()` once) must not leak tape state
    /// into each other. Before `run_backward` reset the tape/registry/leaves
    /// after each successful backward pass, the second computation's
    /// backward walked *all* nodes ever recorded (including the first
    /// computation's, with incompatible shapes) and differentiated against
    /// every leaf ever registered, corrupting the result.
    #[test]
    fn independent_backward_passes_do_not_leak_state() {
        reset_for_test();

        // First, unrelated computation: shape [3].
        let x1 = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
        let y1 = make_tensor(vec![10.0, 20.0, 30.0], &[3]);
        let x1 = add_then_sum_backward(x1, &y1);

        // The tape must be fully reset after the first backward pass.
        let leaf_count_after_first = LEAVES.with(|l| l.borrow().len());
        assert_eq!(
            leaf_count_after_first, 0,
            "leaves must be cleared after a successful backward pass"
        );
        let tape_len_after_first = IMPLICIT_TAPE.with(GradientTape::len);
        assert_eq!(
            tape_len_after_first, 0,
            "tape nodes must be cleared after a successful backward pass"
        );

        // Second, unrelated computation with a *different* shape: [2, 2].
        // Before the fix, this would either error out (shape mismatch while
        // re-processing the first computation's stale nodes) or silently
        // produce a wrong gradient (summed against unrelated stale leaves).
        let x2 = make_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let y2 = make_tensor(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let x2 = add_then_sum_backward(x2, &y2);

        let grad2 = get_grad(&x2).expect("grad must be available for the second computation");
        let grad2_data = grad2.tensor.to_vec().expect("grad data must be readable");
        assert_eq!(
            grad2_data,
            vec![1.0, 1.0, 1.0, 1.0],
            "second computation's gradient must be correct and unaffected by the first"
        );

        // The first computation's grad must still be readable (GRAD_STORE is
        // never cleared by run_backward, only the tape/registry/leaves are).
        let grad1 = get_grad(&x1).expect("first computation's grad must still be readable");
        let grad1_data = grad1.tensor.to_vec().expect("grad data must be readable");
        assert_eq!(grad1_data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn non_tracked_leaf_does_not_accumulate_gradient() {
        reset_for_test();
        let x = make_tensor(vec![1.0, 2.0], &[2]);
        let y = make_tensor(vec![3.0, 4.0], &[2]);
        // Neither tensor is marked as requiring grad.
        let raw = tenflowers_core::ops::add(&x.tensor, &y.tensor).expect("add must succeed");
        let result = PyTensor {
            tensor: Arc::new(raw),
            requires_grad: false,
            is_pinned: false,
        };
        record_and_link_binary(BinaryOpKind::Add, &x, &y, &result)
            .expect("recording must be a no-op success");

        assert!(
            lookup_tracked(&result).is_none(),
            "result of an untracked op must not become tracked"
        );
        assert!(get_grad(&x).is_err(), "x never had requires_grad=True");
    }

    #[test]
    fn backward_on_untracked_tensor_errors_clearly() {
        // `PyErr::to_string()` needs an initialized Python interpreter to
        // format the underlying Python exception object; see
        // `crate::test_module` for the same idiom used elsewhere in this
        // crate's Rust-side test suite.
        pyo3::Python::initialize();
        reset_for_test();
        let x = make_tensor(vec![1.0], &[1]);
        let err = run_backward(&x).expect_err("backward on untracked tensor must error");
        let message = err.to_string();
        assert!(
            message.contains("no recorded computation graph"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn grad_on_never_tracked_tensor_errors_clearly() {
        pyo3::Python::initialize();
        reset_for_test();
        let x = make_tensor(vec![1.0], &[1]);
        let err = get_grad(&x).expect_err("grad on a never-tracked tensor must error");
        let message = err.to_string();
        assert!(
            message.contains("never had requires_grad"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn tensor_key_matches_for_clones_and_differs_for_new_tensors() {
        let x = make_tensor(vec![1.0], &[1]);
        let x_clone = x.clone();
        assert_eq!(
            tensor_key(&x),
            tensor_key(&x_clone),
            "cloning a PyTensor must preserve identity (shared Arc)"
        );

        let y = make_tensor(vec![1.0], &[1]);
        assert_ne!(
            tensor_key(&x),
            tensor_key(&y),
            "two independently constructed tensors must have distinct identities"
        );
    }

    /// Regression test for a real bug caught during development: `tensor_key`
    /// is an `Arc` address, which the allocator is free to reuse once every
    /// strong reference to the original `Arc` is dropped. Before
    /// `IDENTITY_ANCHORS` existed, running many independent `backward()`
    /// computations (each of which drops its tensors afterward) could free
    /// and reallocate addresses such that a brand-new, never-tracked tensor
    /// coincidentally landed at an address a *different*, already-completed
    /// computation's leaf used to occupy — and `get_grad` would then
    /// spuriously return that unrelated stale gradient instead of erroring.
    #[test]
    fn address_reuse_does_not_leak_stale_gradient() {
        reset_for_test();

        // Run many independent tiny computations and let every tensor they
        // touch be dropped immediately afterward, to encourage the
        // allocator to reuse freed addresses for what comes next.
        for i in 0..64 {
            let x = make_tensor(vec![i as f32, (i + 1) as f32], &[2]);
            let y = make_tensor(vec![10.0, 20.0], &[2]);
            let _ = add_then_sum_backward(x, &y);
            // x, y (and every intermediate PyTensor created inside
            // add_then_sum_backward) are dropped here at the end of the
            // loop body, freeing their Arc<Tensor<f32>> allocations unless
            // something (correctly) still anchors them.
        }

        // A brand-new tensor that has never participated in implicit
        // autograd at all must not be considered tracked, no matter what
        // address it happens to occupy.
        let fresh = make_tensor(vec![99.0, 100.0], &[2]);
        assert!(
            lookup_tracked(&fresh).is_none(),
            "a fresh, never-tracked tensor must not appear tracked even if its \
             address was previously used by a completed computation's tensor"
        );
        assert!(
            get_grad(&fresh).is_err(),
            "a fresh, never-tracked tensor must not spuriously return a stale gradient"
        );
    }
}
