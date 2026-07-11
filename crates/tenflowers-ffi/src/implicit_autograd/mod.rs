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
//! ### Explicit-key parameter identities need no anchoring
//!
//! [`crate::neural::layers::PyParameter`] introduces a *second* kind of
//! identity into these same side-tables: instead of a [`tensor_key`] derived
//! from a `PyTensor`'s own `Arc<Tensor<f32>>` address, a parameter's `id` is
//! the address of the `Arc<RwLock<Tensor<f32>>>` allocation backing its
//! *value cell* — an address that a training loop deliberately keeps calling
//! back into the *same* leaf under, forward pass after forward pass, even as
//! the cell's contents are replaced in place by an optimizer step (see that
//! struct's doc for why a plain `Arc<Tensor<f32>>` cannot support this).
//! [`mark_leaf_param`] and [`get_grad_by_id`] operate on this kind of key
//! directly, alongside [`clear_grad_by_id`] for clearing it.
//!
//! Crucially, this explicit-key path deliberately does **not** populate
//! [`IDENTITY_ANCHORS`] the way the [`tensor_key`]-derived path does (see
//! [`register_tracked_by_key`]). The address-reuse hazard above exists
//! because a `tensor_key` address becomes unsafe to reuse as a table key the
//! moment its originating `Arc<Tensor<f32>>` is dropped and something else
//! might get allocated at the same address — so this module has to keep that
//! `Arc` alive itself. A `PyParameter`'s `id` has no such problem *while the
//! parameter is alive*: its `Arc<RwLock<Tensor<f32>>>` allocation is kept
//! alive directly by the `PyParameter` Python object's own lifetime for as
//! long as it exists, completely independent of anything this module
//! tracks. There is simply nothing for `IDENTITY_ANCHORS` to usefully do for
//! this kind of key while its owner is alive — anchoring it would only pin a
//! redundant clone of an allocation that was never at risk of being freed
//! out from under its key.
//!
//! That "while alive" qualifier matters: unlike [`tensor_key`] entries
//! (whose lifetime *in these tables* is governed entirely by this module,
//! via `IDENTITY_ANCHORS`), a `PyParameter`'s `id` genuinely does stop being
//! safe once the parameter itself is dropped — the `RwLock` allocation truly
//! is freed at that point, and the allocator can legitimately hand that same
//! address to a later, unrelated `PyParameter`. Two independent tables can
//! each go stale this way, for two different reasons:
//!
//! * [`GRAD_STORE`] is deliberately never cleared by [`run_backward`] (so
//!   `.grad()` survives after the graph is freed, mirroring PyTorch), so a
//!   dropped parameter's now-stale `GRAD_STORE` entry could otherwise be
//!   spuriously "inherited" by a later, address-colliding parameter.
//! * [`TRACKED_REGISTRY`]/[`LEAVES`] are only cleared by a successful
//!   [`run_backward`] call — which a legitimate forward-only computation
//!   (one whose result is simply never backpropagated through) never
//!   triggers. A parameter [`mark_leaf_param`]'d and then dropped without an
//!   intervening `run_backward` leaves its `id` behind in both tables. Worse
//!   than `GRAD_STORE`'s hazard, this one is not merely "spurious data read
//!   back" but a silent tracking failure: [`mark_leaf_param`]'s own
//!   idempotency check (`TRACKED_REGISTRY.contains_key(&explicit_key)`)
//!   cannot distinguish that stale entry from "this exact parameter was
//!   already marked earlier this cycle", so a later, address-colliding
//!   parameter's very first `mark_leaf_param` call takes the early-return
//!   branch and is silently never watched on the tape at all — confirmed
//!   reproducible via
//!   [`tests::dropped_untracked_backward_param_does_not_poison_tracked_registry_on_address_reuse`].
//!
//! Rather than anchoring (which would require keeping the very allocation
//! alive that `Drop` is trying to free — a contradiction), `PyParameter`'s
//! `Drop` impl (in `layers.rs`) proactively calls [`unregister_param_id`] on
//! its own `id` as the parameter goes away — scrubbing `TRACKED_REGISTRY`,
//! `LEAVES`, *and* `GRAD_STORE` together — closing all three holes at the
//! moment they actually open rather than preventing them from ever opening.
//! This is sound because a `PyParameter`'s `id` is exclusively owned —
//! `PyParameter` does not implement `Clone`, so exactly one live object ever
//! owns a given `id`, and its `Drop` can never race with a different,
//! still-alive parameter that shares that `id`.
//!
//! ([`PyParameter::zero_grad`](crate::neural::layers::PyParameter::zero_grad)
//! — an explicit, user-requested clear of just the accumulated gradient in
//! the *middle* of a still-alive parameter's lifetime — deliberately keeps
//! calling only [`clear_grad_by_id`], not [`unregister_param_id`]: it must
//! not un-track a parameter that is still legitimately in use.)
//!
//! ## Components
//!
//! * [`IMPLICIT_TAPE`] — one [`GradientTape`] per thread, created lazily and
//!   reused for the life of the thread. It auto-activates: nothing must be
//!   explicitly "started", it simply exists and is always recording.
//! * [`TRACKED_REGISTRY`] — maps a tensor's stable identity to the
//!   [`TrackedTensor<f32>`] that represents it on the implicit tape. Populated
//!   by [`mark_leaf`] / [`mark_leaf_param`] (for tensors and parameters with
//!   `requires_grad=True`) and by [`record_and_link_binary`] /
//!   [`record_and_link_unary`] (for the outputs of tracked operations).
//! * [`LEAVES`] — the set of tensors (and parameter snapshots) that became
//!   tracked *directly* via `set_requires_grad(True)` or
//!   [`mark_leaf_param`] (as opposed to becoming tracked because an input was
//!   tracked). These are exactly the tensors `.backward()` differentiates
//!   with respect to, mirroring PyTorch's notion of a "leaf".
//! * [`GRAD_STORE`] — the `.grad` storage itself: after `.backward()` runs,
//!   each leaf's computed gradient is stored here keyed by the leaf's stable
//!   identity, so `PyTensor::grad()` / [`get_grad_by_id`] can look it up
//!   later.
//! * [`IDENTITY_ANCHORS`] — see above; keeps a [`tensor_key`]'s
//!   `Arc<Tensor<f32>>` allocation alive (and therefore its address
//!   un-reusable) for exactly as long as any of the three tables above still
//!   references that key. Not populated for [`crate::neural::layers::PyParameter`]
//!   explicit-key entries — see "Explicit-key parameter identities" above.
//!
//! ## Generic operation hook
//!
//! [`record_and_link_binary`] and [`record_and_link_unary`] are dispatched on
//! a small [`BinaryOpKind`] / [`UnaryOpKind`] tag rather than being one
//! function per named op. Any binary op that has a `TrackedTensor` method
//! (`add`, `sub`, `mul`, `div`, `pow`, `matmul`, ...) can be wired in by
//! adding one match arm and one call site — nothing about the hook itself is
//! specific to `add`/`mul`/`matmul` by name.
//!
//! Three more hooks follow the same "tag enum + dispatch" shape for
//! operations that do not fit the binary/unary arity:
//!
//! * [`record_and_link_ternary`] — up to three *differentiable* operands
//!   (input, weight/gamma, optional bias/beta), for `TrackedTensor` methods
//!   like `conv1d`/`conv2d`/`conv3d`/`layer_norm`/`group_norm`/
//!   `instance_norm`. [`TernaryOpKind::BatchNorm`] is a partial exception:
//!   `TrackedTensor::batch_norm` needs *five* tracked-tensor operands, but
//!   two of them (`running_mean`/`running_var`) are forward-only statistics
//!   that are read for their values but never differentiated through (see
//!   that variant's own doc) — structurally the same "must be present on the
//!   tape, must never be a leaf" treatment [`record_and_link_binary`] already
//!   gives a constant operand, just applied to two extra fixed-role tensors
//!   instead of `lhs`/`rhs` themselves.
//! * [`record_and_link_variadic`] — an arbitrary number of *equal-role*
//!   operands, for `TrackedTensor::concat`/`TrackedTensor::stack`, which are
//!   associated functions taking a `&[&TrackedTensor<T>]` slice rather than
//!   distinguishing e.g. a "primary" operand from the rest.
//! * [`record_and_link_gather`] — `TrackedTensor::gather`'s `indices`
//!   argument is a raw `Tensor<i32>`, never a [`PyTensor`]/`TrackedTensor` at
//!   all, so it cannot be tracked or become a leaf; only `input`'s tracked
//!   status matters, which does not fit any of the tag-enum hooks above.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

use tenflowers_autograd::{GradientTape, Operation, TrackedTensor};
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

/// Register `tracked` under `key` directly in [`TRACKED_REGISTRY`], with
/// **no** [`IDENTITY_ANCHORS`] involvement.
///
/// This is the low-level primitive both [`register_tracked`] (the
/// `PyTensor`/[`tensor_key`]-derived path) and [`mark_leaf_param`] (the
/// explicit-key, [`crate::neural::layers::PyParameter`] path) build on. The
/// two paths need different identity-safety treatment:
///
/// * A [`tensor_key`]-derived key is only safe to keep using as long as
///   *something* keeps the originating `Arc<Tensor<f32>>` allocation alive —
///   that is exactly what [`IDENTITY_ANCHORS`] provides (see the module-level
///   "address-reuse hazard" doc), so the [`tensor_key`] path anchors on every
///   insert.
/// * A [`PyParameter`]'s explicit `id` needs no such anchoring: it is the
///   address of an `Arc<RwLock<Tensor<f32>>>` allocation that the
///   `PyParameter` Python object itself keeps alive for as long as it exists,
///   completely independent of anything stored in this module. There is
///   nothing meaningful to anchor for that key — the caller already
///   guarantees its stability. See
///   [`crate::neural::layers::PyParameter`]'s struct-level doc and the
///   "Explicit-key parameter identities" section of this module's doc for
///   the full reasoning.
///
/// [`PyParameter`]: crate::neural::layers::PyParameter
fn register_tracked_by_key(key: usize, tracked: Arc<TrackedTensor<f32>>) {
    TRACKED_REGISTRY.with(|registry| {
        registry.borrow_mut().insert(key, tracked);
    });
}

/// Register `tracked` as the implicit-tape counterpart of `tensor`.
///
/// Also anchors `tensor`'s identity (see [`anchor_identity`]) so the key
/// just inserted into [`TRACKED_REGISTRY`] cannot be invalidated by the
/// tensor's `Arc` being freed and its address reused for something else
/// while it is still registered. Thin wrapper around
/// [`register_tracked_by_key`] that additionally computes the key from
/// `tensor` and performs that anchoring — see that function's doc for why
/// the explicit-key path does not do the same.
fn register_tracked(tensor: &PyTensor, tracked: Arc<TrackedTensor<f32>>) {
    let key = tensor_key(tensor);
    anchor_identity(tensor);
    register_tracked_by_key(key, tracked);
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

/// Mark a [`PyParameter`]'s current-value snapshot as a leaf that requires
/// gradients, keyed by the parameter's own stable `explicit_key` (its `id`)
/// rather than by [`tensor_key`] on `transient` itself.
///
/// [`PyParameter`]: crate::neural::layers::PyParameter
///
/// # Why this cannot just call [`mark_leaf`]
///
/// A real training loop calls a layer's `forward()` — and therefore this
/// function — on *every* step. Each step's `PyParameter::to_tensor()`
/// snapshot is a **different** `PyTensor`/`Arc<Tensor<f32>>` allocation even
/// though it represents "the same" parameter, because
/// [`crate::neural::layers::PyParameter::to_tensor`] always allocates a
/// fresh `Arc`. [`mark_leaf`]'s idempotency check
/// (`lookup_tracked(tensor)`, itself keyed by `tensor_key(tensor)`) is
/// appropriate for its own callers because the *same* `PyTensor`/`Arc` is
/// what gets re-marked there — but it would be actively wrong here: keyed by
/// `tensor_key(transient)`, the check would see a fresh, never-before-seen
/// address on *every single call* and would never actually be idempotent,
/// re-watching a "new leaf" on the tape every forward pass instead of
/// recognizing it as the same parameter.
///
/// This function sidesteps that entirely by taking `explicit_key` as a
/// caller-supplied parameter (the `PyParameter`'s own `id`, stable across
/// `set_data` calls — see that struct's doc) and checking idempotency
/// against *that* key directly: if `explicit_key` is already present in
/// [`TRACKED_REGISTRY`], this is a no-op, exactly mirroring [`mark_leaf`]'s
/// idempotency guarantee but correctly scoped to the parameter's stable
/// identity instead of the transient snapshot's incidental one.
///
/// # Registered under two keys, on purpose
///
/// The resulting `TrackedTensor` is inserted into [`TRACKED_REGISTRY`] under
/// **both** `explicit_key` and `tensor_key(transient)`, pointing at the same
/// `Arc<TrackedTensor<f32>>` — but pushed onto [`LEAVES`] (and therefore
/// later [`GRAD_STORE`]) only **once**, under `explicit_key`. This is not
/// redundant, and both halves are load-bearing:
///
/// * [`record_and_link_binary`] / [`record_and_link_unary`] — which this
///   function must not require any change to (see this module's "Generic
///   operation hook" doc) — look up an operand purely via [`lookup_tracked`],
///   which is hard-wired to key on `tensor_key(operand)`. When `transient`
///   (this exact `PyTensor` snapshot) is subsequently passed as an operand to
///   one of those functions — which is exactly what happens next in a real
///   `forward()`, e.g. `weight_snapshot.matmul(&input)` — the lookup has no
///   way to know about `explicit_key` at all; it can only ever find the
///   entry via `tensor_key(transient)`. Without this second registration,
///   every operation performed directly on `transient` would silently be
///   treated as untracked.
/// * `explicit_key` is what must end up owning the [`LEAVES`] entry (and,
///   after a successful [`run_backward`], the [`GRAD_STORE`] entry), since
///   that is the stable identity [`get_grad_by_id`] /
///   [`crate::neural::layers::PyParameter::grad`] look the gradient up by
///   later — `tensor_key(transient)` is a fresh, throwaway address specific
///   to this one snapshot and must not be what the gradient survives under.
///
/// The `tensor_key(transient)` half is registered via [`register_tracked`]
/// (so it *is* anchored, exactly like any other `tensor_key`-derived entry —
/// see the address-reuse hazard this guards against in this module's
/// top-level doc: nothing but `TRACKED_REGISTRY` itself is guaranteed to
/// keep `transient`'s `Arc<Tensor<f32>>` alive for the rest of the forward
/// pass, so this entry needs the same protection any other tensor_key entry
/// gets). The `explicit_key` half is registered via
/// [`register_tracked_by_key`] directly, with **no** anchoring — see that
/// function's doc for why a parameter `id` needs none: its `RwLock`
/// allocation is kept alive by the owning
/// [`crate::neural::layers::PyParameter`] object itself, not by anything in
/// this module.
///
/// Both `TRACKED_REGISTRY` entries need no special cleanup beyond what
/// already exists: [`run_backward`] unconditionally clears the *entire*
/// registry after a successful backward pass, so both disappear together
/// along with every other non-leaf intermediate entry. The anchor for the
/// `tensor_key(transient)` half is likewise pruned exactly as any other
/// tensor_key anchor is — by `run_backward`'s existing retain-if-in-
/// GRAD_STORE pass, which correctly does not retain it (only `explicit_key`
/// ends up in `GRAD_STORE`, never `tensor_key(transient)`).
///
/// # Errors
///
/// This function cannot currently fail; it returns `()` like [`mark_leaf`].
pub fn mark_leaf_param(transient: &PyTensor, explicit_key: usize) {
    let already_tracked =
        TRACKED_REGISTRY.with(|registry| registry.borrow().contains_key(&explicit_key));
    if already_tracked {
        // Already tracked under this parameter's id — either from an earlier
        // call within the same forward/backward cycle, or (more commonly in
        // practice) simply not yet cleared by a `run_backward` call. Nothing
        // further to do; in particular, do NOT re-watch `transient` on the
        // tape, which would silently start tracking two different
        // `TrackedTensor`s under variations of "the same" parameter.
        return;
    }

    let tracked = IMPLICIT_TAPE.with(|tape| tape.watch((*transient.tensor).clone()));
    let tracked = Arc::new(tracked);

    // explicit_key half: deliberately NOT `register_tracked` (the
    // tensor_key + anchoring path) — see `register_tracked_by_key`'s doc for
    // why an explicit parameter `id` needs no `IDENTITY_ANCHORS` entry.
    register_tracked_by_key(explicit_key, Arc::clone(&tracked));
    // tensor_key(transient) half: DOES need anchoring like any other
    // tensor_key entry (see the doc section above), so this uses
    // `register_tracked` (not `register_tracked_by_key`) specifically to get
    // that anchoring for free.
    register_tracked(transient, Arc::clone(&tracked));
    LEAVES.with(|leaves| leaves.borrow_mut().push((explicit_key, tracked)));
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
///
/// # Deliberately NOT included here: `ScalarMul`
///
/// `ScalarMul` has no `TrackedTensor` recording method — confirmed by reading
/// `tenflowers_autograd::tape::tracked_tensor` in full — and no `Operation`
/// variant for it exists either, so there is nothing a `TrackedTensor` method
/// could even construct. Per this module's design (see the top-level
/// "Generic operation hook" doc), this hook only ever dispatches to an
/// *existing*, tested `TrackedTensor` method — it introduces no gradient math
/// of its own. Adding a variant here with no such method to call would either
/// fail to compile (nothing to dispatch to) or, worse, tempt inlining ad-hoc
/// gradient-free math that silently drops the tape edge — exactly the failure
/// mode this whole design exists to eliminate. Wiring `ScalarMul` up is
/// scoped to a follow-up task against `tenflowers-autograd` (add the
/// `Operation` variant, its backward implementation, and the `TrackedTensor`
/// recording method), not this file.
///
/// (Every other op that used to be listed here — `Log`, `Abs`, `Clamp`,
/// `Gelu`, `Swish`, `Mish`, `LeakyRelu`, `Elu`, `Relu6`, `HardSwish`, and
/// `LogSoftmax` — now has a real `TrackedTensor` recording method and is
/// wired in below like any other variant; see
/// [`record_and_link_unary`]'s match arms.)
///
/// # Why this no longer derives `PartialEq`/`Eq`
///
/// The original (pre-existing) variant set derived `PartialEq, Eq`, but
/// [`Slice`](UnaryOpKind::Slice)'s payload is a
/// `Vec<tenflowers_autograd::grad_ops::SliceSpec>`, and `SliceSpec` (defined
/// in `tenflowers-autograd`, not here) implements neither trait. No call site
/// in this crate ever compared two `UnaryOpKind` values for equality
/// (confirmed by search), so dropping both derives here is the correct fix
/// rather than working around it (e.g. by hand-rolling a partial
/// `PartialEq` that panics or always-returns-false on `Slice`, which would
/// be actively misleading).
#[derive(Debug, Clone)]
pub enum UnaryOpKind {
    Relu,
    Sigmoid,
    Tanh,
    /// `gelu()` — Gaussian Error Linear Unit, matching `TrackedTensor::gelu`.
    Gelu,
    /// `swish()` — `x * sigmoid(x)`, matching `TrackedTensor::swish`.
    Swish,
    /// `mish()` — `x * tanh(softplus(x))`, matching `TrackedTensor::mish`.
    Mish,
    /// `leaky_relu(alpha)`, matching `TrackedTensor::leaky_relu`'s own `f32`
    /// slope parameter (applied to negative inputs).
    LeakyRelu { negative_slope: f32 },
    /// `elu(alpha)`, matching `TrackedTensor::elu`'s own `f32` scale
    /// parameter for the negative-input exponential branch.
    Elu { alpha: f32 },
    /// `relu6()` — `min(max(x, 0), 6)`, matching `TrackedTensor::relu6`.
    Relu6,
    /// `hard_swish()` — `x * relu6(x + 3) / 6`, matching
    /// `TrackedTensor::hard_swish`.
    HardSwish,
    /// `softmax(axis)`, matching `TrackedTensor::softmax`'s own `Option<i32>`
    /// axis (`None` means "softmax over the flattened tensor" — see that
    /// method's forward kernel).
    Softmax { axis: Option<i32> },
    /// `log_softmax(axis)`, matching `TrackedTensor::log_softmax`'s own
    /// `Option<i32>` axis (same convention as `Softmax` above).
    LogSoftmax { axis: Option<i32> },
    /// `transpose(axes)`, matching `TrackedTensor::transpose`'s own
    /// `Option<Vec<usize>>` axes (`None` means "reverse all axes" — see that
    /// method's forward kernel).
    Transpose { axes: Option<Vec<usize>> },
    /// `reshape(shape)`.
    Reshape { shape: Vec<usize> },
    /// `slice(specs)`, one [`tenflowers_autograd::grad_ops::SliceSpec`] per
    /// dimension (trailing dimensions without an explicit spec are taken in
    /// full — see `TrackedTensor::slice`'s own doc).
    Slice {
        specs: Vec<tenflowers_autograd::grad_ops::SliceSpec>,
    },
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
    /// `log()` — natural logarithm, matching `TrackedTensor::log`.
    Log,
    /// `abs()` — element-wise absolute value, matching `TrackedTensor::abs`.
    Abs,
    /// `clamp(min, max)`, matching `TrackedTensor::clamp`'s own
    /// `Option<f32>` bounds (either side may be `None` to leave that side
    /// unconstrained).
    Clamp {
        min: Option<f32>,
        max: Option<f32>,
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
        UnaryOpKind::Gelu => input_tt.gelu(),
        UnaryOpKind::Swish => input_tt.swish(),
        UnaryOpKind::Mish => input_tt.mish(),
        UnaryOpKind::LeakyRelu { negative_slope } => input_tt.leaky_relu(negative_slope),
        UnaryOpKind::Elu { alpha } => input_tt.elu(alpha),
        UnaryOpKind::Relu6 => input_tt.relu6(),
        UnaryOpKind::HardSwish => input_tt.hard_swish(),
        UnaryOpKind::Softmax { axis } => input_tt.softmax(axis),
        UnaryOpKind::LogSoftmax { axis } => input_tt.log_softmax(axis),
        UnaryOpKind::Transpose { axes } => input_tt.transpose(axes),
        UnaryOpKind::Reshape { shape } => input_tt.reshape(&shape),
        UnaryOpKind::Slice { specs } => input_tt.slice(&specs),
        UnaryOpKind::Sum { axes, keepdims } => input_tt.sum(axes, keepdims),
        UnaryOpKind::Mean { axes, keepdims } => input_tt.mean(axes, keepdims),
        UnaryOpKind::Log => input_tt.log(),
        UnaryOpKind::Abs => input_tt.abs(),
        UnaryOpKind::Clamp { min, max } => input_tt.clamp(min, max),
    }
    .map_err(|e| PyRuntimeError::new_err(format!("implicit autograd recording failed: {e}")))?;

    register_tracked(result, Arc::new(recorded));
    Ok(())
}

/// Watch `tensor` on the shared implicit tape as a non-leaf "constant" and
/// return the resulting [`TrackedTensor`], reusing an already-tracked operand
/// unchanged.
///
/// This is the exact "watch a constant operand so it gets a real, tape-issued
/// id without ever becoming a leaf" idiom [`record_and_link_binary`] already
/// uses inline for `lhs`/`rhs` (see that function's doc for *why* this must
/// go through [`GradientTape::watch`] rather than `TrackedTensor::new`: a
/// `TrackedTensor::new`-synthesized id of `0` would collide with a real
/// leaf's tape-issued id). Factored out here so
/// [`record_and_link_ternary`] and [`record_and_link_variadic`] — which both
/// need to do this for a variable/larger number of operands than
/// `record_and_link_binary`'s fixed two — do not each re-derive it.
fn tracked_or_watch_as_constant(
    tracked: Option<Arc<TrackedTensor<f32>>>,
    tensor: &PyTensor,
) -> Arc<TrackedTensor<f32>> {
    tracked
        .unwrap_or_else(|| Arc::new(IMPLICIT_TAPE.with(|tape| tape.watch((*tensor.tensor).clone()))))
}

/// The ternary (up to three *differentiable* operands) tensor operations
/// wired into the implicit-tracking hook: convolutions (input, weight,
/// optional bias) and the normalization layers with a learnable affine
/// transform (input, gamma, optional beta).
///
/// [`BatchNorm`](TernaryOpKind::BatchNorm) additionally needs two
/// *non*-differentiable, always-required operands (`running_mean`,
/// `running_var`) that do not fit the `a`/`b`/`c` shape — see
/// [`record_and_link_ternary`]'s own doc for how those are threaded through
/// instead.
#[derive(Debug, Clone, PartialEq)]
pub enum TernaryOpKind {
    /// `conv1d(weight, bias, stride, padding)`.
    Conv1D { stride: usize, padding: String },
    /// `conv2d(weight, bias, stride, padding)`.
    Conv2D {
        stride: (usize, usize),
        padding: String,
    },
    /// `conv3d(weight, bias, stride, padding)`.
    Conv3D {
        stride: (usize, usize, usize),
        padding: String,
    },
    /// `layer_norm(gamma, beta, normalized_shape, epsilon)`. `beta` (`c`) is
    /// required, not optional, matching `TrackedTensor::layer_norm`'s own
    /// signature (unlike conv's bias, LayerNorm has no "no beta" forward
    /// path in this crate).
    LayerNorm {
        epsilon: f32,
        normalized_shape: Vec<usize>,
    },
    /// `group_norm(gamma, beta, num_groups, epsilon)`. `beta` (`c`) is
    /// required, matching `TrackedTensor::group_norm`'s own signature.
    GroupNorm { num_groups: usize, epsilon: f32 },
    /// `instance_norm(gamma, beta, epsilon)`. `beta` (`c`) is required,
    /// matching `TrackedTensor::instance_norm`'s own signature.
    InstanceNorm { epsilon: f32 },
    /// `batch_norm(gamma, beta, running_mean, running_var, epsilon,
    /// training)`. `beta` (`c`) is required. `running_mean`/`running_var`
    /// are threaded through [`record_and_link_ternary`]'s separate
    /// `running_stats` parameter rather than `a`/`b`/`c` — see that
    /// function's doc for why (they are forward-only statistics that must be
    /// present on the tape but must never be leaves, structurally unlike
    /// every other operand this hook handles).
    BatchNorm { epsilon: f32, training: bool },
}

/// Ternary counterpart of [`record_and_link_binary`]/[`record_and_link_unary`]:
/// record `op(a, b, c?) = result` onto the implicit tape if at least one of
/// `a`/`b`/`c` is currently tracked. `a` is always the primary input; `b` is
/// the weight/gamma; `c` is the optional bias/beta (required — i.e. always
/// `Some` — for every [`TernaryOpKind`] except the conv variants, which
/// mirror `TrackedTensor::conv1d`/`conv2d`/`conv3d`'s own `Option<&..>` bias).
///
/// `running_stats` carries [`TernaryOpKind::BatchNorm`]'s `running_mean`/
/// `running_var` (in that order); every other `kind` ignores it and it may be
/// `None`. Both tensors are always watched on the tape the same way a
/// [`record_and_link_binary`] constant operand is (see
/// [`tracked_or_watch_as_constant`]) — regardless of whether they are
/// individually tracked — because `TrackedTensor::batch_norm` needs a real id
/// for both no matter what, but neither one is ever pushed onto [`LEAVES`]:
/// per `Operation::BatchNorm`'s backward (`process_batchnorm_backward`), they
/// are forward-only statistics that are read for their values but never
/// differentiated through.
///
/// # Errors
///
/// * Returns an error if `kind` is [`TernaryOpKind::BatchNorm`] and
///   `running_stats` is `None` — this indicates a caller bug (every BatchNorm
///   forward call has running statistics, even if only freshly-initialized
///   ones), not a legitimately absent operand, so this is reported rather
///   than silently treated as a no-op.
/// * Otherwise, returns an error only if at least one operand is tracked but
///   recording the operation on the tape itself fails (e.g. a poisoned
///   lock) — never because gradients are unsupported for `kind`, since only
///   ops with a real `TrackedTensor` backward implementation are wired in
///   here.
pub fn record_and_link_ternary(
    kind: TernaryOpKind,
    a: &PyTensor,
    b: &PyTensor,
    c: Option<&PyTensor>,
    running_stats: Option<(&PyTensor, &PyTensor)>,
    result: &PyTensor,
) -> PyResult<()> {
    let a_tracked = lookup_tracked(a);
    let b_tracked = lookup_tracked(b);
    let c_tracked = c.and_then(lookup_tracked);

    if a_tracked.is_none() && b_tracked.is_none() && c_tracked.is_none() {
        return Ok(());
    }

    let a_tt = tracked_or_watch_as_constant(a_tracked, a);
    let b_tt = tracked_or_watch_as_constant(b_tracked, b);
    let c_tt = c.map(|c_tensor| tracked_or_watch_as_constant(c_tracked, c_tensor));

    let recorded = match kind {
        TernaryOpKind::Conv1D { stride, padding } => {
            a_tt.conv1d(&b_tt, c_tt.as_deref(), stride, &padding)
        }
        TernaryOpKind::Conv2D { stride, padding } => {
            a_tt.conv2d(&b_tt, c_tt.as_deref(), stride, &padding)
        }
        TernaryOpKind::Conv3D { stride, padding } => {
            a_tt.conv3d(&b_tt, c_tt.as_deref(), stride, &padding)
        }
        TernaryOpKind::LayerNorm {
            epsilon,
            normalized_shape,
        } => {
            let Some(beta_tt) = c_tt else {
                return Err(PyRuntimeError::new_err(
                    "implicit autograd recording failed: LayerNorm requires beta (c)",
                ));
            };
            a_tt.layer_norm(&b_tt, &beta_tt, normalized_shape, epsilon)
        }
        TernaryOpKind::GroupNorm {
            num_groups,
            epsilon,
        } => {
            let Some(beta_tt) = c_tt else {
                return Err(PyRuntimeError::new_err(
                    "implicit autograd recording failed: GroupNorm requires beta (c)",
                ));
            };
            a_tt.group_norm(&b_tt, &beta_tt, num_groups, epsilon)
        }
        TernaryOpKind::InstanceNorm { epsilon } => {
            let Some(beta_tt) = c_tt else {
                return Err(PyRuntimeError::new_err(
                    "implicit autograd recording failed: InstanceNorm requires beta (c)",
                ));
            };
            a_tt.instance_norm(&b_tt, &beta_tt, epsilon)
        }
        TernaryOpKind::BatchNorm { epsilon, training } => {
            let Some(beta_tt) = c_tt else {
                return Err(PyRuntimeError::new_err(
                    "implicit autograd recording failed: BatchNorm requires beta (c)",
                ));
            };
            let Some((running_mean, running_var)) = running_stats else {
                return Err(PyRuntimeError::new_err(
                    "implicit autograd recording failed: BatchNorm requires running_stats",
                ));
            };
            let running_mean_tt =
                tracked_or_watch_as_constant(lookup_tracked(running_mean), running_mean);
            let running_var_tt =
                tracked_or_watch_as_constant(lookup_tracked(running_var), running_var);
            a_tt.batch_norm(
                &b_tt,
                &beta_tt,
                &running_mean_tt,
                &running_var_tt,
                epsilon,
                training,
            )
        }
    }
    .map_err(|e| PyRuntimeError::new_err(format!("implicit autograd recording failed: {e}")))?;

    register_tracked(result, Arc::new(recorded));
    Ok(())
}

/// The variadic (arbitrary number of *equal-role* operands) tensor
/// operations wired into the implicit-tracking hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariadicOpKind {
    /// `TrackedTensor::concat(inputs, axis)`.
    Concat { axis: i32 },
    /// `TrackedTensor::stack(inputs, axis)`.
    Stack { axis: i32 },
}

/// Variadic counterpart of [`record_and_link_binary`]: record `op(inputs) =
/// result` onto the implicit tape if at least one element of `inputs` is
/// currently tracked.
///
/// Generalizes [`record_and_link_binary`]'s "watch an untracked operand as a
/// tape constant" logic (via [`tracked_or_watch_as_constant`]) from exactly
/// two operands to however many `inputs` contains: every element is looked
/// up, and every element that is not already tracked is watched as a
/// constant, exactly as `record_and_link_binary` does for a single untracked
/// `lhs`/`rhs` — see that function's doc for why this must go through
/// [`GradientTape::watch`] rather than `TrackedTensor::new`.
///
/// # Errors
///
/// * Returns an error (without recording anything) if `inputs` is empty —
///   there is no operation to record, and both `TrackedTensor::concat`/
///   `stack` already reject an empty slice themselves, so this fails fast
///   with the same intent before doing any tape-watching work.
/// * Otherwise, returns an error only if at least one input is tracked but
///   recording the operation on the tape itself fails.
pub fn record_and_link_variadic(
    kind: VariadicOpKind,
    inputs: &[&PyTensor],
    result: &PyTensor,
) -> PyResult<()> {
    if inputs.is_empty() {
        return Err(PyRuntimeError::new_err(
            "implicit autograd recording failed: at least one input is required",
        ));
    }

    let any_tracked = inputs.iter().any(|input| lookup_tracked(input).is_some());
    if !any_tracked {
        return Ok(());
    }

    let tracked_inputs: Vec<Arc<TrackedTensor<f32>>> = inputs
        .iter()
        .map(|input| tracked_or_watch_as_constant(lookup_tracked(input), input))
        .collect();
    let tracked_refs: Vec<&TrackedTensor<f32>> =
        tracked_inputs.iter().map(|tt| tt.as_ref()).collect();

    let recorded = match kind {
        VariadicOpKind::Concat { axis } => TrackedTensor::concat(&tracked_refs, axis),
        VariadicOpKind::Stack { axis } => TrackedTensor::stack(&tracked_refs, axis),
    }
    .map_err(|e| PyRuntimeError::new_err(format!("implicit autograd recording failed: {e}")))?;

    register_tracked(result, Arc::new(recorded));
    Ok(())
}

/// Gather-specific counterpart of [`record_and_link_unary`]: record
/// `gather(input, indices, axis) = result` onto the implicit tape if `input`
/// is currently tracked.
///
/// `indices` is deliberately a raw `&Tensor<i32>`, never a [`PyTensor`]: it
/// is plain integer index data, not a differentiable operand, mirroring
/// `Operation::Gather`'s own design (see that variant's doc in
/// `tenflowers_autograd::tape::operations`) — it carries no gradient and must
/// never be tracked, watched on the tape, or marked as a leaf. This is why
/// `record_and_link_gather` cannot reuse [`record_and_link_unary`] (whose
/// `UnaryOpKind` dispatch assumes both `input` and `result` are
/// [`PyTensor`]s) and needs its own hook instead of an extra `UnaryOpKind`
/// variant.
///
/// # Errors
///
/// Returns an error only if `input` is tracked but recording the operation
/// on the tape itself fails (e.g. a poisoned lock).
pub fn record_and_link_gather(
    input: &PyTensor,
    indices: &Tensor<i32>,
    axis: usize,
    result: &PyTensor,
) -> PyResult<()> {
    let Some(input_tt) = lookup_tracked(input) else {
        return Ok(());
    };

    let recorded = input_tt
        .gather(indices, axis)
        .map_err(|e| PyRuntimeError::new_err(format!("implicit autograd recording failed: {e}")))?;

    register_tracked(result, Arc::new(recorded));
    Ok(())
}

/// The pooling operations wired into [`record_and_link_pooling`].
///
/// Both variants mirror the exact `(kernel_size, stride, padding)` shape
/// [`Operation::MaxPool2D`]/[`Operation::AvgPool2D`]'s own real, tested
/// backward implementations expect (`tenflowers_autograd::ops::convolution_ops::
/// max_pool2d_backward`/`avg_pool2d_backward`, dispatched to from
/// `tenflowers_autograd::tape::gradient_computation::core::process_operation_backward`)
/// — `padding` is a "valid"/"same" string, exactly like every conv variant in
/// [`TernaryOpKind`]; there is no dilation or ceil-mode field because neither
/// operation's forward ever records one (max pooling's backward hard-codes
/// unit dilation to match — see that match arm's own comment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolingOpKind {
    /// `max_pool2d(kernel_size, stride, padding)`.
    MaxPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: String,
    },
    /// `avg_pool2d(kernel_size, stride, padding)`.
    AvgPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: String,
    },
}

/// Pooling-specific counterpart of [`record_and_link_unary`]: record
/// `pool(input) = result` onto the implicit tape if `input` is currently
/// tracked.
///
/// # Why this cannot reuse [`record_and_link_unary`]/[`UnaryOpKind`]
///
/// Every existing [`UnaryOpKind`] variant dispatches to a real
/// `TrackedTensor::<method>()` that itself computes the forward result *and*
/// calls [`GradientTape::record_op`] internally (see e.g.
/// `TrackedTensor::softmax` in `tenflowers_autograd::tape::tracked_tensor`) —
/// per that enum's own doc, this hook family only ever dispatches to an
/// *existing, tested* `TrackedTensor` method and deliberately introduces no
/// gradient math of its own. No `TrackedTensor::max_pool2d`/`avg_pool2d`
/// method exists (confirmed by reading `tracked_tensor.rs` in full): the
/// `Operation::{MaxPool2D,AvgPool2D}` variants and their real, dispatched
/// backward implementations both exist, but the `TrackedTensor` recording
/// wrapper does not, so there is nothing for a `UnaryOpKind` variant to
/// delegate to.
///
/// This function is therefore written the way [`record_and_link_gather`]
/// is — a bespoke hook that calls [`GradientTape::record_op`] directly
/// (the same lock-and-append primitive every `TrackedTensor` method uses
/// internally, exposed as a public, non-chained entry point on
/// [`GradientTape`] itself) instead of routing through a `TrackedTensor`
/// method that does not exist — rather than as one more tag-enum variant
/// dispatched from inside [`record_and_link_unary`].
///
/// `result` must be exactly the tensor a real pooling forward pass over
/// `input` (using the same `kernel_size`/`stride`/`padding` embedded in
/// `kind`) would produce; this function performs no forward computation of
/// its own; it only records that `result` was already correctly computed
/// that way, mirroring every other `record_and_link_*` hook in this module.
///
/// # Errors
///
/// Returns an error only if `input` is tracked but recording the operation
/// on the tape itself fails (e.g. a poisoned lock).
pub fn record_and_link_pooling(
    kind: PoolingOpKind,
    input: &PyTensor,
    result: &PyTensor,
) -> PyResult<()> {
    let Some(input_tt) = lookup_tracked(input) else {
        return Ok(());
    };

    let operation = match kind {
        PoolingOpKind::MaxPool2D {
            kernel_size,
            stride,
            padding,
        } => Operation::MaxPool2D {
            input: input_tt.id,
            kernel_size,
            stride,
            padding,
        },
        PoolingOpKind::AvgPool2D {
            kernel_size,
            stride,
            padding,
        } => Operation::AvgPool2D {
            input: input_tt.id,
            kernel_size,
            stride,
            padding,
        },
    };

    let recorded = IMPLICIT_TAPE.with(|tape| tape.record_op(operation, (*result.tensor).clone()));
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
/// by each leaf's stable identity (either a [`tensor_key`], or a
/// [`crate::neural::layers::PyParameter`]'s explicit `id` — see
/// [`mark_leaf_param`]) — the key does not depend on the tape or registry
/// (both of which this function clears) at all. What keeps that key safe to
/// keep using differs by kind: a `tensor_key` stays safe because
/// [`IDENTITY_ANCHORS`] keeps its exact `Arc` allocation alive for as long as
/// `GRAD_STORE` references it; a `PyParameter`'s `id` stays safe because
/// nothing frees its `Arc<RwLock<..>>` allocation while the owning
/// `PyParameter` is alive, and that object's own `Drop` impl proactively
/// scrubs its `GRAD_STORE` entry at the moment it is no longer alive (see
/// the module-level "Explicit-key parameter identities" doc for the full
/// reasoning on why these need two different strategies).
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

/// Retrieve the gradient stored under `id` by a previous `.backward()` call,
/// if any.
///
/// Looks `id` up in [`GRAD_STORE`] directly — no [`tensor_key`] derivation
/// involved — rather than via [`TRACKED_REGISTRY`], because [`run_backward`]
/// clears the registry once the graph has been consumed while `GRAD_STORE`
/// deliberately survives that reset (see [`GRAD_STORE`]'s own doc). This is
/// the low-level primitive [`get_grad`] (the `PyTensor`/[`tensor_key`]-derived
/// path) and [`crate::neural::layers::PyParameter::grad`] (the explicit-`id`
/// path) both build on.
///
/// # Errors
///
/// Raises `RuntimeError` if no gradient has been computed for `id` yet —
/// either `.backward()` was never called, or whatever this `id` identifies
/// was not a leaf / was not on the path to whatever was differentiated.
pub fn get_grad_by_id(id: usize) -> PyResult<PyTensor> {
    GRAD_STORE.with(|store| {
        store
            .borrow()
            .get(&id)
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

/// Retrieve the gradient computed for `tensor` by a previous `.backward()`
/// call, if any.
///
/// Thin wrapper around [`get_grad_by_id`] that derives the lookup key from
/// `tensor` via [`tensor_key`] — see that function's doc for the full
/// lookup/error semantics, which are identical here.
///
/// # Errors
///
/// Raises `RuntimeError` if no gradient has been computed for `tensor` yet —
/// either `.backward()` was never called, or `tensor` was not a leaf / was
/// not on the path to whatever was differentiated.
pub fn get_grad(tensor: &PyTensor) -> PyResult<PyTensor> {
    get_grad_by_id(tensor_key(tensor))
}

/// Remove `id`'s entry from [`GRAD_STORE`], if present. Used by
/// [`crate::neural::layers::PyParameter::zero_grad`] — explicit,
/// user-requested clearing of a parameter's accumulated gradient in the
/// middle of that parameter's lifetime (e.g. between an accumulation
/// step and the next forward pass), which must clear *only* the
/// computed-gradient cache and nothing else about the parameter's
/// tracked status. This is deliberately **not** what
/// `PyParameter`'s `Drop` impl calls — see [`unregister_param_id`] for
/// that, and for why `Drop` needs to clear strictly more than
/// `zero_grad` does.
///
/// # Why this does not touch [`IDENTITY_ANCHORS`]
///
/// For the explicit parameter-`id` path this removal exists for,
/// `IDENTITY_ANCHORS` was never populated for that key in the first place
/// (see [`register_tracked_by_key`]'s doc — parameter ids are never
/// anchored, since nothing in this module needs to keep their `RwLock`
/// allocation alive; the [`crate::neural::layers::PyParameter`] Python
/// object does that itself), so there is nothing to prune for a parameter id
/// here.
///
/// More generally, this function is correct-by-construction rather than
/// correct-by-luck even in the astronomically unlikely case that `id`
/// happens to numerically coincide with a currently-anchored [`tensor_key`]
/// from some unrelated transient tensor: [`IDENTITY_ANCHORS`]'s *only*
/// pruning logic already lives entirely in [`run_backward`]'s retain pass,
/// which re-checks fresh [`GRAD_STORE`] membership on every single
/// `run_backward` call rather than being incrementally maintained on each
/// removal — so an anchor is never pruned here, only ever by that one
/// centralized, always-correct retain pass the next time it runs. This
/// function only ever needs to touch `GRAD_STORE` itself.
pub fn clear_grad_by_id(id: usize) {
    GRAD_STORE.with(|store| {
        store.borrow_mut().remove(&id);
    });
}

/// Remove every trace of `id` from [`TRACKED_REGISTRY`], [`LEAVES`], and
/// [`GRAD_STORE`]. Used **only** by `PyParameter`'s `Drop` impl (see
/// `layers.rs`) — proactive, full un-tracking of a parameter's identity
/// at the moment it is actually deallocated, closing an address-reuse
/// hazard analogous to the one [`IDENTITY_ANCHORS`] closes for
/// [`tensor_key`]-derived keys, but via the "scrub eagerly instead of
/// anchor" strategy this module's doc describes for explicit parameter
/// identities (see the module-level "Explicit-key parameter identities"
/// section) — extended here to cover `TRACKED_REGISTRY`/`LEAVES` as well
/// as `GRAD_STORE`.
///
/// # Why `Drop` needs this and not just [`clear_grad_by_id`]
///
/// [`clear_grad_by_id`] alone leaves a real gap: if a parameter is
/// [`mark_leaf_param`]'d (which inserts its `id` into
/// [`TRACKED_REGISTRY`] and pushes it onto [`LEAVES`]) and then dropped
/// *without* an intervening [`run_backward`] call — a legitimate
/// pattern, e.g. a forward pass whose result the caller never actually
/// backpropagates through — its `Arc<RwLock<Tensor<f32>>>` allocation is
/// freed while `TRACKED_REGISTRY`/`LEAVES` still reference that exact
/// address as a key. If the allocator later hands that freed address to
/// a **different**, unrelated `PyParameter`'s own `Arc::new(RwLock::new(..))`
/// allocation (ordinary allocator behaviour, not a contrived edge case —
/// confirmed reproducible via
/// [`super::tests::dropped_untracked_backward_param_does_not_poison_tracked_registry_on_address_reuse`]),
/// [`mark_leaf_param`]'s own idempotency check
/// (`TRACKED_REGISTRY.contains_key(&explicit_key)`) cannot tell "this
/// exact parameter was already marked earlier this cycle" apart from "an
/// unrelated, already-dropped parameter that used to occupy this freed
/// address left a stale entry here" — both look identical: the key is
/// present. The check takes the early-return branch, and the new,
/// completely different parameter is silently **never watched on the
/// tape at all**, even though the caller just called `mark_leaf_param`
/// on it in perfectly good faith. Every later attempt to record an
/// operation on that parameter's snapshot then also silently no-ops (via
/// [`lookup_tracked`] finding nothing under the snapshot's own
/// `tensor_key`, since `mark_leaf_param` never reached the
/// `register_tracked(transient, ..)` call that would have registered
/// it), and the gradient the caller expects to exist after a real
/// `.backward()` call simply never materializes.
///
/// Scrubbing `TRACKED_REGISTRY`/`LEAVES` here — at the moment the
/// address is actually about to be freed, mirroring exactly what this
/// same `Drop` impl already does for `GRAD_STORE` via
/// [`clear_grad_by_id`] and for the identical reason — closes this
/// completely: by the time the allocator could hand this `id` to a new
/// parameter, nothing in these tables still claims it.
///
/// # Why this is safe to call unconditionally on every `Drop`, even for
/// a parameter that is mid-graph (tracked, contributing to a still-live
/// but not-yet-backward'd computation) when it is dropped
///
/// [`TRACKED_REGISTRY`] is purely a *lookup index* from stable identity
/// to `TrackedTensor` — removing an entry does not touch the underlying
/// tape nodes/operations already recorded on [`IMPLICIT_TAPE`], which
/// reference each other by `TensorId`, not by this table. Any downstream
/// tensor that already recorded an edge to this parameter's tape node
/// keeps working exactly as before; only *future* [`lookup_tracked`]
/// calls keyed on this `id` stop finding it (correctly — the
/// `PyParameter` that owned this identity no longer exists to ever
/// receive a gradient back through it anyway, see below).
///
/// Removing the [`LEAVES`] entry means a later `.backward()` call will
/// no longer differentiate with respect to this now-dropped parameter.
/// This cannot lose an observable result: [`crate::neural::layers::PyParameter::grad`]
/// is an instance method — once the `PyParameter` object is dropped,
/// there is no longer any way for a caller to invoke `.grad()` on it in
/// the first place, tracked-and-computed or not. Continuing to
/// differentiate with respect to a leaf whose result can never again be
/// observed would be pure wasted work, not a behaviour anything could
/// depend on.
///
/// # Why this does not touch [`IDENTITY_ANCHORS`]
///
/// Same reasoning as [`clear_grad_by_id`]: parameter ids are never
/// anchored in the first place (see [`register_tracked_by_key`]'s doc),
/// so there is nothing to prune here either.
pub fn unregister_param_id(id: usize) {
    TRACKED_REGISTRY.with(|registry| {
        registry.borrow_mut().remove(&id);
    });
    LEAVES.with(|leaves| {
        leaves.borrow_mut().retain(|(key, _)| *key != id);
    });
    clear_grad_by_id(id);
}

#[cfg(test)]
mod tests;
