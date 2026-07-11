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

// ---------------------------------------------------------------------
// PyParameter / explicit-key ("mark_leaf_param") tests.
//
// These import the *real* `PyParameter` (same crate, no import-cycle
// concern: `crate::neural::layers` and `crate::implicit_autograd` are
// both plain modules of the one `tenflowers-ffi` crate, and Rust's
// module graph — unlike its crate-dependency graph — has no acyclicity
// requirement) so these tests prove the real `PyParameter` +
// `implicit_autograd` integration end-to-end, not just this module's
// internals against a hand-rolled stand-in.
// ---------------------------------------------------------------------
use crate::neural::layers::PyParameter;

/// Build `scalar = sum(param_snapshot + y)` with `param`'s current-value
/// snapshot marked as a leaf via [`mark_leaf_param`] (mirroring how
/// [`add_then_sum_backward`] wires up a plain [`mark_leaf`]-tracked
/// `PyTensor`), and run `.backward()` on it.
///
/// Returns the `PyTensor` snapshot that was marked as the leaf so the
/// caller can additionally sanity-check it if needed; the gradient
/// itself must be read back via `get_grad_by_id(param.id())`, not from
/// the returned snapshot's own `tensor_key` (that would be the wrong
/// key — see [`mark_leaf_param`]'s doc).
fn mark_param_leaf_add_then_sum_backward(param: &PyParameter, y: &PyTensor) -> PyTensor {
    let snapshot = param.to_tensor().expect("to_tensor must succeed");
    mark_leaf_param(&snapshot, param.id());

    let raw = tenflowers_core::ops::add(&snapshot.tensor, &y.tensor).expect("add must succeed");
    let sum_result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_binary(BinaryOpKind::Add, &snapshot, y, &sum_result)
        .expect("recording add must succeed");

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
    snapshot
}

/// Build `scalar = sum(param_snapshot * param_snapshot)` — i.e.
/// `L = sum(param^2)`, so `dL/d(param) = 2 * param` — with `param`'s
/// current-value snapshot marked as a leaf via [`mark_leaf_param`], and
/// run `.backward()` on it.
///
/// Deliberately squares the parameter against *itself* (rather than
/// against a second, independent tensor, as
/// [`mark_param_leaf_add_then_sum_backward`]'s `add` does) so the
/// resulting gradient is a direct function of the parameter's *own
/// current value* — this is what makes it possible to prove that a
/// second cycle after `set_data` picks up the *new* value rather than
/// silently reusing something stale from the first cycle (an `add`
/// against a constant would yield gradient `1` regardless of the
/// parameter's value, which could never distinguish "fresh data" from
/// "stale data").
fn mark_param_leaf_square_then_sum_backward(param: &PyParameter) -> PyTensor {
    let snapshot = param.to_tensor().expect("to_tensor must succeed");
    mark_leaf_param(&snapshot, param.id());

    let raw =
        tenflowers_core::ops::mul(&snapshot.tensor, &snapshot.tensor).expect("mul must succeed");
    let mul_result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_binary(BinaryOpKind::Mul, &snapshot, &snapshot, &mul_result)
        .expect("recording mul must succeed");

    let summed =
        tenflowers_core::ops::sum(&mul_result.tensor, None, false).expect("sum must succeed");
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
        &mul_result,
        &scalar,
    )
    .expect("recording sum must succeed");

    run_backward(&scalar).expect("backward must succeed");
    snapshot
}

/// A `PyParameter`'s `id` — the address of its `Arc<RwLock<Tensor<f32>>>`
/// value cell — must stay exactly the same no matter how many times
/// `set_data` replaces the cell's *contents*. This is the core property
/// [`mark_leaf_param`] / [`get_grad_by_id`] rely on to keep resolving to
/// the same tape leaf across an optimizer's repeated in-place updates.
#[test]
fn parameter_id_stable_across_set_data_calls() {
    reset_for_test();
    let initial = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let param = PyParameter::new(initial, Some(true));
    let id_at_construction = param.id();

    for replacement in [
        vec![10.0, 20.0, 30.0],
        vec![-1.0, -2.0, -3.0],
        vec![0.0, 0.0, 0.0],
        vec![100.5, 200.5, 300.5],
    ] {
        let new_tensor =
            Tensor::from_vec(replacement, &[3]).expect("tensor construction must succeed");
        param
            .set_data(new_tensor)
            .expect("set_data must succeed on a healthy lock");
        assert_eq!(
            param.id(),
            id_at_construction,
            "a parameter's id must never change across set_data calls, \
             no matter how many times its contents are replaced"
        );
    }
}

/// End-to-end proof that [`mark_leaf_param`] + a real forward/backward
/// cycle through the implicit tape correctly populates the gradient
/// reachable via `get_grad_by_id(param.id())` — genuine integration
/// through the real tape machinery, not just a unit test of the
/// plumbing in isolation.
#[test]
fn mark_leaf_param_end_to_end_gradient_flow() {
    reset_for_test();
    let initial = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let param = PyParameter::new(initial, Some(true));
    let y = make_tensor(vec![10.0, 20.0, 30.0], &[3]);

    mark_param_leaf_add_then_sum_backward(&param, &y);

    let grad = get_grad_by_id(param.id()).expect("grad must be available for the parameter");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    // d(sum(param + y))/d(param) = 1 for every element.
    assert_eq!(grad_data, vec![1.0, 1.0, 1.0]);
}

/// The whole point of the `PyParameter` redesign, proven end-to-end: a
/// parameter's `id` must keep resolving to fresh, correct gradients
/// across repeated "training step" cycles — `mark_leaf_param` +
/// forward + backward, then `set_data` (simulating an optimizer's
/// in-place `w -= lr * grad` update), then a *second*, independent
/// `mark_leaf_param` + forward + backward cycle using a fresh
/// `to_tensor()` snapshot — all resolving through the *same*
/// `get_grad_by_id(param.id())` key throughout.
///
/// This specifically exercises the idempotency-by-explicit-key
/// correctness requirement from `mark_leaf_param`'s doc: after
/// `run_backward` clears `TRACKED_REGISTRY`/`LEAVES`/the tape at the end
/// of cycle 1, `mark_leaf_param` must be called again for cycle 2 (a
/// real layer's `forward()` does this at the start of *every* forward
/// pass, not once ever), and it must correctly re-register a fresh leaf
/// under the *same* `explicit_key` rather than either (a) refusing to
/// re-register because it wrongly still thinks the id is tracked from
/// cycle 1, or (b) resolving to some stale, cycle-1 tracked tensor.
///
/// Uses `mark_param_leaf_square_then_sum_backward` (`L = sum(param^2)`,
/// `dL/d(param) = 2*param`) specifically *because* the gradient is a
/// direct function of the parameter's own current value: if cycle 2's
/// gradient reflects the *old* (pre-`set_data`) value instead of the
/// new one, this test will fail with a clearly wrong number rather than
/// silently passing on a coincidence.
#[test]
fn set_data_then_fresh_cycle_resolves_by_same_id() {
    reset_for_test();
    let initial = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let param = PyParameter::new(initial, Some(true));
    let id = param.id();

    // Cycle 1: L = sum(param^2), dL/d(param) = 2*param = [2, 4, 6].
    mark_param_leaf_square_then_sum_backward(&param);
    let grad1 = get_grad_by_id(id).expect("cycle 1 grad must be available");
    let grad1_data = grad1.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(
        grad1_data,
        vec![2.0, 4.0, 6.0],
        "cycle 1 gradient must match d(sum(param^2))/d(param) = 2*param for the initial value"
    );

    // Simulate an optimizer's in-place update: w <- new value. The
    // parameter's id must be completely unaffected (already covered by
    // `parameter_id_stable_across_set_data_calls`, re-asserted here
    // because it is exactly the property this test depends on).
    let updated =
        Tensor::from_vec(vec![5.0, 10.0, 15.0], &[3]).expect("tensor construction must succeed");
    param
        .set_data(updated)
        .expect("set_data must succeed on a healthy lock");
    assert_eq!(
        param.id(),
        id,
        "set_data must not change the parameter's id"
    );

    // Cycle 2: a fresh, independent mark_leaf_param + forward + backward
    // cycle (a real layer's forward() does exactly this at the start of
    // every forward pass) using the *new* data.
    // dL/d(param) = 2*param = [10, 20, 30] for the new value — provably
    // different from cycle 1's [2, 4, 6], so a pass here cannot be
    // explained by accidentally reusing stale cycle-1 state.
    mark_param_leaf_square_then_sum_backward(&param);
    let grad2 = get_grad_by_id(id).expect(
        "cycle 2 grad must be available under the SAME id as cycle 1 \
         — this is the crux of the whole design",
    );
    let grad2_data = grad2.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(
        grad2_data,
        vec![10.0, 20.0, 30.0],
        "cycle 2 gradient must reflect the NEW data written by set_data, \
         not stale data leaked from cycle 1"
    );
}

/// Mirrors [`address_reuse_does_not_leak_stale_gradient`]'s spirit,
/// adapted for the parameter-id path: independent parameters must never
/// cross-contaminate each other's gradients, and — since a
/// `PyParameter`'s `id` is just as much an allocator address as a
/// `tensor_key` is — many short-lived parameters whose `RwLock`
/// allocations get freed and reused by the allocator must not cause a
/// brand-new, never-marked parameter to spuriously resolve to a stale
/// gradient from an earlier, unrelated, already-completed cycle.
#[test]
fn independent_parameters_do_not_cross_contaminate_even_on_address_reuse() {
    reset_for_test();

    // Two independent, concurrently-alive parameters: necessarily
    // different `id`s, since they are different `RwLock` allocations.
    let param_a = PyParameter::new(make_tensor(vec![1.0, 2.0, 3.0], &[3]), Some(true));
    let param_b = PyParameter::new(make_tensor(vec![100.0, 200.0], &[2]), Some(true));
    assert_ne!(
        param_a.id(),
        param_b.id(),
        "two independently constructed parameters must have distinct identities"
    );

    let y_a = make_tensor(vec![10.0, 20.0, 30.0], &[3]);
    mark_param_leaf_add_then_sum_backward(&param_a, &y_a);
    let grad_a = get_grad_by_id(param_a.id()).expect("param_a's grad must be available");
    let grad_a_data = grad_a.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(grad_a_data, vec![1.0, 1.0, 1.0]);

    let y_b = make_tensor(vec![-1.0, -1.0], &[2]);
    mark_param_leaf_add_then_sum_backward(&param_b, &y_b);
    let grad_b = get_grad_by_id(param_b.id()).expect("param_b's grad must be available");
    let grad_b_data = grad_b.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(grad_b_data, vec![1.0, 1.0]);

    // Each id must resolve ONLY to its own gradient.
    assert_eq!(
        get_grad_by_id(param_a.id())
            .expect("param_a's grad must still be readable")
            .tensor
            .to_vec()
            .expect("grad data must be readable"),
        vec![1.0, 1.0, 1.0],
        "param_a's gradient must not have been overwritten by param_b's cycle"
    );

    // Run many independent short-lived parameter + mark_leaf_param +
    // forward + backward cycles, each using a temporary parameter
    // dropped at the end of its loop iteration, to encourage the
    // allocator to reuse freed `RwLock` addresses for later iterations'
    // parameters — the same allocator-pressure technique
    // `address_reuse_does_not_leak_stale_gradient` uses for tensor_key
    // addresses, adapted here to parameter-id addresses.
    for i in 0..64 {
        let transient_param = PyParameter::new(
            make_tensor(vec![i as f32, (i + 1) as f32], &[2]),
            Some(true),
        );
        let transient_y = make_tensor(vec![10.0, 20.0], &[2]);
        mark_param_leaf_add_then_sum_backward(&transient_param, &transient_y);
        // transient_param (and every intermediate PyTensor created
        // inside the helper) is dropped here at the end of the loop
        // body. transient_param's `Arc<RwLock<Tensor<f32>>>`
        // allocation is freed unless something still anchors it — and
        // per this module's design, nothing should: parameter ids are
        // deliberately never anchored (see `register_tracked_by_key`'s
        // doc), because the PyParameter object's own lifetime is what
        // keeps the allocation alive while it matters, not this module.
    }

    // A brand-new parameter that has never participated in implicit
    // autograd at all must not be considered tracked under
    // TRACKED_REGISTRY, no matter what address its RwLock allocation
    // happens to occupy after 64 rounds of allocate-then-free pressure.
    let fresh_param = PyParameter::new(make_tensor(vec![99.0, 100.0], &[2]), Some(true));
    let fresh_is_tracked = TRACKED_REGISTRY.with(|r| r.borrow().contains_key(&fresh_param.id()));
    assert!(
        !fresh_is_tracked,
        "a fresh, never-marked parameter must not appear tracked even if its \
         id (RwLock address) was previously used by a completed cycle's parameter"
    );
    assert!(
        get_grad_by_id(fresh_param.id()).is_err(),
        "a fresh, never-marked parameter must not spuriously return a stale \
         gradient left over from an earlier, address-reused cycle"
    );

    // The two original, still-alive parameters' gradients must still be
    // exactly correct and unaffected by all 64 transient cycles and by
    // the fresh-parameter check above.
    let final_grad_a = get_grad_by_id(param_a.id())
        .expect("param_a's grad must survive 64 unrelated transient cycles");
    assert_eq!(
        final_grad_a
            .tensor
            .to_vec()
            .expect("grad data must be readable"),
        vec![1.0, 1.0, 1.0]
    );
    let final_grad_b = get_grad_by_id(param_b.id())
        .expect("param_b's grad must survive 64 unrelated transient cycles");
    assert_eq!(
        final_grad_b
            .tensor
            .to_vec()
            .expect("grad data must be readable"),
        vec![1.0, 1.0]
    );
}

/// INVESTIGATION: unlike
/// `independent_parameters_do_not_cross_contaminate_even_on_address_reuse`
/// (whose loop always drives every transient parameter through a full
/// `mark_param_leaf_add_then_sum_backward` cycle — i.e. `run_backward`
/// *always* runs, and therefore `TRACKED_REGISTRY` is always fully
/// cleared, before that iteration's parameter is dropped), this test
/// drops a tracked-but-never-backward'd parameter and checks whether a
/// later, address-colliding parameter can be poisoned by its leftover
/// `TRACKED_REGISTRY` entry.
///
/// `PyParameter::drop` (see `layers.rs`) proactively clears
/// [`GRAD_STORE`] for its own `id`, closing the address-reuse hazard
/// for *that* table specifically. It does **not** touch
/// [`TRACKED_REGISTRY`]. If a parameter is `mark_leaf_param`'d (which
/// inserts into `TRACKED_REGISTRY` under its `id`) and then dropped
/// *without* an intervening `run_backward` call (which is the only
/// other thing that clears `TRACKED_REGISTRY`), its `id` — the address
/// of its `Arc<RwLock<Tensor<f32>>>` — is freed while still present as
/// a `TRACKED_REGISTRY` key. `mark_leaf_param`'s own idempotency check
/// (`TRACKED_REGISTRY.contains_key(&explicit_key)`) cannot distinguish
/// "this exact parameter was already marked earlier this cycle" from
/// "a different, already-dropped parameter that used to occupy this
/// freed address left a stale entry here" — both look identical: the
/// key is present. If the allocator reuses the freed address for a new,
/// unrelated parameter, the very first `mark_leaf_param` call for that
/// new parameter takes the early-return branch and never watches it on
/// the tape at all.
#[test]
fn dropped_untracked_backward_param_does_not_poison_tracked_registry_on_address_reuse() {
    // Every `.expect()` below that can fail formats a `PyErr` (via
    // `Debug`/`Display`) as part of the panic message, which needs an
    // initialized Python interpreter — see the same idiom's explanation
    // on `backward_on_untracked_tensor_errors_clearly` above.
    pyo3::Python::initialize();
    reset_for_test();

    // Run many short-lived parameters through mark_leaf_param ONLY —
    // deliberately never calling run_backward — so each one is dropped
    // while still present in TRACKED_REGISTRY, to encourage the
    // allocator to reuse a freed RwLock address for a later iteration's
    // parameter.
    for i in 0..256 {
        let stale_param = PyParameter::new(
            make_tensor(vec![i as f32, (i + 1) as f32], &[2]),
            Some(true),
        );
        let snapshot = stale_param.to_tensor().expect("to_tensor must succeed");
        mark_leaf_param(&snapshot, stale_param.id());
        // stale_param is dropped here. Its Drop impl clears GRAD_STORE
        // for its id (a no-op here — nothing was ever inserted into
        // GRAD_STORE without a run_backward call), but leaves
        // TRACKED_REGISTRY[stale_param.id()] and LEAVES untouched.
    }

    // A brand-new parameter, constructed after the loop, that is meant
    // to be a completely independent, real leaf.
    let real_param = PyParameter::new(make_tensor(vec![7.0, 8.0, 9.0], &[3]), Some(true));
    let y = make_tensor(vec![1.0, 1.0, 1.0], &[3]);
    mark_param_leaf_add_then_sum_backward(&real_param, &y);

    let grad = get_grad_by_id(real_param.id());
    let grad_data = grad
        .expect(
            "real_param must have a gradient after a real forward+backward cycle — if this \
             errors, mark_leaf_param's early-return treated a stale, address-colliding \
             TRACKED_REGISTRY entry from an earlier, dropped, never-backward'd parameter as \
             proof real_param was already tracked, and silently never watched it on the tape \
             at all",
        )
        .tensor
        .to_vec()
        .expect("grad data must be readable");
    assert_eq!(
        grad_data,
        vec![1.0, 1.0, 1.0],
        "real_param's gradient must be exactly correct — d(sum(param + y))/d(param) = 1 — \
         not silently missing and not some stale value inherited from address-colliding \
         earlier parameters"
    );
}

// ---------------------------------------------------------------------
// record_and_link_ternary / record_and_link_variadic / record_and_link_gather
// ---------------------------------------------------------------------

/// Build `scalar = sum(conv1d(input, weight, bias))` with `input`,
/// `weight`, and `bias` all marked as leaves via [`mark_leaf`], recorded
/// through [`record_and_link_ternary`] with
/// [`TernaryOpKind::Conv1D`], and run `.backward()` on it.
///
/// Uses the same shape/value convention as
/// `neural::conv_layers::conv1d_forward_real_values` (single batch,
/// single in/out channel, kernel length 2, all-ones weight, "valid"
/// padding, stride 1): `input = [1, 2, 3, 4]` (shape `[1, 1, 4]`),
/// `weight = [1, 1]` (shape `[1, 1, 2]`), `bias = [0]` (shape `[1]`) so
/// the forward output is exactly `[3, 5, 7]` (each output position is
/// the sum of two adjacent input elements, plus a zero bias) — a real,
/// checkable convolution rather than a degenerate all-zeros case that
/// could pass even with a broken backward.
#[test]
fn ternary_conv1d_gradient_reaches_all_three_operands() {
    reset_for_test();
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let weight = make_tensor(vec![1.0, 1.0], &[1, 1, 2]);
    let bias = make_tensor(vec![0.0], &[1]);
    mark_leaf(&input);
    mark_leaf(&weight);
    mark_leaf(&bias);

    let raw = tenflowers_core::ops::conv1d(
        &input.tensor,
        &weight.tensor,
        Some(&bias.tensor),
        1,
        "valid",
    )
    .expect("conv1d must succeed");
    assert_eq!(
        raw.to_vec().expect("conv output readable"),
        vec![3.0, 5.0, 7.0]
    );
    let conv_result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_ternary(
        TernaryOpKind::Conv1D {
            stride: 1,
            padding: "valid".to_string(),
        },
        &input,
        &weight,
        Some(&bias),
        None,
        &conv_result,
    )
    .expect("recording conv1d must succeed");

    let summed =
        tenflowers_core::ops::sum(&conv_result.tensor, None, false).expect("sum must succeed");
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
        &conv_result,
        &scalar,
    )
    .expect("recording sum must succeed");

    run_backward(&scalar).expect("backward must succeed");

    // All three operands must have a reachable gradient — this is the
    // core property `record_and_link_ternary` exists to guarantee: a
    // caller-supplied `TernaryOpKind` must genuinely link every
    // differentiable operand onto the tape, not just `a`.
    let grad_input = get_grad(&input).expect("input gradient must be reachable");
    let grad_weight = get_grad(&weight).expect("weight gradient must be reachable");
    let grad_bias = get_grad(&bias).expect("bias gradient must be reachable");

    // d(sum(conv1d(input, weight)))/d(weight[k]) = sum over output
    // positions of input[pos + k] (each weight tap is reused at every
    // valid output position); for input=[1,2,3,4], kernel length 2,
    // output length 3: tap 0 sees input[0..3]=[1,2,3] (sum 6), tap 1
    // sees input[1..4]=[2,3,4] (sum 9).
    assert_eq!(
        grad_weight.tensor.to_vec().expect("weight grad readable"),
        vec![6.0, 9.0]
    );
    // d(sum(...))/d(bias) = number of output positions the bias is
    // added at = 3 (one add per output position).
    assert_eq!(
        grad_bias.tensor.to_vec().expect("bias grad readable"),
        vec![3.0]
    );
    // d(sum(...))/d(input[i]) = sum of weight taps that touch input[i].
    // With an all-ones kernel of length 2 and output length 3: input[0]
    // is touched by 1 output (tap 0 of output 0) => 1; input[1] by 2
    // outputs (tap 1 of output 0, tap 0 of output 1) => 2; input[2] by 2
    // outputs => 2; input[3] by 1 output => 1.
    assert_eq!(
        grad_input.tensor.to_vec().expect("input grad readable"),
        vec![1.0, 2.0, 2.0, 1.0]
    );
}

/// Build `scalar = sum(concat([a, b, c], axis=0))` with all three inputs
/// marked as leaves via [`mark_leaf`] and recorded through
/// [`record_and_link_variadic`] with [`VariadicOpKind::Concat`], and run
/// `.backward()` on it. Confirms the gradient is reachable for every
/// one of the three (not just the first two, which is all
/// `record_and_link_binary` could ever prove) — the core property
/// `record_and_link_variadic` exists to guarantee for an arbitrary
/// number of equal-role operands.
#[test]
fn variadic_concat_gradient_reaches_all_three_inputs() {
    reset_for_test();
    let a = make_tensor(vec![1.0, 2.0], &[2]);
    let b = make_tensor(vec![3.0, 4.0, 5.0], &[3]);
    let c = make_tensor(vec![6.0], &[1]);
    mark_leaf(&a);
    mark_leaf(&b);
    mark_leaf(&c);

    let raw = tenflowers_core::ops::concat(&[&a.tensor, &b.tensor, &c.tensor], 0)
        .expect("concat must succeed");
    assert_eq!(
        raw.to_vec().expect("concat output readable"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let concat_result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_variadic(
        VariadicOpKind::Concat { axis: 0 },
        &[&a, &b, &c],
        &concat_result,
    )
    .expect("recording concat must succeed");

    let summed =
        tenflowers_core::ops::sum(&concat_result.tensor, None, false).expect("sum must succeed");
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
        &concat_result,
        &scalar,
    )
    .expect("recording sum must succeed");

    run_backward(&scalar).expect("backward must succeed");

    // d(sum(concat(a, b, c)))/d(x) = ones_like(x) for every one of the
    // three concatenated inputs.
    let grad_a = get_grad(&a).expect("a's gradient must be reachable");
    assert_eq!(
        grad_a.tensor.to_vec().expect("grad readable"),
        vec![1.0, 1.0]
    );
    let grad_b = get_grad(&b).expect("b's gradient must be reachable");
    assert_eq!(
        grad_b.tensor.to_vec().expect("grad readable"),
        vec![1.0, 1.0, 1.0]
    );
    let grad_c = get_grad(&c).expect("c's gradient must be reachable");
    assert_eq!(grad_c.tensor.to_vec().expect("grad readable"), vec![1.0]);
}

/// Build `scalar = sum(gather(input, indices, axis=0))` with `input`
/// marked as a leaf via [`mark_leaf`] and recorded through
/// [`record_and_link_gather`], and run `.backward()` on it.
///
/// Proves two things at once:
/// 1. `input`'s gradient IS reachable (the tracked side of the hook
///    genuinely links onto the tape).
/// 2. `indices` is NEVER tracked, by construction rather than by
///    assertion-after-the-fact: `indices` is a raw `Tensor<i32>`, never
///    wrapped in a `PyTensor` anywhere in this test, so there is no
///    `PyTensor` sharing its identity that `lookup_tracked`/`get_grad`
///    could even be called on — the type signature of
///    `record_and_link_gather` (`indices: &Tensor<i32>`, not `&PyTensor`)
///    makes "indices accidentally becomes a leaf" a compile-time
///    impossibility, not just a runtime property to check. This test
///    additionally double-checks the runtime side: a freshly constructed
///    `PyTensor` that happens to hold the exact same index *values* as
///    `indices` (but is never passed to `record_and_link_gather` at all)
///    must still correctly report "no gradient computed" — i.e. nothing
///    about this cycle spuriously tracks unrelated tensors either.
#[test]
fn gather_indices_are_never_tracked() {
    pyo3::Python::initialize();
    reset_for_test();
    let input = make_tensor(vec![10.0, 20.0, 30.0, 40.0], &[4]);
    mark_leaf(&input);

    let indices = Tensor::<i32>::from_vec(vec![0, 2, 3], &[3]).expect("indices construction");

    let raw =
        tenflowers_core::ops::gather(&input.tensor, &indices, 0).expect("gather must succeed");
    assert_eq!(
        raw.to_vec().expect("gather output readable"),
        vec![10.0, 30.0, 40.0]
    );
    let gather_result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_gather(&input, &indices, 0, &gather_result)
        .expect("recording gather must succeed");

    let summed =
        tenflowers_core::ops::sum(&gather_result.tensor, None, false).expect("sum must succeed");
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
        &gather_result,
        &scalar,
    )
    .expect("recording sum must succeed");

    run_backward(&scalar).expect("backward must succeed");

    // input's gradient must be reachable: d(sum(gather(input,
    // [0,2,3])))/d(input) scatters a 1.0 back to each gathered
    // position (0, 2, 3) and 0.0 to the untouched position (1).
    let grad_input = get_grad(&input).expect("input's gradient must be reachable");
    assert_eq!(
        grad_input.tensor.to_vec().expect("grad readable"),
        vec![1.0, 0.0, 1.0, 1.0]
    );

    // A never-passed-to-record_and_link_gather PyTensor holding the
    // same values as `indices` must NOT spuriously resolve to a
    // gradient — confirming nothing about this cycle accidentally
    // tracked something it shouldn't have.
    let indices_lookalike = make_tensor(vec![0.0, 2.0, 3.0], &[3]);
    assert!(
        lookup_tracked(&indices_lookalike).is_none(),
        "a PyTensor with the same values as `indices` (but never linked \
         through record_and_link_gather) must not be tracked"
    );
    let err = get_grad(&indices_lookalike)
        .expect_err("indices must never receive a gradient — they carry none");
    let message = err.to_string();
    assert!(
        message.contains("never had requires_grad"),
        "unexpected error message: {message}"
    );
}

// ---------------------------------------------------------------------
// UnaryOpKind::{Gelu, Swish, Mish, LeakyRelu, Elu, Relu6, HardSwish,
// LogSoftmax, Log, Abs, Clamp} — one gradient-flow test per variant,
// mirroring `add_then_sum_backward`'s "mark leaf, compute the matching
// raw forward value, record_and_link_unary(kind), reduce to scalar via
// UnaryOpKind::Sum, run_backward" idiom.
// ---------------------------------------------------------------------

/// Build `scalar = sum(op(x))` with `x` marked as a leaf via [`mark_leaf`],
/// the raw forward value supplied by the caller (so it can be computed via
/// whichever raw `Tensor<f32>` method/free-function matches `kind`), and
/// `.backward()` run on the result. Returns the leaf `x` so the caller can
/// inspect `get_grad(&x)`.
///
/// Generalizes [`add_then_sum_backward`] from a binary op to any
/// [`UnaryOpKind`], for the eleven activation/elementwise variants added in
/// this wave (`Gelu`, `Swish`, `Mish`, `LeakyRelu`, `Elu`, `Relu6`,
/// `HardSwish`, `LogSoftmax`, `Log`, `Abs`, `Clamp`).
fn unary_op_then_sum_backward(
    x: PyTensor,
    kind: UnaryOpKind,
    raw_forward: Tensor<f32>,
) -> PyTensor {
    mark_leaf(&x);

    let op_result = PyTensor {
        tensor: Arc::new(raw_forward),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_unary(kind, &x, &op_result).expect("recording op must succeed");

    let summed =
        tenflowers_core::ops::sum(&op_result.tensor, None, false).expect("sum must succeed");
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
        &op_result,
        &scalar,
    )
    .expect("recording sum must succeed");

    run_backward(&scalar).expect("backward must succeed");
    x
}

/// `gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))`; no closed-form-simple derivative,
/// so the expected gradient is a central-difference numerical check
/// (verified independently in Python before writing this test:
/// `d(gelu)/dx` at `[0.5, 1.0, -0.5, 2.0]` is approximately
/// `[0.8675, 1.0833, 0.1325, 1.0852]`), hence the tolerance-based
/// assertion rather than `assert_eq!`.
#[test]
fn gelu_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![0.5, 1.0, -0.5, 2.0], &[4]);
    let raw = x.tensor.gelu().expect("gelu must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Gelu, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    let expected = [0.8674951, 1.0833155, 0.1325049, 1.0852318];
    for (actual, expected) in grad_data.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "gelu gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

/// `swish(x) = x*sigmoid(x)`, `d/dx = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))`.
/// Verified analytically and cross-checked numerically before writing this
/// test: at `[0.5, 1.0, -0.5, 2.0]` the derivative is approximately
/// `[0.7400, 0.9277, 0.2600, 1.0908]`.
#[test]
fn swish_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![0.5, 1.0, -0.5, 2.0], &[4]);
    let raw = x.tensor.swish().expect("swish must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Swish, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    let expected = [0.7399612, 0.9276705, 0.2600388, 1.0907842];
    for (actual, expected) in grad_data.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "swish gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

/// `mish(x) = x*tanh(softplus(x))`; fiddly to hand-derive in closed form, so
/// the expected gradient is a central-difference numerical check (verified
/// independently in Python before writing this test: `d(mish)/dx` at
/// `[0.5, 1.0, -0.5, 2.0]` is approximately
/// `[0.8864, 1.0490, 0.2895, 1.0693]`).
#[test]
fn mish_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![0.5, 1.0, -0.5, 2.0], &[4]);
    let raw = x.tensor.mish().expect("mish must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Mish, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    let expected = [0.8864244, 1.0490362, 0.2895107, 1.0693179];
    for (actual, expected) in grad_data.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "mish gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

/// `leaky_relu(x, negative_slope) = x` for `x>0`, `= negative_slope*x` for
/// `x<0`; `d/dx = 1` for `x>0`, `= negative_slope` for `x<0`. Inputs are
/// clearly positive/negative (`2.0, -3.0, 1.0, -0.5`), away from the `x=0`
/// kink, so the exact piecewise gradient is checkable with `assert_eq!`.
#[test]
fn leaky_relu_gradient_matches_expected() {
    reset_for_test();
    let negative_slope = 0.1f32;
    let x = make_tensor(vec![2.0, -3.0, 1.0, -0.5], &[4]);
    let raw = x
        .tensor
        .leaky_relu(negative_slope)
        .expect("leaky_relu must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::LeakyRelu { negative_slope }, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(grad_data, vec![1.0, 0.1, 1.0, 0.1]);
}

/// `elu(x, alpha) = x` for `x>0`, `= alpha*(exp(x)-1)` for `x<0`;
/// `d/dx = 1` for `x>0`, `= alpha*exp(x)` for `x<0`. Inputs are clearly
/// positive/negative (`2.0, -1.0, 0.5, -2.0`), away from the `x=0` kink.
#[test]
fn elu_gradient_matches_expected() {
    reset_for_test();
    let alpha = 1.0f32;
    let x = make_tensor(vec![2.0, -1.0, 0.5, -2.0], &[4]);
    let raw = x.tensor.elu(alpha).expect("elu must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Elu { alpha }, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    let expected = [1.0, (-1.0f32).exp(), 1.0, (-2.0f32).exp()];
    for (actual, expected) in grad_data.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "elu gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

/// `relu6(x) = min(max(x, 0), 6)`; `d/dx = 1` for `0<x<6`, `= 0` outside
/// that range. Inputs `[-1.0, 2.0, 7.0, 4.0]` deliberately mix a
/// below-range, an in-range, an above-range, and another in-range value so
/// the test cannot pass on a "gradient is always 1" or "always 0" bug —
/// both the in-range and out-of-range gradients are exercised.
#[test]
fn relu6_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![-1.0, 2.0, 7.0, 4.0], &[4]);
    let raw = tenflowers_core::ops::activation::relu6(&x.tensor).expect("relu6 must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Relu6, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(
        grad_data,
        vec![0.0, 1.0, 0.0, 1.0],
        "relu6 gradient must be 1 strictly inside (0, 6) and 0 outside \
         — an all-1s or all-0s result would indicate the range mask is broken"
    );
}

/// `hard_swish(x) = x * relu6(x + 3) / 6`; piecewise: `0` for `x<=-3`,
/// `x` for `x>=3` (derivative `1`), and `(2x+3)/6` in between. Inputs
/// `[-4.0, 0.0, 4.0, 1.0]` hit the below-range, middle, above-range, and a
/// second middle point (chosen with exact rational gradients: `x=0` gives
/// `0.5`, `x=1` gives `5/6`).
#[test]
fn hard_swish_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![-4.0, 0.0, 4.0, 1.0], &[4]);
    let raw = x.tensor.hard_swish().expect("hard_swish must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::HardSwish, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    let expected = [0.0, 0.5, 1.0, 5.0 / 6.0];
    for (actual, expected) in grad_data.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "hard_swish gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

/// `log_softmax(x)_i = x_i - log(sum_k exp(x_k))` (axis `None` means "over
/// the flattened tensor", matching `Softmax`'s own convention — here the
/// input is already 1-D, so "flattened" and "the only axis" coincide).
/// `L = sum_i log_softmax(x)_i = sum_i x_i - n*log(sum_k exp(x_k))`, so
/// `dL/dx_j = 1 - n*softmax(x)_j`. Re-derived and cross-checked against a
/// central-difference numerical gradient in Python before writing this
/// test (they agreed to 1e-9): at `[1.0, 2.0, 3.0, 0.5]` the expected
/// gradient is approximately `[0.6585, 0.0718, -1.5232, 0.7929]`. The raw
/// forward value uses `tenflowers_core::ops::activation::log_softmax`,
/// which computes the same flatten-wide, max-subtraction-stabilized
/// formula `TrackedTensor::log_softmax(None)` computes internally for a
/// 1-D input.
#[test]
fn log_softmax_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![1.0, 2.0, 3.0, 0.5], &[4]);
    let raw =
        tenflowers_core::ops::activation::log_softmax(&x.tensor).expect("log_softmax must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::LogSoftmax { axis: None }, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    let expected = [0.6585244, 0.0717732, -1.5231822, 0.7928846];
    for (actual, expected) in grad_data.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "log_softmax gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

/// `log(x)`, `d/dx = 1/x`. Inputs are all strictly positive
/// (`1.0, 2.0, 4.0, 0.5`), away from the `x=0` singularity.
#[test]
fn log_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![1.0, 2.0, 4.0, 0.5], &[4]);
    let raw = x.tensor.log().expect("log must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Log, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(grad_data, vec![1.0, 0.5, 0.25, 2.0]);
}

/// `abs(x)`, `d/dx = sign(x)`. Inputs `[3.0, -2.0, 5.0, -7.0]` are all
/// clearly away from the `x=0` subgradient-ambiguous point, and mix
/// positive and negative values so the test cannot pass on an "always +1"
/// or "always -1" bug.
#[test]
fn abs_gradient_matches_expected() {
    reset_for_test();
    let x = make_tensor(vec![3.0, -2.0, 5.0, -7.0], &[4]);
    let raw = x.tensor.abs().expect("abs must succeed");
    let x = unary_op_then_sum_backward(x, UnaryOpKind::Abs, raw);

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(grad_data, vec![1.0, -1.0, 1.0, -1.0]);
}

/// `clamp(x, min, max)`; `d/dx = 1` inside `[min, max]`, `0` outside.
/// Inputs `[-3.0, 0.5, 2.0, -0.5]` against `min=-1.0, max=1.0` deliberately
/// mix two out-of-range values (`-3.0` below `min`, `2.0` above `max`) with
/// two in-range values (`0.5`, `-0.5`), so the test proves the clamp mask
/// genuinely gates the gradient rather than passing on a "clamp is a
/// no-op that always returns gradient 1" bug (which an all-in-range input
/// could not distinguish from correct behavior).
#[test]
fn clamp_gradient_matches_expected() {
    reset_for_test();
    let min = -1.0f32;
    let max = 1.0f32;
    let x = make_tensor(vec![-3.0, 0.5, 2.0, -0.5], &[4]);
    let raw = x.tensor.clamp(min, max).expect("clamp must succeed");
    let x = unary_op_then_sum_backward(
        x,
        UnaryOpKind::Clamp {
            min: Some(min),
            max: Some(max),
        },
        raw,
    );

    let grad = get_grad(&x).expect("grad must be available");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(
        grad_data,
        vec![0.0, 1.0, 0.0, 1.0],
        "clamp gradient must be 1 for in-range inputs and 0 for out-of-range \
         inputs — an all-1s result would indicate the clamp mask is broken"
    );
}

// ---------------------------------------------------------------------
// INVESTIGATION: many independent forward-only computations (never call
// `.backward()`) accumulating on the same thread before one real,
// unrelated `.backward()` call finally happens. Unlike every other test
// in this file, this deliberately does NOT call `reset_for_test()`
// between the "stale" forward-only computations and the final real one
// — the whole point is to observe what state genuinely persists across
// them on a single thread, exactly as pytest would run many independent
// test functions in one shared process/thread.
// ---------------------------------------------------------------------

/// Perform one forward-only computation — `mark_leaf` + a couple of
/// `record_and_link_binary`/`record_and_link_unary` calls on fresh
/// tensors — and deliberately never call `run_backward`. Every
/// `PyTensor` this function creates is dropped when it returns (none are
/// returned to the caller), matching a real "forward pass whose result
/// is simply never used for backward" pattern.
fn forward_only_never_backward(seed: f32) {
    let x = make_tensor(vec![seed, seed + 1.0, seed + 2.0], &[3]);
    mark_leaf(&x);
    let y = make_tensor(vec![100.0, 200.0, 300.0], &[3]);

    let raw = tenflowers_core::ops::add(&x.tensor, &y.tensor).expect("add must succeed");
    let sum_result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_binary(BinaryOpKind::Add, &x, &y, &sum_result)
        .expect("recording add must succeed");

    let raw2 = tenflowers_core::ops::mul(&sum_result.tensor, &sum_result.tensor)
        .expect("mul must succeed");
    let mul_result = PyTensor {
        tensor: Arc::new(raw2),
        requires_grad: true,
        is_pinned: false,
    };
    record_and_link_binary(BinaryOpKind::Mul, &sum_result, &sum_result, &mul_result)
        .expect("recording mul must succeed");

    // `x`, `y`, `sum_result`, `mul_result` all drop here. Nothing calls
    // `run_backward`, so nothing in TRACKED_REGISTRY / LEAVES /
    // IMPLICIT_TAPE / IDENTITY_ANCHORS that this call created is ever
    // cleared by this function.
}

/// The actual repro: run many independent forward-only computations
/// (never calling backward) on this thread, then perform ONE real,
/// unrelated tracked computation that DOES call `.backward()`, and
/// assert its gradient is exactly correct — not contaminated,
/// mis-attributed, or corrupted by the N accumulated stale forward-only
/// computations still sitting in the thread-local tables.
///
/// Also instruments (via plain assertions, not eprintln — this is meant
/// to stay in the suite, not be stripped as throwaway debug output)
/// whether TRACKED_REGISTRY / IDENTITY_ANCHORS entry counts grow by
/// approximately what N forward-only passes would be expected to add,
/// distinguishing "unbounded growth" (a leak) from "bounded/reused"
/// (not a leak) as its own, separate observation from the correctness
/// assertion.
#[test]
fn stale_forward_only_computations_do_not_corrupt_later_backward() {
    reset_for_test();

    let registry_len_before = TRACKED_REGISTRY.with(|r| r.borrow().len());
    let anchors_len_before = IDENTITY_ANCHORS.with(|a| a.borrow().len());
    assert_eq!(registry_len_before, 0, "must start from a clean registry");
    assert_eq!(anchors_len_before, 0, "must start from a clean anchor set");

    const N: usize = 1000;
    for i in 0..N {
        forward_only_never_backward(i as f32);
    }

    // Every forward-only pass registers 3 TRACKED_REGISTRY entries (leaf
    // x, sum_result, mul_result) and anchors all 3 (y is a "constant"
    // operand watched on the tape via record_and_link_binary, which does
    // NOT go through `register_tracked`/`anchor_identity` — only `x`,
    // `sum_result`, `mul_result` do). None of this is ever cleared
    // without a `run_backward` call, so growth must be *exactly*
    // proportional to N (bounded neither by a cap nor by reuse) —
    // confirming this is unbounded accumulation, not merely "some
    // growth that happens to plateau".
    let registry_len_after_stale = TRACKED_REGISTRY.with(|r| r.borrow().len());
    let anchors_len_after_stale = IDENTITY_ANCHORS.with(|a| a.borrow().len());
    assert_eq!(
        registry_len_after_stale,
        N * 3,
        "TRACKED_REGISTRY must grow by exactly 3 entries per forward-only \
         pass with no run_backward() ever called to clear it — confirms \
         unbounded accumulation rather than any bound/reuse policy"
    );
    assert_eq!(
        anchors_len_after_stale,
        N * 3,
        "IDENTITY_ANCHORS must grow in lockstep with TRACKED_REGISTRY for \
         the same reason"
    );
    let tape_len_after_stale = IMPLICIT_TAPE.with(GradientTape::len);
    assert_eq!(
        tape_len_after_stale,
        N * 2,
        "IMPLICIT_TAPE must accumulate exactly 2 recorded op nodes (add, \
         mul) per forward-only pass — the leaf `watch()` call does not \
         push a node, only record_op does"
    );
    let leaves_len_after_stale = LEAVES.with(|l| l.borrow().len());
    assert_eq!(
        leaves_len_after_stale, N,
        "LEAVES must accumulate exactly 1 stale leaf per forward-only pass"
    );

    // Now perform ONE real, completely unrelated tracked computation
    // that DOES call backward, sharing the same thread (and therefore
    // the same IMPLICIT_TAPE / TRACKED_REGISTRY / LEAVES /
    // IDENTITY_ANCHORS) as the N stale computations above.
    let x_real = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let y_real = make_tensor(vec![10.0, 20.0, 30.0], &[3]);
    let x_real = add_then_sum_backward(x_real, &y_real);

    // The real computation's gradient must be EXACTLY correct —
    // d(sum(x_real + y_real))/d(x_real) = 1 for every element — despite
    // 1000 accumulated, never-backward'd, unrelated computations sharing
    // every thread-local table it touches.
    let grad = get_grad(&x_real).expect("grad must be available for the real computation");
    let grad_data = grad.tensor.to_vec().expect("grad data must be readable");
    assert_eq!(
        grad_data,
        vec![1.0, 1.0, 1.0],
        "the real computation's gradient must be exactly correct and \
         completely unaffected by 1000 accumulated stale forward-only \
         computations that never called backward — if this fails with \
         anything other than [1, 1, 1], stale state is leaking into an \
         unrelated backward pass"
    );

    // None of the N stale leaves' identities must have spuriously
    // received a gradient from the real backward pass — every one of
    // them is unreachable from `x_real`'s target, so `get_grad` on any
    // of them must still cleanly error (not silently return an
    // unrelated tensor from the real computation, and not panic/crash).
    // Spot-check a handful spread across the run rather than all 1000,
    // to keep the test fast.
    for i in [0usize, 1, 500, 998, 999] {
        let stale_leaf_probe = make_tensor(vec![i as f32], &[1]);
        assert!(
            get_grad(&stale_leaf_probe).is_err(),
            "a freshly constructed, never-tracked probe tensor must not \
             spuriously resolve to a gradient left over from stale state"
        );
    }

    // `run_backward` unconditionally clears TRACKED_REGISTRY, LEAVES,
    // and IMPLICIT_TAPE (see that function's doc) — so after the one
    // real backward call above, ALL of it (both the real computation's
    // now-consumed graph AND the 1000 stale forward-only computations'
    // leftover entries) must be gone, not just the real computation's
    // own subset. This is the mechanism by which the stale accumulation
    // does not compound indefinitely across many independent
    // Python-level tests as long as *something*, eventually, calls
    // backward on this thread.
    let registry_len_after_real = TRACKED_REGISTRY.with(|r| r.borrow().len());
    assert_eq!(
        registry_len_after_real, 0,
        "run_backward's unconditional registry clear must sweep away \
         both the real computation's consumed graph and every stale \
         forward-only entry, since it does not distinguish between them"
    );
    let leaves_len_after_real = LEAVES.with(|l| l.borrow().len());
    assert_eq!(
        leaves_len_after_real, 0,
        "run_backward's unconditional leaves clear must likewise sweep \
         away the stale leaves alongside the real one"
    );
    let tape_len_after_real = IMPLICIT_TAPE.with(GradientTape::len);
    assert_eq!(
        tape_len_after_real, 0,
        "run_backward's IMPLICIT_TAPE.clear() must reset the tape fully"
    );

    // IDENTITY_ANCHORS survives pruned down to exactly what GRAD_STORE
    // still references: only x_real's key (the one leaf that actually
    // received a gradient in GRAD_STORE). Every stale forward-only
    // leaf's key never made it into GRAD_STORE (their gradients were
    // never computed — backward was never called on them), so their
    // anchors must all be pruned away by run_backward's retain-if-in-
    // GRAD_STORE pass, not left pinned forever.
    let anchors_len_after_real = IDENTITY_ANCHORS.with(|a| a.borrow().len());
    assert_eq!(
        anchors_len_after_real, 1,
        "IDENTITY_ANCHORS must be pruned down to exactly the one key \
         GRAD_STORE still references (x_real's), releasing every stale \
         forward-only computation's anchors even though run_backward was \
         never called on any of them directly"
    );
}
