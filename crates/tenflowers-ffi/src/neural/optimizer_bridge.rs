//! Generic bridge from an arbitrary Python model object to its trainable
//! [`PyParameter`] handles.
//!
//! # Why this module exists
//!
//! Every optimizer in [`super::optimizers`] / [`super::extended_optimizers`]
//! needs to accept **any** Python model object — `PyDense`, `PySequential`,
//! or a pure-Python class a user wrote themselves — and pull its trainable
//! parameters out of it, without knowing the model's concrete Rust type at
//! compile time (there is no such single type: a user-defined Python model
//! is not a `#[pyclass]` at all). The only thing every model has in common is
//! a `.parameters()` method, exactly mirroring how real PyTorch's
//! `optimizer = Adam(model.parameters())` /
//! `for p in model.parameters(): ...` works. [`collect_parameters`] is the
//! one place that performs this dynamic `.parameters()` dispatch, so an
//! optimizer's `step()` / `zero_grad()` can be written once against
//! `Vec<Py<PyParameter>>` instead of every optimizer re-implementing its own
//! ad hoc Python introspection.
//!
//! # Relationship to [`super::layers`]
//!
//! [`PyParameter`] — the stable-identity, mutable-in-place parameter cell
//! that an optimizer's `.step()` needs in order to call
//! [`PyParameter::set_data`] — lives in [`super::layers`] and is treated as
//! already complete here (this module adds no methods to it). What this
//! module resolves instead is the *collection* problem: given a
//! `Bound<'_, PyAny>` model of unknown concrete type, produce a
//! `Vec<Py<PyParameter>>` an optimizer can iterate.

use super::layers::PyParameter;
use pyo3::prelude::*;

/// Call `model.parameters()` and extract every element of the returned
/// Python iterable as a [`PyParameter`] handle.
///
/// This is the generic bridge a later optimizer implementation's `step()` /
/// `zero_grad()` should call to turn a `model: Bound<'_, PyAny>` argument
/// into a `Vec<Py<PyParameter>>` it can iterate: read each parameter's
/// current value via [`PyParameter::to_tensor`], its gradient via
/// [`PyParameter::grad`], write an update back via [`PyParameter::set_data`],
/// and clear the gradient via [`PyParameter::zero_grad`].
///
/// # Why `model.call_method0("parameters")` and not a Rust trait
///
/// `tenflowers-ffi` has no single Rust trait implemented by every possible
/// Python-visible "model" (a pure-Python class a user writes is not a
/// `#[pyclass]` at all, so it cannot implement one). Standard PyO3 dynamic
/// method dispatch — call a named method on a `Bound<'_, PyAny>` and see
/// what comes back — is the only mechanism that works uniformly for
/// `PyDense`, `PySequential`, and an arbitrary user-defined Python model
/// alike, exactly matching how real PyTorch resolves `model.parameters()` at
/// the Python level rather than through a shared base class.
///
/// # Resolved: every built-in layer's `.parameters()` returns `Vec<PyParameter>`
///
/// Earlier in this crate's development, [`super::layers::PyDense::parameters`]
/// and [`super::layers::PySequential::parameters`] returned
/// `Vec<`[`crate::tensor_ops::PyTensor`]`>`, **not** `Vec<PyParameter>` —
/// because `PyParameter` did not yet support in-place mutation
/// ([`PyParameter::set_data`]). A bare [`crate::tensor_ops::PyTensor`] cannot
/// serve as a mutable optimizer target: it has no `set_data`, and its
/// identity *is* its `Arc` pointer, which changes on every new computation —
/// there is no stable cell an optimizer could write an in-place update into.
///
/// That gap has since been closed: every built-in layer's `.parameters()` —
/// `PyDense`, `PySequential`, and every layer in `neural/conv_layers`,
/// `neural/normalization`, `neural/embedding`, `neural/attention`,
/// `neural/transformer`, and `neural/recurrent` — now returns
/// `Vec<Py<PyParameter>>`, each handle sharing stable identity with the
/// owning layer's own weight/bias cell (see [`super::layers::PyDense::parameters`]'s
/// doc for the full identity-sharing guarantee that makes an optimizer's
/// `.set_data()` call visible on the very next `.forward()`). A bare
/// `PyTensor`-returning `.parameters()` is no longer produced by any
/// built-in layer.
///
/// The type-mismatch handling below remains as defensive code for a
/// hypothetical third-party Python model whose hand-written `.parameters()`
/// yields a `PyTensor` (or any other non-`PyParameter` value) by mistake:
/// this function returns a clear, typed [`pyo3::exceptions::PyTypeError`]
/// naming the offending index and explaining what is required, rather than
/// silently skipping that element (which would make an optimizer run a
/// no-op step over part of the model) or fabricating a throwaway
/// `PyParameter` around the tensor (a synthetic identity that can never
/// receive a gradient, since [`crate::implicit_autograd`] recorded the
/// backward pass against the *original* `PyTensor`'s `Arc` identity, not
/// this new wrapper's).
///
/// # Errors
///
/// Returns `Err` (never panics) when:
///
/// * `model` has no callable `.parameters()` attribute, or calling it raises
///   in Python — the underlying [`PyErr`] is propagated as-is so the
///   caller/user sees the original Python traceback/message.
/// * `.parameters()`'s return value is not iterable — propagated as-is from
///   [`PyAnyMethods::try_iter`].
/// * Iterating raises partway through — propagated as-is.
/// * Any yielded element is not a [`PyParameter`] instance. If it *is* a
///   [`crate::tensor_ops::PyTensor`] — which no built-in layer produces any
///   more, but a hand-written third-party Python model theoretically could —
///   the error message names that specifically and explains the fix; for any
///   other type, a generic type-mismatch message is used instead.
pub fn collect_parameters(model: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyParameter>>> {
    let params_obj = model.call_method0("parameters").map_err(|err| {
        pyo3::exceptions::PyAttributeError::new_err(format!(
            "collect_parameters: failed to call `.parameters()` on the given model object \
             (type '{}'); an optimizer requires the model to expose a `.parameters()` method \
             returning an iterable of Parameter objects, mirroring PyTorch's \
             `model.parameters()`. Underlying error: {}",
            model.get_type().name().map_or_else(
                |_| "<unknown>".to_string(),
                |name| name.to_string()
            ),
            err
        ))
    })?;

    let iterator = params_obj.try_iter().map_err(|err| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "collect_parameters: `.parameters()` returned a value of type '{}' that is not \
             iterable; expected an iterable (e.g. list) of Parameter objects. Underlying \
             error: {}",
            params_obj.get_type().name().map_or_else(
                |_| "<unknown>".to_string(),
                |name| name.to_string()
            ),
            err
        ))
    })?;

    let mut collected: Vec<Py<PyParameter>> = Vec::new();
    for (index, item_result) in iterator.enumerate() {
        let item = item_result.map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "collect_parameters: iterating `.parameters()`'s return value failed at \
                 index {}: {}",
                index, err
            ))
        })?;

        match item.extract::<Py<PyParameter>>() {
            Ok(param) => collected.push(param),
            Err(extract_err) => {
                // Distinguish the specific, expected-to-happen-today case
                // (a PyTensor, from PyDense/PySequential's current
                // `Vec<PyTensor>` return type) from any other unexpected
                // element type, so the error message tells the caller
                // exactly what is wrong and how to fix it.
                if item.extract::<Py<crate::tensor_ops::PyTensor>>().is_ok() {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "collect_parameters: `.parameters()` element at index {} is a Tensor, \
                         not a Parameter. This layer's `.parameters()` must be updated to \
                         return Parameter objects (with a stable, mutable identity an \
                         optimizer can write updates into via `set_data`) before it can be \
                         used with an optimizer; a plain Tensor's identity changes on every \
                         new computation and cannot be optimized in place.",
                        index
                    )));
                }

                let item_type_name = item
                    .get_type()
                    .name()
                    .map_or_else(|_| "<unknown>".to_string(), |name| name.to_string());
                Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "collect_parameters: `.parameters()` element at index {} has type '{}', \
                     which is not a Parameter object. Underlying error: {}",
                    index, item_type_name, extract_err
                )))?
            }
        }
    }

    Ok(collected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::PyTensor;
    use pyo3::IntoPyObjectExt;
    use std::ffi::CString;

    /// Build a throwaway Python module from source, returning the module's
    /// `__dict__`-resolved attribute `attr_name` (typically a class or
    /// factory function) as a `Py<PyAny>` the test can then call/instantiate
    /// under its own `Python::attach` scope.
    fn load_py_attr(py: Python<'_>, code: &str, attr_name: &str) -> PyResult<Py<PyAny>> {
        let code = CString::new(code).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("test source had a NUL byte: {}", e))
        })?;
        let module = PyModule::from_code(
            py,
            code.as_c_str(),
            c"collect_parameters_test_module.py",
            c"collect_parameters_test_module",
        )?;
        module.getattr(attr_name).map(Bound::unbind)
    }

    #[test]
    fn extracts_parameters_from_a_model_exposing_real_parameter_objects() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            // A pure-Python class exposing `.parameters()` -> list[PyParameter],
            // exactly like a real user model would. Note this test module
            // never subclasses any tenflowers type — the whole point of
            // `collect_parameters` is that it works via duck typing.
            let make_model = load_py_attr(
                py,
                r#"
class FakeModel:
    def __init__(self, params):
        self._params = params
    def parameters(self):
        return list(self._params)
"#,
                "FakeModel",
            )
            .expect("test module should define FakeModel");

            let w = Py::new(
                py,
                PyParameter::new(
                    PyTensor {
                        tensor: std::sync::Arc::new(
                            tenflowers_core::Tensor::<f32>::zeros(&[2, 2]),
                        ),
                        requires_grad: true,
                        is_pinned: false,
                    },
                    Some(true),
                ),
            )
            .expect("PyParameter should construct");
            let b = Py::new(
                py,
                PyParameter::new(
                    PyTensor {
                        tensor: std::sync::Arc::new(tenflowers_core::Tensor::<f32>::zeros(&[2])),
                        requires_grad: true,
                        is_pinned: false,
                    },
                    Some(true),
                ),
            )
            .expect("PyParameter should construct");

            let params_list = pyo3::types::PyList::new(py, [w.clone_ref(py), b.clone_ref(py)])
                .expect("list construction should not fail for two valid items");
            let model = make_model
                .bind(py)
                .call1((params_list,))
                .expect("FakeModel(...) should construct");

            let collected =
                collect_parameters(&model).expect("collect_parameters should succeed");

            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].borrow(py).id(), w.borrow(py).id());
            assert_eq!(collected[1].borrow(py).id(), b.borrow(py).id());
            assert_eq!(collected[0].borrow(py).shape(), vec![2, 2]);
            assert_eq!(collected[1].borrow(py).shape(), vec![2]);
        });
    }

    #[test]
    fn extracts_zero_parameters_from_a_model_with_an_empty_parameter_list() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            let make_model = load_py_attr(
                py,
                r#"
class EmptyModel:
    def parameters(self):
        return []
"#,
                "EmptyModel",
            )
            .expect("test module should define EmptyModel");

            let model = make_model
                .bind(py)
                .call0()
                .expect("EmptyModel() should construct");

            let collected =
                collect_parameters(&model).expect("collect_parameters should succeed");
            assert!(collected.is_empty());
        });
    }

    #[test]
    fn errors_clearly_when_model_has_no_parameters_method() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            let make_model = load_py_attr(
                py,
                r#"
class NoParametersModel:
    pass
"#,
                "NoParametersModel",
            )
            .expect("test module should define NoParametersModel");

            let model = make_model
                .bind(py)
                .call0()
                .expect("NoParametersModel() should construct");

            let err = collect_parameters(&model)
                .expect_err("collect_parameters must fail, not panic, on a missing method");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py),
                "expected PyAttributeError, got: {:?}",
                err
            );
            let message = err.to_string();
            assert!(
                message.contains("parameters"),
                "error message should mention `.parameters()`: {}",
                message
            );
        });
    }

    #[test]
    fn errors_clearly_when_parameters_returns_tensors_instead_of_parameters() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            // Mirrors PyDense/PySequential's CURRENT (pre-upgrade) reality:
            // `.parameters()` returning `Vec<PyTensor>` rather than
            // `Vec<PyParameter>`.
            let make_model = load_py_attr(
                py,
                r#"
class TensorReturningModel:
    def __init__(self, tensors):
        self._tensors = tensors
    def parameters(self):
        return list(self._tensors)
"#,
                "TensorReturningModel",
            )
            .expect("test module should define TensorReturningModel");

            let t = Py::new(
                py,
                PyTensor {
                    tensor: std::sync::Arc::new(tenflowers_core::Tensor::<f32>::zeros(&[3, 3])),
                    requires_grad: true,
                    is_pinned: false,
                },
            )
            .expect("PyTensor should construct");

            let tensors_list = pyo3::types::PyList::new(py, [t])
                .expect("list construction should not fail for one valid item");
            let model = make_model
                .bind(py)
                .call1((tensors_list,))
                .expect("TensorReturningModel(...) should construct");

            let err = collect_parameters(&model).expect_err(
                "collect_parameters must fail, not silently succeed, on Tensor elements",
            );
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "expected PyTypeError, got: {:?}",
                err
            );
            let message = err.to_string();
            assert!(
                message.contains("Parameter") && message.contains("Tensor"),
                "error message should explain the Tensor-vs-Parameter mismatch: {}",
                message
            );
        });
    }

    #[test]
    fn errors_clearly_when_parameters_returns_a_non_iterable() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            let make_model = load_py_attr(
                py,
                r#"
class NonIterableParametersModel:
    def parameters(self):
        return 42
"#,
                "NonIterableParametersModel",
            )
            .expect("test module should define NonIterableParametersModel");

            let model = make_model
                .bind(py)
                .call0()
                .expect("NonIterableParametersModel() should construct");

            let err = collect_parameters(&model)
                .expect_err("collect_parameters must fail, not panic, on a non-iterable return");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "expected PyTypeError, got: {:?}",
                err
            );
        });
    }

    #[test]
    fn errors_clearly_when_parameters_method_itself_raises() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            let make_model = load_py_attr(
                py,
                r#"
class RaisingModel:
    def parameters(self):
        raise ValueError("boom")
"#,
                "RaisingModel",
            )
            .expect("test module should define RaisingModel");

            let model = make_model
                .bind(py)
                .call0()
                .expect("RaisingModel() should construct");

            let err = collect_parameters(&model).expect_err(
                "collect_parameters must propagate a Python-side exception, not panic",
            );
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py),
                "expected PyAttributeError wrapping the underlying ValueError, got: {:?}",
                err
            );
            let message = err.to_string();
            assert!(
                message.contains("boom"),
                "error message should surface the underlying Python exception message: {}",
                message
            );
        });
    }

    #[test]
    fn errors_clearly_on_a_mixed_list_naming_the_offending_index() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `optimizers.rs`, `normalization.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            let make_model = load_py_attr(
                py,
                r#"
class MixedModel:
    def __init__(self, items):
        self._items = items
    def parameters(self):
        return list(self._items)
"#,
                "MixedModel",
            )
            .expect("test module should define MixedModel");

            let good = Py::new(
                py,
                PyParameter::new(
                    PyTensor {
                        tensor: std::sync::Arc::new(tenflowers_core::Tensor::<f32>::zeros(&[1])),
                        requires_grad: true,
                        is_pinned: false,
                    },
                    Some(true),
                ),
            )
            .expect("PyParameter should construct");

            // Second element is an int, not a PyParameter or a PyTensor:
            // exercises the generic (non-Tensor) type-mismatch branch.
            let mixed_list = pyo3::types::PyList::new(
                py,
                [good.into_bound_py_any(py).expect("Parameter bind should not fail"),
                 42i64.into_bound_py_any(py).expect("int bind should not fail")],
            )
            .expect("list construction should not fail for two valid items");
            let model = make_model
                .bind(py)
                .call1((mixed_list,))
                .expect("MixedModel(...) should construct");

            let err = collect_parameters(&model)
                .expect_err("collect_parameters must fail on a non-Parameter, non-Tensor element");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "expected PyTypeError, got: {:?}",
                err
            );
            let message = err.to_string();
            assert!(
                message.contains("index 1"),
                "error message should name the offending index: {}",
                message
            );
        });
    }
}
