//! Gradient Executor Trait — Dependency-Inversion Bridge
//!
//! `tenflowers-core`'s operation registry only stores gradient function names as
//! metadata (e.g. `grad_fn: Option<String>`); it never stores callable
//! forward/backward closures. Real forward+backward execution lives one layer up,
//! in `tenflowers-autograd`'s `GradientTape` -- which depends on `tenflowers-core`,
//! not the other way around, so `tenflowers-core` cannot call into it directly.
//!
//! This module defines a small trait, [`GradientExecutor`], that lets
//! `tenflowers-core` call into a real gradient implementation without depending on
//! `tenflowers-autograd`: a higher-level crate registers an implementation via
//! [`register_gradient_executor`] at an appropriate initialization point (for
//! example, `tenflowers_autograd::init()`). [`crate::gradient_validation_framework`]
//! consults [`get_gradient_executor`] and, if nothing has been registered, returns
//! an honest "cannot verify" status rather than fabricating a `passed: true`
//! result.

use crate::{Result, Tensor};
use std::sync::OnceLock;

/// Bridges `tenflowers-core`'s gradient validation framework to a real
/// forward+backward implementation living in a higher-level crate (normally
/// `tenflowers-autograd`).
///
/// Implementations always operate in `f64`, regardless of the dtype under test in
/// [`crate::gradient_validation_framework::GradientTestCase`]: `f64` has enough
/// precision to be a safe universal choice for a validation/testing framework,
/// and keeping the trait monomorphic avoids needing a type-erased/generic trait
/// object design for this bridge.
pub trait GradientExecutor {
    /// Compute the gradient(s) of `op` applied to `inputs`, with respect to each
    /// element of `inputs`, in order.
    ///
    /// Returns one gradient tensor per input, each with the same shape as its
    /// corresponding input tensor. Returns `Err` if `op` is not recognized by
    /// this executor, or if the gradient computation itself fails -- never
    /// fabricates a placeholder result.
    fn compute_gradient(&self, op: &str, inputs: &[Tensor<f64>]) -> Result<Vec<Tensor<f64>>>;
}

static GRADIENT_EXECUTOR: OnceLock<Box<dyn GradientExecutor + Send + Sync>> = OnceLock::new();

/// Register the global [`GradientExecutor`].
///
/// Intended to be called once, early in application or test startup (for example
/// from `tenflowers_autograd::init()`). Idempotent: if an executor has already
/// been registered, this call is a silent no-op and the original registration is
/// kept (matching [`OnceLock`]'s first-write-wins semantics).
pub fn register_gradient_executor(executor: Box<dyn GradientExecutor + Send + Sync>) {
    let _ = GRADIENT_EXECUTOR.set(executor);
}

/// Get the globally registered [`GradientExecutor`], if any has been registered.
///
/// Returns `None` if no executor has been registered yet -- callers must not
/// treat this as "the property holds"; see
/// [`crate::gradient_validation_framework`] for how this is surfaced honestly.
pub fn get_gradient_executor() -> Option<&'static (dyn GradientExecutor + Send + Sync)> {
    GRADIENT_EXECUTOR.get().map(|b| b.as_ref())
}
