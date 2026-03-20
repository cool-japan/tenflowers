//! Convenience macros for the TenfloweRS framework.
//!
//! This module provides macros that simplify common tensor construction patterns,
//! reducing boilerplate and improving ergonomics when working with tensors.

/// Creates a 1D \[`Tensor`\] from a list of values.
///
/// This macro accepts a comma-separated list of values and produces a 1D tensor
/// by delegating to [`tenflowers_core::Tensor::from_data`].  The element type is
/// inferred from the values; an explicit type suffix (e.g. `1.0f32`) may be used
/// to disambiguate.
///
/// # Examples
///
/// ```rust
/// use tenflowers::tensor;
/// use tenflowers::prelude::Tensor;
///
/// let t = tensor![1.0f32, 2.0, 3.0];
/// assert_eq!(t.shape().dims(), &[3]);
/// ```
///
/// ```rust
/// use tenflowers::tensor;
/// use tenflowers::prelude::Tensor;
///
/// let t = tensor![0i32, 1, 2, 3, 4];
/// assert_eq!(t.shape().dims(), &[5]);
/// ```
///
/// # Errors
///
/// Because the underlying \[`Tensor::from_data`\] call validates the shape, the
/// macro **panics** on a length/shape mismatch — but since the shape is derived
/// directly from the supplied values this cannot happen in practice.
#[macro_export]
macro_rules! tensor {
    // Empty tensor — 1D tensor with 0 elements.
    () => {{
        $crate::core::Tensor::from_data(vec![], &[0])
            .expect("failed to create empty tensor from macro")
    }};

    // One or more comma-separated values.
    ($($val:expr),+ $(,)?) => {{
        let data: Vec<_> = vec![$($val),+];
        let len = data.len();
        $crate::core::Tensor::from_data(data, &[len])
            .expect("failed to create tensor from macro: internal shape mismatch")
    }};
}
