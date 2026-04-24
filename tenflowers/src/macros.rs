//! Convenience macros for the TenfloweRS framework.
//!
//! This module provides macros that simplify common tensor construction patterns,
//! reducing boilerplate and improving ergonomics when working with tensors.
//!
//! # Overview
//!
//! The primary macro is `tensor!`, which constructs a [`tenflowers_core::Tensor`] from
//! literal values with shape inferred from nesting depth:
//!
//! - 1-D: `tensor![1.0f32, 2.0, 3.0]`
//! - 2-D: `tensor![[1.0f32, 2.0], [3.0, 4.0]]`
//! - Explicit dtype: `tensor![dtype = f64; 1.0, 2.0, 3.0]`

/// Creates a [`Tensor`](tenflowers_core::Tensor) from literal values.
///
/// The element type is inferred from the first value; use an explicit type suffix
/// (e.g. `1.0_f32`) or the `dtype = <ty>;` prefix form to override.
///
/// # Arms
///
/// | Syntax | Result |
/// |--------|--------|
/// | `tensor![]` | Empty 1-D `Tensor<f32>` with shape `[0]` |
/// | `tensor![v0, v1, …]` | 1-D tensor, shape `[n]` |
/// | `tensor![[r0, r1], [r2, r3]]` | 2-D tensor, shape `[rows, cols]` |
/// | `tensor![dtype = T; v0, v1, …]` | 1-D tensor of type `T`, shape `[n]` |
/// | `tensor![dtype = T; [r0, r1], [r2, r3]]` | 2-D tensor of type `T`, shape `[rows, cols]` |
///
/// # Examples
///
/// ```rust
/// use tenflowers::tensor;
/// use tenflowers::prelude::Tensor;
///
/// // 1-D vector
/// let v = tensor![1.0_f32, 2.0, 3.0];
/// assert_eq!(v.shape().dims(), &[3]);
///
/// // integer values
/// let iv = tensor![0i32, 1, 2, 3, 4];
/// assert_eq!(iv.shape().dims(), &[5]);
///
/// // 2-D matrix
/// let m = tensor![[1.0_f32, 2.0], [3.0, 4.0]];
/// assert_eq!(m.shape().dims(), &[2, 2]);
///
/// // explicit dtype
/// let dv = tensor![dtype = f64; 1.0, 2.0, 3.0];
/// assert_eq!(dv.shape().dims(), &[3]);
/// ```
///
/// # Panics
///
/// The macro panics if:
/// - In the 2-D form, rows have unequal lengths.
/// - (Internally impossible) the inferred shape does not match the data length.
#[macro_export]
macro_rules! tensor {
    // -----------------------------------------------------------------------
    // Empty tensor — 1-D tensor with 0 elements.
    // -----------------------------------------------------------------------
    () => {{
        $crate::core::Tensor::<f32>::from_data(vec![], &[0usize])
            .expect("failed to create empty tensor from macro")
    }};

    // -----------------------------------------------------------------------
    // dtype = T ; 2-D nested rows
    // e.g. tensor![dtype = f64; [1.0, 2.0], [3.0, 4.0]]
    // -----------------------------------------------------------------------
    (dtype = $ty:ty ; $([$($col:expr),+ $(,)?]),+ $(,)?) => {{
        let rows: Vec<Vec<$ty>> = vec![$( vec![$($col as $ty),+] ),+];
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        for row in rows.iter() {
            assert_eq!(
                row.len(),
                n_cols,
                "tensor! macro: all rows must have equal length (expected {}, got {})",
                n_cols,
                row.len(),
            );
        }
        let flat: Vec<$ty> = rows.into_iter().flatten().collect();
        $crate::core::Tensor::<$ty>::from_data(flat, &[n_rows, n_cols])
            .expect("failed to create 2-D tensor from macro: internal shape mismatch")
    }};

    // -----------------------------------------------------------------------
    // dtype = T ; flat list
    // e.g. tensor![dtype = f64; 1.0, 2.0, 3.0]
    // -----------------------------------------------------------------------
    (dtype = $ty:ty ; $($val:expr),+ $(,)?) => {{
        let data: Vec<$ty> = vec![$($val as $ty),+];
        let len = data.len();
        $crate::core::Tensor::<$ty>::from_data(data, &[len])
            .expect("failed to create tensor from macro: internal shape mismatch")
    }};

    // -----------------------------------------------------------------------
    // 2-D nested rows (inferred type)
    // e.g. tensor![[1.0f32, 2.0], [3.0, 4.0]]
    // -----------------------------------------------------------------------
    ($([$($col:expr),+ $(,)?]),+ $(,)?) => {{
        let rows: Vec<Vec<_>> = vec![$( vec![$($col),+] ),+];
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        for row in rows.iter() {
            assert_eq!(
                row.len(),
                n_cols,
                "tensor! macro: all rows must have equal length (expected {}, got {})",
                n_cols,
                row.len(),
            );
        }
        let flat: Vec<_> = rows.into_iter().flatten().collect();
        $crate::core::Tensor::from_data(flat, &[n_rows, n_cols])
            .expect("failed to create 2-D tensor from macro: internal shape mismatch")
    }};

    // -----------------------------------------------------------------------
    // 1-D: one or more comma-separated values (inferred type)
    // e.g. tensor![1.0f32, 2.0, 3.0]
    // -----------------------------------------------------------------------
    ($($val:expr),+ $(,)?) => {{
        let data: Vec<_> = vec![$($val),+];
        let len = data.len();
        $crate::core::Tensor::from_data(data, &[len])
            .expect("failed to create tensor from macro: internal shape mismatch")
    }};
}
