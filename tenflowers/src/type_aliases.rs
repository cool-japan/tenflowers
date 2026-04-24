//! Common-shape type aliases for [`Tensor`].
//!
//! These aliases are **documentation-only**. The underlying type is the
//! dynamic-rank [`Tensor<T>`](tenflowers_core::Tensor); no compile-time shape
//! enforcement is performed. Names reflect _intent_ (e.g. "matrix" implies 2-D
//! storage), not a distinct type — they are purely ergonomic shorthands.
//!
//! # Example
//!
//! ```rust
//! use tenflowers::type_aliases::*;
//! use tenflowers::prelude::Tensor;
//!
//! let m: Matrix = Tensor::zeros(&[3, 4]);
//! assert_eq!(m.shape().dims(), &[3, 4]);
//! ```

use tenflowers_core::Tensor;

// ── Generic aliases ──────────────────────────────────────────────────────────

/// A 1-dimensional tensor (vector shape).
///
/// The name signals intent; the underlying type is the dynamic-rank
/// [`Tensor<T>`](tenflowers_core::Tensor).
pub type Tensor1D<T> = Tensor<T>;

/// A 2-dimensional tensor (matrix shape).
///
/// The name signals intent; the underlying type is the dynamic-rank
/// [`Tensor<T>`](tenflowers_core::Tensor).
pub type Tensor2D<T> = Tensor<T>;

/// A 3-dimensional tensor.
///
/// The name signals intent; the underlying type is the dynamic-rank
/// [`Tensor<T>`](tenflowers_core::Tensor).
pub type Tensor3D<T> = Tensor<T>;

/// A 4-dimensional tensor (batch × channels × height × width in vision).
///
/// The name signals intent; the underlying type is the dynamic-rank
/// [`Tensor<T>`](tenflowers_core::Tensor).
pub type Tensor4D<T> = Tensor<T>;

// ── f32 convenience aliases ──────────────────────────────────────────────────

/// A 1-D vector alias using `f32`.
pub type Vector = Tensor<f32>;

/// A 2-D matrix alias using `f32`.
pub type Matrix = Tensor<f32>;

/// A batch tensor alias using `f32` (first dimension is the batch).
pub type BatchTensor = Tensor<f32>;

/// A 1-D tensor of `f32`.
pub type Tensor1Df32 = Tensor<f32>;

/// A 2-D tensor of `f32`.
///
/// # Example
///
/// ```rust
/// use tenflowers::type_aliases::Tensor2Df32;
/// use tenflowers::prelude::Tensor;
///
/// let mat: Tensor2Df32 = Tensor::zeros(&[2, 3]);
/// assert_eq!(mat.shape().dims(), &[2, 3]);
/// ```
pub type Tensor2Df32 = Tensor<f32>;

/// A 3-D tensor of `f32`.
pub type Tensor3Df32 = Tensor<f32>;

/// A 4-D tensor of `f32` (e.g. NCHW image batch).
pub type Tensor4Df32 = Tensor<f32>;

// ── f64 convenience aliases ──────────────────────────────────────────────────

/// A 1-D tensor of `f64`.
pub type Tensor1Df64 = Tensor<f64>;

/// A 2-D tensor of `f64`.
pub type Tensor2Df64 = Tensor<f64>;

/// A 3-D tensor of `f64`.
pub type Tensor3Df64 = Tensor<f64>;

/// A 4-D tensor of `f64`.
pub type Tensor4Df64 = Tensor<f64>;
