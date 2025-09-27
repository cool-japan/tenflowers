//! Advanced Linear Algebra Operations
//!
//! This module provides GPU implementations of advanced linear algebra operations
//! including eigenvalue computation, matrix inversion, linear system solving,
//! and determinant calculation.

pub mod determinant;
pub mod eigenvalues;
pub mod linear_solver;
pub mod matrix_inverse;

// Re-export public functions for convenience
pub use determinant::determinant;
pub use eigenvalues::eigenvalues;
pub use linear_solver::solve;
pub use matrix_inverse::inverse;