//! Linear Algebra Operations Module
//!
//! This module contains gradient operations for linear algebra functions including:
//! - Eigenvalue decomposition (eig)
//! - Singular Value Decomposition (SVD)
//! - Matrix inverse (inv)
//! - Matrix determinant (det)
//! - Pseudoinverse (pinv)
//! - Cholesky decomposition
//! - LU decomposition

mod basic_ops;
mod decompositions;
mod einsum;

// Re-export decomposition operations
pub use decompositions::{cholesky_backward, eig_backward, lu_backward, svd_backward};

// Re-export basic operations
pub use basic_ops::{det_backward, inv_backward, matmul_backward, pinv_backward};

// Re-export einsum operations
pub use einsum::einsum_backward;
