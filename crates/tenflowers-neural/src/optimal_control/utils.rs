//! Matrix/vector utility functions and Box-Muller normal sampling.

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// Matrix / Vector Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Add two matrices element-wise: C = A + B.
pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

/// Subtract two matrices element-wise: C = A - B.
pub fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x - y).collect())
        .collect()
}

/// Multiply two matrices: C = A * B (standard matrix multiplication).
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_b = if b.is_empty() { 0 } else { b[0].len() };
    let cols_a = if a.is_empty() { 0 } else { a[0].len() };
    let mut c = vec![vec![0.0f64; cols_b]; rows_a];
    for i in 0..rows_a {
        for k in 0..cols_a {
            for j in 0..cols_b {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Transpose a matrix: B = A^T.
pub fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Scale a matrix by a scalar: B = s * A.
pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|x| x * s).collect())
        .collect()
}

/// Multiply a matrix by a column vector: y = A * x.
pub fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a_ij, x_j)| a_ij * x_j).sum())
        .collect()
}

/// Add two vectors element-wise: c = a + b.
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Scale a vector: b = s * a.
pub fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

/// Dot product of two vectors.
pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a vector.
pub fn vec_norm(v: &[f64]) -> f64 {
    vec_dot(v, v).sqrt()
}

/// Create an identity matrix of size n × n.
pub fn mat_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Frobenius norm of a matrix: sqrt(sum of squared elements).
pub fn mat_frob_norm(m: &[Vec<f64>]) -> f64 {
    m.iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Invert a symmetric positive-definite (SPD) matrix via Cholesky decomposition.
///
/// Returns `Err` if the matrix is not positive definite or if dimensions are inconsistent.
///
/// The algorithm:
/// 1. Cholesky factorization: A = L * L^T
/// 2. Forward substitution to find L^{-1}
/// 3. A^{-1} = (L^{-1})^T * L^{-1}
pub fn mat_sym_pos_def_inv(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Err(TensorError::invalid_argument(
            "mat_sym_pos_def_inv: empty matrix".to_string(),
        ));
    }
    for row in a {
        if row.len() != n {
            return Err(TensorError::invalid_argument(
                "mat_sym_pos_def_inv: matrix must be square".to_string(),
            ));
        }
    }

    // Cholesky: L is lower-triangular with A = L L^T
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(TensorError::numerical_error(
                        "mat_sym_pos_def_inv",
                        "Matrix is not positive definite",
                        vec![
                            "Add regularization (e.g., μI) to ensure positive definiteness"
                                .to_string(),
                        ],
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Compute L^{-1} via forward substitution (lower-triangular inverse).
    let mut l_inv = mat_identity(n);
    for j in 0..n {
        for i in j..n {
            let mut sum = l_inv[i][j];
            for k in j..i {
                sum -= l[i][k] * l_inv[k][j];
            }
            l_inv[i][j] = sum / l[i][i];
        }
    }

    // A^{-1} = L_inv^T * L_inv
    let l_inv_t = mat_transpose(&l_inv);
    Ok(mat_mul(&l_inv_t, &l_inv))
}

// ─────────────────────────────────────────────────────────────────────────────
// Box-Muller Normal Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` i.i.d. N(0,1) samples via Box-Muller transform from seeded RNG.
pub fn normal_samples_f64(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = (rng.random::<f64>()).max(1e-12);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push(r * theta.cos());
        i += 1;
        if i < n {
            out.push(r * theta.sin());
            i += 1;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
