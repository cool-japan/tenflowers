//! Linear-algebra helpers (private, self-contained)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ============================================================================
// Linear-algebra helpers (private, self-contained)
// ============================================================================

/// (m x k) times (k x n)
pub(super) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += aip * b[p][j];
            }
        }
    }
    c
}

pub(super) fn mat_t(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0_f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Solve least-squares  min ||Ax - b||_2  via normal equations (A^T A) x = A^T b.
/// Returns x as a column vector (Vec<f64>).
pub(super) fn solve_lstsq(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let at = mat_t(a);
    let ata = mat_mul(&at, a);
    let atb: Vec<f64> = at
        .iter()
        .map(|row| row.iter().zip(b.iter()).map(|(ri, bi)| ri * bi).sum())
        .collect();
    let n = ata.len();
    // Gaussian elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = ata
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(atb[i]);
            r
        })
        .collect();
    for col in 0..n {
        // pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(TensorError::compute_error_simple(
                "Singular matrix in least-squares solve".to_string(),
            ));
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    Ok(aug.iter().map(|r| r[n]).collect())
}

/// Khatri-Rao product (column-wise Kronecker) of two matrices.
/// A is (I x R), B is (J x R)  ->  (IJ x R).
pub(super) fn khatri_rao(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let i_dim = a.len();
    let j_dim = b.len();
    let r = a[0].len();
    let mut out = vec![vec![0.0_f64; r]; i_dim * j_dim];
    for col in 0..r {
        for ia in 0..i_dim {
            for jb in 0..j_dim {
                out[ia * j_dim + jb][col] = a[ia][col] * b[jb][col];
            }
        }
    }
    out
}

/// Frobenius norm of a flat vector.
pub(super) fn frobenius(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Power-iteration SVD: returns (U, S, Vt) with `k` leading singular components.
/// `mat` is m x n.  U is m x k, S is length k, Vt is k x n.
pub(super) fn truncated_svd(
    mat: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    if mat.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let m = mat.len();
    let n = mat[0].len();
    let rank = k.min(m).min(n);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(rank);
    let mut sigmas: Vec<f64> = Vec::with_capacity(rank);
    let mut vt_rows: Vec<Vec<f64>> = Vec::with_capacity(rank);

    // Deflated power iteration
    let mut residual: Vec<Vec<f64>> = mat.to_vec();

    for _component in 0..rank {
        // Random starting vector
        let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let vnorm = frobenius(&v);
        if vnorm > 1e-15 {
            for x in &mut v {
                *x /= vnorm;
            }
        }

        for _iter in 0..max_iter {
            // u = A * v
            let mut u: Vec<f64> = vec![0.0; m];
            for i in 0..m {
                for j in 0..n {
                    u[i] += residual[i][j] * v[j];
                }
            }
            let u_norm = frobenius(&u);
            if u_norm < 1e-15 {
                break;
            }
            for x in &mut u {
                *x /= u_norm;
            }
            // v = A^T * u
            let mut v_new: Vec<f64> = vec![0.0; n];
            for j in 0..n {
                for i in 0..m {
                    v_new[j] += residual[i][j] * u[i];
                }
            }
            let v_norm = frobenius(&v_new);
            if v_norm < 1e-15 {
                break;
            }
            for x in &mut v_new {
                *x /= v_norm;
            }
            v = v_new;
        }

        // Compute sigma = u^T A v
        let mut u_final: Vec<f64> = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                u_final[i] += residual[i][j] * v[j];
            }
        }
        let sigma = frobenius(&u_final);
        if sigma < 1e-15 {
            break;
        }
        for x in &mut u_final {
            *x /= sigma;
        }

        // Deflate
        for i in 0..m {
            for j in 0..n {
                residual[i][j] -= sigma * u_final[i] * v[j];
            }
        }

        u_cols.push(u_final);
        sigmas.push(sigma);
        vt_rows.push(v);
    }

    // Assemble U (m x rank) and Vt (rank x n)
    let actual_rank = sigmas.len();
    let u_mat: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..actual_rank).map(|r| u_cols[r][i]).collect())
        .collect();
    let vt_mat: Vec<Vec<f64>> = vt_rows;

    Ok((u_mat, sigmas, vt_mat))
}
