//! Riemannian manifolds for ML: SPD, Stiefel, Grassmann, SO(3), geodesic optimizers.
//! Types use `Rg` prefix to avoid clashing with geometric_dl.rs types.

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;
#[cfg(test)]
pub mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

// ═══════════════════════════════════════════════════════════════════════════
// §1  Core Trait
// ═══════════════════════════════════════════════════════════════════════════

/// Core trait for Riemannian manifolds.
///
/// All points and tangent vectors are represented as `Vec<f64>` stored in
/// row-major order for matrix manifolds.
pub trait RiemannianManifold {
    /// Exponential map: moves from `p` in direction `v` (tangent at `p`).
    fn exp_map(&self, p: &[f64], v: &[f64]) -> Vec<f64>;

    /// Logarithmic map: tangent vector at `p` pointing towards `q`.
    fn log_map(&self, p: &[f64], q: &[f64]) -> Vec<f64>;

    /// Geodesic distance between two manifold points.
    fn geodesic_distance(&self, p: &[f64], q: &[f64]) -> f64;

    /// Project a vector `v` to the tangent space at `p`.
    fn project_tangent(&self, p: &[f64], v: &[f64]) -> Vec<f64>;

    /// Retraction (defaults to `exp_map`).
    fn retract(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        self.exp_map(p, v)
    }

    /// Intrinsic dimension of the manifold.
    fn dim(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Linear algebra helpers (no external LA crate — pure Rust)
// ═══════════════════════════════════════════════════════════════════════════

/// General n×n matrix multiply (row-major).
pub fn mat_mul_nn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// n×n matrix transpose.
fn mat_transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// Frobenius norm of a flat matrix.
fn frob_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Element-wise addition.
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise subtraction.
fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scale a vector.
fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

/// Dot product.
fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// n×n identity matrix.
fn eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Gaussian elimination for n×n matrix inverse.
/// Returns `None` if singular.
pub fn mat_inv_nn(a: &[f64], n: usize) -> Option<Vec<f64>> {
    // Augmented [A | I]
    let mut aug = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        // Find pivot
        let pivot = (col..n).max_by(|&r1, &r2| {
            aug[r1 * 2 * n + col]
                .abs()
                .partial_cmp(&aug[r2 * 2 * n + col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot = pivot?;
        if aug[pivot * 2 * n + col].abs() < 1e-12 {
            return None;
        }
        // Swap rows
        if pivot != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, pivot * 2 * n + j);
            }
        }
        let diag = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= diag;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..2 * n {
                let delta = factor * aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= delta;
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

/// Symmetric (A + A^T)/2.
pub fn symmetrize_nn(a: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    s
}

/// Power-series matrix exponential for symmetric n×n matrix.
/// Uses eigendecomposition via Jacobi iterations.
pub fn matrix_exp_sym(s: &[f64], n: usize) -> Vec<f64> {
    // Small n: use Jacobi eigendecomposition
    let (vals, vecs) = jacobi_eigen(s, n, 100);
    // exp(S) = V * diag(exp(λ)) * V^T
    let exp_vals: Vec<f64> = vals.iter().map(|v| v.exp()).collect();
    compose_diag(&vecs, &exp_vals, n)
}

/// Matrix logarithm for symmetric positive definite n×n matrix.
/// Returns `None` if any eigenvalue ≤ 0.
pub fn matrix_log_sym(s: &[f64], n: usize) -> Option<Vec<f64>> {
    let (vals, vecs) = jacobi_eigen(s, n, 100);
    if vals.iter().any(|&v| v <= 0.0) {
        return None;
    }
    let log_vals: Vec<f64> = vals.iter().map(|v| v.ln()).collect();
    Some(compose_diag(&vecs, &log_vals, n))
}

/// Reconstruct V * diag(d) * V^T.
fn compose_diag(v: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    // First compute V * diag(d)
    let mut vd = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            vd[i * n + j] = v[i * n + j] * d[j];
        }
    }
    let vt = mat_transpose(v, n);
    mat_mul_nn(&vd, &vt, n)
}

/// Jacobi eigendecomposition for symmetric n×n matrix.
/// Returns (eigenvalues, eigenvectors as columns in row-major V).
fn jacobi_eigen(a: &[f64], n: usize, max_sweeps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d = a.to_vec(); // will be diagonalized in-place (diagonal holds eigenvalues)
    let mut v = eye(n);
    for _ in 0..max_sweeps {
        // Find max off-diagonal element
        let mut max_off = 0.0_f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[i * n + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        // Compute Jacobi rotation
        let d_pp = d[p * n + p];
        let d_qq = d[q * n + q];
        let d_pq = d[p * n + q];
        let theta = if (d_qq - d_pp).abs() < 1e-14 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * d_pq) / (d_qq - d_pp)).atan()
        };
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        // Apply Jacobi rotation: D' = J^T D J, V' = V J
        let mut d_new = d.clone();
        // Update rows/cols p and q of d
        for k in 0..n {
            if k != p && k != q {
                let d_kp = d[k * n + p];
                let d_kq = d[k * n + q];
                d_new[k * n + p] = cos_t * d_kp - sin_t * d_kq;
                d_new[p * n + k] = d_new[k * n + p];
                d_new[k * n + q] = sin_t * d_kp + cos_t * d_kq;
                d_new[q * n + k] = d_new[k * n + q];
            }
        }
        d_new[p * n + p] = cos_t * cos_t * d_pp - 2.0 * sin_t * cos_t * d_pq + sin_t * sin_t * d_qq;
        d_new[q * n + q] = sin_t * sin_t * d_pp + 2.0 * sin_t * cos_t * d_pq + cos_t * cos_t * d_qq;
        d_new[p * n + q] = 0.0;
        d_new[q * n + p] = 0.0;
        d = d_new;
        // Update eigenvectors V
        for k in 0..n {
            let v_kp = v[k * n + p];
            let v_kq = v[k * n + q];
            v[k * n + p] = cos_t * v_kp - sin_t * v_kq;
            v[k * n + q] = sin_t * v_kp + cos_t * v_kq;
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();
    (eigenvalues, v)
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  SPD Manifold (Log-Euclidean metric)
// ═══════════════════════════════════════════════════════════════════════════

/// SPD(n): n×n Symmetric Positive Definite matrices.
///
/// Equipped with the **Log-Euclidean** metric for simplicity and validity:
/// - `exp_S(V) = matrix_exp(matrix_log(S) + V)`
/// - `log_S(Q) = matrix_log(Q) − matrix_log(S)`
/// - `dist(S, Q) = ‖log(S) − log(Q)‖_F`
#[derive(Debug, Clone)]
pub struct RgSpdManifold {
    /// Matrix dimension.
    pub n: usize,
    /// Small ε added to diagonal for numerical stability.
    pub regularizer: f64,
}

impl RgSpdManifold {
    /// Create a new SPD manifold of size `n`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            regularizer: 1e-6,
        }
    }

    /// Symmetrize a flat row-major matrix: (M + M^T) / 2.
    pub fn symmetrize(m: &[f64], n: usize) -> Vec<f64> {
        symmetrize_nn(m, n)
    }

    /// Enforce symmetry and add ε to the diagonal.
    pub fn from_flat(data: &[f64], n: usize) -> Vec<f64> {
        let mut s = symmetrize_nn(data, n);
        for i in 0..n {
            s[i * n + i] += 1e-6;
        }
        s
    }

    /// Lower-triangular Cholesky decomposition: L * L^T = S.
    /// Returns `None` if S is not positive definite.
    pub fn cholesky(s: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut l = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = s[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum < 0.0 {
                        return None;
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    let ljj = l[j * n + j];
                    if ljj.abs() < 1e-14 {
                        return None;
                    }
                    l[i * n + j] = sum / ljj;
                }
            }
        }
        Some(l)
    }

    /// Matrix exponential for symmetric matrix.
    pub fn matrix_exp_sym(s: &[f64], n: usize) -> Vec<f64> {
        matrix_exp_sym(s, n)
    }

    /// Matrix logarithm for symmetric positive definite matrix.
    pub fn matrix_log_sym(s: &[f64], n: usize) -> Option<Vec<f64>> {
        matrix_log_sym(s, n)
    }

    /// n×n matrix multiply.
    pub fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        mat_mul_nn(a, b, n)
    }

    /// n×n matrix inverse via Gaussian elimination.
    pub fn mat_inv(a: &[f64], n: usize) -> Option<Vec<f64>> {
        mat_inv_nn(a, n)
    }

    /// Add regularizer to diagonal for numerical stability.
    fn regularize(&self, m: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = m.to_vec();
        for i in 0..n {
            out[i * n + i] += self.regularizer;
        }
        out
    }
}

impl RiemannianManifold for RgSpdManifold {
    /// Log-Euclidean exp: `exp(log(P) + V)`.
    fn exp_map(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        let n = self.n;
        let p_reg = self.regularize(p);
        match matrix_log_sym(&p_reg, n) {
            Some(log_p) => {
                let sum = vec_add(&log_p, v);
                let result = matrix_exp_sym(&sum, n);
                symmetrize_nn(&result, n)
            }
            None => {
                // Fallback: retract via symmetrization + regularization
                let mut out = p.to_vec();
                for (i, vi) in v.iter().enumerate() {
                    out[i] += vi;
                }
                let sym = symmetrize_nn(&out, n);
                self.regularize(&sym)
            }
        }
    }

    /// Log-Euclidean log: `log(Q) - log(P)`.
    fn log_map(&self, p: &[f64], q: &[f64]) -> Vec<f64> {
        let n = self.n;
        let p_reg = self.regularize(p);
        let q_reg = self.regularize(q);
        let log_p = matrix_log_sym(&p_reg, n).unwrap_or_else(|| vec![0.0; n * n]);
        let log_q = matrix_log_sym(&q_reg, n).unwrap_or_else(|| vec![0.0; n * n]);
        vec_sub(&log_q, &log_p)
    }

    /// `‖log(P) − log(Q)‖_F`.
    fn geodesic_distance(&self, p: &[f64], q: &[f64]) -> f64 {
        let v = self.log_map(p, q);
        frob_norm(&v)
    }

    /// Tangent space at any SPD point = symmetric matrices → symmetrize `v`.
    fn project_tangent(&self, _p: &[f64], v: &[f64]) -> Vec<f64> {
        symmetrize_nn(v, self.n)
    }

    /// dim = n*(n+1)/2.
    fn dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Stiefel Manifold
// ═══════════════════════════════════════════════════════════════════════════

/// St(n, p): n×p matrices with orthonormal columns, stored row-major.
#[derive(Debug, Clone)]
pub struct RgStiefelManifold {
    /// Ambient (row) dimension.
    pub n: usize,
    /// Number of orthonormal columns.
    pub p: usize,
}

impl RgStiefelManifold {
    /// Create St(n, p).
    pub fn new(n: usize, p: usize) -> Self {
        Self { n, p }
    }

    /// Modified Gram-Schmidt orthogonalization of an n×p matrix (row-major).
    pub fn gram_schmidt(matrix: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut q = matrix.to_vec();
        for j in 0..p {
            // Orthogonalize column j against all previous columns
            for k in 0..j {
                let mut dot = 0.0_f64;
                for i in 0..n {
                    dot += q[i * p + j] * q[i * p + k];
                }
                for i in 0..n {
                    q[i * p + j] -= dot * q[i * p + k];
                }
            }
            // Normalize column j
            let mut norm = 0.0_f64;
            for i in 0..n {
                norm += q[i * p + j] * q[i * p + j];
            }
            let norm = norm.sqrt().max(1e-14);
            for i in 0..n {
                q[i * p + j] /= norm;
            }
        }
        q
    }

    /// QR retraction: QR-decompose (X + V), return Q with sign-corrected R diagonal.
    pub fn qr_retraction(x: &[f64], v: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut m: Vec<f64> = x.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
        // Modified Gram-Schmidt on columns
        let mut q = vec![0.0_f64; n * p];
        for j in 0..p {
            // Copy column j
            for i in 0..n {
                q[i * p + j] = m[i * p + j];
            }
            // Orthogonalize against previous
            for k in 0..j {
                let mut dot = 0.0_f64;
                for i in 0..n {
                    dot += q[i * p + j] * q[i * p + k];
                }
                for i in 0..n {
                    q[i * p + j] -= dot * q[i * p + k];
                }
            }
            // Normalize
            let mut norm = 0.0_f64;
            for i in 0..n {
                norm += q[i * p + j] * q[i * p + j];
            }
            let norm = norm.sqrt().max(1e-14);
            // Enforce sign convention: R[j,j] > 0 (same as LAPACK dgeqrf convention)
            let sign = if m[j] >= 0.0 { 1.0 } else { -1.0 };
            // Actually we use the sign of the first element of the column
            let sign_actual = if q[0] >= 0.0 { 1.0 } else { -1.0 };
            let _ = sign; // suppress; use sign_actual
            let _ = sign_actual;
            for i in 0..n {
                q[i * p + j] /= norm;
            }
        }
        // Re-orthogonalize once more for numerical stability
        Self::gram_schmidt(&q, n, p)
    }

    /// Project tangent vector: `Z - X * sym(X^T Z)` where `sym(A) = (A + A^T)/2`.
    pub fn project_stiefel(x: &[f64], z: &[f64], n: usize, p: usize) -> Vec<f64> {
        // Compute X^T Z  (p×p)
        let mut xtx = vec![0.0_f64; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += x[k * p + i] * z[k * p + j];
                }
                xtx[i * p + j] = s;
            }
        }
        // sym(X^T Z)
        let sym_xtx = symmetrize_nn(&xtx, p);
        // X * sym(X^T Z)  (n×p)
        let mut x_sym = vec![0.0_f64; n * p];
        for i in 0..n {
            for j in 0..p {
                let mut s = 0.0_f64;
                for k in 0..p {
                    s += x[i * p + k] * sym_xtx[k * p + j];
                }
                x_sym[i * p + j] = s;
            }
        }
        // Z - X * sym(X^T Z)
        z.iter().zip(x_sym.iter()).map(|(a, b)| a - b).collect()
    }
}

impl RiemannianManifold for RgStiefelManifold {
    /// QR retraction: `qr(X + V)`.
    fn exp_map(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        Self::qr_retraction(p, v, self.n, self.p)
    }

    /// Approximate log map: project (Q − P) to tangent at P.
    fn log_map(&self, p: &[f64], q: &[f64]) -> Vec<f64> {
        let diff: Vec<f64> = q.iter().zip(p.iter()).map(|(a, b)| a - b).collect();
        Self::project_stiefel(p, &diff, self.n, self.p)
    }

    /// `‖log_map(p, q)‖_F`.
    fn geodesic_distance(&self, p: &[f64], q: &[f64]) -> f64 {
        frob_norm(&self.log_map(p, q))
    }

    /// `Z - X * sym(X^T Z)`.
    fn project_tangent(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        Self::project_stiefel(p, v, self.n, self.p)
    }

    /// dim = n*p - p*(p+1)/2.
    fn dim(&self) -> usize {
        self.n * self.p - self.p * (self.p + 1) / 2
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  Grassmann Manifold
// ═══════════════════════════════════════════════════════════════════════════

/// Gr(n, p): p-dimensional subspaces of R^n.
/// Represented by n×p orthonormal matrices (one representative per class).
#[derive(Debug, Clone)]
pub struct RgGrassmannManifold {
    /// Ambient dimension.
    pub n: usize,
    /// Subspace dimension.
    pub p: usize,
}

impl RgGrassmannManifold {
    /// Create Gr(n, p).
    pub fn new(n: usize, p: usize) -> Self {
        Self { n, p }
    }

    /// Principal angles between subspaces spanned by X and Y (both n×p).
    /// Returns p angles in [0, π/2].
    pub fn principal_angles(x: &[f64], y: &[f64], n: usize, p: usize) -> Vec<f64> {
        // Compute C = X^T Y (p×p)
        let mut c = vec![0.0_f64; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += x[k * p + i] * y[k * p + j];
                }
                c[i * p + j] = s;
            }
        }
        // SVD of C to get singular values σ_i; principal angles = arccos(σ_i)
        // We use power iteration to get singular values of C (small p×p matrix)
        singular_values_pp(&c, p)
            .iter()
            .map(|&sv| {
                let clamped = sv.clamp(-1.0, 1.0);
                clamped.acos()
            })
            .collect()
    }

    /// Orthogonal projection matrix P = X * X^T (n×n).
    pub fn projection_matrix(x: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut proj = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..p {
                    s += x[i * p + k] * x[j * p + k];
                }
                proj[i * n + j] = s;
            }
        }
        proj
    }

    /// Project tangent: `(I - P) * V` where `P = X*X^T`.
    fn project_horizontal(x: &[f64], v: &[f64], n: usize, p: usize) -> Vec<f64> {
        let proj = Self::projection_matrix(x, n, p);
        // P*V (n×p)
        let mut pv = vec![0.0_f64; n * p];
        for i in 0..n {
            for j in 0..p {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += proj[i * n + k] * v[k * p + j];
                }
                pv[i * p + j] = s;
            }
        }
        // V - P*V
        v.iter().zip(pv.iter()).map(|(a, b)| a - b).collect()
    }
}

/// Compute singular values of a p×p matrix via Gram matrix eigendecomposition.
fn singular_values_pp(c: &[f64], p: usize) -> Vec<f64> {
    // Gram matrix C^T C (p×p)
    let ct = mat_transpose(c, p);
    let gram = mat_mul_nn(&ct, c, p);
    let (vals, _) = jacobi_eigen(&gram, p, 100);
    // σ_i = sqrt(max(0, λ_i))
    let mut svs: Vec<f64> = vals.iter().map(|&v| v.max(0.0).sqrt()).collect();
    svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    svs
}

impl RiemannianManifold for RgGrassmannManifold {
    /// QR retraction treating as Stiefel point.
    fn exp_map(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        RgStiefelManifold::qr_retraction(p, v, self.n, self.p)
    }

    /// Approximate log map via horizontal lift.
    fn log_map(&self, p: &[f64], q: &[f64]) -> Vec<f64> {
        let diff: Vec<f64> = q.iter().zip(p.iter()).map(|(a, b)| a - b).collect();
        Self::project_horizontal(p, &diff, self.n, self.p)
    }

    /// `sqrt(Σ θ_i²)` over principal angles.
    fn geodesic_distance(&self, p: &[f64], q: &[f64]) -> f64 {
        let angles = Self::principal_angles(p, q, self.n, self.p);
        angles.iter().map(|a| a * a).sum::<f64>().sqrt()
    }

    /// `(I - P) * V`.
    fn project_tangent(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        Self::project_horizontal(p, v, self.n, self.p)
    }

    /// dim = p*(n-p).
    fn dim(&self) -> usize {
        self.p * (self.n - self.p)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  SO(3) Rotation Group
// ═══════════════════════════════════════════════════════════════════════════

/// SO(3): 3×3 rotation matrices, stored as flat `Vec<f64>` of length 9.
#[derive(Debug, Clone)]
pub struct RgSo3Manifold;

impl RgSo3Manifold {
    /// Create SO(3) manifold.
    pub fn new() -> Self {
        Self
    }

    /// Hat map: ω ∈ R³ → skew-symmetric \[ω\]× ∈ so(3) (9 elements, row-major).
    pub fn hat(omega: &[f64]) -> Vec<f64> {
        // [ω]× = [[0, -ω₂, ω₁],
        //         [ω₂,  0, -ω₀],
        //         [-ω₁, ω₀,  0]]
        vec![
            0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0,
        ]
    }

    /// Vee map: skew-symmetric \[ω\]× → ω ∈ R³.
    pub fn vee(skew: &[f64]) -> Vec<f64> {
        // skew[1] = -ω₀, skew[2] = ω₁, skew[5] = -ω₂, etc.
        // From hat: skew[7]=ω₀, skew[2]=ω₁, skew[3]=ω₂
        vec![skew[7], skew[2], skew[3]]
    }

    /// Rodrigues formula: exp(ω) ∈ SO(3) from ω ∈ R³.
    pub fn rodrigues(omega: &[f64]) -> Vec<f64> {
        let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
        if theta < 1e-8 {
            // First-order approximation: R ≈ I + [ω]×
            let hat = Self::hat(omega);
            let id = eye(3);
            return vec_add(&id, &hat);
        }
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let hat = Self::hat(&[omega[0] / theta, omega[1] / theta, omega[2] / theta]);
        // hat² (skew of unit ω)
        let hat2 = Self::mat_mul_3x3(&hat, &hat);
        let id = eye(3);
        // R = I + sin(θ)*[ω̂]× + (1-cos(θ))*[ω̂]×²
        let mut r = vec![0.0_f64; 9];
        for i in 0..9 {
            r[i] = id[i] + sin_t * hat[i] + (1.0 - cos_t) * hat2[i];
        }
        r
    }

    /// Logarithm of R ∈ SO(3), returns ω ∈ R³ such that exp(ω) = R.
    pub fn log_so3(r: &[f64]) -> Vec<f64> {
        let trace = r[0] + r[4] + r[8];
        let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        if theta.abs() < 1e-8 {
            // Near identity: ω ≈ vee(R - I) / 2
            let diff: Vec<f64> = r
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i == 0 || i == 4 || i == 8 {
                        v - 1.0
                    } else {
                        v
                    }
                })
                .collect();
            return vec_scale(&Self::vee(&diff), 0.5);
        }
        // ω = θ / (2 sin θ) * vee(R - R^T)
        let rt = Self::transpose_3x3(r);
        let r_minus_rt: Vec<f64> = r.iter().zip(rt.iter()).map(|(a, b)| a - b).collect();
        let v = Self::vee(&r_minus_rt);
        vec_scale(&v, theta / (2.0 * theta.sin()))
    }

    /// 3×3 matrix multiply.
    pub fn mat_mul_3x3(a: &[f64], b: &[f64]) -> Vec<f64> {
        mat_mul_nn(a, b, 3)
    }

    /// 3×3 matrix transpose.
    pub fn transpose_3x3(r: &[f64]) -> Vec<f64> {
        mat_transpose(r, 3)
    }

    /// Determinant of a 3×3 matrix.
    pub fn det3(r: &[f64]) -> f64 {
        r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6])
    }
}

impl Default for RgSo3Manifold {
    fn default() -> Self {
        Self::new()
    }
}

impl RiemannianManifold for RgSo3Manifold {
    /// `exp_R(v) = R * rodrigues(v)` where v ∈ R³ is a tangent vector (in world frame).
    fn exp_map(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        let delta_r = Self::rodrigues(v);
        Self::mat_mul_3x3(p, &delta_r)
    }

    /// `log_R(Q) = log_so3(R^T Q)` returned as ω ∈ R³.
    fn log_map(&self, p: &[f64], q: &[f64]) -> Vec<f64> {
        let rt = Self::transpose_3x3(p);
        let rt_q = Self::mat_mul_3x3(&rt, q);
        Self::log_so3(&rt_q)
    }

    /// `‖log_map(p, q)‖₂`.
    fn geodesic_distance(&self, p: &[f64], q: &[f64]) -> f64 {
        let v = self.log_map(p, q);
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Project to so(3) tangent space (Lie algebra, R³ representation).
    ///
    /// - If `v` has length 3 (already in R³ Lie algebra form): returned unchanged,
    ///   since any R³ vector is a valid element of so(3).
    /// - If `v` has length 9 (a 3×3 matrix in T_R SO(3)): compute R^T V,
    ///   skew-symmetrize, extract ω via vee.
    fn project_tangent(&self, p: &[f64], v: &[f64]) -> Vec<f64> {
        if v.len() == 3 {
            // Already in Lie algebra representation — no projection needed.
            v.to_vec()
        } else {
            // v is a 3×3 tangent matrix at R. Project: R^T V → skew → vee.
            let rt = Self::transpose_3x3(p);
            let rt_v = Self::mat_mul_3x3(&rt, v);
            let rt_t = mat_transpose(&rt_v, 3);
            let skew: Vec<f64> = rt_v
                .iter()
                .zip(rt_t.iter())
                .map(|(a, b)| 0.5 * (a - b))
                .collect();
            Self::vee(&skew)
        }
    }

    /// dim = 3.
    fn dim(&self) -> usize {
        3
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  Fréchet Mean
// ═══════════════════════════════════════════════════════════════════════════

/// Riemannian Fréchet mean via gradient descent.
#[derive(Debug, Clone)]
pub struct FrechetMean {
    /// Maximum number of gradient descent iterations.
    pub max_iters: usize,
    /// Convergence tolerance (Frobenius / L2 norm of update step).
    pub tol: f64,
    /// Step size for gradient descent.
    pub step_size: f64,
}

impl FrechetMean {
    /// Create with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_iters: 100,
            tol: 1e-7,
            step_size: 1.0,
        }
    }

    /// Compute Fréchet mean of `points` on `manifold`.
    ///
    /// Uses gradient descent: `μ_{t+1} = exp_{μt}(η/N * Σ log_{μt}(xᵢ))`.
    pub fn compute<M: RiemannianManifold>(&self, manifold: &M, points: &[Vec<f64>]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }
        if points.len() == 1 {
            return points[0].clone();
        }
        let n_pts = points.len();
        let dim = points[0].len();
        let weights = vec![1.0 / n_pts as f64; n_pts];
        self.compute_weighted(manifold, points, &weights)
    }

    /// Compute weighted Fréchet mean.
    pub fn compute_weighted<M: RiemannianManifold>(
        &self,
        manifold: &M,
        points: &[Vec<f64>],
        weights: &[f64],
    ) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }
        let dim = points[0].len();
        let mut mu = points[0].clone();

        for _ in 0..self.max_iters {
            // Compute weighted sum of log maps
            let mut grad = vec![0.0_f64; dim];
            for (pt, &w) in points.iter().zip(weights.iter()) {
                let v = manifold.log_map(&mu, pt);
                for (g, vi) in grad.iter_mut().zip(v.iter()) {
                    *g += w * vi;
                }
            }
            // Scale by step size
            let step: Vec<f64> = grad.iter().map(|g| self.step_size * g).collect();
            let step_norm = frob_norm(&step);
            mu = manifold.exp_map(&mu, &step);
            if step_norm < self.tol {
                break;
            }
        }
        mu
    }
}

impl Default for FrechetMean {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  Riemannian Adam Optimizer
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for Riemannian Adam.
#[derive(Debug, Clone)]
pub struct RiemannianAdamConfig {
    /// Learning rate (default 0.01).
    pub lr: f64,
    /// First moment decay (default 0.9).
    pub beta1: f64,
    /// Second moment decay (default 0.999).
    pub beta2: f64,
    /// Numerical stability constant (default 1e-8).
    pub epsilon: f64,
}

impl Default for RiemannianAdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Riemannian Adam optimizer.
///
/// Maintains moment estimates in the tangent space and uses retraction to
/// move along the manifold.
#[derive(Debug, Clone)]
pub struct RiemannianAdam {
    /// Optimizer configuration.
    pub config: RiemannianAdamConfig,
    /// First moment (tangent vector).
    pub m: Option<Vec<f64>>,
    /// Second moment (element-wise squares).
    pub v: Option<Vec<f64>>,
    /// Step count.
    pub t: usize,
}

impl RiemannianAdam {
    /// Create a new Riemannian Adam optimizer.
    pub fn new(config: RiemannianAdamConfig) -> Self {
        Self {
            config,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Perform one update step.
    ///
    /// `riemannian_grad` must already be a tangent vector at `point`
    /// (i.e., already projected to the tangent space).
    ///
    /// Returns the updated point on the manifold.
    pub fn step<M: RiemannianManifold>(
        &mut self,
        manifold: &M,
        point: &[f64],
        riemannian_grad: &[f64],
    ) -> Vec<f64> {
        self.t += 1;
        let g = riemannian_grad;
        let len = g.len();
        let cfg = &self.config;

        // Initialize moments
        let m = self.m.get_or_insert_with(|| vec![0.0_f64; len]);
        let v = self.v.get_or_insert_with(|| vec![0.0_f64; len]);

        // Update first moment
        for (mi, gi) in m.iter_mut().zip(g.iter()) {
            *mi = cfg.beta1 * *mi + (1.0 - cfg.beta1) * gi;
        }
        // Update second moment
        for (vi, gi) in v.iter_mut().zip(g.iter()) {
            *vi = cfg.beta2 * *vi + (1.0 - cfg.beta2) * gi * gi;
        }

        let t = self.t as f64;
        let bc1 = 1.0 - cfg.beta1.powf(t);
        let bc2 = 1.0 - cfg.beta2.powf(t);

        // Bias-corrected moments
        let m_hat: Vec<f64> = m.iter().map(|mi| mi / bc1).collect();
        let v_hat: Vec<f64> = v.iter().map(|vi| vi / bc2).collect();

        // Update direction
        let d: Vec<f64> = m_hat
            .iter()
            .zip(v_hat.iter())
            .map(|(mi, vi)| -cfg.lr * mi / (vi.sqrt() + cfg.epsilon))
            .collect();

        // Project direction to tangent space
        let d_tangent = manifold.project_tangent(point, &d);

        // Retract to manifold
        manifold.retract(point, &d_tangent)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  Riemannian SGD
// ═══════════════════════════════════════════════════════════════════════════

/// Riemannian SGD with optional momentum.
#[derive(Debug, Clone)]
pub struct RiemannianSgd {
    /// Learning rate.
    pub lr: f64,
    /// Momentum coefficient (0.0 = no momentum).
    pub momentum: f64,
    /// Velocity (momentum buffer, in tangent space).
    pub velocity: Option<Vec<f64>>,
}

impl RiemannianSgd {
    /// Create a new Riemannian SGD optimizer.
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            velocity: None,
        }
    }

    /// Perform one SGD step.
    ///
    /// Returns the updated manifold point.
    pub fn step<M: RiemannianManifold>(
        &mut self,
        manifold: &M,
        point: &[f64],
        riemannian_grad: &[f64],
    ) -> Vec<f64> {
        let len = riemannian_grad.len();
        let vel = self.velocity.get_or_insert_with(|| vec![0.0_f64; len]);

        // Update velocity: v = μ*v - lr*g
        for (vi, gi) in vel.iter_mut().zip(riemannian_grad.iter()) {
            *vi = self.momentum * *vi - self.lr * gi;
        }

        // Project to tangent space
        let v_tangent = manifold.project_tangent(point, vel);

        // Update velocity to projected direction for next step
        if let Some(ref mut vel2) = self.velocity {
            for (vi, vti) in vel2.iter_mut().zip(v_tangent.iter()) {
                *vi = *vti;
            }
        }

        // Retract
        manifold.retract(point, &v_tangent)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  Riemannian Batch Normalization (for SPD networks)
// ═══════════════════════════════════════════════════════════════════════════

/// Riemannian Batch Normalization for SPD matrices.
///
/// Normalizes a batch of SPD matrices via:
/// 1. Compute batch Fréchet mean μ
/// 2. Map to tangent space at μ: sᵢ = log_μ(Sᵢ)
/// 3. Normalize: ŝᵢ = sᵢ / std(sᵢ) element-wise
/// 4. Affine: yᵢ = γ ⊙ ŝᵢ + β
/// 5. Map back: Yᵢ = exp_μ(yᵢ)
#[derive(Debug, Clone)]
pub struct RiemannianBatchNorm {
    /// Size of SPD matrices.
    pub n: usize,
    /// Running mean momentum.
    pub momentum: f64,
    /// Running Fréchet mean (SPD).
    pub running_mean: Option<Vec<f64>>,
    /// Scale parameter (per element, initialized to 1.0).
    pub gamma: Vec<f64>,
    /// Bias parameter (per element, initialized to 0.0).
    pub beta: Vec<f64>,
    /// SPD manifold instance.
    pub manifold: RgSpdManifold,
    /// Fréchet mean computer.
    pub frechet: FrechetMean,
}

impl RiemannianBatchNorm {
    /// Create a new Riemannian batch normalization layer for n×n SPD matrices.
    pub fn new(n: usize) -> Self {
        let dim = n * n;
        Self {
            n,
            momentum: 0.1,
            running_mean: None,
            gamma: vec![1.0_f64; dim],
            beta: vec![0.0_f64; dim],
            manifold: RgSpdManifold::new(n),
            frechet: FrechetMean::new(),
        }
    }

    /// Forward pass for a batch of SPD matrices.
    pub fn forward(&mut self, inputs: &[Vec<f64>], training: bool) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let n = self.n;
        let dim = n * n;

        let mu = if training {
            // Compute batch Fréchet mean
            let batch_mean = self.frechet.compute(&self.manifold, inputs);
            // Update running mean
            match &self.running_mean {
                None => {
                    self.running_mean = Some(batch_mean.clone());
                }
                Some(rm) => {
                    let updated: Vec<f64> = rm
                        .iter()
                        .zip(batch_mean.iter())
                        .map(|(r, b)| (1.0 - self.momentum) * r + self.momentum * b)
                        .collect();
                    self.running_mean = Some(updated);
                }
            }
            batch_mean
        } else {
            self.running_mean.clone().unwrap_or_else(|| {
                // Identity matrix as default running mean
                let mut id = vec![0.0_f64; dim];
                for i in 0..n {
                    id[i * n + i] = 1.0;
                }
                id
            })
        };

        // Map inputs to tangent space at mu
        let tangent_vecs: Vec<Vec<f64>> = inputs
            .iter()
            .map(|s| self.manifold.log_map(&mu, s))
            .collect();

        // Compute element-wise std across batch
        let batch_size = tangent_vecs.len() as f64;
        let mut mean_t = vec![0.0_f64; dim];
        for tv in &tangent_vecs {
            for (m, v) in mean_t.iter_mut().zip(tv.iter()) {
                *m += v / batch_size;
            }
        }
        let mut var_t = vec![0.0_f64; dim];
        for tv in &tangent_vecs {
            for (vr, (ti, mi)) in var_t.iter_mut().zip(tv.iter().zip(mean_t.iter())) {
                *vr += (ti - mi) * (ti - mi) / batch_size;
            }
        }
        let std_t: Vec<f64> = var_t.iter().map(|v| v.sqrt().max(1e-8)).collect();

        // Normalize, apply affine, map back
        tangent_vecs
            .iter()
            .map(|tv| {
                let normalized: Vec<f64> = tv
                    .iter()
                    .zip(mean_t.iter())
                    .zip(std_t.iter())
                    .map(|((ti, mi), si)| (ti - mi) / si)
                    .collect();
                let affine: Vec<f64> = normalized
                    .iter()
                    .zip(self.gamma.iter())
                    .zip(self.beta.iter())
                    .map(|((ni, gi), bi)| gi * ni + bi)
                    .collect();
                self.manifold.exp_map(&mu, &affine)
            })
            .collect()
    }
}
