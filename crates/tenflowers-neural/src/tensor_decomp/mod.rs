//! Tensor Decomposition Algorithms: CP-ALS, Tucker-HOOI, HOSVD, TT-SVD, Randomized SVD, NMF.

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ============================================================================
// Math Utilities
// ============================================================================

/// Multiply two matrices: (m×k) × (k×n) → (m×n).
pub fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i][p];
            if a_ip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a_ip * b[p][j];
            }
        }
    }
    c
}

/// Transpose a matrix: (m×n) → (n×m).
pub fn matrix_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Gram-Schmidt QR decomposition: returns (Q, R) where Q has orthonormal columns.
/// Input is m×n (m rows, n columns); Q is m×n, R is n×n.
pub fn matrix_qr(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    if a.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let m = a.len();
    let n = a[0].len();
    let mut q = vec![vec![0.0f64; n]; m];
    let mut r = vec![vec![0.0f64; n]; n];

    // Work column by column (classic Gram-Schmidt)
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);

    for j in 0..n {
        // Start with column j of a
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();

        // Subtract projections onto previously computed Q columns
        for (k, qk) in q_cols.iter().enumerate() {
            let dot: f64 = v.iter().zip(qk.iter()).map(|(vi, qi)| vi * qi).sum();
            r[k][j] = dot;
            for i in 0..m {
                v[i] -= dot * qk[i];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm > 1e-14 {
            let inv = 1.0 / norm;
            for i in 0..m {
                q[i][j] = v[i] * inv;
            }
            q_cols.push((0..m).map(|i| q[i][j]).collect());
        } else {
            // Zero vector — keep as zero column
            q_cols.push(vec![0.0f64; m]);
        }
    }
    (q, r)
}

/// Truncated SVD via block power iteration for top-k singular triplets.
/// Returns (U [m×k], S \[k\], Vt [k×n]).
/// Uses a fixed-seed internal RNG for reproducibility.
pub fn matrix_svd_truncated(a: &[Vec<f64>], k: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    svd_via_block_power_with_seed(a, k, 0xdeadbeef_cafebabe)
}

/// QR-orthonormalize a matrix given as vec-of-vecs [rows × cols], returning exactly k_eff
/// orthonormal columns. If fewer than k_eff linearly independent columns exist in `mat`,
/// the remaining columns are filled with random unit vectors orthogonalized against all
/// previously accepted columns. Uses a fixed internal seed for reproducibility.
pub(crate) fn qr_cols(mat: &[Vec<f64>], k_eff: usize) -> Vec<Vec<f64>> {
    if mat.is_empty() || k_eff == 0 {
        return Vec::new();
    }
    let m = mat.len();
    let n = mat[0].len().min(k_eff);
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(k_eff);
    let mut rng = StdRng::seed_from_u64(0x00ab_cdef_1234_5678);

    // First pass: orthonormalize existing columns from mat
    for j in 0..n {
        let mut v: Vec<f64> = (0..m)
            .map(|i| if j < mat[i].len() { mat[i][j] } else { 0.0 })
            .collect();
        // Subtract projections (twice for numerical stability)
        for _ in 0..2 {
            for qk in &q_cols {
                let dot: f64 = v.iter().zip(qk.iter()).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(qk.iter()) {
                    *vi -= dot * qi;
                }
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            q_cols.push(v.iter().map(|x| x * inv).collect());
        }
    }

    // Second pass: pad with random orthonormal vectors until we have k_eff columns
    let mut attempts = 0usize;
    while q_cols.len() < k_eff && attempts < k_eff * 20 {
        attempts += 1;
        let mut v: Vec<f64> = (0..m).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        // Subtract projections (twice for numerical stability)
        for _ in 0..2 {
            for qk in &q_cols {
                let dot: f64 = v.iter().zip(qk.iter()).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(qk.iter()) {
                    *vi -= dot * qi;
                }
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            q_cols.push(v.iter().map(|x| x * inv).collect());
        }
    }

    let ncols = q_cols.len();
    let mut result = vec![vec![0.0f64; ncols]; m];
    for (j, qcol) in q_cols.iter().enumerate() {
        for i in 0..m {
            result[i][j] = qcol[i];
        }
    }
    result
}

/// Khatri-Rao product (column-wise Kronecker product).
/// If a is (p×r) and b is (q×r), result is (p*q × r).
pub fn khatri_rao(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let p = a.len();
    let q = b.len();
    let r = a[0].len();
    let mut result = vec![vec![0.0f64; r]; p * q];
    for i in 0..p {
        for j in 0..q {
            let row_idx = i * q + j;
            for c in 0..r {
                result[row_idx][c] = a[i][c] * b[j][c];
            }
        }
    }
    result
}

/// Element-wise (Hadamard) product of two slices.
pub fn hadamard_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Compute the Moore-Penrose pseudo-inverse via truncated SVD.
/// Singular values below max(S) * 1e-10 are treated as zero.
pub fn pseudo_inverse_via_svd(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let k = m.min(n);
    let (u, s, vt) = matrix_svd_truncated(a, k);
    if s.is_empty() {
        return vec![vec![0.0f64; m]; n];
    }
    let s_max = s.iter().cloned().fold(0.0f64, f64::max);
    let threshold = s_max * 1e-10;

    // pinv(A) = V * diag(1/s) * U^T  where Vt rows become V columns
    // shape: n×k, then k×m → n×m
    let k_actual = s.len();
    let mut pinv = vec![vec![0.0f64; m]; n];
    for c in 0..k_actual {
        if s[c] < threshold {
            continue;
        }
        let inv_s = 1.0 / s[c];
        // v_col = vt[c] (length n), u_col = u[:, c] (length m)
        let v_col = &vt[c]; // length n
        for j in 0..n {
            for i in 0..m {
                let u_ic = if i < u.len() && c < u[i].len() {
                    u[i][c]
                } else {
                    0.0
                };
                pinv[j][i] += v_col[j] * inv_s * u_ic;
            }
        }
    }
    pinv
}

// ============================================================================
// DenseTensor
// ============================================================================

/// N-dimensional dense tensor stored as a flat row-major `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct DenseTensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl DenseTensor {
    /// Create a new DenseTensor, validating that data.len() == product(shape).
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::invalid_argument_op(
                "DenseTensor::new",
                &format!(
                    "data length {} does not match shape product {}",
                    data.len(),
                    expected
                ),
            ));
        }
        Ok(Self { data, shape })
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Compute flat row-major index from multi-index.
    fn flat_index(&self, idx: &[usize]) -> Result<usize> {
        if idx.len() != self.shape.len() {
            return Err(TensorError::invalid_argument_op(
                "DenseTensor::flat_index",
                &format!(
                    "index length {} does not match ndim {}",
                    idx.len(),
                    self.shape.len()
                ),
            ));
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (k, (&i, &s)) in idx.iter().zip(self.shape.iter()).enumerate().rev() {
            if i >= s {
                return Err(TensorError::invalid_argument_op(
                    "DenseTensor::flat_index",
                    &format!(
                        "index {} out of bounds for dimension {} with size {}",
                        i, k, s
                    ),
                ));
            }
            flat += i * stride;
            stride *= s;
        }
        Ok(flat)
    }

    /// Bounds-checked element access.
    pub fn get(&self, idx: &[usize]) -> Result<f64> {
        let fi = self.flat_index(idx)?;
        Ok(self.data[fi])
    }

    /// Bounds-checked element assignment.
    pub fn set(&mut self, idx: &[usize], v: f64) -> Result<()> {
        let fi = self.flat_index(idx)?;
        self.data[fi] = v;
        Ok(())
    }

    /// Reshape to a new shape (must have same product).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(TensorError::invalid_argument_op(
                "DenseTensor::reshape",
                &format!(
                    "new shape product {} does not match data length {}",
                    expected,
                    self.data.len()
                ),
            ));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    /// Mode-n unfolding (matricize).
    /// Result shape: [shape\[mode\], product(all other dims)].
    pub fn matricize(&self, mode: usize) -> Result<DenseTensor> {
        let ndim = self.shape.len();
        if mode >= ndim {
            return Err(TensorError::invalid_argument_op(
                "DenseTensor::matricize",
                &format!("mode {} out of bounds for ndim {}", mode, ndim),
            ));
        }
        let n_mode = self.shape[mode];
        let other_dims: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &s)| s)
            .collect();
        let n_other: usize = other_dims.iter().product::<usize>().max(1);
        let mut result = vec![0.0f64; n_mode * n_other];
        let mut strides = vec![1usize; ndim];
        for k in (0..ndim - 1).rev() {
            strides[k] = strides[k + 1] * self.shape[k + 1];
        }
        for (flat, &val) in self.data.iter().enumerate() {
            let mut multi = vec![0usize; ndim];
            let mut rem = flat;
            for k in 0..ndim {
                multi[k] = rem / strides[k];
                rem %= strides[k];
            }
            let row = multi[mode];
            let col_parts: Vec<usize> =
                (0..ndim).filter(|&k| k != mode).map(|k| multi[k]).collect();
            let mut col = 0usize;
            let mut col_stride = 1usize;
            for k in (0..other_dims.len()).rev() {
                col += col_parts[k] * col_stride;
                col_stride *= other_dims[k];
            }
            result[row * n_other + col] = val;
        }
        DenseTensor::new(result, vec![n_mode, n_other])
    }

    /// Frobenius norm: sqrt(sum of squared elements).
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Reconstruct a tensor from CP factors. `a[n]` is shape\[n\] × rank, `weights` is length rank.
    pub fn from_factors_khatri_rao(a: &[Vec<Vec<f64>>], weights: &[f64]) -> DenseTensor {
        if a.is_empty() {
            return DenseTensor {
                data: Vec::new(),
                shape: Vec::new(),
            };
        }
        let rank = weights.len();
        let shape: Vec<usize> = a.iter().map(|f| f.len()).collect();
        let ndim = a.len();
        let total: usize = shape.iter().product();
        let mut strides = vec![1usize; ndim];
        for k in (0..ndim - 1).rev() {
            strides[k] = strides[k + 1] * shape[k + 1];
        }
        let mut data = vec![0.0f64; total];
        for flat in 0..total {
            let mut multi = vec![0usize; ndim];
            let mut rem = flat;
            for k in 0..ndim {
                multi[k] = rem / strides[k];
                rem %= strides[k];
            }
            data[flat] = (0..rank)
                .map(|r| (0..ndim).fold(weights[r], |p, n| p * a[n][multi[n]][r]))
                .sum();
        }
        DenseTensor { data, shape }
    }

    /// Mode-n product: multiply tensor by matrix M along mode n.
    /// If tensor shape\[mode\] = I_n and M is (J × I_n), result shape\[mode\] = J.
    pub fn mode_n_product(&self, m: &[Vec<f64>], mode: usize) -> Result<DenseTensor> {
        let unfolded = self.matricize(mode)?;
        let j = m.len();
        let n_other = unfolded.shape[1];
        let i_n = self.shape[mode];
        let unfolded_2d: Vec<Vec<f64>> = (0..i_n)
            .map(|i| unfolded.data[i * n_other..(i + 1) * n_other].to_vec())
            .collect();
        // result_mat: j × n_other (mode-n unfolding of output tensor)
        let result_mat = matrix_multiply(m, &unfolded_2d);
        // Refold: new_shape replaces shape[mode] with j
        let mut new_shape = self.shape.clone();
        new_shape[mode] = j;
        let ndim = new_shape.len();
        let new_total: usize = new_shape.iter().product();
        let mut new_strides = vec![1usize; ndim];
        for k in (0..ndim - 1).rev() {
            new_strides[k] = new_strides[k + 1] * new_shape[k + 1];
        }
        let other_dims: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &s)| s)
            .collect();
        let mut data = vec![0.0f64; new_total];
        for flat in 0..new_total {
            let mut multi = vec![0usize; ndim];
            let mut rem = flat;
            for k in 0..ndim {
                multi[k] = rem / new_strides[k];
                rem %= new_strides[k];
            }
            let row = multi[mode];
            let mut col = 0usize;
            let mut col_stride = 1usize;
            let other_idx: Vec<usize> =
                (0..ndim).filter(|&k| k != mode).map(|k| multi[k]).collect();
            for k in (0..other_dims.len()).rev() {
                col += other_idx[k] * col_stride;
                col_stride *= other_dims[k];
            }
            data[flat] = result_mat[row][col];
        }
        DenseTensor::new(data, new_shape)
    }
}

// ============================================================================
// CP Decomposition
// ============================================================================

/// Configuration for CP-ALS decomposition.
#[derive(Debug, Clone)]
pub struct CpConfig {
    pub rank: usize,
    pub max_iter: usize,
    pub tolerance: f64,
    pub random_seed: u64,
}

impl Default for CpConfig {
    fn default() -> Self {
        Self {
            rank: 2,
            max_iter: 500,
            tolerance: 1e-6,
            random_seed: 42,
        }
    }
}

/// Result of CP decomposition.
#[derive(Debug, Clone)]
pub struct CpDecomposition {
    /// Factor matrices: factors\[n\] is shape\[n\] × rank.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Column-wise normalization weights (length = rank).
    pub weights: Vec<f64>,
    /// Reconstruction fit (1 - relative error).
    pub fit: f64,
    /// Number of ALS iterations performed.
    pub n_iter: usize,
}

impl CpDecomposition {
    /// Reconstruct the approximated tensor from CP factors.
    pub fn reconstruct(&self) -> DenseTensor {
        DenseTensor::from_factors_khatri_rao(&self.factors, &self.weights)
    }

    /// Return the factor matrix for a given mode.
    pub fn compress_factor(&self, mode: usize) -> &Vec<Vec<f64>> {
        &self.factors[mode]
    }
}

/// CP-ALS: Alternating Least Squares for CP Decomposition.
pub struct CpAls;

impl CpAls {
    /// Fit CP decomposition via ALS.
    pub fn fit(tensor: &DenseTensor, config: &CpConfig) -> Result<CpDecomposition> {
        let ndim = tensor.ndim();
        let rank = config.rank;
        if rank == 0 {
            return Err(TensorError::invalid_argument_op(
                "CpAls::fit",
                "rank must be >= 1",
            ));
        }
        if ndim < 2 {
            return Err(TensorError::invalid_argument_op(
                "CpAls::fit",
                "tensor must have at least 2 dimensions",
            ));
        }

        let tensor_norm = tensor.norm();
        if tensor_norm < 1e-14 {
            // Zero tensor
            let factors: Vec<Vec<Vec<f64>>> = tensor
                .shape
                .iter()
                .map(|&s| vec![vec![0.0f64; rank]; s])
                .collect();
            let weights = vec![0.0f64; rank];
            return Ok(CpDecomposition {
                factors,
                weights,
                fit: 1.0,
                n_iter: 0,
            });
        }

        // Initialize factor matrices randomly
        let mut rng = StdRng::seed_from_u64(config.random_seed);
        let mut factors: Vec<Vec<Vec<f64>>> = tensor
            .shape
            .iter()
            .map(|&s| {
                (0..s)
                    .map(|_| (0..rank).map(|_| rng.random::<f64>() - 0.5).collect())
                    .collect()
            })
            .collect();

        // Column-normalize factors
        let mut weights = vec![1.0f64; rank];
        for n in 0..ndim {
            for r in 0..rank {
                let norm: f64 = factors[n]
                    .iter()
                    .map(|row| row[r] * row[r])
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-14 {
                    for row in factors[n].iter_mut() {
                        row[r] /= norm;
                    }
                    weights[r] *= norm;
                }
            }
        }

        let mut prev_fit = -1.0f64;
        let mut n_iter = 0;

        for iter in 0..config.max_iter {
            n_iter = iter + 1;

            for n in 0..ndim {
                // Compute Khatri-Rao product of all factor matrices except mode n
                let modes_to_use: Vec<usize> = (0..ndim).filter(|&k| k != n).collect();

                let kr = Self::khatri_rao_chain(&factors, &modes_to_use);

                // Unfolded tensor: shape[n] × (product of others)
                let unfolded = tensor.matricize(n)?;
                let n_mode = tensor.shape[n];
                let n_other = unfolded.shape[1];

                // Unfolded as 2D
                let x_n: Vec<Vec<f64>> = (0..n_mode)
                    .map(|i| unfolded.data[i * n_other..(i + 1) * n_other].to_vec())
                    .collect();

                // Gram matrices: V = hadamard product of (A^(k)^T A^(k)) for k ≠ n
                let mut gram = vec![vec![1.0f64; rank]; rank];
                for &k in &modes_to_use {
                    let ak_t = matrix_transpose(&factors[k]);
                    let gk = matrix_multiply(&ak_t, &factors[k]); // rank × rank
                    for i in 0..rank {
                        for j in 0..rank {
                            gram[i][j] *= gk[i][j];
                        }
                    }
                }

                // New A^(n) = X_(n) * KR * pinv(gram)
                let x_kr = matrix_multiply(&x_n, &kr);
                let gram_pinv = pseudo_inverse_via_svd(&gram);
                let new_factor = matrix_multiply(&x_kr, &gram_pinv);

                // Column-normalize and accumulate in weights
                for r in 0..rank {
                    let norm: f64 = new_factor
                        .iter()
                        .map(|row| row[r] * row[r])
                        .sum::<f64>()
                        .sqrt();
                    weights[r] = norm;
                    if norm > 1e-14 {
                        for row in factors[n].iter_mut() {
                            row[r] = 0.0;
                        }
                        for (i, row) in new_factor.iter().enumerate() {
                            factors[n][i][r] = row[r] / norm;
                        }
                    } else {
                        weights[r] = 0.0;
                        for row in factors[n].iter_mut() {
                            row[r] = 0.0;
                        }
                    }
                }
            }

            // Compute reconstruction error
            let reconstructed = DenseTensor::from_factors_khatri_rao(&factors, &weights);
            let error: f64 = tensor
                .data
                .iter()
                .zip(reconstructed.data.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let fit = 1.0 - error / tensor_norm;

            if (fit - prev_fit).abs() < config.tolerance && iter > 0 {
                prev_fit = fit;
                break;
            }
            prev_fit = fit;
        }

        Ok(CpDecomposition {
            factors,
            weights,
            fit: prev_fit,
            n_iter,
        })
    }

    /// Khatri-Rao product chain of selected factor matrices.
    fn khatri_rao_chain(factors: &[Vec<Vec<f64>>], modes_to_use: &[usize]) -> Vec<Vec<f64>> {
        if modes_to_use.is_empty() {
            return Vec::new();
        }
        let rank = factors[modes_to_use[0]][0].len();
        let mut result: Vec<Vec<f64>> = factors[modes_to_use[modes_to_use.len() - 1]].clone();
        for &mode in modes_to_use[..modes_to_use.len() - 1].iter().rev() {
            result = khatri_rao(&factors[mode], &result);
        }
        if result.is_empty() {
            result = vec![vec![1.0f64; rank]];
        }
        result
    }
}

// ============================================================================
// Tucker Decomposition
// ============================================================================

/// Configuration for Tucker-HOOI decomposition.
#[derive(Debug, Clone)]
pub struct TuckerConfig {
    pub ranks: Vec<usize>,
    pub max_iter: usize,
    pub tolerance: f64,
    pub random_seed: u64,
}

impl Default for TuckerConfig {
    fn default() -> Self {
        Self {
            ranks: vec![2, 2],
            max_iter: 200,
            tolerance: 1e-6,
            random_seed: 42,
        }
    }
}

/// Result of Tucker decomposition.
#[derive(Debug, Clone)]
pub struct TuckerDecomposition {
    /// Core tensor G.
    pub core: DenseTensor,
    /// Factor matrices: factors\[n\] is shape\[n\] × ranks\[n\].
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Fit (1 - relative error).
    pub fit: f64,
}

impl TuckerDecomposition {
    /// Reconstruct tensor: T ≈ G ×₁ A^(1) ×₂ A^(2) ...
    pub fn reconstruct(&self) -> DenseTensor {
        let mut result = self.core.clone();
        for (n, factor) in self.factors.iter().enumerate() {
            match result.mode_n_product(factor, n) {
                Ok(r) => result = r,
                Err(_) => return result,
            }
        }
        result
    }
}

/// Tucker-HOOI: Higher-Order Orthogonal Iteration.
pub struct TuckerHooi;

impl TuckerHooi {
    /// Fit Tucker decomposition via HOOI.
    pub fn fit(tensor: &DenseTensor, config: &TuckerConfig) -> Result<TuckerDecomposition> {
        let ndim = tensor.ndim();
        if config.ranks.len() != ndim {
            return Err(TensorError::invalid_argument_op(
                "TuckerHooi::fit",
                &format!(
                    "ranks length {} must equal tensor ndim {}",
                    config.ranks.len(),
                    ndim
                ),
            ));
        }
        for (n, &r) in config.ranks.iter().enumerate() {
            if r == 0 || r > tensor.shape[n] {
                return Err(TensorError::invalid_argument_op(
                    "TuckerHooi::fit",
                    &format!(
                        "rank[{}]={} must be in [1, shape[{}]={}]",
                        n, r, n, tensor.shape[n]
                    ),
                ));
            }
        }

        // Initialize via HOSVD
        let hosvd_init = hosvd(tensor, &config.ranks)?;
        let mut factors = hosvd_init.factors;

        let tensor_norm = tensor.norm();
        let mut prev_norm = 0.0f64;

        for _iter in 0..config.max_iter {
            let mut factor_norm_sum = 0.0f64;

            for n in 0..ndim {
                // Compute Y = X ×_{k≠n} A^(k)^T
                let mut y = tensor.clone();
                for k in 0..ndim {
                    if k == n {
                        continue;
                    }
                    let ak_t = matrix_transpose(&factors[k]);
                    y = y.mode_n_product(&ak_t, k)?;
                }

                let y_unfold = y.matricize(n)?;
                let rows = y_unfold.shape[0];
                let cols = y_unfold.shape[1];

                let y_2d: Vec<Vec<f64>> = (0..rows)
                    .map(|i| y_unfold.data[i * cols..(i + 1) * cols].to_vec())
                    .collect();

                let (u, _s, _vt) = matrix_svd_truncated(&y_2d, config.ranks[n]);

                let r_n = config.ranks[n].min(u.first().map(|r| r.len()).unwrap_or(0));
                let new_factor: Vec<Vec<f64>> = u.iter().map(|row| row[..r_n].to_vec()).collect();

                let norm: f64 = new_factor
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt();
                factor_norm_sum += norm;
                factors[n] = new_factor;
            }

            let delta = (factor_norm_sum - prev_norm).abs();
            if delta < config.tolerance && _iter > 0 {
                break;
            }
            prev_norm = factor_norm_sum;
        }

        // Compute core: G = X ×₁ A^(1)^T ×₂ A^(2)^T ...
        let mut core = tensor.clone();
        for n in 0..ndim {
            let ak_t = matrix_transpose(&factors[n]);
            core = core.mode_n_product(&ak_t, n)?;
        }

        // Compute fit
        let reconstructed = {
            let mut r = core.clone();
            for (n, factor) in factors.iter().enumerate() {
                r = r.mode_n_product(factor, n)?;
            }
            r
        };
        let error: f64 = tensor
            .data
            .iter()
            .zip(reconstructed.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fit = if tensor_norm > 1e-14 {
            1.0 - error / tensor_norm
        } else {
            1.0
        };

        Ok(TuckerDecomposition { core, factors, fit })
    }
}

// ============================================================================
// HOSVD
// ============================================================================

/// Higher-Order SVD (non-iterative one-shot Tucker decomposition).
pub fn hosvd(tensor: &DenseTensor, ranks: &[usize]) -> Result<TuckerDecomposition> {
    let ndim = tensor.ndim();
    if ranks.len() != ndim {
        return Err(TensorError::invalid_argument_op(
            "hosvd",
            &format!(
                "ranks length {} must equal tensor ndim {}",
                ranks.len(),
                ndim
            ),
        ));
    }
    for (n, &r) in ranks.iter().enumerate() {
        if r == 0 || r > tensor.shape[n] {
            return Err(TensorError::invalid_argument_op(
                "hosvd",
                &format!(
                    "rank[{}]={} must be in [1, shape[{}]={}]",
                    n, r, n, tensor.shape[n]
                ),
            ));
        }
    }

    let mut factors: Vec<Vec<Vec<f64>>> = Vec::with_capacity(ndim);

    for n in 0..ndim {
        let unfolded = tensor.matricize(n)?;
        let rows = unfolded.shape[0];
        let cols = unfolded.shape[1];
        let mat_2d: Vec<Vec<f64>> = (0..rows)
            .map(|i| unfolded.data[i * cols..(i + 1) * cols].to_vec())
            .collect();

        let (u, _s, _vt) = matrix_svd_truncated(&mat_2d, ranks[n]);

        let r_n = ranks[n].min(u.first().map(|r| r.len()).unwrap_or(0));
        let factor: Vec<Vec<f64>> = u.iter().map(|row| row[..r_n].to_vec()).collect();
        factors.push(factor);
    }

    // Compute core: G = X ×₁ A^(1)^T ×₂ A^(2)^T ...
    let mut core = tensor.clone();
    for n in 0..ndim {
        let ak_t = matrix_transpose(&factors[n]);
        core = core.mode_n_product(&ak_t, n)?;
    }

    let tensor_norm = tensor.norm();
    let reconstructed = {
        let mut r = core.clone();
        for (n, factor) in factors.iter().enumerate() {
            r = r.mode_n_product(factor, n)?;
        }
        r
    };
    let error: f64 = tensor
        .data
        .iter()
        .zip(reconstructed.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let fit = if tensor_norm > 1e-14 {
        1.0 - error / tensor_norm
    } else {
        1.0
    };

    Ok(TuckerDecomposition { core, factors, fit })
}

// ============================================================================
// Tensor Train (TT-SVD)
// ============================================================================

/// Configuration for TT-SVD.
#[derive(Debug, Clone)]
pub struct TtConfig {
    pub max_rank: usize,
    pub tolerance: f64,
}

impl Default for TtConfig {
    fn default() -> Self {
        Self {
            max_rank: 10,
            tolerance: 1e-6,
        }
    }
}

/// Tensor Train representation.
/// Each core k has shape (r_{k-1}, n_k, r_k) stored in row-major order.
#[derive(Debug, Clone)]
pub struct TtTensor {
    pub cores: Vec<Vec<f64>>,
    pub core_shapes: Vec<(usize, usize, usize)>,
    pub shape: Vec<usize>,
}

impl TtTensor {
    /// Reconstruct the full tensor from TT cores via left-to-right contraction.
    pub fn reconstruct(&self) -> DenseTensor {
        let ndim = self.shape.len();
        if ndim == 0 {
            return DenseTensor {
                data: Vec::new(),
                shape: Vec::new(),
            };
        }

        let (r0_prev, n0, r0) = self.core_shapes[0];
        let mut current_mat: Vec<Vec<f64>> = {
            let core = &self.cores[0];
            (0..n0)
                .map(|i| {
                    (0..r0).map(|j| core[i * r0 + j]).collect()
                })
                .collect()
        };
        let _ = r0_prev; // first bond dim is always 1

        for k in 1..ndim {
            let (rk_prev, nk, rk) = self.core_shapes[k];
            let core = &self.cores[k];
            let rows_prev = current_mat.len();
            let mut new_mat = vec![vec![0.0f64; rk]; rows_prev * nk];
            for i_prev in 0..rows_prev {
                for i_k in 0..nk {
                    for alpha in 0..rk_prev {
                        let c_val = current_mat[i_prev][alpha];
                        if c_val == 0.0 {
                            continue;
                        }
                        for beta in 0..rk {
                            let core_idx = alpha * nk * rk + i_k * rk + beta;
                            new_mat[i_prev * nk + i_k][beta] += c_val * core[core_idx];
                        }
                    }
                }
            }
            current_mat = new_mat;
        }

        let data: Vec<f64> = current_mat.iter().map(|row| row[0]).collect();
        DenseTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Compression ratio: full tensor size / sum of core sizes.
    pub fn compression_ratio(&self) -> f64 {
        let full_size: usize = self.shape.iter().product();
        let compressed_size: usize = self.cores.iter().map(|c| c.len()).sum();
        if compressed_size == 0 {
            return 1.0;
        }
        full_size as f64 / compressed_size as f64
    }
}

/// TT-SVD algorithm (Oseledets 2011).
pub struct TtSvd;

impl TtSvd {
    /// Fit a Tensor Train decomposition via sequential SVD.
    pub fn fit(tensor: &DenseTensor, config: &TtConfig) -> Result<TtTensor> {
        let ndim = tensor.ndim();
        if ndim == 0 {
            return Err(TensorError::invalid_argument_op(
                "TtSvd::fit",
                "tensor must have at least 1 dimension",
            ));
        }

        let shape = tensor.shape.clone();
        let mut cores: Vec<Vec<f64>> = Vec::with_capacity(ndim);
        let mut core_shapes: Vec<(usize, usize, usize)> = Vec::with_capacity(ndim);

        let mut r_prev = 1usize;
        let mut remaining_data = tensor.data.clone();
        let mut remaining_cols: usize = tensor.numel();

        for k in 0..ndim {
            let n_k = shape[k];
            let _product_remaining: usize = shape[k + 1..].iter().product::<usize>().max(1);
            let rows = r_prev * n_k;
            let cols = remaining_cols / n_k;

            let mat_2d: Vec<Vec<f64>> = (0..rows)
                .map(|i| {
                    if i * cols < remaining_data.len() {
                        remaining_data[i * cols..((i + 1) * cols).min(remaining_data.len())]
                            .to_vec()
                    } else {
                        vec![0.0f64; cols]
                    }
                })
                .collect();

            let max_possible = rows.min(cols);
            let trunc_rank = config.max_rank.min(max_possible).max(1);

            let (u, s, vt) = matrix_svd_truncated(&mat_2d, trunc_rank);

            let s_max = s.first().cloned().unwrap_or(0.0);
            let threshold = s_max * config.tolerance;
            let r_k = s
                .iter()
                .filter(|&&sv| sv >= threshold)
                .count()
                .max(1)
                .min(trunc_rank);

            let core_size = r_prev * n_k * r_k;
            let mut core_data = vec![0.0f64; core_size];
            for i in 0..rows {
                for j in 0..r_k {
                    let u_ij = if i < u.len() && j < u[i].len() {
                        u[i][j]
                    } else {
                        0.0
                    };
                    core_data[i * r_k + j] = u_ij;
                }
            }
            cores.push(core_data);
            core_shapes.push((r_prev, n_k, r_k));

            if k < ndim - 1 {
                remaining_data = (0..r_k)
                    .flat_map(|i| {
                        let si = s.get(i).cloned().unwrap_or(0.0);
                        let vt_row: Vec<f64> = if i < vt.len() { vt[i].clone() } else { vec![] };
                        (0..cols).map(move |j| si * vt_row.get(j).cloned().unwrap_or(0.0))
                    })
                    .collect();
                remaining_cols = cols;
                r_prev = r_k;
            } else {
                let sv: Vec<f64> = (0..r_k)
                    .map(|i| {
                        s.get(i).cloned().unwrap_or(0.0)
                            * if i < vt.len() && !vt[i].is_empty() {
                                vt[i][0]
                            } else {
                                0.0
                            }
                    })
                    .collect();
                let last = cores.len() - 1;
                let (lrp, lnk, lrk) = core_shapes[last];
                for i in 0..(lrp * lnk) {
                    for j in 0..lrk {
                        cores[last][i * lrk + j] *= sv.get(j).cloned().unwrap_or(1.0);
                    }
                }
                if lrk > 1 {
                    let new_core: Vec<f64> = (0..(lrp * lnk))
                        .map(|i| (0..lrk).map(|j| cores[last][i * lrk + j]).sum())
                        .collect();
                    cores[last] = new_core;
                    core_shapes[last] = (lrp, lnk, 1);
                }
            }
        }

        Ok(TtTensor {
            cores,
            core_shapes,
            shape,
        })
    }
}

// ============================================================================
// Internal SVD helpers (shared with extensions.rs)
// ============================================================================

/// SVD via simultaneous subspace iteration (block power method) with QR stabilization.
pub(crate) fn svd_via_block_power_with_seed(
    b: &[Vec<f64>],
    k: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    if b.is_empty() || k == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let m = b.len();
    let n = b[0].len();
    let k_eff = k.min(m).min(n);

    let bt = matrix_transpose(b);

    let mut rng_inner = StdRng::seed_from_u64(seed);
    let v_init: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            (0..k_eff)
                .map(|_| rng_inner.random::<f64>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    let mut v_block = qr_cols(&v_init, k_eff);

    let n_iter = 30;
    for _ in 0..n_iter {
        let bv = matrix_multiply(b, &v_block);
        let u_block = qr_cols(&bv, k_eff);
        let btu = matrix_multiply(&bt, &u_block);
        v_block = qr_cols(&btu, k_eff);
    }

    let bv_final = matrix_multiply(b, &v_block);
    let u_block = qr_cols(&bv_final, k_eff);

    let mut s_vals = vec![0.0f64; k_eff];
    let mut vt_rows: Vec<Vec<f64>> = vec![vec![0.0f64; n]; k_eff];

    for j in 0..k_eff {
        let u_j: Vec<f64> = u_block
            .iter()
            .map(|row| if j < row.len() { row[j] } else { 0.0 })
            .collect();
        let mut v_j = vec![0.0f64; n];
        for (p, u_jp) in u_j.iter().enumerate() {
            if *u_jp == 0.0 {
                continue;
            }
            for (q_idx, b_pq) in b[p].iter().enumerate() {
                v_j[q_idx] += b_pq * u_jp;
            }
        }
        let sigma: f64 = v_j.iter().map(|x| x * x).sum::<f64>().sqrt();
        s_vals[j] = sigma;
        if sigma > 1e-14 {
            let inv = 1.0 / sigma;
            vt_rows[j] = v_j.iter().map(|x| x * inv).collect();
        }
    }

    let mut indices: Vec<usize> = (0..k_eff).collect();
    indices.sort_by(|&a_idx, &b_idx| {
        s_vals[b_idx]
            .partial_cmp(&s_vals[a_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let s_sorted: Vec<f64> = indices.iter().map(|&i| s_vals[i]).collect();
    let vt_sorted: Vec<Vec<f64>> = indices.iter().map(|&i| vt_rows[i].clone()).collect();

    let mut u_sorted = vec![vec![0.0f64; k_eff]; m];
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..m {
            u_sorted[i][new_j] = if old_j < u_block[i].len() {
                u_block[i][old_j]
            } else {
                0.0
            };
        }
    }

    (u_sorted, s_sorted, vt_sorted)
}

/// SVD of a matrix via block power iteration, returning orthonormal U and Vt.
pub(crate) fn svd_via_block_power(
    b: &[Vec<f64>],
    k: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    svd_via_block_power_with_seed(b, k, 0xfeedfacade)
}
