//! Edge & Mobile ML Optimization
//!
//! Production-grade tensor decomposition, quantization, hardware-aware NAS,
//! dynamic-width inference, integer arithmetic, and memory-budget planning
//! for deploying models on resource-constrained edge / mobile devices.
//!
//! # Key Components
//!
//! * [`EoCpDecomposition`] --- Canonical Polyadic (CP) decomposition via ALS
//! * [`EoTuckerDecomposition`] --- Tucker decomposition via HOSVD (power-iteration SVD)
//! * [`TtDecomposition`] --- Tensor-Train (TT-SVD) decomposition
//! * [`CodebookQuantization`] --- Weight sharing via k-means codebook (Lloyd)
//! * [`ProductQuantization`] --- Sub-vector PQ with asymmetric distance computation
//! * [`HardwareProfile`] / [`HardwareAwareSearch`] --- NAS under latency/memory budgets
//! * [`DynamicWidthNetwork`] --- Slimmable runtime width adaptation
//! * [`IntegerLinear`] --- Integer-only (Q-format fixed-point) inference layer
//! * [`MemoryBudgetAllocator`] --- Activation-checkpointing & operator-fusion planner
//! * [`EdgeMetrics`] / [`EdgeReport`] --- Compression / efficiency diagnostics

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

mod helpers;
use helpers::*;

// ============================================================================
// 1. EoCpDecomposition
// ============================================================================

/// Rank-1 factors from Canonical Polyadic decomposition: T ~ sum_r  a_r (x) b_r (x) c_r.
#[derive(Debug, Clone)]
pub struct CpFactors {
    /// Factor matrix A (I x R).
    pub factor_a: Vec<Vec<f64>>,
    /// Factor matrix B (J x R).
    pub factor_b: Vec<Vec<f64>>,
    /// Factor matrix C (K x R).
    pub factor_c: Vec<Vec<f64>>,
    /// Component weights (length R).
    pub lambdas: Vec<f64>,
    /// Approximation error (relative Frobenius).
    pub approx_error: f64,
    /// Number of ALS iterations actually performed.
    pub iterations: usize,
}

/// Canonical Polyadic (CP / CANDECOMP-PARAFAC) decomposition via ALS.
///
/// Factorizes a 3-D tensor  T(I x J x K)  into a sum of `rank` rank-1 terms:
///     T ~ sum_{r=1}^{R}  lambda_r  *  a_r (x) b_r (x) c_r
///
/// where (x) denotes outer product.
#[derive(Debug, Clone)]
pub struct EoCpDecomposition {
    pub rank: usize,
    pub max_iters: usize,
    pub tol: f64,
    pub seed: u64,
}

impl EoCpDecomposition {
    /// Create a new CP decomposition solver.
    pub fn new(rank: usize, max_iters: usize) -> Self {
        Self {
            rank,
            max_iters,
            tol: 1e-8,
            seed: 42,
        }
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Decompose a 3-D tensor stored in row-major order.
    /// `tensor` has shape (dim_i, dim_j, dim_k).
    pub fn decompose(
        &self,
        tensor: &[f64],
        dim_i: usize,
        dim_j: usize,
        dim_k: usize,
    ) -> Result<CpFactors> {
        let total = dim_i * dim_j * dim_k;
        if tensor.len() != total {
            return Err(TensorError::compute_error_simple(format!(
                "CP decomposition: tensor length {} != {}*{}*{} = {}",
                tensor.len(),
                dim_i,
                dim_j,
                dim_k,
                total,
            )));
        }

        let r = self.rank;
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize factor matrices randomly
        let mut a: Vec<Vec<f64>> = (0..dim_i)
            .map(|_| (0..r).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        let mut b: Vec<Vec<f64>> = (0..dim_j)
            .map(|_| (0..r).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        let mut c: Vec<Vec<f64>> = (0..dim_k)
            .map(|_| (0..r).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        let tensor_norm = frobenius(tensor);
        let mut prev_error = f64::MAX;
        let mut iterations = 0;

        for iter in 0..self.max_iters {
            iterations = iter + 1;

            // Mode-0 unfolding: X_(0) is (I x JK), factor update: A = X_(0) * (C kr B) * pinv(...)
            // A <- X_(0) (C kr B) [(B^T B * C^T C)]^{-1}
            let kr_cb = khatri_rao(&c, &b);
            let unfold_0 = self.mode_unfold(tensor, dim_i, dim_j, dim_k, 0);
            a = self.update_factor(&unfold_0, &kr_cb, r)?;

            // B <- X_(1) (C kr A) [(A^T A * C^T C)]^{-1}
            let kr_ca = khatri_rao(&c, &a);
            let unfold_1 = self.mode_unfold(tensor, dim_i, dim_j, dim_k, 1);
            b = self.update_factor(&unfold_1, &kr_ca, r)?;

            // C <- X_(2) (B kr A) [(A^T A * B^T B)]^{-1}
            let kr_ba = khatri_rao(&b, &a);
            let unfold_2 = self.mode_unfold(tensor, dim_i, dim_j, dim_k, 2);
            c = self.update_factor(&unfold_2, &kr_ba, r)?;

            // Check convergence via reconstruction error
            let recon = self.reconstruct_flat(&a, &b, &c, &vec![1.0; r], dim_i, dim_j, dim_k);
            let err_vec: Vec<f64> = tensor
                .iter()
                .zip(recon.iter())
                .map(|(t, r)| t - r)
                .collect();
            let error = frobenius(&err_vec) / (tensor_norm + 1e-30);

            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        // Normalize columns and extract lambdas
        let mut lambdas = vec![1.0_f64; r];
        for col in 0..r {
            let na: f64 = a.iter().map(|row| row[col] * row[col]).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|row| row[col] * row[col]).sum::<f64>().sqrt();
            let nc: f64 = c.iter().map(|row| row[col] * row[col]).sum::<f64>().sqrt();
            lambdas[col] = na * nb * nc;
            if na > 1e-15 {
                for row in &mut a {
                    row[col] /= na;
                }
            }
            if nb > 1e-15 {
                for row in &mut b {
                    row[col] /= nb;
                }
            }
            if nc > 1e-15 {
                for row in &mut c {
                    row[col] /= nc;
                }
            }
        }

        let recon = self.reconstruct_flat(&a, &b, &c, &lambdas, dim_i, dim_j, dim_k);
        let err_vec: Vec<f64> = tensor
            .iter()
            .zip(recon.iter())
            .map(|(t, r)| t - r)
            .collect();
        let approx_error = frobenius(&err_vec) / (tensor_norm + 1e-30);

        Ok(CpFactors {
            factor_a: a,
            factor_b: b,
            factor_c: c,
            lambdas,
            approx_error,
            iterations,
        })
    }

    /// Reconstruct the full tensor from CP factors.
    pub fn reconstruct(factors: &CpFactors, dim_i: usize, dim_j: usize, dim_k: usize) -> Vec<f64> {
        let r = factors.lambdas.len();
        let mut out = vec![0.0_f64; dim_i * dim_j * dim_k];
        for comp in 0..r {
            let lam = factors.lambdas[comp];
            for i in 0..dim_i {
                for j in 0..dim_j {
                    for k in 0..dim_k {
                        out[i * dim_j * dim_k + j * dim_k + k] += lam
                            * factors.factor_a[i][comp]
                            * factors.factor_b[j][comp]
                            * factors.factor_c[k][comp];
                    }
                }
            }
        }
        out
    }

    // -- helpers --

    fn mode_unfold(
        &self,
        tensor: &[f64],
        di: usize,
        dj: usize,
        dk: usize,
        mode: usize,
    ) -> Vec<Vec<f64>> {
        match mode {
            0 => {
                // (I x JK)
                (0..di)
                    .map(|i| {
                        let mut row = Vec::with_capacity(dj * dk);
                        for j in 0..dj {
                            for k in 0..dk {
                                row.push(tensor[i * dj * dk + j * dk + k]);
                            }
                        }
                        row
                    })
                    .collect()
            }
            1 => {
                // (J x IK)
                (0..dj)
                    .map(|j| {
                        let mut row = Vec::with_capacity(di * dk);
                        for i in 0..di {
                            for k in 0..dk {
                                row.push(tensor[i * dj * dk + j * dk + k]);
                            }
                        }
                        row
                    })
                    .collect()
            }
            _ => {
                // mode 2: (K x IJ)
                (0..dk)
                    .map(|k| {
                        let mut row = Vec::with_capacity(di * dj);
                        for i in 0..di {
                            for j in 0..dj {
                                row.push(tensor[i * dj * dk + j * dk + k]);
                            }
                        }
                        row
                    })
                    .collect()
            }
        }
    }

    fn update_factor(
        &self,
        unfold: &[Vec<f64>],
        kr: &[Vec<f64>],
        _rank: usize,
    ) -> Result<Vec<Vec<f64>>> {
        // factor = unfold * kr * pinv(kr^T kr)
        let product = mat_mul(unfold, kr);
        let krtk = mat_mul(&mat_t(kr), kr);
        let n = krtk.len();
        // Invert krtk (small RxR) via Gauss-Jordan
        let mut aug: Vec<Vec<f64>> = krtk
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                for j in 0..n {
                    r.push(if i == j { 1.0 } else { 0.0 });
                }
                r
            })
            .collect();
        for col in 0..n {
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
                // Add small regularization instead of failing
                aug[col][col] += 1e-10;
            }
            aug.swap(col, max_row);
            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
        let inv: Vec<Vec<f64>> = aug.iter().map(|r| r[n..].to_vec()).collect();
        Ok(mat_mul(&product, &inv))
    }

    fn reconstruct_flat(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        c: &[Vec<f64>],
        lambdas: &[f64],
        di: usize,
        dj: usize,
        dk: usize,
    ) -> Vec<f64> {
        let r = lambdas.len();
        let mut out = vec![0.0_f64; di * dj * dk];
        for comp in 0..r {
            let lam = lambdas[comp];
            for i in 0..di {
                for j in 0..dj {
                    for k in 0..dk {
                        out[i * dj * dk + j * dk + k] += lam * a[i][comp] * b[j][comp] * c[k][comp];
                    }
                }
            }
        }
        out
    }
}

// ============================================================================
// 2. EoTuckerDecomposition
// ============================================================================

/// Result of Tucker decomposition: core tensor G and factor matrices U1, U2, U3.
#[derive(Debug, Clone)]
pub struct TuckerFactors {
    /// Core tensor G of shape (r1 x r2 x r3), stored flat row-major.
    pub core: Vec<f64>,
    pub core_shape: (usize, usize, usize),
    /// Factor matrices U_n  (dim_n x r_n).
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Compression ratio (original_size / compressed_size).
    pub compression_ratio: f64,
    /// Relative Frobenius approximation error.
    pub approx_error: f64,
}

/// Tucker decomposition via truncated HOSVD (Higher-Order SVD).
///
/// T ~ G x_1 U1 x_2 U2 x_3 U3
/// where G is a small core tensor of shape (r1, r2, r3).
#[derive(Debug, Clone)]
pub struct EoTuckerDecomposition {
    /// Target ranks for each mode.
    pub ranks: (usize, usize, usize),
    pub svd_iters: usize,
    pub seed: u64,
}

impl EoTuckerDecomposition {
    pub fn new(ranks: (usize, usize, usize)) -> Self {
        Self {
            ranks,
            svd_iters: 50,
            seed: 42,
        }
    }

    /// Decompose a 3-D tensor T(dim_i, dim_j, dim_k) via HOSVD.
    pub fn decompose(
        &self,
        tensor: &[f64],
        dim_i: usize,
        dim_j: usize,
        dim_k: usize,
    ) -> Result<TuckerFactors> {
        let total = dim_i * dim_j * dim_k;
        if tensor.len() != total {
            return Err(TensorError::compute_error_simple(format!(
                "Tucker decomposition: tensor length {} != {}",
                tensor.len(),
                total,
            )));
        }

        let (r1, r2, r3) = self.ranks;
        let tensor_norm = frobenius(tensor);

        // Mode-0 unfolding -> truncated SVD -> U1
        let unfold_0 = self.mode_unfold(tensor, dim_i, dim_j, dim_k, 0);
        let (u1, _s1, _vt1) = truncated_svd(&unfold_0, r1, self.svd_iters, self.seed)?;

        // Mode-1 unfolding -> truncated SVD -> U2
        let unfold_1 = self.mode_unfold(tensor, dim_i, dim_j, dim_k, 1);
        let (u2, _s2, _vt2) = truncated_svd(&unfold_1, r2, self.svd_iters, self.seed + 1)?;

        // Mode-2 unfolding -> truncated SVD -> U3
        let unfold_2 = self.mode_unfold(tensor, dim_i, dim_j, dim_k, 2);
        let (u3, _s3, _vt3) = truncated_svd(&unfold_2, r3, self.svd_iters, self.seed + 2)?;

        // Core tensor: G = T x_1 U1^T x_2 U2^T x_3 U3^T
        let core = self.compute_core(tensor, dim_i, dim_j, dim_k, &u1, &u2, &u3);
        let actual_r1 = u1.first().map_or(0, |r| r.len());
        let actual_r2 = u2.first().map_or(0, |r| r.len());
        let actual_r3 = u3.first().map_or(0, |r| r.len());

        // Compression ratio
        let original_size = total;
        let compressed_size = actual_r1 * actual_r2 * actual_r3
            + dim_i * actual_r1
            + dim_j * actual_r2
            + dim_k * actual_r3;
        let compression_ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            0.0
        };

        // Reconstruction error
        let recon = Self::reconstruct_from_parts(
            &core, actual_r1, actual_r2, actual_r3, &u1, &u2, &u3, dim_i, dim_j, dim_k,
        );
        let err_vec: Vec<f64> = tensor
            .iter()
            .zip(recon.iter())
            .map(|(t, r)| t - r)
            .collect();
        let approx_error = frobenius(&err_vec) / (tensor_norm + 1e-30);

        Ok(TuckerFactors {
            core,
            core_shape: (actual_r1, actual_r2, actual_r3),
            factors: vec![u1, u2, u3],
            compression_ratio,
            approx_error,
        })
    }

    /// Reconstruct full tensor from Tucker factors.
    pub fn reconstruct(
        factors: &TuckerFactors,
        dim_i: usize,
        dim_j: usize,
        dim_k: usize,
    ) -> Vec<f64> {
        let (r1, r2, r3) = factors.core_shape;
        Self::reconstruct_from_parts(
            &factors.core,
            r1,
            r2,
            r3,
            &factors.factors[0],
            &factors.factors[1],
            &factors.factors[2],
            dim_i,
            dim_j,
            dim_k,
        )
    }

    // -- helpers --

    fn mode_unfold(
        &self,
        tensor: &[f64],
        di: usize,
        dj: usize,
        dk: usize,
        mode: usize,
    ) -> Vec<Vec<f64>> {
        match mode {
            0 => (0..di)
                .map(|i| {
                    let mut row = Vec::with_capacity(dj * dk);
                    for j in 0..dj {
                        for k in 0..dk {
                            row.push(tensor[i * dj * dk + j * dk + k]);
                        }
                    }
                    row
                })
                .collect(),
            1 => (0..dj)
                .map(|j| {
                    let mut row = Vec::with_capacity(di * dk);
                    for i in 0..di {
                        for k in 0..dk {
                            row.push(tensor[i * dj * dk + j * dk + k]);
                        }
                    }
                    row
                })
                .collect(),
            _ => (0..dk)
                .map(|k| {
                    let mut row = Vec::with_capacity(di * dj);
                    for i in 0..di {
                        for j in 0..dj {
                            row.push(tensor[i * dj * dk + j * dk + k]);
                        }
                    }
                    row
                })
                .collect(),
        }
    }

    fn compute_core(
        &self,
        tensor: &[f64],
        di: usize,
        dj: usize,
        dk: usize,
        u1: &[Vec<f64>],
        u2: &[Vec<f64>],
        u3: &[Vec<f64>],
    ) -> Vec<f64> {
        let r1 = u1.first().map_or(0, |r| r.len());
        let r2 = u2.first().map_or(0, |r| r.len());
        let r3 = u3.first().map_or(0, |r| r.len());
        let mut core = vec![0.0_f64; r1 * r2 * r3];
        // G(a,b,c) = sum_{i,j,k} T(i,j,k) * U1(i,a) * U2(j,b) * U3(k,c)
        for i in 0..di {
            for j in 0..dj {
                let tij_base = i * dj * dk + j * dk;
                for k in 0..dk {
                    let val = tensor[tij_base + k];
                    if val.abs() < 1e-30 {
                        continue;
                    }
                    for a in 0..r1 {
                        let u1_ia = u1[i][a];
                        if u1_ia.abs() < 1e-30 {
                            continue;
                        }
                        for b in 0..r2 {
                            let u2_jb = u2[j][b];
                            if u2_jb.abs() < 1e-30 {
                                continue;
                            }
                            for c in 0..r3 {
                                core[a * r2 * r3 + b * r3 + c] += val * u1_ia * u2_jb * u3[k][c];
                            }
                        }
                    }
                }
            }
        }
        core
    }

    fn reconstruct_from_parts(
        core: &[f64],
        r1: usize,
        r2: usize,
        r3: usize,
        u1: &[Vec<f64>],
        u2: &[Vec<f64>],
        u3: &[Vec<f64>],
        di: usize,
        dj: usize,
        dk: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; di * dj * dk];
        // T(i,j,k) = sum_{a,b,c} G(a,b,c) * U1(i,a) * U2(j,b) * U3(k,c)
        for a in 0..r1 {
            for b in 0..r2 {
                for c in 0..r3 {
                    let g_abc = core[a * r2 * r3 + b * r3 + c];
                    if g_abc.abs() < 1e-30 {
                        continue;
                    }
                    for i in 0..di {
                        let ga_u1 = g_abc * u1[i][a];
                        if ga_u1.abs() < 1e-30 {
                            continue;
                        }
                        for j in 0..dj {
                            let gau2 = ga_u1 * u2[j][b];
                            if gau2.abs() < 1e-30 {
                                continue;
                            }
                            for k in 0..dk {
                                out[i * dj * dk + j * dk + k] += gau2 * u3[k][c];
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

// ============================================================================
// 3. TtDecomposition  (Tensor-Train)
// ============================================================================

/// A single TT core: shape (r_{k-1}, n_k, r_k) stored row-major.
#[derive(Debug, Clone)]
pub struct TtCore {
    pub data: Vec<f64>,
    pub shape: (usize, usize, usize), // (r_left, n, r_right)
}

/// Result of Tensor-Train decomposition.
#[derive(Debug, Clone)]
pub struct TtFactors {
    pub cores: Vec<TtCore>,
    /// Original tensor shape.
    pub original_shape: Vec<usize>,
    /// Relative Frobenius approximation error.
    pub approx_error: f64,
}

/// Tensor-Train (TT) decomposition via TT-SVD.
///
/// T(i1, ..., id) = G1(:,i1,:) * G2(:,i2,:) * ... * Gd(:,id,:)
#[derive(Debug, Clone)]
pub struct TtDecomposition {
    pub max_rank: usize,
    pub tol: f64,
    pub svd_iters: usize,
    pub seed: u64,
}

impl TtDecomposition {
    pub fn new(max_rank: usize) -> Self {
        Self {
            max_rank,
            tol: 1e-8,
            svd_iters: 50,
            seed: 42,
        }
    }

    /// Decompose a tensor with given shape into TT format.
    /// `tensor` is stored flat in row-major order for the given shape.
    pub fn decompose(&self, tensor: &[f64], shape: &[usize]) -> Result<TtFactors> {
        let total: usize = shape.iter().product();
        if tensor.len() != total {
            return Err(TensorError::compute_error_simple(format!(
                "TT decomposition: tensor length {} != product of shape {:?} = {}",
                tensor.len(),
                shape,
                total,
            )));
        }
        if shape.len() < 2 {
            return Err(TensorError::compute_error_simple(
                "TT decomposition requires at least 2 dimensions".to_string(),
            ));
        }

        let tensor_norm = frobenius(tensor);
        let d = shape.len();
        let mut cores: Vec<TtCore> = Vec::with_capacity(d);
        let mut c = tensor.to_vec();
        let mut r_prev = 1_usize;

        // Remaining size of the "tail" dimensions
        let mut remaining: usize = total;

        for k in 0..(d - 1) {
            let n_k = shape[k];
            remaining /= n_k;
            // Reshape c as (r_prev * n_k) x (remaining)
            let rows = r_prev * n_k;
            let cols = remaining;
            let mat: Vec<Vec<f64>> = (0..rows)
                .map(|i| (0..cols).map(|j| c[i * cols + j]).collect())
                .collect();

            let rank = self.max_rank.min(rows).min(cols);
            let (u, s, vt) = truncated_svd(&mat, rank, self.svd_iters, self.seed + k as u64)?;

            let actual_rank = s.len();
            // Core_k: (r_prev, n_k, actual_rank)
            let mut core_data = vec![0.0_f64; r_prev * n_k * actual_rank];
            for i in 0..rows {
                for r in 0..actual_rank {
                    core_data[i * actual_rank + r] =
                        u.get(i).and_then(|row| row.get(r).copied()).unwrap_or(0.0);
                }
            }
            cores.push(TtCore {
                data: core_data,
                shape: (r_prev, n_k, actual_rank),
            });

            // c <- diag(s) * Vt  (actual_rank x remaining)
            let mut new_c = vec![0.0_f64; actual_rank * cols];
            for r in 0..actual_rank {
                for j in 0..cols {
                    new_c[r * cols + j] =
                        s[r] * vt.get(r).and_then(|row| row.get(j).copied()).unwrap_or(0.0);
                }
            }
            c = new_c;
            r_prev = actual_rank;
        }

        // Last core: (r_prev, n_{d-1}, 1)
        let n_last = shape[d - 1];
        let mut last_core_data = vec![0.0_f64; r_prev * n_last];
        let copy_len = c.len().min(r_prev * n_last);
        last_core_data[..copy_len].copy_from_slice(&c[..copy_len]);
        cores.push(TtCore {
            data: last_core_data,
            shape: (r_prev, n_last, 1),
        });

        // Compute reconstruction error
        let recon = Self::reconstruct_flat(&cores, shape);
        let err_vec: Vec<f64> = tensor
            .iter()
            .zip(recon.iter())
            .map(|(t, r)| t - r)
            .collect();
        let approx_error = frobenius(&err_vec) / (tensor_norm + 1e-30);

        Ok(TtFactors {
            cores,
            original_shape: shape.to_vec(),
            approx_error,
        })
    }

    /// Reconstruct a full tensor from TT cores.
    pub fn reconstruct(factors: &TtFactors) -> Vec<f64> {
        Self::reconstruct_flat(&factors.cores, &factors.original_shape)
    }

    /// TT-rounding: reduce ranks by re-decomposing each core.
    pub fn round(&self, factors: &TtFactors, new_max_rank: usize) -> Result<TtFactors> {
        let full = Self::reconstruct_flat(&factors.cores, &factors.original_shape);
        let mut dec = self.clone();
        dec.max_rank = new_max_rank;
        dec.decompose(&full, &factors.original_shape)
    }

    fn reconstruct_flat(cores: &[TtCore], shape: &[usize]) -> Vec<f64> {
        let total: usize = shape.iter().product();
        let d = shape.len();
        let mut result = vec![0.0_f64; total];

        // For each multi-index, multiply chain of core slices
        let mut indices = vec![0_usize; d];
        for flat_idx in 0..total {
            // Compute multi-index from flat_idx
            let mut rem = flat_idx;
            for k in (0..d).rev() {
                indices[k] = rem % shape[k];
                rem /= shape[k];
            }

            // Product of slices: G1(:,i1,:) * G2(:,i2,:) * ... * Gd(:,id,:)
            // Start with [1] (1x1 matrix)
            let mut vec_cur: Vec<f64> = vec![1.0];
            let mut cur_cols = 1_usize;

            for (k, core) in cores.iter().enumerate() {
                let (r_left, _n_k, r_right) = core.shape;
                let ik = indices[k];
                // Extract slice core[:, ik, :] which is (r_left x r_right)
                // core.data layout: (r_left * n_k * r_right) row-major for (r_left, n_k, r_right)
                // Element (a, ik, b) = data[a * n_k * r_right + ik * r_right + b]
                let mut new_vec = vec![0.0_f64; r_right];
                // vec_cur has cur_cols entries, should match r_left
                debug_assert_eq!(cur_cols, r_left);
                for a in 0..r_left {
                    let v_a = vec_cur[a];
                    if v_a.abs() < 1e-30 {
                        continue;
                    }
                    for b in 0..r_right {
                        let core_val = core.data[a * cores[k].shape.1 * r_right + ik * r_right + b];
                        new_vec[b] += v_a * core_val;
                    }
                }
                vec_cur = new_vec;
                cur_cols = r_right;
            }

            result[flat_idx] = vec_cur[0];
        }
        result
    }
}

// ============================================================================
// 4. CodebookQuantization
// ============================================================================

/// Result of codebook quantization.
#[derive(Debug, Clone)]
pub struct CodebookResult {
    /// Centroids (codebook), length = n_clusters.
    pub codebook: Vec<f64>,
    /// Index per weight, pointing into codebook.
    pub indices: Vec<usize>,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
    /// Mean-squared quantization error.
    pub mse: f64,
}

/// Weight sharing via k-means codebook (Lloyd's algorithm).
///
/// Each weight is replaced by the index of its nearest centroid.
/// Compression = original_bits / (codebook_bits + index_bits).
#[derive(Debug, Clone)]
pub struct CodebookQuantization {
    pub n_clusters: usize,
    pub max_iters: usize,
    pub seed: u64,
}

impl CodebookQuantization {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iters: 100,
            seed: 42,
        }
    }

    /// Quantize a weight vector into a codebook + indices.
    pub fn quantize(&self, weights: &[f64]) -> Result<CodebookResult> {
        if weights.is_empty() {
            return Err(TensorError::compute_error_simple(
                "CodebookQuantization: empty weight vector".to_string(),
            ));
        }
        let k = self.n_clusters.min(weights.len());
        if k == 0 {
            return Err(TensorError::compute_error_simple(
                "CodebookQuantization: n_clusters must be > 0".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize centroids via k-means++ style
        let mut centroids: Vec<f64> = Vec::with_capacity(k);
        // Pick first centroid randomly
        let first_idx = rng.random_range(0..weights.len());
        centroids.push(weights[first_idx]);

        for _ in 1..k {
            let mut dist_sq: Vec<f64> = weights
                .iter()
                .map(|w| {
                    centroids
                        .iter()
                        .map(|c| (w - c) * (w - c))
                        .fold(f64::MAX, f64::min)
                })
                .collect();
            let total: f64 = dist_sq.iter().sum();
            if total < 1e-30 {
                // All remaining weights are already close to existing centroids
                centroids.push(weights[rng.random_range(0..weights.len())]);
                continue;
            }
            // Normalize to probabilities
            for d in &mut dist_sq {
                *d /= total;
            }
            let r: f64 = rng.random_range(0.0..1.0);
            let mut cumsum = 0.0;
            let mut chosen = 0;
            for (i, &d) in dist_sq.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    chosen = i;
                    break;
                }
            }
            centroids.push(weights[chosen]);
        }

        // Lloyd's algorithm
        let mut indices = vec![0_usize; weights.len()];
        for _iter in 0..self.max_iters {
            // Assignment step
            let mut changed = false;
            for (i, w) in weights.iter().enumerate() {
                let mut best = 0;
                let mut best_dist = f64::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = (w - centroid) * (w - centroid);
                    if d < best_dist {
                        best_dist = d;
                        best = c;
                    }
                }
                if indices[i] != best {
                    changed = true;
                    indices[i] = best;
                }
            }
            if !changed {
                break;
            }

            // Update step
            let mut sums = vec![0.0_f64; k];
            let mut counts = vec![0_usize; k];
            for (i, w) in weights.iter().enumerate() {
                sums[indices[i]] += w;
                counts[indices[i]] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    centroids[c] = sums[c] / counts[c] as f64;
                }
            }
        }

        // Compute MSE
        let mse: f64 = weights
            .iter()
            .zip(indices.iter())
            .map(|(w, &idx)| {
                let d = w - centroids[idx];
                d * d
            })
            .sum::<f64>()
            / weights.len() as f64;

        // Compression ratio: original 64 bits per weight  ->  codebook(k*64) + indices(n * log2(k))
        let index_bits = (k as f64).log2().ceil().max(1.0);
        let original_bits = weights.len() as f64 * 64.0;
        let compressed_bits = k as f64 * 64.0 + weights.len() as f64 * index_bits;
        let compression_ratio = if compressed_bits > 0.0 {
            original_bits / compressed_bits
        } else {
            0.0
        };

        Ok(CodebookResult {
            codebook: centroids,
            indices,
            compression_ratio,
            mse,
        })
    }

    /// Reconstruct weights from codebook + indices.
    pub fn dequantize(codebook: &[f64], indices: &[usize]) -> Vec<f64> {
        indices
            .iter()
            .map(|&idx| {
                if idx < codebook.len() {
                    codebook[idx]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// ============================================================================
// 5. ProductQuantization
// ============================================================================

/// Encoded vectors: each vector is a sequence of M centroid indices.
#[derive(Debug, Clone)]
pub struct PqCodes {
    /// codes\[i\]\[m\] = centroid index for sub-vector m of vector i.
    pub codes: Vec<Vec<usize>>,
    /// Codebooks: codebooks\[m\] is a (n_centroids x sub_dim) table.
    pub codebooks: Vec<Vec<Vec<f64>>>,
    pub n_subquantizers: usize,
    pub sub_dim: usize,
}

/// Sub-vector Product Quantization (PQ).
///
/// Splits each vector into M sub-vectors and independently quantizes each
/// with k-means. Enables fast asymmetric distance computation (ADC).
#[derive(Debug, Clone)]
pub struct ProductQuantization {
    pub n_subquantizers: usize,
    pub n_centroids: usize,
    pub max_iters: usize,
    pub seed: u64,
}

impl ProductQuantization {
    pub fn new(n_subquantizers: usize, n_centroids: usize) -> Self {
        Self {
            n_subquantizers,
            n_centroids,
            max_iters: 50,
            seed: 42,
        }
    }

    /// Encode a batch of vectors.  Each vector has dimension `dim`.
    /// `vectors` is a flat array: vectors[i * dim + j].
    pub fn encode(&self, vectors: &[f64], n_vectors: usize, dim: usize) -> Result<PqCodes> {
        if vectors.len() != n_vectors * dim {
            return Err(TensorError::compute_error_simple(format!(
                "PQ encode: expected {} elements, got {}",
                n_vectors * dim,
                vectors.len(),
            )));
        }
        if dim % self.n_subquantizers != 0 {
            return Err(TensorError::compute_error_simple(format!(
                "PQ encode: dim {} not divisible by n_subquantizers {}",
                dim, self.n_subquantizers,
            )));
        }
        let sub_dim = dim / self.n_subquantizers;
        let m_count = self.n_subquantizers;
        let k = self.n_centroids.min(n_vectors);

        let mut codebooks: Vec<Vec<Vec<f64>>> = Vec::with_capacity(m_count);
        let mut codes: Vec<Vec<usize>> = vec![vec![0_usize; m_count]; n_vectors];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for m in 0..m_count {
            let offset = m * sub_dim;
            // Extract sub-vectors for this partition
            let sub_vecs: Vec<Vec<f64>> = (0..n_vectors)
                .map(|i| {
                    (0..sub_dim)
                        .map(|d| vectors[i * dim + offset + d])
                        .collect()
                })
                .collect();

            // k-means on sub-vectors
            let mut centroids: Vec<Vec<f64>> = (0..k)
                .map(|_| {
                    let idx = rng.random_range(0..n_vectors);
                    sub_vecs[idx].clone()
                })
                .collect();

            let mut assignments = vec![0_usize; n_vectors];
            for _iter in 0..self.max_iters {
                // Assign
                let mut changed = false;
                for i in 0..n_vectors {
                    let mut best = 0;
                    let mut best_dist = f64::MAX;
                    for (c, centroid) in centroids.iter().enumerate() {
                        let d: f64 = sub_vecs[i]
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        if d < best_dist {
                            best_dist = d;
                            best = c;
                        }
                    }
                    if assignments[i] != best {
                        changed = true;
                        assignments[i] = best;
                    }
                }
                if !changed {
                    break;
                }
                // Update
                let mut sums = vec![vec![0.0_f64; sub_dim]; k];
                let mut counts = vec![0_usize; k];
                for (i, &a) in assignments.iter().enumerate() {
                    for d in 0..sub_dim {
                        sums[a][d] += sub_vecs[i][d];
                    }
                    counts[a] += 1;
                }
                for c in 0..k {
                    if counts[c] > 0 {
                        for d in 0..sub_dim {
                            centroids[c][d] = sums[c][d] / counts[c] as f64;
                        }
                    }
                }
            }

            codebooks.push(centroids);
            for (i, &a) in assignments.iter().enumerate() {
                codes[i][m] = a;
            }
        }

        Ok(PqCodes {
            codes,
            codebooks,
            n_subquantizers: m_count,
            sub_dim,
        })
    }

    /// Asymmetric Distance Computation (ADC): compute L2 distance from `query` to every
    /// encoded vector.  Returns distances of length n_vectors.
    pub fn search_adc(&self, query: &[f64], pq_codes: &PqCodes) -> Result<Vec<f64>> {
        let dim = pq_codes.n_subquantizers * pq_codes.sub_dim;
        if query.len() != dim {
            return Err(TensorError::compute_error_simple(format!(
                "PQ search: query dim {} != expected {}",
                query.len(),
                dim,
            )));
        }

        let m_count = pq_codes.n_subquantizers;
        let sub_dim = pq_codes.sub_dim;

        // Precompute distance tables: dist_table[m][c] = ||q_m - centroid_{m,c}||^2
        let mut dist_table: Vec<Vec<f64>> = Vec::with_capacity(m_count);
        for m in 0..m_count {
            let offset = m * sub_dim;
            let q_sub: Vec<f64> = (0..sub_dim).map(|d| query[offset + d]).collect();
            let table: Vec<f64> = pq_codes.codebooks[m]
                .iter()
                .map(|c| {
                    q_sub
                        .iter()
                        .zip(c.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum()
                })
                .collect();
            dist_table.push(table);
        }

        // Compute distance for each vector
        let n_vectors = pq_codes.codes.len();
        let distances: Vec<f64> = (0..n_vectors)
            .map(|i| {
                (0..m_count)
                    .map(|m| {
                        let idx = pq_codes.codes[i][m];
                        if idx < dist_table[m].len() {
                            dist_table[m][idx]
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .collect();

        Ok(distances)
    }
}

// ============================================================================
// 6. HardwareProfile & HardwareAwareSearch
// ============================================================================

/// Description of target hardware capabilities.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Name of the device (e.g., "Cortex-M7", "Jetson Nano").
    pub name: String,
    /// Available RAM in bytes.
    pub memory_budget_bytes: usize,
    /// Compute budget in MFLOPS.
    pub compute_budget_mflops: f64,
    /// Target latency in milliseconds.
    pub latency_target_ms: f64,
    /// Supported integer bit-widths (e.g., [8, 16, 32]).
    pub supported_int_bits: Vec<usize>,
}

impl HardwareProfile {
    pub fn new(name: &str, memory_bytes: usize, mflops: f64, latency_ms: f64) -> Self {
        Self {
            name: name.to_string(),
            memory_budget_bytes: memory_bytes,
            compute_budget_mflops: mflops,
            latency_target_ms: latency_ms,
            supported_int_bits: vec![8, 16, 32],
        }
    }

    /// Predefined profile for a Cortex-M4 class MCU.
    pub fn cortex_m4() -> Self {
        Self::new("Cortex-M4", 256 * 1024, 100.0, 50.0)
    }

    /// Predefined profile for a mobile phone (mid-range).
    pub fn mobile_midrange() -> Self {
        Self::new("Mobile-MidRange", 2 * 1024 * 1024 * 1024, 50_000.0, 20.0)
    }

    /// Predefined profile for an edge GPU (Jetson Nano class).
    pub fn jetson_nano() -> Self {
        Self::new("Jetson-Nano", 4 * 1024 * 1024 * 1024, 472_000.0, 10.0)
    }
}

/// A candidate architecture in the NAS search space.
#[derive(Debug, Clone)]
pub struct EoArchCandidate {
    /// Width multiplier (e.g., 0.25, 0.5, 0.75, 1.0).
    pub width_mult: f64,
    /// Depth multiplier.
    pub depth_mult: f64,
    /// Estimated latency in ms (FLOPs-proxy).
    pub estimated_latency_ms: f64,
    /// Estimated memory in bytes.
    pub estimated_memory_bytes: usize,
    /// Estimated FLOPs (millions).
    pub estimated_mflops: f64,
    /// Accuracy proxy (e.g., from a look-up table or a zero-cost proxy).
    pub accuracy_proxy: f64,
}

/// Hardware-Aware Neural Architecture Search.
///
/// Generates candidate architectures with different width/depth multipliers,
/// estimates their latency and memory, and extracts a Pareto frontier
/// (accuracy vs. latency).
#[derive(Debug, Clone)]
pub struct HardwareAwareSearch {
    pub profile: HardwareProfile,
    /// Base model FLOPs at width=1.0, depth=1.0 (millions).
    pub base_mflops: f64,
    /// Base model parameters at width=1.0.
    pub base_params: usize,
    /// Width multipliers to try.
    pub width_mults: Vec<f64>,
    /// Depth multipliers to try.
    pub depth_mults: Vec<f64>,
    pub seed: u64,
}

impl HardwareAwareSearch {
    pub fn new(profile: HardwareProfile, base_mflops: f64, base_params: usize) -> Self {
        Self {
            profile,
            base_mflops,
            base_params,
            width_mults: vec![0.25, 0.5, 0.75, 1.0],
            depth_mults: vec![0.5, 0.75, 1.0],
            seed: 42,
        }
    }

    /// Generate all candidate architectures.
    pub fn generate_candidates(&self) -> Vec<EoArchCandidate> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut candidates = Vec::new();

        for &wm in &self.width_mults {
            for &dm in &self.depth_mults {
                // FLOPs scale approximately as width^2 * depth (for conv-heavy models)
                let mflops = self.base_mflops * wm * wm * dm;
                // Memory scales as width * depth * param_size
                let mem = (self.base_params as f64 * wm * dm * 4.0) as usize; // 4 bytes/param (f32)
                                                                              // Latency proxy: FLOPs / compute_budget
                let latency = if self.profile.compute_budget_mflops > 0.0 {
                    mflops / self.profile.compute_budget_mflops * 1000.0 // ms
                } else {
                    f64::MAX
                };
                // Accuracy proxy: simple power-law heuristic + noise
                let accuracy_proxy =
                    (0.5 + 0.4 * (wm * dm).powf(0.3) + rng.random_range(-0.02..0.02))
                        .clamp(0.0, 1.0);

                candidates.push(EoArchCandidate {
                    width_mult: wm,
                    depth_mult: dm,
                    estimated_latency_ms: latency,
                    estimated_memory_bytes: mem,
                    estimated_mflops: mflops,
                    accuracy_proxy,
                });
            }
        }
        candidates
    }

    /// Filter candidates that satisfy hardware constraints.
    pub fn filter_feasible(&self, candidates: &[EoArchCandidate]) -> Vec<EoArchCandidate> {
        candidates
            .iter()
            .filter(|c| {
                c.estimated_latency_ms <= self.profile.latency_target_ms
                    && c.estimated_memory_bytes <= self.profile.memory_budget_bytes
            })
            .cloned()
            .collect()
    }

    /// Extract Pareto frontier: candidates not dominated on (accuracy, latency).
    /// Returns indices into the candidates slice.
    pub fn pareto_frontier(&self, candidates: &[EoArchCandidate]) -> Vec<usize> {
        let n = candidates.len();
        let mut is_dominated = vec![false; n];

        for i in 0..n {
            if is_dominated[i] {
                continue;
            }
            for j in 0..n {
                if i == j || is_dominated[j] {
                    continue;
                }
                // j dominates i if j has >= accuracy AND <= latency (and strictly better in one)
                let j_better_acc = candidates[j].accuracy_proxy >= candidates[i].accuracy_proxy;
                let j_better_lat =
                    candidates[j].estimated_latency_ms <= candidates[i].estimated_latency_ms;
                let j_strictly_better = candidates[j].accuracy_proxy > candidates[i].accuracy_proxy
                    || candidates[j].estimated_latency_ms < candidates[i].estimated_latency_ms;
                if j_better_acc && j_better_lat && j_strictly_better {
                    is_dominated[i] = true;
                    break;
                }
            }
        }

        (0..n).filter(|&i| !is_dominated[i]).collect()
    }

    /// Run full search: generate, filter, extract Pareto set.
    pub fn search(&self) -> Vec<EoArchCandidate> {
        let all = self.generate_candidates();
        let feasible = self.filter_feasible(&all);
        let pareto_idxs = self.pareto_frontier(&feasible);
        pareto_idxs.iter().map(|&i| feasible[i].clone()).collect()
    }
}

// ============================================================================
// 7. DynamicWidthNetwork  (Slimmable network)
// ============================================================================

/// A single linear layer that supports dynamic width selection.
#[derive(Debug, Clone)]
pub struct EoSlimmableLinear {
    /// Full weight matrix (out_features x in_features), row-major.
    pub weight: Vec<f64>,
    /// Full bias vector (out_features).
    pub bias: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl EoSlimmableLinear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Xavier uniform initialization
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let weight: Vec<f64> = (0..out_features * in_features)
            .map(|_| rng.random_range(-limit..limit))
            .collect();
        let bias = vec![0.0_f64; out_features];
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass using only the first `ceil(out * width_mult)` output channels
    /// and first `ceil(in * width_mult)` input channels.
    pub fn forward_at_width(&self, input: &[f64], width_mult: f64) -> Result<Vec<f64>> {
        let active_in = ((self.in_features as f64 * width_mult).ceil() as usize)
            .max(1)
            .min(self.in_features);
        let active_out = ((self.out_features as f64 * width_mult).ceil() as usize)
            .max(1)
            .min(self.out_features);

        if input.len() < active_in {
            return Err(TensorError::compute_error_simple(format!(
                "SlimmableLinear: input len {} < active_in {}",
                input.len(),
                active_in,
            )));
        }

        let mut output = Vec::with_capacity(active_out);
        for o in 0..active_out {
            let mut val = self.bias[o];
            for i in 0..active_in {
                val += self.weight[o * self.in_features + i] * input[i];
            }
            output.push(val);
        }
        Ok(output)
    }
}

/// A slimmable MLP that supports runtime width adaptation.
///
/// Implements the approach from "Slimmable Neural Networks" (Yu et al., 2019):
/// a single network trained at multiple widths (0.25x, 0.5x, 0.75x, 1.0x)
/// and selectable at inference time.
#[derive(Debug, Clone)]
pub struct DynamicWidthNetwork {
    pub layers: Vec<EoSlimmableLinear>,
    /// Supported width multipliers.
    pub width_options: Vec<f64>,
}

impl DynamicWidthNetwork {
    /// Build a slimmable MLP with given layer sizes (at full width).
    pub fn new(layer_sizes: &[usize], seed: u64) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(TensorError::compute_error_simple(
                "DynamicWidthNetwork: need at least 2 layer sizes".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(EoSlimmableLinear::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                seed + i as u64,
            ));
        }
        Ok(Self {
            layers,
            width_options: vec![0.25, 0.5, 0.75, 1.0],
        })
    }

    /// Forward pass at a given width multiplier with ReLU activations.
    pub fn forward_at_width(&self, input: &[f64], width_mult: f64) -> Result<Vec<f64>> {
        let mut x = input.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_at_width(&x, width_mult)?;
            // Apply ReLU to all but last layer
            if idx < self.layers.len() - 1 {
                for v in &mut x {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
        Ok(x)
    }

    /// Inplace distillation loss: KL divergence between widest and given width.
    /// Returns sum of squared differences (simplified distillation loss).
    pub fn inplace_distillation_loss(&self, input: &[f64], width_mult: f64) -> Result<f64> {
        let teacher_out = self.forward_at_width(input, 1.0)?;
        let student_out = self.forward_at_width(input, width_mult)?;
        let n = teacher_out.len().min(student_out.len());
        let loss: f64 = (0..n)
            .map(|i| (teacher_out[i] - student_out[i]).powi(2))
            .sum::<f64>()
            / n.max(1) as f64;
        Ok(loss)
    }
}

// ============================================================================
// 8. IntegerArithmetic  (Fixed-point Q-format inference)
// ============================================================================

/// Fixed-point number with configurable integer and fractional bits.
/// Representation: value = raw / 2^frac_bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPoint {
    pub raw: i32,
    pub frac_bits: u8,
}

impl FixedPoint {
    /// Create from a floating-point value.
    pub fn from_f64(value: f64, frac_bits: u8) -> Self {
        let scale = (1_i64 << frac_bits) as f64;
        let raw = (value * scale)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        Self { raw, frac_bits }
    }

    /// Convert back to f64.
    pub fn to_f64(self) -> f64 {
        self.raw as f64 / (1_i64 << self.frac_bits) as f64
    }

    /// Fixed-point multiply.  Result has same frac_bits.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        let product = (self.raw as i64) * (other.raw as i64);
        let shifted = product >> self.frac_bits;
        Self {
            raw: shifted.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            frac_bits: self.frac_bits,
        }
    }

    /// Fixed-point add.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self {
            raw: self.raw.saturating_add(other.raw),
            frac_bits: self.frac_bits,
        }
    }
}

/// Integer-only linear layer: weights in i8, bias in i32, output scale in f64.
///
/// Performs: output = (W_int8 * input_int8 + bias_int32) * output_scale
#[derive(Debug, Clone)]
pub struct IntegerLinear {
    /// Quantized weights: shape (out_features x in_features), row-major.
    pub weight_i8: Vec<i8>,
    /// Quantized bias: shape (out_features).
    pub bias_i32: Vec<i32>,
    /// Scale factors: output = accumulator * output_scale.
    pub input_scale: f64,
    pub weight_scale: f64,
    pub output_scale: f64,
    pub in_features: usize,
    pub out_features: usize,
}

impl IntegerLinear {
    /// Create from float weights and bias.
    /// Quantizes weights to i8 and bias to i32 with symmetric quantization.
    pub fn from_float(
        weights: &[f64],
        bias: &[f64],
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if weights.len() != out_features * in_features {
            return Err(TensorError::compute_error_simple(format!(
                "IntegerLinear: weight size {} != {}x{}",
                weights.len(),
                out_features,
                in_features,
            )));
        }
        if bias.len() != out_features {
            return Err(TensorError::compute_error_simple(format!(
                "IntegerLinear: bias size {} != {}",
                bias.len(),
                out_features,
            )));
        }

        // Symmetric quantization for weights
        let w_max = weights.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
        let weight_scale = if w_max > 1e-30 { w_max / 127.0 } else { 1.0 };
        let weight_i8: Vec<i8> = weights
            .iter()
            .map(|&w| (w / weight_scale).round().clamp(-128.0, 127.0) as i8)
            .collect();

        // For input quantization, assume input is in [-1, 1] by default
        let input_scale = 1.0 / 127.0;

        // Bias quantization to i32 with combined scale
        let bias_scale = input_scale * weight_scale;
        let bias_i32: Vec<i32> = bias
            .iter()
            .map(|&b| {
                if bias_scale > 1e-30 {
                    (b / bias_scale)
                        .round()
                        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
                } else {
                    0
                }
            })
            .collect();

        let output_scale = input_scale * weight_scale;

        Ok(Self {
            weight_i8,
            bias_i32,
            input_scale,
            weight_scale,
            output_scale,
            in_features,
            out_features,
        })
    }

    /// Integer-only forward pass.  `input_i8` is pre-quantized to i8.
    pub fn forward_int(&self, input_i8: &[i8]) -> Result<Vec<i32>> {
        if input_i8.len() < self.in_features {
            return Err(TensorError::compute_error_simple(format!(
                "IntegerLinear forward: input len {} < in_features {}",
                input_i8.len(),
                self.in_features,
            )));
        }
        let mut output = Vec::with_capacity(self.out_features);
        for o in 0..self.out_features {
            let mut acc: i32 = self.bias_i32[o];
            for i in 0..self.in_features {
                acc = acc.saturating_add(
                    (self.weight_i8[o * self.in_features + i] as i32) * (input_i8[i] as i32),
                );
            }
            output.push(acc);
        }
        Ok(output)
    }

    /// Forward pass from float input: quantize, compute, dequantize.
    pub fn forward_float(&self, input: &[f64]) -> Result<Vec<f64>> {
        let input_i8: Vec<i8> = input
            .iter()
            .map(|&x| (x / self.input_scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        let acc = self.forward_int(&input_i8)?;
        Ok(acc.iter().map(|&a| a as f64 * self.output_scale).collect())
    }

    /// Quantized ReLU: clamp accumulator to [0, max].
    pub fn quantized_relu(acc: &[i32]) -> Vec<i32> {
        acc.iter().map(|&a| a.max(0)).collect()
    }

    /// Piecewise-linear sigmoid approximation for integer accumulators.
    /// Maps i32 accumulator (with implied scale) to [0, 1] range output as i32 with 8 fractional bits.
    pub fn quantized_sigmoid_approx(acc: &[i32], scale: f64) -> Vec<i32> {
        // Piecewise linear: 0 for x < -4, 1 for x > 4, linear in between
        let frac_bits = 8;
        let one = 1 << frac_bits; // 256 = fixed-point 1.0
        acc.iter()
            .map(|&a| {
                let x = a as f64 * scale;
                let y = if x < -4.0 {
                    0.0
                } else if x > 4.0 {
                    1.0
                } else {
                    // Simple piecewise: y = 0.125 * x + 0.5
                    (0.125 * x + 0.5).clamp(0.0, 1.0)
                };
                (y * one as f64).round() as i32
            })
            .collect()
    }
}

// ============================================================================
// 9. MemoryBudgetAllocator
// ============================================================================

/// Types of operations for memory planning.
#[derive(Debug, Clone, PartialEq)]
pub enum EoLayerType {
    Conv {
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
    },
    Linear {
        in_feat: usize,
        out_feat: usize,
    },
    BatchNorm {
        channels: usize,
    },
    Relu,
    Pool {
        factor: usize,
    },
    Custom {
        name: String,
        memory_bytes: usize,
    },
}

/// Description of a model layer for memory planning.
#[derive(Debug, Clone)]
pub struct EoLayerDesc {
    pub name: String,
    pub layer_type: EoLayerType,
    /// Spatial resolution at this layer (H, W) or (seq_len, 1) for 1D.
    pub spatial: (usize, usize),
    /// Batch size.
    pub batch_size: usize,
}

/// Fusion opportunity detected by the allocator.
#[derive(Debug, Clone)]
pub struct EoFusionOp {
    /// Indices of layers fused together.
    pub layer_indices: Vec<usize>,
    /// Description of the fused op.
    pub description: String,
    /// Memory saved (bytes).
    pub memory_saved: usize,
}

/// Memory allocation plan for a model under a budget.
#[derive(Debug, Clone)]
pub struct EoAllocationPlan {
    /// Per-layer activation memory (bytes).
    pub layer_memory: Vec<usize>,
    /// Which layers to checkpoint (recompute instead of storing activations).
    pub checkpoint_layers: Vec<usize>,
    /// Detected fusion opportunities.
    pub fusions: Vec<EoFusionOp>,
    /// Peak memory estimate (bytes).
    pub peak_memory_bytes: usize,
    /// Whether the plan fits within budget.
    pub fits_budget: bool,
}

/// Plans memory allocation under a fixed budget.
///
/// Determines activation checkpointing schedule, detects operator fusion
/// opportunities (conv+bn+relu), and estimates peak memory.
#[derive(Debug, Clone)]
pub struct MemoryBudgetAllocator {
    pub budget_bytes: usize,
}

impl MemoryBudgetAllocator {
    pub fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes }
    }

    /// Estimate activation memory for a single layer (in bytes, assuming f32).
    pub fn estimate_layer_memory(layer: &EoLayerDesc) -> usize {
        let (h, w) = layer.spatial;
        let batch = layer.batch_size;
        match &layer.layer_type {
            EoLayerType::Conv { out_ch, .. } => batch * (*out_ch) * h * w * 4,
            EoLayerType::Linear { out_feat, .. } => batch * (*out_feat) * 4,
            EoLayerType::BatchNorm { channels } => batch * (*channels) * h * w * 4,
            EoLayerType::Relu => batch * h * w * 4, // same size as input, simplified
            EoLayerType::Pool { factor } => {
                let ph = (h + factor - 1) / factor;
                let pw = (w + factor - 1) / factor;
                batch * ph * pw * 4
            }
            EoLayerType::Custom { memory_bytes, .. } => *memory_bytes,
        }
    }

    /// Detect Conv + BatchNorm + ReLU fusion patterns.
    pub fn detect_fusions(layers: &[EoLayerDesc]) -> Vec<EoFusionOp> {
        let mut fusions = Vec::new();
        let n = layers.len();
        let mut i = 0;
        while i + 2 < n {
            let is_conv = matches!(&layers[i].layer_type, EoLayerType::Conv { .. });
            let is_bn = matches!(&layers[i + 1].layer_type, EoLayerType::BatchNorm { .. });
            let is_relu = matches!(&layers[i + 2].layer_type, EoLayerType::Relu);
            if is_conv && is_bn && is_relu {
                let bn_mem = Self::estimate_layer_memory(&layers[i + 1]);
                let relu_mem = Self::estimate_layer_memory(&layers[i + 2]);
                fusions.push(EoFusionOp {
                    layer_indices: vec![i, i + 1, i + 2],
                    description: format!(
                        "Fuse {}/{}/{} -> single Conv+BN+ReLU",
                        layers[i].name,
                        layers[i + 1].name,
                        layers[i + 2].name
                    ),
                    memory_saved: bn_mem + relu_mem,
                });
                i += 3;
            } else {
                i += 1;
            }
        }
        fusions
    }

    /// Plan memory allocation for a model.
    pub fn plan_memory(&self, layers: &[EoLayerDesc]) -> EoAllocationPlan {
        let layer_memory: Vec<usize> = layers
            .iter()
            .map(Self::estimate_layer_memory)
            .collect();
        let fusions = Self::detect_fusions(layers);

        // Total memory without optimization
        let total_no_opt: usize = layer_memory.iter().sum();

        // Subtract fusion savings
        let fusion_savings: usize = fusions.iter().map(|f| f.memory_saved).sum();
        let total_after_fusion = total_no_opt.saturating_sub(fusion_savings);

        // If still over budget, select layers to checkpoint (largest first, skip first/last)
        let mut checkpoint_layers = Vec::new();
        let mut current_mem = total_after_fusion;

        if current_mem > self.budget_bytes && layers.len() > 2 {
            // Sort layer indices by memory (descending), skip first and last
            let mut sorted: Vec<(usize, usize)> = layer_memory
                .iter()
                .enumerate()
                .filter(|&(i, _)| i > 0 && i < layers.len() - 1)
                .map(|(i, &m)| (i, m))
                .collect();
            sorted.sort_by_key(|a| std::cmp::Reverse(a.1));

            for (idx, mem) in sorted {
                if current_mem <= self.budget_bytes {
                    break;
                }
                // Checkpointing saves ~50% of that layer's activation memory
                // (we still need to store enough to recompute)
                let savings = mem / 2;
                current_mem = current_mem.saturating_sub(savings);
                checkpoint_layers.push(idx);
            }
            checkpoint_layers.sort();
        }

        let peak_memory_bytes = current_mem;
        let fits_budget = peak_memory_bytes <= self.budget_bytes;

        EoAllocationPlan {
            layer_memory,
            checkpoint_layers,
            fusions,
            peak_memory_bytes,
            fits_budget,
        }
    }
}

// ============================================================================
// 10. EdgeMetrics & EdgeReport
// ============================================================================

/// Collection of edge/mobile efficiency metrics.
#[derive(Debug, Clone)]
pub struct EdgeMetrics {
    /// Compression ratio (original / compressed).
    pub compression_ratio: f64,
    /// Speedup factor vs baseline.
    pub speedup_factor: f64,
    /// Memory footprint in bytes.
    pub memory_footprint_bytes: usize,
    /// Model size in bytes (parameters only).
    pub model_size_bytes: usize,
    /// Number of operations (FLOPs).
    pub flops: f64,
    /// Accuracy (or proxy).
    pub accuracy: f64,
}

impl EdgeMetrics {
    /// Compute metrics from before/after comparison.
    pub fn compute(
        original_params: usize,
        compressed_params: usize,
        original_flops: f64,
        compressed_flops: f64,
        memory_bytes: usize,
        accuracy: f64,
    ) -> Self {
        let compression_ratio = if compressed_params > 0 {
            original_params as f64 / compressed_params as f64
        } else {
            0.0
        };
        let speedup_factor = if compressed_flops > 0.0 {
            original_flops / compressed_flops
        } else {
            0.0
        };
        Self {
            compression_ratio,
            speedup_factor,
            memory_footprint_bytes: memory_bytes,
            model_size_bytes: compressed_params * 4, // assume f32
            flops: compressed_flops,
            accuracy,
        }
    }

    /// Efficiency score: harmonic mean of normalized accuracy and compression.
    pub fn efficiency_score(&self) -> f64 {
        let a = self.accuracy.clamp(0.0, 1.0);
        let c = (self.compression_ratio / 10.0).clamp(0.0, 1.0); // normalize assuming max 10x
        if a + c > 0.0 {
            2.0 * a * c / (a + c)
        } else {
            0.0
        }
    }
}

/// Comprehensive report for edge deployment analysis.
#[derive(Debug, Clone)]
pub struct EdgeReport {
    pub model_name: String,
    pub target_device: String,
    pub metrics: EdgeMetrics,
    /// Pareto-optimal candidates from HW-aware search (if performed).
    pub pareto_candidates: Vec<EoArchCandidate>,
    /// Memory allocation plan (if computed).
    pub allocation_plan: Option<EoAllocationPlan>,
    /// Decomposition errors (if tensor decomposition was applied).
    pub decomposition_errors: Vec<f64>,
}

impl EdgeReport {
    pub fn new(model_name: &str, target_device: &str, metrics: EdgeMetrics) -> Self {
        Self {
            model_name: model_name.to_string(),
            target_device: target_device.to_string(),
            metrics,
            pareto_candidates: Vec::new(),
            allocation_plan: None,
            decomposition_errors: Vec::new(),
        }
    }

    /// Pretty-print the report.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("=== Edge Report: {} ===\n", self.model_name));
        s.push_str(&format!("Target: {}\n", self.target_device));
        s.push_str(&format!(
            "Compression: {:.2}x\n",
            self.metrics.compression_ratio
        ));
        s.push_str(&format!("Speedup: {:.2}x\n", self.metrics.speedup_factor));
        s.push_str(&format!(
            "Memory: {} bytes\n",
            self.metrics.memory_footprint_bytes
        ));
        s.push_str(&format!(
            "Model size: {} bytes\n",
            self.metrics.model_size_bytes
        ));
        s.push_str(&format!("FLOPs: {:.0}\n", self.metrics.flops));
        s.push_str(&format!("Accuracy: {:.4}\n", self.metrics.accuracy));
        s.push_str(&format!(
            "Efficiency score: {:.4}\n",
            self.metrics.efficiency_score()
        ));
        if !self.pareto_candidates.is_empty() {
            s.push_str(&format!(
                "Pareto candidates: {}\n",
                self.pareto_candidates.len()
            ));
        }
        if let Some(ref plan) = self.allocation_plan {
            s.push_str(&format!(
                "Memory plan: peak={} bytes, fits={}\n",
                plan.peak_memory_bytes, plan.fits_budget
            ));
        }
        s
    }

    /// Check if the Pareto frontier dominance analysis found any candidates.
    pub fn pareto_analysis_summary(&self) -> Vec<(f64, f64)> {
        self.pareto_candidates
            .iter()
            .map(|c| (c.accuracy_proxy, c.estimated_latency_ms))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
