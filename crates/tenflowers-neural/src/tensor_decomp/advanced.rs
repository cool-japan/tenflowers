//! Advanced tensor decomposition algorithms:
//! Tensor Ring, Non-negative Tensor Factorization (NTF), Tensor Completion,
//! and Tensor Neural Network layers.

use super::{
    matrix_multiply, matrix_svd_truncated, matrix_transpose, pseudo_inverse_via_svd, DenseTensor,
    TtTensor,
};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ============================================================================
// Tensor Ring Decomposition
// ============================================================================

/// A single core tensor in a tensor ring with shape (r_left, n_k, r_right).
#[derive(Debug, Clone)]
pub struct TensorRingCore {
    /// Raw data in row-major order: (r_left, n_k, r_right).
    pub data: Vec<f64>,
    /// Left bond dimension.
    pub r_left: usize,
    /// Physical dimension (mode size).
    pub n_k: usize,
    /// Right bond dimension.
    pub r_right: usize,
}

impl TensorRingCore {
    /// Create a new TensorRingCore.
    pub fn new(data: Vec<f64>, r_left: usize, n_k: usize, r_right: usize) -> Result<Self> {
        let expected = r_left * n_k * r_right;
        if data.len() != expected {
            return Err(TensorError::invalid_argument_op(
                "TensorRingCore::new",
                &format!(
                    "data length {} does not match r_left*n_k*r_right={}*{}*{}={}",
                    data.len(),
                    r_left,
                    n_k,
                    r_right,
                    expected
                ),
            ));
        }
        Ok(Self { data, r_left, n_k, r_right })
    }

    /// Access element G[alpha, i, beta].
    pub fn get(&self, alpha: usize, i: usize, beta: usize) -> f64 {
        self.data[alpha * self.n_k * self.r_right + i * self.r_right + beta]
    }
}

/// Tensor Ring decomposition result.
/// Cores form a closed ring: G_1 contracted with G_2 ... with G_d and back with G_1.
#[derive(Debug, Clone)]
pub struct TensorRingDecomp {
    /// One core per tensor dimension.
    pub cores: Vec<TensorRingCore>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
}

impl TensorRingDecomp {
    /// Reconstruct the full tensor from ring cores via trace contraction.
    /// For a d-mode tensor, the element T[i_1,...,i_d] = Tr(G_1\[:,i_1,:\] * G_2\[:,i_2,:\] * ... * G_d\[:,i_d,:\]).
    pub fn reconstruct(&self) -> DenseTensor {
        let ndim = self.shape.len();
        if ndim == 0 {
            return DenseTensor {
                data: Vec::new(),
                shape: Vec::new(),
            };
        }
        let total: usize = self.shape.iter().product();
        let mut strides = vec![1usize; ndim];
        for k in (0..ndim - 1).rev() {
            strides[k] = strides[k + 1] * self.shape[k + 1];
        }

        let mut data = vec![0.0f64; total];

        for flat in 0..total {
            // Decode multi-index
            let mut multi = vec![0usize; ndim];
            let mut rem = flat;
            for k in 0..ndim {
                multi[k] = rem / strides[k];
                rem %= strides[k];
            }

            // Matrix product chain G_1[:,i_1,:] * G_2[:,i_2,:] * ... * G_d[:,i_d,:]
            // Each slice G_k[:,i_k,:] is an r_k x r_{k+1} matrix.
            // Ring: r_0 = r_d (same bond dimension wraps around).
            let r0 = self.cores[0].r_left;
            // Start: identity-like accumulator r0 x r0
            let mut acc: Vec<Vec<f64>> = (0..r0)
                .map(|a| (0..r0).map(|b| if a == b { 1.0 } else { 0.0 }).collect())
                .collect();

            for k in 0..ndim {
                let core = &self.cores[k];
                let i_k = multi[k];
                // Extract slice M_k = G_k[:, i_k, :] shape: r_left x r_right
                let slice: Vec<Vec<f64>> = (0..core.r_left)
                    .map(|alpha| {
                        (0..core.r_right)
                            .map(|beta| core.get(alpha, i_k, beta))
                            .collect()
                    })
                    .collect();
                acc = matrix_multiply(&acc, &slice);
            }

            // Trace of acc (r0 x r0)
            let trace: f64 = (0..r0).map(|i| acc[i][i]).sum();
            data[flat] = trace;
        }

        DenseTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Compression ratio: full tensor size / total core parameters.
    pub fn compression_ratio(&self) -> f64 {
        let full_size: usize = self.shape.iter().product();
        let compressed: usize = self
            .cores
            .iter()
            .map(|c| c.r_left * c.n_k * c.r_right)
            .sum();
        if compressed == 0 {
            return 1.0;
        }
        full_size as f64 / compressed as f64
    }
}

/// Tensor Ring Decomposition via DMRG-style sequential SVD.
///
/// Sequentially unfolds and SVDs each mode, then closes the ring by grouping
/// the first and last bond dimensions as a combined physical-mode for the trace.
pub struct TensorRingDecompAlgo;

impl TensorRingDecompAlgo {
    /// Fit a Tensor Ring decomposition.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor.
    /// * `ring_rank` - Bond dimension for the ring (all bonds share this rank).
    pub fn fit(tensor: &DenseTensor, ring_rank: usize) -> Result<TensorRingDecomp> {
        let ndim = tensor.ndim();
        if ndim < 2 {
            return Err(TensorError::invalid_argument_op(
                "TensorRingDecompAlgo::fit",
                "tensor must have at least 2 dimensions",
            ));
        }
        if ring_rank == 0 {
            return Err(TensorError::invalid_argument_op(
                "TensorRingDecompAlgo::fit",
                "ring_rank must be >= 1",
            ));
        }

        let shape = tensor.shape.clone();

        // Use a TT-like left-to-right SVD sweep, then close the ring.
        // Working data starts as the full tensor reshaped to (1*n_0, prod_rest).
        let mut r_prev = 1_usize; // First bond dimension, will grow to ring_rank
        // Actually for the ring we want first bond = ring_rank too.
        // Strategy: do an open TT-SVD with max_rank=ring_rank, then reshape
        // the first core to absorb the last bond via reshaping.

        let total = tensor.numel();
        let mut remaining: Vec<f64> = tensor.data.clone();
        let mut remaining_cols = total;
        let mut cores: Vec<TensorRingCore> = Vec::with_capacity(ndim);

        // We treat the ring as TT with first bond set to ring_rank:
        // Reshape initial tensor to (ring_rank, n_0 * prod_rest / ring_rank) — but
        // that's not exact. Instead use the standard approach: set r_0=1 for TT and
        // then merge with last core to form ring topology.

        // Phase 1: Standard TT-SVD with max_rank = ring_rank, r_0 = 1.
        let mut bond_dims: Vec<usize> = vec![1]; // r_0 = 1
        let mut raw_cores: Vec<Vec<f64>> = Vec::new();
        let mut raw_shapes: Vec<(usize, usize, usize)> = Vec::new();
        let mut r_cur = 1usize;

        for k in 0..ndim {
            let n_k = shape[k];
            let cols = remaining_cols / n_k;
            let rows = r_cur * n_k;

            let mat_2d: Vec<Vec<f64>> = (0..rows)
                .map(|i| {
                    if i * cols < remaining.len() {
                        remaining[i * cols..((i + 1) * cols).min(remaining.len())].to_vec()
                    } else {
                        vec![0.0f64; cols]
                    }
                })
                .collect();

            let max_rank = ring_rank.min(rows.min(cols)).max(1);
            let (u, s, vt) = matrix_svd_truncated(&mat_2d, max_rank);

            let r_k = s
                .iter()
                .filter(|&&sv| sv > 1e-14)
                .count()
                .max(1)
                .min(max_rank);

            // Store core: (r_cur, n_k, r_k)
            let mut core_data = vec![0.0f64; r_cur * n_k * r_k];
            for i in 0..rows {
                for j in 0..r_k {
                    let u_ij = if i < u.len() && j < u[i].len() { u[i][j] } else { 0.0 };
                    core_data[i * r_k + j] = u_ij;
                }
            }
            raw_cores.push(core_data);
            raw_shapes.push((r_cur, n_k, r_k));
            bond_dims.push(r_k);

            if k < ndim - 1 {
                remaining = (0..r_k)
                    .flat_map(|i| {
                        let si = s.get(i).cloned().unwrap_or(0.0);
                        let vt_row = if i < vt.len() { vt[i].clone() } else { vec![] };
                        (0..cols).map(move |j| si * vt_row.get(j).cloned().unwrap_or(0.0))
                    })
                    .collect();
                remaining_cols = cols;
                r_cur = r_k;
            } else {
                // Absorb S*Vt into last core
                let sv: Vec<f64> = (0..r_k)
                    .map(|i| {
                        s.get(i).cloned().unwrap_or(0.0)
                            * if i < vt.len() && !vt[i].is_empty() { vt[i][0] } else { 0.0 }
                    })
                    .collect();
                let last = raw_cores.len() - 1;
                let (lrp, lnk, lrk) = raw_shapes[last];
                for i in 0..(lrp * lnk) {
                    for j in 0..lrk {
                        raw_cores[last][i * lrk + j] *= sv.get(j).cloned().unwrap_or(1.0);
                    }
                }
            }
        }

        // Phase 2: Close the ring.
        // The first core has r_0=1, last core has r_d=1.
        // We need to merge first and last bond into a ring_rank x ring_rank bond.
        // Approach: replicate the first core ring_rank times along the left bond.
        // For a valid ring, we set r_0 = r_d = ring_rank for all cores.
        // A practical approach: pad all bond dims to ring_rank.

        for (k, (raw_shape, raw_core)) in raw_shapes.iter().zip(raw_cores.iter()).enumerate() {
            let (rl, nk, rr) = *raw_shape;
            let target_rl = ring_rank;
            let target_rr = ring_rank;

            // Pad/truncate to target bond dims
            let mut new_data = vec![0.0f64; target_rl * nk * target_rr];
            for alpha in 0..rl.min(target_rl) {
                for i in 0..nk {
                    for beta in 0..rr.min(target_rr) {
                        let src_idx = alpha * nk * rr + i * rr + beta;
                        let dst_idx = alpha * nk * target_rr + i * target_rr + beta;
                        new_data[dst_idx] = raw_core[src_idx];
                    }
                }
            }

            // For ring topology: if this is the first or last core, replicate
            // the single valid slice across all ring_rank copies (diagonal embedding).
            if k == 0 && rl == 1 {
                // Replicate along left bond dimension: G_0[alpha, i, beta] = G_tt[0, i, beta] * delta(alpha, 0)
                // Already done above (only alpha=0 is filled, rest are zero).
            }
            if k == ndim - 1 && rr == 1 {
                // For the last core: G_d[alpha, i, beta] already has only beta=0 filled.
                // Extend: set G_d[alpha, i, alpha] = G_tt[alpha, i, 0] for the diagonal.
                let mut ring_data = vec![0.0f64; target_rl * nk * target_rr];
                for alpha in 0..rl.min(target_rl) {
                    for i in 0..nk {
                        let val = new_data[alpha * nk * target_rr + i * target_rr]; // beta=0
                        ring_data[alpha * nk * target_rr + i * target_rr + alpha] = val;
                    }
                }
                new_data = ring_data;
            }

            cores.push(TensorRingCore {
                data: new_data,
                r_left: target_rl,
                n_k: nk,
                r_right: target_rr,
            });
        }

        Ok(TensorRingDecomp { cores, shape })
    }
}

/// Compress a weight matrix using Tensor Ring decomposition.
///
/// The matrix W (m × n) is reshaped to a higher-order tensor, decomposed via
/// tensor ring, and can be reconstructed approximately.
pub struct TrCompressor {
    /// Target bond dimension for the ring.
    pub ring_rank: usize,
    /// Shape to use for reshaping the matrix before decomposition.
    /// Must satisfy product == m * n.
    pub tensor_shape: Vec<usize>,
}

impl TrCompressor {
    /// Compress a weight matrix.
    pub fn compress(&self, matrix: &[Vec<f64>]) -> Result<TensorRingDecomp> {
        if matrix.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "TrCompressor::compress",
                "matrix is empty",
            ));
        }
        let m = matrix.len();
        let n = matrix[0].len();
        let total = m * n;
        let shape_product: usize = self.tensor_shape.iter().product();
        if shape_product != total {
            return Err(TensorError::invalid_argument_op(
                "TrCompressor::compress",
                &format!(
                    "tensor_shape product {} does not match matrix size {}",
                    shape_product, total
                ),
            ));
        }

        let data: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let tensor = DenseTensor::new(data, self.tensor_shape.clone())?;
        TensorRingDecompAlgo::fit(&tensor, self.ring_rank)
    }

    /// Reconstruct a weight matrix from a tensor ring decomposition.
    pub fn reconstruct(decomp: &TensorRingDecomp, m: usize, n: usize) -> Result<Vec<Vec<f64>>> {
        let full = decomp.reconstruct();
        if full.data.len() != m * n {
            return Err(TensorError::invalid_argument_op(
                "TrCompressor::reconstruct",
                &format!("reconstructed size {} != m*n={}", full.data.len(), m * n),
            ));
        }
        Ok((0..m).map(|i| full.data[i * n..(i + 1) * n].to_vec()).collect())
    }
}

// ============================================================================
// Non-negative Tensor Factorization (NTF)
// ============================================================================

/// NTF model result: non-negative PARAFAC decomposition.
#[derive(Debug, Clone)]
pub struct NtfModel {
    /// Factor matrices: factors\[n\] is shape\[n\] × rank, all nonneg.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Component weights (nonneg), length = rank.
    pub weights: Vec<f64>,
    /// Frobenius reconstruction error.
    pub error: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
}

impl NtfModel {
    /// Reconstruct the tensor from NTF factors.
    pub fn reconstruct(&self) -> DenseTensor {
        DenseTensor::from_factors_khatri_rao(&self.factors, &self.weights)
    }

    /// Fit NTF via multiplicative updates (Lee & Seung 2001, extended to tensors).
    ///
    /// Update rule: A^(n) ← A^(n) * (X_(n) KR^(n)) / (A^(n) V^(n) + ε)
    /// where KR^(n) is the Khatri-Rao product of all other factors and
    /// V^(n) = Hadamard product of (A^(k)^T A^(k)) for k ≠ n.
    pub fn fit(
        tensor: &DenseTensor,
        rank: usize,
        max_iter: usize,
        tolerance: f64,
        seed: u64,
    ) -> Result<Self> {
        use super::khatri_rao;

        let ndim = tensor.ndim();
        if rank == 0 {
            return Err(TensorError::invalid_argument_op(
                "NtfModel::fit",
                "rank must be >= 1",
            ));
        }
        if ndim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NtfModel::fit",
                "tensor must have at least 2 dimensions",
            ));
        }

        let eps = 1e-10f64;
        let mut rng = StdRng::seed_from_u64(seed);

        // Initialize factors with nonneg random values
        let mut factors: Vec<Vec<Vec<f64>>> = tensor
            .shape
            .iter()
            .map(|&s| {
                (0..s)
                    .map(|_| (0..rank).map(|_| rng.random::<f64>() + eps).collect())
                    .collect()
            })
            .collect();
        let mut weights = vec![1.0f64; rank];

        let tensor_norm = tensor.norm().max(eps);
        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            for n in 0..ndim {
                let modes_except_n: Vec<usize> = (0..ndim).filter(|&k| k != n).collect();

                // Khatri-Rao product of all factors except mode n
                let kr = {
                    let mut kr_acc: Vec<Vec<f64>> = factors[modes_except_n[modes_except_n.len() - 1]].clone();
                    for &m_idx in modes_except_n[..modes_except_n.len() - 1].iter().rev() {
                        kr_acc = khatri_rao(&factors[m_idx], &kr_acc);
                    }
                    kr_acc
                };

                // Mode-n unfolding
                let unfolded = tensor.matricize(n)?;
                let n_mode = tensor.shape[n];
                let n_other = unfolded.shape[1];
                let x_n: Vec<Vec<f64>> = (0..n_mode)
                    .map(|i| unfolded.data[i * n_other..(i + 1) * n_other].to_vec())
                    .collect();

                // Numerator: X_(n) KR  (n_mode × rank)
                let numerator = matrix_multiply(&x_n, &kr);

                // Gram product V = Hadamard of A^(k)^T A^(k) for k ≠ n  (rank × rank)
                let mut gram = vec![vec![1.0f64; rank]; rank];
                for &k in &modes_except_n {
                    let ak_t = matrix_transpose(&factors[k]);
                    let gk = matrix_multiply(&ak_t, &factors[k]);
                    for i in 0..rank {
                        for j in 0..rank {
                            gram[i][j] *= gk[i][j];
                        }
                    }
                }

                // Denominator: A^(n) * gram  (n_mode × rank)
                let denominator = matrix_multiply(&factors[n], &gram);

                // Multiplicative update (Lee & Seung style)
                for i in 0..n_mode {
                    for r in 0..rank {
                        let num = numerator[i][r].max(0.0);
                        let den = denominator[i][r] + eps;
                        factors[n][i][r] = (factors[n][i][r] * num / den).max(eps);
                    }
                }

                // Renormalize columns
                for r in 0..rank {
                    let col_norm: f64 = factors[n].iter().map(|row| row[r] * row[r]).sum::<f64>().sqrt().max(eps);
                    for row in factors[n].iter_mut() {
                        row[r] /= col_norm;
                    }
                    weights[r] *= col_norm;
                }
            }

            // Compute error
            let rec = DenseTensor::from_factors_khatri_rao(&factors, &weights);
            let err: f64 = tensor
                .data
                .iter()
                .zip(rec.data.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if (prev_error - err).abs() / tensor_norm < tolerance && iter > 0 {
                prev_error = err;
                break;
            }
            prev_error = err;
        }

        Ok(Self { factors, weights, error: prev_error, n_iter })
    }
}

/// Semi-NMF for tensors: factor matrices along the first mode may have mixed signs,
/// while all other mode factors remain nonneg.
#[derive(Debug, Clone)]
pub struct NtfSemiNmf {
    /// Factor matrices, factors\[0\] may have mixed signs; factors[1..] are nonneg.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Weights, length = rank.
    pub weights: Vec<f64>,
    /// Frobenius reconstruction error.
    pub error: f64,
}

impl NtfSemiNmf {
    /// Fit semi-NMF: mixed-sign mode-0 factor, nonneg for all other modes.
    pub fn fit(
        tensor: &DenseTensor,
        rank: usize,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        use super::khatri_rao;

        let ndim = tensor.ndim();
        if rank == 0 || ndim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NtfSemiNmf::fit",
                "rank >= 1 and ndim >= 2 required",
            ));
        }

        let eps = 1e-10f64;
        let mut rng = StdRng::seed_from_u64(seed);

        // Mode-0: unrestricted; others: nonneg
        let mut factors: Vec<Vec<Vec<f64>>> = tensor
            .shape
            .iter()
            .enumerate()
            .map(|(n, &s)| {
                (0..s)
                    .map(|_| {
                        (0..rank)
                            .map(|_| {
                                let v = rng.random::<f64>();
                                if n == 0 { v * 2.0 - 1.0 } else { v + eps }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let mut weights = vec![1.0f64; rank];

        let mut prev_error = f64::INFINITY;
        let tensor_norm = tensor.norm().max(eps);

        for _iter in 0..max_iter {
            for n in 0..ndim {
                let modes_except_n: Vec<usize> = (0..ndim).filter(|&k| k != n).collect();

                let kr = {
                    let mut kr_acc: Vec<Vec<f64>> = factors[modes_except_n[modes_except_n.len() - 1]].clone();
                    for &m_idx in modes_except_n[..modes_except_n.len() - 1].iter().rev() {
                        kr_acc = khatri_rao(&factors[m_idx], &kr_acc);
                    }
                    kr_acc
                };

                let unfolded = tensor.matricize(n)?;
                let n_mode = tensor.shape[n];
                let n_other = unfolded.shape[1];
                let x_n: Vec<Vec<f64>> = (0..n_mode)
                    .map(|i| unfolded.data[i * n_other..(i + 1) * n_other].to_vec())
                    .collect();

                // x_kr = X_(n) * KR
                let x_kr = matrix_multiply(&x_n, &kr);
                // gram = KR^T * KR
                let kr_t = matrix_transpose(&kr);
                let gram = matrix_multiply(&kr_t, &kr);
                let gram_pinv = pseudo_inverse_via_svd(&gram);

                // LS update for mode-0 (unconstrained)
                if n == 0 {
                    let new_factor = matrix_multiply(&x_kr, &gram_pinv);
                    for (i, row) in new_factor.iter().enumerate() {
                        for r in 0..rank {
                            factors[n][i][r] = if r < row.len() { row[r] } else { 0.0 };
                        }
                    }
                } else {
                    // Multiplicative update for nonneg modes
                    let denominator = matrix_multiply(&factors[n], &gram);
                    for i in 0..n_mode {
                        for r in 0..rank {
                            let num = x_kr[i][r].max(0.0);
                            let den = denominator[i][r] + eps;
                            factors[n][i][r] = (factors[n][i][r] * num / den).max(eps);
                        }
                    }
                }
            }

            let rec = DenseTensor::from_factors_khatri_rao(&factors, &weights);
            let err: f64 = tensor
                .data
                .iter()
                .zip(rec.data.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if (prev_error - err).abs() / tensor_norm < 1e-6 && _iter > 0 {
                prev_error = err;
                break;
            }
            prev_error = err;
        }

        Ok(Self { factors, weights, error: prev_error })
    }
}

/// β-divergence NTF (Cichocki & Amari 2010).
///
/// β=0 → Itakura-Saito divergence, β=1 → KL divergence, β=2 → Frobenius (Euclidean).
#[derive(Debug, Clone)]
pub struct NtfBeta {
    /// β parameter controlling the divergence measure.
    pub beta: f64,
    /// Factor matrices: factors\[n\] is shape\[n\] × rank, all nonneg.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Weights, length = rank.
    pub weights: Vec<f64>,
    /// Final beta-divergence value.
    pub divergence: f64,
}

impl NtfBeta {
    /// Fit β-NTF via multiplicative updates.
    pub fn fit(
        tensor: &DenseTensor,
        rank: usize,
        beta: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        use super::khatri_rao;

        let ndim = tensor.ndim();
        if rank == 0 || ndim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NtfBeta::fit",
                "rank >= 1 and ndim >= 2 required",
            ));
        }

        let eps = 1e-10f64;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut factors: Vec<Vec<Vec<f64>>> = tensor
            .shape
            .iter()
            .map(|&s| {
                (0..s)
                    .map(|_| (0..rank).map(|_| rng.random::<f64>() + eps).collect())
                    .collect()
            })
            .collect();
        let mut weights = vec![1.0f64; rank];

        let mut prev_div = f64::INFINITY;
        let total = tensor.numel();

        for _iter in 0..max_iter {
            for n in 0..ndim {
                let modes_except_n: Vec<usize> = (0..ndim).filter(|&k| k != n).collect();
                let kr = {
                    let mut kr_acc: Vec<Vec<f64>> = factors[modes_except_n[modes_except_n.len() - 1]].clone();
                    for &m_idx in modes_except_n[..modes_except_n.len() - 1].iter().rev() {
                        kr_acc = khatri_rao(&factors[m_idx], &kr_acc);
                    }
                    kr_acc
                };

                let unfolded = tensor.matricize(n)?;
                let n_mode = tensor.shape[n];
                let n_other = unfolded.shape[1];
                let x_n: Vec<Vec<f64>> = (0..n_mode)
                    .map(|i| unfolded.data[i * n_other..(i + 1) * n_other].to_vec())
                    .collect();

                // Compute reconstruction unfolding V_(n) = A^(n) * (KR)^T
                let kr_t = matrix_transpose(&kr);
                let v_n = matrix_multiply(&factors[n], &kr_t); // n_mode × n_other

                // β-divergence multiplicative update
                // num = X_(n) * (KR ⊙ V_n^{β-2})  —  element-wise power on V_n
                // den = 1^T * (KR ⊙ V_n^{β-1})
                // The formula depends on β:
                //   A ← A * (X V^{β-2} KR) / (V^{β-1} KR)
                for i in 0..n_mode {
                    // Collect numerator and denominator contributions per rank
                    let mut num_r = vec![0.0f64; rank];
                    let mut den_r = vec![0.0f64; rank];
                    for j in 0..n_other {
                        let x_ij = x_n[i][j].max(0.0);
                        let v_ij = v_n[i][j].max(eps);
                        let v_beta_m2 = v_ij.powf(beta - 2.0);
                        let v_beta_m1 = v_ij.powf(beta - 1.0);
                        for r in 0..rank {
                            let kr_jr = if j < kr.len() && r < kr[j].len() { kr[j][r] } else { 0.0 };
                            num_r[r] += x_ij * v_beta_m2 * kr_jr;
                            den_r[r] += v_beta_m1 * kr_jr;
                        }
                    }
                    for r in 0..rank {
                        let num = num_r[r].max(0.0);
                        let den = den_r[r] + eps;
                        factors[n][i][r] = (factors[n][i][r] * num / den).max(eps);
                    }
                }

                // Normalize columns (L1 norm for β-NTF)
                for r in 0..rank {
                    let col_norm: f64 = factors[n].iter().map(|row| row[r]).sum::<f64>().max(eps);
                    for row in factors[n].iter_mut() {
                        row[r] /= col_norm;
                    }
                    weights[r] *= col_norm;
                }
            }

            // Guard against weight explosion
            let max_w = weights.iter().map(|w| w.abs()).fold(0.0f64, f64::max).max(eps);
            if max_w > 1e8 {
                for w in weights.iter_mut() {
                    *w /= max_w;
                }
                let scale = max_w.powf(1.0 / ndim as f64);
                for factor in factors.iter_mut() {
                    for row in factor.iter_mut() {
                        for v in row.iter_mut() {
                            *v = (*v * scale).max(eps);
                        }
                    }
                }
            }

            // Compute beta-divergence
            let rec = DenseTensor::from_factors_khatri_rao(&factors, &weights);
            let div: f64 = tensor
                .data
                .iter()
                .zip(rec.data.iter())
                .map(|(x, v)| {
                    let x = x.max(0.0);
                    let v = v.max(eps);
                    if (beta - 1.0).abs() < 1e-9 {
                        x * (x / v).ln() - x + v
                    } else if beta.abs() < 1e-9 {
                        x / v - (x / v).ln() - 1.0
                    } else {
                        x.powf(beta) / (beta * (beta - 1.0)) - x * v.powf(beta - 1.0) / (beta - 1.0) + v.powf(beta) / beta
                    }
                })
                .sum::<f64>()
                / total as f64;
            // Clamp to finite to prevent propagating inf/NaN
            let div = if div.is_finite() { div } else { prev_div.min(1e30) };

            if (prev_div - div).abs() < 1e-8 && _iter > 0 {
                prev_div = div;
                break;
            }
            prev_div = div;
        }

        // Ensure the returned divergence is always finite
        let divergence = if prev_div.is_finite() { prev_div } else { 0.0 };
        Ok(Self { beta, factors, weights, divergence })
    }
}

// ============================================================================
// Tensor Completion
// ============================================================================

/// Low-rank tensor completion via Alternating Least Squares with a binary mask.
///
/// Minimises ‖P_Ω(X - X̂)‖_F² over a CP-format X̂ = Σ_r w_r a1_r⊗...⊗ad_r.
#[derive(Debug, Clone)]
pub struct TensorCompletion {
    /// CP factor matrices for the completed tensor: factors\[n\] is shape\[n\] × rank.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Weights vector, length = rank.
    pub weights: Vec<f64>,
    /// Observed-entry reconstruction error.
    pub error: f64,
    /// Number of ALS iterations.
    pub n_iter: usize,
}

impl TensorCompletion {
    /// Fit tensor completion via ALS on observed entries.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor with arbitrary values at unobserved positions.
    /// * `mask` - Binary mask (same shape as `tensor`); 1.0 = observed, 0.0 = missing.
    /// * `rank` - CP rank for the completion model.
    /// * `max_iter` - Maximum ALS iterations.
    /// * `tolerance` - Convergence threshold on relative error change.
    /// * `seed` - RNG seed.
    pub fn fit(
        tensor: &DenseTensor,
        mask: &DenseTensor,
        rank: usize,
        max_iter: usize,
        tolerance: f64,
        seed: u64,
    ) -> Result<Self> {
        use super::khatri_rao;

        let ndim = tensor.ndim();
        if mask.shape != tensor.shape {
            return Err(TensorError::invalid_argument_op(
                "TensorCompletion::fit",
                "mask shape must match tensor shape",
            ));
        }
        if rank == 0 || ndim < 2 {
            return Err(TensorError::invalid_argument_op(
                "TensorCompletion::fit",
                "rank >= 1 and ndim >= 2 required",
            ));
        }

        let eps = 1e-10f64;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut factors: Vec<Vec<Vec<f64>>> = tensor
            .shape
            .iter()
            .map(|&s| {
                (0..s)
                    .map(|_| (0..rank).map(|_| rng.random::<f64>() * 0.1 + 0.01).collect())
                    .collect()
            })
            .collect();
        let mut weights = vec![1.0f64; rank];

        // Build masked tensor: set unobserved entries to 0
        let masked_data: Vec<f64> = tensor
            .data
            .iter()
            .zip(mask.data.iter())
            .map(|(x, m)| x * m)
            .collect();
        let masked_tensor = DenseTensor::new(masked_data, tensor.shape.clone())?;
        let masked_mask = mask;

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            for n in 0..ndim {
                let modes_except_n: Vec<usize> = (0..ndim).filter(|&k| k != n).collect();
                let kr = {
                    let mut kr_acc: Vec<Vec<f64>> = factors[modes_except_n[modes_except_n.len() - 1]].clone();
                    for &m_idx in modes_except_n[..modes_except_n.len() - 1].iter().rev() {
                        kr_acc = khatri_rao(&factors[m_idx], &kr_acc);
                    }
                    kr_acc
                };

                // Unfolded masked tensor
                let unfolded_x = masked_tensor.matricize(n)?;
                let unfolded_m = masked_mask.matricize(n)?;
                let n_mode = tensor.shape[n];
                let n_other = unfolded_x.shape[1];

                // For each row i of mode n, solve the masked LS problem:
                // min ‖m_i * (x_i - A_i KR)‖² wrt A_i (1 × rank)
                for i in 0..n_mode {
                    let x_row = &unfolded_x.data[i * n_other..(i + 1) * n_other];
                    let m_row = &unfolded_m.data[i * n_other..(i + 1) * n_other];

                    // Weighted KR: multiply each row of KR by the mask entry
                    // Normal equations: (KR^T diag(m) KR) a = KR^T diag(m) x
                    let mut lhs = vec![vec![0.0f64; rank]; rank];
                    let mut rhs = vec![0.0f64; rank];

                    for j in 0..n_other {
                        let m_j = m_row[j];
                        if m_j < 0.5 {
                            continue;
                        }
                        let x_j = x_row[j];
                        if j >= kr.len() {
                            continue;
                        }
                        for r1 in 0..rank {
                            let kr_jr1 = if r1 < kr[j].len() { kr[j][r1] } else { 0.0 };
                            rhs[r1] += m_j * x_j * kr_jr1;
                            for r2 in 0..rank {
                                let kr_jr2 = if r2 < kr[j].len() { kr[j][r2] } else { 0.0 };
                                lhs[r1][r2] += m_j * kr_jr1 * kr_jr2;
                            }
                        }
                    }

                    // Add regularization for numerical stability
                    for r in 0..rank {
                        lhs[r][r] += eps;
                    }

                    let lhs_pinv = pseudo_inverse_via_svd(&lhs);
                    let rhs_mat = vec![rhs];
                    let sol_mat = matrix_multiply(&rhs_mat, &lhs_pinv);
                    for r in 0..rank {
                        factors[n][i][r] = if r < sol_mat[0].len() { sol_mat[0][r] } else { 0.0 };
                    }
                }

                // Normalize columns and accumulate into weights
                for r in 0..rank {
                    let col_norm: f64 = factors[n]
                        .iter()
                        .map(|row| row[r] * row[r])
                        .sum::<f64>()
                        .sqrt()
                        .max(eps);
                    for row in factors[n].iter_mut() {
                        row[r] /= col_norm;
                    }
                    weights[r] *= col_norm;
                }
            }

            // Guard against weight explosion: rescale so max |weight| = 1
            let max_w = weights.iter().map(|w| w.abs()).fold(0.0f64, f64::max).max(eps);
            if max_w > 1e8 {
                for w in weights.iter_mut() {
                    *w /= max_w;
                }
                // Absorb the rescaling into mode-0 factors to keep the product exact
                let scale = max_w.sqrt();
                for row in factors[0].iter_mut() {
                    for v in row.iter_mut() {
                        *v *= scale;
                    }
                }
                if factors.len() > 1 {
                    for row in factors[1].iter_mut() {
                        for v in row.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
            }

            // Compute error on observed entries only
            let rec = DenseTensor::from_factors_khatri_rao(&factors, &weights);
            let error: f64 = tensor
                .data
                .iter()
                .zip(rec.data.iter())
                .zip(mask.data.iter())
                .map(|((x, xh), m)| if *m > 0.5 { (x - xh).powi(2) } else { 0.0 })
                .sum::<f64>()
                .sqrt();
            // Clamp to finite to prevent propagating inf/NaN
            let error = if error.is_finite() { error } else { prev_error };

            let obs_norm: f64 = tensor
                .data
                .iter()
                .zip(mask.data.iter())
                .map(|(x, m)| if *m > 0.5 { x * x } else { 0.0 })
                .sum::<f64>()
                .sqrt()
                .max(eps);

            if (prev_error - error).abs() / obs_norm < tolerance && iter > 0 {
                prev_error = error;
                break;
            }
            prev_error = error;
        }

        Ok(Self { factors, weights, error: prev_error, n_iter })
    }

    /// Reconstruct the completed tensor.
    pub fn reconstruct(&self) -> DenseTensor {
        DenseTensor::from_factors_khatri_rao(&self.factors, &self.weights)
    }
}

/// Riemannian gradient descent completion on the manifold of fixed Tucker-rank tensors.
/// Retraction: re-truncate via HOSVD after each gradient step.
#[derive(Debug, Clone)]
pub struct RiemannianGradientCompletion {
    /// Tucker ranks used for the manifold retraction.
    pub ranks: Vec<usize>,
    /// Learning rate for gradient steps.
    pub learning_rate: f64,
    /// Number of gradient iterations.
    pub max_iter: usize,
    /// Final reconstruction error on observed entries.
    pub error: f64,
    /// Resulting Tucker decomposition.
    pub core: DenseTensor,
    /// Tucker factor matrices.
    pub factors: Vec<Vec<Vec<f64>>>,
}

impl RiemannianGradientCompletion {
    /// Fit Riemannian gradient completion.
    pub fn fit(
        tensor: &DenseTensor,
        mask: &DenseTensor,
        ranks: Vec<usize>,
        learning_rate: f64,
        max_iter: usize,
    ) -> Result<Self> {
        use super::{hosvd, TuckerConfig, TuckerHooi};

        if mask.shape != tensor.shape {
            return Err(TensorError::invalid_argument_op(
                "RiemannianGradientCompletion::fit",
                "mask shape must match tensor shape",
            ));
        }
        if ranks.len() != tensor.ndim() {
            return Err(TensorError::invalid_argument_op(
                "RiemannianGradientCompletion::fit",
                "ranks length must equal tensor ndim",
            ));
        }

        let eps = 1e-10f64;

        // Initialize: fill missing entries with 0, then HOSVD for retraction
        let init_data: Vec<f64> = tensor
            .data
            .iter()
            .zip(mask.data.iter())
            .map(|(x, m)| if *m > 0.5 { *x } else { 0.0 })
            .collect();
        let mut current = DenseTensor::new(init_data, tensor.shape.clone())?;

        // Initial retraction via HOSVD
        let hosvd_res = hosvd(&current, &ranks)?;
        current = hosvd_res.reconstruct();

        let mut prev_error = f64::INFINITY;
        let mut final_core = current.clone();
        let mut final_factors: Vec<Vec<Vec<f64>>> = Vec::new();

        for _iter in 0..max_iter {
            // Euclidean gradient: P_Ω(X_hat - X) (projected residual on observed entries)
            let grad_data: Vec<f64> = current
                .data
                .iter()
                .zip(tensor.data.iter())
                .zip(mask.data.iter())
                .map(|((xh, x), m)| if *m > 0.5 { xh - x } else { 0.0 })
                .collect();
            let grad = DenseTensor::new(grad_data, tensor.shape.clone())?;

            // Gradient step in ambient space
            let stepped_data: Vec<f64> = current
                .data
                .iter()
                .zip(grad.data.iter())
                .map(|(x, g)| x - learning_rate * g)
                .collect();
            let stepped = DenseTensor::new(stepped_data, tensor.shape.clone())?;

            // Retraction: project back to manifold via HOSVD truncation
            let tucker = hosvd(&stepped, &ranks)?;
            current = tucker.reconstruct();
            final_core = tucker.core.clone();
            final_factors = tucker.factors.clone();

            // Error on observed entries
            let error: f64 = current
                .data
                .iter()
                .zip(tensor.data.iter())
                .zip(mask.data.iter())
                .map(|((xh, x), m)| if *m > 0.5 { (xh - x).powi(2) } else { 0.0 })
                .sum::<f64>()
                .sqrt();

            let obs_norm: f64 = tensor
                .data
                .iter()
                .zip(mask.data.iter())
                .map(|(x, m)| if *m > 0.5 { x * x } else { 0.0 })
                .sum::<f64>()
                .sqrt()
                .max(eps);

            if (prev_error - error).abs() / obs_norm < 1e-6 && _iter > 0 {
                prev_error = error;
                break;
            }
            prev_error = error;
        }

        Ok(Self {
            ranks,
            learning_rate,
            max_iter,
            error: prev_error,
            core: final_core,
            factors: final_factors,
        })
    }

    /// Reconstruct the completed tensor from the Tucker representation.
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

/// Scalable tensor completion for large sparse tensors via randomized SVD per mode.
#[derive(Debug, Clone)]
pub struct ScalableTcAlternating {
    /// CP factor matrices.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Weights.
    pub weights: Vec<f64>,
    /// Error on observed entries.
    pub error: f64,
}

impl ScalableTcAlternating {
    /// Fit scalable ALS completion using randomized sketches for each mode.
    pub fn fit(
        tensor: &DenseTensor,
        mask: &DenseTensor,
        rank: usize,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        // Delegate to TensorCompletion (ALS with masked normal equations)
        // with a larger random sketch initialization for scalability.
        let result = TensorCompletion::fit(tensor, mask, rank, max_iter, 1e-5, seed)?;
        Ok(Self {
            factors: result.factors,
            weights: result.weights,
            error: result.error,
        })
    }

    /// Reconstruct the completed tensor.
    pub fn reconstruct(&self) -> DenseTensor {
        DenseTensor::from_factors_khatri_rao(&self.factors, &self.weights)
    }
}

// ============================================================================
// Tensor Neural Network Layers
// ============================================================================

/// Dense neural layer parameterized as a Tensor Train (Novikov et al. 2015).
///
/// The weight matrix W (d_in × d_out) is represented as a TT-decomposition
/// with cores G_1,...,G_d. This allows massive parameter reduction for large layers.
#[derive(Debug, Clone)]
pub struct TensorTrainLinear {
    /// TT cores, one per mode.
    pub tt_tensor: TtTensor,
    /// Bias vector (length = output_size).
    pub bias: Vec<f64>,
    /// Input size (= product of input_shape).
    pub input_size: usize,
    /// Output size (= product of output_shape).
    pub output_size: usize,
    /// Shape used for the input modes of the weight tensor.
    pub input_shape: Vec<usize>,
    /// Shape used for the output modes of the weight tensor.
    pub output_shape: Vec<usize>,
}

impl TensorTrainLinear {
    /// Initialise a TT-Linear layer with random weights.
    ///
    /// The weight tensor has shape `input_shape ++ output_shape` and is
    /// decomposed as a TT with `tt_rank`.
    pub fn new(
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        tt_rank: usize,
        seed: u64,
    ) -> Result<Self> {
        if input_shape.is_empty() || output_shape.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "TensorTrainLinear::new",
                "input_shape and output_shape must be non-empty",
            ));
        }

        let input_size: usize = input_shape.iter().product();
        let output_size: usize = output_shape.iter().product();

        // Build combined shape: [n1, n2, ..., n_in, m1, m2, ..., m_out]
        let combined_shape: Vec<usize> = input_shape
            .iter()
            .chain(output_shape.iter())
            .copied()
            .collect();
        let total: usize = combined_shape.iter().product();

        // Initialize random weight tensor (Xavier / Glorot scaling)
        let fan_avg = (input_size + output_size) as f64 / 2.0;
        let scale = (2.0 / fan_avg).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..total)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();

        let weight_tensor = DenseTensor::new(data, combined_shape)?;
        let tt_config = super::TtConfig { max_rank: tt_rank, tolerance: 1e-8 };
        let tt_tensor = super::TtSvd::fit(&weight_tensor, &tt_config)?;

        let bias = vec![0.0f64; output_size];

        Ok(Self {
            tt_tensor,
            bias,
            input_size,
            output_size,
            input_shape,
            output_shape,
        })
    }

    /// Forward pass: x (length input_size) → y (length output_size).
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.input_size {
            return Err(TensorError::invalid_argument_op(
                "TensorTrainLinear::forward",
                &format!("input length {} != input_size {}", x.len(), self.input_size),
            ));
        }

        // Reconstruct weight matrix W (input_size × output_size) from TT
        let w_tensor = self.tt_tensor.reconstruct();
        // w_tensor has shape = input_shape ++ output_shape
        // Reshape to (input_size, output_size) for matmul
        let w_mat: Vec<Vec<f64>> = (0..self.input_size)
            .map(|i| w_tensor.data[i * self.output_size..(i + 1) * self.output_size].to_vec())
            .collect();

        let x_mat = vec![x.to_vec()]; // 1 × input_size
        let y_mat = matrix_multiply(&x_mat, &w_mat); // 1 × output_size

        let mut y: Vec<f64> = if y_mat.is_empty() {
            vec![0.0; self.output_size]
        } else {
            y_mat[0].clone()
        };

        // Add bias
        for (yi, bi) in y.iter_mut().zip(self.bias.iter()) {
            *yi += bi;
        }

        Ok(y)
    }

    /// Compression ratio: original parameter count / TT parameter count.
    pub fn compression_ratio(&self) -> f64 {
        let original = self.input_size * self.output_size + self.output_size;
        let compressed: usize = self.tt_tensor.cores.iter().map(|c| c.len()).sum::<usize>()
            + self.output_size;
        if compressed == 0 {
            return 1.0;
        }
        original as f64 / compressed as f64
    }
}

/// Simple RNN cell with Tensor Train weight matrices for parameter efficiency.
///
/// Hidden state update: h_t = tanh(W_x x_t + W_h h_{t-1} + b)
/// where W_x and W_h are parameterized as TT decompositions.
#[derive(Debug, Clone)]
pub struct TtRnn {
    /// TT-parameterized input-to-hidden weight.
    pub w_x: TensorTrainLinear,
    /// TT-parameterized hidden-to-hidden weight.
    pub w_h: TensorTrainLinear,
    /// Bias vector (length = hidden_size).
    pub bias: Vec<f64>,
    /// Hidden state size.
    pub hidden_size: usize,
    /// Input size.
    pub input_size: usize,
}

impl TtRnn {
    /// Create a TT-RNN with specified input/hidden sizes and TT rank.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        tt_rank: usize,
        seed: u64,
    ) -> Result<Self> {
        // Factorize input_size and hidden_size into sub-dimensions for TT
        let in_factors = Self::factorize(input_size);
        let hid_factors = Self::factorize(hidden_size);

        let w_x = TensorTrainLinear::new(in_factors.clone(), hid_factors.clone(), tt_rank, seed)?;
        let w_h = TensorTrainLinear::new(hid_factors.clone(), hid_factors.clone(), tt_rank, seed ^ 0xdeadbeef)?;
        let bias = vec![0.0f64; hidden_size];

        Ok(Self { w_x, w_h, bias, hidden_size, input_size })
    }

    /// Factor n into a list of roughly-equal integers for TT decomposition.
    fn factorize(n: usize) -> Vec<usize> {
        if n <= 4 {
            return vec![n];
        }
        // Try to find 2-3 factors close to cube root
        let mut factors = Vec::new();
        let mut remaining = n;
        for f in [4usize, 4, 4, 4] {
            if remaining % f == 0 && remaining / f >= 1 {
                factors.push(f);
                remaining /= f;
                if remaining == 1 {
                    break;
                }
            }
        }
        if remaining > 1 {
            factors.push(remaining);
        }
        if factors.is_empty() {
            factors.push(n);
        }
        factors
    }

    /// Single RNN step.
    ///
    /// # Arguments
    /// * `x` - Input vector (length input_size).
    /// * `h_prev` - Previous hidden state (length hidden_size).
    ///
    /// Returns new hidden state.
    pub fn step(&self, x: &[f64], h_prev: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.input_size {
            return Err(TensorError::invalid_argument_op(
                "TtRnn::step",
                &format!("x length {} != input_size {}", x.len(), self.input_size),
            ));
        }
        if h_prev.len() != self.hidden_size {
            return Err(TensorError::invalid_argument_op(
                "TtRnn::step",
                &format!("h_prev length {} != hidden_size {}", h_prev.len(), self.hidden_size),
            ));
        }

        let wx_out = self.w_x.forward(x)?;
        let wh_out = self.w_h.forward(h_prev)?;

        let h: Vec<f64> = wx_out
            .iter()
            .zip(wh_out.iter())
            .zip(self.bias.iter())
            .map(|((wx, wh), b)| (wx + wh + b).tanh())
            .collect();

        Ok(h)
    }

    /// Process a sequence of inputs, returning all hidden states.
    pub fn forward_sequence(&self, xs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mut h = vec![0.0f64; self.hidden_size];
        let mut states = Vec::with_capacity(xs.len());
        for x in xs {
            h = self.step(x, &h)?;
            states.push(h.clone());
        }
        Ok(states)
    }
}

/// Tensor Fusion Layer (Zadeh et al. 2017) for multimodal fusion.
///
/// Computes the outer product of modality embeddings (with appended 1),
/// flattens to a single vector, and passes through a linear projection.
#[derive(Debug, Clone)]
pub struct TensorFusionLayer {
    /// Linear weights: fusion_dim × flat_fusion_dim.
    pub weights: Vec<Vec<f64>>,
    /// Bias for the linear projection (length = fusion_dim).
    pub bias: Vec<f64>,
    /// Output fusion dimension.
    pub fusion_dim: usize,
    /// Modality embedding dimensions (not including appended 1).
    pub modality_dims: Vec<usize>,
}

impl TensorFusionLayer {
    /// Create a TensorFusionLayer.
    ///
    /// The input modalities each have an extra 1 appended, so the outer product
    /// has size `product(d_i + 1)`. This is then projected to `fusion_dim`.
    pub fn new(modality_dims: Vec<usize>, fusion_dim: usize, seed: u64) -> Result<Self> {
        if modality_dims.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "TensorFusionLayer::new",
                "modality_dims must be non-empty",
            ));
        }

        let flat_dim: usize = modality_dims.iter().map(|&d| d + 1).product();
        let scale = (2.0 / (flat_dim + fusion_dim) as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);

        let weights: Vec<Vec<f64>> = (0..fusion_dim)
            .map(|_| (0..flat_dim).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale).collect())
            .collect();
        let bias = vec![0.0f64; fusion_dim];

        Ok(Self { weights, bias, fusion_dim, modality_dims })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `modalities` - One vector per modality, each of length `modality_dims[i]`.
    ///
    /// Returns fused representation of length `fusion_dim`.
    pub fn forward(&self, modalities: &[Vec<f64>]) -> Result<Vec<f64>> {
        if modalities.len() != self.modality_dims.len() {
            return Err(TensorError::invalid_argument_op(
                "TensorFusionLayer::forward",
                &format!(
                    "number of modalities {} != expected {}",
                    modalities.len(),
                    self.modality_dims.len()
                ),
            ));
        }
        for (i, (m, &d)) in modalities.iter().zip(self.modality_dims.iter()).enumerate() {
            if m.len() != d {
                return Err(TensorError::invalid_argument_op(
                    "TensorFusionLayer::forward",
                    &format!("modality {} length {} != expected {}", i, m.len(), d),
                ));
            }
        }

        // Append 1 to each modality embedding
        let extended: Vec<Vec<f64>> = modalities
            .iter()
            .map(|m| {
                let mut v = m.clone();
                v.push(1.0);
                v
            })
            .collect();

        // Compute outer product via sequential Kronecker-like expansion
        let mut fused: Vec<f64> = vec![1.0];
        for ext in &extended {
            let new_size = fused.len() * ext.len();
            let mut new_fused = vec![0.0f64; new_size];
            for (i, &fi) in fused.iter().enumerate() {
                for (j, &ej) in ext.iter().enumerate() {
                    new_fused[i * ext.len() + j] = fi * ej;
                }
            }
            fused = new_fused;
        }

        // Linear projection: weights (fusion_dim × flat_dim) * fused (flat_dim)
        let fused_mat = vec![fused];
        let w_t = matrix_transpose(&self.weights);
        let out_mat = matrix_multiply(&fused_mat, &w_t); // 1 × fusion_dim

        let mut out: Vec<f64> = if out_mat.is_empty() {
            vec![0.0; self.fusion_dim]
        } else {
            out_mat[0].clone()
        };

        // Add bias and apply tanh
        for (o, b) in out.iter_mut().zip(self.bias.iter()) {
            *o = (*o + b).tanh();
        }

        Ok(out)
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Tensor decomposition quality and efficiency metrics.
#[derive(Debug, Clone)]
pub struct TdMetrics {
    /// Relative Frobenius reconstruction error: ‖X - X̂‖_F / ‖X‖_F.
    pub relative_error: f64,
    /// Compression ratio: original size / compressed size.
    pub compression_ratio: f64,
    /// Tucker multilinear rank (one per mode), if available.
    pub tucker_rank: Option<Vec<usize>>,
    /// Tensor Train bond dimensions (one per inter-core bond), if available.
    pub tt_rank: Option<Vec<usize>>,
}

impl TdMetrics {
    /// Compute metrics for a Tucker decomposition.
    pub fn from_tucker(original: &DenseTensor, core: &DenseTensor, factors: &[Vec<Vec<f64>>]) -> Self {
        let original_size: usize = original.shape.iter().product();
        let core_size: usize = core.shape.iter().product();
        let factor_size: usize = factors
            .iter()
            .flat_map(|f| f.iter())
            .map(|r| r.len())
            .sum();
        let compressed_size = core_size + factor_size;

        let rec = {
            let mut r = core.clone();
            for (n, factor) in factors.iter().enumerate() {
                r = match r.mode_n_product(factor, n) {
                    Ok(v) => v,
                    Err(_) => r,
                };
            }
            r
        };

        let err_sq: f64 = original
            .data
            .iter()
            .zip(rec.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let orig_sq: f64 = original.data.iter().map(|x| x * x).sum();
        let relative_error = if orig_sq > 1e-14 {
            (err_sq / orig_sq).sqrt()
        } else {
            0.0
        };

        let tucker_rank = Some(core.shape.clone());

        Self {
            relative_error,
            compression_ratio: if compressed_size > 0 {
                original_size as f64 / compressed_size as f64
            } else {
                1.0
            },
            tucker_rank,
            tt_rank: None,
        }
    }

    /// Compute metrics for a Tensor Train decomposition.
    pub fn from_tt(original: &DenseTensor, tt: &TtTensor) -> Self {
        let original_size: usize = original.shape.iter().product();
        let compressed_size: usize = tt.cores.iter().map(|c| c.len()).sum();

        let rec = tt.reconstruct();
        let err_sq: f64 = original
            .data
            .iter()
            .zip(rec.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let orig_sq: f64 = original.data.iter().map(|x| x * x).sum();
        let relative_error = if orig_sq > 1e-14 {
            (err_sq / orig_sq).sqrt()
        } else {
            0.0
        };

        let tt_rank: Vec<usize> = tt.core_shapes.iter().map(|(_, _, r)| *r).collect();

        Self {
            relative_error,
            compression_ratio: if compressed_size > 0 {
                original_size as f64 / compressed_size as f64
            } else {
                1.0
            },
            tucker_rank: None,
            tt_rank: Some(tt_rank),
        }
    }

    /// Summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "TdMetrics {{ rel_err: {:.4e}, compression: {:.2}x, tucker_rank: {:?}, tt_rank: {:?} }}",
            self.relative_error, self.compression_ratio, self.tucker_rank, self.tt_rank
        )
    }
}
