//! Extensions to tensor decomposition: Randomized SVD and NMF.

use super::{matrix_multiply, matrix_transpose, qr_cols, svd_via_block_power};
use tenflowers_core::{Result, TensorError};
use scirs2_core::RngExt;

// ============================================================================
// Randomized SVD
// ============================================================================

/// Result of truncated SVD.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors: m × k.
    pub u: Vec<Vec<f64>>,
    /// Singular values: k.
    pub s: Vec<f64>,
    /// Right singular vectors transposed: k × n.
    pub vt: Vec<Vec<f64>>,
}

/// Randomized SVD via random projection + power iteration (Halko et al. 2011).
#[derive(Debug, Clone)]
pub struct RandomizedSvd {
    /// Number of singular components to extract.
    pub n_components: usize,
    /// Number of oversampling columns for the random sketch.
    pub n_oversampling: usize,
    /// Number of power iteration rounds for spectrum sharpening.
    pub n_power_iter: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomizedSvd {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_oversampling: 10,
            n_power_iter: 4,
            seed: 42,
        }
    }
}

impl RandomizedSvd {
    /// Fit and return truncated SVD via random projection.
    /// Input: m×n matrix as row vectors.
    pub fn fit_transform(&self, matrix: &[Vec<f64>]) -> Result<SvdResult> {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        if matrix.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "RandomizedSvd::fit_transform",
                "matrix is empty",
            ));
        }
        let m = matrix.len();
        let n = matrix[0].len();
        let k = self.n_components;
        let p = self.n_oversampling;
        let l = (k + p).min(n).min(m);

        // Step 1: Generate random Gaussian projection matrix Ω (n × l)
        let mut rng = StdRng::seed_from_u64(self.seed);
        let omega: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..l)
                    .map(|_| {
                        // Box-Muller for Gaussian
                        let u1: f64 = (rng.random::<f64>()).max(1e-300);
                        let u2: f64 = rng.random::<f64>();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    })
                    .collect()
            })
            .collect();

        // Step 2: Y = A * Ω  (m × l)
        let mut y = matrix_multiply(matrix, &omega);

        // Step 3: Power iteration with QR re-orthogonalization.
        let a_t = matrix_transpose(matrix);
        for _ in 0..self.n_power_iter {
            let yq = qr_cols(&y, l);
            let z = matrix_multiply(&a_t, &yq);
            let zq = qr_cols(&z, l);
            y = matrix_multiply(matrix, &zq);
        }

        // Step 4: Orthonormalize Y → Q (m × l).
        let q = qr_cols(&y, l);

        // Step 5: B = Q^T * A  (l × n)
        let q_t = matrix_transpose(&q);
        let b = matrix_multiply(&q_t, matrix);

        // Step 6: SVD(B) — returns exactly k = min(k, l, n) orthonormal components.
        let (u_b, s, vt) = svd_via_block_power(&b, k);

        let k_actual = s.len();
        if k_actual == 0 {
            return Err(TensorError::invalid_argument_op(
                "RandomizedSvd::fit_transform",
                "SVD produced zero components",
            ));
        }

        // Step 7: U = Q * U_b  (m × k_actual).
        let l_q = q.first().map(|r| r.len()).unwrap_or(l);
        let mut u_out = vec![vec![0.0f64; k_actual]; m];
        for i in 0..m {
            for j in 0..k_actual {
                for p_idx in 0..l_q {
                    let u_b_pj = if p_idx < u_b.len() && j < u_b[p_idx].len() {
                        u_b[p_idx][j]
                    } else {
                        0.0
                    };
                    u_out[i][j] += q[i][p_idx] * u_b_pj;
                }
            }
        }

        Ok(SvdResult { u: u_out, s, vt })
    }
}

// ============================================================================
// NMF — Nonnegative Matrix Factorization
// ============================================================================

/// Configuration for NMF.
#[derive(Debug, Clone)]
pub struct NmfConfig {
    /// Number of components / rank.
    pub rank: usize,
    /// Maximum number of multiplicative update iterations.
    pub max_iter: usize,
    /// Convergence tolerance on relative change of reconstruction error.
    pub tolerance: f64,
    /// Random seed for initialization.
    pub seed: u64,
}

impl Default for NmfConfig {
    fn default() -> Self {
        Self {
            rank: 2,
            max_iter: 500,
            tolerance: 1e-6,
            seed: 42,
        }
    }
}

/// Result of NMF decomposition.
#[derive(Debug, Clone)]
pub struct NmfDecomposition {
    /// W: m × rank (nonneg).
    pub w: Vec<Vec<f64>>,
    /// H: rank × n (nonneg).
    pub h: Vec<Vec<f64>>,
    /// Frobenius reconstruction error ‖V - WH‖_F.
    pub reconstruction_error: f64,
}

impl NmfDecomposition {
    /// Reconstruct V ≈ W @ H  (m × n).
    pub fn reconstruct(&self) -> Vec<Vec<f64>> {
        matrix_multiply(&self.w, &self.h)
    }
}

/// NMF via Multiplicative Update rules (Lee & Seung 2001).
pub struct NmfMu;

impl NmfMu {
    /// Fit NMF via multiplicative updates.
    pub fn fit(matrix: &[Vec<f64>], config: &NmfConfig) -> Result<NmfDecomposition> {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        if matrix.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NmfMu::fit",
                "matrix is empty",
            ));
        }
        let m = matrix.len();
        let n = matrix[0].len();
        let rank = config.rank;
        if rank == 0 {
            return Err(TensorError::invalid_argument_op(
                "NmfMu::fit",
                "rank must be >= 1",
            ));
        }

        let eps = 1e-10f64;
        let mut rng = StdRng::seed_from_u64(config.seed);

        // Initialize W (m × rank) and H (rank × n) with random nonneg values
        let mut w: Vec<Vec<f64>> = (0..m)
            .map(|_| (0..rank).map(|_| rng.random::<f64>()).collect())
            .collect();
        let mut h: Vec<Vec<f64>> = (0..rank)
            .map(|_| (0..n).map(|_| rng.random::<f64>()).collect())
            .collect();

        let mut prev_error = f64::INFINITY;

        for _iter in 0..config.max_iter {
            // Update H: H ← H * (W^T V) / (W^T W H + ε)
            {
                let wt = matrix_transpose(&w);
                let wt_v = matrix_multiply(&wt, matrix);
                let wt_w = matrix_multiply(&wt, &w);
                let wt_w_h = matrix_multiply(&wt_w, &h);

                for i in 0..rank {
                    for j in 0..n {
                        let num = wt_v[i][j];
                        let den = wt_w_h[i][j] + eps;
                        h[i][j] *= num / den;
                        if h[i][j] < 0.0 {
                            h[i][j] = 0.0;
                        }
                    }
                }
            }

            // Update W: W ← W * (V H^T) / (W H H^T + ε)
            {
                let ht = matrix_transpose(&h);
                let v_ht = matrix_multiply(matrix, &ht);
                let h_ht = matrix_multiply(&h, &ht);
                let w_h_ht = matrix_multiply(&w, &h_ht);

                for i in 0..m {
                    for j in 0..rank {
                        let num = v_ht[i][j];
                        let den = w_h_ht[i][j] + eps;
                        w[i][j] *= num / den;
                        if w[i][j] < 0.0 {
                            w[i][j] = 0.0;
                        }
                    }
                }
            }

            // Compute Frobenius reconstruction error
            let wh = matrix_multiply(&w, &h);
            let error: f64 = matrix
                .iter()
                .zip(wh.iter())
                .flat_map(|(vi, whi)| vi.iter().zip(whi.iter()).map(|(a, b)| (a - b).powi(2)))
                .sum::<f64>()
                .sqrt();

            let rel_change = (prev_error - error).abs() / (prev_error + eps);
            if rel_change < config.tolerance && _iter > 0 {
                prev_error = error;
                break;
            }
            prev_error = error;
        }

        Ok(NmfDecomposition {
            w,
            h,
            reconstruction_error: prev_error,
        })
    }
}
