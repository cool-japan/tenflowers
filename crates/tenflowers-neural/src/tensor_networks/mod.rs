//! Tensor Network Methods for Machine Learning
//!
//! Implements quantum-inspired tensor network architectures for ML:
//!
//! - **MPS (Matrix Product State / Tensor Train)**: Sequence classification with compressed
//!   parameterization using bond dimensions (§1)
//! - **Tucker Layer**: Weight compression via Tucker-2 decomposition G ×₁ U1 ×₂ U2 (§2)
//! - **Tree Tensor Network**: Hierarchical binary-tree contraction for feature extraction (§3)
//! - **MERA Layer**: Multi-scale Entanglement Renormalization Ansatz coarse-graining (§4)
//! - **TN Convolutional Kernel**: Tensor-train parameterized convolution weights (§5)
//! - **Quantum-Inspired Born Machine**: p(x) = |<x|ψ>|² generative model (§6)
//! - **Low-Rank RNN**: RNN with TT-parameterized recurrent weight matrix (§7)
//! - **Entanglement Measures**: von Neumann entropy, Rényi entropy, Schmidt rank (§8)
//! - **Neural Network TN**: Feed-forward net with Tucker decomposition layers (§9)
//! - **TnMetrics**: Fidelity, reconstruction error, bond dimension profiling (§10)
//!
//! All computations use `f64` Vec<Vec<...>>, 100% pure Rust, no ndarray/rand.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ============================================================================
// §0 Error type
// ============================================================================

/// Errors that can occur in tensor network operations.
#[derive(Debug, Clone)]
pub enum TnError {
    /// Incompatible dimensions for a contraction or operation.
    DimensionMismatch(String),
    /// Numerical issue (NaN, Inf, singular matrix, etc.).
    NumericalError(String),
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl std::fmt::Display for TnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TnError::DimensionMismatch(m) => write!(f, "TN dimension mismatch: {}", m),
            TnError::NumericalError(m) => write!(f, "TN numerical error: {}", m),
            TnError::InvalidConfig(m) => write!(f, "TN invalid config: {}", m),
        }
    }
}

impl std::error::Error for TnError {}

// ============================================================================
// Internal math helpers
// ============================================================================

/// Matrix multiplication (a: m×k) × (b: k×n) → (m×n).
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i][p];
            if a_ip.abs() < 1e-300 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a_ip * b[p][j];
            }
        }
    }
    c
}

/// Matrix-vector multiplication: (m×n) matrix times n-vector → m-vector.
fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Transpose a matrix: (m×n) → (n×m).
fn mat_t(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Gram-Schmidt orthonormalization of columns of matrix `a` (m×n), returns Q (m×n).
fn gram_schmidt(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut q = vec![vec![0.0_f64; n]; m];
    let mut qs: Vec<Vec<f64>> = Vec::with_capacity(n); // orthonormal columns so far

    for j in 0..n {
        // column j of a
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
        // subtract projections
        for qk in &qs {
            let dot: f64 = v.iter().zip(qk.iter()).map(|(vi, qi)| vi * qi).sum();
            for i in 0..m {
                v[i] -= dot * qk[i];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            let col: Vec<f64> = v.iter().map(|x| x * inv).collect();
            for i in 0..m {
                q[i][j] = col[i];
            }
            qs.push(col);
        }
    }
    q
}

/// Truncated SVD (top-k) via block power iteration with seed.
/// Returns (U [m×k], S [k], Vt [k×n]).
fn svd_truncated_seeded(
    a: &[Vec<f64>],
    k: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    if a.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let m = a.len();
    let n = a[0].len();
    let k_eff = k.min(m).min(n).max(1);
    let mut rng = StdRng::seed_from_u64(seed);

    // random initial matrix V: n×k_eff
    let mut v_mat: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            (0..k_eff)
                .map(|_| {
                    let u: f64 = rng.random();
                    u * 2.0 - 1.0
                })
                .collect()
        })
        .collect();

    // orthonormalize columns of v_mat
    let v_col_first: Vec<Vec<f64>> = (0..k_eff)
        .map(|j| (0..n).map(|i| v_mat[i][j]).collect())
        .collect();
    let q_cols_first = {
        // Build matrix n×k_eff for gram_schmidt
        let mat: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..k_eff).map(|j| v_col_first[j][i]).collect())
            .collect();
        gram_schmidt(&mat)
    };
    // update v_mat from q_cols_first
    for i in 0..n {
        for j in 0..k_eff {
            v_mat[i][j] = q_cols_first[i][j];
        }
    }

    // Power iteration: V ← orth(A^T A V)
    for _ in 0..30 {
        // U_tmp = A V: m×k_eff
        let mut u_tmp = vec![vec![0.0_f64; k_eff]; m];
        for i in 0..m {
            for p in 0..n {
                let a_ip = a[i][p];
                for j in 0..k_eff {
                    u_tmp[i][j] += a_ip * v_mat[p][j];
                }
            }
        }
        // V_new = A^T U_tmp: n×k_eff
        let mut v_new = vec![vec![0.0_f64; k_eff]; n];
        for p in 0..n {
            for i in 0..m {
                let at_pi = a[i][p];
                for j in 0..k_eff {
                    v_new[p][j] += at_pi * u_tmp[i][j];
                }
            }
        }
        // Orthonormalize columns of v_new
        let v_gs = gram_schmidt(&v_new);
        v_mat = v_gs;
    }

    // Compute U = A V / sigma
    let mut u_mat = vec![vec![0.0_f64; k_eff]; m];
    for i in 0..m {
        for p in 0..n {
            let a_ip = a[i][p];
            for j in 0..k_eff {
                u_mat[i][j] += a_ip * v_mat[p][j];
            }
        }
    }
    // Singular values = norms of columns of U
    let mut sigma = vec![0.0_f64; k_eff];
    for j in 0..k_eff {
        let norm: f64 = (0..m)
            .map(|i| u_mat[i][j] * u_mat[i][j])
            .sum::<f64>()
            .sqrt();
        sigma[j] = norm;
        if norm > 1e-14 {
            let inv = 1.0 / norm;
            for i in 0..m {
                u_mat[i][j] *= inv;
            }
        }
    }
    // Vt = V^T: k_eff×n
    let vt = mat_t(&v_mat);

    (u_mat, sigma, vt)
}

/// Standard truncated SVD with default seed.
fn svd_truncated(a: &[Vec<f64>], k: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    svd_truncated_seeded(a, k, 0xdeadbeef_cafebabe)
}

/// Generate random f64 in range with a given rng.
fn rand_normal(rng: &mut impl Rng) -> f64 {
    // Box-Muller
    let u1: f64 = rng.random::<f64>().max(1e-14);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ============================================================================
// §1 TnMatrixProductState (MPS / Tensor Train)
// ============================================================================

/// A single site in a Matrix Product State.
/// Tensor has shape [chi_left × d × chi_right].
#[derive(Debug, Clone)]
pub struct TnMpsSite {
    /// Flattened tensor: index order \[chi_left\]\[d\]\[chi_right\].
    pub tensor: Vec<Vec<Vec<f64>>>,
    /// Left bond dimension.
    pub chi_left: usize,
    /// Physical dimension (feature space).
    pub d: usize,
    /// Right bond dimension.
    pub chi_right: usize,
}

impl TnMpsSite {
    /// Create a new site with given dimensions, initialized to zero.
    pub fn zeros(chi_left: usize, d: usize, chi_right: usize) -> Self {
        TnMpsSite {
            tensor: vec![vec![vec![0.0_f64; chi_right]; d]; chi_left],
            chi_left,
            d,
            chi_right,
        }
    }

    /// Get the [chi_left × chi_right] matrix for physical index `s`.
    pub fn get_slice(&self, s: usize) -> Vec<Vec<f64>> {
        (0..self.chi_left)
            .map(|l| (0..self.chi_right).map(|r| self.tensor[l][s][r]).collect())
            .collect()
    }
}

/// Matrix Product State (MPS / Tensor Train) for sequence data.
///
/// Models sequences of length `n_sites` where each position carries
/// a `d`-dimensional feature vector. The contraction produces a
/// `chi_right`-dimensional output vector (for the last site, chi_right=1
/// or a small output dimension).
#[derive(Debug, Clone)]
pub struct TnMatrixProductState {
    /// MPS sites.
    pub sites: Vec<TnMpsSite>,
    /// Number of sites.
    pub n_sites: usize,
    /// Physical dimension per site.
    pub d: usize,
    /// Maximum bond dimension.
    pub chi_max: usize,
}

impl TnMatrixProductState {
    /// Create a new MPS with random Xavier-like initialization.
    ///
    /// # Arguments
    /// - `n_sites`: sequence length
    /// - `d`: physical dimension (feature size per site)
    /// - `chi_max`: maximum bond dimension
    pub fn new(n_sites: usize, d: usize, chi_max: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0x4d505304_c0ffee42);
        let mut sites = Vec::with_capacity(n_sites);
        for i in 0..n_sites {
            let chi_left = if i == 0 { 1 } else { chi_max };
            let chi_right = if i == n_sites - 1 { 1 } else { chi_max };
            let scale = (2.0 / (chi_left * d + d * chi_right) as f64).sqrt();
            let mut site = TnMpsSite::zeros(chi_left, d, chi_right);
            for l in 0..chi_left {
                for s in 0..d {
                    for r in 0..chi_right {
                        site.tensor[l][s][r] = rand_normal(&mut rng) * scale;
                    }
                }
            }
            sites.push(site);
        }
        TnMatrixProductState {
            sites,
            n_sites,
            d,
            chi_max,
        }
    }

    /// Contract MPS with input sequence.
    ///
    /// For each site `i`, selects the tensor slice A_i\[x_i\] (a chi_left×chi_right matrix)
    /// using the feature vector `input[i]` as a soft selector (dot product with d-dim axis).
    /// Returns the final contracted vector (length = chi_right of last site = 1 or output_dim).
    pub fn contract_all(&self, input: &[Vec<f64>]) -> Result<Vec<f64>, TnError> {
        if input.len() != self.n_sites {
            return Err(TnError::DimensionMismatch(format!(
                "input length {} != n_sites {}",
                input.len(),
                self.n_sites
            )));
        }
        for (i, xi) in input.iter().enumerate() {
            if xi.len() != self.d {
                return Err(TnError::DimensionMismatch(format!(
                    "input[{}] length {} != d {}",
                    i,
                    xi.len(),
                    self.d
                )));
            }
        }

        // Start with boundary vector [1.0] (chi=1 on left boundary)
        let mut boundary = vec![1.0_f64];

        for (i, site) in self.sites.iter().enumerate() {
            let xi = &input[i];
            // Compute effective matrix M = sum_s xi[s] * A[l][s][r]
            // Shape: [chi_left × chi_right]
            let chi_l = site.chi_left;
            let chi_r = site.chi_right;
            let mut m = vec![vec![0.0_f64; chi_r]; chi_l];
            for s in 0..site.d {
                let w = xi[s];
                if w.abs() < 1e-300 {
                    continue;
                }
                for l in 0..chi_l {
                    for r in 0..chi_r {
                        m[l][r] += w * site.tensor[l][s][r];
                    }
                }
            }
            // boundary [chi_l] × m [chi_l × chi_r] → new_boundary [chi_r]
            let mut new_boundary = vec![0.0_f64; chi_r];
            for l in 0..chi_l {
                let b = boundary[l];
                for r in 0..chi_r {
                    new_boundary[r] += b * m[l][r];
                }
            }
            boundary = new_boundary;
        }
        Ok(boundary)
    }

    /// Perform a left-to-right SVD sweep to compress bond dimensions.
    ///
    /// Uses power-iteration SVD to truncate each bond to `max_chi` singular values.
    pub fn svd_compress(&mut self, max_chi: usize) -> Result<(), TnError> {
        // Left-to-right sweep: at each bond, reshape site tensor, SVD, truncate, push R to next site
        for i in 0..self.n_sites.saturating_sub(1) {
            let site = &self.sites[i];
            let chi_l = site.chi_left;
            let d = site.d;
            let chi_r = site.chi_right;

            // Reshape site tensor to (chi_l * d) × chi_r matrix
            let rows = chi_l * d;
            let cols = chi_r;
            let mut mat = vec![vec![0.0_f64; cols]; rows];
            for l in 0..chi_l {
                for s in 0..d {
                    for r in 0..chi_r {
                        mat[l * d + s][r] = site.tensor[l][s][r];
                    }
                }
            }

            // Truncated SVD: keep at most max_chi singular values
            let k = max_chi.min(rows).min(cols);
            let (u, sigma, vt) = self.svd_2d(&mat);
            let k_trunc = k.min(sigma.len());

            // New chi_right for site i
            let new_chi_r = k_trunc;

            // U[:, :k_trunc]: rows × k_trunc → reshape to chi_l × d × new_chi_r
            let mut new_tensor = vec![vec![vec![0.0_f64; new_chi_r]; d]; chi_l];
            for l in 0..chi_l {
                for s in 0..d {
                    for r in 0..new_chi_r {
                        new_tensor[l][s][r] = u[l * d + s][r];
                    }
                }
            }

            // S * Vt[:k_trunc, :]: new_chi_r × chi_r (diagonal scale)
            let mut sv = vec![vec![0.0_f64; chi_r]; new_chi_r];
            for r in 0..new_chi_r {
                let s_val = sigma[r];
                for c in 0..chi_r {
                    sv[r][c] = s_val * vt[r][c];
                }
            }

            // Update site i
            self.sites[i].tensor = new_tensor;
            self.sites[i].chi_right = new_chi_r;

            // Contract sv into site i+1: absorb from left
            // site i+1 tensor shape: [old_chi_l × d × chi_r_next]
            // new site i+1: [new_chi_r × d × chi_r_next]
            let site_next = &self.sites[i + 1];
            let old_chi_l_next = site_next.chi_left;
            let d_next = site_next.d;
            let chi_r_next = site_next.chi_right;
            let mut new_tensor_next = vec![vec![vec![0.0_f64; chi_r_next]; d_next]; new_chi_r];
            for new_l in 0..new_chi_r {
                for s in 0..d_next {
                    for r in 0..chi_r_next {
                        let mut val = 0.0_f64;
                        for old_l in 0..old_chi_l_next {
                            if old_l < sv[new_l].len() {
                                val += sv[new_l][old_l] * site_next.tensor[old_l][s][r];
                            }
                        }
                        new_tensor_next[new_l][s][r] = val;
                    }
                }
            }
            self.sites[i + 1].tensor = new_tensor_next;
            self.sites[i + 1].chi_left = new_chi_r;
        }
        self.chi_max = max_chi;
        Ok(())
    }

    /// Internal power-iteration SVD for a 2D matrix. Returns (U, S, Vt).
    pub fn svd_2d(&self, m: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
        let k = m.len().min(if m.is_empty() { 0 } else { m[0].len() });
        svd_truncated(m, k)
    }

    /// Return the bond dimensions [chi between site i and site i+1] for i in 0..n_sites-1.
    pub fn bond_dimensions(&self) -> Vec<usize> {
        (0..self.n_sites.saturating_sub(1))
            .map(|i| self.sites[i].chi_right)
            .collect()
    }

    /// Total number of parameters in all site tensors.
    pub fn total_params(&self) -> usize {
        self.sites
            .iter()
            .map(|s| s.chi_left * s.d * s.chi_right)
            .sum()
    }
}

// ============================================================================
// §2 TnTuckerLayer
// ============================================================================

/// A linear layer whose weight matrix is represented in Tucker-2 form:
/// W ≈ U2 @ G @ U1^T, where G is [r2 × r1], U1 is [in_dim × r1], U2 is [out_dim × r2].
///
/// Forward: y = U2 @ G @ U1^T @ x + bias
#[derive(Debug, Clone)]
pub struct TnTuckerLayer {
    /// Core matrix [r2 × r1].
    pub g: Vec<Vec<f64>>,
    /// Factor for input mode [in_dim × r1].
    pub u1: Vec<Vec<f64>>,
    /// Factor for output mode [out_dim × r2].
    pub u2: Vec<Vec<f64>>,
    /// Bias \[out_dim\].
    pub bias: Vec<f64>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Rank for input mode.
    pub r1: usize,
    /// Rank for output mode.
    pub r2: usize,
}

impl TnTuckerLayer {
    /// Create a new Tucker layer with random orthogonal factors and small core.
    pub fn new(in_dim: usize, out_dim: usize, rank: usize) -> Self {
        let r1 = rank.min(in_dim);
        let r2 = rank.min(out_dim);
        let mut rng = StdRng::seed_from_u64(0x5475636b_65724c79);

        // Random matrices for orthogonalization
        let u1_raw: Vec<Vec<f64>> = (0..in_dim)
            .map(|_| (0..r1).map(|_| rand_normal(&mut rng)).collect())
            .collect();
        let u2_raw: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| (0..r2).map(|_| rand_normal(&mut rng)).collect())
            .collect();

        let u1 = gram_schmidt(&u1_raw);
        let u2 = gram_schmidt(&u2_raw);

        let g_scale = (2.0 / (r1 + r2) as f64).sqrt();
        let g: Vec<Vec<f64>> = (0..r2)
            .map(|_| (0..r1).map(|_| rand_normal(&mut rng) * g_scale).collect())
            .collect();

        let bias = vec![0.0_f64; out_dim];
        TnTuckerLayer {
            g,
            u1,
            u2,
            bias,
            in_dim,
            out_dim,
            r1,
            r2,
        }
    }

    /// Forward pass: y = U2 @ G @ U1^T @ x + bias.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // Step 1: z1 = U1^T @ x: r1-vector
        let u1t = mat_t(&self.u1);
        let z1 = mat_vec(&u1t, x);
        // Step 2: z2 = G @ z1: r2-vector
        let z2 = mat_vec(&self.g, &z1);
        // Step 3: y = U2 @ z2 + bias: out_dim-vector
        let z3 = mat_vec(&self.u2, &z2);
        z3.iter()
            .zip(self.bias.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Compute compression ratio vs. equivalent dense layer.
    /// ratio = (compressed params) / (in_dim * out_dim)
    pub fn compression_ratio(&self) -> f64 {
        let compressed = self.r1 * self.r2 + self.in_dim * self.r1 + self.out_dim * self.r2;
        compressed as f64 / (self.in_dim * self.out_dim) as f64
    }

    /// Create a Tucker layer by approximating a dense weight matrix using truncated SVD.
    ///
    /// Uses SVD: W ≈ U_trunc @ diag(S) @ Vt_trunc, then absorbs sqrt(S) into both factors.
    pub fn from_dense(w: &[Vec<f64>], rank: usize) -> Self {
        let out_dim = w.len();
        let in_dim = if w.is_empty() { 0 } else { w[0].len() };
        let r = rank.min(out_dim).min(in_dim).max(1);

        let (u, s, vt) = svd_truncated(w, r);
        // r_eff might be less than r if matrix is small
        let r_eff = s.len().min(r);

        // U2 = U[:, :r_eff]: out_dim × r_eff
        let u2: Vec<Vec<f64>> = (0..out_dim)
            .map(|i| (0..r_eff).map(|j| u[i][j]).collect())
            .collect();
        // U1^T = Vt[:r_eff, :]: r_eff × in_dim → U1 = Vt^T: in_dim × r_eff
        let u1: Vec<Vec<f64>> = (0..in_dim)
            .map(|j| (0..r_eff).map(|k| vt[k][j]).collect())
            .collect();
        // G = diag(S): r_eff × r_eff
        let g: Vec<Vec<f64>> = (0..r_eff)
            .map(|i| {
                let mut row = vec![0.0_f64; r_eff];
                row[i] = s[i];
                row
            })
            .collect();

        let bias = vec![0.0_f64; out_dim];
        TnTuckerLayer {
            g,
            u1,
            u2,
            bias,
            in_dim,
            out_dim,
            r1: r_eff,
            r2: r_eff,
        }
    }
}

// ============================================================================
// §3 TnTreeTensorNetwork
// ============================================================================

/// An internal node of a binary tree tensor network.
/// Contracts two children (each of `bond_dim` dimension) into an output of `out_dim`.
#[derive(Debug, Clone)]
pub struct TnTreeNode {
    /// Weight tensor [left_dim × right_dim × out_dim].
    pub tensor: Vec<Vec<Vec<f64>>>,
    /// Left child dimension.
    pub left_dim: usize,
    /// Right child dimension.
    pub right_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl TnTreeNode {
    /// Create a tree node with random Xavier-like initialization.
    pub fn new(left_dim: usize, right_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (left_dim * right_dim + out_dim) as f64).sqrt();
        let tensor: Vec<Vec<Vec<f64>>> = (0..left_dim)
            .map(|_| {
                (0..right_dim)
                    .map(|_| (0..out_dim).map(|_| rand_normal(rng) * scale).collect())
                    .collect()
            })
            .collect();
        TnTreeNode {
            tensor,
            left_dim,
            right_dim,
            out_dim,
        }
    }

    /// Contract left and right vectors through this node.
    /// left: \[left_dim\], right: \[right_dim\] → out: \[out_dim\]
    pub fn contract(&self, left: &[f64], right: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.out_dim];
        for l in 0..self.left_dim {
            for r in 0..self.right_dim {
                let lr = left[l] * right[r];
                if lr.abs() < 1e-300 {
                    continue;
                }
                for k in 0..self.out_dim {
                    out[k] += lr * self.tensor[l][r][k];
                }
            }
        }
        out
    }
}

/// Hierarchical binary tree tensor network.
///
/// Leaves embed `in_dim`-dimensional inputs into `bond_dim` vectors.
/// Pairs of leaves are merged by internal nodes; this continues up the binary tree.
/// The root produces a final `out_dim`-dimensional output.
///
/// Requires `n_leaves` to be a power of 2.
#[derive(Debug, Clone)]
pub struct TnTreeTensorNetwork {
    /// Leaf embedding matrices [n_leaves × in_dim × bond_dim] stored flat.
    pub leaves: Vec<Vec<Vec<f64>>>,
    /// Internal tree nodes stored level by level (bottom-up).
    pub internal_nodes: Vec<TnTreeNode>,
    /// Number of leaves (must be power of 2).
    pub n_leaves: usize,
    /// Input dimension per leaf.
    pub in_dim: usize,
    /// Bond dimension (output of leaf embedding, and intermediate node outputs).
    pub bond_dim: usize,
    /// Final output dimension (output of root node).
    pub out_dim: usize,
}

impl TnTreeTensorNetwork {
    /// Create a new tree tensor network.
    ///
    /// # Errors
    /// Returns `TnError::InvalidConfig` if `n_leaves` is not a power of 2 or is 0.
    pub fn new(
        n_leaves: usize,
        in_dim: usize,
        bond_dim: usize,
        out_dim: usize,
    ) -> Result<Self, TnError> {
        if n_leaves == 0 || (n_leaves & (n_leaves - 1)) != 0 {
            return Err(TnError::InvalidConfig(format!(
                "n_leaves={} must be a power of 2 and > 0",
                n_leaves
            )));
        }
        let mut rng = StdRng::seed_from_u64(0x54726565_544e4e00);
        let scale_leaf = (2.0 / (in_dim + bond_dim) as f64).sqrt();

        // Leaf embeddings: [n_leaves][in_dim × bond_dim]
        let leaves: Vec<Vec<Vec<f64>>> = (0..n_leaves)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        (0..bond_dim)
                            .map(|_| rand_normal(&mut rng) * scale_leaf)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Build internal nodes bottom-up
        // depth = log2(n_leaves) levels
        let depth = n_leaves.trailing_zeros() as usize;
        let mut internal_nodes = Vec::new();
        let mut level_out = bond_dim;
        let mut level_size = n_leaves / 2;

        for level in 0..depth {
            let node_out = if level == depth - 1 {
                out_dim
            } else {
                bond_dim
            };
            for _ in 0..level_size {
                internal_nodes.push(TnTreeNode::new(level_out, level_out, node_out, &mut rng));
            }
            level_out = node_out;
            level_size /= 2;
        }

        Ok(TnTreeTensorNetwork {
            leaves,
            internal_nodes,
            n_leaves,
            in_dim,
            bond_dim,
            out_dim,
        })
    }

    /// Forward pass: embed inputs, then bottom-up tree contraction.
    ///
    /// # Errors
    /// Returns `TnError::DimensionMismatch` if input count or dimensions are wrong.
    pub fn forward(&self, inputs: &[Vec<f64>]) -> Result<Vec<f64>, TnError> {
        if inputs.len() != self.n_leaves {
            return Err(TnError::DimensionMismatch(format!(
                "inputs.len()={} != n_leaves={}",
                inputs.len(),
                self.n_leaves
            )));
        }
        for (i, xi) in inputs.iter().enumerate() {
            if xi.len() != self.in_dim {
                return Err(TnError::DimensionMismatch(format!(
                    "inputs[{}].len()={} != in_dim={}",
                    i,
                    xi.len(),
                    self.in_dim
                )));
            }
        }

        // Embed leaves: leaf_mat [in_dim × bond_dim] applied to input xi
        let mut current_level: Vec<Vec<f64>> = inputs
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                // y = leaf[i]^T @ xi: bond_dim vector
                let leaf = &self.leaves[i]; // [in_dim × bond_dim]
                (0..self.bond_dim)
                    .map(|j| (0..self.in_dim).map(|k| leaf[k][j] * xi[k]).sum())
                    .collect()
            })
            .collect();

        // Bottom-up tree contraction
        let depth = self.n_leaves.trailing_zeros() as usize;
        let mut node_idx = 0usize;
        let mut level_size = self.n_leaves / 2;

        for _level in 0..depth {
            let mut next_level = Vec::with_capacity(level_size);
            for pair in 0..level_size {
                let left = &current_level[pair * 2];
                let right = &current_level[pair * 2 + 1];
                let out = self.internal_nodes[node_idx].contract(left, right);
                next_level.push(out);
                node_idx += 1;
            }
            current_level = next_level;
            level_size /= 2;
        }

        // current_level should have exactly 1 element = root output
        current_level
            .into_iter()
            .next()
            .ok_or_else(|| TnError::NumericalError("empty tree output".into()))
    }

    /// Depth of the binary tree = log2(n_leaves).
    pub fn depth(&self) -> usize {
        self.n_leaves.trailing_zeros() as usize
    }
}

// ============================================================================
// §4 TnMeraLayer
// ============================================================================

/// One layer of MERA (Multi-scale Entanglement Renormalization Ansatz).
///
/// Consists of alternating disentanglers (unitary on pairs of sites)
/// and isometries (coarse-graining from 2 sites to 1).
/// Each disentangler is a d²×d² matrix (applied to combined 2-site state).
/// Each isometry is a d²×d matrix (projecting 2-site state into 1-site).
#[derive(Debug, Clone)]
pub struct TnMeraLayer {
    /// Disentanglers: each [d^2 × d^2], stored as list of matrices.
    pub disentanglers: Vec<Vec<Vec<f64>>>,
    /// Isometries: each [d^2 × d], stored as list of matrices.
    pub isometries: Vec<Vec<f64>>,
    /// Physical dimension per site.
    pub d: usize,
}

impl TnMeraLayer {
    /// Create a new MERA layer.
    ///
    /// # Arguments
    /// - `n_sites`: number of sites in the input state (must be even)
    /// - `d`: local Hilbert space dimension
    ///
    /// # Errors
    /// Returns `TnError::InvalidConfig` if `n_sites` is odd or zero.
    pub fn new(n_sites: usize, d: usize) -> Result<Self, TnError> {
        if n_sites == 0 || n_sites % 2 != 0 {
            return Err(TnError::InvalidConfig(format!(
                "n_sites={} must be positive and even",
                n_sites
            )));
        }
        let d2 = d * d;
        let n_pairs = n_sites / 2;
        let mut rng = StdRng::seed_from_u64(0x4d455241_6c617965);

        // Disentanglers: n_pairs of d2×d2 matrices, initialized to identity + small noise
        let disentanglers: Vec<Vec<Vec<f64>>> = (0..n_pairs)
            .map(|_| {
                let mut mat = vec![vec![0.0_f64; d2]; d2];
                for k in 0..d2 {
                    mat[k][k] = 1.0;
                }
                // Add small noise
                for i in 0..d2 {
                    for j in 0..d2 {
                        mat[i][j] += rand_normal(&mut rng) * 0.01;
                    }
                }
                mat
            })
            .collect();

        // Isometries: n_pairs of d2×d matrices, initialized via random Gram-Schmidt
        let isometries: Vec<Vec<f64>> = (0..n_pairs)
            .map(|_| {
                // Random d2×d matrix then column-orthonormalize
                let raw: Vec<Vec<f64>> = (0..d2)
                    .map(|_| (0..d).map(|_| rand_normal(&mut rng)).collect())
                    .collect();
                let q = gram_schmidt(&raw);
                // flatten to 1D (d2×d), row-major
                (0..d2 * d)
                    .map(|idx| {
                        let row = idx / d;
                        let col = idx % d;
                        q[row][col]
                    })
                    .collect()
            })
            .collect();

        Ok(TnMeraLayer {
            disentanglers,
            isometries,
            d,
        })
    }

    /// Forward pass: apply disentanglers to pairs, then isometries to coarse-grain.
    ///
    /// Input state: [n_sites × d] → output: [n_sites/2 × d]
    pub fn forward(&self, state: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TnError> {
        self.coarsen(state)
    }

    /// Coarsen state from n_sites → n_sites/2.
    ///
    /// For each pair (2i, 2i+1):
    /// 1. Combine into d²-vector via outer product flattening.
    /// 2. Apply disentangler (d²×d² matrix multiplication).
    /// 3. Apply isometry (d²×d matrix) to project to d-vector.
    pub fn coarsen(&self, state: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TnError> {
        let n_sites = state.len();
        if n_sites == 0 || n_sites % 2 != 0 {
            return Err(TnError::InvalidConfig(format!(
                "state length {} must be positive and even",
                n_sites
            )));
        }
        let n_pairs = n_sites / 2;
        if self.disentanglers.len() < n_pairs || self.isometries.len() < n_pairs {
            return Err(TnError::DimensionMismatch(format!(
                "MERA layer has {} disentanglers but need {}",
                self.disentanglers.len(),
                n_pairs
            )));
        }
        for (i, site) in state.iter().enumerate() {
            if site.len() != self.d {
                return Err(TnError::DimensionMismatch(format!(
                    "state[{}].len()={} != d={}",
                    i,
                    site.len(),
                    self.d
                )));
            }
        }

        let d = self.d;
        let d2 = d * d;
        let mut output = Vec::with_capacity(n_pairs);

        for pair in 0..n_pairs {
            let s0 = &state[2 * pair];
            let s1 = &state[2 * pair + 1];

            // Outer product: combined[i*d + j] = s0[i] * s1[j]
            let mut combined = vec![0.0_f64; d2];
            for i in 0..d {
                for j in 0..d {
                    combined[i * d + j] = s0[i] * s1[j];
                }
            }

            // Apply disentangler
            let dis = &self.disentanglers[pair];
            let disentangled = mat_vec(dis, &combined);

            // Apply isometry: d2 × d stored flat row-major
            let iso = &self.isometries[pair];
            let mut out = vec![0.0_f64; d];
            for j in 0..d {
                for k in 0..d2 {
                    out[j] += iso[k * d + j] * disentangled[k];
                }
            }
            output.push(out);
        }
        Ok(output)
    }
}

// ============================================================================
// §5 TnConvolutionalKernel
// ============================================================================

/// Tensor-network parameterized convolutional kernel.
///
/// The full weight tensor W[out_ch, in_ch, k] is expressed as a tensor train:
/// W[o, i, s] ≈ sum_{a,b} core1\[a,b\] * core2\[b,i,s\] * core3\[a,o\]
///
/// This gives a compressed representation when `rank` << min(out_ch, in_ch * k).
#[derive(Debug, Clone)]
pub struct TnConvolutionalKernel {
    /// Core 1: [out_rank × in_rank]
    pub core1: Vec<Vec<f64>>,
    /// Core 2: [in_rank × in_channels × kernel_size]
    pub core2: Vec<Vec<Vec<f64>>>,
    /// Core 3: [out_rank × out_channels]
    pub core3: Vec<Vec<f64>>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolutional kernel size.
    pub kernel_size: usize,
    /// Bond dimension for input mode.
    pub in_rank: usize,
    /// Bond dimension for output mode.
    pub out_rank: usize,
}

impl TnConvolutionalKernel {
    /// Create a new TN convolutional kernel with random initialization.
    pub fn new(in_ch: usize, out_ch: usize, k: usize, rank: usize) -> Self {
        let in_rank = rank.min(in_ch * k);
        let out_rank = rank.min(out_ch);
        let mut rng = StdRng::seed_from_u64(0x436f6e764b65726e);

        let scale1 = (2.0 / (out_rank + in_rank) as f64).sqrt();
        let core1: Vec<Vec<f64>> = (0..out_rank)
            .map(|_| {
                (0..in_rank)
                    .map(|_| rand_normal(&mut rng) * scale1)
                    .collect()
            })
            .collect();

        let scale2 = (2.0 / (in_rank + in_ch * k) as f64).sqrt();
        let core2: Vec<Vec<Vec<f64>>> = (0..in_rank)
            .map(|_| {
                (0..in_ch)
                    .map(|_| (0..k).map(|_| rand_normal(&mut rng) * scale2).collect())
                    .collect()
            })
            .collect();

        let scale3 = (2.0 / (out_rank + out_ch) as f64).sqrt();
        let core3: Vec<Vec<f64>> = (0..out_rank)
            .map(|_| {
                (0..out_ch)
                    .map(|_| rand_normal(&mut rng) * scale3)
                    .collect()
            })
            .collect();

        TnConvolutionalKernel {
            core1,
            core2,
            core3,
            in_channels: in_ch,
            out_channels: out_ch,
            kernel_size: k,
            in_rank,
            out_rank,
        }
    }

    /// Reconstruct the full weight tensor W[out_ch × in_ch × kernel_size].
    pub fn materialize(&self) -> Vec<Vec<Vec<f64>>> {
        let oc = self.out_channels;
        let ic = self.in_channels;
        let k = self.kernel_size;
        let mut w = vec![vec![vec![0.0_f64; k]; ic]; oc];

        for o in 0..oc {
            for i in 0..ic {
                for s in 0..k {
                    let mut val = 0.0_f64;
                    for a in 0..self.out_rank {
                        for b in 0..self.in_rank {
                            val += self.core1[a][b] * self.core2[b][i][s] * self.core3[a][o];
                        }
                    }
                    w[o][i][s] = val;
                }
            }
        }
        w
    }

    /// 1D convolution forward pass with same padding.
    ///
    /// Input: [in_ch × length] → Output: [out_ch × length]
    pub fn forward_1d(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let in_ch = self.in_channels;
        let out_ch = self.out_channels;
        let k = self.kernel_size;
        let pad = k / 2;
        let length = if input.is_empty() { 0 } else { input[0].len() };

        // Materialize kernel for efficiency
        let w = self.materialize();

        let mut output = vec![vec![0.0_f64; length]; out_ch];
        for o in 0..out_ch {
            for t in 0..length {
                let mut acc = 0.0_f64;
                for i in 0..in_ch {
                    if i >= input.len() {
                        continue;
                    }
                    for s in 0..k {
                        let t_in = t + s;
                        if t_in < pad || t_in >= length + pad {
                            continue;
                        }
                        acc += w[o][i][s] * input[i][t_in - pad];
                    }
                }
                output[o][t] = acc;
            }
        }
        output
    }

    /// Compression ratio: (TN params) / (full kernel params).
    pub fn compression_ratio(&self) -> f64 {
        let full = self.out_channels * self.in_channels * self.kernel_size;
        let compressed = self.out_rank * self.in_rank
            + self.in_rank * self.in_channels * self.kernel_size
            + self.out_rank * self.out_channels;
        compressed as f64 / full as f64
    }
}

// ============================================================================
// §6 TnQuantumInspiredLayer (Born Machine)
// ============================================================================

/// Born machine: generative model based on MPS.
///
/// The probability of a sequence x is p(x) = |<x|ψ>|² where |ψ> is encoded as MPS.
/// Trained by minimizing negative log-likelihood on discrete data.
#[derive(Debug, Clone)]
pub struct TnQuantumInspiredLayer {
    /// Internal MPS representing the quantum state |ψ>.
    pub mps: TnMatrixProductState,
    /// Number of classes for classification head.
    pub n_classes: usize,
}

impl TnQuantumInspiredLayer {
    /// Create a new Born machine.
    pub fn new(n_features: usize, d: usize, chi: usize, n_classes: usize) -> Self {
        TnQuantumInspiredLayer {
            mps: TnMatrixProductState::new(n_features, d, chi),
            n_classes,
        }
    }

    /// Compute the amplitude <x|ψ> = contract MPS with input x.
    ///
    /// Returns a scalar value.
    pub fn amplitude(&self, x: &[Vec<f64>]) -> Result<f64, TnError> {
        let out = self.mps.contract_all(x)?;
        // The last site has chi_right=1, so out is a 1-element vector
        out.first()
            .copied()
            .ok_or_else(|| TnError::NumericalError("empty MPS output".into()))
    }

    /// Compute log probability: log|<x|ψ>|² = 2 * log|amplitude|.
    pub fn log_probability(&self, x: &[Vec<f64>]) -> Result<f64, TnError> {
        let amp = self.amplitude(x)?;
        if amp == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(2.0 * amp.abs().ln())
    }

    /// Negative log-likelihood loss over a batch.
    pub fn nll_loss(&self, batch: &[Vec<Vec<f64>>]) -> Result<f64, TnError> {
        if batch.is_empty() {
            return Err(TnError::InvalidConfig("empty batch".into()));
        }
        let mut total = 0.0_f64;
        for sample in batch {
            let lp = self.log_probability(sample)?;
            total -= lp;
        }
        Ok(total / batch.len() as f64)
    }
}

// ============================================================================
// §7 TnLowRankRNN
// ============================================================================

/// RNN with tensor-train parameterized recurrent weight matrix.
///
/// Instead of a dense W_hh [hidden × hidden], uses a low-rank TT decomposition
/// with a set of TT cores, reducing parameters from O(hidden²) to O(hidden * rank).
#[derive(Debug, Clone)]
pub struct TnLowRankRnn {
    /// Standard input-to-hidden weight [hidden × input].
    pub w_ih: Vec<Vec<f64>>,
    /// TT cores for hidden-to-hidden: each core [hidden/segments × rank × hidden/segments]
    /// (simplified TT where we split hidden into segments).
    pub tt_cores: Vec<Vec<Vec<f64>>>,
    /// Bias \[hidden\].
    pub bias: Vec<f64>,
    /// Hidden state size.
    pub hidden_size: usize,
    /// Input size.
    pub input_size: usize,
    /// TT bond dimension (rank).
    pub rank: usize,
}

impl TnLowRankRnn {
    /// Create a new low-rank RNN.
    ///
    /// # Errors
    /// Returns `TnError::InvalidConfig` if hidden_size is not divisible by 2.
    pub fn new(input_size: usize, hidden_size: usize, rank: usize) -> Result<Self, TnError> {
        if hidden_size == 0 {
            return Err(TnError::InvalidConfig("hidden_size must be > 0".into()));
        }
        // We split the hidden state into 2 segments for simplicity
        // If hidden_size is odd, pad to even
        let seg = (hidden_size + 1) / 2; // ceiling half
        let mut rng = StdRng::seed_from_u64(0x4c6f77526e6e0000);

        let ih_scale = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let w_ih: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rand_normal(&mut rng) * ih_scale)
                    .collect()
            })
            .collect();

        // Two TT cores: core1 [seg × rank], core2 [rank × seg]
        // These parameterize a [hidden × hidden] matrix in two stages
        let tt_scale = (2.0 / (seg + rank) as f64).sqrt();
        let core1: Vec<Vec<f64>> = (0..seg)
            .map(|_| {
                (0..rank)
                    .map(|_| rand_normal(&mut rng) * tt_scale)
                    .collect()
            })
            .collect();
        let core2: Vec<Vec<f64>> = (0..rank)
            .map(|_| (0..seg).map(|_| rand_normal(&mut rng) * tt_scale).collect())
            .collect();

        // Store as 3D: [2 cores][rows][cols]
        let tt_cores = vec![
            core1.to_vec(),
            core2.to_vec(),
        ];

        let bias = vec![0.0_f64; hidden_size];
        Ok(TnLowRankRnn {
            w_ih,
            tt_cores,
            bias,
            hidden_size,
            input_size,
            rank,
        })
    }

    /// Forward step: h_new = tanh(W_ih @ x + TT_matvec(h) + bias).
    pub fn forward(&self, x: &[f64], h: &[f64]) -> Result<Vec<f64>, TnError> {
        if x.len() != self.input_size {
            return Err(TnError::DimensionMismatch(format!(
                "x.len()={} != input_size={}",
                x.len(),
                self.input_size
            )));
        }
        if h.len() != self.hidden_size {
            return Err(TnError::DimensionMismatch(format!(
                "h.len()={} != hidden_size={}",
                h.len(),
                self.hidden_size
            )));
        }
        let ih = mat_vec(&self.w_ih, x);
        let hh = self.tt_matvec(h)?;
        let h_new: Vec<f64> = ih
            .iter()
            .zip(hh.iter())
            .zip(self.bias.iter())
            .map(|((a, b), c)| (a + b + c).tanh())
            .collect();
        Ok(h_new)
    }

    /// Multiply hidden vector by the TT-parameterized [hidden × hidden] matrix.
    ///
    /// The matrix is approximated as: W_hh ≈ block structure using TT cores.
    pub fn tt_matvec(&self, v: &[f64]) -> Result<Vec<f64>, TnError> {
        let hidden = self.hidden_size;
        let seg = (hidden + 1) / 2;

        if self.tt_cores.len() < 2 {
            return Err(TnError::InvalidConfig("need at least 2 TT cores".into()));
        }

        // core1: seg × rank
        // core2: rank × seg
        let core1 = &self.tt_cores[0]; // [seg × rank]
        let core2 = &self.tt_cores[1]; // [rank × seg]

        // Split v into two halves: v1 [seg], v2 [rest]
        let v1: Vec<f64> = v[..seg.min(hidden)].to_vec();
        let v2: Vec<f64> = v[seg.min(hidden)..].to_vec();

        // Phase 1: r = core1^T @ v1: [rank]
        let core1_t = mat_t(core1);
        let r = mat_vec(&core1_t, &v1);

        // Phase 2: out_seg = core2 @ r: [seg]
        let out_seg2 = mat_vec(core2, &r);

        // Also apply the same to v2 using core2^T and core1
        let core2_t = mat_t(core2);
        let seg2 = v2.len();
        let r2 = if seg2 > 0 {
            let v2_padded: Vec<f64> = {
                let mut p = v2.clone();
                while p.len() < core2_t[0].len().max(1) {
                    p.push(0.0);
                }
                p.truncate(core2_t[0].len().max(1));
                p
            };
            mat_vec(&core2_t, &v2_padded)
        } else {
            vec![0.0_f64; self.rank]
        };
        let out_seg1 = mat_vec(core1, &r2);

        // Assemble output of size hidden
        let mut out = vec![0.0_f64; hidden];
        let copy_len = seg.min(hidden);
        out[..copy_len].copy_from_slice(&out_seg1[..copy_len]);
        for i in 0..out_seg2.len() {
            let idx = seg + i;
            if idx < hidden {
                out[idx] = out_seg2[i];
            }
        }
        Ok(out)
    }

    /// Compression ratio of the TT-parameterized W_hh vs. full dense.
    pub fn compression_ratio(&self) -> f64 {
        let full = self.hidden_size * self.hidden_size;
        let seg = (self.hidden_size + 1) / 2;
        let compressed = seg * self.rank + self.rank * seg;
        compressed as f64 / full as f64
    }
}

// ============================================================================
// §8 TnEntanglementMeasures
// ============================================================================

/// Entanglement measures for tensor networks.
///
/// These quantities characterize the entanglement structure of MPS states
/// via the singular value spectrum at each bond.
pub struct TnEntanglementMeasures;

impl TnEntanglementMeasures {
    /// Von Neumann entanglement entropy.
    ///
    /// S = -∑ᵢ λᵢ² log(λᵢ²) where λᵢ are normalized singular values (Schmidt coefficients).
    /// Uses normalization: λ̃ᵢ = λᵢ / sqrt(∑ⱼ λⱼ²) to ensure ∑λ̃ᵢ² = 1.
    pub fn von_neumann_entropy(singular_values: &[f64]) -> f64 {
        let norm_sq: f64 = singular_values.iter().map(|x| x * x).sum();
        if norm_sq < 1e-15 {
            return 0.0;
        }
        let mut entropy = 0.0_f64;
        for &sv in singular_values {
            let p = (sv * sv) / norm_sq;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Rényi entropy of order `alpha`.
    ///
    /// S_α = 1/(1-α) * log(∑ᵢ pᵢ^α) where pᵢ = λᵢ² / Z (normalized probabilities).
    /// For α→1, this reduces to the von Neumann entropy.
    pub fn renyi_entropy(singular_values: &[f64], alpha: f64) -> f64 {
        let norm_sq: f64 = singular_values.iter().map(|x| x * x).sum();
        if norm_sq < 1e-15 {
            return 0.0;
        }
        if (alpha - 1.0).abs() < 1e-8 {
            // limit as alpha → 1 = von Neumann entropy
            return Self::von_neumann_entropy(singular_values);
        }
        let sum_p_alpha: f64 = singular_values
            .iter()
            .map(|&sv| {
                let p = (sv * sv) / norm_sq;
                p.powf(alpha)
            })
            .sum();
        if sum_p_alpha <= 0.0 {
            return 0.0;
        }
        sum_p_alpha.ln() / (1.0 - alpha)
    }

    /// Schmidt rank: count of singular values above `tol`.
    pub fn schmidt_rank(singular_values: &[f64], tol: f64) -> usize {
        singular_values.iter().filter(|&&sv| sv.abs() > tol).count()
    }

    /// Entanglement spectrum at bond `bond` in the MPS.
    ///
    /// Performs SVD of the reshaped tensor at the specified bond and returns
    /// the singular values.
    pub fn entanglement_spectrum(mps: &TnMatrixProductState, bond: usize) -> Vec<f64> {
        if bond >= mps.n_sites.saturating_sub(1) {
            return Vec::new();
        }
        let site = &mps.sites[bond];
        // Reshape site tensor to (chi_left * d) × chi_right
        let rows = site.chi_left * site.d;
        let cols = site.chi_right;
        let mut mat = vec![vec![0.0_f64; cols]; rows];
        for l in 0..site.chi_left {
            for s in 0..site.d {
                for r in 0..site.chi_right {
                    mat[l * site.d + s][r] = site.tensor[l][s][r];
                }
            }
        }
        let k = rows.min(cols);
        let (_u, sigma, _vt) = svd_truncated(&mat, k);
        sigma
    }
}

// ============================================================================
// §9 TnNeuralNetworkTN
// ============================================================================

/// Feed-forward neural network using Tucker decomposition layers.
///
/// Each hidden layer is a `TnTuckerLayer` instead of a dense layer.
/// Activation: ReLU between layers, linear at output.
#[derive(Debug, Clone)]
pub struct TnNeuralNetworkTN {
    /// Tucker layers.
    pub layers: Vec<TnTuckerLayer>,
    /// Layer dimensions [in_0, in_1, ..., out_last].
    pub layer_dims: Vec<usize>,
}

impl TnNeuralNetworkTN {
    /// Create a new TN neural network.
    ///
    /// # Arguments
    /// - `dims`: slice of dimensions, e.g. [784, 256, 128, 10]
    ///   meaning 3 layers: 784→256, 256→128, 128→10
    /// - `rank`: Tucker decomposition rank for each layer
    ///
    /// # Errors
    /// Returns `TnError::InvalidConfig` if `dims.len() < 2`.
    pub fn new(dims: &[usize], rank: usize) -> Result<Self, TnError> {
        if dims.len() < 2 {
            return Err(TnError::InvalidConfig(
                "dims must have at least 2 elements (in_dim, out_dim)".into(),
            ));
        }
        let layers: Vec<TnTuckerLayer> = dims
            .windows(2)
            .map(|w| TnTuckerLayer::new(w[0], w[1], rank))
            .collect();
        Ok(TnNeuralNetworkTN {
            layers,
            layer_dims: dims.to_vec(),
        })
    }

    /// Forward pass through all layers with ReLU activations (except last layer).
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, TnError> {
        if self.layers.is_empty() {
            return Err(TnError::InvalidConfig("no layers".into()));
        }
        let mut h: Vec<f64> = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            // Apply ReLU for all layers except the last
            if i < self.layers.len() - 1 {
                h = h.into_iter().map(|x| x.max(0.0)).collect();
            }
        }
        Ok(h)
    }

    /// Total number of parameters across all layers.
    pub fn total_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.r1 * l.r2 + l.in_dim * l.r1 + l.out_dim * l.r2 + l.out_dim)
            .sum()
    }

    /// Compression ratio vs. equivalent dense network.
    pub fn compression_ratio(&self, dense_params: usize) -> f64 {
        if dense_params == 0 {
            return 1.0;
        }
        self.total_params() as f64 / dense_params as f64
    }
}

// ============================================================================
// §10 TnMetrics
// ============================================================================

/// Utility metrics for tensor network models.
pub struct TnMetrics;

impl TnMetrics {
    /// MPS fidelity: |<ψ1|ψ2>|² normalized by individual norms.
    ///
    /// Approximated by evaluating both MPS on the same input and computing
    /// the normalized inner product squared.
    pub fn mps_fidelity(
        mps1: &TnMatrixProductState,
        mps2: &TnMatrixProductState,
        input: &[Vec<f64>],
    ) -> Result<f64, TnError> {
        let out1 = mps1.contract_all(input)?;
        let out2 = mps2.contract_all(input)?;
        if out1.len() != out2.len() {
            return Err(TnError::DimensionMismatch(format!(
                "MPS outputs have different lengths: {} vs {}",
                out1.len(),
                out2.len()
            )));
        }
        let inner: f64 = out1.iter().zip(out2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = out1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = out2.iter().map(|x| x * x).sum::<f64>().sqrt();
        let denom = norm1 * norm2;
        if denom < 1e-15 {
            return Ok(0.0);
        }
        Ok((inner / denom).powi(2).clamp(0.0, 1.0))
    }

    /// Tucker layer reconstruction error on a dense weight matrix.
    ///
    /// Materializes W ≈ U2 @ G @ U1^T and computes Frobenius norm of (W_true - W_approx).
    pub fn tucker_reconstruction_error(original: &[Vec<f64>], layer: &TnTuckerLayer) -> f64 {
        let out_dim = original.len();
        let in_dim = if original.is_empty() {
            0
        } else {
            original[0].len()
        };

        // Materialize Tucker approximation: W_approx = U2 @ G @ U1^T
        let u1t = mat_t(&layer.u1);
        let gu1t = mat_mul(&layer.g, &u1t); // [r2 × in_dim]
        let w_approx = mat_mul(&layer.u2, &gu1t); // [out_dim × in_dim]

        let mut sq_err = 0.0_f64;
        for i in 0..out_dim.min(w_approx.len()) {
            for j in 0..in_dim.min(original[i].len()).min(w_approx[i].len()) {
                let diff = original[i][j] - w_approx[i][j];
                sq_err += diff * diff;
            }
        }
        sq_err.sqrt()
    }

    /// Return the bond dimension profile of an MPS.
    pub fn bond_dimension_profile(mps: &TnMatrixProductState) -> Vec<usize> {
        mps.bond_dimensions()
    }

    /// Effective rank: number of singular values above `threshold * max_sv`.
    pub fn effective_rank(singular_values: &[f64], threshold: f64) -> usize {
        if singular_values.is_empty() {
            return 0;
        }
        let max_sv = singular_values.iter().cloned().fold(0.0_f64, f64::max);
        if max_sv < 1e-15 {
            return 0;
        }
        singular_values
            .iter()
            .filter(|&&sv| sv / max_sv > threshold)
            .count()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    include!("tests.rs");
}
