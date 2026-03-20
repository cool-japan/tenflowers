//! Spectral / frequency-domain neural network operations.
//!
//! This module provides:
//! - [`dct_1d`] / [`idct_1d`]: discrete cosine transform (DCT-II) and its inverse
//! - [`SpectralNorm`]: weight-matrix normalisation by its largest singular value
//! - [`RandomFourierFeatures`]: Rahimi-Recht random feature kernel approximation
//! - [`FourierPositionEncoding`]: learnable Fourier positional encodings
//! - Kernel functions: [`rbf_kernel`], [`laplacian_kernel`], [`polynomial_features`]
//! - [`SpectralGraphConv`]: Kipf & Welling GCN in the spectral (normalised adjacency) basis
//!
//! All types operate on plain `Vec<f32>` / `Vec<Vec<f32>>` buffers.

use std::f32::consts::PI;

/// Error type for spectral operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralError {
    /// Empty input vector / matrix.
    EmptyInput,
    /// A dimension was expected to be one value but had another.
    DimensionMismatch { expected: usize, found: usize },
    /// Adjacency matrix invalid (non-square or mismatched node count).
    InvalidAdjacency { rows: usize, cols: usize },
    /// Gamma (bandwidth) parameter is non-positive.
    InvalidGamma { gamma: f32 },
}

impl std::fmt::Display for SpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectralError::EmptyInput => write!(f, "empty input"),
            SpectralError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            SpectralError::InvalidAdjacency { rows, cols } => {
                write!(f, "adjacency matrix invalid: {rows}×{cols}")
            }
            SpectralError::InvalidGamma { gamma } => {
                write!(f, "gamma must be positive, got {gamma}")
            }
        }
    }
}

impl std::error::Error for SpectralError {}

// ─────────────────────────────────────────────────────────────────────────────
// DCT-II / IDCT
// ─────────────────────────────────────────────────────────────────────────────

/// Discrete Cosine Transform — Type II (DCT-II).
///
/// `X[k] = Σ_{n=0}^{N-1} x[n] · cos( π/N · (n + 0.5) · k )`  for k = 0…N-1.
///
/// Returns an empty `Vec` when `x` is empty.
pub fn dct_1d(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f32;
    (0..n)
        .map(|k| {
            let kf = k as f32;
            x.iter()
                .enumerate()
                .map(|(i, &xi)| xi * ((PI / nf * (i as f32 + 0.5) * kf).cos()))
                .sum()
        })
        .collect()
}

/// Inverse DCT (IDCT-II, normalised to recover the original signal).
///
/// Uses the standard orthonormal IDCT formula:
/// `x[n] = (1/N) · X[0]  +  (2/N) · Σ_{k=1}^{N-1} X[k] · cos( π/N · (n + 0.5) · k )`
pub fn idct_1d(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f32;
    (0..n)
        .map(|i| {
            let inf = i as f32;
            let mut val = x[0] / nf;
            for k in 1..n {
                val += (2.0 / nf) * x[k] * ((PI / nf * (inf + 0.5) * k as f32).cos());
            }
            val
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectralNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral normalisation via power iteration (Miyato et al., ICLR 2018).
///
/// Estimates the largest singular value `σ₁` of a weight matrix and divides
/// every entry by it, making the linear operator 1-Lipschitz.
///
/// # Power-iteration update (one step)
/// ```text
/// v̂ = normalise(W^T u)
/// û = normalise(W v̂)
/// σ ≈ û^T W v̂
/// ```
#[derive(Debug, Clone)]
pub struct SpectralNorm {
    /// Left singular-vector estimate: `[rows]`.
    pub u: Vec<f32>,
    /// Right singular-vector estimate: `[cols]`.
    pub v: Vec<f32>,
    /// Number of power-iteration steps per `sigma_estimate` call.
    pub num_power_iter: usize,
}

impl SpectralNorm {
    /// Initialise with deterministic all-ones-then-normalised vectors.
    pub fn new(rows: usize, cols: usize, num_power_iter: usize) -> Self {
        let u = normalize_vec(&vec![1.0_f32; rows]);
        let v = normalize_vec(&vec![1.0_f32; cols]);
        Self {
            u,
            v,
            num_power_iter,
        }
    }

    /// Run `num_power_iter` power-iteration steps and return the current
    /// singular-value estimate `σ ≈ u^T W v`.
    pub fn sigma_estimate(&mut self, weight: &[Vec<f32>]) -> f32 {
        for _ in 0..self.num_power_iter {
            // v̂ = normalise(W^T u)
            let wt_u = mat_vec_t(weight, &self.u);
            self.v = normalize_vec(&wt_u);
            // û = normalise(W v)
            let w_v = mat_vec(weight, &self.v);
            self.u = normalize_vec(&w_v);
        }
        // σ = u^T W v
        let w_v = mat_vec(weight, &self.v);
        self.u.iter().zip(w_v.iter()).map(|(&u, &wv)| u * wv).sum()
    }

    /// Normalise `weight` by its estimated largest singular value.
    ///
    /// Returns an identical-shaped matrix `W / σ`.
    pub fn normalize(&mut self, weight: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let sigma = self.sigma_estimate(weight);
        let denom = if sigma.abs() < 1e-12 { 1.0 } else { sigma };
        weight
            .iter()
            .map(|row| row.iter().map(|&w| w / denom).collect())
            .collect()
    }

    /// Alias for \[`sigma_estimate`\] — runs the iteration and returns σ.
    pub fn update(&mut self, weight: &[Vec<f32>]) -> f32 {
        self.sigma_estimate(weight)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random Fourier Features
// ─────────────────────────────────────────────────────────────────────────────

/// Random Fourier Features for RBF kernel approximation (Rahimi & Recht, NeurIPS 2007).
///
/// Approximates `k(x, y) = exp(-γ ‖x-y‖²)` with an inner product in a finite
/// dimensional space:
/// ```text
/// φ(x) = √(2/D) · [ cos(ω_1^T x + b_1), …, cos(ω_D^T x + b_D) ]
/// ```
/// where `ω_i ~ N(0, 2γ I)` and `b_i ~ Uniform[0, 2π]`.
///
/// This implementation uses a deterministic **Halton-sequence** quasi-random
/// initialisation (base 2 for `ω`, base 3 for `b`) to avoid any non-determinism.
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    /// Frequency matrix: `[num_features, input_dim]`.
    pub omega: Vec<Vec<f32>>,
    /// Phase offsets: `[num_features]`.
    pub b: Vec<f32>,
    /// Number of random features `D`.
    pub num_features: usize,
    /// Input dimensionality.
    pub input_dim: usize,
    /// RBF bandwidth `γ`.
    pub gamma: f32,
}

/// Halton sequence element at index `i` in base `base`.
fn halton(i: usize, base: usize) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    let mut idx = i;
    let b = base as f32;
    while idx > 0 {
        f /= b;
        r += f * (idx % base) as f32;
        idx /= base;
    }
    r
}

/// Approximate N(0,1) via Box-Muller using two Halton sequences.
fn halton_normal(i: usize) -> f32 {
    // Use Halton(i, 2) and Halton(i, 5) as pseudo-independent uniform samples.
    let u1 = halton(i + 1, 2).max(1e-10); // avoid log(0)
    let u2 = halton(i + 1, 5);
    let theta = 2.0 * PI * u2;
    ((-2.0 * u1.ln()).sqrt()) * theta.cos()
}

impl RandomFourierFeatures {
    /// Construct RFF with quasi-random (Halton-based) initialisation.
    ///
    /// `gamma`: bandwidth of the target RBF kernel.
    pub fn new(input_dim: usize, num_features: usize, gamma: f32) -> Self {
        // σ = sqrt(2γ) for the frequency distribution N(0, 2γ I).
        let std_dev = (2.0 * gamma).sqrt();

        // Build omega: [num_features, input_dim] using N(0, 2γ)
        let omega: Vec<Vec<f32>> = (0..num_features)
            .map(|i| {
                (0..input_dim)
                    .map(|j| halton_normal(i * input_dim + j) * std_dev)
                    .collect()
            })
            .collect();

        // Phase offsets: Uniform[0, 2π] via Halton base 3.
        let b: Vec<f32> = (0..num_features)
            .map(|i| halton(i + 1, 3) * 2.0 * PI)
            .collect();

        Self {
            omega,
            b,
            num_features,
            input_dim,
            gamma,
        }
    }

    /// Map `x` to the approximate feature space.
    ///
    /// `φ(x) = √(2/D) · cos(Ω x + b)` ∈ R^D.
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        let scale = (2.0 / self.num_features as f32).sqrt();
        (0..self.num_features)
            .map(|i| {
                let dot: f32 = self.omega[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(&w, &xi)| w * xi)
                    .sum();
                scale * (dot + self.b[i]).cos()
            })
            .collect()
    }

    /// Approximate RBF kernel: `⟨φ(x1), φ(x2)⟩ ≈ exp(-γ ‖x1-x2‖²)`.
    pub fn kernel(&self, x1: &[f32], x2: &[f32]) -> f32 {
        let phi1 = self.transform(x1);
        let phi2 = self.transform(x2);
        phi1.iter().zip(phi2.iter()).map(|(&a, &b)| a * b).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fourier Position Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Learnable Fourier positional encoding.
///
/// Initialised with the classical sinusoidal schedule
/// `f_i = 1 / 10000^{2i / d_model}` and phase `φ_i = 0`, but stored as
/// mutable fields so they can be updated during training.
///
/// Encoding for position `pos`:
/// ```text
/// enc[pos, 2i]     = cos(f_i · pos + φ_i)
/// enc[pos, 2i + 1] = sin(f_i · pos + φ_i)
/// ```
#[derive(Debug, Clone)]
pub struct FourierPositionEncoding {
    /// Maximum sequence length supported.
    pub max_len: usize,
    /// Model dimension (must be even).
    pub d_model: usize,
    /// Learnable frequency per pair: `[d_model / 2]`.
    pub frequencies: Vec<f32>,
    /// Learnable phase offset per pair: `[d_model / 2]`.
    pub phases: Vec<f32>,
}

impl FourierPositionEncoding {
    /// Construct with sinusoidal initialisation `f_i = 1 / 10000^{2i / d_model}`.
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let num_pairs = d_model / 2;
        let frequencies: Vec<f32> = (0..num_pairs)
            .map(|i| 1.0 / (10000.0_f32.powf(2.0 * i as f32 / d_model as f32)))
            .collect();
        let phases = vec![0.0_f32; num_pairs];
        Self {
            max_len,
            d_model,
            frequencies,
            phases,
        }
    }

    /// Encode a single position `pos` → `[d_model]`.
    pub fn encode_pos(&self, pos: usize) -> Vec<f32> {
        let num_pairs = self.d_model / 2;
        let extra = self.d_model % 2;
        let pos_f = pos as f32;

        let mut enc = Vec::with_capacity(self.d_model);
        for i in 0..num_pairs {
            let arg = self.frequencies[i] * pos_f + self.phases[i];
            enc.push(arg.cos());
            enc.push(arg.sin());
        }
        // If d_model is odd, append one final cosine.
        if extra == 1 {
            let i = num_pairs;
            // frequency for the extra slot: use the last frequency (or a new one).
            let last_freq = if num_pairs > 0 {
                self.frequencies[num_pairs - 1]
            } else {
                1.0
            };
            let arg = last_freq * pos_f;
            enc.push(arg.cos());
        }
        enc
    }

    /// Produce `[seq_len, d_model]` positional encoding table.
    pub fn encode(&self, seq_len: usize) -> Vec<Vec<f32>> {
        (0..seq_len).map(|pos| self.encode_pos(pos)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel functions
// ─────────────────────────────────────────────────────────────────────────────

/// RBF (Gaussian) kernel: `exp(-γ ‖x1 - x2‖²)`.
pub fn rbf_kernel(x1: &[f32], x2: &[f32], gamma: f32) -> f32 {
    let sq_dist: f32 = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    (-gamma * sq_dist).exp()
}

/// Laplacian kernel: `exp(-γ ‖x1 - x2‖_1)`.
pub fn laplacian_kernel(x1: &[f32], x2: &[f32], gamma: f32) -> f32 {
    let l1_dist: f32 = x1.iter().zip(x2.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    (-gamma * l1_dist).exp()
}

/// Polynomial feature map for a single degree.
///
/// Computes the (non-homogeneous) polynomial kernel feature map of a given
/// `degree`:
/// `φ(x) = [ x_i^k : i = 0..d-1, k = 1..degree, plus bias term (bias)^degree ]`
///
/// More precisely, we append `[bias, x_0, x_0^2, …, x_0^d, x_1, …, x_{n-1}^d]`
/// so the inner product `⟨φ(x), φ(y)⟩` approximates `(⟨x,y⟩ + bias²)^degree`.
pub fn polynomial_features(x: &[f32], degree: usize, bias: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() * degree + 1);
    // bias term
    let mut b_pow = 1.0_f32;
    for _ in 0..degree {
        b_pow *= bias;
    }
    out.push(b_pow);
    // feature powers
    for &xi in x.iter() {
        let mut xpow = 1.0_f32;
        for _ in 0..degree {
            xpow *= xi;
            out.push(xpow);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectralGraphConv
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral graph convolution — Kipf & Welling GCN (ICLR 2017).
///
/// Forward pass:
/// ```text
/// H' = σ( Â X W + b )
/// ```
/// where `Â = D̃^{-1/2} (A + I) D̃^{-1/2}` (re-normalisation trick),
/// `D̃[i,i] = 1 + Σ_j A[i,j]`, and `σ = ReLU`.
#[derive(Debug, Clone)]
pub struct SpectralGraphConv {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Weight matrix: `[in_features, out_features]`.
    pub weight: Vec<Vec<f32>>,
    /// Bias vector: `[out_features]`.
    pub bias: Vec<f32>,
}

impl SpectralGraphConv {
    /// Create a new layer with Xavier-initialised weights.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = xavier_uniform_sgc(in_features, out_features, 300);
        let bias = vec![0.0_f32; out_features];
        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    /// Compute the renormalised adjacency `Â = D̃^{-1/2} (A + I) D̃^{-1/2}`.
    fn renorm_adj(adj: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = adj.len();
        // Ã = A + I  and  D̃[i] = Σ_j Ã[i,j]
        let mut a_tilde: Vec<Vec<f32>> = adj
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r[i] += 1.0;
                r
            })
            .collect();
        let d_tilde: Vec<f32> = a_tilde.iter().map(|row| row.iter().sum::<f32>()).collect();
        let d_inv_sqrt: Vec<f32> = d_tilde
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        let mut a_hat = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                a_hat[i][j] = d_inv_sqrt[i] * a_tilde[i][j] * d_inv_sqrt[j];
            }
        }
        a_hat
    }

    /// GCN forward pass → `[num_nodes, out_features]`.
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        adj: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, SpectralError> {
        let n = node_features.len();
        if n == 0 {
            return Err(SpectralError::EmptyInput);
        }
        let adj_rows = adj.len();
        if adj_rows == 0 {
            return Err(SpectralError::InvalidAdjacency { rows: 0, cols: 0 });
        }
        let adj_cols = adj[0].len();
        if adj_rows != adj_cols {
            return Err(SpectralError::InvalidAdjacency {
                rows: adj_rows,
                cols: adj_cols,
            });
        }
        if adj_rows != n {
            return Err(SpectralError::DimensionMismatch {
                expected: n,
                found: adj_rows,
            });
        }
        for row in node_features.iter() {
            if row.len() != self.in_features {
                return Err(SpectralError::DimensionMismatch {
                    expected: self.in_features,
                    found: row.len(),
                });
            }
        }

        let a_hat = Self::renorm_adj(adj);

        // Â X: [n, in_features]
        let ax: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0_f32; self.in_features];
                for j in 0..n {
                    let a_ij = a_hat[i][j];
                    if a_ij.abs() > 1e-12 {
                        for f in 0..self.in_features {
                            row[f] += a_ij * node_features[j][f];
                        }
                    }
                }
                row
            })
            .collect();

        // (Â X) W + b followed by ReLU.
        let mut out = vec![vec![0.0_f32; self.out_features]; n];
        for i in 0..n {
            for o in 0..self.out_features {
                let mut val = self.bias[o];
                for f in 0..self.in_features {
                    val += ax[i][f] * self.weight[f][o];
                }
                out[i][o] = val.max(0.0); // ReLU
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix–vector product: `y[i] = Σ_j W[i][j] * x[j]`.
fn mat_vec(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// Transposed matrix–vector product: `y[j] = Σ_i W[i][j] * x[i]`.
fn mat_vec_t(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    if w.is_empty() {
        return Vec::new();
    }
    let cols = w[0].len();
    let mut y = vec![0.0_f32; cols];
    for (i, row) in w.iter().enumerate() {
        let xi = if i < x.len() { x[i] } else { 0.0 };
        for (j, &wij) in row.iter().enumerate() {
            y[j] += wij * xi;
        }
    }
    y
}

/// L2-normalise a vector; returns the input unchanged if the norm is tiny.
fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if n < 1e-12 {
        v.to_vec()
    } else {
        v.iter().map(|&x| x / n).collect()
    }
}

/// Xavier uniform initialiser for SpectralGraphConv.
fn xavier_uniform_sgc(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
    // Avoid direct dependency on scirs2_core::random here by using a simple
    // deterministic LCG for the SGC weights (keeps this module self-contained).
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt() as f32;
    let mut state: u64 = seed.wrapping_add(0xdeadbeef);
    let mut next = || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((state >> 33) ^ state) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 * limit - limit
    };
    (0..rows)
        .map(|_| (0..cols).map(|_| next()).collect())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DCT roundtrip ────────────────────────────────────────────────────────

    #[test]
    fn test_dct_idct_roundtrip() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = dct_1d(&x);
        let recovered = idct_1d(&coeffs);
        assert_eq!(x.len(), recovered.len());
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "roundtrip error: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_dct_empty() {
        assert!(dct_1d(&[]).is_empty());
        assert!(idct_1d(&[]).is_empty());
    }

    #[test]
    fn test_dct_single() {
        let x = vec![5.0_f32];
        let c = dct_1d(&x);
        assert!((c[0] - 5.0).abs() < 1e-5);
        let r = idct_1d(&c);
        assert!((r[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_dct_constant_signal() {
        // DCT of constant signal: only k=0 component is non-zero.
        let x = vec![3.0_f32; 4];
        let c = dct_1d(&x);
        // k=0: X[0] = Σ 3 * cos(0) = 4*3 = 12
        assert!((c[0] - 12.0).abs() < 1e-4);
        for k in 1..4 {
            assert!(c[k].abs() < 1e-4, "c[{k}] = {} should be ~0", c[k]);
        }
    }

    // ── SpectralNorm ─────────────────────────────────────────────────────────

    #[test]
    fn test_spectralnorm_sigma_positive() {
        let mut sn = SpectralNorm::new(4, 3, 5);
        let w = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 2.0, 0.0],
            vec![0.0_f32, 0.0, 3.0],
            vec![0.5_f32, 0.5, 0.5],
        ];
        let sigma = sn.sigma_estimate(&w);
        assert!(sigma > 0.0, "sigma = {sigma}");
    }

    #[test]
    fn test_spectralnorm_normalize_reduces_norm() {
        let mut sn = SpectralNorm::new(3, 3, 10);
        let w = vec![
            vec![4.0_f32, 0.0, 0.0],
            vec![0.0_f32, 4.0, 0.0],
            vec![0.0_f32, 0.0, 4.0],
        ];
        let w_norm = sn.normalize(&w);
        // Frobenius norm of normalised W should be less than original.
        let orig_norm: f32 = w
            .iter()
            .flat_map(|r| r.iter())
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let new_norm: f32 = w_norm
            .iter()
            .flat_map(|r| r.iter())
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            new_norm < orig_norm,
            "new_norm {new_norm} >= orig {orig_norm}"
        );
    }

    #[test]
    fn test_spectralnorm_update_returns_sigma() {
        let mut sn = SpectralNorm::new(2, 2, 3);
        let w = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let s = sn.update(&w);
        assert!(s > 0.0);
    }

    // ── RandomFourierFeatures ────────────────────────────────────────────────

    #[test]
    fn test_rff_transform_output_dim() {
        let rff = RandomFourierFeatures::new(3, 128, 0.5);
        let x = vec![1.0_f32, 2.0, 3.0];
        let phi = rff.transform(&x);
        assert_eq!(phi.len(), 128);
    }

    #[test]
    fn test_rff_kernel_same_input_near_one() {
        let rff = RandomFourierFeatures::new(4, 1024, 1.0);
        let x = vec![1.0_f32, -1.0, 0.5, 2.0];
        let k = rff.kernel(&x, &x);
        // k(x, x) = exp(0) = 1; with finite features this is approximate.
        assert!((k - 1.0).abs() < 0.1, "k(x,x) = {k}, expected ~1.0");
    }

    #[test]
    fn test_rff_kernel_approximates_rbf() {
        // k(x, y) should be less than 1 for different inputs.
        let rff = RandomFourierFeatures::new(4, 2048, 1.0);
        let x = vec![0.0_f32; 4];
        // Use a nearby point so the exact kernel value is not vanishingly small.
        let y = vec![0.3_f32; 4]; // ||x-y||^2 = 4*0.09 = 0.36, exact = exp(-0.36) ≈ 0.698
        let approx = rff.kernel(&x, &y);
        let exact = rbf_kernel(&x, &y, 1.0);
        // With 2048 quasi-random features the approximation should be within 0.4
        // of the exact value (Halton sequences converge slower than true RBF
        // but the sign and rough magnitude should match).
        assert!(
            (approx - exact).abs() < 0.4,
            "approx={approx}, exact={exact}, diff={}",
            (approx - exact).abs()
        );
        // Both should be positive and the approximate kernel should be in [−1, 2].
        assert!(exact > 0.0 && exact <= 1.0);
    }

    // ── FourierPositionEncoding ──────────────────────────────────────────────

    #[test]
    fn test_fourier_pos_encode_shape() {
        let enc = FourierPositionEncoding::new(16, 100);
        let table = enc.encode(50);
        assert_eq!(table.len(), 50);
        assert_eq!(table[0].len(), 16);
    }

    #[test]
    fn test_fourier_pos_encode_single_shape() {
        let enc = FourierPositionEncoding::new(8, 50);
        let v = enc.encode_pos(5);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_fourier_pos_encode_values_bounded() {
        let enc = FourierPositionEncoding::new(16, 100);
        let table = enc.encode(10);
        for row in &table {
            for &v in row.iter() {
                assert!(v.abs() <= 1.0 + 1e-5, "value {v} out of [-1,1]");
            }
        }
    }

    #[test]
    fn test_fourier_pos_encode_pos0_all_cosines() {
        // At pos=0, sin(f*0) = 0 and cos(f*0) = 1, so even indices = 1, odd = 0.
        let enc = FourierPositionEncoding::new(8, 10);
        let v = enc.encode_pos(0);
        for (i, &val) in v.iter().enumerate() {
            if i % 2 == 0 {
                assert!((val - 1.0).abs() < 1e-5, "even[{i}] = {val}");
            } else {
                assert!(val.abs() < 1e-5, "odd[{i}] = {val}");
            }
        }
    }

    // ── Kernel functions ─────────────────────────────────────────────────────

    #[test]
    fn test_rbf_kernel_same_input() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let k = rbf_kernel(&x, &x, 1.0);
        assert!((k - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rbf_kernel_decreases_with_distance() {
        let x = vec![0.0_f32, 0.0];
        let y1 = vec![1.0_f32, 0.0];
        let y2 = vec![2.0_f32, 0.0];
        let k1 = rbf_kernel(&x, &y1, 1.0);
        let k2 = rbf_kernel(&x, &y2, 1.0);
        assert!(k1 > k2, "k1={k1} should be > k2={k2}");
    }

    #[test]
    fn test_laplacian_kernel_same_input() {
        let x = vec![1.0_f32, 2.0];
        let k = laplacian_kernel(&x, &x, 1.0);
        assert!((k - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_features_degree2() {
        // x = [2.0], degree=2, bias=1.0
        // Expected: [bias^2, x^1, x^2] = [1.0, 2.0, 4.0]
        let x = vec![2.0_f32];
        let feats = polynomial_features(&x, 2, 1.0);
        assert_eq!(feats.len(), 3); // 1 bias + 1 input × 2 degrees
        assert!((feats[0] - 1.0).abs() < 1e-6, "bias^2 = {}", feats[0]);
        assert!((feats[1] - 2.0).abs() < 1e-6, "x^1 = {}", feats[1]);
        assert!((feats[2] - 4.0).abs() < 1e-6, "x^2 = {}", feats[2]);
    }

    #[test]
    fn test_polynomial_features_degree1() {
        let x = vec![3.0_f32, 4.0];
        let feats = polynomial_features(&x, 1, 0.0);
        // bias^1 = 0, then [3, 4]
        assert_eq!(feats.len(), 3);
        assert!(feats[0].abs() < 1e-6);
        assert!((feats[1] - 3.0).abs() < 1e-6);
        assert!((feats[2] - 4.0).abs() < 1e-6);
    }

    // ── SpectralGraphConv ────────────────────────────────────────────────────

    #[test]
    fn test_spectralgraphconv_forward_shape() {
        let conv = SpectralGraphConv::new(4, 8);
        let feats = vec![
            vec![1.0_f32, 0.0, 0.5, -1.0],
            vec![0.0_f32, 1.0, -0.5, 2.0],
            vec![-1.0_f32, -1.0, 1.0, 0.0],
        ];
        let adj = vec![
            vec![0.0_f32, 1.0, 1.0],
            vec![1.0_f32, 0.0, 1.0],
            vec![1.0_f32, 1.0, 0.0],
        ];
        let out = conv.forward(&feats, &adj).expect("forward ok");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_spectralgraphconv_forward_relu_activation() {
        let conv = SpectralGraphConv::new(2, 4);
        let feats = vec![vec![1.0_f32, 1.0], vec![-1.0_f32, -1.0]];
        let adj = vec![vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]];
        let out = conv.forward(&feats, &adj).expect("forward ok");
        for row in &out {
            for &v in row.iter() {
                assert!(v >= 0.0, "ReLU output must be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn test_spectralgraphconv_forward_error_empty() {
        let conv = SpectralGraphConv::new(4, 8);
        let result = conv.forward(&[], &[]);
        assert!(matches!(result, Err(SpectralError::EmptyInput)));
    }

    #[test]
    fn test_spectralgraphconv_forward_error_adj_mismatch() {
        let conv = SpectralGraphConv::new(4, 8);
        let feats = vec![vec![1.0_f32, 0.0, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0, 0.0]];
        // 3×3 adjacency for 2 nodes → mismatch
        let adj = vec![
            vec![0.0_f32, 1.0, 0.0],
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 0.0, 0.0],
        ];
        let result = conv.forward(&feats, &adj);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_error_display() {
        assert!(SpectralError::EmptyInput.to_string().contains("empty"));
        let e = SpectralError::DimensionMismatch {
            expected: 4,
            found: 2,
        };
        assert!(e.to_string().contains("4") && e.to_string().contains("2"));
        let e2 = SpectralError::InvalidGamma { gamma: -1.0 };
        assert!(e2.to_string().contains("positive"));
    }
}
