//! Sparse Coding, Dictionary Learning, and Compressed Sensing.
//! OMP/ISTA encoders, K-SVD/Online dict learning, ADMM/CoSaMP recovery,
//! sparse autoencoder, matching pursuit, LASSO/ElasticNet/GroupLASSO regression.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

// ── SparsityMeasure ────────────────────────────────────────────────────────────

/// Regularizer / sparsity measure for encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum SparsityMeasure {
    /// L0 pseudo-norm: at most `k` non-zero coefficients.
    L0(usize),
    /// L1 norm (LASSO) with penalty weight.
    L1(f32),
    /// Elastic Net with L1 and L2 penalty weights.
    ElasticNet(f32, f32),
}

// ── SparseCode ────────────────────────────────────────────────────────────────

/// A sparse representation of a signal: coefficients and their support indices.
#[derive(Debug, Clone)]
pub struct SparseCode {
    /// Non-zero coefficients.
    pub coefficients: Vec<f32>,
    /// Indices of the non-zero entries (support).
    pub support: Vec<usize>,
}

impl SparseCode {
    /// Number of non-zero coefficients.
    pub fn nnz(&self) -> usize {
        self.coefficients.iter().filter(|&&c| c != 0.0).count()
    }

    /// Reconstruct the signal from the dictionary atoms.
    pub fn reconstruct(&self, dictionary: &[Vec<f32>]) -> Vec<f32> {
        if dictionary.is_empty() || self.support.is_empty() {
            return Vec::new();
        }
        let dim = dictionary[0].len();
        let mut out = vec![0.0_f32; dim];
        for (&idx, &coef) in self.support.iter().zip(self.coefficients.iter()) {
            if idx < dictionary.len() {
                for (o, &a) in out.iter_mut().zip(dictionary[idx].iter()) {
                    *o += coef * a;
                }
            }
        }
        out
    }
}

// ── OrthogonalMatchingPursuit ─────────────────────────────────────────────────

/// Greedy sparse encoder using Orthogonal Matching Pursuit.
pub struct OrthogonalMatchingPursuit {
    /// Number of non-zero coefficients (sparsity budget).
    pub n_nonzero: usize,
}

impl OrthogonalMatchingPursuit {
    /// Encode `signal` using at most `n_nonzero` dictionary atoms.
    ///
    /// Algorithm: iteratively select the most correlated atom with the residual,
    /// then project the signal onto the current support via least-squares,
    /// and update the residual.
    pub fn encode(&self, signal: &[f32], dictionary: &[Vec<f32>]) -> SparseCode {
        let n_atoms = dictionary.len();
        let k = self.n_nonzero.min(n_atoms);
        let mut residual = signal.to_vec();
        let mut support: Vec<usize> = Vec::with_capacity(k);

        for _ in 0..k {
            // Select atom with highest correlation to residual
            let mut best_idx = 0;
            let mut best_corr = f32::NEG_INFINITY;
            for (j, atom) in dictionary.iter().enumerate() {
                if support.contains(&j) {
                    continue;
                }
                let corr: f32 = dot(&residual, atom).abs();
                if corr > best_corr {
                    best_corr = corr;
                    best_idx = j;
                }
            }
            support.push(best_idx);

            // Solve least squares: min ||signal - D_S * x||^2 via Gram-Schmidt / normal eqs
            let coefficients = omp_least_squares(signal, dictionary, &support);

            // Update residual
            residual = signal.to_vec();
            for (&idx, &c) in support.iter().zip(coefficients.iter()) {
                for (r, &a) in residual.iter_mut().zip(dictionary[idx].iter()) {
                    *r -= c * a;
                }
            }

            // Early exit if residual is negligible
            if norm_sq(&residual) < 1e-10 {
                let coefs = omp_least_squares(signal, dictionary, &support);
                return SparseCode {
                    coefficients: coefs,
                    support,
                };
            }
        }

        let coefficients = omp_least_squares(signal, dictionary, &support);
        SparseCode {
            coefficients,
            support,
        }
    }
}

/// Solve the least-squares system signal ≈ D_S * x for selected support atoms.
pub(crate) fn omp_least_squares(
    signal: &[f32],
    dictionary: &[Vec<f32>],
    support: &[usize],
) -> Vec<f32> {
    let k = support.len();
    let ds: Vec<Vec<f32>> = support.iter().map(|&i| dictionary[i].clone()).collect();
    let mut gram = vec![vec![0.0_f32; k]; k];
    for i in 0..k {
        for j in 0..k {
            gram[i][j] = dot(&ds[i], &ds[j]);
        }
    }
    let rhs: Vec<f32> = ds.iter().map(|col| dot(col, signal)).collect();
    cholesky_solve(&gram, &rhs).unwrap_or_else(|_| vec![0.0_f32; k])
}

/// Solve A x = b via Gaussian elimination (small systems).
pub(crate) fn cholesky_solve(a: &[Vec<f32>], b: &[f32]) -> Result<Vec<f32>, ()> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut m: Vec<Vec<f32>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();
    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..n {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(());
        }
        m.swap(col, pivot_row);
        let diag = m[col][col];
        for v in m[col].iter_mut() {
            *v /= diag;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            let pivot_row_copy = m[col].clone();
            for (v, &p) in m[row].iter_mut().zip(pivot_row_copy.iter()) {
                *v -= factor * p;
            }
        }
    }
    Ok(m.iter()
        .map(|row| *row.last().unwrap_or(&0.0))
        .collect())
}

// ── LassoEncoder ─────────────────────────────────────────────────────────────

/// ISTA-based LASSO encoder.
pub struct LassoEncoder {
    /// L1 regularization weight.
    pub lambda: f32,
    /// Maximum number of ISTA iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f32,
}

impl LassoEncoder {
    /// ISTA: iterative soft-thresholding.
    ///
    /// Update: x ← soft_threshold(x + D^T(y - Dx), λ/L)
    /// where L is the Lipschitz constant (largest eigenvalue of D^T D).
    pub fn encode_ista(&self, signal: &[f32], dictionary: &[Vec<f32>]) -> SparseCode {
        let n_atoms = dictionary.len();
        if n_atoms == 0 {
            return SparseCode {
                coefficients: Vec::new(),
                support: Vec::new(),
            };
        }

        // Estimate Lipschitz constant via power iteration on D^T D
        let lipschitz = estimate_lipschitz(dictionary);
        let step = if lipschitz > 1e-10 {
            1.0 / lipschitz
        } else {
            0.01
        };
        let threshold = self.lambda * step;

        let mut x = vec![0.0_f32; n_atoms];
        for _ in 0..self.max_iter {
            let x_old = x.clone();
            // residual = signal - D * x
            let mut residual = signal.to_vec();
            for (j, atom) in dictionary.iter().enumerate() {
                let coef = x[j];
                for (r, &a) in residual.iter_mut().zip(atom.iter()) {
                    *r -= coef * a;
                }
            }
            // gradient step: x += D^T * residual * step
            for (j, atom) in dictionary.iter().enumerate() {
                x[j] += dot(atom, &residual) * step;
            }
            // soft threshold
            for v in x.iter_mut() {
                *v = Self::soft_threshold(*v, threshold);
            }
            // convergence check
            let change: f32 = x
                .iter()
                .zip(x_old.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if change < self.tol {
                break;
            }
        }

        let support: Vec<usize> = x
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| i)
            .collect();
        let coefficients: Vec<f32> = support.iter().map(|&i| x[i]).collect();
        SparseCode {
            coefficients,
            support,
        }
    }

    /// Soft thresholding: sign(x) * max(|x| - threshold, 0).
    #[inline]
    pub fn soft_threshold(x: f32, threshold: f32) -> f32 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }
}

/// Estimate the Lipschitz constant of D^T D via power iteration.
pub(crate) fn estimate_lipschitz(dictionary: &[Vec<f32>]) -> f32 {
    let n = dictionary.len();
    if n == 0 {
        return 1.0;
    }
    let mut v = vec![1.0_f32 / (n as f32).sqrt(); n];
    for _ in 0..20 {
        // w = D^T D v
        let mut w = vec![0.0_f32; n];
        for (i, atom_i) in dictionary.iter().enumerate() {
            for (j, atom_j) in dictionary.iter().enumerate() {
                w[i] += dot(atom_i, atom_j) * v[j];
            }
        }
        let nrm = norm(&w);
        if nrm < 1e-12 {
            return 1.0;
        }
        v = w.iter().map(|&x| x / nrm).collect();
    }
    // Rayleigh quotient
    let mut w = vec![0.0_f32; n];
    for (i, atom_i) in dictionary.iter().enumerate() {
        for (j, atom_j) in dictionary.iter().enumerate() {
            w[i] += dot(atom_i, atom_j) * v[j];
        }
    }
    dot(&w, &v).max(1e-10)
}

// ── DictionaryLearning ────────────────────────────────────────────────────────

/// Configuration for dictionary learning algorithms.
#[derive(Debug, Clone)]
pub struct DlConfig {
    /// Number of dictionary atoms.
    pub n_atoms: usize,
    /// Sparsity target (max non-zero coefficients per encoding).
    pub n_nonzero: usize,
    /// Maximum training iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f32,
}

/// A learned dictionary: a set of unit-norm atoms.
#[derive(Debug, Clone)]
pub struct Dictionary {
    /// Dictionary atoms stored as row vectors.
    pub atoms: Vec<Vec<f32>>,
    /// Number of atoms in the dictionary.
    pub n_atoms: usize,
    /// Dimensionality of each atom.
    pub atom_dim: usize,
}

impl Dictionary {
    /// Create a new dictionary with given atoms.
    pub fn new(atoms: Vec<Vec<f32>>) -> Self {
        let n_atoms = atoms.len();
        let atom_dim = atoms.first().map(|a| a.len()).unwrap_or(0);
        Self {
            atoms,
            n_atoms,
            atom_dim,
        }
    }

    /// Get the i-th atom.
    pub fn atom(&self, i: usize) -> &[f32] {
        &self.atoms[i]
    }

    /// Normalize all atoms to unit L2 norm.
    pub fn normalize_atoms(&mut self) {
        for atom in self.atoms.iter_mut() {
            let n = norm(atom);
            if n > 1e-10 {
                for v in atom.iter_mut() {
                    *v /= n;
                }
            }
        }
    }

    /// Mutual coherence: max |<a_i, a_j>| for i ≠ j.
    pub fn coherence(&self) -> f32 {
        let n = self.atoms.len();
        let mut max_corr = 0.0_f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let c = dot(&self.atoms[i], &self.atoms[j]).abs();
                if c > max_corr {
                    max_corr = c;
                }
            }
        }
        max_corr
    }
}

// ── K-SVD ─────────────────────────────────────────────────────────────────────

/// K-SVD dictionary learning.
pub struct KSvd {
    /// Configuration for dictionary learning.
    pub config: DlConfig,
}

impl KSvd {
    /// Fit a dictionary to `data` using the K-SVD algorithm.
    ///
    /// Alternates between:
    /// 1. Sparse coding step (OMP)
    /// 2. Dictionary update step (rank-1 SVD for each atom)
    pub fn fit(&self, data: &[Vec<f32>]) -> (Dictionary, Vec<SparseCode>) {
        if data.is_empty() {
            let d = Dictionary {
                atoms: Vec::new(),
                n_atoms: 0,
                atom_dim: 0,
            };
            return (d, Vec::new());
        }
        let signal_dim = data[0].len();
        let n_atoms = self.config.n_atoms;
        let n_samples = data.len();

        // Initialize dictionary randomly from data samples
        let mut rng = StdRng::seed_from_u64(42);
        let mut atoms: Vec<Vec<f32>> = (0..n_atoms)
            .map(|i| {
                let sample = &data[i % n_samples];
                let mut atom = sample.clone();
                // Add tiny jitter for distinct atoms
                for v in atom.iter_mut() {
                    *v += (rng.random::<f32>() - 0.5) * 1e-4;
                }
                atom
            })
            .collect();
        // Normalize
        for atom in atoms.iter_mut() {
            let n = norm(atom);
            if n > 1e-10 {
                for v in atom.iter_mut() {
                    *v /= n;
                }
            }
        }

        let omp = OrthogonalMatchingPursuit {
            n_nonzero: self.config.n_nonzero,
        };
        let mut codes: Vec<SparseCode> = data.iter().map(|s| omp.encode(s, &atoms)).collect();

        for _iter in 0..self.config.max_iter {
            let old_atoms = atoms.clone();

            // Dictionary update: for each atom k, update using rank-1 SVD
            for k in 0..n_atoms {
                // Find samples that use atom k
                let using: Vec<usize> = (0..n_samples)
                    .filter(|&i| codes[i].support.contains(&k))
                    .collect();
                if using.is_empty() {
                    continue;
                }

                // E_k = data - sum_{j≠k} c_j * a_j
                let e_k: Vec<Vec<f32>> = using
                    .iter()
                    .map(|&i| {
                        let mut e = data[i].clone();
                        for (&sup_idx, &coef) in
                            codes[i].support.iter().zip(codes[i].coefficients.iter())
                        {
                            if sup_idx == k {
                                continue;
                            }
                            for (ev, &av) in e.iter_mut().zip(atoms[sup_idx].iter()) {
                                *ev -= coef * av;
                            }
                        }
                        e
                    })
                    .collect();
                let (u, sigma, v) = rank1_svd(&e_k, signal_dim, 20);
                atoms[k] = u;
                for (sample_pos, &sample_idx) in using.iter().enumerate() {
                    let new_coef = sigma * v[sample_pos];
                    // Find position of k in the support
                    if let Some(pos) = codes[sample_idx].support.iter().position(|&s| s == k) {
                        codes[sample_idx].coefficients[pos] = new_coef;
                    }
                }
            }

            // Check convergence
            let change: f32 = atoms
                .iter()
                .zip(old_atoms.iter())
                .map(|(a, b)| {
                    a.iter()
                        .zip(b.iter())
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
                .sqrt();
            if change < self.config.tol {
                break;
            }

            // Re-encode all signals
            codes = data.iter().map(|s| omp.encode(s, &atoms)).collect();
        }

        let dict = Dictionary::new(atoms);
        (dict, codes)
    }
}

/// Rank-1 SVD via power iteration. Returns (u, sigma, v).
pub(crate) fn rank1_svd(
    matrix: &[Vec<f32>],
    _signal_dim: usize,
    max_iter: usize,
) -> (Vec<f32>, f32, Vec<f32>) {
    let m = matrix.len();
    if m == 0 {
        return (Vec::new(), 0.0, Vec::new());
    }
    let n = matrix[0].len();
    let mut v: Vec<f32> = (0..m).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let mut u = vec![0.0_f32; n];
    for _ in 0..max_iter {
        for i in 0..n {
            u[i] = matrix
                .iter()
                .zip(v.iter())
                .map(|(row, &vj)| row[i] * vj)
                .sum();
        }
        let sigma = norm(&u);
        if sigma < 1e-12 {
            return (vec![0.0; n], 0.0, vec![0.0; m]);
        }
        for x in u.iter_mut() {
            *x /= sigma;
        }

        // v = matrix^T * u = sum_i u[i] * matrix[j][i]
        for j in 0..m {
            v[j] = u
                .iter()
                .zip(matrix[j].iter())
                .map(|(&ui, &aij)| ui * aij)
                .sum();
        }
        let vnorm = norm(&v);
        if vnorm < 1e-12 {
            break;
        }
        for x in v.iter_mut() {
            *x /= vnorm;
        }
    }

    let mut mv = vec![0.0_f32; n];
    for i in 0..n {
        mv[i] = matrix
            .iter()
            .zip(v.iter())
            .map(|(row, &vj)| row[i] * vj)
            .sum();
    }
    let sigma = dot(&u, &mv);
    (u, sigma, v)
}

// ── OnlineDictionaryLearning ──────────────────────────────────────────────────

/// Online dictionary learning (Mairal et al. 2009).
pub struct OnlineDictionaryLearning {
    /// Dictionary learning configuration.
    pub config: DlConfig,
    /// A accumulator: n_atoms × n_atoms
    pub a_matrix: Vec<Vec<f32>>,
    /// B accumulator: signal_dim × n_atoms
    pub b_matrix: Vec<Vec<f32>>,
}

impl OnlineDictionaryLearning {
    /// Create a new online dictionary learner.
    pub fn new(config: DlConfig, signal_dim: usize) -> Self {
        let n_atoms = config.n_atoms;
        let a_matrix = vec![vec![0.0_f32; n_atoms]; n_atoms];
        let b_matrix = vec![vec![0.0_f32; n_atoms]; signal_dim];
        Self {
            config,
            a_matrix,
            b_matrix,
        }
    }

    /// Process one sample and return updated dictionary.
    pub fn update(&mut self, sample: &[f32], current_dict: &Dictionary) -> Dictionary {
        let omp = OrthogonalMatchingPursuit {
            n_nonzero: self.config.n_nonzero,
        };
        let code = omp.encode(sample, &current_dict.atoms);

        let n_atoms = self.config.n_atoms;
        let signal_dim = sample.len();

        // Build dense coefficient vector
        let mut alpha = vec![0.0_f32; n_atoms];
        for (&idx, &coef) in code.support.iter().zip(code.coefficients.iter()) {
            if idx < n_atoms {
                alpha[idx] = coef;
            }
        }

        // Update A: A += alpha * alpha^T
        for i in 0..n_atoms {
            for j in 0..n_atoms {
                self.a_matrix[i][j] += alpha[i] * alpha[j];
            }
        }
        // Update B: B += sample * alpha^T  (signal_dim × n_atoms)
        for i in 0..signal_dim.min(self.b_matrix.len()) {
            for j in 0..n_atoms {
                self.b_matrix[i][j] += sample[i] * alpha[j];
            }
        }

        // Update dictionary via block-coordinate descent
        let mut new_atoms = current_dict.atoms.clone();
        for k in 0..n_atoms {
            let a_kk = self.a_matrix[k][k];
            if a_kk.abs() < 1e-10 {
                continue;
            }

            // u_k = (b_k - D * a_k + d_k * a_kk) / a_kk
            let mut u_k = vec![0.0_f32; signal_dim];
            for i in 0..signal_dim.min(self.b_matrix.len()) {
                u_k[i] = self.b_matrix[i][k];
            }
            for j in 0..n_atoms {
                if j == k {
                    continue;
                }
                let a_jk = self.a_matrix[j][k];
                for i in 0..signal_dim.min(new_atoms[j].len()) {
                    u_k[i] -= new_atoms[j][i] * a_jk;
                }
            }
            for v in u_k.iter_mut() {
                *v /= a_kk;
            }

            // Normalize
            let n = norm(&u_k);
            if n > 1e-10 {
                for v in u_k.iter_mut() {
                    *v /= n;
                }
            }
            new_atoms[k] = u_k;
        }

        Dictionary::new(new_atoms)
    }

    /// Fit a dictionary on a stream of samples.
    pub fn fit_stream(&mut self, data: &[Vec<f32>]) -> Dictionary {
        if data.is_empty() {
            return Dictionary {
                atoms: Vec::new(),
                n_atoms: 0,
                atom_dim: 0,
            };
        }
        let signal_dim = data[0].len();
        let n_atoms = self.config.n_atoms;

        // Initialize dictionary from random samples
        let mut rng = StdRng::seed_from_u64(123);
        let mut atoms: Vec<Vec<f32>> = (0..n_atoms)
            .map(|i| {
                let mut atom = data[i % data.len()].clone();
                for v in atom.iter_mut() {
                    *v += (rng.random::<f32>() - 0.5) * 1e-3;
                }
                atom
            })
            .collect();
        for atom in atoms.iter_mut() {
            let n = norm(atom);
            if n > 1e-10 {
                for v in atom.iter_mut() {
                    *v /= n;
                }
            }
        }

        let mut dict = Dictionary::new(atoms);
        for sample in data.iter() {
            dict = self.update(sample, &dict);
        }
        dict
    }
}

// ── CompressedSensing ─────────────────────────────────────────────────────────

/// Types of measurement matrices for compressed sensing.
#[derive(Debug, Clone)]
pub enum MeasurementMatrix {
    /// Gaussian random matrix: m measurements, n signal length.
    Gaussian(usize, usize),
    /// Bernoulli ±1 random matrix: m measurements, n signal length.
    Bernoulli(usize, usize),
    /// DCT-based measurement matrix with m measurements.
    Dct(usize),
}

/// Result of compressive measurement.
#[derive(Debug, Clone)]
pub struct CsMeasurement {
    /// Compressed measurement vector.
    pub y: Vec<f32>,
    /// Number of measurements taken.
    pub n_measurements: usize,
    /// Original signal length.
    pub signal_len: usize,
}

/// Measurement matrix stored row by row.
#[derive(Debug, Clone)]
pub struct CsMatrix {
    /// Matrix rows (each row is one measurement vector).
    pub rows: Vec<Vec<f32>>,
}

impl CsMatrix {
    /// Number of rows (measurements).
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }
    /// Number of columns (signal dimension).
    pub fn ncols(&self) -> usize {
        self.rows.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Compute A^T * v  (ncols-dim result).
    pub(crate) fn transpose_mul(&self, v: &[f32]) -> Vec<f32> {
        let ncols = self.ncols();
        let mut result = vec![0.0_f32; ncols];
        for (row, &vi) in self.rows.iter().zip(v.iter()) {
            for (r, &a) in result.iter_mut().zip(row.iter()) {
                *r += a * vi;
            }
        }
        result
    }

    /// Compute A * v  (nrows-dim result).
    pub(crate) fn forward_mul(&self, v: &[f32]) -> Vec<f32> {
        self.rows.iter().map(|row| dot(row, v)).collect()
    }
}

/// Generate a measurement matrix from a specification.
pub fn generate_matrix(m: &MeasurementMatrix, rng: &mut impl Rng) -> CsMatrix {
    match m {
        MeasurementMatrix::Gaussian(rows, cols) => {
            let scale = 1.0 / (*rows as f32).sqrt();
            let rows_data: Vec<Vec<f32>> = (0..*rows)
                .map(|_| (0..*cols).map(|_| box_muller(rng) * scale).collect())
                .collect();
            CsMatrix { rows: rows_data }
        }
        MeasurementMatrix::Bernoulli(rows, cols) => {
            let scale = 1.0 / (*rows as f32).sqrt();
            let rows_data: Vec<Vec<f32>> = (0..*rows)
                .map(|_| {
                    (0..*cols)
                        .map(|_| {
                            if rng.random::<f32>() > 0.5 {
                                scale
                            } else {
                                -scale
                            }
                        })
                        .collect()
                })
                .collect();
            CsMatrix { rows: rows_data }
        }
        MeasurementMatrix::Dct(m_rows) => {
            // Use first m_rows rows from the DCT matrix (size estimated as 2*m_rows)
            let n = (2 * m_rows).max(4);
            let dct_rows: Vec<Vec<f32>> = (0..*m_rows)
                .map(|k| {
                    (0..n)
                        .map(|j| {
                            let scale = if k == 0 {
                                (1.0 / n as f32).sqrt()
                            } else {
                                (2.0 / n as f32).sqrt()
                            };
                            let angle = std::f32::consts::PI * k as f32 * (2 * j + 1) as f32
                                / (2 * n) as f32;
                            scale * angle.cos()
                        })
                        .collect()
                })
                .collect();
            CsMatrix { rows: dct_rows }
        }
    }
}

/// Box-Muller transform for standard normal samples.
pub(crate) fn box_muller(rng: &mut impl Rng) -> f32 {
    let u1 = (rng.random::<f32>()).max(1e-30);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Measure a signal using a compressive sensing matrix.
pub fn measure(signal: &[f32], matrix: &CsMatrix) -> CsMeasurement {
    let signal_len = signal.len();
    let y = matrix.forward_mul(signal);
    let n_measurements = y.len();
    CsMeasurement {
        y,
        n_measurements,
        signal_len,
    }
}

// ── BasisPursuit (ADMM) ───────────────────────────────────────────────────────

/// Basis Pursuit via ADMM: minimize ||x||_1 subject to Ax = y.
pub struct BasisPursuit {
    /// Maximum ADMM iterations.
    pub max_iter: usize,
    /// ADMM penalty parameter rho.
    pub rho: f32,
    /// Primal/dual residual convergence tolerance.
    pub tol: f32,
}

impl BasisPursuit {
    /// Recover the sparse signal from compressed measurements using ADMM.
    pub fn recover(&self, measurement: &CsMeasurement, matrix: &CsMatrix) -> Vec<f32> {
        let n = measurement.signal_len;
        let rho = self.rho;

        // ADMM variables
        let mut x = vec![0.0_f32; n];
        let mut z = vec![0.0_f32; n];
        let mut u = vec![0.0_f32; n]; // scaled dual

        // Precompute A^T * y
        let aty: Vec<f32> = matrix.transpose_mul(&measurement.y);

        for _ in 0..self.max_iter {
            // x-update: (A^T A + rho * I) x = A^T y + rho * (z - u)
            // Use conjugate gradient (small/medium problems)
            let rhs: Vec<f32> = (0..n).map(|i| aty[i] + rho * (z[i] - u[i])).collect();
            x = admm_cg(matrix, rho, &rhs, &x, 50);

            // z-update: soft threshold
            let z_old = z.clone();
            for i in 0..n {
                z[i] = LassoEncoder::soft_threshold(x[i] + u[i], 1.0 / rho);
            }

            // u-update
            for i in 0..n {
                u[i] += x[i] - z[i];
            }

            // Primal and dual residual
            let primal: f32 = (0..n).map(|i| (x[i] - z[i]).powi(2)).sum::<f32>().sqrt();
            let dual: f32 = (0..n)
                .map(|i| (rho * (z[i] - z_old[i])).powi(2))
                .sum::<f32>()
                .sqrt();
            if primal < self.tol && dual < self.tol {
                break;
            }
        }
        z
    }
}

/// Conjugate gradient for (A^T A + rho * I) x = b.
fn admm_cg(matrix: &CsMatrix, rho: f32, b: &[f32], x0: &[f32], max_iter: usize) -> Vec<f32> {
    let n = b.len();
    let mut x = x0.to_vec();
    // r = b - (A^T A + rho I) x0
    let ax = matrix.forward_mul(&x);
    let atax = matrix.transpose_mul(&ax);
    let mut r: Vec<f32> = (0..n).map(|i| b[i] - atax[i] - rho * x[i]).collect();
    let mut p = r.clone();
    let mut rsold: f32 = r.iter().map(|&v| v * v).sum();

    for _ in 0..max_iter {
        if rsold < 1e-10 {
            break;
        }
        let ap_full = matrix.forward_mul(&p);
        let atp = matrix.transpose_mul(&ap_full);
        let ap: Vec<f32> = (0..n).map(|i| atp[i] + rho * p[i]).collect();
        let denom: f32 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
        if denom.abs() < 1e-14 {
            break;
        }
        let alpha = rsold / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rsnew: f32 = r.iter().map(|&v| v * v).sum();
        let beta = rsnew / rsold.max(1e-14);
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    x
}

// ── CoSaMP ────────────────────────────────────────────────────────────────────

/// Compressive Sampling Matching Pursuit.
pub struct CoSaMP {
    /// Sparsity level (number of non-zeros to recover).
    pub n_nonzero: usize,
    /// Maximum iterations.
    pub max_iter: usize,
}

impl CoSaMP {
    /// Recover signal via CoSaMP: support extension + LS + pruning.
    pub fn recover(&self, measurement: &CsMeasurement, matrix: &CsMatrix) -> Vec<f32> {
        let n = measurement.signal_len;
        let s = self.n_nonzero;

        let mut x = vec![0.0_f32; n];

        for _ in 0..self.max_iter {
            // Proxy: A^T * (y - Ax)
            let ax = matrix.forward_mul(&x);
            let residual: Vec<f32> = measurement
                .y
                .iter()
                .zip(ax.iter())
                .map(|(&y, &a)| y - a)
                .collect();
            let proxy = matrix.transpose_mul(&residual);

            // Identify 2s largest components of proxy
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                proxy[b]
                    .abs()
                    .partial_cmp(&proxy[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut support: Vec<usize> = indices[..((2 * s).min(n))].to_vec();

            // Merge with current support
            let current_nonzero: Vec<usize> = x
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0.0)
                .map(|(i, _)| i)
                .collect();
            for idx in current_nonzero {
                if !support.contains(&idx) {
                    support.push(idx);
                }
            }

            // Least squares on merged support
            let coefs = cosamp_ls(&measurement.y, matrix, &support);

            // Prune to s largest
            let mut coef_pairs: Vec<(usize, f32)> = support
                .iter()
                .zip(coefs.iter())
                .map(|(&i, &c)| (i, c))
                .collect();
            coef_pairs.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            coef_pairs.truncate(s);

            x = vec![0.0_f32; n];
            for (i, c) in coef_pairs {
                x[i] = c;
            }

            // Convergence check
            let ax_new = matrix.forward_mul(&x);
            let res_norm: f32 = measurement
                .y
                .iter()
                .zip(ax_new.iter())
                .map(|(&y, &a)| (y - a).powi(2))
                .sum::<f32>()
                .sqrt();
            if res_norm < 1e-6 {
                break;
            }
        }
        x
    }
}

/// Least squares for CoSaMP on a subset of columns.
fn cosamp_ls(y: &[f32], matrix: &CsMatrix, support: &[usize]) -> Vec<f32> {
    let k = support.len();
    let m = y.len();
    // Build A_S (m × k)
    let as_cols: Vec<Vec<f32>> = support
        .iter()
        .map(|&i| {
            matrix
                .rows
                .iter()
                .map(|row| *row.get(i).unwrap_or(&0.0))
                .collect()
        })
        .collect();
    // Solve normal equations: A_S^T A_S x = A_S^T y
    let mut gram = vec![vec![0.0_f32; k]; k];
    for i in 0..k {
        for j in 0..k {
            gram[i][j] = (0..m).map(|r| as_cols[i][r] * as_cols[j][r]).sum();
        }
    }
    let rhs: Vec<f32> = (0..k)
        .map(|i| (0..m).map(|r| as_cols[i][r] * y[r]).sum())
        .collect();
    cholesky_solve(&gram, &rhs).unwrap_or_else(|_| vec![0.0_f32; k])
}

/// Estimate the restricted isometry property constant via random sparse vectors.
pub fn rip_constant_estimate(
    matrix: &CsMatrix,
    s: usize,
    n_trials: usize,
    rng: &mut impl Rng,
) -> f32 {
    let n = matrix.ncols();
    if n == 0 {
        return 0.0;
    }
    let mut max_delta = 0.0_f32;

    for _ in 0..n_trials {
        // Generate random s-sparse unit vector
        let mut support: Vec<usize> = (0..n).collect();
        // Shuffle first s elements
        for i in 0..s.min(n) {
            let j = i + (rng.random_range(0..(n - i)));
            support.swap(i, j);
        }
        let support = &support[..s.min(n)];

        let mut x = vec![0.0_f32; n];
        let mut total_sq = 0.0_f32;
        for &i in support {
            let v = rng.random::<f32>() * 2.0 - 1.0;
            x[i] = v;
            total_sq += v * v;
        }
        if total_sq < 1e-10 {
            continue;
        }
        let x_norm_sq = total_sq;
        for v in x.iter_mut() {
            *v /= x_norm_sq.sqrt();
        }

        let ax = matrix.forward_mul(&x);
        let ax_norm_sq: f32 = ax.iter().map(|&v| v * v).sum();
        let delta = (ax_norm_sq - 1.0).abs();
        if delta > max_delta {
            max_delta = delta;
        }
    }
    max_delta
}

// ── SparseAutoencoder ─────────────────────────────────────────────────────────

/// Configuration for the sparse autoencoder.
#[derive(Debug, Clone)]
pub struct SaeConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden layer dimensionality.
    pub hidden_dim: usize,
    /// Target sparsity level (fraction of active units).
    pub sparsity_target: f32,
    /// Weight for the sparsity penalty.
    pub sparsity_weight: f32,
}

/// Sparse autoencoder with k-winner-takes-all activation.
#[derive(Debug, Clone)]
pub struct SparseAutoencoder {
    /// Encoder weight matrix (hidden_dim × input_dim).
    pub encoder_w: Vec<Vec<f32>>,
    /// Encoder bias vector.
    pub encoder_b: Vec<f32>,
    /// Decoder weight matrix (input_dim × hidden_dim).
    pub decoder_w: Vec<Vec<f32>>,
    /// Decoder bias vector.
    pub decoder_b: Vec<f32>,
}

impl SparseAutoencoder {
    /// Create a new sparse autoencoder with Xavier initialization.
    pub fn new(config: &SaeConfig, rng: &mut impl Rng) -> Self {
        let scale_enc = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let scale_dec = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();

        let encoder_w: Vec<Vec<f32>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| box_muller(rng) * scale_enc)
                    .collect()
            })
            .collect();
        let encoder_b = vec![0.0_f32; config.hidden_dim];

        let decoder_w: Vec<Vec<f32>> = (0..config.input_dim)
            .map(|_| {
                (0..config.hidden_dim)
                    .map(|_| box_muller(rng) * scale_dec)
                    .collect()
            })
            .collect();
        let decoder_b = vec![0.0_f32; config.input_dim];

        Self {
            encoder_w,
            encoder_b,
            decoder_w,
            decoder_b,
        }
    }

    /// Encode with ReLU followed by k-winner-takes-all.
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        let hidden_dim = self.encoder_w.len();
        // Linear + ReLU
        let mut h: Vec<f32> = (0..hidden_dim)
            .map(|i| {
                let pre_act = dot(&self.encoder_w[i], x) + self.encoder_b[i];
                pre_act.max(0.0) // ReLU
            })
            .collect();

        // k-winner-takes-all: keep top-k activations, zero the rest
        let k = ((hidden_dim as f32 * 0.1).ceil() as usize)
            .max(1)
            .min(hidden_dim);
        let mut indexed: Vec<(usize, f32)> = h.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = k;
        let zero_indices: Vec<usize> = indexed[threshold_idx..].iter().map(|(i, _)| *i).collect();
        for i in zero_indices {
            h[i] = 0.0;
        }
        h
    }

    /// Decode the latent representation back to the input space.
    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        let input_dim = self.decoder_w.len();
        (0..input_dim)
            .map(|i| dot(&self.decoder_w[i], z) + self.decoder_b[i])
            .collect()
    }

    /// Mean squared reconstruction error.
    pub fn reconstruction_loss(&self, x: &[f32]) -> f32 {
        let z = self.encode(x);
        let x_hat = self.decode(&z);
        let n = x.len();
        if n == 0 {
            return 0.0;
        }
        x.iter()
            .zip(x_hat.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32
    }

    /// KL-divergence sparsity penalty on hidden activations.
    pub fn sparsity_loss(&self, z: &[f32]) -> f32 {
        if z.is_empty() {
            return 0.0;
        }
        let rho_hat = z.iter().map(|&v| v.clamp(0.0, 1.0)).sum::<f32>() / z.len() as f32;
        let rho = 0.05_f32; // default target
        Self::kl_divergence_bernoulli(rho, rho_hat)
    }

    /// Total loss = reconstruction + sparsity_weight * sparsity.
    pub fn total_loss(&self, x: &[f32], config: &SaeConfig) -> f32 {
        let z = self.encode(x);
        let rec = self.reconstruction_loss(x);
        let spar = self.sparsity_loss_with_target(&z, config.sparsity_target);
        rec + config.sparsity_weight * spar
    }

    fn sparsity_loss_with_target(&self, z: &[f32], target: f32) -> f32 {
        if z.is_empty() {
            return 0.0;
        }
        let rho_hat = z.iter().map(|&v| v.clamp(0.0, 1.0)).sum::<f32>() / z.len() as f32;
        Self::kl_divergence_bernoulli(target, rho_hat)
    }

    /// KL divergence between Bernoulli(rho) and Bernoulli(rho_hat).
    pub fn kl_divergence_bernoulli(rho: f32, rho_hat: f32) -> f32 {
        let eps = 1e-8_f32;
        let rho = rho.clamp(eps, 1.0 - eps);
        let rho_hat = rho_hat.clamp(eps, 1.0 - eps);
        rho * (rho / rho_hat).ln() + (1.0 - rho) * ((1.0 - rho) / (1.0 - rho_hat)).ln()
    }
}

// ── SparseLearningMetrics ─────────────────────────────────────────────────────

/// Normalized L2 reconstruction error.
pub fn reconstruction_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.is_empty() {
        return 0.0;
    }
    let err: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>();
    let norm_orig: f32 = original.iter().map(|v| v * v).sum::<f32>();
    if norm_orig < 1e-10 {
        return err.sqrt();
    }
    (err / norm_orig).sqrt()
}

/// Sparsity ratio: nnz / signal_len.
pub fn sparsity_ratio(code: &SparseCode, signal_len: usize) -> f32 {
    if signal_len == 0 {
        return 0.0;
    }
    code.nnz() as f32 / signal_len as f32
}

/// Welch bound (coherence lower bound): sqrt((n-d) / (d*(n-1))).
pub fn coherence_bound(n_atoms: usize, signal_dim: usize) -> f32 {
    let n = n_atoms as f32;
    let d = signal_dim as f32;
    if n <= 1.0 || d <= 0.0 || n <= d {
        return 0.0;
    }
    ((n - d) / (d * (n - 1.0))).sqrt()
}

/// Recovery quality in PSNR (dB): 10 * log10(peak² / MSE).
pub fn recovery_quality(original: &[f32], recovered: &[f32]) -> f32 {
    if original.is_empty() {
        return 0.0;
    }
    let mse: f32 = original
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;
    if mse < 1e-14 {
        return 100.0;
    }
    let peak = original
        .iter()
        .cloned()
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));
    let peak = if peak < 1e-10 { 1.0 } else { peak };
    10.0 * (peak * peak / mse).log10()
}

/// Summary report for sparse learning evaluation.
#[derive(Debug, Clone)]
pub struct SparseLearningReport {
    /// Average reconstruction error across all signals.
    pub reconstruction_error: f32,
    /// Average sparsity ratio.
    pub sparsity: f32,
    /// Dictionary coherence (max off-diagonal Gram matrix entry).
    pub coherence: f32,
    /// Average signal-to-noise ratio in dB.
    pub snr_db: f32,
}

/// Evaluate a dictionary and sparse coding on a dataset.
pub fn evaluate(data: &[Vec<f32>], dict: &Dictionary, n_nonzero: usize) -> SparseLearningReport {
    let omp = OrthogonalMatchingPursuit { n_nonzero };
    let mut total_rec_err = 0.0_f32;
    let mut total_sparsity = 0.0_f32;
    let mut total_snr = 0.0_f32;
    let n = data.len();

    for signal in data.iter() {
        let code = omp.encode(signal, &dict.atoms);
        let reconstructed = code.reconstruct(&dict.atoms);
        let signal_len = signal.len();
        total_rec_err += reconstruction_error(signal, &reconstructed);
        total_sparsity += sparsity_ratio(&code, signal_len);
        total_snr += recovery_quality(signal, &reconstructed);
    }

    let (rec_err, sparsity, snr) = if n > 0 {
        (
            total_rec_err / n as f32,
            total_sparsity / n as f32,
            total_snr / n as f32,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    SparseLearningReport {
        reconstruction_error: rec_err,
        sparsity,
        coherence: dict.coherence(),
        snr_db: snr,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
pub(crate) fn norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[inline]
pub(crate) fn norm_sq(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}
