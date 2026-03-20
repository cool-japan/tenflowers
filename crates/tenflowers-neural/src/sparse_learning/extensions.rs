//! Extension sparse learning algorithms: matching pursuit variants, regression methods,
//! and subspace pursuit for sparse recovery.

use super::{
    dot, norm, norm_sq, omp_least_squares, OrthogonalMatchingPursuit, LassoEncoder,
};

// ── MatchingPursuit ───────────────────────────────────────────────────────────

/// Result of a matching pursuit decomposition.
#[derive(Debug, Clone)]
pub struct MpResult {
    /// Indices of selected dictionary atoms.
    pub atoms: Vec<usize>,
    /// Coefficients corresponding to the selected atoms.
    pub coefficients: Vec<f32>,
    /// L2 norm of the final residual.
    pub residual_norm: f32,
}

/// Basic Matching Pursuit.
pub struct MatchingPursuit {
    /// Number of atoms to select.
    pub n_atoms: usize,
}

impl MatchingPursuit {
    /// Decompose `signal` into `n_atoms` dictionary atoms (greedy).
    pub fn decompose(&self, signal: &[f32], dictionary: &[Vec<f32>]) -> MpResult {
        let mut residual = signal.to_vec();
        let mut atoms = Vec::with_capacity(self.n_atoms);
        let mut coefficients = Vec::with_capacity(self.n_atoms);

        for _ in 0..self.n_atoms {
            if dictionary.is_empty() {
                break;
            }
            let (best_idx, best_coef) = dictionary
                .iter()
                .enumerate()
                .map(|(i, atom)| {
                    let c = dot(&residual, atom);
                    (i, c)
                })
                .max_by(|a, b| {
                    a.1.abs()
                        .partial_cmp(&b.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or((0, 0.0));

            atoms.push(best_idx);
            coefficients.push(best_coef);

            // Update residual: r -= coef * atom
            for (r, &a) in residual.iter_mut().zip(dictionary[best_idx].iter()) {
                *r -= best_coef * a;
            }

            if norm_sq(&residual) < 1e-12 {
                break;
            }
        }

        let residual_norm = norm(&residual);
        MpResult {
            atoms,
            coefficients,
            residual_norm,
        }
    }
}

/// Weak Matching Pursuit: selects any atom with |inner product| > threshold * max.
pub struct WeakMatchingPursuit {
    /// Fraction of maximum correlation to use as selection threshold.
    pub threshold_ratio: f32,
    /// Maximum number of atoms to select.
    pub n_atoms: usize,
}

impl WeakMatchingPursuit {
    /// Decompose `signal` using weak MP.
    pub fn decompose(&self, signal: &[f32], dictionary: &[Vec<f32>]) -> MpResult {
        let mut residual = signal.to_vec();
        let mut atoms = Vec::with_capacity(self.n_atoms);
        let mut coefficients = Vec::with_capacity(self.n_atoms);

        for _ in 0..self.n_atoms {
            if dictionary.is_empty() {
                break;
            }
            // Find max correlation
            let max_corr: f32 = dictionary
                .iter()
                .map(|atom| dot(&residual, atom).abs())
                .fold(0.0_f32, f32::max);
            let threshold = self.threshold_ratio * max_corr;

            // Pick first atom that exceeds the threshold
            let best = dictionary
                .iter()
                .enumerate()
                .find(|(_, atom)| dot(&residual, atom).abs() >= threshold);

            let (best_idx, best_coef) = match best {
                Some((i, atom)) => (i, dot(&residual, atom)),
                None => break,
            };

            atoms.push(best_idx);
            coefficients.push(best_coef);
            for (r, &a) in residual.iter_mut().zip(dictionary[best_idx].iter()) {
                *r -= best_coef * a;
            }
            if norm_sq(&residual) < 1e-12 {
                break;
            }
        }

        let residual_norm = norm(&residual);
        MpResult {
            atoms,
            coefficients,
            residual_norm,
        }
    }
}

/// Subspace Pursuit for sparse recovery.
pub struct SubspacePursuit {
    /// Sparsity level (number of non-zero coefficients).
    pub n_nonzero: usize,
    /// Maximum iterations.
    pub max_iter: usize,
}

impl SubspacePursuit {
    /// Recover a sparse signal using Subspace Pursuit.
    pub fn recover(&self, signal: &[f32], dictionary: &[Vec<f32>]) -> MpResult {
        let n_atoms = dictionary.len();
        let s = self.n_nonzero.min(n_atoms);

        // Initialize support from OMP
        let omp = OrthogonalMatchingPursuit { n_nonzero: s };
        let initial_code = omp.encode(signal, dictionary);
        let mut support: Vec<usize> = initial_code.support.clone();
        if support.len() < s {
            for i in 0..n_atoms {
                if support.len() >= s {
                    break;
                }
                if !support.contains(&i) {
                    support.push(i);
                }
            }
        }

        let mut coefficients = vec![0.0_f32; s];

        for _ in 0..self.max_iter {
            // Compute proxy: D^T * residual
            let mut residual = signal.to_vec();
            for (&idx, &c) in support.iter().zip(coefficients.iter()) {
                for (r, &a) in residual.iter_mut().zip(dictionary[idx].iter()) {
                    *r -= c * a;
                }
            }
            let proxy: Vec<(usize, f32)> = (0..n_atoms)
                .map(|i| (i, dot(dictionary[i].as_slice(), &residual).abs()))
                .collect();

            // Extend support with s best atoms from proxy
            let mut extended = support.clone();
            let mut sorted_proxy = proxy;
            sorted_proxy
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (idx, _) in sorted_proxy.iter().take(s) {
                if !extended.contains(idx) {
                    extended.push(*idx);
                }
                if extended.len() >= 2 * s {
                    break;
                }
            }

            // LS on extended support
            let ext_coefs = omp_least_squares(signal, dictionary, &extended);

            // Prune to s largest
            let mut pairs: Vec<(usize, f32)> = extended
                .iter()
                .zip(ext_coefs.iter())
                .map(|(&i, &c)| (i, c))
                .collect();
            pairs.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            pairs.truncate(s);

            let new_support: Vec<usize> = pairs.iter().map(|(i, _)| *i).collect();
            let new_coefs: Vec<f32> = pairs.iter().map(|(_, c)| *c).collect();

            if new_support == support {
                break;
            }
            support = new_support;
            coefficients = new_coefs;
        }

        let mut residual = signal.to_vec();
        for (&idx, &c) in support.iter().zip(coefficients.iter()) {
            for (r, &a) in residual.iter_mut().zip(dictionary[idx].iter()) {
                *r -= c * a;
            }
        }
        let residual_norm = norm(&residual);
        MpResult {
            atoms: support,
            coefficients,
            residual_norm,
        }
    }
}

// ── SparseRegression ──────────────────────────────────────────────────────────

/// LASSO regression via coordinate descent.
pub struct LassoRegression {
    /// L1 regularization weight.
    pub lambda: f32,
    /// Maximum coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f32,
}

impl LassoRegression {
    /// Fit LASSO by coordinate descent.
    pub fn fit(&self, x_data: &[Vec<f32>], y: &[f32]) -> Vec<f32> {
        if x_data.is_empty() || y.is_empty() {
            return Vec::new();
        }
        let n_samples = x_data.len();
        let n_features = x_data[0].len();
        let mut coefs = vec![0.0_f32; n_features];

        for _ in 0..self.max_iter {
            let old_coefs = coefs.clone();
            for j in 0..n_features {
                // Compute partial residual: r_i = y_i - sum_{k≠j} X_ik * coef_k
                let rho: f32 = (0..n_samples)
                    .map(|i| {
                        let pred_without_j: f32 = x_data[i]
                            .iter()
                            .enumerate()
                            .map(|(k, &xik)| if k == j { 0.0 } else { xik * coefs[k] })
                            .sum();
                        x_data[i][j] * (y[i] - pred_without_j)
                    })
                    .sum();
                // Column norm squared
                let z: f32 = x_data.iter().map(|row| row[j] * row[j]).sum();
                if z.abs() < 1e-10 {
                    coefs[j] = 0.0;
                    continue;
                }
                coefs[j] = LassoEncoder::soft_threshold(rho / z, self.lambda / z);
            }
            let change: f32 = coefs
                .iter()
                .zip(old_coefs.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if change < self.tol {
                break;
            }
        }
        coefs
    }

    /// Predict using fitted coefficients.
    pub fn predict(&self, x_data: &[Vec<f32>], coefs: &[f32]) -> Vec<f32> {
        x_data.iter().map(|row| dot(row, coefs)).collect()
    }
}

/// Elastic Net regression via coordinate descent.
pub struct ElasticNetRegression {
    /// Combined L1+L2 regularization weight.
    pub alpha: f32,
    /// Ratio of L1 penalty (0 = Ridge, 1 = LASSO).
    pub l1_ratio: f32,
    /// Maximum coordinate descent iterations.
    pub max_iter: usize,
}

impl ElasticNetRegression {
    /// Fit Elastic Net by coordinate descent.
    pub fn fit(&self, x_data: &[Vec<f32>], y: &[f32]) -> Vec<f32> {
        if x_data.is_empty() || y.is_empty() {
            return Vec::new();
        }
        let n_samples = x_data.len();
        let n_features = x_data[0].len();
        let mut coefs = vec![0.0_f32; n_features];
        let lambda1 = self.alpha * self.l1_ratio;
        let lambda2 = self.alpha * (1.0 - self.l1_ratio);

        for _ in 0..self.max_iter {
            let old_coefs = coefs.clone();
            for j in 0..n_features {
                let rho: f32 = (0..n_samples)
                    .map(|i| {
                        let pred_without_j: f32 = x_data[i]
                            .iter()
                            .enumerate()
                            .map(|(k, &xik)| if k == j { 0.0 } else { xik * coefs[k] })
                            .sum();
                        x_data[i][j] * (y[i] - pred_without_j)
                    })
                    .sum();
                let z: f32 = x_data.iter().map(|row| row[j] * row[j]).sum::<f32>()
                    + lambda2 * n_samples as f32;
                if z.abs() < 1e-10 {
                    coefs[j] = 0.0;
                    continue;
                }
                coefs[j] = LassoEncoder::soft_threshold(rho / z, lambda1 / z);
            }
            let change: f32 = coefs
                .iter()
                .zip(old_coefs.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if change < 1e-6 {
                break;
            }
        }
        coefs
    }

    /// Predict using fitted coefficients.
    pub fn predict(&self, x_data: &[Vec<f32>], coefs: &[f32]) -> Vec<f32> {
        x_data.iter().map(|row| dot(row, coefs)).collect()
    }
}

/// Sparse Group LASSO via block coordinate descent.
pub struct SparseGroupLasso {
    /// Feature groups: each entry is a list of feature indices forming one group.
    pub groups: Vec<Vec<usize>>,
    /// Individual L1 penalty weight.
    pub lambda1: f32,
    /// Group L2 penalty weight.
    pub lambda2: f32,
    /// Maximum block coordinate descent iterations.
    pub max_iter: usize,
}

impl SparseGroupLasso {
    /// Fit Sparse Group LASSO.
    pub fn fit(&self, x_data: &[Vec<f32>], y: &[f32]) -> Vec<f32> {
        if x_data.is_empty() || y.is_empty() {
            return Vec::new();
        }
        let n_features = x_data[0].len();
        let mut coefs = vec![0.0_f32; n_features];

        for _ in 0..self.max_iter {
            let old_coefs = coefs.clone();
            for group in self.groups.iter() {
                // Block update for the group
                // Compute group gradient and apply group-level soft threshold
                let group_update = self.group_block_update(x_data, y, &coefs, group);
                for (&j, &v) in group.iter().zip(group_update.iter()) {
                    if j < coefs.len() {
                        coefs[j] = v;
                    }
                }
            }
            let change: f32 = coefs
                .iter()
                .zip(old_coefs.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if change < 1e-6 {
                break;
            }
        }
        coefs
    }

    fn group_block_update(
        &self,
        x_data: &[Vec<f32>],
        y: &[f32],
        coefs: &[f32],
        group: &[usize],
    ) -> Vec<f32> {
        let n_samples = x_data.len();
        let g_size = group.len();
        // Partial residual for the group
        let mut rho = vec![0.0_f32; g_size];
        for i in 0..n_samples {
            let pred_without_group: f32 = x_data[i]
                .iter()
                .enumerate()
                .map(|(k, &xik)| {
                    if group.contains(&k) {
                        0.0
                    } else {
                        xik * coefs[k]
                    }
                })
                .sum();
            let resid = y[i] - pred_without_group;
            for (gi, &j) in group.iter().enumerate() {
                rho[gi] += x_data[i].get(j).copied().unwrap_or(0.0) * resid;
            }
        }
        // Apply individual LASSO soft threshold (lambda1)
        for v in rho.iter_mut() {
            *v = LassoEncoder::soft_threshold(*v, self.lambda1);
        }
        // Apply group-level L2 (lambda2): shrink the group vector
        let group_norm = norm(&rho);
        if group_norm > self.lambda2 {
            let scale = 1.0 - self.lambda2 / group_norm;
            for v in rho.iter_mut() {
                *v *= scale;
            }
        } else {
            rho = vec![0.0_f32; g_size];
        }
        rho
    }

    /// Predict using fitted coefficients.
    pub fn predict(&self, x_data: &[Vec<f32>], coefs: &[f32]) -> Vec<f32> {
        x_data.iter().map(|row| dot(row, coefs)).collect()
    }
}
