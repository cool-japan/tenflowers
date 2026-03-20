//! # Neural Collapse Theory & Applications
//!
//! Implements the Neural Collapse phenomenon (Papyan, Han & Donoho 2020) and
//! its applications to learning, classification, and feature geometry.
//!
//! Neural Collapse describes the geometric structure of final-layer features
//! at the terminal phase of training, characterized by four properties:
//!
//! - **NC1**: Within-class variability collapses to zero
//! - **NC2**: Class means converge to an Equiangular Tight Frame (ETF)
//! - **NC3**: Classifier weights align with class means (self-duality)
//! - **NC4**: Simplified Maximum Class Correlation — decision approaches NCC
//!
//! ## Key References
//!
//! - Papyan, Han & Donoho (2020) "Prevalence of Neural Collapse during the Terminal Phase of DNN Training"
//! - Zhu et al. (2021) "A Geometric Analysis of Neural Collapse with Unconstrained Features"
//! - Yang et al. (2022) "Inducing Neural Collapse in Imbalanced Learning"
//! - Xie et al. (2023) "Neural Collapse Inspired Feature-Classifier Imbalance"
//!
//! All public types use the `Ncl` prefix (Neural CoLlapse) to avoid collisions
//! with other modules (e.g. `neural_compression` which uses `Nc`).

#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]

use std::fmt;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors emitted by the neural-collapse module.
#[derive(Debug, Clone, PartialEq)]
pub enum NclError {
    /// Dimension mismatch, out-of-bounds, or incompatible shapes.
    InvalidDimension(String),
    /// Numerical failure: NaN, divergence, singular matrix, non-positive variance.
    NumericalError(String),
    /// Not enough classes to form a valid ETF or measurement.
    NotEnoughClasses(String),
}

impl fmt::Display for NclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NclError::InvalidDimension(m) => write!(f, "NclError::InvalidDimension: {}", m),
            NclError::NumericalError(m) => write!(f, "NclError::NumericalError: {}", m),
            NclError::NotEnoughClasses(m) => write!(f, "NclError::NotEnoughClasses: {}", m),
        }
    }
}

impl std::error::Error for NclError {}

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Rust math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 pseudo-random number generator (Marsaglia 2003).
/// Returns the next u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Standard-normal sample via Box-Muller (uses two u64 calls).
pub fn ncl_randn(seed: &mut u64) -> f64 {
    let u1 = (xorshift64(seed) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    let u2 = (xorshift64(seed) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Uniform [0, 1) sample from xorshift64.
pub fn ncl_rand01(seed: &mut u64) -> f64 {
    xorshift64(seed) as f64 / (u64::MAX as f64 + 1.0)
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Normalize a vector in-place. Returns Err if zero-length.
fn normalize_inplace(v: &mut [f64]) -> Result<(), NclError> {
    let n = l2_norm(v);
    if n < 1e-12 {
        return Err(NclError::NumericalError(
            "Cannot normalize near-zero vector".to_string(),
        ));
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    Ok(())
}

/// Return a new normalized copy. Returns Err if zero-length.
fn normalize(v: &[f64]) -> Result<Vec<f64>, NclError> {
    let n = l2_norm(v);
    if n < 1e-12 {
        return Err(NclError::NumericalError(
            "Cannot normalize near-zero vector".to_string(),
        ));
    }
    Ok(v.iter().map(|x| x / n).collect())
}

/// Matrix–vector multiply: A [m×n] @ x [n] → y [m].
fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

/// Matrix multiply A [m×k] @ B [k×n] → C [m×n].
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = if m == 0 { 0 } else { a[0].len() };
    let n = if k == 0 || b.is_empty() {
        0
    } else {
        b[0].len()
    };
    let mut c = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k {
                s += a[i][l] * b[l][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Transpose of a matrix [m×n] → [n×m].
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return vec![];
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

/// Frobenius norm of a matrix.
fn frobenius_norm(m: &[Vec<f64>]) -> f64 {
    m.iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// QR decomposition via Gram-Schmidt (returns Q [m×n], R [n×n]).
/// Works on columns of the input matrix A [m×n] (m >= n).
fn gram_schmidt_qr(a: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), NclError> {
    let m = a.len();
    if m == 0 {
        return Err(NclError::InvalidDimension("Empty matrix".to_string()));
    }
    let n = a[0].len();
    // Work column-wise: we need the columns of A
    // a is [m rows × n cols]
    // We QR-decompose so that A = Q R, Q is [m × n] orthogonal
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut r = vec![vec![0.0f64; n]; n];

    for j in 0..n {
        // Extract column j of A
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
        // Subtract projections onto previous q columns
        for (qi, qcol) in q_cols.iter().enumerate() {
            let proj = dot(&v, qcol);
            r[qi][j] = proj;
            for k in 0..m {
                v[k] -= proj * qcol[k];
            }
        }
        let nrm = l2_norm(&v);
        r[j][j] = nrm;
        if nrm < 1e-14 {
            // Numerically rank-deficient column: fill with a random orthogonal vector
            // This is acceptable for ETF construction
            let mut fallback: Vec<f64> = vec![0.0; m];
            fallback[j % m] = 1.0;
            // Re-orthogonalize against existing columns
            for qcol in &q_cols {
                let proj = dot(&fallback, qcol);
                for k in 0..m {
                    fallback[k] -= proj * qcol[k];
                }
            }
            let fb_nrm = l2_norm(&fallback);
            if fb_nrm < 1e-14 {
                return Err(NclError::NumericalError(format!(
                    "Rank-deficient matrix: cannot orthogonalize column {}",
                    j
                )));
            }
            for x in fallback.iter_mut() {
                *x /= fb_nrm;
            }
            q_cols.push(fallback);
        } else {
            for x in v.iter_mut() {
                *x /= nrm;
            }
            q_cols.push(v);
        }
    }

    // Convert q_cols (each of length m) into Q [m × n]
    let mut q = vec![vec![0.0f64; n]; m];
    for j in 0..n {
        for i in 0..m {
            q[i][j] = q_cols[j][i];
        }
    }

    Ok((q, r))
}

/// Soft-max over a slice.
fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / v.len() as f64; v.len()];
    }
    exps.iter().map(|x| x / sum).collect()
}

/// Log-sum-exp for a slice.
fn log_sum_exp(v: &[f64]) -> f64 {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max + v.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 NclEquiangularTightFrame
// ─────────────────────────────────────────────────────────────────────────────

/// An Equiangular Tight Frame (ETF): num_classes unit vectors in R^d satisfying
/// `<h_i, h_j> = -1/(num_classes-1)` for all i ≠ j, and `||h_i|| = 1` for all i.
///
/// The ETF describes the optimal geometry for neural-collapse class prototypes
/// (NC2 property). When d ≥ num_classes-1, a proper ETF exists.
pub struct NclEquiangularTightFrame {
    /// Prototype vectors: [num_classes × d].
    pub prototypes: Vec<Vec<f64>>,
    /// Number of classes.
    pub num_classes: usize,
    /// Feature dimension.
    pub d: usize,
}

impl NclEquiangularTightFrame {
    /// Construct an ETF with num_classes unit vectors in R^d.
    ///
    /// Algorithm (when d ≥ num_classes):
    /// 1. Sample a random matrix R ∈ R^{d × num_classes}.
    /// 2. QR-decompose R → Q ∈ R^{d × num_classes} (orthonormal columns).
    /// 3. The num_classes orthonormal columns give <h_i, h_j> = δ_{ij}, not −1/(num_classes−1).
    ///    To achieve ETF geometry we apply the simplex-ETF construction:
    ///    h_k = sqrt(num_classes/(num_classes-1)) * (e_k - 1/num_classes * 1) lifted into R^d via Q.
    ///    Specifically we form M = Q * S where S is the num_classes×num_classes simplex ETF matrix.
    ///
    /// For num_classes = 2: h_1 = +q_1, h_2 = −q_1 (antipodal). Inner product = −1.
    pub fn new(num_classes: usize, d: usize, seed: &mut u64) -> Result<Self, NclError> {
        if num_classes < 2 {
            return Err(NclError::NotEnoughClasses(format!(
                "ETF requires num_classes ≥ 2, got num_classes = {}",
                num_classes
            )));
        }
        if d < num_classes - 1 {
            return Err(NclError::InvalidDimension(format!(
                "ETF with num_classes = {} requires d ≥ num_classes-1 = {}, got d = {}",
                num_classes,
                num_classes - 1,
                d
            )));
        }

        // Build the num_classes × num_classes simplex ETF matrix S:
        // S_ij = sqrt(num_classes/(num_classes-1)) * (delta_ij - 1/num_classes)
        // S has orthonormal rows and columns 0..num_classes-1 in R^num_classes
        // Actually we need columns (prototypes) that are unit-norm with the right inner products.
        // Classic construction: use the d×num_classes simplex ETF embedded in R^d.
        //
        // Step 1: Build S_K = sqrt(num_classes/(num_classes-1)) * (I_K - 1/num_classes * 1_K 1_K^T) ∈ R^{num_classes×num_classes}
        // This matrix has rank num_classes-1 and its non-zero columns form the ETF.
        //
        // Step 2: Embed into R^d via a random d×num_classes isometry Q.

        // Build S_K (num_classes×num_classes) simplex ETF matrix
        let scale = ((num_classes as f64) / ((num_classes - 1) as f64)).sqrt();
        // S_K[i][j] = scale * (delta_ij - 1/num_classes)
        let mut s_k: Vec<Vec<f64>> = vec![vec![0.0f64; num_classes]; num_classes];
        for i in 0..num_classes {
            for j in 0..num_classes {
                s_k[i][j] = scale * (if i == j { 1.0 } else { 0.0 } - 1.0 / num_classes as f64);
            }
        }
        // Columns of S_K are the ETF prototypes in R^num_classes (unit norm, pairwise inner product -1/(num_classes-1))
        // We need to embed these in R^d (d >= num_classes-1).
        // Use an orthonormal embedding: random d×num_classes matrix Q via Gram-Schmidt.

        // Sample random d × num_classes matrix
        let mut rand_mat: Vec<Vec<f64>> = vec![vec![0.0f64; num_classes]; d];
        for i in 0..d {
            for j in 0..num_classes {
                rand_mat[i][j] = ncl_randn(seed);
            }
        }

        // QR decompose to get orthonormal Q of shape d × num_classes
        let (q_mat, _r_mat) = gram_schmidt_qr(&rand_mat)?;
        // q_mat is d × num_classes orthonormal

        // Prototypes[k] = Q * s_k_col_k  (d-dimensional)
        // s_k_col_k = column k of S_K = [s_k[0][k], s_k[1][k], ..., s_k[num_classes-1][k]]
        let mut prototypes: Vec<Vec<f64>> = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            // Extract column k of s_k
            let s_col: Vec<f64> = (0..num_classes).map(|i| s_k[i][k]).collect();
            // Compute Q * s_col: [d × num_classes] × [num_classes] → [d]
            let proto = matvec(&q_mat, &s_col);
            prototypes.push(proto);
        }

        // Normalize each prototype (they should already be unit-norm, but ensure numerics)
        for proto in prototypes.iter_mut() {
            normalize_inplace(proto)?;
        }

        Ok(NclEquiangularTightFrame {
            prototypes,
            num_classes,
            d,
        })
    }

    /// Verify the ETF property:
    /// - ||h_i|| ≈ 1 for all i
    /// - <h_i, h_j> ≈ -1/(num_classes-1) for i ≠ j
    pub fn verify_etf_property(&self, tol: f64) -> bool {
        let target_ip = Self::theoretical_inner_product(self.num_classes);
        for i in 0..self.num_classes {
            // Check norm
            let nrm = l2_norm(&self.prototypes[i]);
            if (nrm - 1.0).abs() > tol {
                return false;
            }
            for j in (i + 1)..self.num_classes {
                let ip = dot(&self.prototypes[i], &self.prototypes[j]);
                if (ip - target_ip).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Theoretical pairwise inner product for an ETF with num_classes classes: -1/(num_classes-1).
    pub fn theoretical_inner_product(num_classes: usize) -> f64 {
        if num_classes <= 1 {
            return 0.0;
        }
        -1.0 / (num_classes as f64 - 1.0)
    }

    /// Compute the Gram matrix G\[i\]\[j\] = <h_i, h_j>.
    pub fn gram_matrix(&self) -> Vec<Vec<f64>> {
        let mut g = vec![vec![0.0f64; self.num_classes]; self.num_classes];
        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                g[i][j] = dot(&self.prototypes[i], &self.prototypes[j]);
            }
        }
        g
    }

    /// Find the nearest class via argmax_k <query, h_k>.
    pub fn nearest_class(&self, query: &[f64]) -> usize {
        let mut best_k = 0usize;
        let mut best_ip = f64::NEG_INFINITY;
        for (k, proto) in self.prototypes.iter().enumerate() {
            let ip = dot(query, proto);
            if ip > best_ip {
                best_ip = ip;
                best_k = k;
            }
        }
        best_k
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 NclNeuralCollapseMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-class statistics used for measuring Neural Collapse properties.
pub struct NclFeatureStats {
    /// Per-class mean features: [num_classes × d].
    pub class_means: Vec<Vec<f64>>,
    /// Global mean feature: \[d\].
    pub global_mean: Vec<f64>,
    /// Within-class scatter matrix S_W: [d × d].
    pub within_class_cov: Vec<Vec<f64>>,
    /// Between-class scatter matrix S_B: [d × d].
    pub between_class_cov: Vec<Vec<f64>>,
    /// Number of classes.
    pub num_classes: usize,
    /// Feature dimension.
    pub d: usize,
}

/// Computes Neural Collapse metrics (NC1–NC4) from feature vectors.
pub struct NclNeuralCollapseMetrics;

impl NclNeuralCollapseMetrics {
    /// Compute per-class and global feature statistics from labeled feature data.
    ///
    /// # Arguments
    /// - `features`: [n_samples × d] feature vectors
    /// - `labels`: class labels in [0, num_classes)
    /// - `num_classes`: total number of classes
    pub fn compute_stats(
        features: &[Vec<f64>],
        labels: &[usize],
        num_classes: usize,
    ) -> Result<NclFeatureStats, NclError> {
        if features.is_empty() {
            return Err(NclError::InvalidDimension("Empty feature set".to_string()));
        }
        if features.len() != labels.len() {
            return Err(NclError::InvalidDimension(format!(
                "features len {} != labels len {}",
                features.len(),
                labels.len()
            )));
        }
        if num_classes < 2 {
            return Err(NclError::NotEnoughClasses(
                "Need at least num_classes=2 classes".to_string(),
            ));
        }
        let d = features[0].len();
        if d == 0 {
            return Err(NclError::InvalidDimension(
                "Feature dimension is 0".to_string(),
            ));
        }
        // Verify all labels < num_classes and all feature dimensions match
        for (i, lbl) in labels.iter().enumerate() {
            if *lbl >= num_classes {
                return Err(NclError::InvalidDimension(format!(
                    "Label {} at index {} is out of range [0, {})",
                    lbl, i, num_classes
                )));
            }
            if features[i].len() != d {
                return Err(NclError::InvalidDimension(format!(
                    "Feature at index {} has dim {}, expected {}",
                    i,
                    features[i].len(),
                    d
                )));
            }
        }

        // Count per-class samples
        let mut counts = vec![0usize; num_classes];
        for &lbl in labels.iter() {
            counts[lbl] += 1;
        }
        // Ensure every class has at least one sample
        for (k, &cnt) in counts.iter().enumerate() {
            if cnt == 0 {
                return Err(NclError::InvalidDimension(format!(
                    "Class {} has no samples",
                    k
                )));
            }
        }

        let n = features.len();

        // Global mean
        let mut global_mean = vec![0.0f64; d];
        for feat in features.iter() {
            for j in 0..d {
                global_mean[j] += feat[j];
            }
        }
        for j in 0..d {
            global_mean[j] /= n as f64;
        }

        // Per-class means
        let mut class_means = vec![vec![0.0f64; d]; num_classes];
        for (feat, &lbl) in features.iter().zip(labels.iter()) {
            for j in 0..d {
                class_means[lbl][j] += feat[j];
            }
        }
        for k in 0..num_classes {
            for j in 0..d {
                class_means[k][j] /= counts[k] as f64;
            }
        }

        // Within-class scatter S_W = 1/n * sum_i (x_i - mu_k) (x_i - mu_k)^T
        let mut sw = vec![vec![0.0f64; d]; d];
        for (feat, &lbl) in features.iter().zip(labels.iter()) {
            let mu = &class_means[lbl];
            let diff: Vec<f64> = (0..d).map(|j| feat[j] - mu[j]).collect();
            for r in 0..d {
                for c in 0..d {
                    sw[r][c] += diff[r] * diff[c];
                }
            }
        }
        for r in 0..d {
            for c in 0..d {
                sw[r][c] /= n as f64;
            }
        }

        // Between-class scatter S_B = 1/num_classes * sum_k n_k * (mu_k - mu_g) (mu_k - mu_g)^T
        let mut sb = vec![vec![0.0f64; d]; d];
        for k in 0..num_classes {
            let mu = &class_means[k];
            let diff: Vec<f64> = (0..d).map(|j| mu[j] - global_mean[j]).collect();
            let nk = counts[k] as f64;
            for r in 0..d {
                for c in 0..d {
                    sb[r][c] += nk * diff[r] * diff[c];
                }
            }
        }
        for r in 0..d {
            for c in 0..d {
                sb[r][c] /= n as f64;
            }
        }

        Ok(NclFeatureStats {
            class_means,
            global_mean,
            within_class_cov: sw,
            between_class_cov: sb,
            num_classes,
            d,
        })
    }

    /// **NC1** — Within-class variability collapse.
    ///
    /// Computes `trace(S_W) / trace(S_B)`. Lower values (→ 0) indicate NC1 collapse.
    /// Returns 0.0 if trace(S_B) ≈ 0 (already fully collapsed).
    pub fn nc1_within_class_variability(stats: &NclFeatureStats) -> f64 {
        let trace_sw: f64 = (0..stats.d).map(|j| stats.within_class_cov[j][j]).sum();
        let trace_sb: f64 = (0..stats.d).map(|j| stats.between_class_cov[j][j]).sum();
        if trace_sb < 1e-12 {
            // If between-class variance is zero, classes are not separated; NC1 is ill-defined.
            // Return the raw within-class trace normalized by 1 to avoid NaN.
            return trace_sw.max(0.0);
        }
        (trace_sw / trace_sb).max(0.0)
    }

    /// **NC2** — Convergence of class means to ETF geometry.
    ///
    /// Centers class means by subtracting the global mean, then computes:
    /// `mean over i≠j of |<h_i, h_j> / (||h_i|| ||h_j||) - (-1/(num_classes-1))|`
    /// Lower = closer to ETF. Returns deviation from ideal ETF.
    pub fn nc2_convergence_to_etf(stats: &NclFeatureStats) -> f64 {
        let num_classes = stats.num_classes;
        if num_classes < 2 {
            return 0.0;
        }
        let target = -1.0 / (num_classes as f64 - 1.0);

        // Center class means
        let centered: Vec<Vec<f64>> = stats
            .class_means
            .iter()
            .map(|mu| {
                (0..stats.d)
                    .map(|j| mu[j] - stats.global_mean[j])
                    .collect::<Vec<f64>>()
            })
            .collect();

        let norms: Vec<f64> = centered.iter().map(|v| l2_norm(v)).collect();

        let mut total_dev = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_classes {
            for j in (i + 1)..num_classes {
                let ni = norms[i];
                let nj = norms[j];
                if ni < 1e-12 || nj < 1e-12 {
                    continue;
                }
                let cos_ij = dot(&centered[i], &centered[j]) / (ni * nj);
                total_dev += (cos_ij - target).abs();
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        total_dev / count as f64
    }

    /// **NC3** — Self-duality (alignment of classifier weights with class means).
    ///
    /// Measures `||W_norm - H_norm||_F / sqrt(num_classes)` where:
    /// - W_norm\[k\] = classifier_weights\[k\] / ||classifier_weights\[k\]||
    /// - H_norm\[k\] = (class_mean\[k\] - global_mean) / ||(class_mean\[k\] - global_mean)||
    ///
    /// Returns a normalized Frobenius distance in [0, sqrt(2)] (0 = perfect alignment).
    pub fn nc3_duality_gap(stats: &NclFeatureStats, classifier_weights: &[Vec<f64>]) -> f64 {
        let num_classes = stats.num_classes;
        if classifier_weights.len() != num_classes {
            return f64::NAN;
        }
        let d = stats.d;

        // Normalize classifier weights
        let w_norms: Vec<Vec<f64>> = classifier_weights
            .iter()
            .map(|w| {
                let nrm = l2_norm(w);
                if nrm < 1e-12 {
                    w.clone()
                } else {
                    w.iter().map(|x| x / nrm).collect()
                }
            })
            .collect();

        // Normalize centered class means
        let h_norms: Vec<Vec<f64>> = stats
            .class_means
            .iter()
            .map(|mu| {
                let centered: Vec<f64> = (0..d).map(|j| mu[j] - stats.global_mean[j]).collect();
                let nrm = l2_norm(&centered);
                if nrm < 1e-12 {
                    centered
                } else {
                    centered.iter().map(|x| x / nrm).collect()
                }
            })
            .collect();

        // ||W_norm - H_norm||_F
        let mut frob_sq = 0.0f64;
        for k in 0..num_classes {
            if w_norms[k].len() != d || h_norms[k].len() != d {
                return f64::NAN;
            }
            for j in 0..d {
                let diff = w_norms[k][j] - h_norms[k][j];
                frob_sq += diff * diff;
            }
        }
        (frob_sq / num_classes as f64).sqrt()
    }

    /// **NC4** — Fraction of class pairs whose feature-mean angle approximates the ETF angle.
    ///
    /// Returns the fraction of (i, j) pairs where `|<h_i, h_j>/(||h_i|| ||h_j||) - target| < 0.1`.
    /// Higher = more ETF-like.
    pub fn nc4_simple_mnc(stats: &NclFeatureStats) -> f64 {
        let num_classes = stats.num_classes;
        if num_classes < 2 {
            return 1.0;
        }
        let target = NclEquiangularTightFrame::theoretical_inner_product(num_classes);
        let centered: Vec<Vec<f64>> = stats
            .class_means
            .iter()
            .map(|mu| {
                (0..stats.d)
                    .map(|j| mu[j] - stats.global_mean[j])
                    .collect::<Vec<f64>>()
            })
            .collect();
        let norms: Vec<f64> = centered.iter().map(|v| l2_norm(v)).collect();

        let tol = 0.1;
        let mut good = 0usize;
        let mut total = 0usize;
        for i in 0..num_classes {
            for j in (i + 1)..num_classes {
                let ni = norms[i];
                let nj = norms[j];
                if ni < 1e-12 || nj < 1e-12 {
                    continue;
                }
                let cos_ij = dot(&centered[i], &centered[j]) / (ni * nj);
                if (cos_ij - target).abs() < tol {
                    good += 1;
                }
                total += 1;
            }
        }
        if total == 0 {
            return 1.0;
        }
        good as f64 / total as f64
    }

    /// Combined Neural Collapse index.
    ///
    /// Combines NC1 (inverted), NC2 (inverted), NC4, and a structural score.
    /// Lower = better neural collapse overall.
    pub fn collapse_index(stats: &NclFeatureStats) -> f64 {
        let nc1 = Self::nc1_within_class_variability(stats);
        let nc2 = Self::nc2_convergence_to_etf(stats);
        let nc4 = Self::nc4_simple_mnc(stats);
        // nc4 in [0,1]: higher = better ETF alignment → invert
        let nc4_inv = 1.0 - nc4;
        // Normalize nc1 and nc2 to [0,1] using sigmoid-like mapping
        let nc1_norm = nc1 / (1.0 + nc1);
        let nc2_norm = nc2.min(1.0);
        (nc1_norm + nc2_norm + nc4_inv) / 3.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 NclEtfClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// Neural Collapse-inspired classifier with fixed ETF weights.
///
/// The classifier weights are fixed as an ETF (not learned). Only the feature
/// projection layer is trained. This simplifies optimization and can improve
/// generalization by enforcing the NC2 geometry from the start.
pub struct NclEtfClassifier {
    /// Fixed ETF geometry — num_classes unit vectors in R^d.
    pub etf: NclEquiangularTightFrame,
    /// Feature projection weights: [feat_dim × input_dim].
    pub feat_proj_w: Vec<Vec<f64>>,
    /// Feature projection bias: \[feat_dim\].
    pub feat_proj_b: Vec<f64>,
    /// Input dimension.
    pub input_dim: usize,
    /// Feature dimension (must satisfy feat_dim ≥ num_classes-1).
    pub feat_dim: usize,
    /// Number of classes.
    pub num_classes: usize,
}

impl NclEtfClassifier {
    /// Create a new ETF classifier with random initialization.
    pub fn new(
        input_dim: usize,
        feat_dim: usize,
        num_classes: usize,
        seed: &mut u64,
    ) -> Result<Self, NclError> {
        if input_dim == 0 {
            return Err(NclError::InvalidDimension(
                "input_dim must be > 0".to_string(),
            ));
        }
        if feat_dim < num_classes.saturating_sub(1).max(1) {
            return Err(NclError::InvalidDimension(format!(
                "feat_dim {} must be >= num_classes-1 = {}",
                feat_dim,
                num_classes.saturating_sub(1)
            )));
        }

        let etf = NclEquiangularTightFrame::new(num_classes, feat_dim, seed)?;

        // Xavier uniform initialization for projection weights
        let limit = (6.0 / (input_dim + feat_dim) as f64).sqrt();
        let mut feat_proj_w = vec![vec![0.0f64; input_dim]; feat_dim];
        for row in feat_proj_w.iter_mut() {
            for x in row.iter_mut() {
                *x = (ncl_rand01(seed) * 2.0 - 1.0) * limit;
            }
        }
        let feat_proj_b = vec![0.0f64; feat_dim];

        Ok(NclEtfClassifier {
            etf,
            feat_proj_w,
            feat_proj_b,
            input_dim,
            feat_dim,
            num_classes,
        })
    }

    /// Extract L2-normalized features: z = normalize(W @ x + b).
    pub fn extract_features(&self, x: &[f64]) -> Vec<f64> {
        // Compute W @ x + b
        let mut z: Vec<f64> = (0..self.feat_dim)
            .map(|i| {
                let wx: f64 = self.feat_proj_w[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xi)| w * xi)
                    .sum();
                wx + self.feat_proj_b[i]
            })
            .collect();
        // L2 normalize
        let nrm = l2_norm(&z);
        if nrm > 1e-12 {
            for v in z.iter_mut() {
                *v /= nrm;
            }
        }
        z
    }

    /// Classify input by nearest ETF prototype.
    pub fn classify(&self, x: &[f64]) -> usize {
        let z = self.extract_features(x);
        self.etf.nearest_class(&z)
    }

    /// Compute logits: <z, h_k> for all k.
    pub fn logits(&self, x: &[f64]) -> Vec<f64> {
        let z = self.extract_features(x);
        self.etf.prototypes.iter().map(|h| dot(&z, h)).collect()
    }

    /// MSE loss toward the ETF target: ||z - h_label||^2.
    pub fn mse_nc_loss(&self, x: &[f64], label: usize) -> f64 {
        let z = self.extract_features(x);
        let h = &self.etf.prototypes[label.min(self.num_classes - 1)];
        z.iter()
            .zip(h.iter())
            .map(|(zi, hi)| (zi - hi).powi(2))
            .sum()
    }

    /// Cross-entropy loss: -log softmax(logits)\[label\].
    pub fn cross_entropy_loss(&self, x: &[f64], label: usize) -> f64 {
        let lgts = self.logits(x);
        let lse = log_sum_exp(&lgts);
        let safe_label = label.min(self.num_classes - 1);
        lse - lgts[safe_label]
    }

    /// Perform a gradient step on the MSE-NC loss toward the ETF target.
    ///
    /// Gradient: dL/dz = 2(z - h_y); chain rule through normalize, then through W.
    /// Returns the loss value.
    pub fn update(&mut self, x: &[f64], label: usize, lr: f64) -> f64 {
        let safe_label = label.min(self.num_classes - 1);
        let h = self.etf.prototypes[safe_label].clone();

        // Forward: pre-norm = W @ x + b
        let pre_norm: Vec<f64> = (0..self.feat_dim)
            .map(|i| {
                let wx: f64 = self.feat_proj_w[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xi)| w * xi)
                    .sum();
                wx + self.feat_proj_b[i]
            })
            .collect();

        let nrm = l2_norm(&pre_norm);
        let z = if nrm > 1e-12 {
            pre_norm.iter().map(|v| v / nrm).collect::<Vec<_>>()
        } else {
            pre_norm.clone()
        };

        // Loss: ||z - h||^2
        let loss: f64 = z
            .iter()
            .zip(h.iter())
            .map(|(zi, hi)| (zi - hi).powi(2))
            .sum();

        // Gradient of loss w.r.t. z: dL/dz = 2(z - h)
        let dldz: Vec<f64> = z
            .iter()
            .zip(h.iter())
            .map(|(zi, hi)| 2.0 * (zi - hi))
            .collect();

        // Gradient through L2 normalization: dz/d(pre_norm)
        // d(z)/d(p) = (I - z z^T) / nrm  where z = p / nrm
        // dL/dp = (I - z z^T) / nrm * dL/dz
        let z_dot_dldz: f64 = dot(&z, &dldz);
        let dldp: Vec<f64> = if nrm > 1e-12 {
            dldz.iter()
                .zip(z.iter())
                .map(|(g, zi)| (g - z_dot_dldz * zi) / nrm)
                .collect()
        } else {
            dldz.clone()
        };

        // dL/dW[i][j] = dldp[i] * x[j]
        // dL/db[i] = dldp[i]
        for i in 0..self.feat_dim {
            for j in 0..self.input_dim {
                self.feat_proj_w[i][j] -= lr * dldp[i] * x[j];
            }
            self.feat_proj_b[i] -= lr * dldp[i];
        }

        loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 NclDrLoss
// ─────────────────────────────────────────────────────────────────────────────

/// DR Loss (Spectral Decoupled Neural Collapse loss, Xie et al. 2023).
///
/// Decomposes the CE loss into discriminative and regularization (Gram) terms:
/// `L_DR = CE(logits/T, labels) + lambda * ||G - G_ETF||_F^2`
pub struct NclDrLoss {
    /// Regularization strength for the Gram-matrix loss.
    pub lambda: f64,
}

impl NclDrLoss {
    /// Construct a DR loss with regularization coefficient λ.
    pub fn new(lambda: f64) -> Self {
        NclDrLoss { lambda }
    }

    /// Compute DR loss over a batch.
    ///
    /// # Arguments
    /// - `logits`: [n_samples × num_classes] pre-softmax outputs
    /// - `labels`: \[n_samples\] ground-truth class indices
    /// - `temperature`: temperature for softmax scaling
    pub fn compute(&self, logits: &[Vec<f64>], labels: &[usize], temperature: f64) -> f64 {
        if logits.is_empty() || labels.is_empty() {
            return 0.0;
        }
        let n = logits.len().min(labels.len());
        let k = logits[0].len();
        let t = temperature.max(1e-6);

        // Cross-entropy loss (with temperature)
        let mut ce = 0.0f64;
        for i in 0..n {
            let scaled: Vec<f64> = logits[i].iter().map(|x| x / t).collect();
            let lse = log_sum_exp(&scaled);
            let safe_lbl = labels[i].min(k.saturating_sub(1));
            ce += lse - scaled[safe_lbl];
        }
        ce /= n as f64;

        // Gram-matrix regularization
        let gram_reg = Self::gram_loss(logits, labels, k);

        ce + self.lambda * gram_reg
    }

    /// Compute ||G_pred - G_ETF||_F^2 where G is the Gram matrix of class-mean logits.
    pub fn gram_loss(logits: &[Vec<f64>], labels: &[usize], num_classes: usize) -> f64 {
        if logits.is_empty() || num_classes < 2 {
            return 0.0;
        }
        let class_means = Self::class_mean_logits(logits, labels, num_classes);
        let target_ip = -1.0 / (num_classes as f64 - 1.0);

        // Normalize class means
        let norms: Vec<f64> = class_means.iter().map(|v| l2_norm(v)).collect();
        let mut loss = 0.0f64;
        for i in 0..num_classes {
            for j in 0..num_classes {
                let ni = norms[i];
                let nj = norms[j];
                let g_ij = if ni < 1e-12 || nj < 1e-12 {
                    0.0
                } else {
                    dot(&class_means[i], &class_means[j]) / (ni * nj)
                };
                let target = if i == j { 1.0 } else { target_ip };
                loss += (g_ij - target).powi(2);
            }
        }
        loss
    }

    /// Compute per-class mean logit vectors: [num_classes × num_classes].
    pub fn class_mean_logits(
        logits: &[Vec<f64>],
        labels: &[usize],
        num_classes: usize,
    ) -> Vec<Vec<f64>> {
        if logits.is_empty() || num_classes == 0 {
            return vec![vec![]; num_classes];
        }
        let logit_dim = logits[0].len();
        let mut means = vec![vec![0.0f64; logit_dim]; num_classes];
        let mut counts = vec![0usize; num_classes];
        let n = logits.len().min(labels.len());
        for i in 0..n {
            let lbl = labels[i].min(num_classes - 1);
            for j in 0..logit_dim.min(means[lbl].len()) {
                means[lbl][j] += logits[i][j];
            }
            counts[lbl] += 1;
        }
        for k in 0..num_classes {
            if counts[k] > 0 {
                for j in 0..logit_dim {
                    means[k][j] /= counts[k] as f64;
                }
            }
        }
        means
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 NclSupcon
// ─────────────────────────────────────────────────────────────────────────────

/// Neural Collapse with Supervised Contrastive Loss (Khosla et al. 2020),
/// modified to encourage ETF geometry.
pub struct NclSupcon {
    /// Temperature parameter τ.
    pub temperature: f64,
    /// Number of positive pairs per anchor (informational, not used in loss directly).
    pub n_positives: usize,
}

impl NclSupcon {
    /// Construct with the given temperature.
    pub fn new(temperature: f64) -> Self {
        NclSupcon {
            temperature: temperature.max(1e-6),
            n_positives: 1,
        }
    }

    /// Supervised Contrastive Loss (SupCon).
    ///
    /// For each anchor `i`, attract same-class examples, repel different-class ones:
    /// `L = -1/N * sum_i [ 1/|P(i)| * sum_{p in P(i)} log[ exp(z_i·z_p/T) / sum_{k≠i} exp(z_i·z_k/T) ] ]`
    ///
    /// # Arguments
    /// - `features`: [2N × d] L2-normalized features (2 augmented views per sample)
    /// - `labels`:   \[2N\] labels (same label for both views of each sample)
    pub fn loss(&self, features: &[Vec<f64>], labels: &[usize]) -> Result<f64, NclError> {
        let n = features.len();
        if n < 2 {
            return Err(NclError::InvalidDimension(
                "SupCon requires at least 2 samples".to_string(),
            ));
        }
        if n != labels.len() {
            return Err(NclError::InvalidDimension(format!(
                "features len {} != labels len {}",
                n,
                labels.len()
            )));
        }
        let t = self.temperature;
        let mut total_loss = 0.0f64;
        let mut valid_anchors = 0usize;

        for i in 0..n {
            // Build positive set P(i) = {k ≠ i : labels[k] == labels[i]}
            let positives: Vec<usize> = (0..n)
                .filter(|&k| k != i && labels[k] == labels[i])
                .collect();
            if positives.is_empty() {
                continue;
            }
            // Denominator: sum over k ≠ i of exp(z_i · z_k / T)
            let denom_logits: Vec<f64> = (0..n)
                .filter(|&k| k != i)
                .map(|k| dot(&features[i], &features[k]) / t)
                .collect();
            let log_denom = log_sum_exp(&denom_logits);

            // Numerator: sum over positives
            let mut pos_loss = 0.0f64;
            for &p in &positives {
                let log_num = dot(&features[i], &features[p]) / t;
                pos_loss += log_num - log_denom;
            }
            total_loss -= pos_loss / positives.len() as f64;
            valid_anchors += 1;
        }

        if valid_anchors == 0 {
            return Ok(0.0);
        }
        Ok(total_loss / valid_anchors as f64)
    }

    /// NC-SupCon: replace positive examples with ETF prototypes.
    ///
    /// For each anchor z_i with label y_i:
    /// `L = -log [ exp(z_i · h_{y_i} / T) / sum_k exp(z_i · h_k / T) ]`
    pub fn nc_supcon_loss(
        &self,
        features: &[Vec<f64>],
        labels: &[usize],
        etf: &NclEquiangularTightFrame,
    ) -> Result<f64, NclError> {
        let n = features.len();
        if n == 0 {
            return Err(NclError::InvalidDimension("Empty features".to_string()));
        }
        if n != labels.len() {
            return Err(NclError::InvalidDimension(format!(
                "features len {} != labels len {}",
                n,
                labels.len()
            )));
        }
        let t = self.temperature;
        let num_classes = etf.num_classes;
        let mut total_loss = 0.0f64;

        for i in 0..n {
            let lbl = labels[i].min(num_classes.saturating_sub(1));
            // Logits: z_i · h_k / T for all k
            let all_logits: Vec<f64> = etf
                .prototypes
                .iter()
                .map(|h| dot(&features[i], h) / t)
                .collect();
            let lse = log_sum_exp(&all_logits);
            // Loss for this anchor
            total_loss += lse - all_logits[lbl];
        }
        Ok(total_loss / n as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 NclPrototypeClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// Neural Collapse-style nearest-prototype classifier.
///
/// Maintains a set of num_classes class prototypes (one per class) that are updated via
/// exponential moving average (EMA) as new features arrive.
pub struct NclPrototypeClassifier {
    /// Learned/updated class prototypes: [num_classes × d].
    pub prototypes: Vec<Vec<f64>>,
    /// Number of classes.
    pub num_classes: usize,
    /// Feature dimension.
    pub d: usize,
    /// EMA momentum (0 = fully new, 1 = no update).
    pub update_momentum: f64,
}

impl NclPrototypeClassifier {
    /// Construct a classifier with zero-initialized prototypes.
    pub fn new(num_classes: usize, d: usize, momentum: f64) -> Self {
        NclPrototypeClassifier {
            prototypes: vec![vec![0.0f64; d]; num_classes],
            num_classes,
            d,
            update_momentum: momentum.clamp(0.0, 1.0),
        }
    }

    /// Initialize prototypes as class means from labeled data.
    pub fn initialize_from_data(
        &mut self,
        features: &[Vec<f64>],
        labels: &[usize],
    ) -> Result<(), NclError> {
        if features.is_empty() {
            return Err(NclError::InvalidDimension("Empty feature set".to_string()));
        }
        let d = features[0].len();
        if d != self.d {
            return Err(NclError::InvalidDimension(format!(
                "Feature dim {} != classifier dim {}",
                d, self.d
            )));
        }
        let mut sums = vec![vec![0.0f64; d]; self.num_classes];
        let mut counts = vec![0usize; self.num_classes];
        for (feat, &lbl) in features.iter().zip(labels.iter()) {
            if lbl >= self.num_classes {
                return Err(NclError::InvalidDimension(format!(
                    "Label {} out of range [0, {})",
                    lbl, self.num_classes
                )));
            }
            if feat.len() != d {
                return Err(NclError::InvalidDimension(
                    "Inconsistent feature dimension".to_string(),
                ));
            }
            for j in 0..d {
                sums[lbl][j] += feat[j];
            }
            counts[lbl] += 1;
        }
        for k in 0..self.num_classes {
            if counts[k] > 0 {
                for j in 0..d {
                    self.prototypes[k][j] = sums[k][j] / counts[k] as f64;
                }
            }
        }
        Ok(())
    }

    /// Classify by nearest prototype (L2 distance).
    pub fn classify(&self, feature: &[f64]) -> usize {
        let mut best_k = 0usize;
        let mut best_dist = f64::INFINITY;
        for k in 0..self.num_classes {
            let dist: f64 = feature
                .iter()
                .zip(self.prototypes[k].iter())
                .map(|(fi, pi)| (fi - pi).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }
        best_k
    }

    /// Update the prototype for class `label` via EMA:
    /// `proto_k = m * proto_k + (1-m) * z_k`
    pub fn update_prototype_ema(&mut self, feature: &[f64], label: usize) {
        let k = label.min(self.num_classes - 1);
        let m = self.update_momentum;
        let d = self.d.min(feature.len());
        for j in 0..d {
            self.prototypes[k][j] = m * self.prototypes[k][j] + (1.0 - m) * feature[j];
        }
    }

    /// Compute [num_classes × num_classes] pairwise L2-distance matrix between prototypes.
    pub fn prototype_distances(&self) -> Vec<Vec<f64>> {
        let mut dist = vec![vec![0.0f64; self.num_classes]; self.num_classes];
        for i in 0..self.num_classes {
            for j in (i + 1)..self.num_classes {
                let d: f64 = self.prototypes[i]
                    .iter()
                    .zip(self.prototypes[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        dist
    }

    /// Measure how close the prototypes are to ETF geometry.
    ///
    /// Returns the mean absolute deviation from the theoretical inner product -1/(num_classes-1).
    pub fn measure_etf_alignment(&self) -> f64 {
        let num_classes = self.num_classes;
        if num_classes < 2 {
            return 0.0;
        }
        let target = NclEquiangularTightFrame::theoretical_inner_product(num_classes);
        let norms: Vec<f64> = self.prototypes.iter().map(|p| l2_norm(p)).collect();

        let mut total_dev = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_classes {
            for j in (i + 1)..num_classes {
                let ni = norms[i];
                let nj = norms[j];
                if ni < 1e-12 || nj < 1e-12 {
                    continue;
                }
                let ip = dot(&self.prototypes[i], &self.prototypes[j]) / (ni * nj);
                total_dev += (ip - target).abs();
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        total_dev / count as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 NclLayerAnalysis
// ─────────────────────────────────────────────────────────────────────────────

/// Analyze Neural Collapse properties across multiple layers.
///
/// Tracks NC1/NC2 progression from early to final layers to understand
/// when neural collapse begins and how it develops.
pub struct NclLayerAnalysis {
    /// Per-layer feature statistics (stored in order of addition).
    pub layer_stats: Vec<NclFeatureStats>,
    /// Human-readable names for each layer.
    pub layer_names: Vec<String>,
    /// NC1 (within-class variability) per layer.
    pub nc1_per_layer: Vec<f64>,
    /// NC2 (ETF convergence) per layer.
    pub nc2_per_layer: Vec<f64>,
}

impl NclLayerAnalysis {
    /// Create an empty analysis.
    pub fn new() -> Self {
        NclLayerAnalysis {
            layer_stats: Vec::new(),
            layer_names: Vec::new(),
            nc1_per_layer: Vec::new(),
            nc2_per_layer: Vec::new(),
        }
    }

    /// Add statistics for a layer (appended in order).
    pub fn add_layer_stats(&mut self, stats: NclFeatureStats, name: &str) {
        self.layer_stats.push(stats);
        self.layer_names.push(name.to_string());
    }

    /// (Re-)compute NC1 and NC2 for each stored layer.
    pub fn compute_nc_progression(&mut self, _k: usize) {
        self.nc1_per_layer.clear();
        self.nc2_per_layer.clear();
        for stats in &self.layer_stats {
            let nc1 = NclNeuralCollapseMetrics::nc1_within_class_variability(stats);
            let nc2 = NclNeuralCollapseMetrics::nc2_convergence_to_etf(stats);
            self.nc1_per_layer.push(nc1);
            self.nc2_per_layer.push(nc2);
        }
    }

    /// Find the first layer index where NC1 < 0.1 (significant collapse onset).
    pub fn find_collapse_onset(&self) -> Option<usize> {
        self.nc1_per_layer.iter().position(|&nc1| nc1 < 0.1)
    }

    /// Layer-to-layer NC1 improvement rate (positive = improvement).
    pub fn nc_improvement_rate(&self) -> Vec<f64> {
        if self.nc1_per_layer.len() < 2 {
            return vec![];
        }
        self.nc1_per_layer
            .windows(2)
            .map(|w| w[0] - w[1]) // positive = decreasing NC1 = improvement
            .collect()
    }

    /// Generate a summary report string.
    pub fn report(&self) -> String {
        let mut s = String::from("=== Neural Collapse Layer Analysis ===\n");
        let n = self.layer_names.len();
        for i in 0..n {
            let nc1 = self.nc1_per_layer.get(i).copied().unwrap_or(f64::NAN);
            let nc2 = self.nc2_per_layer.get(i).copied().unwrap_or(f64::NAN);
            s.push_str(&format!(
                "  Layer {:>2} ({:<20}): NC1 = {:.4}, NC2 = {:.4}\n",
                i, self.layer_names[i], nc1, nc2
            ));
        }
        match self.find_collapse_onset() {
            Some(idx) => s.push_str(&format!(
                "  Collapse onset at layer {} ({})\n",
                idx, self.layer_names[idx]
            )),
            None => s.push_str("  No collapse onset detected (NC1 never < 0.1)\n"),
        }
        s
    }
}

impl Default for NclLayerAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8 NclFewShotNc
// ─────────────────────────────────────────────────────────────────────────────

/// Few-shot learning with Neural Collapse geometry.
///
/// Uses support set class means as prototypes. Optionally interpolates with
/// ETF prototypes to leverage the NC2 geometric prior.
pub struct NclFewShotNc {
    /// Class means from the support set: [num_classes × d].
    pub support_means: Vec<Vec<f64>>,
    /// Feature dimension.
    pub d: usize,
}

impl NclFewShotNc {
    /// Create an empty few-shot NC classifier.
    pub fn new(d: usize) -> Self {
        NclFewShotNc {
            support_means: Vec::new(),
            d,
        }
    }

    /// Compute class means from the labeled support set.
    pub fn fit_support(
        &mut self,
        support_features: &[Vec<f64>],
        support_labels: &[usize],
        num_classes: usize,
    ) -> Result<(), NclError> {
        if support_features.is_empty() {
            return Err(NclError::InvalidDimension("Empty support set".to_string()));
        }
        if support_features.len() != support_labels.len() {
            return Err(NclError::InvalidDimension(
                "support_features and support_labels length mismatch".to_string(),
            ));
        }
        let d = support_features[0].len();
        if d != self.d {
            return Err(NclError::InvalidDimension(format!(
                "Support feature dim {} != classifier dim {}",
                d, self.d
            )));
        }

        let mut sums = vec![vec![0.0f64; d]; num_classes];
        let mut counts = vec![0usize; num_classes];
        for (feat, &lbl) in support_features.iter().zip(support_labels.iter()) {
            if lbl >= num_classes {
                return Err(NclError::InvalidDimension(format!(
                    "Label {} >= num_classes = {}",
                    lbl, num_classes
                )));
            }
            for j in 0..d {
                sums[lbl][j] += feat[j];
            }
            counts[lbl] += 1;
        }
        self.support_means = (0..num_classes)
            .map(|k| {
                if counts[k] > 0 {
                    sums[k].iter().map(|s| s / counts[k] as f64).collect()
                } else {
                    vec![0.0f64; d]
                }
            })
            .collect();
        Ok(())
    }

    /// Classify a query feature by nearest support mean (L2 distance).
    pub fn classify(&self, query_feature: &[f64]) -> Result<usize, NclError> {
        if self.support_means.is_empty() {
            return Err(NclError::InvalidDimension(
                "No support means — call fit_support first".to_string(),
            ));
        }
        let mut best_k = 0usize;
        let mut best_dist = f64::INFINITY;
        for (k, mean) in self.support_means.iter().enumerate() {
            let dist: f64 = query_feature
                .iter()
                .zip(mean.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }
        Ok(best_k)
    }

    /// Classify using ETF-interpolated prototypes:
    /// `proto_k = (1-alpha) * mu_k + alpha * h_k`
    ///
    /// When alpha=0: pure support means. When alpha=1: pure ETF.
    pub fn classify_with_nc_prior(
        &self,
        query_feature: &[f64],
        etf: &NclEquiangularTightFrame,
        alpha: f64,
    ) -> Result<usize, NclError> {
        if self.support_means.is_empty() {
            return Err(NclError::InvalidDimension(
                "No support means — call fit_support first".to_string(),
            ));
        }
        let num_classes = self.support_means.len().min(etf.num_classes);
        let d = self.d;
        let alpha = alpha.clamp(0.0, 1.0);

        let mut best_k = 0usize;
        let mut best_dist = f64::INFINITY;
        for k in 0..num_classes {
            let proto: Vec<f64> = (0..d)
                .map(|j| {
                    let mu_j = self.support_means[k].get(j).copied().unwrap_or(0.0);
                    let h_j = etf.prototypes[k].get(j).copied().unwrap_or(0.0);
                    (1.0 - alpha) * mu_j + alpha * h_j
                })
                .collect();
            let dist: f64 = query_feature
                .iter()
                .zip(proto.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }
        Ok(best_k)
    }

    /// Compute nearest-mean accuracy on a query set.
    pub fn k_shot_accuracy(&self, query_features: &[Vec<f64>], query_labels: &[usize]) -> f64 {
        if query_features.is_empty() {
            return 0.0;
        }
        let n = query_features.len().min(query_labels.len());
        let mut correct = 0usize;
        for i in 0..n {
            if let Ok(pred) = self.classify(&query_features[i]) {
                if pred == query_labels[i] {
                    correct += 1;
                }
            }
        }
        correct as f64 / n as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9 NclFisherRao
// ─────────────────────────────────────────────────────────────────────────────

/// Fisher-Rao geometry meets Neural Collapse.
///
/// Uses Fisher information to measure class separability under
/// Gaussian assumptions, complementing the standard Euclidean/cosine metrics.
pub struct NclFisherRao;

impl NclFisherRao {
    /// Fisher-Rao distance between two diagonal Gaussians N(μ1, Σ1) and N(μ2, Σ2)
    /// (diagonal covariances σ1^2 and σ2^2 component-wise).
    ///
    /// Approximation:
    /// `d_FR ≈ 2 * sum_i |log(sigma1_i/sigma2_i)| + ||(mu1 - mu2) / sqrt(sigma1 * sigma2)||_1`
    pub fn fisher_rao_distance(mu1: &[f64], mu2: &[f64], sigma1: &[f64], sigma2: &[f64]) -> f64 {
        let d = mu1.len().min(mu2.len()).min(sigma1.len()).min(sigma2.len());
        let mut dist = 0.0f64;
        for i in 0..d {
            let s1 = sigma1[i].abs().max(1e-12);
            let s2 = sigma2[i].abs().max(1e-12);
            let log_term = (s1 / s2).ln().abs();
            let mean_term = ((mu1[i] - mu2[i]) / (s1 * s2).sqrt()).abs();
            dist += 2.0 * log_term + mean_term;
        }
        dist
    }

    /// Compute mean pairwise Fisher-Rao distance between class distributions.
    ///
    /// Uses per-class means as μ and sqrt(diagonal of S_W) as σ (std deviation).
    pub fn class_separability(stats: &NclFeatureStats) -> f64 {
        let num_classes = stats.num_classes;
        if num_classes < 2 {
            return 0.0;
        }
        let d = stats.d;

        // Use diagonal of within-class covariance as variance per dimension
        let sigma: Vec<f64> = (0..d)
            .map(|j| stats.within_class_cov[j][j].max(0.0).sqrt().max(1e-8))
            .collect();

        let mut total = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_classes {
            for j in (i + 1)..num_classes {
                let fr_dist = Self::fisher_rao_distance(
                    &stats.class_means[i],
                    &stats.class_means[j],
                    &sigma,
                    &sigma,
                );
                total += fr_dist;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        total / count as f64
    }

    /// NC Fisher index: mean pairwise Fisher-Rao distance / (within-class variation).
    ///
    /// Higher = better class separability relative to within-class spread.
    pub fn nc_fisher_index(stats: &NclFeatureStats) -> f64 {
        let sep = Self::class_separability(stats);
        let within_var: f64 = (0..stats.d)
            .map(|j| stats.within_class_cov[j][j].max(0.0))
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        sep / within_var
    }

    /// Wasserstein-2 distance for identity-covariance Gaussians: ||mu1 - mu2||_2.
    pub fn optimal_transport_distance(mu1: &[f64], mu2: &[f64]) -> f64 {
        mu1.iter()
            .zip(mu2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10 NclMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of Neural Collapse measurement utilities.
pub struct NclMetrics;

impl NclMetrics {
    /// Compute accuracy using a nearest-class-mean (NCM) classifier.
    ///
    /// For each feature, predict the nearest class mean and compare to true label.
    pub fn accuracy_from_nearest_class_mean(
        stats: &NclFeatureStats,
        features: &[Vec<f64>],
        labels: &[usize],
    ) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        let n = features.len().min(labels.len());
        let mut correct = 0usize;
        for i in 0..n {
            let mut best_k = 0usize;
            let mut best_dist = f64::INFINITY;
            for k in 0..stats.num_classes {
                let dist: f64 = features[i]
                    .iter()
                    .zip(stats.class_means[k].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = k;
                }
            }
            if best_k == labels[i] {
                correct += 1;
            }
        }
        correct as f64 / n as f64
    }

    /// Return a human-readable description of the NC1 collapse state.
    pub fn variability_collapse_ratio(nc1: f64) -> String {
        if nc1 < 0.01 {
            "collapsed".to_string()
        } else if nc1 < 0.1 {
            "collapsing".to_string()
        } else if nc1 < 0.5 {
            "partial".to_string()
        } else {
            "not_collapsed".to_string()
        }
    }

    /// Mean absolute deviation from theoretical ETF inner products.
    ///
    /// Scans the Gram matrix and computes mean |G\[i\]\[j\] - target| for i≠j
    /// and |G\[i\]\[i\] - 1| for diagonal.
    pub fn etf_angle_deviation(gram: &[Vec<f64>], num_classes: usize) -> f64 {
        if num_classes < 2 || gram.is_empty() {
            return 0.0;
        }
        let target_offdiag = -1.0 / (num_classes as f64 - 1.0);
        let mut total = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_classes.min(gram.len()) {
            for j in 0..num_classes.min(gram[i].len()) {
                let target = if i == j { 1.0 } else { target_offdiag };
                total += (gram[i][j] - target).abs();
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        total / count as f64
    }

    /// Find the most confused class pairs by their class-mean L2 distance.
    ///
    /// Returns the `k` pairs (i, j, distance) with the *smallest* inter-class distances,
    /// sorted ascending (most confused first).
    pub fn top_k_class_confusion(stats: &NclFeatureStats, k: usize) -> Vec<(usize, usize, f64)> {
        let num_classes = stats.num_classes;
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..num_classes {
            for j in (i + 1)..num_classes {
                let dist = NclFisherRao::optimal_transport_distance(
                    &stats.class_means[i],
                    &stats.class_means[j],
                );
                pairs.push((i, j, dist));
            }
        }
        // Sort ascending by distance (smallest = most confused)
        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs
    }

    /// Detect when neural collapse starts in a training trajectory.
    ///
    /// Returns the first epoch index (0-based) where NC1 drops below 0.1.
    pub fn collapse_phase_detection(nc1_trajectory: &[f64]) -> Option<usize> {
        nc1_trajectory.iter().position(|&v| v < 0.1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for convenience
// ─────────────────────────────────────────────────────────────────────────────

// All public types are re-exported via `pub mod neural_collapse` in lib.rs.
// The _ re-exports below silence unused-import warnings for helper functions
// ncl_rand01 and ncl_randn are already pub fn at the top of this module.
