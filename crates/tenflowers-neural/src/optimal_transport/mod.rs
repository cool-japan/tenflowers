//! Optimal Transport algorithms for distribution comparison and alignment.
//!
//! This module provides a comprehensive suite of Optimal Transport (OT) methods
//! including Sinkhorn-Knopp regularized transport, Wasserstein distances, Earth
//! Mover's Distance, Wasserstein barycenters, partial OT, Gromov-Wasserstein,
//! and differentiable Sinkhorn losses for training.
//!
//! # Algorithms
//!
//! - **Sinkhorn-Knopp**: Log-domain iterative scaling for entropy-regularized OT.
//! - **Wasserstein 1D**: Exact O(n log n) computation via sorted CDFs.
//! - **Sliced Wasserstein**: Monte Carlo random projections to 1D Wasserstein.
//! - **Earth Mover's Distance**: Exact 1D CDF-difference formulation.
//! - **Wasserstein Barycenter**: Fixed-support barycenter via iterative Sinkhorn.
//! - **Sinkhorn Divergence**: Debiased symmetric OT loss for neural training.
//! - **Partial OT**: Unbalanced transport with KL marginal relaxation.
//! - **Gromov-Wasserstein**: Structure-preserving cross-domain transport.
//!
//! # Example
//!
//! ```rust,ignore
//! use tenflowers_neural::optimal_transport::{OtConfig, sinkhorn, cost_matrix_euclidean};
//!
//! let a = vec![0.5, 0.5];
//! let b = vec![0.5, 0.5];
//! let x = vec![vec![0.0f64], vec![1.0f64]];
//! let y = vec![vec![0.0f64], vec![2.0f64]];
//! let cost = cost_matrix_euclidean(&x, &y);
//! let cfg = OtConfig::default();
//! let result = sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");
//! println!("OT cost: {}", result.cost);
//! ```

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::error::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Optimal Transport solvers.
#[derive(Debug, Clone)]
pub struct OtConfig {
    /// Entropy regularization parameter ε > 0. Smaller values give sharper transport plans.
    pub epsilon: f64,
    /// Maximum number of Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance for marginal constraint violation.
    pub tolerance: f64,
    /// Whether to use the log-domain formulation for numerical stability.
    pub log_domain: bool,
}

impl Default for OtConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            max_iter: 1000,
            tolerance: 1e-9,
            log_domain: true,
        }
    }
}

/// Result returned by Sinkhorn and related solvers.
#[derive(Debug, Clone)]
pub struct OtResult {
    /// Primal transport cost ⟨C, T⟩.
    pub cost: f64,
    /// Optimal transport plan as a row-major n×m matrix (flattened as Vec of rows).
    pub transport_plan: Vec<Vec<f64>>,
    /// Log-domain dual potential u (source).
    pub u: Vec<f64>,
    /// Log-domain dual potential v (target).
    pub v: Vec<f64>,
    /// Number of Sinkhorn iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

/// A pair of distributions with support weights.
#[derive(Debug, Clone)]
pub struct DistributionPair {
    /// Source support points.
    pub source: Vec<f64>,
    /// Target support points.
    pub target: Vec<f64>,
    /// Source marginal weights (must sum to 1).
    pub weights_a: Vec<f64>,
    /// Target marginal weights (must sum to 1).
    pub weights_b: Vec<f64>,
}

/// Sinkhorn divergence decomposition: `OT(a,b) - 0.5·OT(a,a) - 0.5·OT(b,b)`.
///
/// This debiased quantity is positive semi-definite and equals 0 iff a = b.
#[derive(Debug, Clone)]
pub struct SinkhornDivergence {
    /// The symmetric divergence value.
    pub value: f64,
    /// Transport result for (a, b).
    pub transport_ab: OtResult,
    /// Transport result for (a, a).
    pub transport_aa: OtResult,
    /// Transport result for (b, b).
    pub transport_bb: OtResult,
}

/// Configuration for partial / unbalanced Optimal Transport.
#[derive(Debug, Clone)]
pub struct PartialOtConfig {
    /// Fraction of total mass to transport (0 < m ≤ 1).
    pub mass_ratio: f64,
    /// Regularization ε for entropy penalty.
    pub epsilon: f64,
    /// KL marginal penalty coefficient τ.
    pub tau: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for PartialOtConfig {
    fn default() -> Self {
        Self {
            mass_ratio: 1.0,
            epsilon: 0.05,
            tau: 1.0,
            max_iter: 1000,
            tolerance: 1e-9,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerics helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp: log Σ exp(v_i).
///
/// Uses the max-shift trick to avoid overflow/underflow.
pub fn log_sum_exp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = v.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Soft-minimum: `−ε · log Σ exp(−v_i / ε)`.
///
/// As ε → 0 this approaches the true minimum.
pub fn soft_min(v: &[f64], epsilon: f64) -> f64 {
    let scaled: Vec<f64> = v.iter().map(|&x| -x / epsilon).collect();
    -epsilon * log_sum_exp(&scaled)
}

/// Project weights onto the probability simplex (normalize to sum 1).
///
/// Returns a uniform distribution if the sum is zero or negative.
pub fn normalize_weights(w: &[f64]) -> Vec<f64> {
    let sum: f64 = w.iter().sum();
    if sum <= 0.0 {
        let n = w.len();
        return vec![1.0 / n as f64; n];
    }
    w.iter().map(|&x| x / sum).collect()
}

/// Pairwise squared-Euclidean cost matrix: `C[i][j] = ‖x_i − y_j‖²`.
pub fn cost_matrix_squared(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|xi| {
            y.iter()
                .map(|yj| {
                    xi.iter()
                        .zip(yj.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Pairwise squared-Euclidean cost matrix (alias for [`cost_matrix_squared`]).
pub fn cost_matrix_euclidean(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    cost_matrix_squared(x, y)
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-domain Sinkhorn helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute log-sum-exp over rows of `M` weighted by log-weights `log_v`.
///
/// Returns a column vector of length `n` where the i-th entry is
/// `log Σ_j exp(M[i][j] + log_v[j])`.
pub(crate) fn log_dot_rows(m: &[Vec<f64>], log_v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| {
            let vals: Vec<f64> = row.iter().zip(log_v.iter()).map(|(&a, &b)| a + b).collect();
            log_sum_exp(&vals)
        })
        .collect()
}

/// Compute log-sum-exp over columns of `M` weighted by log-weights `log_u`.
///
/// Returns a row vector of length `m` where the j-th entry is
/// `log Σ_i exp(M[i][j] + log_u[i])`.
pub(crate) fn log_dot_cols(m: &[Vec<f64>], log_u: &[f64]) -> Vec<f64> {
    if m.is_empty() {
        return Vec::new();
    }
    let ncols = m[0].len();
    (0..ncols)
        .map(|j| {
            let vals: Vec<f64> = m
                .iter()
                .zip(log_u.iter())
                .map(|(row, &lu)| row[j] + lu)
                .collect();
            log_sum_exp(&vals)
        })
        .collect()
}

/// Build the log-kernel matrix: `K[i][j] = −C[i][j] / ε`.
pub(crate) fn build_log_kernel(cost: &[Vec<f64>], epsilon: f64) -> Vec<Vec<f64>> {
    cost.iter()
        .map(|row| row.iter().map(|&c| -c / epsilon).collect())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sinkhorn-Knopp
// ─────────────────────────────────────────────────────────────────────────────

/// Solve entropy-regularized Optimal Transport via log-domain Sinkhorn-Knopp.
///
/// Given source weights `a` (n), target weights `b` (m), and cost matrix `cost` (n×m),
/// the solver minimises `⟨C, T⟩ − ε · H(T)` subject to `T·1 = a`, `T^⊤·1 = b`.
///
/// # Errors
///
/// Returns [`TensorError::InvalidArgument`] when inputs are empty, sizes mismatch,
/// or `epsilon ≤ 0`.
pub fn sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[Vec<f64>],
    config: &OtConfig,
) -> Result<OtResult, TensorError> {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return Err(TensorError::invalid_argument_op(
            "sinkhorn",
            "source and target weights must be non-empty",
        ));
    }
    if cost.len() != n {
        return Err(TensorError::invalid_argument_op(
            "sinkhorn",
            "cost matrix row count must equal len(a)",
        ));
    }
    for (i, row) in cost.iter().enumerate() {
        if row.len() != m {
            return Err(TensorError::invalid_argument_op(
                "sinkhorn",
                &format!(
                    "cost matrix row {i} has {} columns, expected {m}",
                    row.len()
                ),
            ));
        }
    }
    if config.epsilon <= 0.0 {
        return Err(TensorError::invalid_argument_op(
            "sinkhorn",
            "epsilon must be positive",
        ));
    }

    let log_a: Vec<f64> = a.iter().map(|&x| x.max(1e-300).ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|&x| x.max(1e-300).ln()).collect();
    let log_k = build_log_kernel(cost, config.epsilon);

    // Dual potentials in log-domain: f = ε·u_log, g = ε·v_log
    let mut u_log = vec![0.0_f64; n]; // log of scaling factor u
    let mut v_log = vec![0.0_f64; m]; // log of scaling factor v

    let mut converged = false;
    let mut iters = 0usize;

    for iter in 0..config.max_iter {
        iters = iter + 1;
        // u ← a / (K v)  in log-domain: u_log ← log_a − log(K exp(v_log))
        let kv_log = log_dot_rows(&log_k, &v_log);
        let u_log_new: Vec<f64> = log_a
            .iter()
            .zip(kv_log.iter())
            .map(|(&la, &kv)| la - kv)
            .collect();

        // v ← b / (K^⊤ u)
        let ktu_log = log_dot_cols(&log_k, &u_log_new);
        let v_log_new: Vec<f64> = log_b
            .iter()
            .zip(ktu_log.iter())
            .map(|(&lb, &ku)| lb - ku)
            .collect();

        // Check convergence via change in v
        let delta: f64 = v_log_new
            .iter()
            .zip(v_log.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        u_log = u_log_new;
        v_log = v_log_new;

        if delta < config.tolerance {
            converged = true;
            break;
        }
    }

    // Reconstruct transport plan T[i][j] = exp(u_log[i] + K[i][j] + v_log[j])
    let mut transport_plan = vec![vec![0.0_f64; m]; n];
    let mut primal_cost = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            let log_t = u_log[i] + log_k[i][j] + v_log[j];
            let t_ij = log_t.exp();
            transport_plan[i][j] = t_ij;
            primal_cost += t_ij * cost[i][j];
        }
    }

    Ok(OtResult {
        cost: primal_cost,
        transport_plan,
        u: u_log.iter().map(|&x| x * config.epsilon).collect(),
        v: v_log.iter().map(|&x| x * config.epsilon).collect(),
        iterations: iters,
        converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D Wasserstein distance (exact)
// ─────────────────────────────────────────────────────────────────────────────

/// Exact 1D Wasserstein-1 distance between two weighted distributions via sorted CDFs.
///
/// Both `positions_a` / `positions_b` are support locations with corresponding
/// weights `weights_a` / `weights_b`.  All weights must be non-negative and sum to 1.
/// Complexity: O(n log n + m log m).
///
/// # Errors
///
/// Returns an error if lengths mismatch or weights are empty.
pub fn wasserstein_1d(
    positions_a: &[f64],
    weights_a: &[f64],
    positions_b: &[f64],
    weights_b: &[f64],
) -> Result<f64, TensorError> {
    if positions_a.len() != weights_a.len() {
        return Err(TensorError::invalid_argument_op(
            "wasserstein_1d",
            "positions_a and weights_a must have the same length",
        ));
    }
    if positions_b.len() != weights_b.len() {
        return Err(TensorError::invalid_argument_op(
            "wasserstein_1d",
            "positions_b and weights_b must have the same length",
        ));
    }
    if positions_a.is_empty() || positions_b.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "wasserstein_1d",
            "distributions must be non-empty",
        ));
    }

    // Sort both distributions by position
    let mut pairs_a: Vec<(f64, f64)> = positions_a
        .iter()
        .zip(weights_a.iter())
        .map(|(&p, &w)| (p, w))
        .collect();
    let mut pairs_b: Vec<(f64, f64)> = positions_b
        .iter()
        .zip(weights_b.iter())
        .map(|(&p, &w)| (p, w))
        .collect();

    pairs_a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs_b.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let sum_a: f64 = weights_a.iter().sum();
    let sum_b: f64 = weights_b.iter().sum();

    // Build merged sorted event list
    let mut events: Vec<(f64, f64, f64)> = Vec::with_capacity(pairs_a.len() + pairs_b.len());
    for (p, w) in &pairs_a {
        events.push((*p, w / sum_a, 0.0));
    }
    for (p, w) in &pairs_b {
        events.push((*p, 0.0, w / sum_b));
    }
    events.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Sweep through events computing integral of |CDF_a − CDF_b|
    let mut cdf_a = 0.0_f64;
    let mut cdf_b = 0.0_f64;
    let mut prev_pos = events[0].0;
    let mut distance = 0.0_f64;

    for &(pos, wa, wb) in &events {
        let gap = pos - prev_pos;
        if gap > 0.0 {
            distance += gap * (cdf_a - cdf_b).abs();
        }
        cdf_a += wa;
        cdf_b += wb;
        prev_pos = pos;
    }

    Ok(distance)
}

// ─────────────────────────────────────────────────────────────────────────────
// Wasserstein distance via Sinkhorn
// ─────────────────────────────────────────────────────────────────────────────

/// Regularized Wasserstein distance between distributions `a` and `b` via Sinkhorn.
///
/// This is a convenience wrapper around [`sinkhorn`] returning only the transport cost.
///
/// # Errors
///
/// Propagates errors from [`sinkhorn`].
pub fn wasserstein_distance(
    a: &[f64],
    b: &[f64],
    cost_matrix: &[Vec<f64>],
    epsilon: f64,
) -> Result<f64, TensorError> {
    let cfg = OtConfig {
        epsilon,
        max_iter: 1000,
        tolerance: 1e-9,
        log_domain: true,
    };
    let result = sinkhorn(a, b, cost_matrix, &cfg)?;
    Ok(result.cost)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sliced Wasserstein distance
// ─────────────────────────────────────────────────────────────────────────────

/// Sliced Wasserstein Distance (SWD) via random 1D projections.
///
/// Projects samples `a_samples` (shape n×d) and `b_samples` (shape m×d) onto
/// `n_projections` random unit vectors and averages the 1D Wasserstein distances.
/// Uses `scirs2_core` seeded randomness.
///
/// # Errors
///
/// Returns an error if samples are empty or have inconsistent dimensionality.
pub fn sliced_wasserstein(
    a_samples: &[Vec<f64>],
    b_samples: &[Vec<f64>],
    n_projections: usize,
    seed: u64,
) -> Result<f64, TensorError> {
    if a_samples.is_empty() || b_samples.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "sliced_wasserstein",
            "sample sets must be non-empty",
        ));
    }
    let d = a_samples[0].len();
    if d == 0 {
        return Err(TensorError::invalid_argument_op(
            "sliced_wasserstein",
            "sample dimension must be at least 1",
        ));
    }
    for s in a_samples.iter().chain(b_samples.iter()) {
        if s.len() != d {
            return Err(TensorError::invalid_argument_op(
                "sliced_wasserstein",
                "all samples must have the same dimensionality",
            ));
        }
    }
    if n_projections == 0 {
        return Err(TensorError::invalid_argument_op(
            "sliced_wasserstein",
            "n_projections must be at least 1",
        ));
    }

    let na = a_samples.len();
    let nb = b_samples.len();

    let wa: Vec<f64> = vec![1.0 / na as f64; na];
    let wb: Vec<f64> = vec![1.0 / nb as f64; nb];

    let mut rng = StdRng::seed_from_u64(seed);
    let mut total = 0.0_f64;

    for _ in 0..n_projections {
        // Sample a random direction from the unit sphere using Box-Muller
        let mut direction: Vec<f64> = (0..d)
            .map(|_| {
                // Sample standard normal via Box-Muller
                let u1: f64 = rng.random::<f64>().max(1e-300);
                let u2: f64 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();

        // Normalize
        let norm: f64 = direction.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            continue;
        }
        for v in direction.iter_mut() {
            *v /= norm;
        }

        // Project samples onto direction
        let proj_a: Vec<f64> = a_samples
            .iter()
            .map(|s| s.iter().zip(direction.iter()).map(|(&x, &d)| x * d).sum())
            .collect();
        let proj_b: Vec<f64> = b_samples
            .iter()
            .map(|s| s.iter().zip(direction.iter()).map(|(&x, &d)| x * d).sum())
            .collect();

        let w1 = wasserstein_1d(&proj_a, &wa, &proj_b, &wb)?;
        total += w1;
    }

    Ok(total / n_projections as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Earth Mover's Distance (1D exact)
// ─────────────────────────────────────────────────────────────────────────────

/// Exact 1D Earth Mover's Distance via the CDF-difference integral.
///
/// For discrete distributions this equals the L1 distance between their CDFs.
/// Weights are automatically normalized to the probability simplex.
///
/// # Errors
///
/// Returns an error if inputs are empty or lengths mismatch.
pub fn emd_1d(
    positions_a: &[f64],
    weights_a: &[f64],
    positions_b: &[f64],
    weights_b: &[f64],
) -> Result<f64, TensorError> {
    // EMD in 1D equals Wasserstein-1
    wasserstein_1d(positions_a, weights_a, positions_b, weights_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Wasserstein Barycenter
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-support Wasserstein barycenter computed via iterative Sinkhorn updates.
///
/// Given `K` distributions (each a weight vector over a shared `n`-point support)
/// and a mixture weight vector, returns the barycenter weights.
pub struct WassersteinBarycenter {
    /// Regularization for internal Sinkhorn calls.
    pub epsilon: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on barycenter weight change.
    pub tolerance: f64,
}

impl Default for WassersteinBarycenter {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            max_iter: 500,
            tolerance: 1e-7,
        }
    }
}

impl WassersteinBarycenter {
    /// Create a new barycenter solver with the given configuration.
    pub fn new(epsilon: f64, max_iter: usize, tolerance: f64) -> Self {
        Self {
            epsilon,
            max_iter,
            tolerance,
        }
    }

    /// Compute the barycenter weights over the given `support` positions.
    ///
    /// `distributions` — list of K source weight vectors, each of length n.
    /// `mixture_weights` — K non-negative mixture coefficients (automatically normalized).
    /// `support` — n support positions (1D) as `Vec<f64>`.
    ///
    /// Returns the n barycenter weights summing to 1.
    ///
    /// # Errors
    ///
    /// Returns an error on empty inputs or size mismatches.
    pub fn compute(
        &self,
        distributions: &[Vec<f64>],
        mixture_weights: &[f64],
        support: &[f64],
    ) -> Result<Vec<f64>, TensorError> {
        let k = distributions.len();
        if k == 0 {
            return Err(TensorError::invalid_argument_op(
                "WassersteinBarycenter::compute",
                "distributions list must be non-empty",
            ));
        }
        if mixture_weights.len() != k {
            return Err(TensorError::invalid_argument_op(
                "WassersteinBarycenter::compute",
                "mixture_weights length must match number of distributions",
            ));
        }
        let n = support.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "WassersteinBarycenter::compute",
                "support must be non-empty",
            ));
        }
        for (ki, dist) in distributions.iter().enumerate() {
            if dist.len() != n {
                return Err(TensorError::invalid_argument_op(
                    "WassersteinBarycenter::compute",
                    &format!("distribution {ki} has length {}, expected {n}", dist.len()),
                ));
            }
        }

        let lam = normalize_weights(mixture_weights);

        // Cost matrix C[i][j] = (support[i] - support[j])^2
        let cost: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let d = support[i] - support[j];
                        d * d
                    })
                    .collect()
            })
            .collect();

        let cfg = OtConfig {
            epsilon: self.epsilon,
            max_iter: 200,
            tolerance: 1e-9,
            log_domain: true,
        };

        // Initialise barycenter weights uniformly
        let mut bar: Vec<f64> = vec![1.0 / n as f64; n];

        for _outer in 0..self.max_iter {
            // For each source distribution, run Sinkhorn and collect the log-scaling u
            let mut log_prod: Vec<f64> = vec![0.0_f64; n];

            for ki in 0..k {
                let src = &distributions[ki];
                let result = sinkhorn(src, &bar, &cost, &cfg)?;

                // Column sums of transport plan = target marginals
                let col_sums: Vec<f64> = (0..n)
                    .map(|j| result.transport_plan.iter().map(|row| row[j]).sum::<f64>())
                    .collect();

                for j in 0..n {
                    let cs = col_sums[j].max(1e-300);
                    log_prod[j] += lam[ki] * cs.ln();
                }
            }

            // New bar = exp(log_prod), normalized
            let new_bar: Vec<f64> = log_prod.iter().map(|&x| x.exp()).collect();
            let new_bar = normalize_weights(&new_bar);

            // Convergence check
            let delta: f64 = new_bar
                .iter()
                .zip(bar.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            bar = new_bar;
            if delta < self.tolerance {
                break;
            }
        }

        Ok(bar)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Differentiable OT Losses
// ─────────────────────────────────────────────────────────────────────────────

/// Sinkhorn Divergence: debiased, symmetric, differentiable OT loss.
///
/// Computes `S_ε(a, b) = OT_ε(a,b) − ½·OT_ε(a,a) − ½·OT_ε(b,b)`.
/// This quantity is positive semi-definite and equals 0 iff a = b.
pub struct SinkhornLoss {
    /// Sinkhorn regularization ε.
    pub epsilon: f64,
    /// Maximum Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for SinkhornLoss {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            max_iter: 500,
            tolerance: 1e-9,
        }
    }
}

impl SinkhornLoss {
    /// Create a new Sinkhorn loss with the given ε.
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            ..Default::default()
        }
    }

    /// Compute the Sinkhorn divergence between prediction weights `pred` and target weights `target`.
    ///
    /// Both are weight vectors of the same length n (over a shared support).
    /// The squared-Euclidean cost on integer positions `{0, …, n-1}` is used when
    /// no cost matrix is supplied; for custom geometries use \[`compute_with_cost`\].
    ///
    /// # Errors
    ///
    /// Returns an error on empty or mismatched inputs.
    pub fn compute(
        &self,
        predictions: &[f64],
        targets: &[f64],
        epsilon: f64,
    ) -> Result<SinkhornDivergence, TensorError> {
        if predictions.len() != targets.len() {
            return Err(TensorError::invalid_argument_op(
                "SinkhornLoss::compute",
                "predictions and targets must have the same length",
            ));
        }
        let n = predictions.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "SinkhornLoss::compute",
                "inputs must be non-empty",
            ));
        }
        let eps = if epsilon > 0.0 { epsilon } else { self.epsilon };

        // Default cost: integer grid positions
        let support: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cost = cost_matrix_euclidean(&support, &support);

        self.compute_with_cost(predictions, targets, &cost, eps)
    }

    /// Compute Sinkhorn divergence with a custom cost matrix.
    ///
    /// # Errors
    ///
    /// Returns an error on empty, mismatched, or ill-formed inputs.
    pub fn compute_with_cost(
        &self,
        predictions: &[f64],
        targets: &[f64],
        cost: &[Vec<f64>],
        epsilon: f64,
    ) -> Result<SinkhornDivergence, TensorError> {
        if predictions.len() != targets.len() {
            return Err(TensorError::invalid_argument_op(
                "SinkhornLoss::compute_with_cost",
                "predictions and targets must have the same length",
            ));
        }
        let pred = normalize_weights(predictions);
        let tgt = normalize_weights(targets);
        let eps = if epsilon > 0.0 { epsilon } else { self.epsilon };

        let cfg = OtConfig {
            epsilon: eps,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            log_domain: true,
        };

        let transport_ab = sinkhorn(&pred, &tgt, cost, &cfg)?;
        let transport_aa = sinkhorn(&pred, &pred, cost, &cfg)?;
        let transport_bb = sinkhorn(&tgt, &tgt, cost, &cfg)?;

        let value = transport_ab.cost - 0.5 * transport_aa.cost - 0.5 * transport_bb.cost;
        let value = value.max(0.0); // clamp numerical noise

        Ok(SinkhornDivergence {
            value,
            transport_ab,
            transport_aa,
            transport_bb,
        })
    }
}

/// Standard Optimal Transport losses for training neural networks.
pub struct OtLoss;

impl OtLoss {
    /// Wasserstein loss: primal transport cost `⟨C, T⟩` from Sinkhorn.
    ///
    /// # Errors
    ///
    /// Returns an error on empty or mismatched inputs.
    pub fn wasserstein_loss(
        pred: &[f64],
        target: &[f64],
        cost: &[Vec<f64>],
        epsilon: f64,
    ) -> Result<f64, TensorError> {
        let p = normalize_weights(pred);
        let t = normalize_weights(target);
        wasserstein_distance(&p, &t, cost, epsilon)
    }

    /// Symmetric Sinkhorn divergence loss (wrapper for convenience).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`SinkhornLoss::compute`].
    pub fn sinkhorn_divergence(
        pred: &[f64],
        target: &[f64],
        epsilon: f64,
    ) -> Result<f64, TensorError> {
        let loss = SinkhornLoss::new(epsilon);
        let div = loss.compute(pred, target, epsilon)?;
        Ok(div.value)
    }
}
