//! Extension algorithms for Optimal Transport: Partial OT, Gromov-Wasserstein,
//! and Fused Gromov-Wasserstein.

use tenflowers_core::error::TensorError;

use super::{
    build_log_kernel, cost_matrix_euclidean, log_dot_cols, log_dot_rows, log_sum_exp,
    normalize_weights, sinkhorn, OtConfig, OtResult, PartialOtConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// Partial (unbalanced) Optimal Transport
// ─────────────────────────────────────────────────────────────────────────────

/// Solve partial (unbalanced) Optimal Transport via KL-regularized Sinkhorn.
///
/// The marginal constraints are relaxed by a KL-divergence penalty with weight `τ`,
/// allowing transport of only a fraction `m` of the total mass.
///
/// # Errors
///
/// Returns an error on empty or mismatched inputs or invalid configuration.
pub fn partial_sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[Vec<f64>],
    config: &PartialOtConfig,
) -> Result<OtResult, TensorError> {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return Err(TensorError::invalid_argument_op(
            "partial_sinkhorn",
            "source and target weights must be non-empty",
        ));
    }
    if cost.len() != n {
        return Err(TensorError::invalid_argument_op(
            "partial_sinkhorn",
            "cost matrix row count must equal len(a)",
        ));
    }
    for (i, row) in cost.iter().enumerate() {
        if row.len() != m {
            return Err(TensorError::invalid_argument_op(
                "partial_sinkhorn",
                &format!(
                    "cost matrix row {i} has {} columns, expected {m}",
                    row.len()
                ),
            ));
        }
    }
    if config.epsilon <= 0.0 {
        return Err(TensorError::invalid_argument_op(
            "partial_sinkhorn",
            "epsilon must be positive",
        ));
    }
    if !(0.0 < config.mass_ratio && config.mass_ratio <= 1.0) {
        return Err(TensorError::invalid_argument_op(
            "partial_sinkhorn",
            "mass_ratio must be in (0, 1]",
        ));
    }

    let eps = config.epsilon;
    let tau = config.tau;
    let rho = tau * eps / (tau + eps); // effective marginal regularization

    let log_k = build_log_kernel(cost, eps);
    let log_a: Vec<f64> = a.iter().map(|&x| x.max(1e-300).ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|&x| x.max(1e-300).ln()).collect();

    let mut u_log = vec![0.0_f64; n];
    let mut v_log = vec![0.0_f64; m];

    let mut converged = false;
    let mut iters = 0usize;

    // KL-proximal Sinkhorn iterations
    for iter in 0..config.max_iter {
        iters = iter + 1;

        // u update: proximal KL step
        let kv_log = log_dot_rows(&log_k, &v_log);
        let u_log_new: Vec<f64> = log_a
            .iter()
            .zip(kv_log.iter())
            .map(|(&la, &kv)| rho / eps * la - rho / eps * kv)
            .collect();

        // v update
        let ktu_log = log_dot_cols(&log_k, &u_log_new);
        let v_log_new: Vec<f64> = log_b
            .iter()
            .zip(ktu_log.iter())
            .map(|(&lb, &ku)| rho / eps * lb - rho / eps * ku)
            .collect();

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

    // Reconstruct transport plan
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
        u: u_log.iter().map(|&x| x * eps).collect(),
        v: v_log.iter().map(|&x| x * eps).collect(),
        iterations: iters,
        converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Gromov-Wasserstein
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Gromov-Wasserstein (GW) distance between two metric measure spaces.
///
/// `c_s` (n×n) — pairwise distances within source space.
/// `c_t` (m×m) — pairwise distances within target space.
/// `a` (n) — source weights.
/// `b` (m) — target weights.
/// `epsilon` — entropy regularization.
/// `max_iter` — outer alternating iterations.
///
/// Returns the GW cost (a scalar) and the optimal coupling matrix.
///
/// # Errors
///
/// Returns an error on shape mismatches or invalid epsilon.
pub fn gromov_wasserstein_distance(
    c_s: &[Vec<f64>],
    c_t: &[Vec<f64>],
    a: &[f64],
    b: &[f64],
    epsilon: f64,
    max_iter: usize,
) -> Result<(f64, Vec<Vec<f64>>), TensorError> {
    let n = a.len();
    let m = b.len();

    if c_s.len() != n || c_s.iter().any(|r| r.len() != n) {
        return Err(TensorError::invalid_argument_op(
            "gromov_wasserstein_distance",
            "c_s must be n×n where n = len(a)",
        ));
    }
    if c_t.len() != m || c_t.iter().any(|r| r.len() != m) {
        return Err(TensorError::invalid_argument_op(
            "gromov_wasserstein_distance",
            "c_t must be m×m where m = len(b)",
        ));
    }
    if epsilon <= 0.0 {
        return Err(TensorError::invalid_argument_op(
            "gromov_wasserstein_distance",
            "epsilon must be positive",
        ));
    }
    if n == 0 || m == 0 {
        return Err(TensorError::invalid_argument_op(
            "gromov_wasserstein_distance",
            "distributions must be non-empty",
        ));
    }

    let cfg = OtConfig {
        epsilon,
        max_iter: 200,
        tolerance: 1e-9,
        log_domain: true,
    };

    // Initialise transport plan as outer product of marginals
    let mut t: Vec<Vec<f64>> = a
        .iter()
        .map(|&ai| b.iter().map(|&bj| ai * bj).collect())
        .collect();

    for _outer in 0..max_iter {
        // Compute linearization of GW loss: M[i][j] = Σ_{k,l} C_s[i,k] T[k,l] C_t[l,j]
        // = (C_s T C_t)[i,j]
        // Computed as two matrix products (plain loops — no ndarray)
        let tc_t: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|l| {
                        // (T C_t)[i,l] = Σ_j T[i,j] C_t[j,l]  (n×m times m×m → n×m)
                        t[i].iter()
                            .zip(c_t.iter())
                            .map(|(&tij, ct_row)| tij * ct_row[l])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        let m_mat: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| {
                        // M[i,j] = Σ_k C_s[i,k] * (TC_t)[k,j]  (n×n times n×m → n×m)
                        c_s[i]
                            .iter()
                            .zip(tc_t.iter())
                            .map(|(&cs_ik, tc_row)| cs_ik * tc_row[j])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        // Solve Sinkhorn with linearized cost
        let result = sinkhorn(a, b, &m_mat, &cfg)?;

        // Convergence: check change in T
        let delta: f64 = result
            .transport_plan
            .iter()
            .zip(t.iter())
            .flat_map(|(new_row, old_row)| {
                new_row
                    .iter()
                    .zip(old_row.iter())
                    .map(|(&a, &b)| (a - b).abs())
            })
            .fold(0.0_f64, f64::max);

        t = result.transport_plan;

        if delta < 1e-7 {
            break;
        }
    }

    // Compute final GW cost
    let gw_cost = compute_gw_cost(c_s, c_t, &t);

    Ok((gw_cost, t))
}

/// Compute the Gromov-Wasserstein loss: `Σ_{i,j,k,l} (C_s[i,k] - C_t[j,l])^2 T[i,j] T[k,l]`.
pub(crate) fn compute_gw_cost(c_s: &[Vec<f64>], c_t: &[Vec<f64>], t: &[Vec<f64>]) -> f64 {
    let n = c_s.len();
    let m = c_t.len();
    let mut cost = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            for k in 0..n {
                for l in 0..m {
                    let diff = c_s[i][k] - c_t[j][l];
                    cost += diff * diff * t[i][j] * t[k][l];
                }
            }
        }
    }
    cost
}

/// Fused Gromov-Wasserstein distance combining structural and feature costs.
///
/// Minimises `α · GW(C_s, C_t, T) + (1−α) · ⟨F, T⟩` where:
/// - `c_s` (n×n), `c_t` (m×m) are intra-domain distance matrices,
/// - `f_s` (n×d), `f_t` (m×d) are node feature matrices,
/// - `F[i,j] = ‖f_s[i] − f_t[j]‖²` is the feature cost,
/// - `alpha` ∈ \[0,1\] balances structural vs feature cost.
///
/// # Errors
///
/// Returns an error on shape mismatches or out-of-range alpha.
pub fn fused_gromov_wasserstein(
    c_s: &[Vec<f64>],
    c_t: &[Vec<f64>],
    f_s: &[Vec<f64>],
    f_t: &[Vec<f64>],
    a: &[f64],
    b: &[f64],
    alpha: f64,
    epsilon: f64,
    max_iter: usize,
) -> Result<(f64, Vec<Vec<f64>>), TensorError> {
    let n = a.len();
    let m = b.len();

    if !(0.0..=1.0).contains(&alpha) {
        return Err(TensorError::invalid_argument_op(
            "fused_gromov_wasserstein",
            "alpha must be in [0, 1]",
        ));
    }
    if c_s.len() != n || c_s.iter().any(|r| r.len() != n) {
        return Err(TensorError::invalid_argument_op(
            "fused_gromov_wasserstein",
            "c_s must be n×n where n = len(a)",
        ));
    }
    if c_t.len() != m || c_t.iter().any(|r| r.len() != m) {
        return Err(TensorError::invalid_argument_op(
            "fused_gromov_wasserstein",
            "c_t must be m×m where m = len(b)",
        ));
    }
    if f_s.len() != n {
        return Err(TensorError::invalid_argument_op(
            "fused_gromov_wasserstein",
            "f_s row count must equal len(a)",
        ));
    }
    if f_t.len() != m {
        return Err(TensorError::invalid_argument_op(
            "fused_gromov_wasserstein",
            "f_t row count must equal len(b)",
        ));
    }
    if epsilon <= 0.0 {
        return Err(TensorError::invalid_argument_op(
            "fused_gromov_wasserstein",
            "epsilon must be positive",
        ));
    }

    // Feature cost matrix
    let feature_cost = cost_matrix_euclidean(f_s, f_t);

    let cfg = OtConfig {
        epsilon,
        max_iter: 200,
        tolerance: 1e-9,
        log_domain: true,
    };

    // Initialise transport plan
    let mut t: Vec<Vec<f64>> = a
        .iter()
        .map(|&ai| b.iter().map(|&bj| ai * bj).collect())
        .collect();

    for _outer in 0..max_iter {
        // GW linearized cost: M_gw[i,j] = Σ_{k,l} C_s[i,k] T[k,l] C_t[l,j]
        let tc_t: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|l| {
                        t[i].iter()
                            .zip(c_t.iter())
                            .map(|(&tij, ct_row)| tij * ct_row[l])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        let m_gw: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| {
                        c_s[i]
                            .iter()
                            .zip(tc_t.iter())
                            .map(|(&cs_ik, tc_row)| cs_ik * tc_row[j])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        // Fused cost: (1-alpha)*F + alpha*M_gw
        let fused_cost: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| (1.0 - alpha) * feature_cost[i][j] + alpha * m_gw[i][j])
                    .collect()
            })
            .collect();

        let result = sinkhorn(a, b, &fused_cost, &cfg)?;

        let delta: f64 = result
            .transport_plan
            .iter()
            .zip(t.iter())
            .flat_map(|(new_row, old_row)| {
                new_row
                    .iter()
                    .zip(old_row.iter())
                    .map(|(&a, &b)| (a - b).abs())
            })
            .fold(0.0_f64, f64::max);

        t = result.transport_plan;

        if delta < 1e-7 {
            break;
        }
    }

    let gw_cost = compute_gw_cost(c_s, c_t, &t);
    let feat_cost: f64 = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| feature_cost[i][j] * t[i][j])
        .sum();

    let total = alpha * gw_cost + (1.0 - alpha) * feat_cost;

    Ok((total, t))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimal_transport::{
        cost_matrix_euclidean, emd_1d, log_sum_exp, normalize_weights, sliced_wasserstein,
        sinkhorn, wasserstein_1d, wasserstein_distance, OtConfig, OtLoss, PartialOtConfig,
        SinkhornLoss, WassersteinBarycenter,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── log_sum_exp ───────────────────────────────────────────────────────────

    #[test]
    fn test_log_sum_exp_basic() {
        let v = vec![0.0, 0.0, 0.0];
        let result = log_sum_exp(&v);
        assert!(approx_eq(result, (3.0_f64).ln(), 1e-10));
    }

    #[test]
    fn test_log_sum_exp_large_values() {
        // Should not overflow
        let v = vec![1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&v);
        assert!(result.is_finite());
        assert!(result > 1002.0);
    }

    #[test]
    fn test_log_sum_exp_small_values() {
        // Should not underflow to -inf
        let v = vec![-1000.0, -1001.0, -1002.0];
        let result = log_sum_exp(&v);
        assert!(result.is_finite());
        assert!(result < -999.0);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let result = log_sum_exp(&[]);
        assert!(result.is_infinite() && result < 0.0);
    }

    // ── normalize_weights ─────────────────────────────────────────────────────

    #[test]
    fn test_normalize_weights_sum_to_one() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = normalize_weights(&w);
        let sum: f64 = normalized.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-12));
    }

    #[test]
    fn test_normalize_weights_already_normalized() {
        let w = vec![0.25, 0.25, 0.25, 0.25];
        let normalized = normalize_weights(&w);
        let sum: f64 = normalized.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-12));
        for (&orig, &norm) in w.iter().zip(normalized.iter()) {
            assert!(approx_eq(orig, norm, 1e-12));
        }
    }

    #[test]
    fn test_normalize_weights_zero_sum_returns_uniform() {
        let w = vec![0.0, 0.0, 0.0];
        let normalized = normalize_weights(&w);
        let sum: f64 = normalized.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-12));
        for &v in &normalized {
            assert!(approx_eq(v, 1.0 / 3.0, 1e-12));
        }
    }

    // ── cost_matrix_euclidean ─────────────────────────────────────────────────

    #[test]
    fn test_cost_matrix_euclidean_1d() {
        let x = vec![vec![0.0_f64], vec![1.0_f64]];
        let y = vec![vec![0.0_f64], vec![2.0_f64]];
        let c = cost_matrix_euclidean(&x, &y);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].len(), 2);
        assert!(approx_eq(c[0][0], 0.0, 1e-12)); // (0-0)^2
        assert!(approx_eq(c[0][1], 4.0, 1e-12)); // (0-2)^2
        assert!(approx_eq(c[1][0], 1.0, 1e-12)); // (1-0)^2
        assert!(approx_eq(c[1][1], 1.0, 1e-12)); // (1-2)^2
    }

    #[test]
    fn test_cost_matrix_diagonal_is_zero() {
        let pts: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let c = cost_matrix_euclidean(&pts, &pts);
        for i in 0..4 {
            assert!(approx_eq(c[i][i], 0.0, 1e-12));
        }
    }

    // ── sinkhorn ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sinkhorn_source_marginal() {
        let a = uniform(3);
        let b = uniform(3);
        let cost = cost_matrix_euclidean(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[vec![0.0], vec![1.0], vec![2.0]],
        );
        let cfg = OtConfig::default();
        let res = sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");

        // Row sums should approximate a
        for (i, &ai) in a.iter().enumerate() {
            let row_sum: f64 = res.transport_plan[i].iter().sum();
            assert!(approx_eq(row_sum, ai, 1e-4), "row {i}: {row_sum} ≠ {ai}");
        }
    }

    #[test]
    fn test_sinkhorn_target_marginal() {
        let a = uniform(3);
        let b = uniform(3);
        let cost = cost_matrix_euclidean(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[vec![0.0], vec![1.0], vec![2.0]],
        );
        let cfg = OtConfig::default();
        let res = sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");
        let n = b.len();
        for j in 0..n {
            let col_sum: f64 = res.transport_plan.iter().map(|row| row[j]).sum();
            assert!(
                approx_eq(col_sum, b[j], 1e-4),
                "col {j}: {col_sum} ≠ {}",
                b[j]
            );
        }
    }

    #[test]
    fn test_sinkhorn_cost_nonnegative() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = cost_matrix_euclidean(&[vec![0.0], vec![3.0]], &[vec![1.0], vec![4.0]]);
        let cfg = OtConfig::default();
        let res = sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");
        assert!(res.cost >= 0.0);
    }

    #[test]
    fn test_sinkhorn_identical_distributions_near_zero_cost() {
        // Identical distributions: transport plan is diagonal, cost minimal
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let pts = vec![vec![0.0_f64], vec![0.0_f64]]; // same support
        let cost = cost_matrix_euclidean(&pts, &pts);
        let cfg = OtConfig {
            epsilon: 0.01,
            ..Default::default()
        };
        let res = sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");
        assert!(
            res.cost < 1e-5,
            "cost should be near 0 for identical distributions, got {}",
            res.cost
        );
    }

    #[test]
    fn test_sinkhorn_error_empty() {
        let result = sinkhorn(&[], &[0.5], &[], &OtConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_sinkhorn_error_shape_mismatch() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5];
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]]; // 2×2 but b has 1 element
        let result = sinkhorn(&a, &b, &cost, &OtConfig::default());
        assert!(result.is_err());
    }

    // ── wasserstein_1d ────────────────────────────────────────────────────────

    #[test]
    fn test_wasserstein_1d_dirac() {
        // W_1(δ_0, δ_1) = 1
        let a =
            wasserstein_1d(&[0.0], &[1.0], &[1.0], &[1.0]).expect("operation should succeed");
        assert!(approx_eq(a, 1.0, 1e-10));
    }

    #[test]
    fn test_wasserstein_1d_symmetric() {
        let pos_a = vec![0.0, 1.0];
        let w_a = vec![0.5, 0.5];
        let pos_b = vec![2.0, 3.0];
        let w_b = vec![0.5, 0.5];
        let d_ab =
            wasserstein_1d(&pos_a, &w_a, &pos_b, &w_b).expect("operation should succeed");
        let d_ba =
            wasserstein_1d(&pos_b, &w_b, &pos_a, &w_a).expect("operation should succeed");
        assert!(approx_eq(d_ab, d_ba, 1e-10));
    }

    #[test]
    fn test_wasserstein_1d_identical_distributions() {
        let pos = vec![0.0, 1.0, 2.0];
        let w = vec![0.3, 0.4, 0.3];
        let d = wasserstein_1d(&pos, &w, &pos, &w).expect("operation should succeed");
        assert!(approx_eq(d, 0.0, 1e-10));
    }

    #[test]
    fn test_wasserstein_1d_known_value() {
        // W_1(uniform on {0,1}, δ_{0.5}) = 0.5
        let d = wasserstein_1d(&[0.0, 1.0], &[0.5, 0.5], &[0.5], &[1.0])
            .expect("operation should succeed");
        assert!(approx_eq(d, 0.5, 1e-10));
    }

    #[test]
    fn test_wasserstein_1d_non_negative() {
        let d = wasserstein_1d(
            &[1.0, 3.0, 5.0],
            &[0.2, 0.5, 0.3],
            &[0.0, 2.0, 4.0, 6.0],
            &[0.25, 0.25, 0.25, 0.25],
        )
        .expect("computation failed");
        assert!(d >= 0.0);
    }

    #[test]
    fn test_wasserstein_1d_error_empty() {
        let result = wasserstein_1d(&[], &[], &[1.0], &[1.0]);
        assert!(result.is_err());
    }

    // ── emd_1d ────────────────────────────────────────────────────────────────

    #[test]
    fn test_emd_1d_equal_to_wasserstein_1d() {
        let pa = vec![0.0, 1.0, 2.0];
        let wa = vec![0.2, 0.5, 0.3];
        let pb = vec![0.5, 1.5, 3.0];
        let wb = vec![0.4, 0.3, 0.3];
        let emd = emd_1d(&pa, &wa, &pb, &wb).expect("operation should succeed");
        let w1 = wasserstein_1d(&pa, &wa, &pb, &wb).expect("operation should succeed");
        assert!(approx_eq(emd, w1, 1e-12));
    }

    #[test]
    fn test_emd_1d_nonnegative() {
        let d = emd_1d(&[0.0], &[1.0], &[10.0], &[1.0]).expect("operation should succeed");
        assert!(d >= 0.0);
    }

    // ── sliced_wasserstein ────────────────────────────────────────────────────

    #[test]
    fn test_sliced_wasserstein_non_negative() {
        let a: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let b: Vec<Vec<f64>> = (0..8).map(|i| vec![(i + 1) as f64, i as f64]).collect();
        let swd = sliced_wasserstein(&a, &b, 50, 42).expect("operation should succeed");
        assert!(swd >= 0.0);
    }

    #[test]
    fn test_sliced_wasserstein_symmetric() {
        let a: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let b: Vec<Vec<f64>> = vec![vec![0.5, 0.5], vec![1.5, 0.5]];
        let swd_ab = sliced_wasserstein(&a, &b, 100, 99).expect("operation should succeed");
        let swd_ba = sliced_wasserstein(&b, &a, 100, 99).expect("operation should succeed");
        assert!(approx_eq(swd_ab, swd_ba, 1e-8));
    }

    #[test]
    fn test_sliced_wasserstein_identical_zero() {
        let a: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let swd = sliced_wasserstein(&a, &a, 50, 7).expect("operation should succeed");
        assert!(approx_eq(swd, 0.0, 1e-10));
    }

    #[test]
    fn test_sliced_wasserstein_error_empty() {
        let result = sliced_wasserstein(&[], &[vec![1.0]], 10, 0);
        assert!(result.is_err());
    }

    // ── WassersteinBarycenter ─────────────────────────────────────────────────

    #[test]
    fn test_barycenter_weights_sum_to_one() {
        let support = vec![0.0, 1.0, 2.0, 3.0];
        let d1 = vec![0.6, 0.3, 0.1, 0.0];
        let d2 = vec![0.0, 0.1, 0.3, 0.6];
        let bary = WassersteinBarycenter::default();
        let result = bary
            .compute(&[d1, d2], &[0.5, 0.5], &support)
            .expect("operation should succeed");
        let sum: f64 = result.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
    }

    #[test]
    fn test_barycenter_single_distribution() {
        // Barycenter of a single distribution is itself
        let support = vec![0.0, 1.0, 2.0];
        let d = vec![0.2, 0.5, 0.3];
        let bary = WassersteinBarycenter::default();
        let result = bary
            .compute(std::slice::from_ref(&d), &[1.0], &support)
            .expect("operation should succeed");
        let sum: f64 = result.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
    }

    #[test]
    fn test_barycenter_non_negative() {
        let support = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let d1 = vec![0.5, 0.4, 0.05, 0.03, 0.02];
        let d2 = vec![0.02, 0.03, 0.05, 0.4, 0.5];
        let bary = WassersteinBarycenter::new(0.1, 100, 1e-6);
        let result = bary
            .compute(&[d1, d2], &[0.5, 0.5], &support)
            .expect("operation should succeed");
        for &v in &result {
            assert!(v >= 0.0, "negative weight {v}");
        }
    }

    // ── SinkhornLoss ──────────────────────────────────────────────────────────

    #[test]
    fn test_sinkhorn_loss_identical_is_zero() {
        let p = vec![0.3, 0.4, 0.3];
        let loss = SinkhornLoss::new(0.05);
        let div = loss.compute(&p, &p, 0.05).expect("operation should succeed");
        assert!(
            approx_eq(div.value, 0.0, 1e-4),
            "expected ~0, got {}",
            div.value
        );
    }

    #[test]
    fn test_sinkhorn_loss_non_negative() {
        let p = vec![0.6, 0.3, 0.1];
        let q = vec![0.1, 0.3, 0.6];
        let loss = SinkhornLoss::default();
        let div = loss.compute(&p, &q, 0.05).expect("operation should succeed");
        assert!(div.value >= 0.0, "Sinkhorn divergence must be non-negative");
    }

    #[test]
    fn test_sinkhorn_loss_symmetric() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.1, 0.4, 0.5];
        let n = p.len();
        let support: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cost = cost_matrix_euclidean(&support, &support);
        let loss = SinkhornLoss::new(0.1);
        let div_pq = loss
            .compute_with_cost(&p, &q, &cost, 0.1)
            .expect("operation should succeed");
        let div_qp = loss
            .compute_with_cost(&q, &p, &cost, 0.1)
            .expect("operation should succeed");
        assert!(
            approx_eq(div_pq.value, div_qp.value, 1e-3),
            "S(p,q)={} ≠ S(q,p)={} (diff={})",
            div_pq.value,
            div_qp.value,
            (div_pq.value - div_qp.value).abs()
        );
    }

    // ── OtLoss ────────────────────────────────────────────────────────────────

    #[test]
    fn test_ot_loss_wasserstein_non_negative() {
        let pred = vec![0.6, 0.4];
        let target = vec![0.3, 0.7];
        let cost = cost_matrix_euclidean(
            &[vec![0.0_f64], vec![1.0_f64]],
            &[vec![0.0_f64], vec![1.0_f64]],
        );
        let loss = OtLoss::wasserstein_loss(&pred, &target, &cost, 0.05)
            .expect("operation should succeed");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_ot_loss_sinkhorn_divergence_identical() {
        let p = vec![0.4, 0.3, 0.3];
        let d = OtLoss::sinkhorn_divergence(&p, &p, 0.05).expect("operation should succeed");
        assert!(approx_eq(d, 0.0, 1e-4));
    }

    // ── partial_sinkhorn ──────────────────────────────────────────────────────

    #[test]
    fn test_partial_sinkhorn_full_mass() {
        // With mass_ratio=1.0 partial OT should approximate standard OT
        let a = uniform(3);
        let b = uniform(3);
        let cost = cost_matrix_euclidean(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[vec![0.0], vec![1.0], vec![2.0]],
        );
        let cfg = PartialOtConfig {
            mass_ratio: 1.0,
            epsilon: 0.05,
            tau: 10.0,
            max_iter: 500,
            tolerance: 1e-8,
        };
        let res = partial_sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");
        assert!(res.cost >= 0.0);
    }

    #[test]
    fn test_partial_sinkhorn_small_mass_ratio() {
        let a = uniform(4);
        let b = uniform(4);
        let cost = cost_matrix_euclidean(
            &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        );
        let cfg = PartialOtConfig {
            mass_ratio: 0.5,
            epsilon: 0.1,
            tau: 1.0,
            max_iter: 200,
            tolerance: 1e-7,
        };
        let res = partial_sinkhorn(&a, &b, &cost, &cfg).expect("operation should succeed");
        assert!(res.cost >= 0.0);
        // Transport plan entries are non-negative
        for row in &res.transport_plan {
            for &t in row {
                assert!(t >= -1e-10, "negative plan entry {t}");
            }
        }
    }

    #[test]
    fn test_partial_sinkhorn_error_invalid_mass() {
        let a = uniform(2);
        let b = uniform(2);
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let cfg = PartialOtConfig {
            mass_ratio: 0.0,
            ..Default::default()
        };
        let result = partial_sinkhorn(&a, &b, &cost, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_sinkhorn_error_empty() {
        let cfg = PartialOtConfig::default();
        let result = partial_sinkhorn(&[], &[1.0], &[], &cfg);
        assert!(result.is_err());
    }

    // ── gromov_wasserstein_distance ───────────────────────────────────────────

    #[test]
    fn test_gromov_wasserstein_non_negative() {
        let n = 3;
        let m = 3;
        // Source: line graph {0,1,2}
        let c_s: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| (i as f64 - j as f64).abs()).collect())
            .collect();
        // Target: slightly different
        let c_t: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..m).map(|j| (i as f64 - j as f64).abs() * 0.5).collect())
            .collect();
        let a = uniform(n);
        let b = uniform(m);
        let (gw, _t) =
            gromov_wasserstein_distance(&c_s, &c_t, &a, &b, 0.05, 10)
                .expect("operation should succeed");
        assert!(gw >= 0.0);
    }

    #[test]
    fn test_gromov_wasserstein_isomorphic_spaces_near_zero() {
        // Identical metric spaces: GW cost should be smaller than two very different spaces.
        let n = 3;
        let c: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| (i as f64 - j as f64).abs()).collect())
            .collect();
        // Very different space (large distances)
        let c_diff: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (i as f64 - j as f64).abs() * 100.0)
                    .collect()
            })
            .collect();
        let a = uniform(n);
        let (gw_same, _) =
            gromov_wasserstein_distance(&c, &c, &a, &a, 0.05, 15)
                .expect("operation should succeed");
        let (gw_diff, _) =
            gromov_wasserstein_distance(&c, &c_diff, &a, &a, 0.05, 15)
                .expect("operation should succeed");
        assert!(gw_same >= 0.0, "GW must be non-negative: {gw_same}");
        assert!(
            gw_same <= gw_diff,
            "GW on same space ({gw_same}) should be ≤ GW on different spaces ({gw_diff})"
        );
    }

    #[test]
    fn test_gromov_wasserstein_error_shape_mismatch() {
        let c_s = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let c_t = vec![vec![0.0]]; // 1×1 but b has 2 elements
        let a = uniform(2);
        let b = uniform(2);
        let result = gromov_wasserstein_distance(&c_s, &c_t, &a, &b, 0.05, 5);
        assert!(result.is_err());
    }

    // ── fused_gromov_wasserstein ──────────────────────────────────────────────

    #[test]
    fn test_fused_gromov_wasserstein_non_negative() {
        let n = 3;
        let m = 3;
        let c_s: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| (i as f64 - j as f64).abs()).collect())
            .collect();
        let c_t = c_s.clone();
        let f_s: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let f_t: Vec<Vec<f64>> = (0..m).map(|i| vec![i as f64 + 0.1]).collect();
        let a = uniform(n);
        let b = uniform(m);
        let (cost, _) =
            fused_gromov_wasserstein(&c_s, &c_t, &f_s, &f_t, &a, &b, 0.5, 0.1, 10)
                .expect("operation should succeed");
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_fused_gromov_wasserstein_alpha_zero_reduces_to_wasserstein() {
        // When alpha=0, FGW reduces to standard Wasserstein on features
        let n = 2;
        let c_s = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let c_t = c_s.clone();
        let f_s = vec![vec![0.0_f64], vec![1.0_f64]];
        let f_t = vec![vec![0.0_f64], vec![1.0_f64]];
        let a = uniform(n);
        let b = uniform(n);
        let (cost, _) =
            fused_gromov_wasserstein(&c_s, &c_t, &f_s, &f_t, &a, &b, 0.0, 0.1, 10)
                .expect("operation should succeed");
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_fused_gromov_wasserstein_error_invalid_alpha() {
        let c = vec![vec![0.0]];
        let f = vec![vec![0.0_f64]];
        let a = vec![1.0];
        let result = fused_gromov_wasserstein(&c, &c, &f, &f, &a, &a, 1.5, 0.05, 5);
        assert!(result.is_err());
    }

    // ── soft_min ──────────────────────────────────────────────────────────────

    #[test]
    fn test_soft_min_approaches_min() {
        use crate::optimal_transport::soft_min;
        let v = vec![1.0, 3.0, 5.0];
        // With small epsilon, soft_min should be close to actual min = 1.0
        let sm = soft_min(&v, 0.001);
        assert!(approx_eq(sm, 1.0, 0.01));
    }

    #[test]
    fn test_soft_min_symmetric_equal() {
        use crate::optimal_transport::soft_min;
        // soft_min([c,c,c], ε) = c − ε·ln(n), which is always ≤ c.
        let v = vec![2.0, 2.0, 2.0];
        let eps = 0.1_f64;
        let sm = soft_min(&v, eps);
        let expected = 2.0 - eps * (3.0_f64).ln();
        assert!(
            approx_eq(sm, expected, 1e-8),
            "soft_min={sm}, expected {expected}"
        );
    }

    // ── wasserstein_distance ──────────────────────────────────────────────────

    #[test]
    fn test_wasserstein_distance_non_negative() {
        let a = uniform(3);
        let b = uniform(3);
        let cost = cost_matrix_euclidean(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[vec![0.0], vec![1.0], vec![2.0]],
        );
        let d = wasserstein_distance(&a, &b, &cost, 0.05).expect("operation should succeed");
        assert!(d >= 0.0);
    }
}
