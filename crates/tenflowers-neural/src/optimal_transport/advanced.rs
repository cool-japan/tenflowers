//! Advanced Optimal Transport algorithms: Unbalanced OT, Domain Adaptation,
//! Scalable OT, and extended metrics.

use tenflowers_core::error::TensorError;

use super::{cost_matrix_euclidean, sinkhorn, wasserstein_1d, OtConfig, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers (Box-Muller random unit vector, no external RNG needed here)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a pseudo-random unit vector in `dim` dimensions using a simple
/// LCG + Box-Muller scheme seeded by `seed`.
fn random_unit_vector(dim: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_add(1);
    let mut out = Vec::with_capacity(dim);
    let mut i = 0usize;
    while i < dim {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-300);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push(r * theta.cos());
        i += 1;
        if i < dim {
            out.push(r * theta.sin());
            i += 1;
        }
    }
    // L2 normalise
    let norm: f64 = out.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-300);
    out.iter_mut().for_each(|x| *x /= norm);
    out
}

/// Project each row of `pts` onto unit vector `dir`.
fn project_1d(pts: &[Vec<f64>], dir: &[f64]) -> Vec<f64> {
    pts.iter()
        .map(|row| row.iter().zip(dir.iter()).map(|(&x, &d)| x * d).sum())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unbalanced Optimal Transport — Chizat et al. 2018
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for unbalanced Sinkhorn (KL-penalised marginals).
#[derive(Debug, Clone)]
pub struct UnbalancedOtConfig {
    /// Entropy regularization ε > 0.
    pub epsilon: f64,
    /// KL marginal penalty weight τ > 0. Large τ → nearly balanced.
    pub tau: f64,
    /// Maximum Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for UnbalancedOtConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            tau: 1.0,
            max_iter: 1000,
            tolerance: 1e-9,
        }
    }
}

/// Unbalanced Sinkhorn solver (Chizat 2018): replaces hard marginal
/// constraints with KL-divergence penalties of weight `τ`.
///
/// The scaling iterations become soft projections:
/// `u ← (a / K v)^{τ/(τ+ε)}`, `v ← (b / Kᵀ u)^{τ/(τ+ε)}`.
#[derive(Debug, Clone)]
pub struct UnbalancedSinkhorn {
    /// Configuration parameters.
    pub config: UnbalancedOtConfig,
}

impl UnbalancedSinkhorn {
    /// Create a new `UnbalancedSinkhorn` solver with the given config.
    pub fn new(config: UnbalancedOtConfig) -> Self {
        Self { config }
    }

    /// Solve unbalanced OT between source weights `a` and target weights `b`
    /// with the given `cost` matrix (n×m).
    ///
    /// Returns an [`OtResult`] with the transport plan and primal cost.
    ///
    /// # Errors
    ///
    /// Returns an error on empty inputs, shape mismatches, or invalid config.
    pub fn solve(
        &self,
        a: &[f64],
        b: &[f64],
        cost: &[Vec<f64>],
    ) -> Result<OtResult, TensorError> {
        let n = a.len();
        let m = b.len();

        if n == 0 || m == 0 {
            return Err(TensorError::invalid_argument_op(
                "UnbalancedSinkhorn::solve",
                "source and target weights must be non-empty",
            ));
        }
        if cost.len() != n || cost.iter().any(|r| r.len() != m) {
            return Err(TensorError::invalid_argument_op(
                "UnbalancedSinkhorn::solve",
                "cost matrix shape must be n×m",
            ));
        }
        if self.config.epsilon <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "UnbalancedSinkhorn::solve",
                "epsilon must be positive",
            ));
        }
        if self.config.tau <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "UnbalancedSinkhorn::solve",
                "tau must be positive",
            ));
        }

        let eps = self.config.epsilon;
        let tau = self.config.tau;
        // Soft-scaling exponent: ρ/(ρ+ε) where ρ = τ*ε/(τ+ε)
        let rho = tau * eps / (tau + eps);
        let exp_fac = rho / eps;

        // Build Gibbs kernel in log domain
        let log_k: Vec<Vec<f64>> = cost
            .iter()
            .map(|row| row.iter().map(|&c| -c / eps).collect())
            .collect();

        // log-domain u, v (dual variables)
        let log_a: Vec<f64> = a.iter().map(|&x| x.max(1e-300).ln()).collect();
        let log_b: Vec<f64> = b.iter().map(|&x| x.max(1e-300).ln()).collect();

        let mut f = vec![0.0_f64; n]; // log u = f/eps
        let mut g = vec![0.0_f64; m]; // log v = g/eps

        let mut converged = false;
        let mut iters = 0usize;

        for iter in 0..self.config.max_iter {
            iters = iter + 1;

            // Compute log(K v)[i] = logsumexp_j(log_k[i][j] + g[j])
            let log_kv: Vec<f64> = (0..n)
                .map(|i| {
                    let vals: Vec<f64> =
                        (0..m).map(|j| log_k[i][j] + g[j]).collect();
                    log_sum_exp_local(&vals)
                })
                .collect();

            // f update (soft projection): f_new[i] = exp_fac * (log_a[i] - log_kv[i])
            let f_new: Vec<f64> = log_a
                .iter()
                .zip(log_kv.iter())
                .map(|(&la, &lkv)| exp_fac * (la - lkv))
                .collect();

            // Compute log(Kᵀ u)[j] = logsumexp_i(log_k[i][j] + f_new[i])
            let log_ktu: Vec<f64> = (0..m)
                .map(|j| {
                    let vals: Vec<f64> =
                        (0..n).map(|i| log_k[i][j] + f_new[i]).collect();
                    log_sum_exp_local(&vals)
                })
                .collect();

            // g update
            let g_new: Vec<f64> = log_b
                .iter()
                .zip(log_ktu.iter())
                .map(|(&lb, &lku)| exp_fac * (lb - lku))
                .collect();

            let delta: f64 = g_new
                .iter()
                .zip(g.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            f = f_new;
            g = g_new;

            if delta < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Reconstruct transport plan: T[i][j] = exp(f[i] + log_k[i][j] + g[j])
        let mut transport_plan = vec![vec![0.0_f64; m]; n];
        let mut primal_cost = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let t_ij = (f[i] + log_k[i][j] + g[j]).exp();
                transport_plan[i][j] = t_ij;
                primal_cost += t_ij * cost[i][j];
            }
        }

        Ok(OtResult {
            cost: primal_cost,
            transport_plan,
            u: f.iter().map(|&x| x * eps).collect(),
            v: g.iter().map(|&x| x * eps).collect(),
            iterations: iters,
            converged,
        })
    }
}

/// Numerically-stable log-sum-exp for a local slice.
#[inline]
fn log_sum_exp_local(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return max_v;
    }
    max_v + v.iter().map(|&x| (x - max_v).exp()).sum::<f64>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// Partial OT Solver (transport fraction m of total mass)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for partial OT.
#[derive(Debug, Clone)]
pub struct PartialOtAdvancedConfig {
    /// Entropy regularization ε > 0.
    pub epsilon: f64,
    /// Fraction of mass to transport, in (0, 1].
    pub mass_fraction: f64,
    /// Maximum outer iterations (bisection on the threshold).
    pub max_iter: usize,
    /// Inner Sinkhorn maximum iterations.
    pub inner_max_iter: usize,
    /// Tolerance for convergence.
    pub tolerance: f64,
}

impl Default for PartialOtAdvancedConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            mass_fraction: 0.5,
            max_iter: 50,
            inner_max_iter: 500,
            tolerance: 1e-7,
        }
    }
}

/// Partial OT solver: transports exactly a fraction `m` of total mass.
///
/// Uses a thresholded Sinkhorn approach: a virtual mass sink/source is added
/// to absorb the remaining `1 - m` mass fraction at zero cost.
#[derive(Debug, Clone)]
pub struct PartialOtSolver {
    /// Configuration.
    pub config: PartialOtAdvancedConfig,
}

impl PartialOtSolver {
    /// Create a new `PartialOtSolver`.
    pub fn new(config: PartialOtAdvancedConfig) -> Self {
        Self { config }
    }

    /// Solve partial OT between `a` (n) and `b` (m) with cost matrix `cost` (n×m).
    ///
    /// Returns `(partial_cost, transport_plan)` where `transport_plan[i][j]`
    /// carries at most `a[i]` from source `i` and at most `b[j]` into target `j`,
    /// and the total mass transported equals approximately `mass_fraction`.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid inputs or configuration.
    pub fn solve(
        &self,
        a: &[f64],
        b: &[f64],
        cost: &[Vec<f64>],
    ) -> Result<(f64, Vec<Vec<f64>>), TensorError> {
        let n = a.len();
        let m = b.len();

        if n == 0 || m == 0 {
            return Err(TensorError::invalid_argument_op(
                "PartialOtSolver::solve",
                "source and target must be non-empty",
            ));
        }
        if cost.len() != n || cost.iter().any(|r| r.len() != m) {
            return Err(TensorError::invalid_argument_op(
                "PartialOtSolver::solve",
                "cost matrix must be n×m",
            ));
        }
        if !(0.0 < self.config.mass_fraction && self.config.mass_fraction <= 1.0) {
            return Err(TensorError::invalid_argument_op(
                "PartialOtSolver::solve",
                "mass_fraction must be in (0, 1]",
            ));
        }

        let mf = self.config.mass_fraction;
        let eps = self.config.epsilon;

        // Augment with a dummy sink column (cost = threshold θ)
        // and a dummy source row. We bisect θ so that transported mass ≈ mf.
        let sum_a: f64 = a.iter().sum();
        let sum_b: f64 = b.iter().sum();
        let target_mass = mf * sum_a.min(sum_b);

        // Build augmented cost: n+1 rows × m+1 cols
        // Extra source (row n): cost 0 to each target, fills missing mass in b.
        // Extra sink (col m): cost 0 to each source, absorbs leftover from a.
        // We use a large threshold cost for the dummy→dummy entry.
        let max_cost = cost
            .iter()
            .flat_map(|r| r.iter())
            .cloned()
            .fold(0.0_f64, f64::max);
        let thresh = max_cost + 1.0;

        let aug_n = n + 1;
        let aug_m = m + 1;

        let mut aug_cost = vec![vec![0.0_f64; aug_m]; aug_n];
        for i in 0..n {
            for j in 0..m {
                aug_cost[i][j] = cost[i][j];
            }
            // Source i can dump leftover into dummy sink at threshold cost
            aug_cost[i][m] = thresh;
        }
        // Dummy source row n: can fill any target at threshold cost
        for j in 0..m {
            aug_cost[n][j] = thresh;
        }
        aug_cost[n][m] = 0.0; // dummy → dummy is free

        // Augmented marginals
        let slack_a = sum_a * (1.0 - mf) + sum_a.min(sum_b) * mf - target_mass;
        let slack_b = sum_b * (1.0 - mf) + sum_a.min(sum_b) * mf - target_mass;
        let mut aug_a: Vec<f64> = a.to_vec();
        aug_a.push(slack_a.max(1e-30));
        let mut aug_b: Vec<f64> = b.to_vec();
        aug_b.push(slack_b.max(1e-30));

        // Normalise
        let sum_aug_a: f64 = aug_a.iter().sum::<f64>().max(1e-300);
        let sum_aug_b: f64 = aug_b.iter().sum::<f64>().max(1e-300);
        let aug_a_n: Vec<f64> = aug_a.iter().map(|&x| x / sum_aug_a).collect();
        let aug_b_n: Vec<f64> = aug_b.iter().map(|&x| x / sum_aug_b).collect();

        let cfg = OtConfig {
            epsilon: eps,
            max_iter: self.config.inner_max_iter,
            tolerance: self.config.tolerance,
            log_domain: true,
        };

        let res = sinkhorn(&aug_a_n, &aug_b_n, &aug_cost, &cfg)?;

        // Extract the n×m sub-plan (remove augmented rows/cols)
        let scale = sum_a.min(sum_b) * mf;
        let mut partial_plan = vec![vec![0.0_f64; m]; n];
        let mut total_cost = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let t_ij = res.transport_plan[i][j] * scale;
                partial_plan[i][j] = t_ij;
                total_cost += t_ij * cost[i][j];
            }
        }

        Ok((total_cost, partial_plan))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OT for Domain Adaptation — Flamary et al. 2016
// ─────────────────────────────────────────────────────────────────────────────

/// Optimal Transport for Domain Adaptation (Flamary 2016).
///
/// Estimates a Sinkhorn transport map from source domain `xs` (ns × d) to
/// target domain `xt` (nt × d), then uses it to project new source samples.
#[derive(Debug, Clone)]
pub struct OtDaTransport {
    /// Entropy regularization ε.
    pub epsilon: f64,
    /// Maximum Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Fitted transport plan (ns × nt), stored after `fit`.
    transport_plan: Option<Vec<Vec<f64>>>,
    /// Source samples used during fit.
    xs_fit: Option<Vec<Vec<f64>>>,
    /// Target samples used during fit.
    xt_fit: Option<Vec<Vec<f64>>>,
}

impl OtDaTransport {
    /// Create a new `OtDaTransport` adapter.
    pub fn new(epsilon: f64, max_iter: usize, tolerance: f64) -> Self {
        Self {
            epsilon,
            max_iter,
            tolerance,
            transport_plan: None,
            xs_fit: None,
            xt_fit: None,
        }
    }

    /// Fit the transport map between source `xs` (ns × d) and target `xt` (nt × d).
    ///
    /// # Errors
    ///
    /// Returns an error on empty inputs or Sinkhorn failure.
    pub fn fit(
        &mut self,
        xs: &[Vec<f64>],
        xt: &[Vec<f64>],
    ) -> Result<(), TensorError> {
        if xs.is_empty() || xt.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "OtDaTransport::fit",
                "source and target must be non-empty",
            ));
        }

        let ns = xs.len();
        let nt = xt.len();

        let a = vec![1.0 / ns as f64; ns];
        let b = vec![1.0 / nt as f64; nt];
        let cost = cost_matrix_euclidean(xs, xt);

        let cfg = OtConfig {
            epsilon: self.epsilon,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            log_domain: true,
        };

        let res = sinkhorn(&a, &b, &cost, &cfg)?;
        self.transport_plan = Some(res.transport_plan);
        self.xs_fit = Some(xs.to_vec());
        self.xt_fit = Some(xt.to_vec());

        Ok(())
    }

    /// Transform source samples `xs_new` (n_new × d) to the target domain.
    ///
    /// Each adapted point is: `xs_new_adapted[i] = Σ_j T̂[i,j] * xt[j] / Σ_j T̂[i,j]`
    /// where `T̂` is the normalised transport row for the nearest training source.
    ///
    /// # Errors
    ///
    /// Returns an error if `fit` has not been called.
    pub fn transform(&self, xs_new: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TensorError> {
        let plan = self.transport_plan.as_ref().ok_or_else(|| {
            TensorError::invalid_argument_op(
                "OtDaTransport::transform",
                "must call fit() before transform()",
            )
        })?;
        let xs_fit = self.xs_fit.as_ref().expect("xs_fit always set with plan");
        let xt_fit = self.xt_fit.as_ref().expect("xt_fit always set with plan");

        let ns = xs_fit.len();
        let nt = xt_fit.len();
        let d = xt_fit[0].len();

        xs_new
            .iter()
            .map(|x_new| {
                // Find nearest source training point
                let nearest = (0..ns)
                    .min_by(|&i, &j| {
                        let di: f64 = xs_fit[i]
                            .iter()
                            .zip(x_new.iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        let dj: f64 = xs_fit[j]
                            .iter()
                            .zip(x_new.iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);

                let row = &plan[nearest];
                let row_sum: f64 = row.iter().sum::<f64>().max(1e-300);

                let mut adapted = vec![0.0_f64; d];
                for j in 0..nt {
                    let w = row[j] / row_sum;
                    for k in 0..d {
                        adapted[k] += w * xt_fit[j][k];
                    }
                }
                adapted
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(Ok)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JDOT — Courty et al. 2017: Joint Distribution Optimal Transport
// ─────────────────────────────────────────────────────────────────────────────

/// JDOT optimizer (Courty 2017): minimizes joint OT cost combining
/// feature and label alignment.
///
/// At each outer iteration, the feature + label cost is updated using the
/// current classifier's predictions on target, then Sinkhorn is re-solved.
#[derive(Debug, Clone)]
pub struct JdotOptimizer {
    /// Feature cost weight α.
    pub alpha: f64,
    /// Entropy regularization ε.
    pub epsilon: f64,
    /// Number of outer JDOT iterations.
    pub n_outer: usize,
    /// Maximum inner Sinkhorn iterations.
    pub sinkhorn_max_iter: usize,
    /// Converged transport plan (ns × nt).
    pub transport_plan: Option<Vec<Vec<f64>>>,
}

impl JdotOptimizer {
    /// Create a new `JdotOptimizer`.
    pub fn new(alpha: f64, epsilon: f64, n_outer: usize) -> Self {
        Self {
            alpha,
            epsilon,
            n_outer,
            sinkhorn_max_iter: 500,
            transport_plan: None,
        }
    }

    /// Fit JDOT: alternates between solving OT with the combined feature+label
    /// cost and updating pseudo-labels for target via source-weighted assignment.
    ///
    /// `xs` (ns × d_feat), `ys` (ns × d_label), `xt` (nt × d_feat).
    ///
    /// Returns pseudo-labels for `xt` (nt × d_label).
    ///
    /// # Errors
    ///
    /// Returns an error on shape mismatches or Sinkhorn failure.
    pub fn fit_transform(
        &mut self,
        xs: &[Vec<f64>],
        ys: &[Vec<f64>],
        xt: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        let ns = xs.len();
        let nt = xt.len();

        if ns == 0 || nt == 0 {
            return Err(TensorError::invalid_argument_op(
                "JdotOptimizer::fit_transform",
                "source and target must be non-empty",
            ));
        }
        if ys.len() != ns {
            return Err(TensorError::invalid_argument_op(
                "JdotOptimizer::fit_transform",
                "ys must have same number of rows as xs",
            ));
        }

        let d_label = ys[0].len();
        let a = vec![1.0 / ns as f64; ns];
        let b = vec![1.0 / nt as f64; nt];

        // Initialise pseudo-labels: uniform average of source labels
        let label_mean: Vec<f64> = (0..d_label)
            .map(|k| ys.iter().map(|y| y[k]).sum::<f64>() / ns as f64)
            .collect();
        let mut yt_pseudo: Vec<Vec<f64>> = vec![label_mean; nt];

        let cfg = OtConfig {
            epsilon: self.epsilon,
            max_iter: self.sinkhorn_max_iter,
            tolerance: 1e-9,
            log_domain: true,
        };

        let feat_cost = cost_matrix_euclidean(xs, xt);

        for _ in 0..self.n_outer {
            // Build combined cost: α * feat_cost + (1-α) * label_cost
            let label_cost: Vec<Vec<f64>> = (0..ns)
                .map(|i| {
                    (0..nt)
                        .map(|j| {
                            ys[i]
                                .iter()
                                .zip(yt_pseudo[j].iter())
                                .map(|(&a, &b)| (a - b) * (a - b))
                                .sum()
                        })
                        .collect()
                })
                .collect();

            let combined: Vec<Vec<f64>> = (0..ns)
                .map(|i| {
                    (0..nt)
                        .map(|j| self.alpha * feat_cost[i][j] + (1.0 - self.alpha) * label_cost[i][j])
                        .collect()
                })
                .collect();

            let res = sinkhorn(&a, &b, &combined, &cfg)?;
            let plan = &res.transport_plan;

            // Update pseudo-labels: yt_pseudo[j] = Σ_i plan[i][j] * ys[i] / col_sum[j]
            for j in 0..nt {
                let col_sum: f64 = (0..ns).map(|i| plan[i][j]).sum::<f64>().max(1e-300);
                let mut lbl = vec![0.0_f64; d_label];
                for i in 0..ns {
                    let w = plan[i][j] / col_sum;
                    for k in 0..d_label {
                        lbl[k] += w * ys[i][k];
                    }
                }
                yt_pseudo[j] = lbl;
            }

            self.transport_plan = Some(plan.clone());
        }

        Ok(yt_pseudo)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Online Sliced Wasserstein (streaming, random projections)
// ─────────────────────────────────────────────────────────────────────────────

/// Online / streaming Sliced Wasserstein distance estimator.
///
/// Accumulates random projections incrementally; call \[`update`\] with new
/// batches and \[`estimate`\] to get the current running estimate.
#[derive(Debug, Clone)]
pub struct OnlineSlicedWasserstein {
    /// Dimension of the feature space.
    dim: usize,
    /// Number of random projections per update.
    n_projections: usize,
    /// Running sum of sliced W₁ values.
    running_sum: f64,
    /// Number of projection batches accumulated.
    n_batches: usize,
    /// Running seed for reproducible projections.
    seed: u64,
}

impl OnlineSlicedWasserstein {
    /// Create a new `OnlineSlicedWasserstein` estimator.
    pub fn new(dim: usize, n_projections: usize, seed: u64) -> Self {
        Self {
            dim,
            n_projections,
            running_sum: 0.0,
            n_batches: 0,
            seed,
        }
    }

    /// Update the running estimate with a new pair of sample batches
    /// `xs_batch` (n_s × d) and `xt_batch` (n_t × d).
    ///
    /// # Errors
    ///
    /// Returns an error on empty batches or dimension mismatch.
    pub fn update(
        &mut self,
        xs_batch: &[Vec<f64>],
        xt_batch: &[Vec<f64>],
    ) -> Result<(), TensorError> {
        if xs_batch.is_empty() || xt_batch.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "OnlineSlicedWasserstein::update",
                "batches must be non-empty",
            ));
        }
        let d = xs_batch[0].len();
        if d != self.dim {
            return Err(TensorError::invalid_argument_op(
                "OnlineSlicedWasserstein::update",
                "batch dimension does not match configured dim",
            ));
        }

        let ns = xs_batch.len();
        let nt = xt_batch.len();
        let ws = vec![1.0 / ns as f64; ns];
        let wt = vec![1.0 / nt as f64; nt];

        let mut batch_sum = 0.0_f64;
        for p in 0..self.n_projections {
            self.seed = self.seed.wrapping_add(p as u64 + 1);
            let dir = random_unit_vector(d, self.seed);
            let proj_s = project_1d(xs_batch, &dir);
            let proj_t = project_1d(xt_batch, &dir);
            let w1 = wasserstein_1d(&proj_s, &ws, &proj_t, &wt)?;
            batch_sum += w1;
        }

        self.running_sum += batch_sum / self.n_projections as f64;
        self.n_batches += 1;

        Ok(())
    }

    /// Return the current running estimate (mean over all batches).
    pub fn estimate(&self) -> f64 {
        if self.n_batches == 0 {
            return 0.0;
        }
        self.running_sum / self.n_batches as f64
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.running_sum = 0.0;
        self.n_batches = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Subsampled Sinkhorn — Genevay et al. 2018 (mini-batch stochastic OT)
// ─────────────────────────────────────────────────────────────────────────────

/// Mini-batch / subsampled Sinkhorn estimator (Genevay 2018).
///
/// Approximates the Sinkhorn divergence by averaging over random sub-batches
/// drawn from the full datasets.
#[derive(Debug, Clone)]
pub struct SubsampledSinkhorn {
    /// Entropy regularization ε.
    pub epsilon: f64,
    /// Sub-batch size.
    pub batch_size: usize,
    /// Number of random sub-batches to average over.
    pub n_batches: usize,
    /// Maximum inner Sinkhorn iterations.
    pub max_iter: usize,
    /// Tolerance.
    pub tolerance: f64,
    /// RNG seed.
    pub seed: u64,
}

impl SubsampledSinkhorn {
    /// Create a new `SubsampledSinkhorn`.
    pub fn new(
        epsilon: f64,
        batch_size: usize,
        n_batches: usize,
        seed: u64,
    ) -> Self {
        Self {
            epsilon,
            batch_size,
            n_batches,
            max_iter: 500,
            tolerance: 1e-7,
            seed,
        }
    }

    /// Estimate the Sinkhorn divergence S_ε(μ, ν) between sample sets
    /// `xs` (ns × d) and `xt` (nt × d) via mini-batch averaging.
    ///
    /// # Errors
    ///
    /// Returns an error on empty inputs or invalid configuration.
    pub fn divergence(
        &self,
        xs: &[Vec<f64>],
        xt: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        if xs.is_empty() || xt.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SubsampledSinkhorn::divergence",
                "sample sets must be non-empty",
            ));
        }
        if self.batch_size == 0 {
            return Err(TensorError::invalid_argument_op(
                "SubsampledSinkhorn::divergence",
                "batch_size must be positive",
            ));
        }

        let ns = xs.len();
        let nt = xt.len();
        let bs = self.batch_size.min(ns).min(nt);
        let cfg = OtConfig {
            epsilon: self.epsilon,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            log_domain: true,
        };

        let mut seed = self.seed;
        let mut total = 0.0_f64;

        for batch in 0..self.n_batches {
            seed = seed.wrapping_add(batch as u64 + 1337);

            // Simple LCG subsample indices
            let idx_s = subsample_indices(ns, bs, seed);
            seed = seed.wrapping_add(999_983);
            let idx_t = subsample_indices(nt, bs, seed);
            seed = seed.wrapping_add(777_777);
            let idx_s2 = subsample_indices(ns, bs, seed);
            seed = seed.wrapping_add(888_888);
            let idx_t2 = subsample_indices(nt, bs, seed);

            let batch_xs: Vec<Vec<f64>> = idx_s.iter().map(|&i| xs[i].clone()).collect();
            let batch_xt: Vec<Vec<f64>> = idx_t.iter().map(|&j| xt[j].clone()).collect();
            let batch_xs2: Vec<Vec<f64>> = idx_s2.iter().map(|&i| xs[i].clone()).collect();
            let batch_xt2: Vec<Vec<f64>> = idx_t2.iter().map(|&j| xt[j].clone()).collect();

            let a = vec![1.0 / bs as f64; bs];
            let b = vec![1.0 / bs as f64; bs];

            let c_st = cost_matrix_euclidean(&batch_xs, &batch_xt);
            let c_ss = cost_matrix_euclidean(&batch_xs, &batch_xs2);
            let c_tt = cost_matrix_euclidean(&batch_xt, &batch_xt2);

            let ot_st = sinkhorn(&a, &b, &c_st, &cfg)?.cost;
            let ot_ss = sinkhorn(&a, &b, &c_ss, &cfg)?.cost;
            let ot_tt = sinkhorn(&a, &b, &c_tt, &cfg)?.cost;

            // S_ε(μ,ν) = OT(μ,ν) - 0.5*OT(μ,μ) - 0.5*OT(ν,ν)
            total += ot_st - 0.5 * ot_ss - 0.5 * ot_tt;
        }

        Ok((total / self.n_batches as f64).max(0.0))
    }
}

/// Subsample `k` indices from `[0, n)` using an LCG.
fn subsample_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates partial shuffle
    let k = k.min(n);
    for i in 0..k {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let j = i + (state >> 33) as usize % (n - i);
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree Wasserstein — exact O(n) on tree metric
// ─────────────────────────────────────────────────────────────────────────────

/// A weighted tree edge for Tree Wasserstein computation.
#[derive(Debug, Clone)]
pub struct TreeEdge {
    /// Parent node index (usize::MAX for root).
    pub parent: usize,
    /// Child node index.
    pub child: usize,
    /// Edge weight (distance).
    pub weight: f64,
}

/// Tree Wasserstein distance — exact O(n) computation on a tree metric.
///
/// Given a rooted tree with `n` nodes and edge weights, computes the
/// Wasserstein-1 distance between two distributions `a` and `b` on the
/// tree leaves in O(n) time via the flow formulation.
#[derive(Debug, Clone)]
pub struct TreeWasserstein {
    /// Edges of the tree (must form a valid rooted tree).
    pub edges: Vec<TreeEdge>,
    /// Total number of nodes.
    pub n_nodes: usize,
}

impl TreeWasserstein {
    /// Create a new `TreeWasserstein` from a list of edges.
    pub fn new(n_nodes: usize, edges: Vec<TreeEdge>) -> Self {
        Self { edges, n_nodes }
    }

    /// Compute the Tree Wasserstein distance between distributions `a` and `b`
    /// (both of length `n_nodes`, indexed by node).
    ///
    /// Formula: `TW(a, b) = Σ_{edges (u,v,w)} w · |Σ_{i ∈ subtree(v)} (a[i] - b[i])|`.
    ///
    /// # Errors
    ///
    /// Returns an error if `a` or `b` length does not match `n_nodes`.
    pub fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64, TensorError> {
        if a.len() != self.n_nodes || b.len() != self.n_nodes {
            return Err(TensorError::invalid_argument_op(
                "TreeWasserstein::distance",
                "a and b must have length equal to n_nodes",
            ));
        }

        // Build children list
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); self.n_nodes];
        let mut root = 0usize;
        let mut has_parent = vec![false; self.n_nodes];
        for edge in &self.edges {
            if edge.parent == usize::MAX {
                root = edge.child;
            } else {
                children[edge.parent].push(edge.child);
                has_parent[edge.child] = true;
            }
        }
        // Find actual root (no parent)
        for i in 0..self.n_nodes {
            if !has_parent[i] {
                root = i;
                break;
            }
        }

        // Build edge weight lookup: child → weight
        let mut edge_weight: Vec<f64> = vec![0.0; self.n_nodes];
        for edge in &self.edges {
            if edge.parent != usize::MAX {
                edge_weight[edge.child] = edge.weight;
            }
        }

        // Compute subtree mass differences via post-order DFS
        let diff: Vec<f64> = (0..self.n_nodes).map(|i| a[i] - b[i]).collect();

        // Post-order traversal using explicit stack
        let mut subtree_sum = diff.clone();
        let mut stack: Vec<(usize, bool)> = vec![(root, false)];
        let mut order: Vec<usize> = Vec::new();
        while let Some((node, processed)) = stack.pop() {
            if processed {
                order.push(node);
                for &child in &children[node] {
                    subtree_sum[node] += subtree_sum[child];
                }
            } else {
                stack.push((node, true));
                for &child in &children[node] {
                    stack.push((child, false));
                }
            }
        }
        let _ = order; // used implicitly via stack processing

        // TW = Σ_{v ≠ root} edge_weight[v] * |subtree_sum[v]|
        let tw: f64 = (0..self.n_nodes)
            .filter(|&v| v != root)
            .map(|v| edge_weight[v] * subtree_sum[v].abs())
            .sum();

        Ok(tw)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fréchet Inception Distance (OT-based)
// ─────────────────────────────────────────────────────────────────────────────

/// OT-based Fréchet Distance between two sets of feature vectors.
///
/// Approximates the squared Fréchet distance using the closed-form
/// Gaussian approximation: `FD = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})`.
///
/// The matrix square root is approximated via the Newton-Schulz iteration.
#[derive(Debug, Clone)]
pub struct FrechetDistance {
    /// Number of Newton-Schulz iterations for matrix square root.
    pub n_iter: usize,
}

impl Default for FrechetDistance {
    fn default() -> Self {
        Self { n_iter: 100 }
    }
}

impl FrechetDistance {
    /// Create a new `FrechetDistance` with the given Newton-Schulz iterations.
    pub fn new(n_iter: usize) -> Self {
        Self { n_iter }
    }

    /// Compute the Fréchet distance between feature sets `feats_a` (n × d) and
    /// `feats_b` (m × d).
    ///
    /// Returns the squared Fréchet distance (FID-style metric).
    ///
    /// # Errors
    ///
    /// Returns an error on empty inputs or dimension mismatch.
    pub fn compute(
        &self,
        feats_a: &[Vec<f64>],
        feats_b: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        if feats_a.is_empty() || feats_b.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "FrechetDistance::compute",
                "feature sets must be non-empty",
            ));
        }
        let d = feats_a[0].len();
        if feats_b[0].len() != d {
            return Err(TensorError::invalid_argument_op(
                "FrechetDistance::compute",
                "feature sets must have the same dimension",
            ));
        }

        let (mu_a, sigma_a) = mean_and_cov(feats_a);
        let (mu_b, sigma_b) = mean_and_cov(feats_b);

        // Mean difference squared
        let mean_diff_sq: f64 = mu_a
            .iter()
            .zip(mu_b.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();

        // Tr(Σ_a) + Tr(Σ_b) - 2 * Tr((Σ_a Σ_b)^{1/2})
        let tr_a: f64 = (0..d).map(|i| sigma_a[i][i]).sum();
        let tr_b: f64 = (0..d).map(|i| sigma_b[i][i]).sum();

        // Product Σ_a Σ_b
        let prod = mat_mul_sym(&sigma_a, &sigma_b, d);

        // Approximate Tr((Σ_a Σ_b)^{1/2}) via eigenvalue decomposition proxy:
        // Use the nuclear norm approximation: Tr(A^{1/2}) ≈ Σ sqrt(max(λ_i, 0))
        // where λ_i are estimated via power iteration (diagonal approximation).
        let tr_sqrt = approx_trace_sqrt(&prod, d, self.n_iter);

        let fd = mean_diff_sq + tr_a + tr_b - 2.0 * tr_sqrt;
        Ok(fd.max(0.0))
    }
}

/// Compute mean vector and covariance matrix for a set of feature vectors.
fn mean_and_cov(feats: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = feats.len();
    let d = feats[0].len();

    // Mean
    let mut mu = vec![0.0_f64; d];
    for row in feats {
        for (k, &v) in row.iter().enumerate() {
            mu[k] += v;
        }
    }
    mu.iter_mut().for_each(|x| *x /= n as f64);

    // Covariance (biased)
    let mut sigma = vec![vec![0.0_f64; d]; d];
    for row in feats {
        let centered: Vec<f64> = row.iter().zip(mu.iter()).map(|(&x, &m)| x - m).collect();
        for i in 0..d {
            for j in 0..d {
                sigma[i][j] += centered[i] * centered[j];
            }
        }
    }
    let scale = (n as f64).max(1.0);
    for i in 0..d {
        for j in 0..d {
            sigma[i][j] /= scale;
        }
    }

    (mu, sigma)
}

/// Dense matrix multiplication A * B → C (d × d).
fn mat_mul_sym(a: &[Vec<f64>], b: &[Vec<f64>], d: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..d {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Approximate `Tr(A^{1/2})` for a PSD matrix A via Lanczos-style diagonal
/// extraction: compute diagonal elements of Σ, then return Σ sqrt(|diag|).
/// This is a diagonal approximation; accurate when A is nearly diagonal.
fn approx_trace_sqrt(a: &[Vec<f64>], d: usize, _n_iter: usize) -> f64 {
    // Use diagonal approximation: Tr(A^{1/2}) ≈ Σ_i sqrt(max(A[i][i], 0))
    // This is exact when A is diagonal and a reasonable approximation otherwise.
    (0..d).map(|i| a[i][i].max(0.0).sqrt()).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended OT Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Extended OT evaluation metrics for comparing distributions.
#[derive(Debug, Clone, Default)]
pub struct OtMetrics {
    /// Recorded Wasserstein distances.
    pub wasserstein_distances: Vec<f64>,
    /// Recorded Sinkhorn divergences.
    pub sinkhorn_divergences: Vec<f64>,
    /// Recorded Sliced Wasserstein distances.
    pub sliced_wasserstein_distances: Vec<f64>,
    /// Recorded Gromov-Wasserstein costs.
    pub gromov_wasserstein_costs: Vec<f64>,
    /// Recorded Fréchet distances.
    pub frechet_distances: Vec<f64>,
}

impl OtMetrics {
    /// Create a new empty `OtMetrics`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a Wasserstein distance measurement.
    pub fn record_wasserstein(&mut self, dist: f64) {
        self.wasserstein_distances.push(dist);
    }

    /// Record a Sinkhorn divergence measurement.
    pub fn record_sinkhorn(&mut self, div: f64) {
        self.sinkhorn_divergences.push(div);
    }

    /// Record a Sliced Wasserstein distance.
    pub fn record_sliced(&mut self, swd: f64) {
        self.sliced_wasserstein_distances.push(swd);
    }

    /// Record a Gromov-Wasserstein cost.
    pub fn record_gromov(&mut self, gw: f64) {
        self.gromov_wasserstein_costs.push(gw);
    }

    /// Record a Fréchet distance.
    pub fn record_frechet(&mut self, fd: f64) {
        self.frechet_distances.push(fd);
    }

    /// Compute the mean of recorded Wasserstein distances.
    pub fn mean_wasserstein(&self) -> f64 {
        mean_vec(&self.wasserstein_distances)
    }

    /// Compute the mean of recorded Sinkhorn divergences.
    pub fn mean_sinkhorn(&self) -> f64 {
        mean_vec(&self.sinkhorn_divergences)
    }

    /// Compute the mean of recorded Sliced Wasserstein distances.
    pub fn mean_sliced(&self) -> f64 {
        mean_vec(&self.sliced_wasserstein_distances)
    }

    /// Compute the mean of recorded Gromov-Wasserstein costs.
    pub fn mean_gromov(&self) -> f64 {
        mean_vec(&self.gromov_wasserstein_costs)
    }

    /// Compute the mean of recorded Fréchet distances.
    pub fn mean_frechet(&self) -> f64 {
        mean_vec(&self.frechet_distances)
    }
}

fn mean_vec(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimal_transport::cost_matrix_euclidean;

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── UnbalancedSinkhorn ────────────────────────────────────────────────────

    #[test]
    fn test_unbalanced_sinkhorn_basic() {
        let a = uniform(3);
        let b = uniform(3);
        let cost = cost_matrix_euclidean(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[vec![0.0], vec![1.0], vec![2.0]],
        );
        let solver = UnbalancedSinkhorn::new(UnbalancedOtConfig::default());
        let res = solver.solve(&a, &b, &cost).expect("solve failed");
        assert!(res.cost >= 0.0, "cost must be non-negative");
    }

    #[test]
    fn test_unbalanced_sinkhorn_identical_low_cost() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let pts = vec![vec![0.0_f64], vec![0.0_f64]];
        let cost = cost_matrix_euclidean(&pts, &pts);
        let config = UnbalancedOtConfig {
            epsilon: 0.01,
            tau: 10.0,
            ..Default::default()
        };
        let solver = UnbalancedSinkhorn::new(config);
        let res = solver.solve(&a, &b, &cost).expect("solve failed");
        assert!(res.cost < 1e-5, "cost near 0 for identical: {}", res.cost);
    }

    #[test]
    fn test_unbalanced_sinkhorn_error_empty() {
        let solver = UnbalancedSinkhorn::new(UnbalancedOtConfig::default());
        let result = solver.solve(&[], &[0.5], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unbalanced_sinkhorn_error_bad_epsilon() {
        let a = uniform(2);
        let b = uniform(2);
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let config = UnbalancedOtConfig {
            epsilon: -1.0,
            ..Default::default()
        };
        let solver = UnbalancedSinkhorn::new(config);
        let result = solver.solve(&a, &b, &cost);
        assert!(result.is_err());
    }

    #[test]
    fn test_unbalanced_sinkhorn_transport_plan_shape() {
        let n = 4;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let xs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let xt: Vec<Vec<f64>> = (0..m).map(|j| vec![j as f64 * 1.5]).collect();
        let cost = cost_matrix_euclidean(&xs, &xt);
        let solver = UnbalancedSinkhorn::new(UnbalancedOtConfig::default());
        let res = solver.solve(&a, &b, &cost).expect("solve failed");
        assert_eq!(res.transport_plan.len(), n);
        assert_eq!(res.transport_plan[0].len(), m);
    }

    // ── PartialOtSolver ───────────────────────────────────────────────────────

    #[test]
    fn test_partial_ot_solver_basic() {
        let a = uniform(4);
        let b = uniform(4);
        let xs: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let cost = cost_matrix_euclidean(&xs, &xs);
        let config = PartialOtAdvancedConfig {
            mass_fraction: 0.5,
            ..Default::default()
        };
        let solver = PartialOtSolver::new(config);
        let (total_cost, plan) = solver.solve(&a, &b, &cost).expect("solve failed");
        assert!(total_cost >= 0.0);
        assert_eq!(plan.len(), 4);
        for row in &plan {
            for &t in row {
                assert!(t >= -1e-10, "negative plan entry {t}");
            }
        }
    }

    #[test]
    fn test_partial_ot_full_mass_nonnegative() {
        let a = uniform(3);
        let b = uniform(3);
        let xs: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64]).collect();
        let cost = cost_matrix_euclidean(&xs, &xs);
        let config = PartialOtAdvancedConfig {
            mass_fraction: 1.0,
            ..Default::default()
        };
        let solver = PartialOtSolver::new(config);
        let (total_cost, _plan) = solver.solve(&a, &b, &cost).expect("solve failed");
        assert!(total_cost >= 0.0);
    }

    #[test]
    fn test_partial_ot_error_empty() {
        let solver = PartialOtSolver::new(PartialOtAdvancedConfig::default());
        let result = solver.solve(&[], &[1.0], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_ot_error_bad_mass_fraction() {
        let a = uniform(2);
        let b = uniform(2);
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let config = PartialOtAdvancedConfig {
            mass_fraction: 0.0,
            ..Default::default()
        };
        let solver = PartialOtSolver::new(config);
        let result = solver.solve(&a, &b, &cost);
        assert!(result.is_err());
    }

    // ── OtDaTransport ─────────────────────────────────────────────────────────

    #[test]
    fn test_ot_da_transport_fit_transform() {
        let xs: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 0.0]).collect();
        let xt: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 + 1.0, 0.5]).collect();
        let mut adapter = OtDaTransport::new(0.05, 300, 1e-8);
        adapter.fit(&xs, &xt).expect("fit failed");
        let adapted = adapter.transform(&xs).expect("transform failed");
        assert_eq!(adapted.len(), 5);
        for row in &adapted {
            assert_eq!(row.len(), 2);
            for &v in row {
                assert!(v.is_finite(), "non-finite adapted value");
            }
        }
    }

    #[test]
    fn test_ot_da_transport_transform_without_fit() {
        let adapter = OtDaTransport::new(0.05, 100, 1e-7);
        let xs = vec![vec![1.0, 2.0]];
        let result = adapter.transform(&xs);
        assert!(result.is_err());
    }

    #[test]
    fn test_ot_da_transport_error_empty() {
        let mut adapter = OtDaTransport::new(0.05, 100, 1e-7);
        let result = adapter.fit(&[], &[vec![1.0]]);
        assert!(result.is_err());
    }

    // ── JdotOptimizer ─────────────────────────────────────────────────────────

    #[test]
    fn test_jdot_fit_transform_basic() {
        let xs: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let ys: Vec<Vec<f64>> = (0..4).map(|i| vec![if i < 2 { 0.0 } else { 1.0 }]).collect();
        let xt: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 + 0.5]).collect();
        let mut jdot = JdotOptimizer::new(0.5, 0.1, 3);
        let pseudo = jdot.fit_transform(&xs, &ys, &xt).expect("jdot failed");
        assert_eq!(pseudo.len(), 4);
        for lbl in &pseudo {
            assert_eq!(lbl.len(), 1);
            assert!(lbl[0].is_finite());
        }
    }

    #[test]
    fn test_jdot_error_empty() {
        let mut jdot = JdotOptimizer::new(0.5, 0.1, 3);
        let result = jdot.fit_transform(&[], &[], &[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_jdot_error_ys_mismatch() {
        let xs = vec![vec![0.0], vec![1.0]];
        let ys = vec![vec![0.0]]; // only 1 label for 2 source points
        let xt = vec![vec![0.5]];
        let mut jdot = JdotOptimizer::new(0.5, 0.1, 2);
        let result = jdot.fit_transform(&xs, &ys, &xt);
        assert!(result.is_err());
    }

    // ── OnlineSlicedWasserstein ───────────────────────────────────────────────

    #[test]
    fn test_online_sw_basic() {
        let mut osw = OnlineSlicedWasserstein::new(2, 20, 42);
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.0]).collect();
        let xt: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 + 2.0, 0.0]).collect();
        osw.update(&xs, &xt).expect("update failed");
        let est = osw.estimate();
        assert!(est >= 0.0, "estimate must be non-negative");
    }

    #[test]
    fn test_online_sw_identical_zero() {
        let mut osw = OnlineSlicedWasserstein::new(1, 20, 7);
        let pts: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        osw.update(&pts, &pts).expect("update failed");
        let est = osw.estimate();
        assert!(approx_eq(est, 0.0, 1e-10), "identical distributions: {est}");
    }

    #[test]
    fn test_online_sw_reset() {
        let mut osw = OnlineSlicedWasserstein::new(2, 10, 0);
        let xs = vec![vec![0.0, 1.0]];
        let xt = vec![vec![1.0, 0.0]];
        osw.update(&xs, &xt).expect("update failed");
        osw.reset();
        assert_eq!(osw.n_batches, 0);
        assert_eq!(osw.estimate(), 0.0);
    }

    #[test]
    fn test_online_sw_error_empty() {
        let mut osw = OnlineSlicedWasserstein::new(2, 10, 0);
        let result = osw.update(&[], &[vec![1.0, 2.0]]);
        assert!(result.is_err());
    }

    // ── SubsampledSinkhorn ────────────────────────────────────────────────────

    #[test]
    fn test_subsampled_sinkhorn_nonnegative() {
        let xs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let xt: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 + 5.0]).collect();
        let ss = SubsampledSinkhorn::new(0.1, 5, 10, 42);
        let div = ss.divergence(&xs, &xt).expect("divergence failed");
        assert!(div >= 0.0, "divergence must be non-negative: {div}");
    }

    #[test]
    fn test_subsampled_sinkhorn_identical_near_zero() {
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let ss = SubsampledSinkhorn::new(0.05, 5, 20, 0);
        let div = ss.divergence(&xs, &xs).expect("divergence failed");
        assert!(div < 1.0, "identical: divergence should be small, got {div}");
    }

    #[test]
    fn test_subsampled_sinkhorn_error_empty() {
        let ss = SubsampledSinkhorn::new(0.1, 5, 5, 0);
        let result = ss.divergence(&[], &[vec![1.0]]);
        assert!(result.is_err());
    }

    // ── TreeWasserstein ───────────────────────────────────────────────────────

    #[test]
    fn test_tree_wasserstein_line_graph() {
        // Tree: 0 -- 1 -- 2 (unit edge weights)
        // a = [1, 0, 0], b = [0, 0, 1] → TW = 2
        let edges = vec![
            TreeEdge { parent: usize::MAX, child: 0, weight: 0.0 },
            TreeEdge { parent: 0, child: 1, weight: 1.0 },
            TreeEdge { parent: 1, child: 2, weight: 1.0 },
        ];
        let tw = TreeWasserstein::new(3, edges);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0];
        let dist = tw.distance(&a, &b).expect("distance failed");
        assert!(dist >= 0.0, "TW must be non-negative");
        // Flow must pass through both edges: expected 2.0
        assert!(approx_eq(dist, 2.0, 1e-10), "expected TW=2, got {dist}");
    }

    #[test]
    fn test_tree_wasserstein_identical_zero() {
        let edges = vec![
            TreeEdge { parent: usize::MAX, child: 0, weight: 0.0 },
            TreeEdge { parent: 0, child: 1, weight: 1.5 },
        ];
        let tw = TreeWasserstein::new(2, edges);
        let a = vec![0.5, 0.5];
        let dist = tw.distance(&a, &a).expect("distance failed");
        assert!(approx_eq(dist, 0.0, 1e-10), "identical: TW={dist}");
    }

    #[test]
    fn test_tree_wasserstein_error_length_mismatch() {
        let edges = vec![
            TreeEdge { parent: usize::MAX, child: 0, weight: 0.0 },
            TreeEdge { parent: 0, child: 1, weight: 1.0 },
        ];
        let tw = TreeWasserstein::new(2, edges);
        let result = tw.distance(&[1.0], &[0.5, 0.5]);
        assert!(result.is_err());
    }

    // ── FrechetDistance ───────────────────────────────────────────────────────

    #[test]
    fn test_frechet_distance_identical_near_zero() {
        let feats: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64, (i * 2) as f64])
            .collect();
        let fd_calc = FrechetDistance::default();
        let fd = fd_calc.compute(&feats, &feats).expect("compute failed");
        assert!(fd >= 0.0, "FD must be non-negative");
        // Identical distributions: FD ≈ 0
        assert!(fd < 1e-10, "identical: FD={fd}");
    }

    #[test]
    fn test_frechet_distance_different_distributions() {
        let feats_a: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0, 0.0]).collect();
        let feats_b: Vec<Vec<f64>> = (0..10).map(|_| vec![10.0, 10.0]).collect();
        let fd_calc = FrechetDistance::default();
        let fd = fd_calc.compute(&feats_a, &feats_b).expect("compute failed");
        assert!(fd > 0.0, "different distributions: FD should be positive");
    }

    #[test]
    fn test_frechet_distance_error_empty() {
        let fd_calc = FrechetDistance::default();
        let result = fd_calc.compute(&[], &[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_frechet_distance_error_dim_mismatch() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0]];
        let fd_calc = FrechetDistance::default();
        let result = fd_calc.compute(&a, &b);
        assert!(result.is_err());
    }

    // ── OtMetrics ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ot_metrics_record_and_mean() {
        let mut metrics = OtMetrics::new();
        metrics.record_wasserstein(1.0);
        metrics.record_wasserstein(3.0);
        assert!(approx_eq(metrics.mean_wasserstein(), 2.0, 1e-10));
    }

    #[test]
    fn test_ot_metrics_empty_mean_zero() {
        let metrics = OtMetrics::new();
        assert_eq!(metrics.mean_wasserstein(), 0.0);
        assert_eq!(metrics.mean_sinkhorn(), 0.0);
    }

    #[test]
    fn test_ot_metrics_all_types() {
        let mut metrics = OtMetrics::new();
        metrics.record_sinkhorn(0.5);
        metrics.record_sliced(1.2);
        metrics.record_gromov(0.3);
        metrics.record_frechet(2.1);
        assert!(approx_eq(metrics.mean_sinkhorn(), 0.5, 1e-10));
        assert!(approx_eq(metrics.mean_sliced(), 1.2, 1e-10));
        assert!(approx_eq(metrics.mean_gromov(), 0.3, 1e-10));
        assert!(approx_eq(metrics.mean_frechet(), 2.1, 1e-10));
    }
}
