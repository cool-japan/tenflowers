//! Causal discovery extensions for time series (§5-§8).
//!
//! PCMCI, LiNGAM-TS, Intervention Effect estimation,
//! and causal graph evaluation metrics.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::{CdtsResult, CdtsError, pearson_corr, ols_cholesky, compute_residuals,
    normal_cdf, fisher_z, mean, std_dev, entropy_from_counts, rss,
    CdtsTransferEntropy, CdtsTransferEntropyConfig, CdtsGrangerTest,
    CdtsGrangerResult};


// ─────────────────────────────────────────────────────────────────────────────
// §5. CdtsPcmci — PCMCI Algorithm (Runge 2019)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for PCMCI.
#[derive(Debug, Clone)]
pub struct CdtsPcmciConfig {
    /// Maximum lag to consider.
    pub max_lag: usize,
    /// Significance level for conditional independence tests.
    pub alpha: f64,
    /// Minimum partial correlation for an edge to be retained in PC phase.
    pub pc_alpha: f64,
}

impl Default for CdtsPcmciConfig {
    fn default() -> Self {
        Self {
            max_lag: 3,
            alpha: 0.05,
            pc_alpha: 0.05,
        }
    }
}

/// A directed causal link from variable `src` at lag `lag` to variable `tgt`.
#[derive(Debug, Clone)]
pub struct CdtsCausalLink {
    pub src_var: usize,
    pub src_lag: usize,
    pub tgt_var: usize,
    pub partial_corr: f64,
    pub p_value: f64,
    pub is_significant: bool,
}

/// PCMCI result: causal graph as a list of significant links.
#[derive(Debug, Clone)]
pub struct CdtsPcmciResult {
    pub links: Vec<CdtsCausalLink>,
    /// Number of variables.
    pub n_vars: usize,
    /// Maximum lag tested.
    pub max_lag: usize,
}

impl CdtsPcmciResult {
    /// Get adjacency matrix: adj\[tgt\]\[src_lag_idx\] where src_lag_idx encodes (var, lag).
    pub fn adjacency_matrix(&self, n_vars: usize, max_lag: usize) -> Vec<Vec<bool>> {
        let cols = n_vars * (max_lag + 1);
        let mut adj = vec![vec![false; cols]; n_vars];
        for link in &self.links {
            if link.is_significant {
                let col = link.src_var * (max_lag + 1) + link.src_lag;
                if col < cols {
                    adj[link.tgt_var][col] = true;
                }
            }
        }
        adj
    }
}

/// PCMCI causal discovery algorithm for multivariate time series.
pub struct CdtsPcmci {
    pub config: CdtsPcmciConfig,
}

impl CdtsPcmci {
    pub fn new(config: CdtsPcmciConfig) -> Self {
        Self { config }
    }

    /// Compute partial correlation between variables `x` and `y` conditioning on `z_set`.
    ///
    /// Uses sequential residualization: regress x on z, regress y on z,
    /// then compute Pearson correlation of residuals.
    pub(super) fn partial_correlation(x: &[f64], y: &[f64], z_set: &[Vec<f64>]) -> CdtsResult<f64> {
        if z_set.is_empty() {
            return Ok(pearson_corr(x, y));
        }
        let n = x.len();
        // Build design matrix for z_set
        let mut xmat: Vec<Vec<f64>> = vec![vec![1.0]; n];
        for z in z_set {
            if z.len() != n {
                return Err(CdtsError("partial_correlation: z length mismatch".into()));
            }
            for (i, &zi) in z.iter().enumerate() {
                xmat[i].push(zi);
            }
        }
        let bx = ols_cholesky(&xmat, x)?;
        let by = ols_cholesky(&xmat, y)?;
        let rx = compute_residuals(&xmat, x, &bx);
        let ry = compute_residuals(&xmat, y, &by);
        Ok(pearson_corr(&rx, &ry))
    }

    /// Fisher Z significance test for partial correlation.
    /// H0: partial_corr = 0.  Returns p-value (two-tailed).
    fn partial_corr_pvalue(pcorr: f64, n: usize, n_cond: usize) -> f64 {
        let df = (n as f64) - (n_cond as f64) - 3.0;
        if df <= 0.0 {
            return 1.0;
        }
        let z = fisher_z(pcorr) * df.sqrt();
        2.0 * (1.0 - normal_cdf(z.abs()))
    }

    /// PC phase: iteratively remove edges with low partial correlation given growing conditioning sets.
    fn pc_phase(&self, series: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<bool>> {
        let p = self.config.max_lag;
        // parents[tgt] = set of (src_var, lag) pairs that are candidate parents
        let mut adj = vec![vec![true; k * (p + 1)]; k];
        // Remove self-lag-0 (same variable, lag 0 = contemporaneous)
        for var in 0..k {
            adj[var][var * (p + 1)] = false;
        }

        // Iteratively test and remove edges
        for cond_size in 0..=2 {
            for tgt in 0..k {
                // Snapshot candidates from current adj state
                let candidates: Vec<(usize, usize)> = {
                    let adj_row = &adj[tgt];
                    (0..k)
                        .flat_map(|v| {
                            (1..=p).filter_map(move |lag| {
                                let col = v * (p + 1) + lag;
                                if adj_row[col] {
                                    Some((v, lag))
                                } else {
                                    None
                                }
                            })
                        })
                        .collect()
                };

                // Collect edges to remove (avoid mutating adj while iterating candidates)
                let mut to_remove: Vec<usize> = Vec::new();

                for &(src, lag) in &candidates {
                    // Build time-aligned series with lag
                    let start = p;
                    let y_tgt: Vec<f64> = (start..n).map(|t| series[tgt][t]).collect();
                    let x_src: Vec<f64> = (start..n).map(|t| series[src][t - lag]).collect();
                    let n_eff = y_tgt.len();

                    // Choose conditioning set: up to cond_size other parents
                    let cond_parents: Vec<(usize, usize)> = candidates
                        .iter()
                        .filter(|&&(v, l)| !(v == src && l == lag))
                        .take(cond_size)
                        .cloned()
                        .collect();

                    let z_set: Vec<Vec<f64>> = cond_parents
                        .iter()
                        .map(|&(v, l)| (start..n).map(|t| series[v][t - l]).collect())
                        .collect();

                    let pcorr = match Self::partial_correlation(&x_src, &y_tgt, &z_set) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let p_val = Self::partial_corr_pvalue(pcorr, n_eff, cond_parents.len());

                    if p_val > self.config.pc_alpha {
                        let col = src * (p + 1) + lag;
                        to_remove.push(col);
                    }
                }

                // Apply removals
                for col in to_remove {
                    adj[tgt][col] = false;
                }
            }
        }
        adj
    }

    /// Discover causal structure in a multivariate time series.
    ///
    /// `data[t][k]` = value of variable k at time t.
    pub fn discover(&self, data: &[Vec<f64>]) -> CdtsResult<CdtsPcmciResult> {
        let n = data.len();
        if n == 0 {
            return Err(CdtsError("PCMCI: empty data".into()));
        }
        let k = data[0].len();
        if k < 2 {
            return Err(CdtsError("PCMCI: need >= 2 variables".into()));
        }
        let p = self.config.max_lag;
        if n <= p {
            return Err(CdtsError(format!(
                "PCMCI: n={} must exceed max_lag={}",
                n, p
            )));
        }

        // Transpose to series[var][t]
        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
        for t in 0..n {
            for var in 0..k {
                series[var].push(data[t][var]);
            }
        }

        // PC phase: build candidate parent sets
        let adj = self.pc_phase(&series, n, k);

        // MCI phase: for each remaining edge, compute MCI partial correlation
        let mut links = Vec::new();
        let start = p;
        let n_eff = n - start;

        for tgt in 0..k {
            for src in 0..k {
                for lag in 1..=p {
                    let col = src * (p + 1) + lag;
                    if !adj[tgt][col] {
                        continue;
                    }

                    let y_tgt: Vec<f64> = (start..n).map(|t| series[tgt][t]).collect();
                    let x_src: Vec<f64> = (start..n).map(|t| series[src][t - lag]).collect();

                    // MCI conditioning: parents of tgt (lag 1) and parents of src (lag 1)
                    let mut z_set: Vec<Vec<f64>> = Vec::new();
                    // Past of target (all lags 1..=p for tgt itself)
                    for l in 1..=p.min(2) {
                        z_set.push((start..n).map(|t| series[tgt][t - l]).collect());
                    }
                    // Past of source
                    for l in 1..=p.min(2) {
                        if !(src == tgt && l == lag) {
                            z_set.push((start..n).map(|t| series[src][t - l]).collect());
                        }
                    }

                    let pcorr = Self::partial_correlation(&x_src, &y_tgt, &z_set).unwrap_or(0.0);
                    let p_val = Self::partial_corr_pvalue(pcorr, n_eff, z_set.len());
                    let is_significant = p_val < self.config.alpha;

                    links.push(CdtsCausalLink {
                        src_var: src,
                        src_lag: lag,
                        tgt_var: tgt,
                        partial_corr: pcorr,
                        p_value: p_val,
                        is_significant,
                    });
                }
            }
        }

        Ok(CdtsPcmciResult {
            links,
            n_vars: k,
            max_lag: p,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6. CdtsLinguisticTS — Time-Series LiNGAM (TiMINo)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for time-series LiNGAM.
#[derive(Debug, Clone)]
pub struct CdtsLingamTsConfig {
    /// Number of lags to consider.
    pub n_lags: usize,
    /// Negentropy approximation: use kurtosis-based proxy.
    pub use_kurtosis: bool,
    /// Significance threshold for non-Gaussianity test.
    pub ng_threshold: f64,
}

impl Default for CdtsLingamTsConfig {
    fn default() -> Self {
        Self {
            n_lags: 2,
            use_kurtosis: true,
            ng_threshold: 0.1,
        }
    }
}

/// Non-Gaussianity score for a series (approximated via excess kurtosis).
pub(super) fn negentropy_approx(x: &[f64]) -> f64 {
    if x.len() < 4 {
        return 0.0;
    }
    let m = mean(x);
    let s = std_dev(x).max(1e-15);
    let n = x.len() as f64;
    // Excess kurtosis
    let kurt: f64 = x.iter().map(|&xi| ((xi - m) / s).powi(4)).sum::<f64>() / n - 3.0;
    // Negentropy ~ kurt^2 / 12  (Jones & Sibson approximation)
    kurt.powi(2) / 12.0
}

/// Causal ordering result from time-series LiNGAM.
#[derive(Debug, Clone)]
pub struct CdtsLingamTsResult {
    /// Causal ordering of variables (from most exogenous to most endogenous).
    pub causal_order: Vec<usize>,
    /// Non-Gaussianity scores for each variable.
    pub ng_scores: Vec<f64>,
    /// Residual-based adjacency matrix (is_causal\[i\]\[j\] = i causes j).
    pub is_causal: Vec<Vec<bool>>,
}

/// Time-series LiNGAM estimator (TiMINo-style).
///
/// Identifies causal order among variables by residual non-Gaussianity:
/// the most exogenous variable has the most non-Gaussian residuals when
/// regressed on its own lags.
pub struct CdtsLingamTs {
    pub config: CdtsLingamTsConfig,
}

impl CdtsLingamTs {
    pub fn new(config: CdtsLingamTsConfig) -> Self {
        Self { config }
    }

    /// Compute lagged regression residuals: regress y on its own p lags + lags of other vars.
    fn lagged_residuals(
        series: &[Vec<f64>],
        target_var: usize,
        exclude_var: Option<usize>,
        p: usize,
    ) -> CdtsResult<Vec<f64>> {
        let n = series[0].len();
        let k = series.len();
        if n <= p {
            return Err(CdtsError("lingam_ts: too few observations".into()));
        }
        let n_eff = n - p;

        let mut xmat: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row = vec![1.0_f64];
            for var in 0..k {
                if Some(var) == exclude_var {
                    continue;
                }
                for lag in 1..=p {
                    row.push(series[var][t - lag]);
                }
            }
            xmat.push(row);
        }
        let y: Vec<f64> = (p..n).map(|t| series[target_var][t]).collect();
        let beta = ols_cholesky(&xmat, &y)?;
        Ok(compute_residuals(&xmat, &y, &beta))
    }

    /// Identify causal ordering of variables via sequential residual non-Gaussianity.
    pub fn identify_causal_order(&self, data: &[Vec<f64>]) -> CdtsResult<CdtsLingamTsResult> {
        let n = data.len();
        if n == 0 {
            return Err(CdtsError("LingamTs: empty data".into()));
        }
        let k = data[0].len();
        if k < 2 {
            return Err(CdtsError("LingamTs: need >= 2 variables".into()));
        }
        let p = self.config.n_lags;

        // Transpose to series[var][t]
        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
        for t in 0..n {
            for var in 0..k {
                series[var].push(data[t][var]);
            }
        }

        let mut remaining: Vec<usize> = (0..k).collect();
        let mut causal_order: Vec<usize> = Vec::with_capacity(k);
        let mut ng_scores: Vec<f64> = vec![0.0; k];
        let mut is_causal = vec![vec![false; k]; k];

        while !remaining.is_empty() {
            // For each remaining variable, compute negentropy of residuals
            // after regressing on own lags only (not on other remaining vars)
            let mut best_var = remaining[0];
            let mut best_ng = f64::NEG_INFINITY;

            for &var in &remaining {
                let resid = Self::lagged_residuals(&series, var, None, p)
                    .unwrap_or_else(|_| series[var].clone());
                let ng = negentropy_approx(&resid);
                if ng > best_ng {
                    best_ng = ng;
                    best_var = var;
                }
            }

            ng_scores[best_var] = best_ng;
            causal_order.push(best_var);
            remaining.retain(|&v| v != best_var);

            // Mark causality: best_var does NOT cause already-ordered vars
            // but remaining vars may be caused by best_var
            for &later in &remaining {
                // Check if best_var's lags reduce non-Gaussianity of `later`
                let ng_with = {
                    let resid = Self::lagged_residuals(&series, later, None, p)
                        .unwrap_or_else(|_| series[later].clone());
                    negentropy_approx(&resid)
                };
                let ng_without = {
                    let resid = Self::lagged_residuals(&series, later, Some(best_var), p)
                        .unwrap_or_else(|_| series[later].clone());
                    negentropy_approx(&resid)
                };
                // If including best_var's lags reduces ng of `later`, best_var -> later
                if ng_without - ng_with > self.config.ng_threshold {
                    is_causal[best_var][later] = true;
                }
            }
        }

        Ok(CdtsLingamTsResult {
            causal_order,
            ng_scores,
            is_causal,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7. CdtsInterventionEffect — Causal effect estimation from time series
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for intervention effect analysis.
#[derive(Debug, Clone)]
pub struct CdtsInterventionConfig {
    /// Number of pre-intervention periods to use.
    pub n_pre: Option<usize>,
    /// Number of post-intervention periods to use.
    pub n_post: Option<usize>,
    /// Confidence level for confidence intervals (e.g., 0.95).
    pub confidence_level: f64,
    /// Method: ITS (interrupted time series) or SynControl (synthetic control).
    pub method: CdtsInterventionMethod,
}

/// Intervention effect estimation method.
#[derive(Debug, Clone, PartialEq)]
pub enum CdtsInterventionMethod {
    /// Interrupted time series: simple before/after comparison with trend adjustment.
    Its,
    /// Synthetic control: weighted combination of control units.
    SyntheticControl,
}

impl Default for CdtsInterventionConfig {
    fn default() -> Self {
        Self {
            n_pre: None,
            n_post: None,
            confidence_level: 0.95,
            method: CdtsInterventionMethod::Its,
        }
    }
}

/// Causal effect estimate.
#[derive(Debug, Clone)]
pub struct CdtsEffectResult {
    /// Average treatment effect (post-intervention mean difference).
    pub ate: f64,
    /// Lower bound of confidence interval.
    pub ci_lower: f64,
    /// Upper bound of confidence interval.
    pub ci_upper: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Counterfactual series (predicted without intervention).
    pub counterfactual: Vec<f64>,
    /// Actual post-intervention series.
    pub observed_post: Vec<f64>,
    /// Point-wise causal effect.
    pub pointwise_effect: Vec<f64>,
}

/// Intervention effect estimator using ITS and synthetic control.
pub struct CdtsInterventionEffect {
    pub config: CdtsInterventionConfig,
}

impl CdtsInterventionEffect {
    pub fn new(config: CdtsInterventionConfig) -> Self {
        Self { config }
    }

    /// Estimate causal effect using Interrupted Time Series (ITS).
    ///
    /// Fits a regression model Y ~ time + post_indicator + time*post on treated series.
    /// The counterfactual is the pre-intervention trend extrapolated post-intervention.
    fn its_analysis(
        treated: &[f64],
        intervention_time: usize,
        confidence_level: f64,
    ) -> CdtsResult<CdtsEffectResult> {
        let n = treated.len();
        if intervention_time == 0 || intervention_time >= n {
            return Err(CdtsError(format!(
                "ITS: intervention_time={} must be in 1..{}",
                intervention_time, n
            )));
        }
        // Pre-period regression: Y ~ time
        let pre = &treated[..intervention_time];
        let n_pre = pre.len();
        let mut xmat_pre: Vec<Vec<f64>> = (0..n_pre).map(|t| vec![1.0, t as f64]).collect();
        let beta_pre = ols_cholesky(&xmat_pre, pre)?;

        // Project counterfactual into post period
        let n_post = n - intervention_time;
        let mut counterfactual = Vec::with_capacity(n_post);
        for t in intervention_time..n {
            let cf = beta_pre[0] + beta_pre[1] * t as f64;
            counterfactual.push(cf);
        }

        let observed_post = treated[intervention_time..].to_vec();
        let pointwise_effect: Vec<f64> = observed_post
            .iter()
            .zip(counterfactual.iter())
            .map(|(&y, &cf)| y - cf)
            .collect();

        let ate = mean(&pointwise_effect);

        // Standard error from residuals
        let resid_pre = compute_residuals(&xmat_pre, pre, &beta_pre);
        let sigma2 = rss(&resid_pre) / (n_pre as f64 - 2.0).max(1.0);
        let std_error = (sigma2 / n_post as f64).sqrt();

        // Normal approximation CI
        let z_alpha = {
            let alpha = 1.0 - confidence_level;
            // Bisect for z: normal_cdf(z) = 1 - alpha/2
            let target = 1.0 - alpha / 2.0;
            let mut lo = 0.0_f64;
            let mut hi = 10.0_f64;
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                if normal_cdf(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (lo + hi) / 2.0
        };
        let ci_lower = ate - z_alpha * std_error;
        let ci_upper = ate + z_alpha * std_error;

        Ok(CdtsEffectResult {
            ate,
            ci_lower,
            ci_upper,
            std_error,
            counterfactual,
            observed_post,
            pointwise_effect,
        })
    }

    /// Synthetic control: find weights w such that Σ w_j * control_j[t] ≈ treated[t] for t < intervention_time.
    ///
    /// Weights are found via non-negative least squares (projected gradient descent).
    fn synthetic_control(
        treated: &[f64],
        controls: &[Vec<f64>],
        intervention_time: usize,
        confidence_level: f64,
    ) -> CdtsResult<CdtsEffectResult> {
        let n = treated.len();
        let n_controls = controls.len();
        if n_controls == 0 {
            return Err(CdtsError("SynControl: no control units provided".into()));
        }
        if intervention_time == 0 || intervention_time >= n {
            return Err(CdtsError(format!(
                "SynControl: intervention_time={} must be in 1..{}",
                intervention_time, n
            )));
        }
        for (i, ctrl) in controls.iter().enumerate() {
            if ctrl.len() != n {
                return Err(CdtsError(format!(
                    "SynControl: control[{}] has wrong length",
                    i
                )));
            }
        }

        let n_pre = intervention_time;
        let pre_treated = &treated[..n_pre];

        // Build control matrix: ctrl_mat[t][j] = control j at time t (pre-period)
        let ctrl_mat: Vec<Vec<f64>> = (0..n_pre)
            .map(|t| (0..n_controls).map(|j| controls[j][t]).collect())
            .collect();

        // NNLS via projected gradient descent
        let max_iter = 1000;
        let lr = 0.01;
        let mut weights = vec![1.0 / n_controls as f64; n_controls];

        for _iter in 0..max_iter {
            // Gradient: 2 * C' (C w - y)
            let mut grad = vec![0.0_f64; n_controls];
            for t in 0..n_pre {
                let pred: f64 = (0..n_controls).map(|j| ctrl_mat[t][j] * weights[j]).sum();
                let err = pred - pre_treated[t];
                for j in 0..n_controls {
                    grad[j] += 2.0 * ctrl_mat[t][j] * err;
                }
            }
            // Gradient step + project onto simplex
            for j in 0..n_controls {
                weights[j] = (weights[j] - lr * grad[j]).max(0.0);
            }
            // Project onto probability simplex
            let w_sum: f64 = weights.iter().sum();
            if w_sum > 1e-15 {
                for j in 0..n_controls {
                    weights[j] /= w_sum;
                }
            } else {
                weights = vec![1.0 / n_controls as f64; n_controls];
            }
        }

        // Compute counterfactual
        let counterfactual: Vec<f64> = (intervention_time..n)
            .map(|t| (0..n_controls).map(|j| controls[j][t] * weights[j]).sum())
            .collect();

        let observed_post = treated[intervention_time..].to_vec();
        let pointwise_effect: Vec<f64> = observed_post
            .iter()
            .zip(counterfactual.iter())
            .map(|(&y, &cf)| y - cf)
            .collect();

        let ate = mean(&pointwise_effect);

        // Placebo std error from pre-period fit error
        let pre_cf: Vec<f64> = (0..n_pre)
            .map(|t| {
                (0..n_controls)
                    .map(|j| controls[j][t] * weights[j])
                    .sum::<f64>()
            })
            .collect();
        let pre_resid: Vec<f64> = pre_treated
            .iter()
            .zip(pre_cf.iter())
            .map(|(&y, &cf)| y - cf)
            .collect();
        let sigma = std_dev(&pre_resid).max(1e-15);
        let std_error = sigma / (observed_post.len() as f64).sqrt();

        let z_alpha = {
            let alpha = 1.0 - confidence_level;
            let target = 1.0 - alpha / 2.0;
            let mut lo = 0.0_f64;
            let mut hi = 10.0_f64;
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                if normal_cdf(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (lo + hi) / 2.0
        };
        let ci_lower = ate - z_alpha * std_error;
        let ci_upper = ate + z_alpha * std_error;

        Ok(CdtsEffectResult {
            ate,
            ci_lower,
            ci_upper,
            std_error,
            counterfactual,
            observed_post,
            pointwise_effect,
        })
    }

    /// Estimate causal effect of an intervention.
    ///
    /// `treated` = outcome series for the treated unit.
    /// `controls` = outcome series for control units (needed for synthetic control; may be empty for ITS).
    /// `intervention_time` = index of first post-intervention observation.
    pub fn estimate_effect(
        &self,
        treated: &[f64],
        controls: &[Vec<f64>],
        intervention_time: usize,
    ) -> CdtsResult<CdtsEffectResult> {
        match self.config.method {
            CdtsInterventionMethod::Its => {
                Self::its_analysis(treated, intervention_time, self.config.confidence_level)
            }
            CdtsInterventionMethod::SyntheticControl => Self::synthetic_control(
                treated,
                controls,
                intervention_time,
                self.config.confidence_level,
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8. CdtsMetrics — Causal discovery evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Binary causal graph for evaluation purposes.
/// `edges[i][j] = true` means i causes j.
#[derive(Debug, Clone)]
pub struct CdtsCausalGraph {
    pub n_vars: usize,
    pub edges: Vec<Vec<bool>>,
}

impl CdtsCausalGraph {
    /// Create an empty causal graph with n_vars variables.
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            edges: vec![vec![false; n_vars]; n_vars],
        }
    }

    /// Add a directed edge from `src` to `tgt`.
    pub fn add_edge(&mut self, src: usize, tgt: usize) {
        if src < self.n_vars && tgt < self.n_vars {
            self.edges[src][tgt] = true;
        }
    }

    /// Count total number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.edges
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&e| e)
            .count()
    }

    /// Flatten edges into a boolean vector (row-major).
    pub fn to_flat(&self) -> Vec<bool> {
        self.edges
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect()
    }
}

/// Evaluation metrics for causal graph discovery.
#[derive(Debug, Clone)]
pub struct CdtsDiscoveryMetrics {
    /// Structural Hamming Distance (missing edges + extra edges + reversed edges).
    pub shd: usize,
    /// Precision on directed edges.
    pub precision: f64,
    /// Recall on directed edges.
    pub recall: f64,
    /// F1 score on directed edges.
    pub f1: f64,
    /// Number of true positive edges.
    pub tp: usize,
    /// Number of false positive edges.
    pub fp: usize,
    /// Number of false negative edges.
    pub fn_count: usize,
    /// Number of reversed edges.
    pub reversed: usize,
}

/// Metrics computation for causal discovery evaluation.
pub struct CdtsMetrics;

impl CdtsMetrics {
    /// Compute evaluation metrics given predicted and true causal graphs.
    pub fn evaluate(
        predicted: &CdtsCausalGraph,
        true_graph: &CdtsCausalGraph,
    ) -> CdtsResult<CdtsDiscoveryMetrics> {
        if predicted.n_vars != true_graph.n_vars {
            return Err(CdtsError(
                "CdtsMetrics: graphs have different n_vars".into(),
            ));
        }
        let k = predicted.n_vars;
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;
        let mut reversed = 0usize;

        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue;
                }
                let pred = predicted.edges[i][j];
                let truth = true_graph.edges[i][j];
                let rev = true_graph.edges[j][i];
                if pred && truth {
                    tp += 1;
                } else if pred && !truth {
                    if rev {
                        reversed += 1;
                    } else {
                        fp += 1;
                    }
                } else if !pred && truth {
                    fn_count += 1;
                }
            }
        }

        let shd = fp + fn_count + reversed;
        let precision = if tp + fp + reversed > 0 {
            tp as f64 / (tp + fp + reversed) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(CdtsDiscoveryMetrics {
            shd,
            precision,
            recall,
            f1,
            tp,
            fp,
            fn_count,
            reversed,
        })
    }

    /// Compute time-lagged cross-correlation matrix.
    ///
    /// Returns matrix\[i\]\[j\] = max correlation between variable i and variable j at lags 0..=max_lag.
    pub fn lagged_correlation_matrix(
        data: &[Vec<f64>],
        max_lag: usize,
    ) -> CdtsResult<Vec<Vec<f64>>> {
        if data.is_empty() {
            return Err(CdtsError("lagged_corr_matrix: empty data".into()));
        }
        let k = data[0].len();
        let n = data.len();
        let mut matrix = vec![vec![0.0_f64; k]; k];

        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
        for t in 0..n {
            for var in 0..k {
                series[var].push(data[t][var]);
            }
        }

        for i in 0..k {
            for j in 0..k {
                if i == j {
                    matrix[i][j] = 1.0;
                    continue;
                }
                let mut max_corr = 0.0_f64;
                for lag in 0..=max_lag {
                    if lag >= n {
                        break;
                    }
                    let xi = &series[i][lag..];
                    let xj = &series[j][..n - lag];
                    let c = pearson_corr(xi, xj).abs();
                    if c > max_corr {
                        max_corr = c;
                    }
                }
                matrix[i][j] = max_corr;
            }
        }
        Ok(matrix)
    }

    /// Compute AUC-ROC for edge detection given predicted p-values and true graph.
    ///
    /// Lower p-values → more likely causal. Thresholds sweep [0, 1].
    pub fn auroc_edges(
        predicted_pvalues: &[Vec<f64>],
        true_graph: &CdtsCausalGraph,
    ) -> CdtsResult<f64> {
        let k = true_graph.n_vars;
        if predicted_pvalues.len() != k {
            return Err(CdtsError(
                "auroc_edges: p-value matrix dimension mismatch".into(),
            ));
        }
        let mut scores: Vec<(f64, bool)> = Vec::new();
        for i in 0..k {
            if predicted_pvalues[i].len() != k {
                return Err(CdtsError(format!(
                    "auroc_edges: p-value row {} wrong length",
                    i
                )));
            }
            for j in 0..k {
                if i == j {
                    continue;
                }
                scores.push((1.0 - predicted_pvalues[i][j], true_graph.edges[i][j]));
            }
        }
        // Sort by score descending
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_pos = scores.iter().filter(|&&(_, label)| label).count();
        let n_neg = scores.len() - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return Ok(0.5);
        }

        let mut auc = 0.0_f64;
        let mut tp_count = 0usize;
        let mut fp_count = 0usize;
        let mut prev_fp = 0usize;
        let mut prev_tp = 0usize;

        for &(_, label) in &scores {
            if label {
                tp_count += 1;
            } else {
                fp_count += 1;
            }
            // Trapezoid rule
            auc += (fp_count - prev_fp) as f64 * (tp_count + prev_tp) as f64 / 2.0;
            prev_fp = fp_count;
            prev_tp = tp_count;
        }
        Ok(auc / (n_pos as f64 * n_neg as f64))
    }

    /// Compute time-lagged partial autocorrelation for a single series (Yule-Walker method).
    pub fn partial_autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
        let n = series.len();
        if n < 2 {
            return vec![];
        }
        let mut pacf = Vec::with_capacity(max_lag);
        let m = mean(series);
        let centered: Vec<f64> = series.iter().map(|&x| x - m).collect();

        // Durbin-Levinson recursion
        let mut phi = vec![vec![0.0_f64; max_lag + 1]; max_lag + 1];
        let mut acf = vec![0.0_f64; max_lag + 1];
        let var = centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        acf[0] = var;
        for k in 1..=max_lag {
            if n <= k {
                break;
            }
            let c: f64 = (0..n - k)
                .map(|t| centered[t] * centered[t + k])
                .sum::<f64>()
                / n as f64;
            acf[k] = c;
        }
        // PACF via Levinson recursion
        for k in 1..=max_lag {
            if k >= n {
                break;
            }
            let mut num = acf[k];
            let mut den = acf[0];
            if k > 1 {
                let prev_phi: Vec<f64> = (1..k).map(|j| phi[k - 1][j]).collect();
                for (j, &pj) in prev_phi.iter().enumerate().map(|(i, v)| (i + 1, v)) {
                    num -= pj * acf[k - j];
                    den -= pj * acf[j];
                }
            }
            phi[k][k] = if den.abs() > 1e-15 { num / den } else { 0.0 };
            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }
            pacf.push(phi[k][k]);
        }
        pacf
    }
}

/// Comprehensive report of CDTS causal discovery analysis.
#[derive(Debug, Clone)]
pub struct CdtsReport {
    /// Number of variables analyzed.
    pub n_vars: usize,
    /// Number of time steps.
    pub n_time: usize,
    /// Maximum lag considered.
    pub max_lag: usize,
    /// Granger causality results (K x K).
    pub granger_matrix: Option<Vec<Vec<CdtsGrangerResult>>>,
    /// Transfer entropy matrix (K x K values).
    pub transfer_entropy_matrix: Option<Vec<Vec<f64>>>,
    /// PCMCI significant links.
    pub pcmci_links: Option<Vec<CdtsCausalLink>>,
    /// LiNGAM causal ordering.
    pub lingam_order: Option<Vec<usize>>,
    /// Time-lagged correlation matrix.
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
}

impl CdtsReport {
    /// Run full causal discovery pipeline on multivariate time series.
    pub fn analyze(data: &[Vec<f64>], max_lag: usize, alpha: f64) -> CdtsResult<Self> {
        if data.is_empty() {
            return Err(CdtsError("CdtsReport: empty data".into()));
        }
        let n_time = data.len();
        let n_vars = data[0].len();

        let granger_matrix = if n_vars >= 2 {
            Some(CdtsGrangerTest::pairwise_test(data, max_lag, alpha)?)
        } else {
            None
        };

        let correlation_matrix = Some(CdtsMetrics::lagged_correlation_matrix(data, max_lag)?);

        let pcmci_links = if n_vars >= 2 && n_time > 2 * max_lag {
            let config = CdtsPcmciConfig {
                max_lag,
                alpha,
                pc_alpha: alpha,
            };
            let result = CdtsPcmci::new(config).discover(data)?;
            Some(
                result
                    .links
                    .into_iter()
                    .filter(|l| l.is_significant)
                    .collect(),
            )
        } else {
            None
        };

        let lingam_order = if n_vars >= 2 {
            let config = CdtsLingamTsConfig {
                n_lags: max_lag.min(3),
                ..Default::default()
            };
            let result = CdtsLingamTs::new(config).identify_causal_order(data)?;
            Some(result.causal_order)
        } else {
            None
        };

        Ok(Self {
            n_vars,
            n_time,
            max_lag,
            granger_matrix,
            transfer_entropy_matrix: None,
            pcmci_links,
            lingam_order,
            correlation_matrix,
        })
    }

    /// Compute transfer entropy matrix (expensive; call separately).
    pub fn with_transfer_entropy(mut self, data: &[Vec<f64>]) -> CdtsResult<Self> {
        if data.is_empty() {
            return Ok(self);
        }
        let k = data[0].len();
        let n = data.len();
        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
        for t in 0..n {
            for var in 0..k {
                series[var].push(data[t][var]);
            }
        }

        let te_config = CdtsTransferEntropyConfig::default();
        let estimator = CdtsTransferEntropy::new(te_config);
        let mut te_matrix = vec![vec![0.0_f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    te_matrix[i][j] = estimator.compute(&series[i], &series[j]).unwrap_or(0.0);
                }
            }
        }
        self.transfer_entropy_matrix = Some(te_matrix);
        Ok(self)
    }
}

