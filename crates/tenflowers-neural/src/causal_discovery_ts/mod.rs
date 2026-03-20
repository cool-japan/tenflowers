//! Causal Discovery for Time Series
//!
//! Implements time-series-specific causal discovery algorithms:
//!
//! - **CdtsVarModel**: Vector Autoregression VAR(p) with OLS estimation and multi-step forecasting.
//! - **CdtsGrangerTest**: Granger causality via F-test (restricted vs unrestricted AR).
//! - **CdtsTransferEntropy**: Schreiber (2000) transfer entropy via binned histograms.
//! - **CdtsConvergentCC**: Convergent Cross Mapping (Sugihara 2012) for nonlinear systems.
//! - **CdtsPcmci**: PCMCI algorithm (Runge 2019) — PC + MCI conditional independence.
//! - **CdtsLinguisticTS**: Time-series LiNGAM (TiMINo) via non-Gaussianity identification.
//! - **CdtsInterventionEffect**: Interrupted time series & synthetic control causal effects.
//! - **CdtsMetrics**: Evaluation metrics (SHD, precision/recall/F1, correlation matrix).

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

/// Error type for causal discovery time series operations.
#[derive(Debug, Clone)]
pub struct CdtsError(pub String);

impl std::fmt::Display for CdtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CdtsError: {}", self.0)
    }
}

impl std::error::Error for CdtsError {}

pub type CdtsResult<T> = Result<T, CdtsError>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal math utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Normal CDF via Abramowitz & Stegun 26.2.16 rational approximation.
fn normal_cdf(x: f64) -> f64 {
    const P: f64 = 0.2316419;
    const B: [f64; 5] = [
        0.319381530,
        -0.356563782,
        1.781477937,
        -1.821255978,
        1.330274429,
    ];
    let t = 1.0 / (1.0 + P * x.abs());
    let poly = t * (B[0] + t * (B[1] + t * (B[2] + t * (B[3] + t * B[4]))));
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if x >= 0.0 {
        1.0 - phi * poly
    } else {
        phi * poly
    }
}

/// F-distribution CDF approximation via Wilson-Hilferty cube-root normal transform.
/// Returns P(F_{d1,d2} <= x).
fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Use beta distribution regularized incomplete via normal approx for large dof
    // Pearson approximation for F -> chi2/d1 -> normal
    let u = x * d1 / (x * d1 + d2);
    // Wilson-Hilferty: Beta(d1/2, d2/2) ~ normal
    let a = d1 / 2.0;
    let b = d2 / 2.0;
    let mu_b = a / (a + b);
    let var_b = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));
    let std_b = var_b.sqrt().max(1e-15);
    // cube-root transform on beta variable u
    let mu_cb = mu_b * (1.0 - var_b / (mu_b * mu_b));
    let z = (u.powf(1.0 / 3.0) - mu_cb) / std_b;
    normal_cdf(z)
}

/// F-distribution survival function (p-value for right tail): P(F > x).
fn f_pvalue(x: f64, d1: f64, d2: f64) -> f64 {
    1.0 - f_cdf(x, d1, d2)
}

/// Ordinary least squares via Cholesky-decomposition of normal equations.
/// Design matrix `xmat` is n x k; response `y` is length n.
fn ols_cholesky(xmat: &[Vec<f64>], y: &[f64]) -> CdtsResult<Vec<f64>> {
    let n = xmat.len();
    if n == 0 {
        return Err(CdtsError("OLS: empty data".into()));
    }
    if y.len() != n {
        return Err(CdtsError(format!("OLS: row mismatch {} vs {}", n, y.len())));
    }
    let k = xmat[0].len();
    if k == 0 {
        return Err(CdtsError("OLS: zero columns".into()));
    }

    // Accumulate X'X and X'y
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];
    for i in 0..n {
        for j in 0..k {
            xty[j] += xmat[i][j] * y[i];
            for l in 0..k {
                xtx[j][l] += xmat[i][j] * xmat[i][l];
            }
        }
    }
    // Ridge regularization for numerical stability
    for j in 0..k {
        xtx[j][j] += 1e-10;
    }
    cholesky_solve(&xtx, &xty)
}

/// Solve symmetric positive definite system A x = b via Cholesky decomposition.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> CdtsResult<Vec<f64>> {
    let k = b.len();
    if a.len() != k {
        return Err(CdtsError("cholesky_solve: dimension mismatch".into()));
    }
    // L L' = A
    let mut l = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i][j];
            for m in 0..j {
                s -= l[i][m] * l[j][m];
            }
            if i == j {
                s = s.max(1e-14);
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j].max(1e-14);
            }
        }
    }
    // Forward substitution L z = b
    let mut z = vec![0.0_f64; k];
    for i in 0..k {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * z[j];
        }
        z[i] = s / l[i][i].max(1e-14);
    }
    // Back substitution L' x = z
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut s = z[i];
        for j in (i + 1)..k {
            s -= l[j][i] * x[j];
        }
        x[i] = s / l[i][i].max(1e-14);
    }
    Ok(x)
}

/// Compute residuals: e_i = y_i - xmat[i] · beta.
fn compute_residuals(xmat: &[Vec<f64>], y: &[f64], beta: &[f64]) -> Vec<f64> {
    xmat.iter()
        .zip(y.iter())
        .map(|(row, &yi)| {
            let yhat: f64 = row.iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum();
            yi - yhat
        })
        .collect()
}

/// Residual sum of squares.
fn rss(resid: &[f64]) -> f64 {
    resid.iter().map(|&r| r * r).sum()
}

/// Mean of a slice.
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

/// Variance of a slice (sample variance, ddof=1).
fn variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = mean(x);
    x.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}

/// Standard deviation (sample).
fn std_dev(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

/// Pearson correlation coefficient.
fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let mx = mean(x);
    let my = mean(y);
    let num: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum();
    let dx: f64 = x.iter().map(|&xi| (xi - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = y.iter().map(|&yi| (yi - my).powi(2)).sum::<f64>().sqrt();
    let denom = dx * dy;
    if denom < 1e-15 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

/// Fisher Z-transform of a correlation r.
fn fisher_z(r: f64) -> f64 {
    let rc = r.clamp(-0.9999, 0.9999);
    0.5 * ((1.0 + rc) / (1.0 - rc)).ln()
}

/// Entropy of a discrete distribution given counts.
fn entropy_from_counts(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let tot_f = total as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / tot_f;
            -p * p.ln()
        })
        .sum()
}

/// Log-sum of natural logs, i.e. ln(sum(x_i)) computed as ln(sum(exp(ln_x_i))).
fn log_sum_exp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return f64::NEG_INFINITY;
    }
    max + vals.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1. CdtsVarModel — Vector Autoregression VAR(p)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for VAR model fitting.
#[derive(Debug, Clone)]
pub struct CdtsVarConfig {
    /// Number of lags p.
    pub n_lags: usize,
    /// Whether to include an intercept term.
    pub include_intercept: bool,
    /// Ridge regularization strength for OLS (0 = pure OLS).
    pub ridge_lambda: f64,
}

impl Default for CdtsVarConfig {
    fn default() -> Self {
        Self {
            n_lags: 2,
            include_intercept: true,
            ridge_lambda: 0.0,
        }
    }
}

/// Fitted VAR(p) model: Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + c + ε_t.
///
/// Coefficient matrices A_k are stored as `coefficients[k][eq][var]`:
/// - `k` = lag index 0..p
/// - `eq` = equation index 0..K (which output variable)
/// - `var` = input variable index 0..K
#[derive(Debug, Clone)]
pub struct CdtsVarModel {
    /// Model configuration.
    pub config: CdtsVarConfig,
    /// Number of variables K.
    pub n_vars: usize,
    /// Coefficient matrices \[lag\]\[eq\]\[var\], shape p x K x K.
    pub coefficients: Vec<Vec<Vec<f64>>>,
    /// Intercept terms \[eq\], length K.
    pub intercepts: Vec<f64>,
    /// Training residuals, shape (T - p) x K.
    pub residuals: Vec<Vec<f64>>,
    /// Last p observations for forecasting, shape p x K.
    pub last_window: Vec<Vec<f64>>,
}

impl CdtsVarModel {
    /// Fit VAR(p) model on multivariate time series data.
    ///
    /// `data[t][k]` = value at time t for variable k.
    /// Returns fitted model.
    pub fn fit(data: &[Vec<f64>], config: CdtsVarConfig) -> CdtsResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err(CdtsError("VarModel: empty data".into()));
        }
        let k = data[0].len();
        if k == 0 {
            return Err(CdtsError("VarModel: zero variables".into()));
        }
        let p = config.n_lags;
        if n <= p {
            return Err(CdtsError(format!(
                "VarModel: n={} must exceed n_lags={}",
                n, p
            )));
        }

        let n_eff = n - p;
        let intercept_cols = if config.include_intercept { 1 } else { 0 };
        let n_regressors = intercept_cols + p * k;

        // Build design matrix X of shape n_eff x n_regressors
        let mut xmat: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row = Vec::with_capacity(n_regressors);
            if config.include_intercept {
                row.push(1.0);
            }
            for lag in 1..=p {
                for var in 0..k {
                    row.push(data[t - lag][var]);
                }
            }
            xmat.push(row);
        }

        // Apply ridge to XtX if lambda > 0
        let mut coefficients = vec![vec![vec![0.0_f64; k]; k]; p];
        let mut intercepts = vec![0.0_f64; k];
        let mut residuals = vec![vec![0.0_f64; k]; n_eff];

        for eq in 0..k {
            let y_eq: Vec<f64> = (p..n).map(|t| data[t][eq]).collect();
            let mut xmat_eq = xmat.clone();
            if config.ridge_lambda > 0.0 {
                // Augment design matrix with ridge rows
                for j in 0..n_regressors {
                    let mut aug = vec![0.0_f64; n_regressors];
                    aug[j] = (config.ridge_lambda * n_eff as f64).sqrt();
                    xmat_eq.push(aug);
                }
                let mut y_aug = y_eq.clone();
                y_aug.resize(y_eq.len() + n_regressors, 0.0);
                let beta = ols_cholesky(&xmat_eq, &y_aug)?;
                let resid = compute_residuals(&xmat, &y_eq, &beta);
                // Unpack beta
                let mut col = 0;
                if config.include_intercept {
                    intercepts[eq] = beta[0];
                    col += 1;
                }
                for lag in 0..p {
                    for var in 0..k {
                        coefficients[lag][eq][var] = beta[col];
                        col += 1;
                    }
                }
                for (t_idx, r) in resid.into_iter().enumerate() {
                    residuals[t_idx][eq] = r;
                }
            } else {
                let beta = ols_cholesky(&xmat, &y_eq)?;
                let resid = compute_residuals(&xmat, &y_eq, &beta);
                let mut col = 0;
                if config.include_intercept {
                    intercepts[eq] = beta[0];
                    col += 1;
                }
                for lag in 0..p {
                    for var in 0..k {
                        coefficients[lag][eq][var] = beta[col];
                        col += 1;
                    }
                }
                for (t_idx, r) in resid.into_iter().enumerate() {
                    residuals[t_idx][eq] = r;
                }
            }
        }

        let last_window = data[(n - p)..n].to_vec();

        Ok(Self {
            config,
            n_vars: k,
            coefficients,
            intercepts,
            residuals,
            last_window,
        })
    }

    /// Produce h-step-ahead point forecasts.
    ///
    /// `history` must contain at least `n_lags` observations; the last `n_lags` are used.
    pub fn forecast(&self, history: &[Vec<f64>], h_steps: usize) -> CdtsResult<Vec<Vec<f64>>> {
        let p = self.config.n_lags;
        if history.len() < p {
            return Err(CdtsError(format!(
                "VarModel::forecast: need >= {} rows, got {}",
                p,
                history.len()
            )));
        }
        let k = self.n_vars;
        let mut window: Vec<Vec<f64>> = history[(history.len() - p)..].to_vec();
        let mut forecasts = Vec::with_capacity(h_steps);

        for _ in 0..h_steps {
            let mut y_next = self.intercepts.clone();
            for lag in 0..p {
                let t_lag = window.len() - 1 - lag;
                for eq in 0..k {
                    for var in 0..k {
                        y_next[eq] += self.coefficients[lag][eq][var] * window[t_lag][var];
                    }
                }
            }
            forecasts.push(y_next.clone());
            window.push(y_next);
            if window.len() > p {
                window.remove(0);
            }
        }
        Ok(forecasts)
    }

    /// Return training residuals (T - p rows, K columns).
    pub fn get_residuals(&self) -> &Vec<Vec<f64>> {
        &self.residuals
    }

    /// Compute the coefficient matrix for lag `lag_index` (1-based).
    /// Returns K x K matrix where entry \[eq\]\[var\] is the coefficient.
    pub fn lag_matrix(&self, lag_index: usize) -> CdtsResult<Vec<Vec<f64>>> {
        if lag_index == 0 || lag_index > self.config.n_lags {
            return Err(CdtsError(format!(
                "lag_index must be in 1..={}",
                self.config.n_lags
            )));
        }
        Ok(self.coefficients[lag_index - 1].clone())
    }

    /// Granger causality check: does variable `src` cause variable `tgt` via F-test on VAR coefficients?
    ///
    /// Returns (F_stat, p_value, is_causal).
    pub fn granger_from_coefficients(
        &self,
        src: usize,
        tgt: usize,
        alpha: f64,
        data: &[Vec<f64>],
    ) -> CdtsResult<(f64, f64, bool)> {
        let k = self.n_vars;
        let p = self.config.n_lags;
        if src >= k || tgt >= k {
            return Err(CdtsError(
                "granger_from_coefficients: variable index out of range".into(),
            ));
        }
        let n = data.len();
        let n_eff = n - p;
        // Collect restricted residuals (exclude src lags from tgt equation)
        let intercept_cols = if self.config.include_intercept { 1 } else { 0 };
        let mut xmat_full: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        let mut xmat_restr: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row_full = Vec::new();
            let mut row_restr = Vec::new();
            if self.config.include_intercept {
                row_full.push(1.0);
                row_restr.push(1.0);
            }
            for lag in 1..=p {
                for var in 0..k {
                    let val = data[t - lag][var];
                    row_full.push(val);
                    if var != src {
                        row_restr.push(val);
                    }
                }
            }
            xmat_full.push(row_full);
            xmat_restr.push(row_restr);
        }
        let y_tgt: Vec<f64> = (p..n).map(|t| data[t][tgt]).collect();
        let beta_full = ols_cholesky(&xmat_full, &y_tgt)?;
        let beta_restr = ols_cholesky(&xmat_restr, &y_tgt)?;
        let resid_full = compute_residuals(&xmat_full, &y_tgt, &beta_full);
        let resid_restr = compute_residuals(&xmat_restr, &y_tgt, &beta_restr);
        let rss_u = rss(&resid_full);
        let rss_r = rss(&resid_restr);
        let q = p as f64; // number of restrictions (p lags of src)
        let df_u = n_eff as f64 - (intercept_cols + p * k) as f64;
        let df_u = df_u.max(1.0);
        let f_stat = ((rss_r - rss_u) / q) / (rss_u / df_u).max(1e-15);
        let f_stat = f_stat.max(0.0);
        let p_val = f_pvalue(f_stat, q, df_u);
        Ok((f_stat, p_val, p_val < alpha))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2. CdtsGrangerTest — Granger causality F-test
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a Granger causality test.
#[derive(Debug, Clone)]
pub struct CdtsGrangerResult {
    /// F-statistic from incremental F-test.
    pub f_stat: f64,
    /// Two-sided p-value via F-distribution CDF approximation.
    pub p_value: f64,
    /// Whether X Granger-causes Y at the tested significance level.
    pub is_causal: bool,
    /// Number of lags used.
    pub n_lags: usize,
    /// Number of effective observations (T - max_lag).
    pub n_obs: usize,
    /// RSS of restricted model.
    pub rss_restricted: f64,
    /// RSS of unrestricted model.
    pub rss_unrestricted: f64,
}

/// Granger causality test: does X Granger-cause Y?
///
/// Restricted model: Y_t ~ Y_{t-1}, ..., Y_{t-p}
/// Unrestricted model: Y_t ~ Y_{t-1}, ..., Y_{t-p}, X_{t-1}, ..., X_{t-p}
/// F = ((RSS_r - RSS_u)/q) / (RSS_u/(T-k))
pub struct CdtsGrangerTest;

impl CdtsGrangerTest {
    /// Test whether `x_series` Granger-causes `y_series` using `max_lag` lags at significance `alpha`.
    pub fn test(
        x_series: &[f64],
        y_series: &[f64],
        max_lag: usize,
        alpha: f64,
    ) -> CdtsResult<CdtsGrangerResult> {
        let n = y_series.len();
        if x_series.len() != n {
            return Err(CdtsError(format!(
                "GrangerTest: length mismatch x={} y={}",
                x_series.len(),
                n
            )));
        }
        if max_lag == 0 {
            return Err(CdtsError("GrangerTest: max_lag must be >= 1".into()));
        }
        if n <= max_lag + 1 {
            return Err(CdtsError(format!(
                "GrangerTest: need n > max_lag+1, got n={} max_lag={}",
                n, max_lag
            )));
        }
        let p = max_lag;
        let n_eff = n - p;

        // Restricted: intercept + p lags of Y
        let mut xmat_r: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        // Unrestricted: intercept + p lags of Y + p lags of X
        let mut xmat_u: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        let mut y_vec: Vec<f64> = Vec::with_capacity(n_eff);

        for t in p..n {
            let mut row_r = vec![1.0_f64];
            let mut row_u = vec![1.0_f64];
            for lag in 1..=p {
                row_r.push(y_series[t - lag]);
                row_u.push(y_series[t - lag]);
            }
            for lag in 1..=p {
                row_u.push(x_series[t - lag]);
            }
            xmat_r.push(row_r);
            xmat_u.push(row_u);
            y_vec.push(y_series[t]);
        }

        let beta_r = ols_cholesky(&xmat_r, &y_vec)?;
        let beta_u = ols_cholesky(&xmat_u, &y_vec)?;
        let resid_r = compute_residuals(&xmat_r, &y_vec, &beta_r);
        let resid_u = compute_residuals(&xmat_u, &y_vec, &beta_u);
        let rss_r = rss(&resid_r);
        let rss_u = rss(&resid_u);

        // q = number of restrictions = p lags of X
        let q = p as f64;
        // Degrees of freedom for unrestricted: T - k, k = 1 + 2p (intercept + p Y lags + p X lags)
        let k_u = 1.0 + 2.0 * p as f64;
        let df_u = (n_eff as f64 - k_u).max(1.0);

        let f_stat = ((rss_r - rss_u) / q) / (rss_u / df_u).max(1e-15);
        let f_stat = f_stat.max(0.0);
        let p_value = f_pvalue(f_stat, q, df_u);

        Ok(CdtsGrangerResult {
            f_stat,
            p_value,
            is_causal: p_value < alpha,
            n_lags: p,
            n_obs: n_eff,
            rss_restricted: rss_r,
            rss_unrestricted: rss_u,
        })
    }

    /// Compute Granger causality for all pairs in a multivariate time series.
    ///
    /// Returns a K x K matrix of (F_stat, p_value, is_causal) where entry \[i\]\[j\]
    /// indicates whether variable i Granger-causes variable j.
    pub fn pairwise_test(
        data: &[Vec<f64>],
        max_lag: usize,
        alpha: f64,
    ) -> CdtsResult<Vec<Vec<CdtsGrangerResult>>> {
        if data.is_empty() {
            return Err(CdtsError("pairwise_test: empty data".into()));
        }
        let k = data[0].len();
        if k < 2 {
            return Err(CdtsError("pairwise_test: need >= 2 variables".into()));
        }

        // Transpose: series[var][t]
        let n = data.len();
        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
        for t in 0..n {
            for var in 0..k {
                series[var].push(data[t][var]);
            }
        }

        let mut results = vec![vec![]; k];
        for i in 0..k {
            results[i] = Vec::with_capacity(k);
            for j in 0..k {
                if i == j {
                    // Self-causality: trivial
                    results[i].push(CdtsGrangerResult {
                        f_stat: 0.0,
                        p_value: 1.0,
                        is_causal: false,
                        n_lags: max_lag,
                        n_obs: n - max_lag,
                        rss_restricted: 0.0,
                        rss_unrestricted: 0.0,
                    });
                } else {
                    let r = Self::test(&series[i], &series[j], max_lag, alpha)?;
                    results[i].push(r);
                }
            }
        }
        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3. CdtsTransferEntropy — Transfer Entropy (Schreiber 2000)
// ─────────────────────────────────────────────────────────────────────────────

/// Transfer entropy estimator configuration.
#[derive(Debug, Clone)]
pub struct CdtsTransferEntropyConfig {
    /// Number of bins for histogram discretization.
    pub n_bins: usize,
    /// Source lag k (how many past values of X to use).
    pub k_lag: usize,
    /// Target lag l (how many past values of Y to use).
    pub l_lag: usize,
    /// Whether to compute normalized TE.
    pub normalize: bool,
}

impl Default for CdtsTransferEntropyConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            k_lag: 1,
            l_lag: 1,
            normalize: false,
        }
    }
}

/// Transfer entropy estimator via binned histograms.
///
/// TE(X→Y) = I(Y_future ; X_past | Y_past)
///          = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
///          = H(Y_future, Y_past) + H(Y_past, X_past) - H(Y_future, Y_past, X_past) - H(Y_past)
pub struct CdtsTransferEntropy {
    pub config: CdtsTransferEntropyConfig,
}

impl CdtsTransferEntropy {
    pub fn new(config: CdtsTransferEntropyConfig) -> Self {
        Self { config }
    }

    /// Bin a continuous series into `n_bins` equal-width bins based on min/max.
    fn bin_series(series: &[f64], n_bins: usize) -> Vec<usize> {
        if series.is_empty() {
            return vec![];
        }
        let min = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-15);
        series
            .iter()
            .map(|&v| {
                let b = ((v - min) / range * n_bins as f64) as usize;
                b.min(n_bins - 1)
            })
            .collect()
    }

    /// Marginal entropy H(X) from binned data.
    fn marginal_entropy(binned: &[usize], n_bins: usize) -> f64 {
        let mut counts = vec![0usize; n_bins];
        for &b in binned {
            counts[b] += 1;
        }
        entropy_from_counts(&counts)
    }

    /// Joint entropy H(X, Y) for two binned series (same length).
    fn joint_entropy_2(binned_x: &[usize], binned_y: &[usize], n_bins: usize) -> f64 {
        let mut counts = vec![0usize; n_bins * n_bins];
        for (&bx, &by) in binned_x.iter().zip(binned_y.iter()) {
            counts[bx * n_bins + by] += 1;
        }
        entropy_from_counts(&counts)
    }

    /// Joint entropy H(X, Y, Z) for three binned series.
    fn joint_entropy_3(
        binned_x: &[usize],
        binned_y: &[usize],
        binned_z: &[usize],
        n_bins: usize,
    ) -> f64 {
        let nb2 = n_bins * n_bins;
        let mut counts = vec![0usize; n_bins * n_bins * n_bins];
        for ((&bx, &by), &bz) in binned_x.iter().zip(binned_y.iter()).zip(binned_z.iter()) {
            counts[bx * nb2 + by * n_bins + bz] += 1;
        }
        entropy_from_counts(&counts)
    }

    /// Compute TE(X→Y).
    ///
    /// Uses the formula: TE = H(Yf, Yp) + H(Yp, Xp) - H(Yf, Yp, Xp) - H(Yp)
    /// where Yf = Y_{t+1}, Yp = Y_t (past l lags), Xp = X_t (past k lags).
    pub fn compute(&self, x: &[f64], y: &[f64]) -> CdtsResult<f64> {
        let n = y.len();
        if x.len() != n {
            return Err(CdtsError(format!(
                "TransferEntropy: length mismatch x={} y={}",
                x.len(),
                n
            )));
        }
        let max_lag = self.config.k_lag.max(self.config.l_lag);
        if n <= max_lag + 1 {
            return Err(CdtsError(format!(
                "TransferEntropy: too few observations (n={}, need > {})",
                n,
                max_lag + 1
            )));
        }
        let nb = self.config.n_bins;
        let bx = Self::bin_series(x, nb);
        let by = Self::bin_series(y, nb);

        // Build aligned vectors: Yf = y[t], Yp = y[t-1], Xp = x[t-1]
        // Using k_lag=l_lag=1 for simplicity; generalize to first lag
        let k = self.config.k_lag;
        let l = self.config.l_lag;
        let start = k.max(l);
        let t_count = n - start - 1;
        if t_count == 0 {
            return Err(CdtsError(
                "TransferEntropy: insufficient effective samples".into(),
            ));
        }

        let yf: Vec<usize> = (start..n - 1).map(|t| by[t + 1]).collect();
        let yp: Vec<usize> = (start..n - 1).map(|t| by[t]).collect();
        let xp: Vec<usize> = (start..n - 1).map(|t| bx[t]).collect();

        // TE = H(Yf, Yp) + H(Yp, Xp) - H(Yf, Yp, Xp) - H(Yp)
        let h_yf_yp = Self::joint_entropy_2(&yf, &yp, nb);
        let h_yp_xp = Self::joint_entropy_2(&yp, &xp, nb);
        let h_yf_yp_xp = Self::joint_entropy_3(&yf, &yp, &xp, nb);
        let h_yp = Self::marginal_entropy(&yp, nb);

        let te = h_yf_yp + h_yp_xp - h_yf_yp_xp - h_yp;
        let te = te.max(0.0);

        if self.config.normalize {
            // Normalized TE = TE / H(Yf | Yp)
            let h_cond = h_yf_yp - h_yp;
            let h_cond = h_cond.max(1e-15);
            Ok(te / h_cond)
        } else {
            Ok(te)
        }
    }

    /// Test significance via permutation (shuffle X, recompute TE).
    pub fn significance_test(
        &self,
        x: &[f64],
        y: &[f64],
        n_permutations: usize,
        seed: u64,
    ) -> CdtsResult<(f64, f64)> {
        let te_obs = self.compute(x, y)?;
        let n = x.len();
        let mut rng_state = seed;
        let mut count_exceeding = 0usize;

        for _ in 0..n_permutations {
            // Simple LCG shuffle
            let mut x_perm = x.to_vec();
            for i in (1..n).rev() {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                x_perm.swap(i, j);
            }
            let te_perm = self.compute(&x_perm, y)?;
            if te_perm >= te_obs {
                count_exceeding += 1;
            }
        }
        let p_value = (count_exceeding + 1) as f64 / (n_permutations + 1) as f64;
        Ok((te_obs, p_value))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4. CdtsConvergentCC — Convergent Cross Mapping (Sugihara 2012)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CCM.
#[derive(Debug, Clone)]
pub struct CdtsCcmConfig {
    /// Embedding dimension E.
    pub embedding_dim: usize,
    /// Time delay tau.
    pub tau: usize,
    /// Library sizes to test (number of points used for manifold reconstruction).
    pub lib_sizes: Vec<usize>,
    /// Number of random subsamples per library size.
    pub n_samples: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for CdtsCcmConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 3,
            tau: 1,
            lib_sizes: vec![10, 25, 50, 100],
            n_samples: 10,
            seed: 42,
        }
    }
}

/// Cross-map skill result for one library size.
#[derive(Debug, Clone)]
pub struct CdtsCcmPoint {
    pub lib_size: usize,
    pub skill_xy: f64, // correlation when predicting X from Y's manifold
    pub skill_yx: f64, // correlation when predicting Y from X's manifold
}

/// Convergent Cross Mapping result.
#[derive(Debug, Clone)]
pub struct CdtsCcmResult {
    /// Skill vs library size.
    pub skill_curve: Vec<CdtsCcmPoint>,
    /// Whether X causes Y (Y's manifold predicts X, skill converges).
    pub x_causes_y: bool,
    /// Whether Y causes X.
    pub y_causes_x: bool,
    /// Convergence score for X->Y: final skill - initial skill.
    pub convergence_xy: f64,
    /// Convergence score for Y->X.
    pub convergence_yx: f64,
}

/// Convergent Cross Mapping estimator for nonlinear causal detection.
pub struct CdtsConvergentCC {
    pub config: CdtsCcmConfig,
}

impl CdtsConvergentCC {
    pub fn new(config: CdtsCcmConfig) -> Self {
        Self { config }
    }

    /// Build time-delay embedding vectors for a univariate series.
    /// Returns vectors of length E, sampled at times [tau*(E-1) .. n-1].
    fn embed(series: &[f64], e: usize, tau: usize) -> Vec<Vec<f64>> {
        let n = series.len();
        let start = tau * (e - 1);
        if n <= start {
            return vec![];
        }
        (start..n)
            .map(|t| (0..e).map(|i| series[t - i * tau]).collect())
            .collect()
    }

    /// Euclidean distance between two embedding vectors.
    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Cross-map: use `shadow_manifold` to predict the values of `target_series`.
    ///
    /// For each point in the library manifold, find E+1 nearest neighbors,
    /// use exponential-distance weighted average to predict target.
    /// Returns Pearson correlation between predicted and actual target values.
    fn cross_map_skill(
        manifold: &[Vec<f64>],
        target: &[f64],
        lib_indices: &[usize],
        query_indices: &[usize],
        e: usize,
        offset: usize,
    ) -> f64 {
        let n_nn = e + 1;
        let mut predicted = Vec::with_capacity(query_indices.len());
        let mut actual = Vec::with_capacity(query_indices.len());

        for &qi in query_indices {
            let query = &manifold[qi];
            // Find n_nn nearest neighbors in lib_indices (excluding qi itself)
            let mut nn_dists: Vec<(f64, usize)> = lib_indices
                .iter()
                .filter(|&&li| li != qi)
                .map(|&li| (Self::dist(query, &manifold[li]), li))
                .collect();
            if nn_dists.len() < n_nn {
                continue;
            }
            nn_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let nn = &nn_dists[..n_nn];

            // Exponential weights
            let d0 = nn[0].0.max(1e-15);
            let weights: Vec<f64> = nn.iter().map(|(d, _)| (-d / d0).exp()).collect();
            let w_sum: f64 = weights.iter().sum();
            if w_sum < 1e-15 {
                continue;
            }

            // Weighted sum of target values at neighbor time points
            let mut pred = 0.0_f64;
            for (w, (_, li)) in weights.iter().zip(nn.iter()) {
                let t_idx = li + offset;
                if t_idx < target.len() {
                    pred += w * target[t_idx];
                }
            }
            pred /= w_sum;
            predicted.push(pred);
            let t_qi = qi + offset;
            if t_qi < target.len() {
                actual.push(target[t_qi]);
            }
        }

        if predicted.len() < 2 || actual.len() != predicted.len() {
            return 0.0;
        }
        pearson_corr(&predicted, &actual)
    }

    /// Run CCM analysis: compute cross-map skill for multiple library sizes.
    pub fn ccm(&self, x: &[f64], y: &[f64]) -> CdtsResult<CdtsCcmResult> {
        let n = x.len();
        if y.len() != n {
            return Err(CdtsError(format!(
                "CCM: length mismatch x={} y={}",
                n,
                y.len()
            )));
        }
        let e = self.config.embedding_dim;
        let tau = self.config.tau;
        let offset = tau * (e - 1);

        let mx = Self::embed(x, e, tau);
        let my = Self::embed(y, e, tau);
        let emb_len = mx.len().min(my.len());
        if emb_len < e + 2 {
            return Err(CdtsError(format!(
                "CCM: too few embedded points ({}) for e={}",
                emb_len, e
            )));
        }

        let mx = &mx[..emb_len];
        let my = &my[..emb_len];
        let all_indices: Vec<usize> = (0..emb_len).collect();

        let mut rng_state = self.config.seed;
        let mut skill_curve = Vec::new();

        for &lib_size in &self.config.lib_sizes {
            let lib_size = lib_size.min(emb_len);
            if lib_size < e + 2 {
                continue;
            }

            let mut sum_xy = 0.0_f64;
            let mut sum_yx = 0.0_f64;
            let mut valid_count = 0usize;

            for _ in 0..self.config.n_samples {
                // Randomly select lib_size indices
                let mut indices = all_indices.clone();
                // LCG shuffle then take first lib_size
                for i in (1..emb_len).rev() {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = (rng_state >> 33) as usize % (i + 1);
                    indices.swap(i, j);
                }
                let lib_idx = indices[..lib_size].to_vec();
                let query_idx = indices[..lib_size.min(20)].to_vec();

                let sk_xy = Self::cross_map_skill(my, x, &lib_idx, &query_idx, e, offset);
                let sk_yx = Self::cross_map_skill(mx, y, &lib_idx, &query_idx, e, offset);
                sum_xy += sk_xy;
                sum_yx += sk_yx;
                valid_count += 1;
            }

            if valid_count > 0 {
                skill_curve.push(CdtsCcmPoint {
                    lib_size,
                    skill_xy: sum_xy / valid_count as f64,
                    skill_yx: sum_yx / valid_count as f64,
                });
            }
        }

        if skill_curve.is_empty() {
            return Err(CdtsError("CCM: no valid skill estimates computed".into()));
        }

        // Convergence = last skill - first skill (should be positive for true causality)
        let first = &skill_curve[0];
        let last = &skill_curve[skill_curve.len() - 1];
        let convergence_xy = last.skill_xy - first.skill_xy;
        let convergence_yx = last.skill_yx - first.skill_yx;
        let threshold = 0.05;
        let x_causes_y = last.skill_xy > 0.1 && convergence_xy > threshold;
        let y_causes_x = last.skill_yx > 0.1 && convergence_yx > threshold;

        Ok(CdtsCcmResult {
            skill_curve,
            x_causes_y,
            y_causes_x,
            convergence_xy,
            convergence_yx,
        })
    }
}

pub mod extensions;
pub use extensions::*;

#[cfg(test)]
mod tests;
