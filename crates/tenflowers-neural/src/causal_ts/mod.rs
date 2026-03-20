//! Causal Time Series & Econometrics Neural Methods
//!
//! Granger causality, structural time series, intervention analysis,
//! neural forecasting (WaveNet, TCN, DeepSSM), and counterfactual methods.

pub mod extensions;
pub mod advanced;
pub use extensions::*;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

pub(crate) type CtsResult<T> = Result<T, String>;

#[inline]
pub(crate) fn sigmoid_f64(x: f64) -> f64 {
    let c = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-c).exp())
}

/// Chi-squared p-value via Wilson-Hilferty cube-root approximation.
pub(crate) fn chi2_p_value(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    let mu = 1.0 - 2.0 / (9.0 * df);
    let s2 = 2.0 / (9.0 * df);
    1.0 - normal_cdf_f64(((x / df).powf(1.0 / 3.0) - mu) / s2.sqrt())
}

/// Normal CDF (Abramowitz & Stegun 26.2.16).
pub(crate) fn normal_cdf_f64(x: f64) -> f64 {
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

/// Ordinary least squares: returns coefficients for design matrix X (n x k) and response y (n).
/// Uses normal equations: beta = (X'X)^{-1} X'y via Cholesky-like approach.
pub(crate) fn ols(x_mat: &[Vec<f64>], y: &[f64]) -> CtsResult<Vec<f64>> {
    let n = y.len();
    if x_mat.len() != n {
        return Err(format!("OLS: row mismatch {} vs {}", x_mat.len(), n));
    }
    if n == 0 {
        return Err("OLS: empty data".to_string());
    }
    let k = x_mat[0].len();
    // X'X  (k x k)
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];
    for i in 0..n {
        for j in 0..k {
            xty[j] += x_mat[i][j] * y[i];
            for l in 0..k {
                xtx[j][l] += x_mat[i][j] * x_mat[i][l];
            }
        }
    }
    // Solve via Cholesky (regularized)
    let eps = 1e-10;
    for i in 0..k {
        xtx[i][i] += eps;
    }
    cholesky_solve(&xtx, &xty)
}

/// Solve A x = b where A is symmetric positive definite (k x k) via Cholesky.
pub(crate) fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> CtsResult<Vec<f64>> {
    let k = b.len();
    // Cholesky decomposition: A = L L'
    let mut l = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut s: f64 = a[i][j];
            for m in 0..j {
                s -= l[i][m] * l[j][m];
            }
            if i == j {
                if s <= 0.0 {
                    s = 1e-12;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Forward substitution: L y = b
    let mut y_fwd = vec![0.0_f64; k];
    for i in 0..k {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * y_fwd[j];
        }
        y_fwd[i] = s / l[i][i];
    }
    // Back substitution: L' x = y
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut s = y_fwd[i];
        for j in (i + 1)..k {
            s -= l[j][i] * x[j];
        }
        x[i] = s / l[i][i];
    }
    Ok(x)
}

/// Compute residuals from design matrix and coefficients.
pub(crate) fn residuals(x_mat: &[Vec<f64>], y: &[f64], beta: &[f64]) -> Vec<f64> {
    x_mat
        .iter()
        .zip(y.iter())
        .map(|(row, &yi)| {
            let yhat: f64 = row.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
            yi - yhat
        })
        .collect()
}

/// Sum of squared residuals.
pub(crate) fn ssr(resid: &[f64]) -> f64 {
    resid.iter().map(|&r| r * r).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Granger Causality
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a Granger causality test.
#[derive(Debug, Clone)]
pub struct GrangerResult {
    /// F-statistic from the incremental F-test.
    pub f_statistic: f64,
    /// Approximate p-value (chi-squared approximation: T * F ~ chi2(p)).
    pub p_value_approx: f64,
    /// Number of lags used.
    pub lags: usize,
    /// Number of observations.
    pub n_obs: usize,
}

impl GrangerResult {
    /// Returns true if X Granger-causes Y at significance level `alpha`.
    pub fn is_causal(&self, alpha: f64) -> bool {
        self.p_value_approx < alpha
    }
}

/// Tests Granger causality: does X Granger-cause Y?
///
/// Compares restricted AR(p) on Y vs unrestricted model with lagged X.
pub struct GrangerCausalityTest;

impl GrangerCausalityTest {
    /// Test whether `x` Granger-causes `y` with `p` lags.
    pub fn test(x: &[f64], y: &[f64], p: usize) -> CtsResult<GrangerResult> {
        let n = y.len();
        if x.len() != n {
            return Err(format!("Granger: length mismatch x={} y={}", x.len(), n));
        }
        if n <= p + 1 {
            return Err(format!("Granger: need n > p+1, got n={} p={}", n, p));
        }
        let n_eff = n - p;
        // Build restricted design matrix (intercept + p lags of Y)
        let mut x_restr: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        let mut y_restr: Vec<f64> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row = vec![1.0_f64]; // intercept
            for lag in 1..=p {
                row.push(y[t - lag]);
            }
            x_restr.push(row);
            y_restr.push(y[t]);
        }
        let beta_r = ols(&x_restr, &y_restr)?;
        let resid_r = residuals(&x_restr, &y_restr, &beta_r);
        let ssr_r = ssr(&resid_r);

        // Build unrestricted design matrix (intercept + p lags of Y + p lags of X)
        let mut x_unres: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row = vec![1.0_f64]; // intercept
            for lag in 1..=p {
                row.push(y[t - lag]);
            }
            for lag in 1..=p {
                row.push(x[t - lag]);
            }
            x_unres.push(row);
        }
        let beta_u = ols(&x_unres, &y_restr)?;
        let resid_u = residuals(&x_unres, &y_restr, &beta_u);
        let ssr_u = ssr(&resid_u);

        // F = ((SSR_R - SSR_U) / p) / (SSR_U / (n_eff - 2p - 1))
        let df1 = p as f64;
        let df2 = (n_eff as f64) - 2.0 * (p as f64) - 1.0;
        if df2 <= 0.0 {
            return Err(format!("Granger: not enough df2={}", df2));
        }
        let f_stat = if ssr_u < 1e-15 {
            0.0
        } else {
            ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        };
        let f_stat = f_stat.max(0.0);
        // Chi-squared approximation: T * F ~ chi2(p)
        let _chi2_stat = (n_eff as f64) * f_stat / (df2 / df1 + f_stat).max(1e-15);
        let p_value = chi2_p_value(f_stat * df1, df1);

        Ok(GrangerResult {
            f_statistic: f_stat,
            p_value_approx: p_value,
            lags: p,
            n_obs: n_eff,
        })
    }
}

/// Vector Autoregression model VAR(p).
pub struct VectorAutoregression {
    /// Number of lags.
    pub p: usize,
    /// Number of variables (K).
    pub k: usize,
    /// Coefficient matrices A_1, ..., A_p each of shape K x K.
    /// Stored as Vec<Vec<`Vec<f64>>>` = \[lag\]\[eq\]\[var\].
    pub coefficients: Vec<Vec<Vec<f64>>>,
    /// Intercepts, length K.
    pub intercepts: Vec<f64>,
    /// Last p observations (for forecasting), shape p x K.
    pub last_obs: Vec<Vec<f64>>,
}

impl VectorAutoregression {
    /// Fit VAR(p) on multivariate data.
    /// `data[t][k]` = observation at time t for variable k.
    /// Returns (VectorAutoregression, residuals flat).
    pub fn fit(data: &[Vec<f64>], p: usize) -> CtsResult<(Self, Vec<f64>)> {
        let n = data.len();
        if n == 0 {
            return Err("VAR: empty data".to_string());
        }
        let k = data[0].len();
        if k == 0 {
            return Err("VAR: zero variables".to_string());
        }
        if n <= p {
            return Err(format!("VAR: need n > p, got n={} p={}", n, p));
        }
        let n_eff = n - p;
        // For each equation eq in 0..k, run OLS on [1, y_{t-1}, ..., y_{t-p}] (all K vars)
        let mut all_resid = Vec::new();
        let mut coefficients = vec![vec![vec![0.0_f64; k]; k]; p];
        let mut intercepts = vec![0.0_f64; k];

        // Build design matrix once: rows = t in p..n, cols = [1, data[t-1][0..k], ..., data[t-p][0..k]]
        let num_cols = 1 + p * k;
        let mut design: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row = Vec::with_capacity(num_cols);
            row.push(1.0);
            for lag in 1..=p {
                for var in 0..k {
                    row.push(data[t - lag][var]);
                }
            }
            design.push(row);
        }

        for eq in 0..k {
            let y_eq: Vec<f64> = (p..n).map(|t| data[t][eq]).collect();
            let beta = ols(&design, &y_eq)?;
            intercepts[eq] = beta[0];
            for lag in 0..p {
                for var in 0..k {
                    coefficients[lag][eq][var] = beta[1 + lag * k + var];
                }
            }
            let resid_eq = residuals(&design, &y_eq, &beta);
            all_resid.extend_from_slice(&resid_eq);
        }

        let last_obs: Vec<Vec<f64>> = (n - p..n).map(|t| data[t].clone()).collect();

        Ok((
            Self {
                p,
                k,
                coefficients,
                intercepts,
                last_obs,
            },
            all_resid,
        ))
    }

    /// Multi-step forecast.
    pub fn forecast(&self, steps: usize) -> Vec<Vec<f64>> {
        let mut history: Vec<Vec<f64>> = self.last_obs.clone();
        let mut forecasts = Vec::with_capacity(steps);
        for _ in 0..steps {
            let h = history.len();
            let mut next = self.intercepts.clone();
            for lag in 0..self.p {
                if lag < h {
                    let obs = &history[h - 1 - lag];
                    for eq in 0..self.k {
                        for var in 0..self.k {
                            next[eq] += self.coefficients[lag][eq][var] * obs[var];
                        }
                    }
                }
            }
            history.push(next.clone());
            forecasts.push(next);
        }
        forecasts
    }
}

/// Transfer Entropy Estimator: TE(X→Y) = I(Y_t; X_{t-1} | Y_{t-1}).
pub struct TransferEntropyEstimator {
    /// Number of bins for discretization.
    pub n_bins: usize,
}

impl TransferEntropyEstimator {
    /// Create with given number of discretization bins.
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins: n_bins.max(2),
        }
    }

    /// Compute TE(x → y) with given lag.
    pub fn compute(&self, x: &[f64], y: &[f64], lag: usize) -> f64 {
        let n = y.len().min(x.len());
        let lag = lag.max(1);
        if n <= lag + 1 {
            return 0.0;
        }
        let bins = self.n_bins;
        // Discretize x and y to [0, bins)
        let disc = |v: &[f64]| -> Vec<usize> {
            let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max - min).max(1e-15);
            v.iter()
                .map(|&vi| (((vi - min) / range) * (bins as f64 - 1.0)).round() as usize)
                .collect()
        };
        let xd = disc(x);
        let yd = disc(y);

        // Triplets: (y_t, y_{t-lag}, x_{t-lag})
        let t_start = lag;
        let n_obs = n - t_start;
        let mut count_yx_prev_x_prev = vec![vec![vec![0usize; bins]; bins]; bins];
        let mut count_y_yprev = vec![vec![0usize; bins]; bins];
        let mut count_yprev_xprev = vec![vec![0usize; bins]; bins];
        let mut count_yprev = vec![0usize; bins];

        for t in t_start..n {
            let yt = yd[t];
            let yt_prev = yd[t - lag];
            let xt_prev = xd[t - lag];
            count_yx_prev_x_prev[yt][yt_prev][xt_prev] += 1;
            count_y_yprev[yt][yt_prev] += 1;
            count_yprev_xprev[yt_prev][xt_prev] += 1;
            count_yprev[yt_prev] += 1;
        }

        let n_f = n_obs as f64;
        let mut te = 0.0_f64;
        for yt in 0..bins {
            for yt_prev in 0..bins {
                for xt_prev in 0..bins {
                    let p_joint = count_yx_prev_x_prev[yt][yt_prev][xt_prev] as f64 / n_f;
                    if p_joint < 1e-15 {
                        continue;
                    }
                    let p_y_yprev = count_y_yprev[yt][yt_prev] as f64 / n_f;
                    let p_yprev_xprev = count_yprev_xprev[yt_prev][xt_prev] as f64 / n_f;
                    let p_yprev = count_yprev[yt_prev] as f64 / n_f;
                    if p_y_yprev < 1e-15 || p_yprev_xprev < 1e-15 || p_yprev < 1e-15 {
                        continue;
                    }
                    te += p_joint * (p_joint * p_yprev / (p_y_yprev * p_yprev_xprev)).ln();
                }
            }
        }
        te.max(0.0)
    }
}

/// Convergent Cross Mapping for detecting causality in nonlinear dynamical systems.
pub struct ConvergentCrossMapping {
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Time delay for embedding.
    pub tau: usize,
}

impl ConvergentCrossMapping {
    /// Create CCM with embedding dimension and delay.
    pub fn new(embed_dim: usize, tau: usize) -> Self {
        Self {
            embed_dim: embed_dim.max(2),
            tau: tau.max(1),
        }
    }

    /// Reconstruct shadow manifold from `z` using delay embedding.
    fn embed(z: &[f64], dim: usize, tau: usize) -> Vec<Vec<f64>> {
        let n = z.len();
        let min_idx = (dim - 1) * tau;
        if n <= min_idx {
            return Vec::new();
        }
        (min_idx..n)
            .map(|t| (0..dim).map(|d| z[t - d * tau]).collect())
            .collect()
    }

    /// Compute correlation between x_pred and x_true.
    fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mean_a = a[..n].iter().sum::<f64>() / n as f64;
        let mean_b = b[..n].iter().sum::<f64>() / n as f64;
        let num: f64 = a[..n]
            .iter()
            .zip(b[..n].iter())
            .map(|(&ai, &bi)| (ai - mean_a) * (bi - mean_b))
            .sum();
        let den_a: f64 = a[..n]
            .iter()
            .map(|&ai| (ai - mean_a).powi(2))
            .sum::<f64>()
            .sqrt();
        let den_b: f64 = b[..n]
            .iter()
            .map(|&bi| (bi - mean_b).powi(2))
            .sum::<f64>()
            .sqrt();
        if den_a < 1e-15 || den_b < 1e-15 {
            return 0.0;
        }
        num / (den_a * den_b)
    }

    /// Test CCM: x → y (does x cause y?).
    /// Returns cross-map correlation for each library size.
    pub fn test_ccm(&self, x: &[f64], y: &[f64], lib_sizes: &[usize]) -> Vec<f64> {
        let n = x.len().min(y.len());
        let dim = self.embed_dim;
        let tau = self.tau;
        let min_idx = (dim - 1) * tau;
        if n <= min_idx + dim {
            return vec![0.0; lib_sizes.len()];
        }
        // Embed y to reconstruct x (CCM: x causes y means y's manifold reconstructs x)
        let my = Self::embed(&y[..n], dim, tau);
        let n_pts = my.len();
        if n_pts == 0 {
            return vec![0.0; lib_sizes.len()];
        }
        // x values corresponding to embedded points
        let x_vals: Vec<f64> = ((dim - 1) * tau..n).map(|t| x[t]).collect();

        lib_sizes
            .iter()
            .map(|&lib| {
                let lib = lib.min(n_pts);
                if lib < dim + 1 {
                    return 0.0;
                }
                // Use first `lib` points as library
                let lib_pts = &my[..lib];
                let lib_x = &x_vals[..lib];
                // For each point in full set, find dim+1 nearest neighbors in library
                let mut x_pred = Vec::with_capacity(n_pts);
                for pt in my.iter() {
                    // Find dim+1 NNs
                    let nn = dim + 1;
                    let mut dists: Vec<(f64, usize)> = lib_pts
                        .iter()
                        .enumerate()
                        .map(|(j, lp)| {
                            let d: f64 =
                                lp.iter().zip(pt.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                            (d, j)
                        })
                        .collect();
                    dists
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let nn_count = nn.min(dists.len());
                    if nn_count == 0 {
                        x_pred.push(0.0);
                        continue;
                    }
                    // Weights: exp(-d / d_min) normalized
                    let d_min = dists[0].0.max(1e-15);
                    let weights: Vec<f64> = dists[..nn_count]
                        .iter()
                        .map(|(d, _)| (-d / d_min).exp())
                        .collect();
                    let w_sum: f64 = weights.iter().sum::<f64>().max(1e-15);
                    let xhat: f64 = dists[..nn_count]
                        .iter()
                        .zip(weights.iter())
                        .map(|((_, j), w)| w * lib_x[*j])
                        .sum::<f64>()
                        / w_sum;
                    x_pred.push(xhat);
                }
                Self::pearson_corr(&x_pred, &x_vals)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Structural Time Series
// ─────────────────────────────────────────────────────────────────────────────

/// Local Level Model: Kalman filter for μ_t = μ_{t-1} + η_t, y_t = μ_t + ε_t.
pub struct LocalLevelModel {
    /// Observation noise variance σ²_ε.
    pub sigma2_obs: f64,
    /// State noise variance σ²_η.
    pub sigma2_state: f64,
}

impl LocalLevelModel {
    /// Create with given noise variances.
    pub fn new(sigma2_obs: f64, sigma2_state: f64) -> Self {
        Self {
            sigma2_obs: sigma2_obs.max(1e-10),
            sigma2_state: sigma2_state.max(1e-10),
        }
    }

    /// Run Kalman filter on observations `y`.
    /// Returns (filtered_level, filtered_variance).
    pub fn filter(&self, y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut level = Vec::with_capacity(n);
        let mut variance = Vec::with_capacity(n);
        // Initialize
        let mut m = if n > 0 { y[0] } else { 0.0 };
        let mut p = self.sigma2_obs + self.sigma2_state;
        for &yt in y.iter() {
            // Predict
            let p_pred = p + self.sigma2_state;
            // Update
            let s = p_pred + self.sigma2_obs;
            let k = p_pred / s;
            let v = yt - m;
            m += k * v;
            p = (1.0 - k) * p_pred;
            level.push(m);
            variance.push(p);
        }
        (level, variance)
    }
}

/// Local Linear Trend model: level + slope state-space.
/// State: [μ_t, ν_t], μ_t = μ_{t-1} + ν_{t-1} + η_t, ν_t = ν_{t-1} + ζ_t.
pub struct LocalLinearTrend {
    /// Observation noise variance.
    pub sigma2_obs: f64,
    /// Level noise variance.
    pub sigma2_level: f64,
    /// Slope noise variance.
    pub sigma2_slope: f64,
}

impl LocalLinearTrend {
    /// Create with noise variances.
    pub fn new(sigma2_obs: f64, sigma2_level: f64, sigma2_slope: f64) -> Self {
        Self {
            sigma2_obs: sigma2_obs.max(1e-10),
            sigma2_level: sigma2_level.max(1e-10),
            sigma2_slope: sigma2_slope.max(1e-10),
        }
    }

    /// Kalman filter for local linear trend.
    /// Returns Vec<`Vec<f64>>` where each inner vec is [level, slope].
    pub fn filter(&self, y: &[f64]) -> Vec<Vec<f64>> {
        let n = y.len();
        let mut result = Vec::with_capacity(n);
        // State: [mu, nu]
        let mut mu = if n > 0 { y[0] } else { 0.0 };
        let mut nu = 0.0_f64;
        // P = 2x2 covariance, stored as [p00, p01, p10, p11]
        let mut p00 = self.sigma2_obs;
        let mut p01 = 0.0_f64;
        let mut p11 = self.sigma2_slope;

        for &yt in y.iter() {
            // Predict: mu_pred = mu + nu, nu_pred = nu
            let mu_pred = mu + nu;
            let nu_pred = nu;
            // P_pred = F P F' + Q, F = [[1,1],[0,1]], Q = diag(sigma2_level, sigma2_slope)
            let pp00 = p00 + p01 + p01 + p11 + self.sigma2_level;
            let pp01 = p01 + p11;
            let pp11 = p11 + self.sigma2_slope;
            // Update: H = [1, 0], S = H P_pred H' + R = pp00 + sigma2_obs
            let s = pp00 + self.sigma2_obs;
            let k0 = pp00 / s;
            let k1 = pp01 / s;
            let v = yt - mu_pred;
            mu = mu_pred + k0 * v;
            nu = nu_pred + k1 * v;
            // P = (I - K H) P_pred
            p00 = (1.0 - k0) * pp00;
            p01 = (1.0 - k0) * pp01;
            p11 = pp11 - k1 * pp01;
            result.push(vec![mu, nu]);
        }
        result
    }
}

/// Unobserved Components from UCM decomposition.
#[derive(Debug, Clone)]
pub struct UcComponents {
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Residual (irregular) component.
    pub residual: Vec<f64>,
}

/// Unobserved Components Model: decompose y into trend + seasonal + irregular.
pub struct UnobservedComponentsModel {
    /// Observation noise variance.
    pub sigma2_obs: f64,
    /// Trend noise variance.
    pub sigma2_trend: f64,
    /// Seasonal noise variance.
    pub sigma2_seasonal: f64,
}

impl UnobservedComponentsModel {
    /// Create UCM.
    pub fn new(sigma2_obs: f64, sigma2_trend: f64, sigma2_seasonal: f64) -> Self {
        Self {
            sigma2_obs: sigma2_obs.max(1e-10),
            sigma2_trend: sigma2_trend.max(1e-10),
            sigma2_seasonal: sigma2_seasonal.max(1e-10),
        }
    }

    /// Decompose y with given period.
    pub fn decompose(&self, y: &[f64], period: usize) -> UcComponents {
        let n = y.len();
        if n == 0 {
            return UcComponents {
                trend: Vec::new(),
                seasonal: Vec::new(),
                residual: Vec::new(),
            };
        }
        let period = period.max(2);
        // Extract trend via local level Kalman filter
        let llm = LocalLevelModel::new(self.sigma2_obs, self.sigma2_trend);
        let (trend, _) = llm.filter(y);

        // Detrend
        let detrended: Vec<f64> = y.iter().zip(trend.iter()).map(|(yi, ti)| yi - ti).collect();

        // Estimate seasonal: average within each season
        let mut season_sum = vec![0.0_f64; period];
        let mut season_cnt = vec![0usize; period];
        for (i, &d) in detrended.iter().enumerate() {
            season_sum[i % period] += d;
            season_cnt[i % period] += 1;
        }
        let season_avg: Vec<f64> = season_sum
            .iter()
            .zip(season_cnt.iter())
            .map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 })
            .collect();
        // Center seasonal
        let smean = season_avg.iter().sum::<f64>() / period as f64;
        let seasonal_pattern: Vec<f64> = season_avg.iter().map(|s| s - smean).collect();
        let seasonal: Vec<f64> = (0..n).map(|i| seasonal_pattern[i % period]).collect();
        // Residual
        let residual: Vec<f64> = y
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((yi, ti), si)| yi - ti - si)
            .collect();

        UcComponents {
            trend,
            seasonal,
            residual,
        }
    }
}

/// Seasonal Kalman Filter with trigonometric seasonal component (Harvey 1989).
pub struct SeasonalKalmanFilter {
    /// Seasonal period.
    pub period: usize,
    /// Observation noise variance.
    pub sigma2_obs: f64,
    /// Seasonal component noise variance.
    pub sigma2_seasonal: f64,
}

impl SeasonalKalmanFilter {
    /// Create with period and noise variances.
    pub fn new(period: usize, sigma2_obs: f64, sigma2_seasonal: f64) -> Self {
        Self {
            period: period.max(2),
            sigma2_obs: sigma2_obs.max(1e-10),
            sigma2_seasonal: sigma2_seasonal.max(1e-10),
        }
    }

    /// Filter with trigonometric seasonal (Harvey 1989). Returns (levels, seasonals).
    pub fn filter(&self, y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let p = self.period;
        let n_harm = p / 2;
        let ns = 2 * n_harm + 1;
        let mut state = vec![0.0_f64; ns];
        state[0] = y[0];
        let mut pcov = vec![vec![0.0_f64; ns]; ns];
        for i in 0..ns {
            pcov[i][i] = 1.0;
        }
        let mut levels = Vec::with_capacity(n);
        let mut seasonals = Vec::with_capacity(n);
        let two_pi = 2.0 * std::f64::consts::PI;
        for &yt in y {
            // Predict state
            let mut sp = vec![0.0_f64; ns];
            sp[0] = state[0];
            for j in 0..n_harm {
                let lam = two_pi * (j + 1) as f64 / p as f64;
                let (sl, cl) = lam.sin_cos();
                let g = state[1 + 2 * j];
                let gs = state[2 + 2 * j];
                sp[1 + 2 * j] = cl * g + sl * gs;
                sp[2 + 2 * j] = -sl * g + cl * gs;
            }
            // Predict covariance
            let mut pp = pcov.clone();
            pp[0][0] += self.sigma2_obs;
            for j in 0..n_harm {
                pp[1 + 2 * j][1 + 2 * j] += self.sigma2_seasonal;
                pp[2 + 2 * j][2 + 2 * j] += self.sigma2_seasonal;
            }
            // H = [1, 1, 0, 1, 0, ...]
            let h: Vec<f64> = (0..ns)
                .map(|i| if i == 0 || (i % 2 == 1) { 1.0 } else { 0.0 })
                .collect();
            let ph: Vec<f64> = (0..ns)
                .map(|i| pp[i].iter().zip(h.iter()).map(|(pv, &hj)| pv * hj).sum())
                .collect();
            let sv = ph
                .iter()
                .zip(h.iter())
                .map(|(pv, &hj)| pv * hj)
                .sum::<f64>()
                + self.sigma2_obs;
            let k: Vec<f64> = ph.iter().map(|pv| pv / sv).collect();
            let y_pred: f64 = sp[0] + (0..n_harm).map(|j| sp[1 + 2 * j]).sum::<f64>();
            let v = yt - y_pred;
            for i in 0..ns {
                state[i] = sp[i] + k[i] * v;
            }
            for i in 0..ns {
                for j in 0..ns {
                    pcov[i][j] = pp[i][j] - k[i] * ph[j];
                }
            }
            levels.push(state[0]);
            seasonals.push((0..n_harm).map(|j| state[1 + 2 * j]).sum::<f64>());
        }
        (levels, seasonals)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Intervention Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Impact summary from CausalImpactModel.
#[derive(Debug, Clone)]
pub struct ImpactSummary {
    /// Estimated absolute treatment effect.
    pub estimated_effect: f64,
    /// Relative treatment effect (effect / counterfactual).
    pub relative_effect: f64,
    /// Probability of a causal effect (from normal CDF).
    pub prob_of_causal_effect: f64,
}

/// Bayesian Structural Time Series for causal impact: pre-fit → post-counterfactual.
pub struct CausalImpactModel {
    filtered_level: Vec<f64>,
    filtered_var: Vec<f64>,
    pre_mean: f64,
    sigma2_noise: f64,
    post_len: usize,
    last_level: f64,
    last_var: f64,
}

impl CausalImpactModel {
    /// Create a new CausalImpactModel.
    pub fn new() -> Self {
        Self {
            filtered_level: Vec::new(),
            filtered_var: Vec::new(),
            pre_mean: 0.0,
            sigma2_noise: 1.0,
            post_len: 0,
            last_level: 0.0,
            last_var: 1.0,
        }
    }

    /// Fit on pre-intervention series.
    pub fn fit(&mut self, y_pre: &[f64], y_post_len: usize) {
        let n = y_pre.len();
        let s2 = if n > 1 {
            y_pre.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>() / (2.0 * (n - 1) as f64)
        } else {
            1.0
        };
        let (level, var) = LocalLevelModel::new(s2, s2 * 0.01).filter(y_pre);
        self.pre_mean = if n > 0 {
            y_pre.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        self.sigma2_noise = s2;
        self.last_level = level.last().copied().unwrap_or(self.pre_mean);
        self.last_var = var.last().copied().unwrap_or(s2);
        self.filtered_level = level;
        self.filtered_var = var;
        self.post_len = y_post_len;
    }

    /// Compute causal impact summary for post-intervention series.
    pub fn impact(&self, y_post: &[f64]) -> ImpactSummary {
        let np = y_post.len().min(self.post_len).max(1);
        let cf = self.last_level;
        let post_mean = y_post[..np].iter().sum::<f64>() / np as f64;
        let eff = post_mean - cf;
        let rel = if cf.abs() > 1e-15 { eff / cf } else { 0.0 };
        let z = eff
            / (self.last_var + self.sigma2_noise * np as f64)
                .sqrt()
                .max(1e-15);
        ImpactSummary {
            estimated_effect: eff,
            relative_effect: rel,
            prob_of_causal_effect: normal_cdf_f64(z),
        }
    }
}

impl Default for CausalImpactModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Synthetic Control: find simplex weights on donors matching treated pre-period.
pub struct SyntheticControl {
    /// Simplex weights over donor units.
    pub weights: Vec<f64>,
    /// Number of donor units.
    pub n_donors: usize,
}

impl SyntheticControl {
    /// Create a new SyntheticControl.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            n_donors: 0,
        }
    }

    /// Fit via projected gradient descent.
    pub fn fit(&mut self, treated_pre: &[f64], donors_pre: &[Vec<f64>]) -> CtsResult<()> {
        let j = donors_pre.len();
        if j == 0 {
            return Err("SyntheticControl: no donors".to_string());
        }
        let n = treated_pre.len();
        if n == 0 {
            return Err("SyntheticControl: empty treated series".to_string());
        }
        let mut w = vec![1.0_f64 / j as f64; j];
        for _ in 0..500 {
            let synth: Vec<f64> = (0..n)
                .map(|t| {
                    donors_pre
                        .iter()
                        .zip(&w)
                        .map(|(d, &wj)| wj * d.get(t).copied().unwrap_or(0.0))
                        .sum()
                })
                .collect();
            let resid: Vec<f64> = treated_pre.iter().zip(&synth).map(|(a, b)| a - b).collect();
            let mut grad = vec![0.0_f64; j];
            for (jj, dj) in donors_pre.iter().enumerate() {
                for t in 0..n {
                    grad[jj] -= 2.0 * resid[t] * dj.get(t).copied().unwrap_or(0.0);
                }
            }
            for jj in 0..j {
                w[jj] -= 0.1 * grad[jj];
            }
            w = project_simplex(&w);
        }
        self.weights = w;
        self.n_donors = j;
        Ok(())
    }

    /// Compute counterfactual predictions using fitted weights.
    pub fn counterfactual(&self, donors_post: &[Vec<f64>]) -> Vec<f64> {
        if donors_post.is_empty() || self.weights.is_empty() {
            return Vec::new();
        }
        let tm = donors_post.iter().map(|d| d.len()).min().unwrap_or(0);
        (0..tm)
            .map(|t| {
                donors_post
                    .iter()
                    .zip(&self.weights)
                    .map(|(d, &w)| w * d[t])
                    .sum()
            })
            .collect()
    }
}

impl Default for SyntheticControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Project vector onto probability simplex (Duchi et al. 2008).
/// Guarantees result sums to 1 and all components >= 0.
pub(crate) fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    // Sort descending
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // Find rho = max{j: u_j - (sum_{i<=j} u_i - 1)/j > 0}
    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        // condition: u_i > (cssv - 1) / (i+1)
        if ui > (cssv - 1.0) / (i + 1) as f64 {
            rho = i;
        }
    }
    // theta = (sum_{i<=rho} u_i - 1) / (rho + 1)
    let theta = (u[..=rho].iter().sum::<f64>() - 1.0) / (rho + 1) as f64;
    // Clip and return
    let result: Vec<f64> = v.iter().map(|&vi| (vi - theta).max(0.0)).collect();
    // Numerical safety: ensure sum is exactly 1
    let s: f64 = result.iter().sum::<f64>();
    if s > 1e-15 {
        result.iter().map(|&r| r / s).collect()
    } else {
        // Degenerate fallback: uniform
        vec![1.0 / n as f64; n]
    }
}

/// RD Estimate result.
#[derive(Debug, Clone)]
pub struct RdEstimate {
    /// Estimated treatment effect at the cutoff (right intercept - left intercept).
    pub treatment_effect: f64,
    /// Left-side polynomial intercept at cutoff.
    pub left_intercept: f64,
    /// Right-side polynomial intercept at cutoff.
    pub right_intercept: f64,
}

/// Regression Discontinuity Design: local polynomial regression on each side of cutoff.
pub struct RegressionDiscontinuity {
    /// Polynomial degree for local regression.
    pub degree: usize,
}

impl RegressionDiscontinuity {
    /// Create with polynomial degree.
    pub fn new(degree: usize) -> Self {
        Self {
            degree: degree.max(1),
        }
    }

    /// Estimate RD treatment effect at `cutoff` with `bandwidth`.
    pub fn estimate(
        &self,
        x: &[f64],
        y: &[f64],
        cutoff: f64,
        bandwidth: f64,
    ) -> CtsResult<RdEstimate> {
        let n = x.len().min(y.len());
        if n == 0 {
            return Err("RD: empty data".to_string());
        }
        let bw = bandwidth.max(1e-10);
        let deg = self.degree;
        let (mut xl, mut yl, mut wl, mut xr, mut yr, mut wr) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let u = (x[i] - cutoff) / bw;
            let wt = (1.0 - u.abs()).max(0.0);
            if wt <= 0.0 {
                continue;
            }
            if x[i] < cutoff {
                xl.push(x[i] - cutoff);
                yl.push(y[i]);
                wl.push(wt);
            } else {
                xr.push(x[i] - cutoff);
                yr.push(y[i]);
                wr.push(wt);
            }
        }
        if xl.is_empty() || xr.is_empty() {
            return Err("RD: insufficient data on one or both sides".to_string());
        }
        let wpoly = |xs: &[f64], ys: &[f64], ws: &[f64]| -> CtsResult<f64> {
            let d: Vec<Vec<f64>> = xs
                .iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let wi = ws[i].sqrt();
                    (0..=deg).map(|dd| wi * xi.powi(dd as i32)).collect()
                })
                .collect();
            let yw: Vec<f64> = ys.iter().zip(ws).map(|(&yi, &wi)| yi * wi.sqrt()).collect();
            Ok(ols(&d, &yw)?[0])
        };
        let li = wpoly(&xl, &yl, &wl)?;
        let ri = wpoly(&xr, &yr, &wr)?;
        Ok(RdEstimate {
            treatment_effect: ri - li,
            left_intercept: li,
            right_intercept: ri,
        })
    }
}

/// Difference-in-Differences estimator (regression-based).
pub struct DifferenceInDifferences;

impl DifferenceInDifferences {
    /// Estimate DiD treatment effect from pre/post outcomes and treatment indicators.
    pub fn estimate(y_pre: &[f64], y_post: &[f64], treated: &[bool]) -> CtsResult<f64> {
        let n = y_pre.len().min(y_post.len()).min(treated.len());
        if n == 0 {
            return Err("DiD: empty data".to_string());
        }
        let mut design: Vec<Vec<f64>> = Vec::with_capacity(2 * n);
        let mut y_all: Vec<f64> = Vec::with_capacity(2 * n);
        for i in 0..n {
            let ti = if treated[i] { 1.0 } else { 0.0 };
            design.push(vec![1.0, 0.0, ti, 0.0]);
            y_all.push(y_pre[i]);
            design.push(vec![1.0, 1.0, ti, ti]);
            y_all.push(y_post[i]);
        }
        Ok(ols(&design, &y_all)?[3])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Neural Time Series Forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// WaveNet: dilated causal convolution with gating (tanh × sigmoid).
pub struct WaveNet {
    kernels: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    out_weights: Vec<f64>,
    kernel_size: usize,
}

impl WaveNet {
    /// Create WaveNet with given kernel size, layers, and random seed.
    pub fn new(kernel_size: usize, n_layers: usize, seed: u64) -> CtsResult<Self> {
        let ks = kernel_size.max(2);
        let nl = n_layers.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let sc = (2.0_f64 / ks as f64).sqrt();
        let kernels: Vec<Vec<f64>> = (0..nl)
            .map(|_| {
                (0..ks * 2)
                    .map(|_| (rng.random::<f64>() - 0.5) * 2.0 * sc)
                    .collect()
            })
            .collect();
        let biases: Vec<Vec<f64>> = (0..nl).map(|_| vec![0.0_f64; 2]).collect();
        let out_weights: Vec<f64> = (0..nl).map(|_| (rng.random::<f64>() - 0.5) * 0.1).collect();
        Ok(Self {
            kernels,
            biases,
            out_weights,
            kernel_size: ks,
        })
    }

    /// Forward pass with given dilations.
    pub fn forward(&self, x: &[f64], dilations: &[usize]) -> CtsResult<Vec<f64>> {
        if x.is_empty() {
            return Err("WaveNet: empty input".to_string());
        }
        let n = x.len();
        let nl = self.kernels.len();
        let nd = dilations.len().max(1);
        let mut sig = x.to_vec();
        for layer in 0..nl {
            let dil = dilations.get(layer % nd).copied().unwrap_or(1).max(1);
            let ks = self.kernel_size;
            let kern = &self.kernels[layer];
            let bias = &self.biases[layer];
            let mut out = vec![0.0_f64; n];
            for t in 0..n {
                let (mut h, mut g) = (bias[0], bias[1]);
                for ki in 0..ks {
                    let src = t as isize - (ki * dil) as isize;
                    let val = if src >= 0 { sig[src as usize] } else { 0.0 };
                    h += kern[ki] * val;
                    g += kern[ks + ki] * val;
                }
                out[t] = h.tanh() * sigmoid_f64(g);
            }
            for t in 0..n {
                sig[t] += out[t];
            }
        }
        let wsum: f64 = self.out_weights.iter().sum();
        Ok(sig.iter().map(|&s| s * wsum).collect())
    }
}

/// Deep State Space Model (deep Kalman filter): neural transition + emission.
pub struct DeepStateSpaceModel {
    /// Latent state dimension.
    pub latent_dim: usize,
    trans_w: Vec<f64>,
    emit_mean_w: Vec<f64>,
    emit_logvar_w: Vec<f64>,
}

impl DeepStateSpaceModel {
    /// Create DeepSSM with given latent dimension and random seed.
    pub fn new(latent_dim: usize, seed: u64) -> Self {
        let ld = latent_dim.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let sc = (1.0_f64 / ld as f64).sqrt();
        Self {
            latent_dim: ld,
            trans_w: (0..ld * ld)
                .map(|_| (rng.random::<f64>() - 0.5) * sc)
                .collect(),
            emit_mean_w: (0..ld).map(|_| (rng.random::<f64>() - 0.5) * sc).collect(),
            emit_logvar_w: (0..ld)
                .map(|_| (rng.random::<f64>() - 0.5) * 0.1 - 1.0)
                .collect(),
        }
    }

    /// Forward: returns (mean_seq, variance_seq).
    pub fn forward(&self, x: &[f64]) -> CtsResult<(Vec<f64>, Vec<f64>)> {
        let n = x.len();
        if n == 0 {
            return Err("DeepSSM: empty input".to_string());
        }
        let ld = self.latent_dim;
        let mut state = vec![0.0_f64; ld];
        let mut means = Vec::with_capacity(n);
        let mut vars = Vec::with_capacity(n);
        for &xt in x {
            let mut next = vec![0.0_f64; ld];
            for i in 0..ld {
                let s: f64 = (0..ld).map(|j| self.trans_w[i * ld + j] * state[j]).sum();
                next[i] = (s + xt).tanh();
            }
            state = next;
            means.push(
                state
                    .iter()
                    .zip(&self.emit_mean_w)
                    .map(|(s, w)| s * w)
                    .sum(),
            );
            let lv: f64 = state
                .iter()
                .zip(&self.emit_logvar_w)
                .map(|(s, w)| s * w)
                .sum();
            vars.push(lv.exp().max(1e-15));
        }
        Ok((means, vars))
    }
}

/// Temporal Convolutional Network (TCN): residual dilated causal conv blocks.
pub struct TemporalConvNet {
    layer_weights: Vec<Vec<f64>>,
    channels: usize,
    kernel_size: usize,
}

impl TemporalConvNet {
    /// Create TCN with given layers, channels, kernel size, and seed.
    pub fn new(n_layers: usize, channels: usize, kernel_size: usize, seed: u64) -> Self {
        let nl = n_layers.max(1);
        let ch = channels.max(1);
        let ks = kernel_size.max(2);
        let mut rng = StdRng::seed_from_u64(seed);
        let sc = (2.0_f64 / (ch * ks) as f64).sqrt();
        Self {
            layer_weights: (0..nl)
                .map(|_| {
                    (0..ch * ch * ks)
                        .map(|_| (rng.random::<f64>() - 0.5) * sc)
                        .collect()
                })
                .collect(),
            channels: ch,
            kernel_size: ks,
        }
    }

    /// Forward pass with explicit layer/channel/kernel parameters.
    pub fn forward(
        &self,
        x: &[f64],
        n_layers: usize,
        channels: usize,
        kernel_size: usize,
    ) -> CtsResult<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Err("TCN: empty input".to_string());
        }
        let nl = n_layers.min(self.layer_weights.len()).max(1);
        let ch = channels.min(self.channels).max(1);
        let ks = kernel_size.max(2);
        let mut sig: Vec<Vec<f64>> = (0..ch).map(|_| x.to_vec()).collect();
        for layer in 0..nl {
            let dil = 1usize << layer;
            let w = &self.layer_weights[layer];
            let mut out = vec![vec![0.0_f64; n]; ch];
            for co in 0..ch {
                for t in 0..n {
                    let mut val = 0.0_f64;
                    for ci in 0..ch {
                        for ki in 0..ks {
                            let src = t as isize - (ki * dil) as isize;
                            let v = if src >= 0 { sig[ci][src as usize] } else { 0.0 };
                            val += w.get((co * ch + ci) * ks + ki).copied().unwrap_or(0.0) * v;
                        }
                    }
                    out[co][t] = val.max(0.0);
                }
            }
            for c in 0..ch {
                for t in 0..n {
                    sig[c][t] += out[c][t];
                }
            }
        }
        Ok((0..n)
            .map(|t| sig.iter().map(|c| c[t]).sum::<f64>() / ch as f64)
            .collect())
    }
}

/// Recurrent GAN for Time Series (TimeGAN-simplified): LSTM generator + discriminator readout.
pub struct RecurrentGanForTimeSeries {
    /// Hidden dimension of the LSTM.
    pub hidden_dim: usize,
    gen_w: Vec<f64>,
    disc_w: Vec<f64>,
}

impl RecurrentGanForTimeSeries {
    /// Create RecurrentGAN with hidden dim and seed.
    pub fn new(hidden_dim: usize, seed: u64) -> Self {
        let hd = hidden_dim.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let sc = (1.0_f64 / hd as f64).sqrt();
        Self {
            hidden_dim: hd,
            gen_w: (0..(1 + hd) * 4 * hd)
                .map(|_| (rng.random::<f64>() - 0.5) * sc)
                .collect(),
            disc_w: (0..hd * 2)
                .map(|_| (rng.random::<f64>() - 0.5) * sc)
                .collect(),
        }
    }

    /// Generate a synthetic time series of given length.
    pub fn generate(&self, length: usize, noise_seed: u64) -> Vec<f64> {
        let hd = self.hidden_dim;
        let mut rng = StdRng::seed_from_u64(noise_seed);
        let mut h = vec![0.0_f64; hd];
        let mut c = vec![0.0_f64; hd];
        let mut out = Vec::with_capacity(length);
        for _ in 0..length {
            let x_in = rng.random::<f64>() * 2.0 - 1.0;
            let (hn, cn, y) = self.lstm_step(x_in, &h, &c);
            h = hn;
            c = cn;
            out.push(y);
        }
        out
    }

    fn lstm_step(&self, x: f64, h: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        let hd = self.hidden_dim;
        let isz = 1 + hd;
        let w = &self.gen_w;
        let xh: Vec<f64> = std::iter::once(x).chain(h.iter().copied()).collect();
        let mut gates = vec![0.0_f64; 4 * hd];
        for g in 0..4 {
            for i in 0..hd {
                gates[g * hd + i] = (0..isz)
                    .map(|j| w.get((g * hd + i) * isz + j).copied().unwrap_or(0.0) * xh[j])
                    .sum();
            }
        }
        let mut hn = vec![0.0_f64; hd];
        let mut cn = vec![0.0_f64; hd];
        for i in 0..hd {
            cn[i] = sigmoid_f64(gates[hd + i]) * c[i]
                + sigmoid_f64(gates[i]) * gates[2 * hd + i].tanh();
            hn[i] = sigmoid_f64(gates[3 * hd + i]) * cn[i].tanh();
        }
        let y: f64 = hn
            .iter()
            .zip(self.disc_w.iter().take(hd))
            .map(|(hi, wi)| hi * wi)
            .sum();
        (hn, cn, y)
    }
}

/// Anomaly Transformer: association discrepancy (symmetric KL) between prior and series.
pub struct AnomalyTransformer {
    /// Sequence length.
    pub seq_len: usize,
    /// Window size for Gaussian prior.
    pub window_size: usize,
}

impl AnomalyTransformer {
    /// Create AnomalyTransformer with sequence length and window size.
    pub fn new(seq_len: usize, window_size: usize) -> Self {
        Self {
            seq_len: seq_len.max(1),
            window_size: window_size.max(1),
        }
    }

    /// Compute symmetric KL discrepancy between prior and series associations.
    pub fn compute_discrepancy(&self, prior_assoc: &[f64], series_assoc: &[f64]) -> Vec<f64> {
        let n = prior_assoc.len().min(series_assoc.len());
        if n == 0 {
            return Vec::new();
        }
        let norm = |v: &[f64]| -> Vec<f64> {
            let s = v.iter().sum::<f64>().max(1e-15);
            v.iter().map(|&x| x.max(1e-15) / s).collect()
        };
        let p = norm(&prior_assoc[..n]);
        let q = norm(&series_assoc[..n]);
        p.iter()
            .zip(&q)
            .map(|(&pi, &qi)| (pi * (pi / qi).ln() + qi * (qi / pi).ln()).abs())
            .collect()
    }

    /// Compute Gaussian prior association centered at time step `t`.
    pub fn prior_association(&self, t: usize, n: usize) -> Vec<f64> {
        let sig = self.window_size as f64 / 3.0;
        let a: Vec<f64> = (0..n)
            .map(|i| {
                let d = (i as f64 - t as f64) / sig;
                (-0.5 * d * d).exp()
            })
            .collect();
        let s = a.iter().sum::<f64>().max(1e-15);
        a.iter().map(|v| v / s).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Counterfactual Time Series
// ─────────────────────────────────────────────────────────────────────────────

/// GP-based counterfactual prediction using squared exponential kernel.
pub struct GpCounterfactual {
    /// Length scale of the SE kernel.
    pub length_scale: f64,
    /// Signal variance of the SE kernel.
    pub signal_var: f64,
    /// Noise variance.
    pub noise_var: f64,
    t_train: Vec<f64>,
    y_train: Vec<f64>,
    alpha: Vec<f64>,
    chol: Vec<Vec<f64>>,
}

impl GpCounterfactual {
    /// Create GP counterfactual with kernel hyperparameters.
    pub fn new(length_scale: f64, signal_var: f64, noise_var: f64) -> Self {
        Self {
            length_scale: length_scale.max(1e-3),
            signal_var: signal_var.max(1e-10),
            noise_var: noise_var.max(1e-10),
            t_train: Vec::new(),
            y_train: Vec::new(),
            alpha: Vec::new(),
            chol: Vec::new(),
        }
    }

    #[inline]
    fn se_kernel(&self, t1: f64, t2: f64) -> f64 {
        let d = (t1 - t2) / self.length_scale;
        self.signal_var * (-0.5 * d * d).exp()
    }

    /// Fit GP on control time series.
    pub fn fit(&mut self, control: &[f64], time_control: &[f64]) -> CtsResult<()> {
        let n = control.len().min(time_control.len());
        if n == 0 {
            return Err("GpCounterfactual: empty training data".to_string());
        }
        self.t_train = time_control[..n].to_vec();
        self.y_train = control[..n].to_vec();
        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.se_kernel(self.t_train[i], self.t_train[j]);
            }
            k[i][i] += self.noise_var;
        }
        let mut l = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = k[i][j];
                for m in 0..j {
                    s -= l[i][m] * l[j][m];
                }
                l[i][j] = if i == j {
                    s.max(1e-12).sqrt()
                } else {
                    s / l[j][j]
                };
            }
        }
        let mut vf = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = self.y_train[i];
            for j in 0..i {
                s -= l[i][j] * vf[j];
            }
            vf[i] = s / l[i][i];
        }
        let mut alpha = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = vf[i];
            for j in (i + 1)..n {
                s -= l[j][i] * alpha[j];
            }
            alpha[i] = s / l[i][i];
        }
        self.alpha = alpha;
        self.chol = l;
        Ok(())
    }

    /// Predict at new time points. Returns (means, stds).
    pub fn predict(&self, time_treated: &[f64]) -> CtsResult<(Vec<f64>, Vec<f64>)> {
        let n_train = self.t_train.len();
        if n_train == 0 {
            return Err("GpCounterfactual: not fitted".to_string());
        }
        let mut means = Vec::with_capacity(time_treated.len());
        let mut stds = Vec::with_capacity(time_treated.len());
        for &t_star in time_treated {
            let k_star: Vec<f64> = self
                .t_train
                .iter()
                .map(|&ti| self.se_kernel(t_star, ti))
                .collect();
            let mean: f64 = k_star.iter().zip(&self.alpha).map(|(k, a)| k * a).sum();
            let mut vv = vec![0.0_f64; n_train];
            for i in 0..n_train {
                let mut s = k_star[i];
                for j in 0..i {
                    s -= self.chol[i][j] * vv[j];
                }
                vv[i] = s / self.chol[i][i];
            }
            let var = (self.se_kernel(t_star, t_star) + self.noise_var
                - vv.iter().map(|v| v * v).sum::<f64>())
            .max(0.0);
            means.push(mean);
            stds.push(var.sqrt());
        }
        Ok((means, stds))
    }
}

/// Neural Synthetic Control: encoder-based matching in embedding space.
pub struct NeuralSyntheticControl {
    /// Embedding dimension.
    pub embed_dim: usize,
    encoder_w: Vec<f64>,
    weights: Vec<f64>,
    n_donors: usize,
}

impl NeuralSyntheticControl {
    /// Create NeuralSyntheticControl with embedding dim and seed.
    pub fn new(embed_dim: usize, seed: u64) -> Self {
        let ed = embed_dim.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f64 / ed as f64).sqrt();
        let encoder_w: Vec<f64> = (0..ed * ed)
            .map(|_| (rng.random::<f64>() - 0.5) * scale)
            .collect();
        Self {
            embed_dim: ed,
            encoder_w,
            weights: Vec::new(),
            n_donors: 0,
        }
    }

    fn encode(&self, ts: &[f64]) -> Vec<f64> {
        let ed = self.embed_dim;
        let n = ts.len();
        if n == 0 {
            return vec![0.0; ed];
        }
        let cs = ((n as f64 / ed as f64).ceil() as usize).max(1);
        let features: Vec<f64> = (0..ed)
            .map(|i| {
                let s = i * cs;
                let e = ((i + 1) * cs).min(n);
                if s >= n {
                    0.0
                } else {
                    ts[s..e].iter().sum::<f64>() / (e - s) as f64
                }
            })
            .collect();
        let mut emb = vec![0.0_f64; ed];
        for i in 0..ed {
            for j in 0..ed {
                emb[i] += self.encoder_w[i * ed + j] * features[j];
            }
            emb[i] = emb[i].tanh();
        }
        emb
    }

    /// Fit via cosine-similarity softmax weights in embedding space.
    pub fn fit(&mut self, treated_pre: &[f64], donors: &[Vec<f64>]) -> CtsResult<()> {
        let j = donors.len();
        if j == 0 {
            return Err("NeuralSC: no donors".to_string());
        }
        let et = self.encode(treated_pre);
        let eds: Vec<Vec<f64>> = donors.iter().map(|d| self.encode(d)).collect();
        let sims: Vec<f64> = eds
            .iter()
            .map(|ed| {
                let dot: f64 = et.iter().zip(ed).map(|(a, b)| a * b).sum();
                let na = et.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-15);
                let nb = ed.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-15);
                dot / (na * nb)
            })
            .collect();
        let mx = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = sims.iter().map(|s| (s - mx).exp()).collect();
        let se: f64 = exps.iter().sum::<f64>().max(1e-15);
        self.weights = exps.iter().map(|e| e / se).collect();
        self.n_donors = j;
        Ok(())
    }

    /// Predict post-intervention counterfactual using donor series.
    pub fn predict(&self, donors_post: &[Vec<f64>]) -> Vec<f64> {
        if donors_post.is_empty() || self.weights.is_empty() {
            return Vec::new();
        }
        let t_max = donors_post.iter().map(|d| d.len()).min().unwrap_or(0);
        (0..t_max)
            .map(|t| {
                donors_post
                    .iter()
                    .zip(&self.weights)
                    .map(|(d, &w)| w * d[t])
                    .sum()
            })
            .collect()
    }
}

/// Time-varying propensity score for observational time series.
pub struct PropensityScoreTs {
    /// Learned logistic regression weights.
    pub weights: Vec<Vec<f64>>,
    /// Number of input features.
    pub n_features: usize,
}

impl PropensityScoreTs {
    /// Create PropensityScoreTs with feature count.
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: Vec::new(),
            n_features: n_features.max(1),
        }
    }

    /// Fit via gradient descent logistic regression (time-varying wrapper).
    pub fn fit(&mut self, x_ts: &[Vec<f64>], treated_ts: &[bool]) -> CtsResult<()> {
        let n = x_ts.len().min(treated_ts.len());
        if n == 0 {
            return Err("PropensityScoreTs: empty data".to_string());
        }
        let p = self.n_features;
        let design: Vec<Vec<f64>> = x_ts[..n]
            .iter()
            .map(|xi| {
                let mut row = vec![1.0];
                row.extend_from_slice(&xi[..p.min(xi.len())]);
                while row.len() <= p {
                    row.push(0.0);
                }
                row
            })
            .collect();
        let y: Vec<f64> = treated_ts[..n]
            .iter()
            .map(|&t| if t { 1.0 } else { 0.0 })
            .collect();
        let mut w = vec![0.0_f64; p + 1];
        for _ in 0..200 {
            let mut grad = vec![0.0_f64; p + 1];
            for i in 0..n {
                let pred: f64 = design[i].iter().zip(&w).map(|(xi, wi)| xi * wi).sum();
                let err = sigmoid_f64(pred) - y[i];
                for j in 0..=p {
                    grad[j] += err * design[i].get(j).copied().unwrap_or(0.0);
                }
            }
            for j in 0..=p {
                w[j] -= 0.01 * grad[j] / n as f64;
            }
        }
        self.weights = vec![w];
        Ok(())
    }

    /// Score observations with fitted propensity model.
    pub fn score(&self, x: &[Vec<f64>]) -> Vec<f64> {
        if self.weights.is_empty() {
            return vec![0.5; x.len()];
        }
        let w = &self.weights[0];
        x.iter()
            .map(|xi| {
                let logit = w[0]
                    + xi.iter()
                        .enumerate()
                        .map(|(j, &v)| w.get(j + 1).copied().unwrap_or(0.0) * v)
                        .sum::<f64>();
                sigmoid_f64(logit)
            })
            .collect()
    }
}

/// Inverse Intensity Weighting for event sequences.
pub struct InverseIntensityWeighting {
    /// Bandwidth for Gaussian kernel density estimation.
    pub bandwidth: f64,
}

impl InverseIntensityWeighting {
    /// Create with bandwidth for KDE.
    pub fn new(bandwidth: f64) -> Self {
        Self {
            bandwidth: bandwidth.max(1e-3),
        }
    }

    /// Estimate event intensity at query times using Gaussian KDE.
    pub fn estimate_intensity(&self, event_times: &[f64], query_times: &[f64]) -> Vec<f64> {
        let bw = self.bandwidth;
        let norm = (2.0 * std::f64::consts::PI).sqrt() * bw;
        query_times
            .iter()
            .map(|&qt| {
                event_times
                    .iter()
                    .map(|&et| {
                        let u = (qt - et) / bw;
                        (-0.5 * u * u).exp() / norm
                    })
                    .sum::<f64>()
                    .max(1e-15)
            })
            .collect()
    }

    /// Compute inverse-intensity weights at event times.
    pub fn weights(&self, event_times: &[f64]) -> Vec<f64> {
        self.estimate_intensity(event_times, event_times)
            .iter()
            .map(|&iv| 1.0 / iv)
            .collect()
    }
}

/// Causal Attention Mechanism with causal masking and attribution via attention rollout.
pub struct CausalAttentionMechanism {
    /// Sequence length.
    pub seq_len: usize,
    /// Head dimension.
    pub head_dim: usize,
    q_w: Vec<f64>,
    k_w: Vec<f64>,
    v_w: Vec<f64>,
}

impl CausalAttentionMechanism {
    /// Create CausalAttentionMechanism with sequence length, head dim, and seed.
    pub fn new(seq_len: usize, head_dim: usize, seed: u64) -> Self {
        let sl = seq_len.max(1);
        let hd = head_dim.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0_f64 / hd as f64).sqrt();
        let mk = |r: &mut StdRng| {
            (0..hd * hd)
                .map(|_| (r.random::<f64>() - 0.5) * scale)
                .collect::<Vec<_>>()
        };
        Self {
            seq_len: sl,
            head_dim: hd,
            q_w: mk(&mut rng),
            k_w: mk(&mut rng),
            v_w: mk(&mut rng),
        }
    }

    fn project(&self, x: &[Vec<f64>], w: &[f64]) -> Vec<Vec<f64>> {
        let hd = self.head_dim;
        x.iter()
            .map(|xi| {
                let mut out = vec![0.0_f64; hd];
                for i in 0..hd {
                    for j in 0..hd {
                        out[i] += w[i * hd + j] * xi.get(j).copied().unwrap_or(0.0);
                    }
                }
                out
            })
            .collect()
    }

    /// Causal masked attention. Returns (output, attention_weights).
    pub fn forward(&self, x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = x.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let hd = self.head_dim;
        let q = self.project(x, &self.q_w);
        let k = self.project(x, &self.k_w);
        let v = self.project(x, &self.v_w);
        let sc = (hd as f64).sqrt();
        let mut attn = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let max_r = (0..=i)
                .map(|j| q[i].iter().zip(&k[j]).map(|(qi, kj)| qi * kj).sum::<f64>() / sc)
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = (0..=i)
                .map(|j| {
                    let dot: f64 =
                        q[i].iter().zip(&k[j]).map(|(qi, kj)| qi * kj).sum::<f64>() / sc;
                    (dot - max_r).exp()
                })
                .collect();
            let se = exps.iter().sum::<f64>().max(1e-15);
            for j in 0..=i {
                attn[i][j] = exps[j] / se;
            }
        }
        let output = (0..n)
            .map(|i| {
                let mut out = vec![0.0_f64; hd];
                for j in 0..=i {
                    for d in 0..hd {
                        out[d] += attn[i][j] * v[j].get(d).copied().unwrap_or(0.0);
                    }
                }
                out
            })
            .collect();
        (output, attn)
    }

    /// Attribution via attention rollout (last row of attention weights).
    pub fn compute_attribution(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let (_, attn) = self.forward(x);
        if attn.is_empty() {
            Vec::new()
        } else {
            attn[attn.len() - 1].clone()
        }
    }
}
