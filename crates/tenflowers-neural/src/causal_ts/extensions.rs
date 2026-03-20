//! Extensions to the causal time series module.
//!
//! Additional utilities: lag-selection criteria (AIC/BIC for VAR), rolling-window
//! Granger summaries, multivariate transfer entropy, and causal graph helpers.

use super::{ols, residuals, ssr, CtsResult, GrangerCausalityTest};

// ─────────────────────────────────────────────────────────────────────────────
// VAR Lag Selection
// ─────────────────────────────────────────────────────────────────────────────

/// AIC/BIC lag selection result.
#[derive(Debug, Clone)]
pub struct LagSelectionResult {
    /// Optimal lag selected by AIC.
    pub best_lag_aic: usize,
    /// Optimal lag selected by BIC.
    pub best_lag_bic: usize,
    /// AIC values for each lag tested.
    pub aic_values: Vec<f64>,
    /// BIC values for each lag tested.
    pub bic_values: Vec<f64>,
}

/// VAR Lag-order selection via AIC and BIC criteria (Lütkepohl 2005).
///
/// For each candidate lag p, fit a bivariate AR(p) on two series and compute
/// AIC = -2 log L + 2k and BIC = -2 log L + k log T.
pub struct VarLagSelector;

impl VarLagSelector {
    /// Select optimal lag for `x` Granger-causing `y` over lags 1..=`max_lag`.
    pub fn select(x: &[f64], y: &[f64], max_lag: usize) -> CtsResult<LagSelectionResult> {
        let n = x.len().min(y.len());
        if n == 0 {
            return Err("VarLagSelector: empty series".to_string());
        }
        let max_lag = max_lag.max(1).min(n / 4);
        let mut aic_values = Vec::with_capacity(max_lag);
        let mut bic_values = Vec::with_capacity(max_lag);

        for p in 1..=max_lag {
            if n <= p + 1 {
                aic_values.push(f64::INFINITY);
                bic_values.push(f64::INFINITY);
                continue;
            }
            let n_eff = n - p;
            // Build unrestricted design matrix
            let mut design: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
            let mut y_vec: Vec<f64> = Vec::with_capacity(n_eff);
            for t in p..n {
                let mut row = vec![1.0_f64];
                for lag in 1..=p {
                    row.push(y[t - lag]);
                    row.push(x[t - lag]);
                }
                design.push(row);
                y_vec.push(y[t]);
            }
            let k = design[0].len() as f64; // number of parameters
            let beta = match ols(&design, &y_vec) {
                Ok(b) => b,
                Err(_) => {
                    aic_values.push(f64::INFINITY);
                    bic_values.push(f64::INFINITY);
                    continue;
                }
            };
            let resid = residuals(&design, &y_vec, &beta);
            let sigma2 = ssr(&resid) / n_eff as f64;
            // Gaussian log-likelihood: -T/2 * log(2π σ²) - SSR/(2σ²) ≈ -T/2 * (1 + log(2π σ²))
            let log_like =
                -(n_eff as f64) / 2.0 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());
            let aic = -2.0 * log_like + 2.0 * k;
            let bic = -2.0 * log_like + k * (n_eff as f64).ln();
            aic_values.push(aic);
            bic_values.push(bic);
        }

        let best_lag_aic = aic_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i + 1)
            .unwrap_or(1);
        let best_lag_bic = bic_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i + 1)
            .unwrap_or(1);

        Ok(LagSelectionResult {
            best_lag_aic,
            best_lag_bic,
            aic_values,
            bic_values,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rolling-Window Granger
// ─────────────────────────────────────────────────────────────────────────────

/// Single window of a rolling Granger test.
#[derive(Debug, Clone)]
pub struct RollingGrangerWindow {
    /// Center time index of this window.
    pub center: usize,
    /// F-statistic.
    pub f_statistic: f64,
    /// Approximate p-value.
    pub p_value: f64,
    /// True if significant at the stored alpha level.
    pub is_significant: bool,
}

/// Rolling-window Granger causality test.
///
/// Slides a window of fixed size `window` over the series and tests Granger
/// causality at each position. Useful for detecting time-varying causality.
pub struct RollingGrangerTest {
    /// Window width.
    pub window: usize,
    /// Number of lags.
    pub lags: usize,
    /// Significance threshold.
    pub alpha: f64,
}

impl RollingGrangerTest {
    /// Create RollingGrangerTest.
    pub fn new(window: usize, lags: usize, alpha: f64) -> Self {
        Self {
            window: window.max(10),
            lags: lags.max(1),
            alpha: alpha.clamp(1e-10, 1.0),
        }
    }

    /// Run rolling Granger test of x → y.
    pub fn test(&self, x: &[f64], y: &[f64]) -> Vec<RollingGrangerWindow> {
        let n = x.len().min(y.len());
        if n < self.window {
            return Vec::new();
        }
        let mut results = Vec::new();
        let step = (self.window / 4).max(1);
        let mut start = 0;
        while start + self.window <= n {
            let end = start + self.window;
            let x_win = &x[start..end];
            let y_win = &y[start..end];
            let center = start + self.window / 2;
            match GrangerCausalityTest::test(x_win, y_win, self.lags) {
                Ok(r) => results.push(RollingGrangerWindow {
                    center,
                    f_statistic: r.f_statistic,
                    p_value: r.p_value_approx,
                    is_significant: r.p_value_approx < self.alpha,
                }),
                Err(_) => results.push(RollingGrangerWindow {
                    center,
                    f_statistic: 0.0,
                    p_value: 1.0,
                    is_significant: false,
                }),
            }
            start += step;
        }
        results
    }

    /// Fraction of windows showing significant causality.
    pub fn causal_fraction(&self, windows: &[RollingGrangerWindow]) -> f64 {
        if windows.is_empty() {
            return 0.0;
        }
        windows.iter().filter(|w| w.is_significant).count() as f64 / windows.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multivariate Transfer Entropy
// ─────────────────────────────────────────────────────────────────────────────

/// Multivariate transfer entropy matrix.
#[derive(Debug, Clone)]
pub struct TransferEntropyMatrix {
    /// TE values: `te[i][j]` = TE(variable i → variable j).
    pub te: Vec<Vec<f64>>,
    /// Number of variables.
    pub n_vars: usize,
}

impl TransferEntropyMatrix {
    /// Compute pairwise TE matrix for a multivariate time series.
    ///
    /// `data[t][v]` is variable `v` at time `t`. Uses `n_bins` discretization bins.
    pub fn compute(data: &[Vec<f64>], n_bins: usize, lag: usize) -> CtsResult<Self> {
        if data.is_empty() {
            return Err("TransferEntropyMatrix: empty data".to_string());
        }
        let n_vars = data[0].len();
        if n_vars == 0 {
            return Err("TransferEntropyMatrix: zero variables".to_string());
        }
        let n_t = data.len();
        let bins = n_bins.max(2);
        let lag = lag.max(1);

        // Extract per-variable series
        let series: Vec<Vec<f64>> = (0..n_vars)
            .map(|v| data.iter().map(|row| row.get(v).copied().unwrap_or(0.0)).collect())
            .collect();

        // Discretize each variable
        let disc_series: Vec<Vec<usize>> = series
            .iter()
            .map(|s| {
                let mn = s.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let rng = (mx - mn).max(1e-15);
                s.iter()
                    .map(|&v| (((v - mn) / rng) * (bins as f64 - 1.0)).round() as usize)
                    .collect()
            })
            .collect();

        let mut te = vec![vec![0.0_f64; n_vars]; n_vars];

        for src in 0..n_vars {
            for dst in 0..n_vars {
                if src == dst {
                    continue;
                }
                // TE(src→dst): bins³ histogram
                let mut c_joint = vec![vec![vec![0usize; bins]; bins]; bins];
                let mut c_y_yprev = vec![vec![0usize; bins]; bins];
                let mut c_yprev_xprev = vec![vec![0usize; bins]; bins];
                let mut c_yprev = vec![0usize; bins];

                let n_obs = n_t - lag;
                for t in lag..n_t {
                    let yt = disc_series[dst][t];
                    let yt_p = disc_series[dst][t - lag];
                    let xt_p = disc_series[src][t - lag];
                    c_joint[yt][yt_p][xt_p] += 1;
                    c_y_yprev[yt][yt_p] += 1;
                    c_yprev_xprev[yt_p][xt_p] += 1;
                    c_yprev[yt_p] += 1;
                }

                let nf = n_obs as f64;
                let mut te_val = 0.0_f64;
                for yt in 0..bins {
                    for yt_p in 0..bins {
                        for xt_p in 0..bins {
                            let p_j = c_joint[yt][yt_p][xt_p] as f64 / nf;
                            if p_j < 1e-15 {
                                continue;
                            }
                            let p_yy = c_y_yprev[yt][yt_p] as f64 / nf;
                            let p_yx = c_yprev_xprev[yt_p][xt_p] as f64 / nf;
                            let p_y = c_yprev[yt_p] as f64 / nf;
                            if p_yy < 1e-15 || p_yx < 1e-15 || p_y < 1e-15 {
                                continue;
                            }
                            te_val +=
                                p_j * (p_j * p_y / (p_yy * p_yx)).ln();
                        }
                    }
                }
                te[src][dst] = te_val.max(0.0);
            }
        }

        Ok(Self { te, n_vars })
    }

    /// Returns the total outgoing TE from variable `v` (sum over all targets).
    pub fn outgoing(&self, v: usize) -> f64 {
        if v >= self.n_vars {
            return 0.0;
        }
        self.te[v].iter().sum()
    }

    /// Returns the total incoming TE to variable `v` (sum over all sources).
    pub fn incoming(&self, v: usize) -> f64 {
        if v >= self.n_vars {
            return 0.0;
        }
        (0..self.n_vars).map(|s| self.te[s][v]).sum()
    }

    /// Net causal influence of `v`: outgoing - incoming.
    pub fn net_influence(&self, v: usize) -> f64 {
        self.outgoing(v) - self.incoming(v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Graph Utility
// ─────────────────────────────────────────────────────────────────────────────

/// A simple directed causal graph inferred from pairwise Granger tests.
#[derive(Debug, Clone)]
pub struct CausalTsGraph {
    /// Number of variables.
    pub n_vars: usize,
    /// Adjacency matrix: `edges[i][j]` = true means i Granger-causes j.
    pub edges: Vec<Vec<bool>>,
    /// p-values for each edge.
    pub p_values: Vec<Vec<f64>>,
}

impl CausalTsGraph {
    /// Infer causal graph from pairwise Granger tests at `alpha` significance.
    pub fn from_granger(
        series: &[Vec<f64>],
        lags: usize,
        alpha: f64,
    ) -> CtsResult<Self> {
        let k = series.len();
        if k == 0 {
            return Err("CausalTsGraph: empty series list".to_string());
        }
        let n = series[0].len();
        for s in series {
            if s.len() != n {
                return Err("CausalTsGraph: mismatched series lengths".to_string());
            }
        }
        let mut edges = vec![vec![false; k]; k];
        let mut p_values = vec![vec![1.0_f64; k]; k];

        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue;
                }
                match GrangerCausalityTest::test(&series[i], &series[j], lags) {
                    Ok(r) => {
                        p_values[i][j] = r.p_value_approx;
                        edges[i][j] = r.p_value_approx < alpha;
                    }
                    Err(_) => {
                        p_values[i][j] = 1.0;
                        edges[i][j] = false;
                    }
                }
            }
        }

        Ok(Self {
            n_vars: k,
            edges,
            p_values,
        })
    }

    /// In-degree of variable `v` (number of Granger causes).
    pub fn in_degree(&self, v: usize) -> usize {
        if v >= self.n_vars {
            return 0;
        }
        (0..self.n_vars).filter(|&i| self.edges[i][v]).count()
    }

    /// Out-degree of variable `v` (number of variables it Granger-causes).
    pub fn out_degree(&self, v: usize) -> usize {
        if v >= self.n_vars {
            return 0;
        }
        self.edges[v].iter().filter(|&&e| e).count()
    }

    /// Check if variable `src` is an ancestor of `dst` (reachability via BFS).
    pub fn is_ancestor(&self, src: usize, dst: usize) -> bool {
        if src >= self.n_vars || dst >= self.n_vars {
            return false;
        }
        let mut visited = vec![false; self.n_vars];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(src);
        while let Some(curr) = queue.pop_front() {
            if curr == dst {
                return true;
            }
            if visited[curr] {
                continue;
            }
            visited[curr] = true;
            for j in 0..self.n_vars {
                if self.edges[curr][j] && !visited[j] {
                    queue.push_back(j);
                }
            }
        }
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Time Series Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for causal time series methods.
#[derive(Debug, Clone)]
pub struct CausalTsMetrics {
    /// True positive rate (sensitivity) for causal edge detection.
    pub sensitivity: f64,
    /// True negative rate (specificity) for causal edge detection.
    pub specificity: f64,
    /// F1 score for causal edge detection.
    pub f1: f64,
    /// Mean absolute error of counterfactual predictions.
    pub cf_mae: f64,
    /// Root mean squared error of counterfactual predictions.
    pub cf_rmse: f64,
}

impl CausalTsMetrics {
    /// Compute metrics from predicted and true causal edges plus counterfactual errors.
    pub fn compute(
        pred_edges: &[Vec<bool>],
        true_edges: &[Vec<bool>],
        cf_pred: &[f64],
        cf_true: &[f64],
    ) -> Self {
        let k = pred_edges.len().min(true_edges.len());
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        let mut fn_ = 0usize;

        for i in 0..k {
            let pj = pred_edges[i].len().min(true_edges[i].len());
            for j in 0..pj {
                match (pred_edges[i][j], true_edges[i][j]) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => tn += 1,
                }
            }
        }

        let sensitivity = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let specificity = if tn + fp > 0 {
            tn as f64 / (tn + fp) as f64
        } else {
            0.0
        };
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let f1 = if precision + sensitivity > 1e-15 {
            2.0 * precision * sensitivity / (precision + sensitivity)
        } else {
            0.0
        };

        let n_cf = cf_pred.len().min(cf_true.len());
        let cf_mae = if n_cf > 0 {
            cf_pred[..n_cf]
                .iter()
                .zip(&cf_true[..n_cf])
                .map(|(p, t)| (p - t).abs())
                .sum::<f64>()
                / n_cf as f64
        } else {
            0.0
        };
        let cf_rmse = if n_cf > 0 {
            (cf_pred[..n_cf]
                .iter()
                .zip(&cf_true[..n_cf])
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n_cf as f64)
                .sqrt()
        } else {
            0.0
        };

        Self {
            sensitivity,
            specificity,
            f1,
            cf_mae,
            cf_rmse,
        }
    }
}

