//! Risk models: Historical VaR, Parametric VaR, CVaR, Max Drawdown, Correlation Risk.

use super::order_book::FinResult;

// ─────────────────────────────────────────────────────────────────────────────
// Historical VaR
// ─────────────────────────────────────────────────────────────────────────────

/// Historical simulation Value at Risk.
#[derive(Debug, Clone)]
pub struct HistoricalVaR;

impl HistoricalVaR {
    pub fn new() -> Self {
        Self
    }

    /// Compute VaR at given confidence level (e.g. 0.95 → 5% left tail).
    pub fn compute(&self, returns: &[f64], confidence: f64) -> FinResult<f64> {
        if returns.is_empty() {
            return Err("HistoricalVaR: empty returns".to_string());
        }
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        Ok(-sorted[idx])
    }
}

impl Default for HistoricalVaR {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parametric VaR
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian parametric Value at Risk.
#[derive(Debug, Clone)]
pub struct ParametricVaR;

impl ParametricVaR {
    pub fn new() -> Self {
        Self
    }

    /// Compute VaR given mean, std dev, and confidence level.
    pub fn compute(&self, mu: f64, sigma: f64, confidence: f64) -> f64 {
        // z-score via approximation
        let z = probit(confidence);
        -(mu - z * sigma)
    }
}

impl Default for ParametricVaR {
    fn default() -> Self {
        Self::new()
    }
}

/// Rational approximation of the probit (inverse normal CDF).
fn probit(p: f64) -> f64 {
    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let t = if p <= 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c = [2.515517, 0.802853, 0.010328_f64];
    let d = [1.432788, 0.189269, 0.001308_f64];
    let num = c[0] + c[1] * t + c[2] * t * t;
    let den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
    let z = t - num / den;
    if p <= 0.5 {
        -z
    } else {
        z
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CVaR
// ─────────────────────────────────────────────────────────────────────────────

/// Conditional VaR (Expected Shortfall).
#[derive(Debug, Clone)]
pub struct CvarCalculator;

impl CvarCalculator {
    pub fn new() -> Self {
        Self
    }

    /// CVaR = mean of losses exceeding VaR.
    pub fn compute(&self, returns: &[f64], confidence: f64) -> FinResult<f64> {
        if returns.is_empty() {
            return Err("CVaR: empty returns".to_string());
        }
        let var = HistoricalVaR.compute(returns, confidence)?;
        let tail: Vec<f64> = returns
            .iter()
            .filter(|&&r| -r >= var)
            .map(|&r| -r)
            .collect();
        if tail.is_empty() {
            return Ok(var);
        }
        Ok(tail.iter().sum::<f64>() / tail.len() as f64)
    }
}

impl Default for CvarCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Max Drawdown
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum drawdown calculator.
#[derive(Debug, Clone)]
pub struct MaxDrawdown;

impl MaxDrawdown {
    pub fn new() -> Self {
        Self
    }

    /// Returns `(max_drawdown, peak_index, trough_index)`.
    pub fn compute(&self, prices: &[f64]) -> FinResult<(f64, usize, usize)> {
        if prices.len() < 2 {
            return Err("MaxDrawdown: need at least 2 prices".to_string());
        }
        let mut max_dd = 0.0_f64;
        let mut peak_idx = 0usize;
        let mut trough_idx = 0usize;
        let mut cur_peak = prices[0];
        let mut cur_peak_idx = 0usize;

        for i in 1..prices.len() {
            if prices[i] > cur_peak {
                cur_peak = prices[i];
                cur_peak_idx = i;
            }
            let dd = (cur_peak - prices[i]) / cur_peak.abs().max(1e-12);
            if dd > max_dd {
                max_dd = dd;
                peak_idx = cur_peak_idx;
                trough_idx = i;
            }
        }
        Ok((max_dd, peak_idx, trough_idx))
    }
}

impl Default for MaxDrawdown {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Correlation Risk Model
// ─────────────────────────────────────────────────────────────────────────────

/// Dynamic Conditional Correlation (DCC) approximation via EMA rolling correlations.
#[derive(Debug, Clone)]
pub struct CorrelationRiskModel {
    pub n_assets: usize,
    pub ema_decay: f64,
    /// EMA-smoothed covariance matrix (n_assets × n_assets).
    pub cov: Vec<f64>,
}

impl CorrelationRiskModel {
    pub fn new(n_assets: usize, ema_decay: f64) -> Self {
        let decay = ema_decay.clamp(0.01, 0.999);
        Self {
            n_assets,
            ema_decay: decay,
            cov: vec![0.0_f64; n_assets * n_assets],
        }
    }

    /// Update with a new cross-sectional return vector (length n_assets).
    pub fn update(&mut self, returns: &[f64]) -> FinResult<()> {
        let n = self.n_assets;
        if returns.len() != n {
            return Err("CorrelationRiskModel: wrong return vector length".to_string());
        }
        let d = self.ema_decay;
        for i in 0..n {
            for j in 0..n {
                self.cov[i * n + j] = d * self.cov[i * n + j] + (1.0 - d) * returns[i] * returns[j];
            }
        }
        Ok(())
    }

    /// Extract current correlation matrix.
    pub fn correlation(&self) -> Vec<f64> {
        let n = self.n_assets;
        let mut corr = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let si = self.cov[i * n + i].sqrt().max(1e-12);
                let sj = self.cov[j * n + j].sqrt().max(1e-12);
                corr[i * n + j] = self.cov[i * n + j] / (si * sj);
            }
        }
        corr
    }
}
