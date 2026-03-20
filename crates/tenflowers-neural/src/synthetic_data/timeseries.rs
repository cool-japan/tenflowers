//! Time-series simulation: ARIMA, GARCH, OU, fBm, Multivariate GBM.

use super::math::{cholesky, lower_tri_mv, sample_normal};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use tenflowers_core::{Result, TensorError};

/// ARIMA(p, d, q) time-series simulator.
pub struct ArimaSimulator;

impl ArimaSimulator {
    /// Simulate an ARIMA series of length `n`.
    ///
    /// - `ar_params`: AR coefficients φ₁…φₚ
    /// - `d`: integration order
    /// - `ma_params`: MA coefficients θ₁…θ_q
    /// - `sigma`: noise standard deviation
    pub fn simulate(
        n: usize,
        ar_params: &[f64],
        d: usize,
        ma_params: &[f64],
        sigma: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let p = ar_params.len();
        let q = ma_params.len();
        let burn_in = 100_usize;
        let total = n + burn_in;
        let mut eps = vec![0.0_f64; total + q];
        for e in eps.iter_mut() {
            *e = sigma * sample_normal(&mut rng);
        }
        let mut y = vec![0.0_f64; total + p];
        for t in 0..total {
            let ti = t + p;
            let mut val = eps[t + q];
            for (i, &phi) in ar_params.iter().enumerate() {
                val += phi * y[ti - 1 - i];
            }
            for (i, &theta) in ma_params.iter().enumerate() {
                val += theta * eps[t + q - 1 - i];
            }
            y[ti] = val;
        }
        let stationary: Vec<f64> = y[p..].to_vec();
        let mut result = stationary[burn_in..].to_vec();
        for _ in 0..d {
            let mut integrated = vec![0.0_f64; result.len()];
            integrated[0] = result[0];
            for i in 1..result.len() {
                integrated[i] = integrated[i - 1] + result[i];
            }
            result = integrated;
        }
        result
    }
}

/// GARCH(1,1) variance process simulator.
pub struct GarchSimulator;

impl GarchSimulator {
    /// Simulate `n` steps. Returns `(returns, volatilities)`.
    ///
    /// Model: σ²ₜ = ω + α·εₜ₋₁² + β·σ²ₜ₋₁
    pub fn simulate(
        n: usize,
        omega: f64,
        alpha: f64,
        beta: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let long_run_var = omega / (1.0 - alpha - beta).max(1e-6);
        let mut sigma2 = long_run_var;
        let mut returns = Vec::with_capacity(n);
        let mut vols = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_normal(&mut rng);
            let sig = sigma2.sqrt();
            let r = sig * z;
            returns.push(r);
            vols.push(sig);
            sigma2 = (omega + alpha * r * r + beta * sigma2).max(1e-10);
        }
        (returns, vols)
    }
}

/// Ornstein-Uhlenbeck mean-reverting SDE simulator.
///
/// SDE: dX = θ(μ - X)dt + σdW
pub struct OrnsteinUhlenbeck;

impl OrnsteinUhlenbeck {
    /// Euler-Maruyama simulation of length `n`.
    pub fn simulate(
        n: usize,
        theta: f64,
        mu: f64,
        sigma: f64,
        dt: f64,
        x0: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = x0;
        let sqrt_dt = dt.sqrt();
        let mut path = Vec::with_capacity(n);
        path.push(x);
        for _ in 1..n {
            let dw = sample_normal(&mut rng) * sqrt_dt;
            x += theta * (mu - x) * dt + sigma * dw;
            path.push(x);
        }
        path
    }
}

/// Fractional Brownian Motion via Cholesky method (O(n²), suitable for small n).
pub struct FractionalBrownianMotion;

impl FractionalBrownianMotion {
    /// Simulate an fBm path of length `n` with Hurst exponent `H ∈ (0, 1)`.
    pub fn simulate(n: usize, h: f64, seed: u64) -> Result<Vec<f64>> {
        if n == 0 {
            return Ok(vec![]);
        }
        if h <= 0.0 || h >= 1.0 {
            return Err(TensorError::invalid_argument_op(
                "FractionalBrownianMotion::simulate",
                "H must be in (0, 1)",
            ));
        }
        let cov: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = (idx / n) as f64 + 1.0;
                let j = (idx % n) as f64 + 1.0;
                0.5 * (i.powf(2.0 * h) + j.powf(2.0 * h) - (i - j).abs().powf(2.0 * h))
            })
            .collect();
        let l = cholesky(&cov, n)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let z: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng)).collect();
        let path = lower_tri_mv(&l, &z, n);
        Ok(path)
    }
}

/// Multivariate Geometric Brownian Motion with correlation.
pub struct MultivariateGbm;

impl MultivariateGbm {
    /// Simulate correlated GBM paths.
    ///
    /// Returns `Vec<Vec<f64>>` of shape `[d][n+1]` (paths including S₀).
    pub fn simulate(
        n: usize,
        mu: &[f64],
        sigma: &[f64],
        corr: &[f64],
        dt: f64,
        s0: &[f64],
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        let d = mu.len();
        if sigma.len() != d || s0.len() != d || corr.len() != d * d {
            return Err(TensorError::invalid_argument_op(
                "MultivariateGbm::simulate",
                "dimension mismatch",
            ));
        }
        let l = cholesky(corr, d)?;
        let sqrt_dt = dt.sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut paths: Vec<Vec<f64>> = s0.iter().map(|&s| vec![s]).collect();
        let mut prices = s0.to_vec();
        for _ in 0..n {
            let z_raw: Vec<f64> = (0..d).map(|_| sample_normal(&mut rng)).collect();
            let z_corr = lower_tri_mv(&l, &z_raw, d);
            let new_prices: Vec<f64> = (0..d)
                .map(|i| {
                    prices[i]
                        * ((mu[i] - 0.5 * sigma[i] * sigma[i]) * dt
                            + sigma[i] * sqrt_dt * z_corr[i])
                            .exp()
                })
                .collect();
            for (i, &p) in new_prices.iter().enumerate() {
                paths[i].push(p);
            }
            prices = new_prices;
        }
        Ok(paths)
    }
}
