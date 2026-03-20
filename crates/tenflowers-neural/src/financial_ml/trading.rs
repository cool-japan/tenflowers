//! Algorithmic trading: BacktestEngine, MomentumStrategy, MeanReversionStrategy, ExecutionSimulator.

use super::order_book::FinResult;
use super::risk::MaxDrawdown;

// ─────────────────────────────────────────────────────────────────────────────
// Backtest
// ─────────────────────────────────────────────────────────────────────────────

/// Results of a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
}

/// Simple event-driven backtest engine.
#[derive(Debug, Clone)]
pub struct BacktestEngine {
    pub risk_free_rate: f64,
    pub transaction_cost: f64,
}

impl BacktestEngine {
    pub fn new(risk_free_rate: f64, transaction_cost: f64) -> Self {
        Self {
            risk_free_rate,
            transaction_cost,
        }
    }

    /// Run backtest.
    ///
    /// * `prices`  – price series
    /// * `signals` – +1 long, 0 flat, -1 short (same length as prices)
    pub fn run(&self, prices: &[f64], signals: &[i32]) -> FinResult<BacktestResult> {
        let n = prices.len();
        if n < 2 || signals.len() != n {
            return Err("BacktestEngine: prices/signals length mismatch".to_string());
        }
        let mut portfolio_values = vec![1.0_f64; n];
        let mut n_trades = 0usize;
        let mut prev_sig = signals[0];
        let mut cash = 1.0_f64;

        for i in 1..n {
            let sig = signals[i - 1];
            if sig != prev_sig {
                n_trades += 1;
                cash *= 1.0 - self.transaction_cost;
                prev_sig = sig;
            }
            let ret = (prices[i] - prices[i - 1]) / prices[i - 1].abs().max(1e-12);
            cash *= 1.0 + sig as f64 * ret;
            portfolio_values[i] = cash;
        }

        let total_return = cash - 1.0;
        let daily_rets: Vec<f64> = (1..n)
            .map(|i| {
                (portfolio_values[i] - portfolio_values[i - 1])
                    / portfolio_values[i - 1].abs().max(1e-12)
            })
            .collect();
        let mean_ret = daily_rets.iter().sum::<f64>() / daily_rets.len() as f64;
        let var_ret = daily_rets
            .iter()
            .map(|&r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / daily_rets.len() as f64;
        let std_ret = var_ret.sqrt().max(1e-12);
        let sharpe = (mean_ret - self.risk_free_rate / 252.0) / std_ret * 252.0_f64.sqrt();

        let downside: Vec<f64> = daily_rets
            .iter()
            .filter(|&&r| r < 0.0).copied()
            .collect();
        let downside_var = if downside.is_empty() {
            1e-12
        } else {
            downside.iter().map(|&r| r * r).sum::<f64>() / downside.len() as f64
        };
        let sortino = (mean_ret - self.risk_free_rate / 252.0) / downside_var.sqrt().max(1e-12)
            * 252.0_f64.sqrt();

        let (max_dd, _, _) = MaxDrawdown.compute(&portfolio_values)?;

        Ok(BacktestResult {
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            n_trades,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Momentum Strategy
// ─────────────────────────────────────────────────────────────────────────────

/// Dual momentum strategy: 12-month minus 1-month return signal.
#[derive(Debug, Clone)]
pub struct MomentumStrategy {
    pub short_window: usize,
    pub long_window: usize,
}

impl MomentumStrategy {
    pub fn new() -> Self {
        Self {
            short_window: 21,
            long_window: 252,
        }
    }

    /// Generate signals (+1/-1/0) for each price in the series.
    pub fn signals(&self, prices: &[f64]) -> FinResult<Vec<i32>> {
        let n = prices.len();
        let lw = self.long_window;
        let sw = self.short_window;
        if n < lw {
            return Err("MomentumStrategy: not enough price history".to_string());
        }
        let mut sigs = vec![0i32; n];
        for i in lw..n {
            let long_ret = (prices[i] - prices[i - lw]) / prices[i - lw].abs().max(1e-12);
            let short_ret = (prices[i] - prices[i - sw]) / prices[i - sw].abs().max(1e-12);
            let momentum = long_ret - short_ret;
            sigs[i] = if momentum > 0.0 {
                1
            } else if momentum < 0.0 {
                -1
            } else {
                0
            };
        }
        Ok(sigs)
    }
}

impl Default for MomentumStrategy {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mean Reversion Strategy
// ─────────────────────────────────────────────────────────────────────────────

/// Bollinger band mean-reversion strategy.
#[derive(Debug, Clone)]
pub struct MeanReversionStrategy {
    pub window: usize,
    pub n_std: f64,
}

impl MeanReversionStrategy {
    pub fn new(window: usize, n_std: f64) -> Self {
        Self { window, n_std }
    }

    /// Generate signals (+1 buy, -1 sell, 0 neutral) based on z-score.
    pub fn signals(&self, prices: &[f64]) -> FinResult<Vec<i32>> {
        let n = prices.len();
        if n < self.window {
            return Err("MeanReversionStrategy: not enough data".to_string());
        }
        let mut sigs = vec![0i32; n];
        for i in self.window..n {
            let window = &prices[i - self.window..i];
            let mean = window.iter().sum::<f64>() / self.window as f64;
            let std = (window.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                / self.window as f64)
                .sqrt()
                .max(1e-12);
            let z = (prices[i] - mean) / std;
            sigs[i] = if z < -self.n_std {
                1
            } else if z > self.n_std {
                -1
            } else {
                0
            };
        }
        Ok(sigs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution Simulator
// ─────────────────────────────────────────────────────────────────────────────

/// Simulates order execution with market impact.
#[derive(Debug, Clone)]
pub struct ExecutionSimulator {
    /// Market impact coefficient (sqrt model).
    pub impact_coeff: f64,
    /// Average daily volume for normalisation.
    pub adv: f64,
}

impl ExecutionSimulator {
    pub fn new(impact_coeff: f64, adv: f64) -> Self {
        Self {
            impact_coeff,
            adv: adv.max(1.0),
        }
    }

    /// Execute an order. Returns fill price accounting for spread and impact.
    ///
    /// * `order_size` – signed (positive = buy, negative = sell)
    /// * `price`      – current mid price
    /// * `spread`     – bid-ask spread
    pub fn execute(&self, order_size: f64, price: f64, spread: f64) -> f64 {
        let half_spread = spread / 2.0;
        let participation = (order_size.abs() / self.adv).min(1.0);
        let impact = self.impact_coeff * participation.sqrt();
        if order_size > 0.0 {
            price + half_spread + impact * price
        } else {
            price - half_spread - impact * price
        }
    }
}
