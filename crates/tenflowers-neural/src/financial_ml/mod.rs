//! Financial ML & Advanced Time Series — TenfloweRS Neural.
//!
//! Implements deep learning models for financial data, regime detection,
//! risk models, algorithmic trading, and temporal fusion forecasting.

pub mod forecasting;
pub mod order_book;
pub mod regime;
pub mod risk;
mod tests;
pub mod trading;

// ── Order book / deep learning ────────────────────────────────────────────────
pub use order_book::{
    AlphaFactorNet, DeepLobModel, LobbEncoder, OrderBookEncoder, PortfolioOptimizer,
};

// ── Regime detection ──────────────────────────────────────────────────────────
pub use regime::{
    ChangePointDetector, HiddenMarkovModel, MarketRegimeClassifier, RegimeSwitchingModel,
    VolatilityRegimeDetector,
};

// ── Risk models ───────────────────────────────────────────────────────────────
pub use risk::{CorrelationRiskModel, CvarCalculator, HistoricalVaR, MaxDrawdown, ParametricVaR};

// ── Algorithmic trading ───────────────────────────────────────────────────────
pub use trading::{
    BacktestEngine, BacktestResult, ExecutionSimulator, MeanReversionStrategy, MomentumStrategy,
};

// ── Forecasting ───────────────────────────────────────────────────────────────
pub use forecasting::{
    ForecastMetrics, ForecastScores, FrequencyDomainForecaster, NHitsLayer, PatchTsT, TimeMixer,
};
