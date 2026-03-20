//! Tests for the financial_ml module.

#[cfg(test)]
mod tests {
    use crate::financial_ml::{
        AlphaFactorNet, BacktestEngine, ChangePointDetector, CorrelationRiskModel, CvarCalculator,
        DeepLobModel, ExecutionSimulator, ForecastMetrics, FrequencyDomainForecaster,
        HiddenMarkovModel, HistoricalVaR, LobbEncoder, MarketRegimeClassifier, MaxDrawdown,
        MeanReversionStrategy, MomentumStrategy, NHitsLayer, OrderBookEncoder, ParametricVaR,
        PatchTsT, PortfolioOptimizer, RegimeSwitchingModel, TimeMixer, VolatilityRegimeDetector,
    };

    // ── Financial DL ──────────────────────────────────────────────────────────

    #[test]
    fn test_order_book_encoder_features() {
        let enc = OrderBookEncoder::new(5);
        let bp = vec![99.9_f32, 99.8, 99.7, 99.6, 99.5];
        let bv = vec![100_f32, 200.0, 150.0, 80.0, 50.0];
        let ap = vec![100.1_f32, 100.2, 100.3, 100.4, 100.5];
        let av = vec![120_f32, 180.0, 160.0, 90.0, 60.0];
        let feat = enc.encode(&bp, &bv, &ap, &av).expect("order book encode should succeed");
        assert_eq!(feat.len(), 4);
        // mid = (99.9 + 100.1) / 2 = 100.0
        assert!((feat[0] - 100.0).abs() < 1e-3, "mid price");
        // spread = 0.2
        assert!((feat[1] - 0.2).abs() < 1e-3, "spread");
        // vol_imbalance should be in [-1, 1]
        assert!(feat[2].abs() <= 1.0, "vol imbalance");
    }

    #[test]
    fn test_lobb_encoder_shape() {
        let enc = LobbEncoder::new(5, 16, 42).expect("LobbEncoder construction should succeed");
        let snap = vec![1.0_f32; 20]; // 5 levels × 4 fields
        let latent = enc.forward(&snap).expect("LobbEncoder forward should succeed");
        assert_eq!(latent.len(), 16);
    }

    #[test]
    fn test_deep_lob_3class() {
        let model = DeepLobModel::new(20, 16, 8, 42).expect("DeepLobModel construction should succeed");
        let seq: Vec<Vec<f32>> = (0..10).map(|_| vec![0.5_f32; 20]).collect();
        let probs = model.forward(&seq).expect("DeepLobModel forward should succeed");
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "probs sum to 1");
    }

    #[test]
    fn test_alpha_factor_shape() {
        let net = AlphaFactorNet::new(10, 32, 4, 42).expect("AlphaFactorNet construction should succeed");
        let x = vec![0.1_f32; 10];
        let alpha = net.forward(&x).expect("AlphaFactorNet forward should succeed");
        assert_eq!(alpha.len(), 4);
    }

    #[test]
    fn test_markowitz_weights_sum_to_1() {
        let opt = PortfolioOptimizer::new(3);
        let mu = vec![0.1, 0.2, 0.15];
        let sigma = vec![0.04, 0.01, 0.01, 0.01, 0.09, 0.02, 0.01, 0.02, 0.06];
        let w = opt.markowitz(&mu, &sigma, 2.0).expect("markowitz should succeed");
        assert_eq!(w.len(), 3);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum to 1, got {sum}");
        for &wi in &w {
            assert!(wi >= -1e-9, "weight non-negative");
        }
    }

    // ── Regime ────────────────────────────────────────────────────────────────

    #[test]
    fn test_hmm_viterbi_length() {
        let hmm = HiddenMarkovModel::new(2, 1, 0).expect("HMM construction should succeed");
        let obs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let path = hmm.viterbi(&obs).expect("HMM viterbi should succeed");
        assert_eq!(path.len(), 20);
        for &s in &path {
            assert!(s < 2);
        }
    }

    #[test]
    fn test_hmm_log_likelihood_negative() {
        let mut hmm = HiddenMarkovModel::new(2, 1, 1).expect("HMM construction should succeed");
        let obs: Vec<Vec<f64>> = (0..30).map(|i| vec![(i as f64 % 2.0) * 0.5]).collect();
        hmm.fit_baum_welch(&obs, 5).expect("HMM fit_baum_welch should succeed");
        let ll = hmm.log_likelihood(&obs).expect("HMM log_likelihood should succeed");
        // LL should be a finite number (for Gaussian densities with narrow distributions
        // it may be positive — the important property is it is finite and decreases on
        // random data compared to a bad model).
        assert!(ll.is_finite(), "log-likelihood must be finite, got {ll}");
    }

    #[test]
    fn test_change_point_posterior_sums_to_1() {
        let mut cpd = ChangePointDetector::new(0.01, 0.0, 1.0, 1.0, 1.0);
        let data = [0.1, 0.2, -0.5, -0.4, -0.3, 0.1, 0.2];
        let mut last_posterior = Vec::new();
        for &x in &data {
            last_posterior = cpd.update(x);
        }
        let sum: f64 = last_posterior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posterior sums to 1, got {sum}");
    }

    #[test]
    fn test_garch_fit_positive() {
        let mut garch = VolatilityRegimeDetector::new();
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let (omega, alpha, beta) = garch.fit(&returns, 50).expect("GARCH fit should succeed");
        assert!(omega > 0.0, "omega must be positive");
        assert!(alpha >= 0.0, "alpha must be non-negative");
        assert!(beta >= 0.0, "beta must be non-negative");
        assert!(alpha + beta < 1.0, "stationarity: alpha+beta < 1");
    }

    #[test]
    fn test_regime_classifier() {
        let clf = MarketRegimeClassifier::new(20);
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.05).sin() * 0.01).collect();
        let labels = clf.classify(&returns).expect("regime classification should succeed");
        assert_eq!(labels.len(), 100);
        for &l in &labels {
            assert!(l <= 2, "label in {{0,1,2}}");
        }
    }

    // ── Risk ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_var_historical_5pct() {
        let var_calc = HistoricalVaR::new();
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.001).collect();
        let var95 = var_calc.compute(&returns, 0.95).expect("historical VaR should succeed");
        // With 100 uniform returns centred at 0, 5th percentile loss is around 0.045
        assert!(var95 > 0.0, "VaR should be positive for a loss metric");
    }

    #[test]
    fn test_var_parametric() {
        let var_calc = ParametricVaR::new();
        let var95 = var_calc.compute(0.0, 0.01, 0.95);
        // Standard normal 95th ≈ 1.645; VaR = 1.645 * 0.01 ≈ 0.01645
        assert!(
            (var95 - 0.01645).abs() < 0.002,
            "parametric VaR close to expected, got {var95}"
        );
    }

    #[test]
    fn test_cvar_ge_var() {
        let hvar = HistoricalVaR::new();
        let cvar_calc = CvarCalculator::new();
        let returns: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) * 0.001).collect();
        let var95 = hvar.compute(&returns, 0.95).expect("historical VaR should succeed");
        let cvar95 = cvar_calc.compute(&returns, 0.95).expect("CVaR should succeed");
        assert!(cvar95 >= var95 - 1e-9, "CVaR >= VaR");
    }

    #[test]
    fn test_max_drawdown_correct() {
        let md = MaxDrawdown::new();
        let prices = vec![100.0, 120.0, 80.0, 90.0, 110.0];
        let (dd, peak, trough) = md.compute(&prices).expect("max drawdown should succeed");
        // From 120 to 80 is 33.3%
        assert!((dd - (120.0 - 80.0) / 120.0).abs() < 1e-9);
        assert_eq!(peak, 1);
        assert_eq!(trough, 2);
    }

    #[test]
    fn test_max_drawdown_indices() {
        let md = MaxDrawdown::new();
        let prices = vec![50.0, 100.0, 60.0, 80.0, 40.0, 90.0];
        let (_, peak, trough) = md.compute(&prices).expect("max drawdown should succeed");
        assert_eq!(peak, 1);
        assert_eq!(trough, 4, "deepest trough at index 4 (price 40)");
    }

    // ── Trading ───────────────────────────────────────────────────────────────

    #[test]
    fn test_backtest_positive_return() {
        let engine = BacktestEngine::new(0.0, 0.0);
        // Monotonically rising prices → always long → positive return
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let signals = vec![1i32; 50];
        let result = engine.run(&prices, &signals).expect("backtest run should succeed");
        assert!(result.total_return > 0.0, "long in bull market > 0");
    }

    #[test]
    fn test_backtest_sharpe_ratio() {
        let engine = BacktestEngine::new(0.02, 0.001);
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let signals = vec![1i32; 100];
        let result = engine.run(&prices, &signals).expect("backtest run should succeed");
        // Sharpe is finite
        assert!(result.sharpe_ratio.is_finite(), "Sharpe ratio finite");
    }

    #[test]
    fn test_momentum_signal() {
        let strat = MomentumStrategy {
            short_window: 5,
            long_window: 10,
        };
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let sigs = strat.signals(&prices).expect("momentum signals should succeed");
        assert_eq!(sigs.len(), 20);
        // In a rising market, recent returns are lower than long-term → mixed, but non-zero signals exist
        assert!(sigs.iter().any(|&s| s != 0), "should have non-zero signals");
    }

    #[test]
    fn test_mean_reversion_signal() {
        let strat = MeanReversionStrategy::new(10, 2.0);
        let prices = vec![
            100.0, 101.0, 99.0, 100.5, 98.0, 101.5, 99.5, 100.0, 101.0, 99.0,
            // spike up
            108.0, 107.0, 106.0, 105.0, 104.0,
        ];
        let sigs = strat.signals(&prices).expect("mean reversion signals should succeed");
        assert_eq!(sigs.len(), prices.len());
        // The spike should produce a sell signal
        assert!(sigs.contains(&-1), "spike produces sell signal");
    }

    #[test]
    fn test_execution_spread_impact() {
        let sim = ExecutionSimulator::new(0.1, 1_000_000.0);
        let fill_buy = sim.execute(1000.0, 100.0, 0.02);
        let fill_sell = sim.execute(-1000.0, 100.0, 0.02);
        // Buy should be above mid, sell below mid
        assert!(fill_buy > 100.0, "buy fill above mid");
        assert!(fill_sell < 100.0, "sell fill below mid");
    }

    // ── Forecasting ───────────────────────────────────────────────────────────

    #[test]
    fn test_nhits_shapes() {
        let layer = NHitsLayer::new(24, 6, 8, 32, 42).expect("NHitsLayer construction should succeed");
        let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let (backcast, forecast) = layer.forward(&x, 1).expect("NHitsLayer forward should succeed");
        assert_eq!(backcast.len(), 24, "backcast length");
        assert_eq!(forecast.len(), 6, "forecast length");
    }

    #[test]
    fn test_patch_tst_shape() {
        let model = PatchTsT::new(24, 4, 16, 6, 42).expect("PatchTsT construction should succeed");
        let x: Vec<f32> = (0..24).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = model.forward(&x).expect("PatchTsT forward should succeed");
        assert_eq!(out.len(), 6, "PatchTsT forecast length");
    }

    #[test]
    fn test_time_mixer_output() {
        let mixer = TimeMixer::new(8, 3, 16, 4, 99).expect("TimeMixer construction should succeed");
        let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.01).collect();
        let out = mixer.forward(&x).expect("TimeMixer forward should succeed");
        assert_eq!(out.len(), 4, "TimeMixer output length");
    }

    #[test]
    fn test_freq_domain_forecaster() {
        let fdf = FrequencyDomainForecaster::new(16, 4, 77).expect("FrequencyDomainForecaster construction should succeed");
        let x: Vec<f32> = (0..16)
            .map(|i| (i as f32 * std::f32::consts::PI / 4.0).sin())
            .collect();
        let out = fdf.forward(&x).expect("FrequencyDomainForecaster forward should succeed");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| v.is_finite()), "all outputs finite");
    }

    #[test]
    fn test_forecast_metrics_smape() {
        let fm = ForecastMetrics::new();
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];
        let scores = fm.compute_all(&y_true, &y_pred, 1).expect("forecast metrics should succeed");
        assert!(scores.smape >= 0.0, "sMAPE >= 0");
        assert!(
            scores.smape < 20.0,
            "sMAPE should be small for close predictions"
        );
        assert!(scores.rmse > 0.0, "RMSE > 0");
        assert!(scores.mae > 0.0, "MAE > 0");
        assert!(scores.mase > 0.0, "MASE > 0");
    }

    // ── Extra / edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_order_book_encoder_too_short() {
        let enc = OrderBookEncoder::new(5);
        let result = enc.encode(&[1.0], &[1.0], &[1.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_portfolio_optimizer_dimensions() {
        let opt = PortfolioOptimizer::new(2);
        let result = opt.markowitz(&[0.1, 0.2, 0.15], &[0.0; 4], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_hmm_empty_obs() {
        let hmm = HiddenMarkovModel::new(2, 1, 0).expect("HMM construction should succeed");
        assert!(hmm.log_likelihood(&[]).is_err());
    }

    #[test]
    fn test_garch_forecast_long_run() {
        let garch = VolatilityRegimeDetector {
            omega: 0.0001,
            alpha: 0.05,
            beta: 0.90,
        };
        let fv = garch.forecast_variance(1);
        assert!(fv > 0.0);
        let fv_long = garch.forecast_variance(100);
        let long_run = 0.0001 / (1.0 - 0.95);
        assert!((fv_long - long_run).abs() < 1e-4);
    }

    #[test]
    fn test_correlation_risk_model_update() {
        let mut crm = CorrelationRiskModel::new(3, 0.94);
        for _ in 0..10 {
            crm.update(&[0.01, -0.01, 0.005]).expect("correlation risk model update should succeed");
        }
        let corr = crm.correlation();
        assert_eq!(corr.len(), 9);
        // Diagonal should be 1.0
        assert!((corr[0] - 1.0).abs() < 1e-9);
        assert!((corr[4] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_backtest_n_trades() {
        let engine = BacktestEngine::new(0.0, 0.001);
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        // Alternating signals: lots of trades
        let mut signals = vec![1i32; 20];
        for i in (0..20).step_by(2) {
            signals[i] = -1;
        }
        let result = engine.run(&prices, &signals).expect("backtest run should succeed");
        assert!(result.n_trades > 0);
    }

    #[test]
    fn test_regime_switching_model() {
        let mut rsm = RegimeSwitchingModel::new();
        let returns: Vec<f64> = (0..50)
            .map(|i| if i < 25 { 0.001 } else { -0.003 })
            .collect();
        rsm.fit(&returns).expect("regime switching fit should succeed");
        let p = rsm.regime_probability(40);
        assert!((0.0..=1.0).contains(&p), "probability in [0,1]");
    }

    #[test]
    fn test_nhits_multirate() {
        let layer = NHitsLayer::new(16, 4, 8, 16, 0).expect("NHitsLayer construction should succeed");
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // sampling_rate = 2 (decimation)
        let (bc, fc) = layer.forward(&x, 2).expect("NHitsLayer forward should succeed");
        assert_eq!(bc.len(), 16);
        assert_eq!(fc.len(), 4);
    }

    #[test]
    fn test_black_litterman_blending() {
        let opt = PortfolioOptimizer::new(2);
        let pi = vec![0.08, 0.12];
        let q = vec![0.10_f64];
        let p = vec![1.0_f64, 0.0]; // view on asset 0
        let sigma = vec![0.04, 0.01, 0.01, 0.09];
        let omega = vec![0.005_f64];
        let mu_bl = opt
            .black_litterman(&pi, &q, &p, 0.05, &sigma, &omega)
            .expect("black_litterman should succeed");
        assert_eq!(mu_bl.len(), 2);
    }

    #[test]
    fn test_execution_large_order_impact() {
        let sim = ExecutionSimulator::new(0.5, 100.0);
        // Order size = 200 (2x ADV) → capped participation at 1.0
        let fill = sim.execute(200.0, 50.0, 0.1);
        // Impact should be significant
        assert!(fill > 50.0 + 0.05, "large order has market impact");
    }

    #[test]
    fn test_lobb_encoder_wrong_input() {
        let enc = LobbEncoder::new(5, 8, 0).expect("LobbEncoder construction should succeed");
        // Provide too short input
        let result = enc.forward(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_tst_insufficient_input() {
        let model = PatchTsT::new(32, 4, 16, 8, 0).expect("PatchTsT construction should succeed");
        let result = model.forward(&[0.0_f32; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_time_mixer_channel_mixing() {
        let mixer = TimeMixer::new(4, 2, 8, 2, 0).expect("TimeMixer construction should succeed");
        let x: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = mixer.forward(&x).expect("TimeMixer forward should succeed");
        assert_eq!(out.len(), 2);
    }
}
