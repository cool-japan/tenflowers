//! Adaptive prefetch auto-tuning policy
//!
//! This module provides automatic tuning of prefetch parameters based on
//! runtime performance metrics. It dynamically adjusts prefetch depth,
//! aggressiveness, and strategies to optimize for the current access pattern.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Adaptive prefetch tuning policy
#[derive(Debug, Clone)]
pub struct AdaptivePrefetchPolicy {
    /// Current prefetch depth (number of items to prefetch ahead)
    pub prefetch_depth: usize,
    /// Current prefetch aggressiveness (0.0 = conservative, 1.0 = aggressive)
    pub aggressiveness: f64,
    /// Minimum prefetch depth allowed
    pub min_depth: usize,
    /// Maximum prefetch depth allowed
    pub max_depth: usize,
    /// Target hit rate (0.0 - 1.0)
    pub target_hit_rate: f64,
    /// Current adaptation strategy
    pub strategy: AdaptationStrategy,
}

/// Strategy for adapting prefetch parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationStrategy {
    /// Conservative: Increase prefetch only when high confidence
    Conservative,
    /// Balanced: Moderate adaptation based on metrics
    Balanced,
    /// Aggressive: Rapidly adapt to maximize performance
    Aggressive,
    /// Custom: Use custom thresholds
    Custom,
}

/// Performance metrics for tuning decisions
#[derive(Debug, Clone)]
pub struct PrefetchMetrics {
    /// Hit rate (successful prefetches / total prefetches)
    pub hit_rate: f64,
    /// Miss rate (unsuccessful prefetches / total prefetches)
    pub miss_rate: f64,
    /// Waste rate (prefetched but unused / total prefetches)
    pub waste_rate: f64,
    /// Average latency when data is in cache (microseconds)
    pub cache_hit_latency_us: f64,
    /// Average latency when data must be fetched (microseconds)
    pub cache_miss_latency_us: f64,
    /// Memory pressure (0.0 = low, 1.0 = high)
    pub memory_pressure: f64,
    /// Access pattern predictability (0.0 = random, 1.0 = perfectly predictable)
    pub predictability: f64,
}

impl Default for PrefetchMetrics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            miss_rate: 0.0,
            waste_rate: 0.0,
            cache_hit_latency_us: 0.0,
            cache_miss_latency_us: 0.0,
            memory_pressure: 0.0,
            predictability: 0.0,
        }
    }
}

/// Tuning decision made by the adaptive policy
#[derive(Debug, Clone)]
pub struct TuningDecision {
    /// New prefetch depth
    pub new_depth: usize,
    /// New aggressiveness level
    pub new_aggressiveness: f64,
    /// Reason for the decision
    pub reason: String,
    /// Confidence in this decision (0.0 - 1.0)
    pub confidence: f64,
}

/// Adaptive prefetch tuner that adjusts parameters automatically
pub struct AdaptivePrefetchTuner {
    /// Current policy configuration
    policy: Arc<Mutex<AdaptivePrefetchPolicy>>,
    /// Historical metrics for trend analysis
    metrics_history: Arc<Mutex<VecDeque<(Instant, PrefetchMetrics)>>>,
    /// Maximum history size
    max_history_size: usize,
    /// Tuning interval (how often to adjust parameters)
    tuning_interval: Duration,
    /// Last tuning time
    last_tuning: Arc<Mutex<Instant>>,
}

impl AdaptivePrefetchPolicy {
    /// Create a new adaptive policy with defaults
    pub fn new() -> Self {
        Self {
            prefetch_depth: 4,
            aggressiveness: 0.5,
            min_depth: 1,
            max_depth: 32,
            target_hit_rate: 0.8,
            strategy: AdaptationStrategy::Balanced,
        }
    }

    /// Create a conservative policy
    pub fn conservative() -> Self {
        Self {
            prefetch_depth: 2,
            aggressiveness: 0.3,
            min_depth: 1,
            max_depth: 8,
            target_hit_rate: 0.9,
            strategy: AdaptationStrategy::Conservative,
        }
    }

    /// Create an aggressive policy
    pub fn aggressive() -> Self {
        Self {
            prefetch_depth: 8,
            aggressiveness: 0.8,
            min_depth: 2,
            max_depth: 64,
            target_hit_rate: 0.7,
            strategy: AdaptationStrategy::Aggressive,
        }
    }

    /// Create a custom policy
    pub fn custom(min_depth: usize, max_depth: usize, target_hit_rate: f64) -> Self {
        Self {
            prefetch_depth: (min_depth + max_depth) / 2,
            aggressiveness: 0.5,
            min_depth,
            max_depth,
            target_hit_rate: target_hit_rate.clamp(0.0, 1.0),
            strategy: AdaptationStrategy::Custom,
        }
    }
}

impl Default for AdaptivePrefetchPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptivePrefetchTuner {
    /// Create a new adaptive tuner with the given policy
    pub fn new(policy: AdaptivePrefetchPolicy) -> Self {
        Self {
            policy: Arc::new(Mutex::new(policy)),
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            max_history_size: 100,
            tuning_interval: Duration::from_secs(5),
            last_tuning: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Create with custom tuning interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.tuning_interval = interval;
        self
    }

    /// Create with custom history size
    pub fn with_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Update metrics and potentially trigger retuning
    pub fn update_metrics(&self, metrics: PrefetchMetrics) -> Option<TuningDecision> {
        // Record metrics
        let mut history = self.metrics_history.lock().unwrap();
        history.push_back((Instant::now(), metrics.clone()));

        // Maintain history size
        while history.len() > self.max_history_size {
            history.pop_front();
        }
        drop(history);

        // Check if it's time to retune
        let mut last_tuning = self.last_tuning.lock().unwrap();
        if last_tuning.elapsed() < self.tuning_interval {
            return None;
        }
        *last_tuning = Instant::now();
        drop(last_tuning);

        // Make tuning decision
        let decision = self.make_tuning_decision(&metrics)?;

        // Apply decision
        let mut policy = self.policy.lock().unwrap();
        policy.prefetch_depth = decision.new_depth;
        policy.aggressiveness = decision.new_aggressiveness;
        drop(policy);

        Some(decision)
    }

    /// Make a tuning decision based on current metrics
    fn make_tuning_decision(&self, metrics: &PrefetchMetrics) -> Option<TuningDecision> {
        let policy = self.policy.lock().unwrap();
        let current_depth = policy.prefetch_depth;
        let current_aggressiveness = policy.aggressiveness;
        let strategy = policy.strategy;
        let target_hit_rate = policy.target_hit_rate;
        let min_depth = policy.min_depth;
        let max_depth = policy.max_depth;
        drop(policy);

        // Calculate how far we are from target hit rate
        let hit_rate_delta = metrics.hit_rate - target_hit_rate;

        // Decision logic based on strategy
        let (depth_adjustment, aggressiveness_adjustment, reason, confidence) = match strategy {
            AdaptationStrategy::Conservative => self.conservative_decision(
                metrics,
                hit_rate_delta,
                current_depth,
                min_depth,
                max_depth,
            ),
            AdaptationStrategy::Balanced => {
                self.balanced_decision(metrics, hit_rate_delta, current_depth, min_depth, max_depth)
            }
            AdaptationStrategy::Aggressive => self.aggressive_decision(
                metrics,
                hit_rate_delta,
                current_depth,
                min_depth,
                max_depth,
            ),
            AdaptationStrategy::Custom => {
                self.balanced_decision(metrics, hit_rate_delta, current_depth, min_depth, max_depth)
            }
        };

        let new_depth = (current_depth as i32 + depth_adjustment)
            .max(min_depth as i32)
            .min(max_depth as i32) as usize;

        let new_aggressiveness =
            (current_aggressiveness + aggressiveness_adjustment).clamp(0.0, 1.0);

        // Only return decision if there's a meaningful change
        if new_depth == current_depth && (new_aggressiveness - current_aggressiveness).abs() < 0.05
        {
            return None;
        }

        Some(TuningDecision {
            new_depth,
            new_aggressiveness,
            reason,
            confidence,
        })
    }

    fn conservative_decision(
        &self,
        metrics: &PrefetchMetrics,
        hit_rate_delta: f64,
        current_depth: usize,
        min_depth: usize,
        max_depth: usize,
    ) -> (i32, f64, String, f64) {
        // Conservative: Only increase if hit rate is very good and predictability is high
        if hit_rate_delta > 0.1 && metrics.predictability > 0.7 {
            (
                1,
                0.05,
                "High hit rate with good predictability".to_string(),
                0.8,
            )
        } else if hit_rate_delta < -0.2 {
            (
                -1,
                -0.1,
                "Significant hit rate degradation".to_string(),
                0.9,
            )
        } else if metrics.memory_pressure > 0.8 {
            (-1, -0.05, "High memory pressure".to_string(), 0.95)
        } else {
            (0, 0.0, "No change needed".to_string(), 0.6)
        }
    }

    fn balanced_decision(
        &self,
        metrics: &PrefetchMetrics,
        hit_rate_delta: f64,
        current_depth: usize,
        min_depth: usize,
        max_depth: usize,
    ) -> (i32, f64, String, f64) {
        // Balanced: Moderate adjustments based on multiple factors
        let mut depth_adj = 0i32;
        let mut aggr_adj = 0.0;
        let mut reasons = Vec::new();
        let mut confidence: f64 = 0.5;

        // Hit rate consideration
        if hit_rate_delta > 0.05 {
            depth_adj += 1;
            aggr_adj += 0.1;
            reasons.push("Good hit rate");
            confidence += 0.1;
        } else if hit_rate_delta < -0.1 {
            depth_adj -= 1;
            aggr_adj -= 0.1;
            reasons.push("Poor hit rate");
            confidence += 0.2;
        }

        // Predictability consideration
        if metrics.predictability > 0.6 {
            depth_adj += 1;
            reasons.push("High predictability");
            confidence += 0.15;
        } else if metrics.predictability < 0.3 {
            depth_adj -= 1;
            reasons.push("Low predictability");
            confidence += 0.1;
        }

        // Memory pressure consideration
        if metrics.memory_pressure > 0.7 {
            depth_adj -= 2;
            aggr_adj -= 0.2;
            reasons.push("High memory pressure");
            confidence += 0.2;
        }

        // Waste rate consideration
        if metrics.waste_rate > 0.3 {
            depth_adj -= 1;
            aggr_adj -= 0.15;
            reasons.push("High waste rate");
            confidence += 0.15;
        }

        let reason = if reasons.is_empty() {
            "Maintaining current settings".to_string()
        } else {
            reasons.join(", ")
        };

        (depth_adj, aggr_adj, reason, confidence.min(1.0))
    }

    fn aggressive_decision(
        &self,
        metrics: &PrefetchMetrics,
        hit_rate_delta: f64,
        current_depth: usize,
        min_depth: usize,
        max_depth: usize,
    ) -> (i32, f64, String, f64) {
        // Aggressive: Rapidly adapt to maximize performance
        let (depth_adj, aggr_adj, reason, confidence) =
            if metrics.predictability > 0.5 && hit_rate_delta > -0.1 {
                // Increase aggressively if there's some predictability
                (
                    2,
                    0.2,
                    "Increasing prefetch for better performance".to_string(),
                    0.7,
                )
            } else if hit_rate_delta < -0.15 || metrics.memory_pressure > 0.8 {
                // Pull back if hitting problems
                (
                    -2,
                    -0.2,
                    "Reducing prefetch due to issues".to_string(),
                    0.85,
                )
            } else {
                // Try increasing by default
                (1, 0.1, "Exploring higher prefetch levels".to_string(), 0.6)
            };

        (depth_adj, aggr_adj, reason, confidence)
    }

    /// Get current policy
    pub fn get_policy(&self) -> AdaptivePrefetchPolicy {
        self.policy.lock().unwrap().clone()
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> Vec<(Instant, PrefetchMetrics)> {
        self.metrics_history
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get average metrics over recent history
    pub fn get_average_metrics(&self, window: Duration) -> Option<PrefetchMetrics> {
        let history = self.metrics_history.lock().unwrap();
        let cutoff = Instant::now() - window;

        let recent: Vec<_> = history
            .iter()
            .filter(|(time, _)| *time > cutoff)
            .map(|(_, metrics)| metrics)
            .collect();

        if recent.is_empty() {
            return None;
        }

        let count = recent.len() as f64;
        Some(PrefetchMetrics {
            hit_rate: recent.iter().map(|m| m.hit_rate).sum::<f64>() / count,
            miss_rate: recent.iter().map(|m| m.miss_rate).sum::<f64>() / count,
            waste_rate: recent.iter().map(|m| m.waste_rate).sum::<f64>() / count,
            cache_hit_latency_us: recent.iter().map(|m| m.cache_hit_latency_us).sum::<f64>()
                / count,
            cache_miss_latency_us: recent.iter().map(|m| m.cache_miss_latency_us).sum::<f64>()
                / count,
            memory_pressure: recent.iter().map(|m| m.memory_pressure).sum::<f64>() / count,
            predictability: recent.iter().map(|m| m.predictability).sum::<f64>() / count,
        })
    }

    /// Reset tuner state
    pub fn reset(&self) {
        self.metrics_history.lock().unwrap().clear();
        *self.last_tuning.lock().unwrap() = Instant::now();
    }

    /// Generate tuning report
    pub fn generate_report(&self) -> String {
        let policy = self.policy.lock().unwrap();
        let avg_metrics = self
            .get_average_metrics(Duration::from_secs(60))
            .unwrap_or_default();

        let mut report = String::new();
        report.push_str("=== Adaptive Prefetch Tuning Report ===\n\n");

        report.push_str("## Current Policy\n");
        report.push_str(&format!("  Strategy: {:?}\n", policy.strategy));
        report.push_str(&format!("  Prefetch Depth: {}\n", policy.prefetch_depth));
        report.push_str(&format!("  Aggressiveness: {:.2}\n", policy.aggressiveness));
        report.push_str(&format!(
            "  Depth Range: {} - {}\n",
            policy.min_depth, policy.max_depth
        ));
        report.push_str(&format!(
            "  Target Hit Rate: {:.1}%\n\n",
            policy.target_hit_rate * 100.0
        ));

        report.push_str("## Average Metrics (Last 60s)\n");
        report.push_str(&format!(
            "  Hit Rate: {:.1}%\n",
            avg_metrics.hit_rate * 100.0
        ));
        report.push_str(&format!(
            "  Miss Rate: {:.1}%\n",
            avg_metrics.miss_rate * 100.0
        ));
        report.push_str(&format!(
            "  Waste Rate: {:.1}%\n",
            avg_metrics.waste_rate * 100.0
        ));
        report.push_str(&format!(
            "  Predictability: {:.1}%\n",
            avg_metrics.predictability * 100.0
        ));
        report.push_str(&format!(
            "  Memory Pressure: {:.1}%\n",
            avg_metrics.memory_pressure * 100.0
        ));
        report.push_str(&format!(
            "  Cache Hit Latency: {:.2}μs\n",
            avg_metrics.cache_hit_latency_us
        ));
        report.push_str(&format!(
            "  Cache Miss Latency: {:.2}μs\n",
            avg_metrics.cache_miss_latency_us
        ));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_creation() {
        let policy = AdaptivePrefetchPolicy::new();
        assert_eq!(policy.prefetch_depth, 4);
        assert_eq!(policy.strategy, AdaptationStrategy::Balanced);

        let conservative = AdaptivePrefetchPolicy::conservative();
        assert_eq!(conservative.strategy, AdaptationStrategy::Conservative);
        assert!(conservative.prefetch_depth < policy.prefetch_depth);

        let aggressive = AdaptivePrefetchPolicy::aggressive();
        assert_eq!(aggressive.strategy, AdaptationStrategy::Aggressive);
        assert!(aggressive.prefetch_depth > policy.prefetch_depth);
    }

    #[test]
    fn test_tuner_creation() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy);

        let current_policy = tuner.get_policy();
        assert_eq!(current_policy.prefetch_depth, 4);
    }

    #[test]
    fn test_metrics_update() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy).with_interval(Duration::from_millis(100));

        let metrics = PrefetchMetrics {
            hit_rate: 0.9,
            predictability: 0.8,
            ..Default::default()
        };

        tuner.update_metrics(metrics.clone());

        // Should not retune immediately
        let decision = tuner.update_metrics(metrics);
        assert!(decision.is_none());
    }

    #[test]
    fn test_average_metrics() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy);

        for i in 1..=5 {
            let metrics = PrefetchMetrics {
                hit_rate: i as f64 * 0.1,
                ..Default::default()
            };
            tuner.update_metrics(metrics);
        }

        let avg = tuner.get_average_metrics(Duration::from_secs(60)).unwrap();
        assert!((avg.hit_rate - 0.3).abs() < 0.01); // Average of 0.1, 0.2, 0.3, 0.4, 0.5
    }

    #[test]
    fn test_high_hit_rate_increases_depth() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy).with_interval(Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(2));

        let metrics = PrefetchMetrics {
            hit_rate: 0.95, // Very high hit rate
            predictability: 0.9,
            memory_pressure: 0.1,
            ..Default::default()
        };

        let decision = tuner.update_metrics(metrics);
        if let Some(d) = decision {
            assert!(d.new_depth > 4, "Should increase depth on high hit rate");
        }
    }

    #[test]
    fn test_high_memory_pressure_reduces_depth() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy).with_interval(Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(2));

        let metrics = PrefetchMetrics {
            hit_rate: 0.5,
            memory_pressure: 0.9, // Very high memory pressure
            ..Default::default()
        };

        let decision = tuner.update_metrics(metrics);
        if let Some(d) = decision {
            assert!(
                d.new_depth < 4,
                "Should reduce depth on high memory pressure"
            );
        }
    }

    #[test]
    fn test_reset() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy);

        tuner.update_metrics(PrefetchMetrics::default());
        assert!(!tuner.get_metrics_history().is_empty());

        tuner.reset();
        assert!(tuner.get_metrics_history().is_empty());
    }

    #[test]
    fn test_generate_report() {
        let policy = AdaptivePrefetchPolicy::new();
        let tuner = AdaptivePrefetchTuner::new(policy);

        tuner.update_metrics(PrefetchMetrics {
            hit_rate: 0.85,
            ..Default::default()
        });

        let report = tuner.generate_report();
        assert!(report.contains("Adaptive Prefetch Tuning Report"));
        assert!(report.contains("Prefetch Depth"));
        assert!(report.contains("Hit Rate"));
    }

    #[test]
    fn test_custom_policy() {
        let policy = AdaptivePrefetchPolicy::custom(2, 16, 0.75);
        assert_eq!(policy.min_depth, 2);
        assert_eq!(policy.max_depth, 16);
        assert!((policy.target_hit_rate - 0.75).abs() < 0.01);
        assert_eq!(policy.strategy, AdaptationStrategy::Custom);
    }
}
