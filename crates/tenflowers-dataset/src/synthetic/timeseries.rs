//! Time Series Data Generation
//!
//! This module contains functionality for generating various types of synthetic
//! time series data with different patterns and characteristics.

use super::core::{DatasetGenerator, SyntheticConfig, SyntheticDataset};
use scirs2_core::random::{Rng, RngCore, SeedableRng};
use std::f64::consts::PI;
use tenflowers_core::{Result, Tensor};

impl DatasetGenerator {
    /// Generate time series data with various patterns
    pub fn make_time_series(
        config: SyntheticConfig,
        pattern: TimeSeriesPattern,
        sequence_length: usize,
    ) -> Result<SyntheticDataset<f64>> {
        let mut rng: Box<dyn RngCore> = if let Some(seed) = config.random_seed {
            Box::new(scirs2_core::random::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(scirs2_core::random::rng())
        };

        let mut all_sequences = Vec::new();
        let mut all_targets = Vec::new();

        for _ in 0..config.n_samples {
            let (sequence, target) =
                pattern.generate_sequence(sequence_length, &mut *rng, config.noise_level);
            all_sequences.extend(sequence);
            all_targets.push(target);
        }

        let feature_tensor = Tensor::from_vec(all_sequences, &[config.n_samples, sequence_length])?;
        let label_tensor = Tensor::from_vec(all_targets, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate multivariate time series data
    pub fn make_multivariate_time_series(
        config: SyntheticConfig,
        n_features: usize,
        sequence_length: usize,
        correlation_matrix: Option<Vec<Vec<f64>>>,
    ) -> Result<SyntheticDataset<f64>> {
        let mut rng: Box<dyn RngCore> = if let Some(seed) = config.random_seed {
            Box::new(scirs2_core::random::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(scirs2_core::random::rng())
        };

        let mut all_sequences = Vec::new();
        let mut all_targets = Vec::new();

        for _ in 0..config.n_samples {
            let mut sequence = Vec::new();

            // Generate correlated time series
            for t in 0..sequence_length {
                let time_point = t as f64 / sequence_length as f64;

                for feature_idx in 0..n_features {
                    let base_value = match feature_idx {
                        0 => (time_point * 2.0 * PI).sin(),
                        1 => (time_point * 4.0 * PI).cos(),
                        2 => time_point * 2.0 - 1.0,      // Linear trend
                        _ => rng.random_range(-1.0..1.0), // Random walk
                    };

                    // Add correlation effects if matrix provided
                    let mut correlated_value = base_value;
                    if let Some(ref corr_matrix) = correlation_matrix {
                        if feature_idx < corr_matrix.len() && t > 0 {
                            let prev_idx = (t - 1) * n_features;
                            for (i, &corr) in corr_matrix[feature_idx].iter().enumerate() {
                                if i < n_features && prev_idx + i < sequence.len() {
                                    correlated_value += corr * sequence[prev_idx + i] * 0.5;
                                }
                            }
                        }
                    }

                    // Add noise
                    let noise = rng.random_range(-config.noise_level..config.noise_level);
                    sequence.push(correlated_value + noise);
                }
            }

            // Target is the mean of the last few values of the first feature
            let target = if sequence_length * n_features >= n_features * 3 {
                let last_values: Vec<f64> = (0..3)
                    .map(|i| sequence[(sequence_length - 1 - i) * n_features])
                    .collect();
                last_values.iter().sum::<f64>() / last_values.len() as f64
            } else {
                sequence[sequence.len() - n_features] // Last value of first feature
            };

            all_sequences.extend(sequence);
            all_targets.push(target);
        }

        let feature_tensor = Tensor::from_vec(
            all_sequences,
            &[config.n_samples, sequence_length * n_features],
        )?;
        let label_tensor = Tensor::from_vec(all_targets, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate time series with anomalies for anomaly detection tasks
    pub fn make_time_series_anomalies(
        config: SyntheticConfig,
        sequence_length: usize,
        anomaly_probability: f64,
        anomaly_magnitude: f64,
    ) -> Result<SyntheticDataset<f64>> {
        let mut rng: Box<dyn RngCore> = if let Some(seed) = config.random_seed {
            Box::new(scirs2_core::random::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(scirs2_core::random::rng())
        };

        let mut all_sequences = Vec::new();
        let mut all_labels = Vec::new(); // 1 for anomaly, 0 for normal

        for _ in 0..config.n_samples {
            let mut sequence = Vec::new();
            let mut has_anomaly = false;

            for t in 0..sequence_length {
                let time_point = t as f64 / sequence_length as f64;

                // Base pattern (sine wave with trend)
                let base_value = (time_point * 2.0 * PI).sin() + time_point * 0.5;

                // Add normal noise
                let noise = rng.random_range(-config.noise_level..config.noise_level);
                let mut value = base_value + noise;

                // Inject anomaly with probability
                if rng.random_range(0.0..1.0) < anomaly_probability {
                    let anomaly_factor = if rng.random_range(0.0..1.0) < 0.5 {
                        1.0
                    } else {
                        -1.0
                    };
                    value += anomaly_factor * anomaly_magnitude;
                    has_anomaly = true;
                }

                sequence.push(value);
            }

            all_sequences.extend(sequence);
            all_labels.push(if has_anomaly { 1.0 } else { 0.0 });
        }

        let feature_tensor = Tensor::from_vec(all_sequences, &[config.n_samples, sequence_length])?;
        let label_tensor = Tensor::from_vec(all_labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }
}

/// Time series patterns for synthetic data generation
#[derive(Debug, Clone)]
pub enum TimeSeriesPattern {
    /// Sinusoidal wave with specified frequency
    Sine { frequency: f64 },
    /// Cosine wave with specified frequency
    Cosine { frequency: f64 },
    /// Linear trend with slope
    Linear { slope: f64 },
    /// Exponential growth/decay
    Exponential { rate: f64 },
    /// Random walk
    RandomWalk { step_size: f64 },
    /// Autoregressive process AR(1)
    AR1 { coefficient: f64 },
    /// Combination of sine and linear trend
    SineWithTrend { frequency: f64, slope: f64 },
    /// Seasonal pattern
    Seasonal { period: usize, amplitude: f64 },
}

impl TimeSeriesPattern {
    pub fn generate_sequence(
        &self,
        length: usize,
        rng: &mut dyn RngCore,
        noise_level: f64,
    ) -> (Vec<f64>, f64) {
        let mut sequence = Vec::with_capacity(length);

        match self {
            TimeSeriesPattern::Sine { frequency } => {
                for t in 0..length {
                    let time_point = t as f64 / length as f64;
                    let value = (time_point * frequency * 2.0 * PI).sin();
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(value + noise);
                }
            }
            TimeSeriesPattern::Cosine { frequency } => {
                for t in 0..length {
                    let time_point = t as f64 / length as f64;
                    let value = (time_point * frequency * 2.0 * PI).cos();
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(value + noise);
                }
            }
            TimeSeriesPattern::Linear { slope } => {
                for t in 0..length {
                    let time_point = t as f64 / length as f64;
                    let value = slope * time_point;
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(value + noise);
                }
            }
            TimeSeriesPattern::Exponential { rate } => {
                for t in 0..length {
                    let time_point = t as f64 / length as f64;
                    let value = (rate * time_point).exp();
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(value + noise);
                }
            }
            TimeSeriesPattern::RandomWalk { step_size } => {
                let mut current_value = 0.0;
                for _ in 0..length {
                    let step = rng.random_range(-step_size..*step_size);
                    current_value += step;
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(current_value + noise);
                }
            }
            TimeSeriesPattern::AR1 { coefficient } => {
                let mut prev_value = rng.random_range(-1.0..1.0);
                sequence.push(prev_value);

                for _ in 1..length {
                    let innovation = rng.random_range(-1.0..1.0);
                    let value = coefficient * prev_value + innovation;
                    let noise = rng.random_range(-noise_level..noise_level);
                    prev_value = value;
                    sequence.push(value + noise);
                }
            }
            TimeSeriesPattern::SineWithTrend { frequency, slope } => {
                for t in 0..length {
                    let time_point = t as f64 / length as f64;
                    let sine_component = (time_point * frequency * 2.0 * PI).sin();
                    let trend_component = slope * time_point;
                    let value = sine_component + trend_component;
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(value + noise);
                }
            }
            TimeSeriesPattern::Seasonal { period, amplitude } => {
                for t in 0..length {
                    let seasonal_phase = (t % period) as f64 / *period as f64;
                    let value = amplitude * (seasonal_phase * 2.0 * PI).sin();
                    let noise = rng.random_range(-noise_level..noise_level);
                    sequence.push(value + noise);
                }
            }
        }

        // Target is the last value for forecasting tasks
        let target = sequence.last().copied().unwrap_or(0.0);

        (sequence, target)
    }
}
