//! Random search over a hyperparameter space.
//!
//! Samples hyperparameter sets from the specified search space using a
//! seeded, reproducible random number generator provided by `scirs2_core`.

use scirs2_core::random::{seeded_rng, Distribution, RandUniform, Rng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use crate::hparam::space::{HParamConfig, HParamSet, HParamSpec, HParamValue};

// ─────────────────────────────────────────────────────────────────────────────
// RandomSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Randomly samples hyperparameter sets from the specified search space.
///
/// Uses a seeded PRNG (`scirs2_core::random::seeded_rng`) for full
/// reproducibility across runs.
pub struct RandomSearch {
    configs: Vec<HParamConfig>,
    seed: u64,
}

impl RandomSearch {
    /// Create a new `RandomSearch` with the given configs and seed.
    pub fn new(configs: Vec<HParamConfig>, seed: u64) -> Self {
        RandomSearch { configs, seed }
    }

    /// Append an additional parameter configuration.
    pub fn add_param(&mut self, config: HParamConfig) {
        self.configs.push(config);
    }

    /// Sample a single `HParamSet` using the current seed.
    ///
    /// Calling this multiple times with the same seed always returns the same
    /// value (the seed is not advanced between external calls — use
    /// [`sample`] for multiple distinct samples).
    ///
    /// [`sample`]: RandomSearch::sample
    pub fn sample_one(&self) -> Result<HParamSet> {
        let mut sets = self.sample(1)?;
        sets.pop().ok_or_else(|| {
            TensorError::invalid_argument_op(
                "RandomSearch::sample_one",
                "no configs defined; cannot sample a hyperparameter set",
            )
        })
    }

    /// Sample `n` distinct `HParamSet`s.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidArgument`] if:
    /// - No configs are defined and `n > 0`.
    /// - A `LogRange` spec has non-positive bounds.
    /// - A `Categorical` spec is empty.
    pub fn sample(&self, n: usize) -> Result<Vec<HParamSet>> {
        if n == 0 {
            return Ok(vec![]);
        }
        if self.configs.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "RandomSearch::sample",
                "no hyperparameter configs defined; add at least one HParamConfig",
            ));
        }

        let mut rng = seeded_rng(self.seed);
        let mut results = Vec::with_capacity(n);

        for trial in 0..n {
            // Mix trial index into RNG state so successive samples differ.
            // We advance the RNG by sampling a disposable value per trial.
            let _skip: u64 = rng.random_range(0..u64::MAX);
            // Re-seed per trial to get deterministic per-trial values
            // while still varying across trials.
            let trial_seed = self
                .seed
                .wrapping_add(trial as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut trial_rng = seeded_rng(trial_seed);

            let mut set = HParamSet::new();
            for cfg in &self.configs {
                let value = sample_value(&cfg.spec, &mut trial_rng)?;
                set.set(cfg.name.as_str(), value);
            }
            results.push(set);
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a single value from the given spec using `rng`.
fn sample_value<R: Rng>(spec: &HParamSpec, rng: &mut R) -> Result<HParamValue> {
    match spec {
        HParamSpec::Categorical(vals) => {
            if vals.is_empty() {
                return Err(TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    "Categorical spec has no choices",
                ));
            }
            let idx = rng.random_range(0..vals.len());
            Ok(vals[idx].clone())
        }

        HParamSpec::IntRange { low, high, step } => {
            if *step <= 0 || *low >= *high {
                return Err(TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    "IntRange has an empty or invalid grid (check low, high, step)",
                ));
            }
            // Number of valid steps.
            let range = (*high - *low) as u64;
            let step_u = *step as u64;
            let num_steps = (range + step_u - 1) / step_u;
            let chosen_step = rng.random_range(0..num_steps);
            Ok(HParamValue::Int(low + (chosen_step as i64) * step))
        }

        HParamSpec::FloatRange { low, high } => {
            if low >= high {
                return Err(TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    "FloatRange: low must be less than high",
                ));
            }
            let dist = RandUniform::new(*low, *high).map_err(|e| {
                TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    &format!("FloatRange distribution error: {e}"),
                )
            })?;
            Ok(HParamValue::Float(dist.sample(rng)))
        }

        HParamSpec::LogRange { low, high } => {
            if *low <= 0.0 || *high <= 0.0 {
                return Err(TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    "LogRange: both low and high must be strictly positive",
                ));
            }
            if low >= high {
                return Err(TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    "LogRange: low must be less than high",
                ));
            }
            // Sample uniformly in log-space: exp(uniform(ln(low), ln(high)))
            let log_low = low.ln();
            let log_high = high.ln();
            let dist = RandUniform::new(log_low, log_high).map_err(|e| {
                TensorError::invalid_argument_op(
                    "RandomSearch::sample_value",
                    &format!("LogRange distribution error: {e}"),
                )
            })?;
            let log_val: f64 = dist.sample(rng);
            Ok(HParamValue::Float(log_val.exp()))
        }

        HParamSpec::Bool => Ok(HParamValue::Bool(rng.random_range(0..2u64) == 1)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hparam::space::HParamSpec;

    #[test]
    fn test_random_search_sample_returns_n_results() -> Result<()> {
        let cfg = HParamConfig::new(
            "lr",
            HParamSpec::FloatRange {
                low: 1e-4,
                high: 1e-1,
            },
        );
        let rs = RandomSearch::new(vec![cfg], 42);
        let samples = rs.sample(10)?;
        assert_eq!(samples.len(), 10);
        Ok(())
    }

    #[test]
    fn test_random_search_sample_zero() -> Result<()> {
        let cfg = HParamConfig::new(
            "lr",
            HParamSpec::FloatRange {
                low: 1e-4,
                high: 1e-1,
            },
        );
        let rs = RandomSearch::new(vec![cfg], 7);
        let samples = rs.sample(0)?;
        assert!(samples.is_empty());
        Ok(())
    }

    #[test]
    fn test_random_search_sample_one() -> Result<()> {
        let cfg = HParamConfig::new("dropout", HParamSpec::Bool);
        let rs = RandomSearch::new(vec![cfg], 99);
        let set = rs.sample_one()?;
        assert!(set.get_bool("dropout").is_some());
        Ok(())
    }

    #[test]
    fn test_random_search_float_range_bounds() -> Result<()> {
        let low = 0.001_f64;
        let high = 0.1_f64;
        let cfg = HParamConfig::new("lr", HParamSpec::FloatRange { low, high });
        let rs = RandomSearch::new(vec![cfg], 1234);
        let samples = rs.sample(50)?;
        for s in &samples {
            let v = s.get_float("lr").expect("lr must be present");
            assert!(
                v >= low && v < high,
                "FloatRange value {v} out of bounds [{low}, {high})"
            );
        }
        Ok(())
    }

    #[test]
    fn test_random_search_log_range_in_correct_range() -> Result<()> {
        let low = 1e-5_f64;
        let high = 1e-1_f64;
        let cfg = HParamConfig::new("lr", HParamSpec::LogRange { low, high });
        let rs = RandomSearch::new(vec![cfg], 5678);
        let samples = rs.sample(50)?;
        for s in &samples {
            let v = s.get_float("lr").expect("lr must be present");
            assert!(
                v >= low && v < high,
                "LogRange value {v} out of bounds [{low}, {high})"
            );
        }
        Ok(())
    }

    #[test]
    fn test_random_search_categorical() -> Result<()> {
        let opts = ["adam", "sgd", "rmsprop"];
        let cfg = HParamConfig::new(
            "opt",
            HParamSpec::Categorical(
                opts.iter()
                    .map(|s| HParamValue::String((*s).to_string()))
                    .collect(),
            ),
        );
        let rs = RandomSearch::new(vec![cfg], 42);
        let samples = rs.sample(30)?;
        for s in &samples {
            let v = s.get_str("opt").expect("opt must be present");
            assert!(opts.contains(&v), "unexpected optimizer: {v}");
        }
        Ok(())
    }

    #[test]
    fn test_random_search_int_range() -> Result<()> {
        let cfg = HParamConfig::new(
            "batch",
            HParamSpec::IntRange {
                low: 16,
                high: 256,
                step: 16,
            },
        );
        let rs = RandomSearch::new(vec![cfg], 11);
        let samples = rs.sample(40)?;
        for s in &samples {
            let v = s.get_int("batch").expect("batch must be present");
            assert!((16..256).contains(&v), "batch {v} out of range");
            assert_eq!((v - 16) % 16, 0, "batch {v} is not aligned to step 16");
        }
        Ok(())
    }

    #[test]
    fn test_random_search_reproducible_with_same_seed() -> Result<()> {
        let cfg = HParamConfig::new(
            "lr",
            HParamSpec::FloatRange {
                low: 1e-4,
                high: 1e-1,
            },
        );
        let rs1 = RandomSearch::new(vec![cfg.clone()], 999);
        let rs2 = RandomSearch::new(vec![cfg], 999);
        let s1 = rs1.sample(5)?;
        let s2 = rs2.sample(5)?;
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(
                a.get_float("lr"),
                b.get_float("lr"),
                "same seed must produce same samples"
            );
        }
        Ok(())
    }

    #[test]
    fn test_random_search_no_configs_error() {
        let rs = RandomSearch::new(vec![], 0);
        assert!(rs.sample(1).is_err());
    }

    #[test]
    fn test_random_search_log_range_invalid_bounds() {
        let cfg = HParamConfig::new(
            "lr",
            HParamSpec::LogRange {
                low: -1.0,
                high: 1.0,
            },
        );
        let rs = RandomSearch::new(vec![cfg], 0);
        assert!(rs.sample(1).is_err());
    }
}
