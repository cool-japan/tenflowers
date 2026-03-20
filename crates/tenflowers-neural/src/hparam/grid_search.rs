//! Grid search over a discrete hyperparameter space.
//!
//! Computes the full Cartesian product of discrete grids, producing every
//! combination of hyperparameter values exactly once.

use tenflowers_core::{Result, TensorError};

use crate::hparam::space::{HParamConfig, HParamSet, HParamSpec};

// ─────────────────────────────────────────────────────────────────────────────
// GridSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Exhaustive grid search over a discrete hyperparameter space.
///
/// Each `HParamConfig` must have a discrete-valued spec (`Categorical`,
/// `IntRange`, or `Bool`).  Continuous specs (`FloatRange`, `LogRange`) are
/// rejected when [`generate_all`] is called.
///
/// [`generate_all`]: GridSearch::generate_all
pub struct GridSearch {
    configs: Vec<HParamConfig>,
}

impl GridSearch {
    /// Create a new `GridSearch` with the given parameter configurations.
    pub fn new(configs: Vec<HParamConfig>) -> Self {
        GridSearch { configs }
    }

    /// Append an additional parameter configuration.
    pub fn add_param(&mut self, config: HParamConfig) {
        self.configs.push(config);
    }

    /// Return the total number of combinations (product of grid sizes).
    ///
    /// Returns `0` when there are no configs or when any config has an empty grid.
    pub fn num_combinations(&self) -> usize {
        if self.configs.is_empty() {
            return 0;
        }
        let mut total = 1usize;
        for cfg in &self.configs {
            match cfg.spec.discrete_values() {
                Some(vals) => {
                    if vals.is_empty() {
                        return 0;
                    }
                    total = total.saturating_mul(vals.len());
                }
                None => {
                    // Continuous specs are treated as size 0 for this estimate.
                    return 0;
                }
            }
        }
        total
    }

    /// Generate all hyperparameter combinations via Cartesian product.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidArgument`] if:
    /// - Any spec is continuous (`FloatRange` or `LogRange`).
    /// - Any grid is empty (e.g. `IntRange` where `low >= high`).
    pub fn generate_all(&self) -> Result<Vec<HParamSet>> {
        if self.configs.is_empty() {
            return Ok(vec![]);
        }

        // Build discrete grids for each parameter, validating them.
        let mut grids: Vec<(&str, Vec<_>)> = Vec::with_capacity(self.configs.len());
        for cfg in &self.configs {
            let vals = match &cfg.spec {
                HParamSpec::FloatRange { .. } | HParamSpec::LogRange { .. } => {
                    return Err(TensorError::invalid_argument_op(
                        "GridSearch::generate_all",
                        &format!(
                            "parameter '{}' has a continuous spec and cannot be used with grid search; \
                             use IntRange, Categorical, or Bool instead",
                            cfg.name
                        ),
                    ));
                }
                _ => cfg.spec.discrete_values().unwrap_or_default(),
            };

            if vals.is_empty() {
                return Err(TensorError::invalid_argument_op(
                    "GridSearch::generate_all",
                    &format!(
                        "parameter '{}' has an empty discrete grid (check IntRange bounds/step)",
                        cfg.name
                    ),
                ));
            }

            grids.push((cfg.name.as_str(), vals));
        }

        // Iterative Cartesian product (avoids recursive stack growth for large grids).
        let mut results: Vec<HParamSet> = vec![HParamSet::new()];

        for (name, vals) in &grids {
            let mut next_results: Vec<HParamSet> = Vec::with_capacity(results.len() * vals.len());
            for existing in &results {
                for val in vals {
                    let mut extended = existing.clone();
                    extended.set(name, val.clone());
                    next_results.push(extended);
                }
            }
            results = next_results;
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hparam::space::{HParamSpec, HParamValue};

    #[test]
    fn test_grid_search_empty_configs_returns_empty() -> Result<()> {
        let gs = GridSearch::new(vec![]);
        let all = gs.generate_all()?;
        assert!(all.is_empty());
        Ok(())
    }

    #[test]
    fn test_grid_search_single_categorical() -> Result<()> {
        let cfg = HParamConfig::new(
            "optimizer",
            HParamSpec::Categorical(vec![
                HParamValue::String("adam".to_owned()),
                HParamValue::String("sgd".to_owned()),
                HParamValue::String("rmsprop".to_owned()),
            ]),
        );
        let gs = GridSearch::new(vec![cfg]);
        let all = gs.generate_all()?;
        assert_eq!(all.len(), 3);
        assert_eq!(gs.num_combinations(), 3);
        Ok(())
    }

    #[test]
    fn test_grid_search_categorical_x_int_range() -> Result<()> {
        let cfg_opt = HParamConfig::new(
            "optimizer",
            HParamSpec::Categorical(vec![
                HParamValue::String("adam".to_owned()),
                HParamValue::String("sgd".to_owned()),
            ]),
        );
        let cfg_layers = HParamConfig::new(
            "num_layers",
            HParamSpec::IntRange {
                low: 1,
                high: 4,
                step: 1,
            },
        );
        let gs = GridSearch::new(vec![cfg_opt, cfg_layers]);
        let all = gs.generate_all()?;
        // 2 optimizers × 3 layer counts (1,2,3) = 6
        assert_eq!(all.len(), 6);
        assert_eq!(gs.num_combinations(), 6);
        Ok(())
    }

    #[test]
    fn test_grid_search_bool_param() -> Result<()> {
        let cfg = HParamConfig::new("use_dropout", HParamSpec::Bool);
        let gs = GridSearch::new(vec![cfg]);
        let all = gs.generate_all()?;
        assert_eq!(all.len(), 2);
        assert_eq!(gs.num_combinations(), 2);
        Ok(())
    }

    #[test]
    fn test_grid_search_continuous_spec_is_error() {
        let cfg = HParamConfig::new(
            "lr",
            HParamSpec::FloatRange {
                low: 1e-4,
                high: 1e-1,
            },
        );
        let gs = GridSearch::new(vec![cfg]);
        assert!(gs.generate_all().is_err());
    }

    #[test]
    fn test_grid_search_log_range_is_error() {
        let cfg = HParamConfig::new(
            "lr",
            HParamSpec::LogRange {
                low: 1e-5,
                high: 1e-1,
            },
        );
        let gs = GridSearch::new(vec![cfg]);
        assert!(gs.generate_all().is_err());
    }

    #[test]
    fn test_grid_search_empty_int_range_is_error() {
        // low >= high → empty grid
        let cfg = HParamConfig::new(
            "layers",
            HParamSpec::IntRange {
                low: 5,
                high: 3,
                step: 1,
            },
        );
        let gs = GridSearch::new(vec![cfg]);
        assert!(gs.generate_all().is_err());
    }

    #[test]
    fn test_grid_search_num_combinations_matches_generated() -> Result<()> {
        let cfg1 = HParamConfig::new(
            "a",
            HParamSpec::Categorical(vec![
                HParamValue::Int(1),
                HParamValue::Int(2),
                HParamValue::Int(3),
            ]),
        );
        let cfg2 = HParamConfig::new(
            "b",
            HParamSpec::IntRange {
                low: 0,
                high: 6,
                step: 2,
            },
        );
        let cfg3 = HParamConfig::new("flag", HParamSpec::Bool);
        let gs = GridSearch::new(vec![cfg1, cfg2, cfg3]);
        let predicted = gs.num_combinations();
        let actual = gs.generate_all()?.len();
        assert_eq!(predicted, actual);
        // 3 × 3 × 2 = 18
        assert_eq!(actual, 18);
        Ok(())
    }

    #[test]
    fn test_grid_search_each_combo_has_all_params() -> Result<()> {
        let cfg1 = HParamConfig::new(
            "opt",
            HParamSpec::Categorical(vec![HParamValue::String("adam".to_owned())]),
        );
        let cfg2 = HParamConfig::new(
            "lr",
            HParamSpec::Categorical(vec![HParamValue::Float(0.01), HParamValue::Float(0.001)]),
        );
        let gs = GridSearch::new(vec![cfg1, cfg2]);
        let all = gs.generate_all()?;
        for set in &all {
            assert!(set.get("opt").is_some(), "missing 'opt'");
            assert!(set.get("lr").is_some(), "missing 'lr'");
        }
        Ok(())
    }

    #[test]
    fn test_grid_search_add_param_updates_count() -> Result<()> {
        let mut gs = GridSearch::new(vec![]);
        gs.add_param(HParamConfig::new(
            "x",
            HParamSpec::Categorical(vec![HParamValue::Int(1), HParamValue::Int(2)]),
        ));
        gs.add_param(HParamConfig::new(
            "y",
            HParamSpec::Categorical(vec![
                HParamValue::Int(10),
                HParamValue::Int(20),
                HParamValue::Int(30),
            ]),
        ));
        assert_eq!(gs.num_combinations(), 6);
        assert_eq!(gs.generate_all()?.len(), 6);
        Ok(())
    }
}
