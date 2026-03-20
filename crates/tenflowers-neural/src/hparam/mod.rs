//! Hyperparameter search utilities.
//!
//! This module provides tools for defining, searching, and managing
//! hyperparameter spaces for machine-learning experiments.
//!
//! ## Modules
//!
//! - [`space`] — Core types: [`HParamSpec`], [`HParamValue`], [`HParamConfig`], [`HParamSet`].
//! - [`grid_search`] — Exhaustive grid search via Cartesian product.
//! - [`random_search`] — Random sampling over continuous and discrete spaces.
//! - [`trial`] — Trial tracking, study management, and result reporting.
//!
//! ## Example
//!
//! ```rust,ignore
//! use tenflowers_neural::hparam::{
//!     space::{HParamConfig, HParamSpec, HParamValue},
//!     grid_search::GridSearch,
//!     random_search::RandomSearch,
//!     trial::{HParamStudy, OptimizationDirection, TrialResult},
//! };
//!
//! // Define a 2-dimensional grid
//! let configs = vec![
//!     HParamConfig::new("optimizer", HParamSpec::Categorical(vec![
//!         HParamValue::String("adam".into()),
//!         HParamValue::String("sgd".into()),
//!     ])),
//!     HParamConfig::new("lr", HParamSpec::Categorical(vec![
//!         HParamValue::Float(0.01),
//!         HParamValue::Float(0.001),
//!     ])),
//! ];
//!
//! let gs = GridSearch::new(configs.clone());
//! let combos = gs.generate_all().expect("generate_all failed"); // 2 x 2 = 4 combinations
//!
//! // Random search over a continuous log-range
//! let rs = RandomSearch::new(vec![
//!     HParamConfig::new("lr", HParamSpec::LogRange { low: 1e-5, high: 1e-1 }),
//! ], 42);
//! let samples = rs.sample(20).expect("sample failed");
//!
//! // Track results in a study
//! let mut study = HParamStudy::new("my-experiment", OptimizationDirection::Minimize);
//! for (i, params) in combos.into_iter().enumerate() {
//!     let fake_loss = 0.5 / (i as f64 + 1.0);
//!     study.add_trial(TrialResult::new(i, params, fake_loss, 1.0));
//! }
//! println!("{}", study.summary());
//! ```

pub mod grid_search;
pub mod random_search;
pub mod space;
pub mod trial;

// Convenient re-exports of the most commonly used types.
pub use grid_search::GridSearch;
pub use random_search::RandomSearch;
pub use space::{HParamConfig, HParamSet, HParamSpec, HParamValue};
pub use trial::{HParamStudy, OptimizationDirection, TrialResult};
