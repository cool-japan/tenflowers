//! Bayesian Deep Learning — SGLD, SGHMC, Laplace, SWAG, Calibration.
//!
//! Implements principled uncertainty quantification for neural networks using:
//! - **SGLD** (Welling & Teh 2011): Stochastic Gradient Langevin Dynamics
//! - **SGHMC** (Chen et al. 2014): Stochastic Gradient Hamiltonian Monte Carlo
//! - **Laplace Approximation**: MAP + diagonal Hessian posterior
//! - **SWAG** (Maddox et al. 2019): Stochastic Weight Averaging Gaussian
//! - **Calibration**: ECE, temperature scaling, deep ensembles
//!
//! All types use the `Bdl` prefix to avoid conflicts with `bayesian.rs`.

mod calibration;
pub(crate) mod helpers;
mod laplace;
mod mcmc;
mod shared;
mod swag;
mod tests;

pub use calibration::{
    compute_calibration, nll_classification, CalibrationResult, DeepEnsemble, DeepEnsembleConfig,
    TemperatureScaling,
};
pub use laplace::LaplaceApproximation;
pub use mcmc::{SghcmConfig, SghcmSampler, SgldConfig, SgldSampler};
pub use shared::{BdlLinear, BdlMlp};
pub use swag::{SwagConfig, SwagModel};
