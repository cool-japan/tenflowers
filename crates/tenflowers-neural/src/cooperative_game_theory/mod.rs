//! Cooperative and Non-Cooperative Game Theory Module
//!
//! Provides comprehensive game-theoretic algorithms including Nash equilibrium finding,
//! counterfactual regret minimization, cooperative game solution concepts, and
//! mechanism design — all implemented in pure Rust.

pub mod cfr;
pub mod cooperative;
pub mod dynamics;
pub mod equilibrium;
pub mod types;

mod tests;

// Re-export all public items so downstream code can use `cooperative_game_theory::*`
pub use types::{CgtError, CgtGame, NashEquilibriumSolver};

pub use cfr::{CfrInfoSet, CfrNode, CfrSolver, RegretAlgorithm, RegretMinimizer};

pub use cooperative::{BanzhafIndex, CgtCooperativeGame, CoreSolver, ShapleyValueCalculator};

pub use equilibrium::{CorrelatedEquilibrium, MeanFieldEquilibrium, MechanismDesign};

pub use dynamics::{CgtMetrics, CgtNeuralNash, EvolutionaryGameDynamics, NashNet, SymmetricGame};
