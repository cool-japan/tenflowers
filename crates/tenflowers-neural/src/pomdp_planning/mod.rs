//! POMDP (Partially Observable Markov Decision Process) Planning
//!
//! Comprehensive planning algorithms for POMDPs covering:
//! - Exact methods: PBVI, Perseus, FIB, QMDP
//! - Online methods: POMCP (particle-based UCT), AO* belief tree search
//! - Policy representations: Alpha vectors, finite-state controllers (FSC)
//! - Approximate methods: SARSOP, Belief-MDP
//! - Generators: Tiger problem, Grid maze, Rock sample
//!
//! # Algorithm Summary
//! | Algorithm | Type | Notes |
//! |-----------|------|-------|
//! | QMDP | Offline approx | Fast, ignores partial obs |
//! | FIB | Offline bound | Tighter than QMDP |
//! | PBVI | Offline exact approx | Point-based VI |
//! | Perseus | Offline randomized | Faster than PBVI |
//! | SARSOP | Offline anytime | Best offline |
//! | POMCP | Online particle | Silver&Veness 2010 |
//! | AO* | Online tree | Heuristic search |

pub mod metrics;
pub mod offline_solvers;
pub mod online_solvers;
pub mod policy;
pub mod types;

mod tests;

// Re-export all public types from submodules
pub use metrics::{PomdpGenerator, PomdpMetrics};
pub use offline_solvers::{FibAlgorithm, PbviSolver, PerseusAlgorithm, QmdpApproximation};
pub use online_solvers::{PomcpActionNode, PomcpNode, PomcpSolver, SarsopAlgorithm};
pub use policy::{BeliefMdpSolver, BtNode, FscNode, OnlineBeliefTreeSearch, PomdpPolicyGraph};
pub use types::{
    AlphaVector, BeliefState, PomdpAction, PomdpError, PomdpModel, PomdpObs, PomdpState, Result,
};
