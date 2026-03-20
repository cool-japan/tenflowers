//! Federated Learning algorithms — production-grade implementations.
//!
//! This module provides implementations of the most widely-used federated
//! learning (FL) algorithms with differential privacy support and
//! communication efficiency utilities.
//!
//! # Algorithms
//!
//! | Algorithm | Reference |
//! |-----------|-----------|
//! | [`FedAvg`] | McMahan et al., 2017 |
//! | [`FedProx`] | Li et al., 2020 |
//! | [`Scaffold`] | Karimireddy et al., 2020 |
//! | [`FedNova`] | Wang et al., 2020 |
//!
//! # Differential Privacy
//!
//! Gradient clipping ([`clip_gradients`]) and Gaussian noise addition
//! ([`add_dp_noise`]) implement the DP-SGD mechanism (Abadi et al., 2016).
//! Privacy accounting is provided via [`compute_privacy_loss`] using the
//! moments-accountant approximation.
//!
//! # Communication Efficiency
//!
//! [`GradientCompressor`] implements top-k sparsification, keeping only the
//! largest-magnitude gradient components to reduce communication cost.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::federated::{FedAvg, FederatedConfig, ClientUpdate};
//!
//! let config = FederatedConfig::default();
//! let initial_params = vec![vec![0.0_f32; 10], vec![0.0_f32; 5]];
//! let mut server = FedAvg::new(config, initial_params);
//!
//! // Clients perform local training and return updates
//! let updates = vec![
//!     ClientUpdate { client_id: 0, params: server.distribute().clone(),
//!                    num_samples: 100, loss: 0.5, num_local_steps: 5 },
//! ];
//! let global = server.aggregate(&updates)?;
//! ```

pub mod byzantine;
pub mod clustered_fl;
pub mod compression;
pub mod fedavg;
pub mod fedma;
pub mod fednova;
pub mod fedprox;
pub mod personalized;
pub mod privacy;
pub mod scaffold;
pub mod types;

mod tests;

// Re-export all public types
pub use byzantine::{
    BulyanAggregator, ByzantineMetrics, FlameAggregator, KrumAggregator, MedianAggregator,
    TrimmedMeanAggregator,
};
pub use clustered_fl::{
    ClusterEnsemble, ClusteredFLMetrics, HypCluster, IfcaAlgorithm, IfcaAssignment,
};
pub use compression::GradientCompressor;
pub use fedavg::FedAvg;
pub use fedma::{FedMaAggregator, LayerMatching, PermutationMatrix};
pub use fednova::FedNova;
pub use fedprox::FedProx;
pub use personalized::{ApflClient, FedBnClient, HeurFl, PFedMeClient, PersonalizedMetrics};
pub use privacy::{add_dp_noise, clip_gradients, compute_privacy_loss};
pub use scaffold::{Scaffold, ScaffoldState};
pub use types::{
    ClientUpdate, DpBudget, FederatedConfig, FederatedError, GlobalUpdate, ModelParams,
};
