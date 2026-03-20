//! Advanced Graph Neural Network (GNN) layers
//!
//! This module implements advanced GNN architectures operating directly on flat
//! `Vec<f32>` node-feature arrays with explicit neighbor-list adjacency, making
//! them easy to use without constructing dense adjacency matrices.
//!
//! ## Included layers
//!
//! * [`GraphSageLayer`] — GraphSAGE (Hamilton et al., 2017) with Mean, Max, and Sum
//!   neighbourhood aggregation, optional bias, and per-node L2 normalisation.
//!
//! * [`GatLayer`] — Graph Attention Network (Veličković et al., 2018) with multi-head
//!   attention, concat-or-mean head merging, and optional dropout during training.
//!
//! ## Message-passing framework
//!
//! * [`MessagePassing`] — a trait that exposes `message`, `aggregate`, and `update`
//!   hooks so that custom GNNs can be expressed in the standard MPNN form.
//!
//! * [`message_passing`] — a free function that runs a full forward pass of any
//!   [`MessagePassing`] implementation on a graph described by neighbour lists.

pub mod gat;
pub mod message_passing;
pub mod sage;
pub mod types;

mod tests;

// Re-export public API
pub use gat::GatLayer;
pub use message_passing::{message_passing, MessagePassing};
pub use sage::GraphSageLayer;
pub use types::AggregationMethod;
