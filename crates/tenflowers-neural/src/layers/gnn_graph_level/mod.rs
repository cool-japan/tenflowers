//! Graph-level GNN operations for graph classification and regression.
//!
//! This module provides:
//! - [`Graph`]: adjacency matrix + node features representation with spectral helpers
//! - Global pooling: [`global_mean_pool`], [`global_max_pool`], [`global_sum_pool`]
//! - Hierarchical pooling: [`DiffPool`] (Ying et al. 2018), [`MinCutPool`] (Bianchi et al. 2020)
//! - [`Gatv2Layer`]: GATv2 dynamic attention (Brody et al. 2021)
//! - [`GinLayer`]: Graph Isomorphism Network layer (Xu et al. 2018)
//!
//! All types operate on plain `Vec<f32>` buffers (no `Tensor` dependency),
//! using deterministic weight initialisation via `scirs2_core::random`.

pub mod layers;
pub mod pooling;
pub mod types;

mod tests;

// Re-export all public types
pub use layers::{Gatv2Layer, GinLayer};
pub use pooling::{
    global_add_pool, global_max_pool, global_mean_pool, global_sum_pool, DiffPool, MinCutPool,
};
pub use types::{Graph, GnnError};
