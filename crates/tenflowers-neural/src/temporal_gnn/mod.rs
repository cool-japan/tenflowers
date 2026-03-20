//! Temporal & Dynamic Graph Neural Networks.
//!
//! Implements state-of-the-art models for learning on temporal/dynamic graphs:
//!
//! - [`TemporalGraphNetwork`]: TGN (Rossi 2020) — GRU memory + time encoding + graph attention.
//! - [`DynamicGraphTransformer`]: DyGFormer-style stack with co-occurrence encoding.
//! - [`StGcnModel`]: Spatio-Temporal GCN for skeleton action recognition (Yan 2018).
//! - [`OdeGnn`]: ODE-based continuous-time node embeddings (Euler / RK4).
//! - [`HeteroTgnModel`]: TGN extended to heterogeneous multi-relational temporal graphs.
//!
//! All fallible operations return `Result<_, TensorError>`.
//! No `unsafe` code; no `unwrap()` calls.

pub mod dygformer;
pub mod hetero;
pub mod ode_gnn;
pub mod stgcn;
pub mod tgn;
pub mod types;

mod tests;

// ── Re-exports ────────────────────────────────────────────────────────────────

// Shared types
pub use types::{
    HeteroEdgeType, HeteroNodeType, MessageFunction, OdeSolver, PartitionStrategy, TemporalEdge,
    TgnConfig,
};

// TGN components
pub use tgn::{MemoryUpdateModule, NodeMemory, TemporalGraphNetwork, TimeEncoder};

// DyGFormer components
pub use dygformer::{
    CoOccurrenceEncoder, DyGFormerLayer, DynamicGraphTransformer, NeighborSampler,
};

// ST-GCN components
pub use stgcn::{StGcnBlock, StGcnLayer, StGcnModel, TemporalConv};

// ODE-GNN components
pub use ode_gnn::{EventGraph, GraphOdeFunc, InterpNodeFeatures, OdeGnn};

// Heterogeneous graph components
pub use hetero::{HeteroTemporalGraph, HeteroTgnModel, RelationalTemporalConv};
