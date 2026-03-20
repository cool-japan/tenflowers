//! Computation Graph Module
//!
//! This module provides a comprehensive computation graph implementation for TenfloweRS.
//! The graph functionality has been modularized for better maintainability and organization.
//!
//! ## Module Organization
//!
//! - **core**: Fundamental graph types and basic operations (add_node, add_edge, get_node, etc.)
//! - **analysis**: Graph analysis operations (topological sorting, validation, input/output identification)
//! - **subgraph**: Subgraph extraction and manipulation (subgraph by criteria, connected components, etc.)
//! - **manipulation**: Graph transformation operations (extending, merging, replacing)
//! - **node_edge_ops**: Node and edge management (removal, replacement, redirection)
//! - **control_deps**: Control dependency management
//! - **serialization**: Graph persistence and serialization (save/load, JSON, GraphDef)
//! - **optimization**: Graph optimization operations (existing module)

// Import the modularized graph functionality
mod analysis;
mod control_deps;
mod core;
mod manipulation;
mod node_edge_ops;
mod serialization;
mod subgraph;

// Keep the existing optimization module
pub mod optimization;

// Re-export all public types and functions for backward compatibility
pub use core::*;
pub use serialization::*;
