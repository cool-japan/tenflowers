//! Cross-Datacenter Replication - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by feature area:
//!
//! ## Module Organization
//!
//! - **topology**: Datacenter topology, network links, and reduction tree structures
//! - **config**: Configuration management for replication, compression, and fault tolerance
//! - **connection**: Connection management, bandwidth monitoring, and traffic optimization
//! - **compression**: Compression engine with various algorithms and adaptive strategies
//! - **operations**: Replication operations, payloads, and metadata structures
//! - **core**: Main CrossDatacenterReplicator implementation and consistency models
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

#[cfg(feature = "distributed")]
pub mod distributed_replication {
    // Import the modularized cross-datacenter replication functionality
    mod cross_datacenter_replication;

    // Re-export all types and functionality for backward compatibility
    pub use cross_datacenter_replication::*;
}