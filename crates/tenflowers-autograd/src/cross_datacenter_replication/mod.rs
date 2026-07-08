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

// Import the modularized cross-datacenter replication functionality
pub mod compression;
pub mod config;
pub mod connection;
pub mod core;
pub mod operations;
pub mod topology;

// Re-export all types and functionality for backward compatibility

// Topology structures
pub use topology::{
    AggregationStrategy, DatacenterCapacity, DatacenterInfo, DatacenterTopology, NetworkLink,
    ReductionTree,
};

// Configuration structures
pub use config::{
    AdaptiveCompressionConfig, BandwidthOptimizationConfig, CompressionAlgorithm,
    CompressionConfig, FaultToleranceConfig, ReplicationConfig, RetryPolicy,
};

// Connection and bandwidth management
pub use connection::{
    BandwidthMeasurement, BandwidthMonitor, BandwidthOptimizer, BandwidthStats, ConnectionStatus,
    DatacenterConnection,
};

// Compression functionality
pub use compression::{CompressionCodec, CompressionEngine, NetworkConditions};

// Operation structures
pub use operations::{
    CompressionInfo, DatacenterStatus, OperationType, ParameterMetadata, Priority,
    ReplicationOperation, ReplicationPayload,
};

// Core replicator and consistency models
pub use core::{ConsistencyModel, CrossDatacenterReplicator, PrepareResult, ReplicationHealth};
