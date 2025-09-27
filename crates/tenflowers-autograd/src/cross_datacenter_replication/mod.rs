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
pub mod topology;
pub mod config;
pub mod connection;
pub mod compression;
pub mod operations;
pub mod core;

// Re-export all types and functionality for backward compatibility

// Topology structures
pub use topology::{
    DatacenterTopology, DatacenterInfo, DatacenterCapacity, NetworkLink,
    ReductionTree, AggregationStrategy
};

// Configuration structures
pub use config::{
    ReplicationConfig, CompressionConfig, CompressionAlgorithm,
    FaultToleranceConfig, RetryPolicy, BandwidthOptimizationConfig,
    AdaptiveCompressionConfig
};

// Connection and bandwidth management
pub use connection::{
    DatacenterConnection, ConnectionStatus, BandwidthMonitor,
    BandwidthStats, BandwidthMeasurement, BandwidthOptimizer
};

// Compression functionality
pub use compression::{
    CompressionEngine, CompressionCodec, NetworkConditions
};

// Operation structures
pub use operations::{
    ReplicationOperation, OperationType, ReplicationPayload,
    ParameterMetadata, CompressionInfo, Priority, DatacenterStatus
};

// Core replicator and consistency models
pub use core::{
    CrossDatacenterReplicator, ConsistencyModel,
    PrepareResult, ReplicationHealth
};