//! Datacenter Topology and Network Management
//!
//! This module defines the datacenter topology structures and network link
//! characteristics for cross-datacenter replication. It provides the foundation
//! for understanding datacenter relationships, capacities, and network properties.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Datacenter topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatacenterTopology {
    /// Map of datacenter IDs to their information
    pub datacenters: HashMap<String, DatacenterInfo>,
    /// Network links between datacenters with bandwidth and latency
    pub network_links: HashMap<(String, String), NetworkLink>,
    /// Hierarchical structure for efficient reduction
    pub reduction_tree: ReductionTree,
}

/// Information about a specific datacenter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatacenterInfo {
    pub id: String,
    pub region: String,
    pub zone: String,
    pub capacity: DatacenterCapacity,
    pub endpoints: Vec<String>,
    pub priority: u8, // Higher priority datacenters act as coordinators
}

/// Datacenter compute and network capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatacenterCapacity {
    pub compute_nodes: usize,
    pub total_gpus: usize,
    pub network_bandwidth_gbps: f64,
    pub storage_capacity_tb: f64,
}

/// Network link characteristics between datacenters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLink {
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub reliability: f64, // 0.0 to 1.0
    pub cost_per_gb: f64,
}

/// Hierarchical reduction tree for efficient parameter aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionTree {
    /// Root datacenter that coordinates global aggregation
    pub root: String,
    /// Tree structure: parent -> children mapping
    pub tree_structure: HashMap<String, Vec<String>>,
    /// Aggregation strategy per level
    pub aggregation_strategy: AggregationStrategy,
}

/// Strategy for aggregating parameters across the tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple averaging
    Average,
    /// Weighted averaging based on datacenter capacity
    WeightedAverage,
    /// Hierarchical reduction with compression
    HierarchicalCompressed,
    /// Bandwidth-aware aggregation
    BandwidthOptimized,
}
