//! NUMA-aware scheduling for multi-threaded data loading
//!
//! This module provides NUMA topology detection and thread affinity management
//! to optimize data loading performance on multi-socket systems by ensuring
//! worker threads are properly bound to specific NUMA nodes and CPU cores.

#![allow(unsafe_code)]

use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: usize,
    /// CPU cores available on this node
    pub cpu_cores: Vec<usize>,
    /// Memory capacity in bytes (if available)
    pub memory_capacity: Option<usize>,
    /// Current load estimate (0.0 to 1.0)
    pub load_estimate: f32,
}

/// NUMA topology information for the system
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Available NUMA nodes
    pub nodes: Vec<NumaNode>,
    /// Total number of CPU cores across all nodes
    pub total_cores: usize,
    /// Whether NUMA is actually supported/beneficial on this system
    pub numa_available: bool,
}

/// Configuration for NUMA-aware scheduling
#[derive(Debug, Clone)]
pub struct NumaConfig {
    /// Enable NUMA-aware scheduling
    pub enabled: bool,
    /// Strategy for assigning workers to NUMA nodes
    pub assignment_strategy: NumaAssignmentStrategy,
    /// Whether to strictly bind threads to assigned cores
    pub strict_affinity: bool,
    /// Preferred NUMA nodes (empty means use all available)
    pub preferred_nodes: Vec<usize>,
    /// Whether to balance workers across NUMA nodes
    pub balance_nodes: bool,
}

/// Strategy for assigning workers to NUMA nodes
#[derive(Debug, Clone, PartialEq)]
pub enum NumaAssignmentStrategy {
    /// Round-robin assignment across NUMA nodes
    RoundRobin,
    /// Fill one NUMA node before moving to the next
    FillFirst,
    /// Interleave workers across nodes for memory bandwidth
    Interleave,
    /// Use load-based assignment (prefer less loaded nodes)
    LoadBalanced,
    /// Custom assignment based on user-defined mapping
    Custom(HashMap<usize, usize>), // worker_id -> numa_node
}

/// NUMA-aware worker assignment information
#[derive(Debug, Clone)]
pub struct NumaWorkerAssignment {
    /// Worker ID
    pub worker_id: usize,
    /// Assigned NUMA node
    pub numa_node: usize,
    /// Assigned CPU cores within the node
    pub cpu_cores: Vec<usize>,
    /// Whether affinity was successfully set
    pub affinity_set: bool,
}

/// NUMA-aware scheduler for data loading workers
pub struct NumaScheduler {
    /// System NUMA topology
    topology: NumaTopology,
    /// Configuration
    config: NumaConfig,
    /// Current worker assignments
    assignments: Vec<NumaWorkerAssignment>,
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            assignment_strategy: NumaAssignmentStrategy::RoundRobin,
            strict_affinity: false, // Default to soft affinity for compatibility
            preferred_nodes: Vec::new(),
            balance_nodes: true,
        }
    }
}

impl NumaTopology {
    /// Detect system NUMA topology
    pub fn detect() -> Self {
        // Try to detect actual NUMA topology
        if let Ok(topology) = Self::detect_linux_numa() {
            return topology;
        }

        // Fallback: create pseudo-NUMA based on CPU topology
        Self::create_pseudo_numa()
    }

    /// Detect NUMA topology on Linux systems
    #[cfg(target_os = "linux")]
    fn detect_linux_numa() -> Result<Self> {
        use std::fs;
        use std::path::Path;

        let numa_path = Path::new("/sys/devices/system/node");
        if !numa_path.exists() {
            return Err(TensorError::invalid_argument(
                "NUMA sysfs not available".to_string(),
            ));
        }

        let mut nodes = Vec::new();
        let entries = fs::read_dir(numa_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read NUMA info: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                TensorError::invalid_argument(format!("Failed to read NUMA entry: {}", e))
            })?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("node") && name_str.len() > 4 {
                if let Ok(node_id) = name_str[4..].parse::<usize>() {
                    let cpu_cores = Self::read_node_cpus(entry.path())?;
                    let memory_capacity = Self::read_node_memory(entry.path()).ok();

                    nodes.push(NumaNode {
                        id: node_id,
                        cpu_cores,
                        memory_capacity,
                        load_estimate: 0.0,
                    });
                }
            }
        }

        if nodes.is_empty() {
            return Err(TensorError::invalid_argument(
                "No NUMA nodes detected".to_string(),
            ));
        }

        let total_cores = nodes.iter().map(|n| n.cpu_cores.len()).sum();

        Ok(Self {
            nodes,
            total_cores,
            numa_available: true,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn detect_linux_numa() -> Result<Self> {
        Err(TensorError::invalid_argument(
            "Linux NUMA detection not available on this platform".to_string(),
        ))
    }

    #[cfg(target_os = "linux")]
    fn read_node_cpus(node_path: std::path::PathBuf) -> Result<Vec<usize>> {
        use std::fs;

        let cpulist_path = node_path.join("cpulist");
        let content = fs::read_to_string(cpulist_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read CPU list: {}", e))
        })?;

        let mut cpus = Vec::new();
        for part in content.trim().split(',') {
            if part.contains('-') {
                let range: Vec<&str> = part.split('-').collect();
                if range.len() == 2 {
                    let start = range[0].parse::<usize>().map_err(|e| {
                        TensorError::invalid_argument(format!("Invalid CPU range: {}", e))
                    })?;
                    let end = range[1].parse::<usize>().map_err(|e| {
                        TensorError::invalid_argument(format!("Invalid CPU range: {}", e))
                    })?;
                    for cpu in start..=end {
                        cpus.push(cpu);
                    }
                }
            } else {
                let cpu = part
                    .parse::<usize>()
                    .map_err(|e| TensorError::invalid_argument(format!("Invalid CPU ID: {}", e)))?;
                cpus.push(cpu);
            }
        }

        Ok(cpus)
    }

    #[cfg(target_os = "linux")]
    fn read_node_memory(node_path: std::path::PathBuf) -> Result<usize> {
        use std::fs;

        let meminfo_path = node_path.join("meminfo");
        let content = fs::read_to_string(meminfo_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read memory info: {}", e))
        })?;

        // Parse MemTotal from the first line
        for line in content.lines() {
            if line.starts_with("Node") && line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let kb = parts[2].parse::<usize>().map_err(|e| {
                        TensorError::invalid_argument(format!("Failed to parse memory size: {}", e))
                    })?;
                    return Ok(kb * 1024); // Convert KB to bytes
                }
            }
        }

        Err(TensorError::invalid_argument(
            "MemTotal not found in meminfo".to_string(),
        ))
    }

    /// Create pseudo-NUMA topology based on available CPU cores
    fn create_pseudo_numa() -> Self {
        let total_cores = num_cpus::get();

        // Create 2 pseudo-NUMA nodes for systems with 4+ cores
        let num_nodes = if total_cores >= 4 { 2 } else { 1 };
        let cores_per_node = total_cores / num_nodes;

        let mut nodes = Vec::new();
        for node_id in 0..num_nodes {
            let start_core = node_id * cores_per_node;
            let end_core = if node_id == num_nodes - 1 {
                total_cores // Last node gets any remaining cores
            } else {
                start_core + cores_per_node
            };

            let cpu_cores: Vec<usize> = (start_core..end_core).collect();

            nodes.push(NumaNode {
                id: node_id,
                cpu_cores,
                memory_capacity: None,
                load_estimate: 0.0,
            });
        }

        Self {
            nodes,
            total_cores,
            numa_available: false, // This is pseudo-NUMA
        }
    }
}

impl NumaScheduler {
    /// Create a new NUMA scheduler
    pub fn new(config: NumaConfig) -> Result<Self> {
        let topology = NumaTopology::detect();

        Ok(Self {
            topology,
            config,
            assignments: Vec::new(),
        })
    }

    /// Create a scheduler with default configuration
    pub fn with_default_config() -> Result<Self> {
        Self::new(NumaConfig::default())
    }

    /// Assign workers to NUMA nodes based on the configuration
    pub fn assign_workers(&mut self, num_workers: usize) -> Result<Vec<NumaWorkerAssignment>> {
        if !self.config.enabled {
            // If NUMA is disabled, return empty assignments
            return Ok(Vec::new());
        }

        let available_nodes = if self.config.preferred_nodes.is_empty() {
            self.topology.nodes.clone()
        } else {
            self.topology
                .nodes
                .iter()
                .filter(|node| self.config.preferred_nodes.contains(&node.id))
                .cloned()
                .collect()
        };

        if available_nodes.is_empty() {
            return Err(TensorError::invalid_argument(
                "No available NUMA nodes for assignment".to_string(),
            ));
        }

        let mut assignments = Vec::new();

        match &self.config.assignment_strategy {
            NumaAssignmentStrategy::RoundRobin => {
                for worker_id in 0..num_workers {
                    let node_idx = worker_id % available_nodes.len();
                    let numa_node = &available_nodes[node_idx];
                    let cpu_cores = Self::select_cores_from_node(numa_node, 1);

                    assignments.push(NumaWorkerAssignment {
                        worker_id,
                        numa_node: numa_node.id,
                        cpu_cores,
                        affinity_set: false,
                    });
                }
            }
            NumaAssignmentStrategy::FillFirst => {
                let mut current_node_idx = 0;
                let mut workers_on_current_node = 0;
                let workers_per_node = num_workers / available_nodes.len() + 1;

                for worker_id in 0..num_workers {
                    if workers_on_current_node >= workers_per_node
                        && current_node_idx < available_nodes.len() - 1
                    {
                        current_node_idx += 1;
                        workers_on_current_node = 0;
                    }

                    let numa_node = &available_nodes[current_node_idx];
                    let cpu_cores = Self::select_cores_from_node(numa_node, 1);

                    assignments.push(NumaWorkerAssignment {
                        worker_id,
                        numa_node: numa_node.id,
                        cpu_cores,
                        affinity_set: false,
                    });

                    workers_on_current_node += 1;
                }
            }
            NumaAssignmentStrategy::Interleave => {
                // Similar to round-robin but with memory locality consideration
                for worker_id in 0..num_workers {
                    let node_idx = worker_id % available_nodes.len();
                    let numa_node = &available_nodes[node_idx];
                    let cpu_cores = Self::select_cores_from_node(numa_node, 1);

                    assignments.push(NumaWorkerAssignment {
                        worker_id,
                        numa_node: numa_node.id,
                        cpu_cores,
                        affinity_set: false,
                    });
                }
            }
            NumaAssignmentStrategy::LoadBalanced => {
                // Sort nodes by load estimate (ascending)
                let mut sorted_nodes = available_nodes.clone();
                sorted_nodes.sort_by(|a, b| a.load_estimate.partial_cmp(&b.load_estimate).unwrap());

                for worker_id in 0..num_workers {
                    let node_idx = worker_id % sorted_nodes.len();
                    let numa_node = &sorted_nodes[node_idx];
                    let cpu_cores = Self::select_cores_from_node(numa_node, 1);

                    assignments.push(NumaWorkerAssignment {
                        worker_id,
                        numa_node: numa_node.id,
                        cpu_cores,
                        affinity_set: false,
                    });
                }
            }
            NumaAssignmentStrategy::Custom(mapping) => {
                for worker_id in 0..num_workers {
                    let numa_node_id = mapping.get(&worker_id).copied().unwrap_or(0);
                    let numa_node = available_nodes
                        .iter()
                        .find(|node| node.id == numa_node_id)
                        .unwrap_or(&available_nodes[0]);

                    let cpu_cores = Self::select_cores_from_node(numa_node, 1);

                    assignments.push(NumaWorkerAssignment {
                        worker_id,
                        numa_node: numa_node.id,
                        cpu_cores,
                        affinity_set: false,
                    });
                }
            }
        }

        self.assignments = assignments.clone();
        Ok(assignments)
    }

    /// Select CPU cores from a NUMA node for a worker
    fn select_cores_from_node(node: &NumaNode, num_cores: usize) -> Vec<usize> {
        let available_cores = &node.cpu_cores;
        let cores_to_take = num_cores.min(available_cores.len());
        available_cores[0..cores_to_take].to_vec()
    }

    /// Set CPU affinity for the current thread
    pub fn set_thread_affinity(assignment: &NumaWorkerAssignment) -> Result<()> {
        if assignment.cpu_cores.is_empty() {
            return Ok(());
        }

        // Platform-specific affinity setting
        #[cfg(target_os = "linux")]
        {
            Self::set_linux_affinity(&assignment.cpu_cores)
        }

        #[cfg(not(target_os = "linux"))]
        {
            // For non-Linux platforms, this is a no-op
            // Could implement platform-specific affinity setting here
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    fn set_linux_affinity(cpu_cores: &[usize]) -> Result<()> {
        use std::mem;

        // Use libc for CPU affinity on Linux
        let mut cpu_set: libc::cpu_set_t = unsafe { mem::zeroed() };

        for &cpu in cpu_cores {
            unsafe {
                libc::CPU_SET(cpu, &mut cpu_set);
            }
        }

        let result = unsafe {
            libc::sched_setaffinity(
                0, // Current thread
                mem::size_of::<libc::cpu_set_t>(),
                &cpu_set,
            )
        };

        if result != 0 {
            return Err(TensorError::invalid_argument(format!(
                "Failed to set CPU affinity: {}",
                std::io::Error::last_os_error()
            )));
        }

        Ok(())
    }

    /// Get NUMA topology information
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Get current worker assignments
    pub fn assignments(&self) -> &[NumaWorkerAssignment] {
        &self.assignments
    }

    /// Update load estimates for NUMA nodes (for load balancing)
    pub fn update_load_estimates(&mut self, load_info: HashMap<usize, f32>) {
        for node in &mut self.topology.nodes {
            if let Some(&load) = load_info.get(&node.id) {
                node.load_estimate = load.clamp(0.0, 1.0);
            }
        }
    }

    /// Get statistics about NUMA assignment
    pub fn get_assignment_stats(&self) -> NumaAssignmentStats {
        let mut workers_per_node = HashMap::new();
        let mut total_workers = 0;

        for assignment in &self.assignments {
            *workers_per_node.entry(assignment.numa_node).or_insert(0) += 1;
            total_workers += 1;
        }

        let affinity_success_count = self.assignments.iter().filter(|a| a.affinity_set).count();

        let numa_nodes_used = workers_per_node.len();

        NumaAssignmentStats {
            total_workers,
            workers_per_node,
            affinity_success_rate: if total_workers > 0 {
                affinity_success_count as f32 / total_workers as f32
            } else {
                0.0
            },
            numa_nodes_used,
            total_numa_nodes: self.topology.nodes.len(),
        }
    }
}

/// Statistics about NUMA worker assignment
#[derive(Debug, Clone)]
pub struct NumaAssignmentStats {
    /// Total number of workers
    pub total_workers: usize,
    /// Distribution of workers across NUMA nodes
    pub workers_per_node: HashMap<usize, usize>,
    /// Success rate of CPU affinity setting (0.0 to 1.0)
    pub affinity_success_rate: f32,
    /// Number of NUMA nodes actually used
    pub numa_nodes_used: usize,
    /// Total number of NUMA nodes available
    pub total_numa_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_config_default() {
        let config = NumaConfig::default();
        assert!(config.enabled);
        assert_eq!(
            config.assignment_strategy,
            NumaAssignmentStrategy::RoundRobin
        );
        assert!(!config.strict_affinity);
        assert!(config.balance_nodes);
    }

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        assert!(topology.total_cores > 0);
        assert!(!topology.nodes.is_empty());
    }

    #[test]
    fn test_worker_assignment_round_robin() {
        let config = NumaConfig {
            enabled: true,
            assignment_strategy: NumaAssignmentStrategy::RoundRobin,
            strict_affinity: false,
            preferred_nodes: Vec::new(),
            balance_nodes: true,
        };

        let mut scheduler = NumaScheduler::new(config).unwrap();
        let assignments = scheduler.assign_workers(4).unwrap();

        assert_eq!(assignments.len(), 4);

        // Check that worker IDs are correct
        for (i, assignment) in assignments.iter().enumerate() {
            assert_eq!(assignment.worker_id, i);
        }
    }

    #[test]
    fn test_assignment_stats() {
        let config = NumaConfig::default();
        let mut scheduler = NumaScheduler::new(config).unwrap();
        let _assignments = scheduler.assign_workers(4).unwrap();

        let stats = scheduler.get_assignment_stats();
        assert_eq!(stats.total_workers, 4);
        assert!(stats.numa_nodes_used > 0);
        assert!(stats.total_numa_nodes > 0);
    }
}
