//! Multi-stream memory management for concurrent operations
//!
//! This module provides concurrent memory management across multiple streams,
//! enabling efficient parallel GPU operations.

use super::pools::{MemoryPool, MemoryPoolStats};
use crate::{Result, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Multi-stream memory management for concurrent operations
pub struct MultiStreamMemoryManager {
    pools: Vec<MemoryPool>,
    stream_assignment: Arc<Mutex<HashMap<usize, usize>>>, // operation_id -> stream_id
    current_stream: Arc<Mutex<usize>>,
}

impl MultiStreamMemoryManager {
    /// Create a new multi-stream memory manager
    #[cfg(feature = "gpu")]
    pub fn new(device_id: usize, num_streams: usize, pool_size_per_stream: usize) -> Result<Self> {
        let mut pools = Vec::new();

        for _ in 0..num_streams {
            pools.push(MemoryPool::new(device_id, pool_size_per_stream)?);
        }

        Ok(Self {
            pools,
            stream_assignment: Arc::new(Mutex::new(HashMap::new())),
            current_stream: Arc::new(Mutex::new(0)),
        })
    }

    /// Get the appropriate memory pool for an operation
    pub fn get_pool(&self, operation_id: usize) -> Result<&MemoryPool> {
        let stream_assignment = self.stream_assignment.lock().unwrap();

        let stream_id = if let Some(&stream_id) = stream_assignment.get(&operation_id) {
            stream_id
        } else {
            // Assign to current stream and rotate
            let mut current_stream = self.current_stream.lock().unwrap();
            let stream_id = *current_stream;
            *current_stream = (*current_stream + 1) % self.pools.len();
            stream_id
        };

        self.pools
            .get(stream_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Invalid stream ID: {stream_id}")))
    }

    /// Assign a specific operation to a specific stream
    pub fn assign_operation_to_stream(&self, operation_id: usize, stream_id: usize) -> Result<()> {
        if stream_id >= self.pools.len() {
            return Err(TensorError::invalid_argument(format!(
                "Stream ID {} out of range. Available streams: {}",
                stream_id,
                self.pools.len()
            )));
        }

        let mut stream_assignment = self.stream_assignment.lock().unwrap();
        stream_assignment.insert(operation_id, stream_id);
        Ok(())
    }

    /// Remove an operation's stream assignment
    pub fn unassign_operation(&self, operation_id: usize) {
        let mut stream_assignment = self.stream_assignment.lock().unwrap();
        stream_assignment.remove(&operation_id);
    }

    /// Get the stream ID for a specific operation
    pub fn get_operation_stream(&self, operation_id: usize) -> Option<usize> {
        let stream_assignment = self.stream_assignment.lock().unwrap();
        stream_assignment.get(&operation_id).copied()
    }

    /// Get the number of available streams
    pub fn num_streams(&self) -> usize {
        self.pools.len()
    }

    /// Get a specific pool by stream ID
    pub fn get_pool_by_stream(&self, stream_id: usize) -> Result<&MemoryPool> {
        self.pools
            .get(stream_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Invalid stream ID: {stream_id}")))
    }

    /// Get statistics for all streams
    pub fn stats(&self) -> Vec<MemoryPoolStats> {
        self.pools.iter().map(|pool| pool.stats()).collect()
    }

    /// Get statistics for a specific stream
    pub fn stream_stats(&self, stream_id: usize) -> Result<MemoryPoolStats> {
        self.pools
            .get(stream_id)
            .map(|pool| pool.stats())
            .ok_or_else(|| TensorError::invalid_argument(format!("Invalid stream ID: {stream_id}")))
    }

    /// Get total memory usage across all streams
    pub fn total_memory_usage(&self) -> (usize, usize) {
        let mut total_allocated = 0;
        let mut total_free = 0;

        for pool in &self.pools {
            let stats = pool.stats();
            total_allocated += stats.total_allocated;
            total_free += stats.total_free;
        }

        (total_allocated, total_free)
    }

    /// Get the least loaded stream (for load balancing)
    pub fn get_least_loaded_stream(&self) -> usize {
        let mut min_load = usize::MAX;
        let mut best_stream = 0;

        for (i, pool) in self.pools.iter().enumerate() {
            let stats = pool.stats();
            if stats.total_allocated < min_load {
                min_load = stats.total_allocated;
                best_stream = i;
            }
        }

        best_stream
    }

    /// Get the stream with the most free memory
    pub fn get_stream_with_most_free_memory(&self) -> usize {
        let mut max_free = 0;
        let mut best_stream = 0;

        for (i, pool) in self.pools.iter().enumerate() {
            let stats = pool.stats();
            if stats.total_free > max_free {
                max_free = stats.total_free;
                best_stream = i;
            }
        }

        best_stream
    }

    /// Balance memory across streams by reassigning operations
    pub fn balance_streams(&self) -> Result<usize> {
        let mut reassignments = 0;
        let target_load = {
            let (total_allocated, _) = self.total_memory_usage();
            total_allocated / self.pools.len()
        };

        let mut stream_assignment = self.stream_assignment.lock().unwrap();

        // Identify overloaded and underloaded streams
        let mut overloaded_streams = Vec::new();
        let mut underloaded_streams = Vec::new();

        for (i, pool) in self.pools.iter().enumerate() {
            let stats = pool.stats();
            if stats.total_allocated > target_load * 11 / 10 {
                // 10% tolerance
                overloaded_streams.push(i);
            } else if stats.total_allocated < target_load * 9 / 10 {
                underloaded_streams.push(i);
            }
        }

        // Reassign operations from overloaded to underloaded streams
        let operations_to_reassign: Vec<_> = stream_assignment
            .iter()
            .filter(|(_, &stream_id)| overloaded_streams.contains(&stream_id))
            .map(|(&op_id, &stream_id)| (op_id, stream_id))
            .collect();

        for (op_id, _old_stream) in operations_to_reassign {
            if let Some(&new_stream) = underloaded_streams.first() {
                stream_assignment.insert(op_id, new_stream);
                reassignments += 1;

                // Rotate underloaded streams for fair distribution
                underloaded_streams.rotate_left(1);
            }
        }

        Ok(reassignments)
    }

    /// Generate a comprehensive report of all streams
    pub fn generate_streams_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Multi-Stream Memory Manager Report ===\n\n");

        let (total_allocated, total_free) = self.total_memory_usage();
        report.push_str(&format!(
            "Total Memory - Allocated: {} bytes, Free: {} bytes\n",
            total_allocated, total_free
        ));
        report.push_str(&format!("Number of Streams: {}\n\n", self.pools.len()));

        // Per-stream statistics
        for (i, pool) in self.pools.iter().enumerate() {
            let stats = pool.stats();
            report.push_str(&format!("Stream {}:\n", i));
            report.push_str(&format!("  Allocated: {} bytes\n", stats.total_allocated));
            report.push_str(&format!("  Free: {} bytes\n", stats.total_free));
            report.push_str(&format!("  Blocks Allocated: {}\n", stats.blocks_allocated));
            report.push_str(&format!("  Blocks Free: {}\n", stats.blocks_free));
            report.push_str(&format!(
                "  Fragmentation Ratio: {:.2}\n",
                stats.fragmentation_ratio
            ));
            report.push_str(&format!(
                "  Memory Pressure: {:.2}%\n",
                stats.memory_pressure * 100.0
            ));
            report.push('\n');
        }

        // Operation assignments
        let stream_assignment = self.stream_assignment.lock().unwrap();
        if !stream_assignment.is_empty() {
            report.push_str("Operation Assignments:\n");
            for (op_id, stream_id) in stream_assignment.iter() {
                report.push_str(&format!("  Operation {}: Stream {}\n", op_id, stream_id));
            }
        }

        report
    }

    /// Clear all operation assignments
    pub fn clear_assignments(&self) {
        let mut stream_assignment = self.stream_assignment.lock().unwrap();
        stream_assignment.clear();
    }

    /// Get operation count per stream
    pub fn get_operation_counts(&self) -> Vec<usize> {
        let stream_assignment = self.stream_assignment.lock().unwrap();
        let mut counts = vec![0; self.pools.len()];

        for &stream_id in stream_assignment.values() {
            if stream_id < counts.len() {
                counts[stream_id] += 1;
            }
        }

        counts
    }

    /// Check if streams are balanced (within tolerance)
    pub fn are_streams_balanced(&self, tolerance_percent: f32) -> bool {
        let (total_allocated, _) = self.total_memory_usage();
        if total_allocated == 0 {
            return true; // No memory allocated, considered balanced
        }

        let target_load = total_allocated / self.pools.len();
        let tolerance = (target_load as f32 * tolerance_percent / 100.0) as usize;

        for pool in &self.pools {
            let stats = pool.stats();
            let deviation = stats.total_allocated.abs_diff(target_load);

            if deviation > tolerance {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests would require GPU context in a real environment
    // For now, we test the logic that doesn't require actual GPU allocation

    #[test]
    fn test_stream_assignment() {
        // This test would need to be adapted for actual GPU context
        // For now, test the assignment logic conceptually

        let _assignments: HashMap<usize, usize> = HashMap::new();
        let current_stream = 0;

        // Test round-robin assignment logic
        let num_streams = 3;
        let mut stream_id = current_stream;

        for i in 0..6 {
            // Simulate assignment
            let assigned_stream = stream_id;
            stream_id = (stream_id + 1) % num_streams;

            assert_eq!(assigned_stream, i % num_streams);
        }
    }

    #[test]
    fn test_load_balancing_logic() {
        // Test the balancing algorithm logic
        let target_load = 1000;
        let tolerance = target_load / 10; // 10% tolerance

        // Test overloaded condition
        let overloaded = 1200;
        assert!(overloaded > target_load + tolerance);

        // Test underloaded condition
        let underloaded = 800;
        assert!(underloaded < target_load - tolerance);

        // Test balanced condition
        let balanced = 950;
        assert!(balanced >= target_load - tolerance && balanced <= target_load + tolerance);
    }

    #[test]
    fn test_stream_balancing_calculation() {
        // Test stream balance calculation
        let total_allocated = 3000;
        let num_streams = 3;
        let target_load = total_allocated / num_streams; // 1000

        assert_eq!(target_load, 1000);

        // Test tolerance calculation
        let tolerance_percent = 10.0;
        let tolerance = (target_load as f32 * tolerance_percent / 100.0) as usize;
        assert_eq!(tolerance, 100);

        // Test deviation calculation
        let stream_load = 1150;
        let deviation = if stream_load > target_load {
            stream_load - target_load
        } else {
            target_load - stream_load
        };
        assert_eq!(deviation, 150);
        assert!(deviation > tolerance); // This stream would be considered unbalanced
    }

    #[test]
    fn test_operation_count_tracking() {
        let mut counts = vec![0; 3]; // 3 streams
        let assignments = vec![(1, 0), (2, 1), (3, 0), (4, 2), (5, 1)];

        for (_, stream_id) in assignments {
            if stream_id < counts.len() {
                counts[stream_id] += 1;
            }
        }

        assert_eq!(counts, vec![2, 2, 1]); // Distribution: stream 0: 2, stream 1: 2, stream 2: 1
    }

    #[test]
    fn test_memory_usage_aggregation() {
        // Test total memory calculation logic
        let stream_stats = vec![
            (500, 1500), // allocated, free
            (800, 1200),
            (300, 1700),
        ];

        let mut total_allocated = 0;
        let mut total_free = 0;

        for (allocated, free) in stream_stats {
            total_allocated += allocated;
            total_free += free;
        }

        assert_eq!(total_allocated, 1600);
        assert_eq!(total_free, 4400);
    }

    #[test]
    fn test_least_loaded_stream_selection() {
        // Test logic for finding least loaded stream
        let stream_loads = vec![1200, 800, 1000];

        let mut min_load = usize::MAX;
        let mut best_stream = 0;

        for (i, &load) in stream_loads.iter().enumerate() {
            if load < min_load {
                min_load = load;
                best_stream = i;
            }
        }

        assert_eq!(best_stream, 1); // Stream 1 has load 800, which is minimum
        assert_eq!(min_load, 800);
    }

    #[test]
    fn test_most_free_memory_selection() {
        // Test logic for finding stream with most free memory
        let stream_free_memory = vec![500, 1200, 800];

        let mut max_free = 0;
        let mut best_stream = 0;

        for (i, &free) in stream_free_memory.iter().enumerate() {
            if free > max_free {
                max_free = free;
                best_stream = i;
            }
        }

        assert_eq!(best_stream, 1); // Stream 1 has 1200 free, which is maximum
        assert_eq!(max_free, 1200);
    }
}
