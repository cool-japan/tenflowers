/// Per-Operation GPU Memory Tracker
///
/// Tracks allocation / deallocation statistics broken down by operation name
/// (e.g. `"matmul"`, `"conv2d"`, `"relu"`).  All operations on the shared hash
/// map are protected by a `RwLock` for maximum read concurrency.

use once_cell::sync::Lazy;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// OpMemoryStats
// ---------------------------------------------------------------------------

/// Memory statistics for a single named operation.
#[derive(Debug, Clone, Default)]
pub struct OpMemoryStats {
    /// Name of the operation.
    pub op_name: String,
    /// Total number of `record_alloc` calls for this operation.
    pub total_alloc_calls: u64,
    /// Total number of `record_dealloc` calls for this operation.
    pub total_dealloc_calls: u64,
    /// Total bytes allocated across all calls.
    pub total_bytes_allocated: u64,
    /// Highest `current_bytes_in_use` value ever observed.
    pub peak_bytes_in_use: u64,
    /// Current bytes in use (allocs minus deallocs). Can become negative due to
    /// accounting edge-cases (e.g. deallocs recorded for allocs not tracked here).
    pub current_bytes_in_use: i64,
}

// ---------------------------------------------------------------------------
// PerOperationTracker
// ---------------------------------------------------------------------------

/// Thread-safe per-operation memory tracker backed by a `RwLock<HashMap>`.
pub struct PerOperationTracker {
    ops: RwLock<HashMap<String, OpMemoryStats>>,
}

impl PerOperationTracker {
    /// Create a new, empty tracker.
    pub fn new() -> Self {
        Self {
            ops: RwLock::new(HashMap::new()),
        }
    }

    /// Record an allocation of `size` bytes for `op_name`.
    pub fn record_alloc(&self, op_name: &str, size: usize) {
        let mut guard = self
            .ops
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = guard.entry(op_name.to_string()).or_insert_with(|| OpMemoryStats {
            op_name: op_name.to_string(),
            ..Default::default()
        });
        entry.total_alloc_calls += 1;
        entry.total_bytes_allocated += size as u64;
        entry.current_bytes_in_use += size as i64;
        if entry.current_bytes_in_use > 0 {
            let current = entry.current_bytes_in_use as u64;
            if current > entry.peak_bytes_in_use {
                entry.peak_bytes_in_use = current;
            }
        }
    }

    /// Record a deallocation of `size` bytes for `op_name`.
    pub fn record_dealloc(&self, op_name: &str, size: usize) {
        let mut guard = self
            .ops
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = guard.entry(op_name.to_string()).or_insert_with(|| OpMemoryStats {
            op_name: op_name.to_string(),
            ..Default::default()
        });
        entry.total_dealloc_calls += 1;
        entry.current_bytes_in_use -= size as i64;
    }

    /// Return the stats for a single operation, or `None` if not seen.
    pub fn get_stats(&self, op_name: &str) -> Option<OpMemoryStats> {
        let guard = self
            .ops
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(op_name).cloned()
    }

    /// Return stats for all tracked operations.
    pub fn get_all_stats(&self) -> Vec<OpMemoryStats> {
        let guard = self
            .ops
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.values().cloned().collect()
    }

    /// Return the top-`n` operations sorted by `total_bytes_allocated` descending.
    pub fn top_consumers(&self, n: usize) -> Vec<OpMemoryStats> {
        let mut all = self.get_all_stats();
        all.sort_by_key(|b| std::cmp::Reverse(b.total_bytes_allocated));
        all.truncate(n);
        all
    }

    /// Reset all per-operation statistics.
    pub fn reset(&self) {
        let mut guard = self
            .ops
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.clear();
    }
}

impl Default for PerOperationTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------

/// Process-wide per-operation memory tracker.
pub static GLOBAL_OP_TRACKER: Lazy<PerOperationTracker> =
    Lazy::new(PerOperationTracker::new);

/// Obtain a reference to the global [`PerOperationTracker`].
pub fn get_per_op_tracker() -> &'static PerOperationTracker {
    &GLOBAL_OP_TRACKER
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> PerOperationTracker {
        PerOperationTracker::new()
    }

    #[test]
    fn test_per_op_tracker_alloc() {
        let tracker = fresh();
        tracker.record_alloc("matmul", 4096);
        let stats = tracker
            .get_stats("matmul")
            .expect("matmul stats should exist");
        assert_eq!(stats.total_alloc_calls, 1);
        assert_eq!(stats.total_bytes_allocated, 4096);
        assert_eq!(stats.current_bytes_in_use, 4096);
        assert_eq!(stats.peak_bytes_in_use, 4096);
    }

    #[test]
    fn test_per_op_tracker_multiple_ops() {
        let tracker = fresh();
        tracker.record_alloc("relu", 512);
        tracker.record_alloc("conv2d", 8192);
        tracker.record_alloc("relu", 512);

        let relu = tracker.get_stats("relu").expect("relu stats");
        let conv2d = tracker.get_stats("conv2d").expect("conv2d stats");

        assert_eq!(relu.total_alloc_calls, 2);
        assert_eq!(relu.total_bytes_allocated, 1024);
        assert_eq!(conv2d.total_alloc_calls, 1);
        assert_eq!(conv2d.total_bytes_allocated, 8192);
    }

    #[test]
    fn test_per_op_tracker_top_consumers() {
        let tracker = fresh();
        tracker.record_alloc("small_op", 100);
        tracker.record_alloc("big_op", 9999);
        tracker.record_alloc("medium_op", 1000);

        let top = tracker.top_consumers(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].op_name, "big_op");
        assert_eq!(top[1].op_name, "medium_op");
    }

    #[test]
    fn test_per_op_tracker_reset() {
        let tracker = fresh();
        tracker.record_alloc("op_x", 200);
        tracker.record_alloc("op_y", 400);
        tracker.reset();
        assert!(tracker.get_all_stats().is_empty(), "all stats must be cleared");
        assert!(tracker.get_stats("op_x").is_none());
    }

    #[test]
    fn test_per_op_tracker_dealloc() {
        let tracker = fresh();
        tracker.record_alloc("op", 1000);
        tracker.record_dealloc("op", 600);
        let stats = tracker.get_stats("op").expect("stats");
        assert_eq!(stats.current_bytes_in_use, 400);
        assert_eq!(stats.peak_bytes_in_use, 1000);
    }

    #[test]
    fn test_per_op_tracker_top_consumers_fewer_than_n() {
        let tracker = fresh();
        tracker.record_alloc("only_one", 42);
        let top = tracker.top_consumers(10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_per_op_tracker_get_all_stats() {
        let tracker = fresh();
        tracker.record_alloc("a", 1);
        tracker.record_alloc("b", 2);
        tracker.record_alloc("c", 3);
        let all = tracker.get_all_stats();
        assert_eq!(all.len(), 3);
    }
}
