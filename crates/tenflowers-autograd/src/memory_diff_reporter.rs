//! # Memory Diff Reporter
//!
//! This module provides comprehensive memory usage tracking and reporting
//! for gradient computation optimizations, showing before/after metrics.
//!
//! ## Features
//!
//! - **Memory Usage Tracking**: Track memory allocation and deallocation
//! - **Before/After Comparisons**: Compare memory usage before and after optimizations
//! - **Detailed Breakdown**: Per-operation and per-layer memory statistics
//! - **Optimization Impact**: Measure the impact of memory optimizations
//! - **Visualization**: Generate reports and visualizations of memory usage
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::memory_diff_reporter::{MemoryDiffReporter, MemorySnapshot};
//!
//! let mut reporter = MemoryDiffReporter::new();
//!
//! // Take snapshot before optimization
//! reporter.snapshot("before_optimization");
//!
//! // ... perform gradient computation and optimizations ...
//!
//! // Take snapshot after optimization
//! reporter.snapshot("after_optimization");
//!
//! // Generate diff report
//! let diff = reporter.diff("before_optimization", "after_optimization").unwrap();
//! println!("{}", diff.format_report());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Memory usage snapshot at a specific point in time
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Total allocated bytes
    pub total_allocated: usize,
    /// Peak allocated bytes
    pub peak_allocated: usize,
    /// Number of active allocations
    pub num_allocations: usize,
    /// Per-operation memory usage (operation_id -> bytes)
    pub operation_memory: HashMap<String, usize>,
    /// Per-layer memory usage (layer_id -> bytes)
    pub layer_memory: HashMap<String, usize>,
    /// Gradient memory usage
    pub gradient_memory: usize,
    /// Activation memory usage
    pub activation_memory: usize,
    /// Parameter memory usage
    pub parameter_memory: usize,
    /// Temporary buffer memory usage
    pub temporary_memory: usize,
    /// Custom tags for categorization
    pub tags: HashMap<String, String>,
}

impl MemorySnapshot {
    /// Create a new empty snapshot
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            total_allocated: 0,
            peak_allocated: 0,
            num_allocations: 0,
            operation_memory: HashMap::new(),
            layer_memory: HashMap::new(),
            gradient_memory: 0,
            activation_memory: 0,
            parameter_memory: 0,
            temporary_memory: 0,
            tags: HashMap::new(),
        }
    }

    /// Create a snapshot from current memory state
    pub fn capture() -> Self {
        let mut snapshot = Self::new();
        snapshot.update_from_system();
        snapshot
    }

    /// Update snapshot with current system memory usage
    fn update_from_system(&mut self) {
        // In a real implementation, this would query actual memory usage
        // For now, we'll use placeholder values
        self.timestamp = Instant::now();
    }

    /// Add memory usage for an operation
    pub fn add_operation_memory(&mut self, operation_id: &str, bytes: usize) {
        *self
            .operation_memory
            .entry(operation_id.to_string())
            .or_insert(0) += bytes;
        self.total_allocated += bytes;
    }

    /// Add memory usage for a layer
    pub fn add_layer_memory(&mut self, layer_id: &str, bytes: usize) {
        *self.layer_memory.entry(layer_id.to_string()).or_insert(0) += bytes;
        self.total_allocated += bytes;
    }

    /// Add a custom tag
    pub fn add_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    /// Get total memory usage in MB
    pub fn total_mb(&self) -> f64 {
        self.total_allocated as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn peak_mb(&self) -> f64 {
        self.peak_allocated as f64 / (1024.0 * 1024.0)
    }
}

impl Default for MemorySnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Difference between two memory snapshots
#[derive(Debug, Clone)]
pub struct MemoryDiff {
    /// Name of the first snapshot
    pub before_name: String,
    /// Name of the second snapshot
    pub after_name: String,
    /// Snapshot before optimization
    pub before: MemorySnapshot,
    /// Snapshot after optimization
    pub after: MemorySnapshot,
    /// Time elapsed between snapshots
    pub time_elapsed: Duration,
}

impl MemoryDiff {
    /// Create a new memory diff
    pub fn new(
        before_name: String,
        after_name: String,
        before: MemorySnapshot,
        after: MemorySnapshot,
    ) -> Self {
        let time_elapsed = after.timestamp.duration_since(before.timestamp);
        Self {
            before_name,
            after_name,
            before,
            after,
            time_elapsed,
        }
    }

    /// Get the change in total allocated memory (bytes)
    pub fn total_allocated_diff(&self) -> i64 {
        self.after.total_allocated as i64 - self.before.total_allocated as i64
    }

    /// Get the change in peak allocated memory (bytes)
    pub fn peak_allocated_diff(&self) -> i64 {
        self.after.peak_allocated as i64 - self.before.peak_allocated as i64
    }

    /// Get the change in number of allocations
    pub fn num_allocations_diff(&self) -> i64 {
        self.after.num_allocations as i64 - self.before.num_allocations as i64
    }

    /// Get the change in gradient memory (bytes)
    pub fn gradient_memory_diff(&self) -> i64 {
        self.after.gradient_memory as i64 - self.before.gradient_memory as i64
    }

    /// Get the change in activation memory (bytes)
    pub fn activation_memory_diff(&self) -> i64 {
        self.after.activation_memory as i64 - self.before.activation_memory as i64
    }

    /// Get the change in parameter memory (bytes)
    pub fn parameter_memory_diff(&self) -> i64 {
        self.after.parameter_memory as i64 - self.before.parameter_memory as i64
    }

    /// Get the change in temporary memory (bytes)
    pub fn temporary_memory_diff(&self) -> i64 {
        self.after.temporary_memory as i64 - self.before.temporary_memory as i64
    }

    /// Get percentage change in total allocated memory
    pub fn total_allocated_pct_change(&self) -> f64 {
        if self.before.total_allocated == 0 {
            return if self.after.total_allocated > 0 {
                100.0
            } else {
                0.0
            };
        }
        (self.total_allocated_diff() as f64 / self.before.total_allocated as f64) * 100.0
    }

    /// Get memory savings (negative diff means savings)
    pub fn memory_savings(&self) -> usize {
        if self.total_allocated_diff() < 0 {
            (-self.total_allocated_diff()) as usize
        } else {
            0
        }
    }

    /// Get memory savings in MB
    pub fn memory_savings_mb(&self) -> f64 {
        self.memory_savings() as f64 / (1024.0 * 1024.0)
    }

    /// Check if optimization reduced memory usage
    pub fn is_improvement(&self) -> bool {
        self.total_allocated_diff() < 0
    }

    /// Format a human-readable diff report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Memory Diff Report ===\n\n");
        report.push_str(&format!("Before: {}\n", self.before_name));
        report.push_str(&format!("After: {}\n", self.after_name));
        report.push_str(&format!(
            "Time Elapsed: {:.2}s\n\n",
            self.time_elapsed.as_secs_f64()
        ));

        report.push_str("Memory Usage:\n");
        report.push_str(&format!(
            "  Total:      {:>10.2} MB -> {:>10.2} MB ({:>+.2} MB, {:>+.1}%)\n",
            self.before.total_mb(),
            self.after.total_mb(),
            self.total_allocated_diff() as f64 / (1024.0 * 1024.0),
            self.total_allocated_pct_change()
        ));

        report.push_str(&format!(
            "  Peak:       {:>10.2} MB -> {:>10.2} MB ({:>+.2} MB)\n",
            self.before.peak_mb(),
            self.after.peak_mb(),
            self.peak_allocated_diff() as f64 / (1024.0 * 1024.0)
        ));

        report.push_str(&format!(
            "  Allocations: {:>8} -> {:>8} ({:>+})\n\n",
            self.before.num_allocations,
            self.after.num_allocations,
            self.num_allocations_diff()
        ));

        report.push_str("Memory Breakdown:\n");
        report.push_str(&format!(
            "  Gradients:   {:>10.2} MB -> {:>10.2} MB ({:>+.2} MB)\n",
            self.before.gradient_memory as f64 / (1024.0 * 1024.0),
            self.after.gradient_memory as f64 / (1024.0 * 1024.0),
            self.gradient_memory_diff() as f64 / (1024.0 * 1024.0)
        ));

        report.push_str(&format!(
            "  Activations: {:>10.2} MB -> {:>10.2} MB ({:>+.2} MB)\n",
            self.before.activation_memory as f64 / (1024.0 * 1024.0),
            self.after.activation_memory as f64 / (1024.0 * 1024.0),
            self.activation_memory_diff() as f64 / (1024.0 * 1024.0)
        ));

        report.push_str(&format!(
            "  Parameters:  {:>10.2} MB -> {:>10.2} MB ({:>+.2} MB)\n",
            self.before.parameter_memory as f64 / (1024.0 * 1024.0),
            self.after.parameter_memory as f64 / (1024.0 * 1024.0),
            self.parameter_memory_diff() as f64 / (1024.0 * 1024.0)
        ));

        report.push_str(&format!(
            "  Temporary:   {:>10.2} MB -> {:>10.2} MB ({:>+.2} MB)\n\n",
            self.before.temporary_memory as f64 / (1024.0 * 1024.0),
            self.after.temporary_memory as f64 / (1024.0 * 1024.0),
            self.temporary_memory_diff() as f64 / (1024.0 * 1024.0)
        ));

        if self.is_improvement() {
            report.push_str(&format!(
                "✓ Optimization SUCCESS: Saved {:.2} MB ({:.1}% reduction)\n",
                self.memory_savings_mb(),
                -self.total_allocated_pct_change()
            ));
        } else {
            report.push_str(&format!(
                "✗ Optimization INCREASED memory by {:.2} MB ({:.1}%)\n",
                -self.memory_savings_mb(),
                self.total_allocated_pct_change()
            ));
        }

        report
    }

    /// Get a summary of top memory changes by operation
    pub fn top_operation_changes(&self, n: usize) -> Vec<(String, i64)> {
        let mut changes = Vec::new();

        // Collect all operation IDs
        let mut all_ops: std::collections::HashSet<String> =
            self.before.operation_memory.keys().cloned().collect();
        all_ops.extend(self.after.operation_memory.keys().cloned());

        for op_id in all_ops {
            let before = *self.before.operation_memory.get(&op_id).unwrap_or(&0);
            let after = *self.after.operation_memory.get(&op_id).unwrap_or(&0);
            let diff = after as i64 - before as i64;
            changes.push((op_id, diff));
        }

        // Sort by absolute value of change
        changes.sort_by(|a, b| b.1.abs().cmp(&a.1.abs()));

        changes.into_iter().take(n).collect()
    }

    /// Get a summary of top memory changes by layer
    pub fn top_layer_changes(&self, n: usize) -> Vec<(String, i64)> {
        let mut changes = Vec::new();

        // Collect all layer IDs
        let mut all_layers: std::collections::HashSet<String> =
            self.before.layer_memory.keys().cloned().collect();
        all_layers.extend(self.after.layer_memory.keys().cloned());

        for layer_id in all_layers {
            let before = *self.before.layer_memory.get(&layer_id).unwrap_or(&0);
            let after = *self.after.layer_memory.get(&layer_id).unwrap_or(&0);
            let diff = after as i64 - before as i64;
            changes.push((layer_id, diff));
        }

        // Sort by absolute value of change
        changes.sort_by(|a, b| b.1.abs().cmp(&a.1.abs()));

        changes.into_iter().take(n).collect()
    }
}

/// Memory diff reporter for tracking optimizations
pub struct MemoryDiffReporter {
    /// Named snapshots
    snapshots: HashMap<String, MemorySnapshot>,
    /// History of diffs
    diffs: Vec<MemoryDiff>,
    /// Whether to enable automatic snapshots
    auto_snapshot: bool,
}

impl MemoryDiffReporter {
    /// Create a new memory diff reporter
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            diffs: Vec::new(),
            auto_snapshot: false,
        }
    }

    /// Enable automatic snapshot capturing
    pub fn enable_auto_snapshot(&mut self) {
        self.auto_snapshot = true;
    }

    /// Disable automatic snapshot capturing
    pub fn disable_auto_snapshot(&mut self) {
        self.auto_snapshot = false;
    }

    /// Take a named snapshot of current memory state
    pub fn snapshot(&mut self, name: &str) -> &MemorySnapshot {
        let snapshot = MemorySnapshot::capture();
        self.snapshots.insert(name.to_string(), snapshot);
        self.snapshots.get(name).unwrap()
    }

    /// Take a snapshot with custom data
    pub fn snapshot_with(&mut self, name: &str, snapshot: MemorySnapshot) {
        self.snapshots.insert(name.to_string(), snapshot);
    }

    /// Get a named snapshot
    pub fn get_snapshot(&self, name: &str) -> Option<&MemorySnapshot> {
        self.snapshots.get(name)
    }

    /// Generate a diff between two snapshots
    pub fn diff(&mut self, before_name: &str, after_name: &str) -> Option<MemoryDiff> {
        let before = self.snapshots.get(before_name)?.clone();
        let after = self.snapshots.get(after_name)?.clone();

        let diff = MemoryDiff::new(
            before_name.to_string(),
            after_name.to_string(),
            before,
            after,
        );

        self.diffs.push(diff.clone());
        Some(diff)
    }

    /// Get all recorded diffs
    pub fn get_diffs(&self) -> &[MemoryDiff] {
        &self.diffs
    }

    /// Clear all snapshots and diffs
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.diffs.clear();
    }

    /// Get the number of snapshots
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the number of diffs
    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }

    /// Generate a summary report of all optimizations
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Memory Optimization Summary ===\n\n");
        report.push_str(&format!("Total Snapshots: {}\n", self.snapshot_count()));
        report.push_str(&format!("Total Comparisons: {}\n\n", self.diff_count()));

        if self.diffs.is_empty() {
            report.push_str("No comparisons recorded.\n");
            return report;
        }

        let total_savings: i64 = self.diffs.iter().map(|d| -d.total_allocated_diff()).sum();
        let total_savings_mb = total_savings as f64 / (1024.0 * 1024.0);

        report.push_str(&format!(
            "Total Memory Savings: {:.2} MB\n\n",
            total_savings_mb
        ));

        report.push_str("Individual Optimizations:\n");
        for (i, diff) in self.diffs.iter().enumerate() {
            report.push_str(&format!(
                "  {}. {} -> {}: {:+.2} MB ({:+.1}%)\n",
                i + 1,
                diff.before_name,
                diff.after_name,
                diff.total_allocated_diff() as f64 / (1024.0 * 1024.0),
                diff.total_allocated_pct_change()
            ));
        }

        report
    }
}

impl Default for MemoryDiffReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot() {
        let mut snapshot = MemorySnapshot::new();
        snapshot.add_operation_memory("op1", 1024);
        snapshot.add_layer_memory("layer1", 2048);

        assert_eq!(snapshot.total_allocated, 3072);
        assert_eq!(*snapshot.operation_memory.get("op1").unwrap(), 1024);
        assert_eq!(*snapshot.layer_memory.get("layer1").unwrap(), 2048);
    }

    #[test]
    fn test_memory_diff() {
        let mut before = MemorySnapshot::new();
        before.total_allocated = 1024 * 1024; // 1 MB
        before.gradient_memory = 512 * 1024; // 512 KB

        let mut after = MemorySnapshot::new();
        after.total_allocated = 768 * 1024; // 768 KB
        after.gradient_memory = 256 * 1024; // 256 KB

        let diff = MemoryDiff::new("before".to_string(), "after".to_string(), before, after);

        assert!(diff.is_improvement());
        assert_eq!(diff.memory_savings(), 256 * 1024);
        assert!(diff.total_allocated_pct_change() < 0.0);
    }

    #[test]
    fn test_memory_diff_reporter() {
        let mut reporter = MemoryDiffReporter::new();

        let mut snapshot1 = MemorySnapshot::new();
        snapshot1.total_allocated = 1024;

        let mut snapshot2 = MemorySnapshot::new();
        snapshot2.total_allocated = 512;

        reporter.snapshot_with("before", snapshot1);
        reporter.snapshot_with("after", snapshot2);

        let diff = reporter.diff("before", "after").unwrap();
        assert!(diff.is_improvement());
        assert_eq!(diff.memory_savings(), 512);
    }

    #[test]
    fn test_reporter_summary() {
        let mut reporter = MemoryDiffReporter::new();

        let mut snapshot1 = MemorySnapshot::new();
        snapshot1.total_allocated = 2048;

        let mut snapshot2 = MemorySnapshot::new();
        snapshot2.total_allocated = 1024;

        reporter.snapshot_with("s1", snapshot1);
        reporter.snapshot_with("s2", snapshot2);
        reporter.diff("s1", "s2");

        let summary = reporter.summary_report();
        assert!(summary.contains("Memory Optimization Summary"));
        assert!(summary.contains("Total Snapshots: 2"));
    }

    #[test]
    fn test_top_operation_changes() {
        let mut before = MemorySnapshot::new();
        before.add_operation_memory("op1", 1000);
        before.add_operation_memory("op2", 2000);
        before.add_operation_memory("op3", 3000);

        let mut after = MemorySnapshot::new();
        after.add_operation_memory("op1", 500);
        after.add_operation_memory("op2", 2500);
        after.add_operation_memory("op3", 2000);

        let diff = MemoryDiff::new("before".to_string(), "after".to_string(), before, after);

        let top_changes = diff.top_operation_changes(2);
        assert_eq!(top_changes.len(), 2);
    }

    #[test]
    fn test_diff_report_formatting() {
        let mut before = MemorySnapshot::new();
        before.total_allocated = 1024 * 1024;
        before.gradient_memory = 512 * 1024;

        let mut after = MemorySnapshot::new();
        after.total_allocated = 512 * 1024;
        after.gradient_memory = 256 * 1024;

        let diff = MemoryDiff::new(
            "before_opt".to_string(),
            "after_opt".to_string(),
            before,
            after,
        );

        let report = diff.format_report();
        assert!(report.contains("Memory Diff Report"));
        assert!(report.contains("before_opt"));
        assert!(report.contains("after_opt"));
        assert!(report.contains("SUCCESS"));
    }
}
