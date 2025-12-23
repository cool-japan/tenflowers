//! GPU Memory Diagnostics and Allocation Tracing
//!
//! This module provides comprehensive memory allocation tracking, diagnostics,
//! and reporting for GPU memory management.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::Result;

/// Detailed allocation event for tracing
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Unique allocation ID
    pub id: u64,
    /// Timestamp when allocated
    pub timestamp: Instant,
    /// Requested size in bytes
    pub requested_size: usize,
    /// Actual allocated size in bytes (may be larger due to alignment)
    pub allocated_size: usize,
    /// Allocation alignment requirement
    pub alignment: usize,
    /// Stack trace or allocation source (for debugging)
    pub source: Option<String>,
    /// Device ID where allocated
    pub device_id: usize,
    /// Pool ID within device
    pub pool_id: Option<usize>,
    /// Whether this is still active
    pub is_active: bool,
    /// When deallocated (if applicable)
    pub deallocated_at: Option<Instant>,
    /// Lifetime in microseconds (if deallocated)
    pub lifetime_us: Option<u64>,
}

impl AllocationEvent {
    /// Create a new allocation event
    pub fn new(
        id: u64,
        requested_size: usize,
        allocated_size: usize,
        alignment: usize,
        device_id: usize,
        source: Option<String>,
    ) -> Self {
        Self {
            id,
            timestamp: Instant::now(),
            requested_size,
            allocated_size,
            alignment,
            source,
            device_id,
            pool_id: None,
            is_active: true,
            deallocated_at: None,
            lifetime_us: None,
        }
    }

    /// Mark allocation as deallocated
    pub fn mark_deallocated(&mut self) {
        let now = Instant::now();
        self.deallocated_at = Some(now);
        self.lifetime_us = Some(now.duration_since(self.timestamp).as_micros() as u64);
        self.is_active = false;
    }

    /// Get lifetime duration
    pub fn lifetime(&self) -> Option<Duration> {
        self.deallocated_at
            .map(|dealloc| dealloc.duration_since(self.timestamp))
    }
}

/// Memory leak detection report
#[derive(Debug, Clone)]
pub struct LeakReport {
    /// Number of leaked allocations
    pub leak_count: usize,
    /// Total leaked memory in bytes
    pub leaked_bytes: usize,
    /// Individual leak details
    pub leaks: Vec<AllocationEvent>,
    /// Oldest leak age
    pub oldest_leak_age: Option<Duration>,
    /// Average leak size
    pub average_leak_size: f64,
}

/// Allocation statistics summary
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total number of allocations
    pub total_allocations: u64,
    /// Total number of deallocations
    pub total_deallocations: u64,
    /// Currently active allocations
    pub active_allocations: usize,
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Total bytes ever allocated
    pub total_allocated: usize,
    /// Total bytes deallocated
    pub total_deallocated: usize,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Average lifetime in microseconds
    pub average_lifetime_us: f64,
    /// Allocation rate (allocs/second)
    pub allocation_rate: f64,
    /// Deallocation rate (deallocs/second)
    pub deallocation_rate: f64,
    /// Fragmentation score (0.0 = no fragmentation, 1.0 = severe)
    pub fragmentation_score: f64,
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            current_usage: 0,
            peak_usage: 0,
            total_allocated: 0,
            total_deallocated: 0,
            average_allocation_size: 0.0,
            average_lifetime_us: 0.0,
            allocation_rate: 0.0,
            deallocation_rate: 0.0,
            fragmentation_score: 0.0,
        }
    }
}

/// Allocation size distribution bucket
#[derive(Debug, Clone)]
pub struct SizeBucket {
    /// Minimum size for this bucket (inclusive)
    pub min_size: usize,
    /// Maximum size for this bucket (exclusive)
    pub max_size: usize,
    /// Number of allocations in this bucket
    pub count: usize,
    /// Total bytes in this bucket
    pub total_bytes: usize,
}

/// GPU Memory Diagnostics Tracker
pub struct MemoryDiagnostics {
    /// All allocation events (both active and historical)
    events: Arc<Mutex<Vec<AllocationEvent>>>,
    /// Active allocations indexed by ID
    active: Arc<Mutex<HashMap<u64, usize>>>,
    /// Next allocation ID
    next_id: Arc<Mutex<u64>>,
    /// Start time for rate calculations
    start_time: Instant,
    /// Diagnostic session metadata
    session_metadata: Arc<Mutex<HashMap<String, String>>>,
    /// Maximum events to keep in history
    max_history_size: usize,
    /// Enable detailed tracing
    detailed_tracing: bool,
}

impl MemoryDiagnostics {
    /// Create a new diagnostics tracker
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            active: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
            session_metadata: Arc::new(Mutex::new(HashMap::new())),
            max_history_size: 100_000,
            detailed_tracing: false,
        }
    }

    /// Create diagnostics with custom configuration
    pub fn with_config(max_history_size: usize, detailed_tracing: bool) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            active: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
            session_metadata: Arc::new(Mutex::new(HashMap::new())),
            max_history_size,
            detailed_tracing,
        }
    }

    /// Record a new allocation
    pub fn record_allocation(
        &self,
        requested_size: usize,
        allocated_size: usize,
        alignment: usize,
        device_id: usize,
        source: Option<String>,
    ) -> u64 {
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let event = AllocationEvent::new(
            id,
            requested_size,
            allocated_size,
            alignment,
            device_id,
            source,
        );

        let mut events = self.events.lock().unwrap();
        let event_idx = events.len();
        events.push(event);
        drop(events);

        let mut active = self.active.lock().unwrap();
        active.insert(id, event_idx);

        id
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, id: u64) {
        let mut active = self.active.lock().unwrap();
        if let Some(event_idx) = active.remove(&id) {
            drop(active);

            let mut events = self.events.lock().unwrap();
            if let Some(event) = events.get_mut(event_idx) {
                event.mark_deallocated();
            }
        }
    }

    /// Get current allocation statistics
    pub fn get_statistics(&self) -> AllocationStats {
        let events = self.events.lock().unwrap();
        let active = self.active.lock().unwrap();

        let mut stats = AllocationStats {
            active_allocations: active.len(),
            ..Default::default()
        };

        let mut current_usage = 0usize;
        let mut total_lifetime_us = 0u64;
        let mut lifetime_count = 0usize;

        for event in events.iter() {
            stats.total_allocations += 1;
            stats.total_allocated += event.allocated_size;

            if event.is_active {
                current_usage += event.allocated_size;
            } else {
                stats.total_deallocations += 1;
                stats.total_deallocated += event.allocated_size;
                if let Some(lifetime) = event.lifetime_us {
                    total_lifetime_us += lifetime;
                    lifetime_count += 1;
                }
            }
        }

        stats.current_usage = current_usage;
        stats.peak_usage = stats.peak_usage.max(current_usage);

        if stats.total_allocations > 0 {
            stats.average_allocation_size =
                stats.total_allocated as f64 / stats.total_allocations as f64;
        }

        if lifetime_count > 0 {
            stats.average_lifetime_us = total_lifetime_us as f64 / lifetime_count as f64;
        }

        // Calculate rates
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            stats.allocation_rate = stats.total_allocations as f64 / elapsed;
            stats.deallocation_rate = stats.total_deallocations as f64 / elapsed;
        }

        // Estimate fragmentation (simplified metric)
        if stats.active_allocations > 0 {
            stats.fragmentation_score =
                1.0 - (stats.current_usage as f64 / stats.peak_usage as f64).min(1.0);
        }

        stats
    }

    /// Detect memory leaks (allocations alive longer than threshold)
    pub fn detect_leaks(&self, age_threshold: Duration) -> LeakReport {
        let events = self.events.lock().unwrap();
        let now = Instant::now();

        let mut leaks = Vec::new();
        let mut leaked_bytes = 0usize;
        let mut oldest_leak_age = None;

        for event in events.iter() {
            if event.is_active {
                let age = now.duration_since(event.timestamp);
                if age > age_threshold {
                    leaked_bytes += event.allocated_size;
                    oldest_leak_age = Some(
                        oldest_leak_age
                            .map(|current: Duration| current.max(age))
                            .unwrap_or(age),
                    );
                    leaks.push(event.clone());
                }
            }
        }

        let average_leak_size = if !leaks.is_empty() {
            leaked_bytes as f64 / leaks.len() as f64
        } else {
            0.0
        };

        LeakReport {
            leak_count: leaks.len(),
            leaked_bytes,
            leaks,
            oldest_leak_age,
            average_leak_size,
        }
    }

    /// Get allocation size distribution
    pub fn get_size_distribution(&self) -> Vec<SizeBucket> {
        let events = self.events.lock().unwrap();

        // Define size buckets (in bytes)
        let bucket_ranges = [
            (0, 1024),                       // < 1KB
            (1024, 16 * 1024),               // 1KB - 16KB
            (16 * 1024, 256 * 1024),         // 16KB - 256KB
            (256 * 1024, 1024 * 1024),       // 256KB - 1MB
            (1024 * 1024, 16 * 1024 * 1024), // 1MB - 16MB
            (16 * 1024 * 1024, usize::MAX),  // > 16MB
        ];

        let mut buckets: Vec<SizeBucket> = bucket_ranges
            .iter()
            .map(|&(min, max)| SizeBucket {
                min_size: min,
                max_size: max,
                count: 0,
                total_bytes: 0,
            })
            .collect();

        for event in events.iter().filter(|e| e.is_active) {
            for bucket in buckets.iter_mut() {
                if event.allocated_size >= bucket.min_size && event.allocated_size < bucket.max_size
                {
                    bucket.count += 1;
                    bucket.total_bytes += event.allocated_size;
                    break;
                }
            }
        }

        buckets
    }

    /// Get allocation timeline (for visualization)
    pub fn get_timeline(&self, bucket_duration: Duration) -> Vec<(Instant, usize, usize)> {
        let events = self.events.lock().unwrap();

        if events.is_empty() {
            return Vec::new();
        }

        let start_time = events[0].timestamp;
        let end_time = Instant::now();
        let total_duration = end_time.duration_since(start_time);
        let num_buckets =
            (total_duration.as_secs_f64() / bucket_duration.as_secs_f64()).ceil() as usize;

        let mut timeline = Vec::with_capacity(num_buckets);
        for i in 0..num_buckets {
            let bucket_start = start_time + bucket_duration * i as u32;
            timeline.push((bucket_start, 0, 0)); // (timestamp, allocations, deallocations)
        }

        for event in events.iter() {
            let alloc_offset = event.timestamp.duration_since(start_time);
            let alloc_bucket =
                (alloc_offset.as_secs_f64() / bucket_duration.as_secs_f64()) as usize;
            if alloc_bucket < timeline.len() {
                timeline[alloc_bucket].1 += 1;
            }

            if let Some(dealloc_time) = event.deallocated_at {
                let dealloc_offset = dealloc_time.duration_since(start_time);
                let dealloc_bucket =
                    (dealloc_offset.as_secs_f64() / bucket_duration.as_secs_f64()) as usize;
                if dealloc_bucket < timeline.len() {
                    timeline[dealloc_bucket].2 += 1;
                }
            }
        }

        timeline
    }

    /// Clear old events to manage memory
    pub fn cleanup_old_events(&self, retention_duration: Duration) {
        let mut events = self.events.lock().unwrap();
        let now = Instant::now();

        events.retain(|event| {
            event.is_active
                || event
                    .deallocated_at
                    .map(|dealloc| now.duration_since(dealloc) < retention_duration)
                    .unwrap_or(false)
        });

        // Enforce max history size
        if events.len() > self.max_history_size {
            let excess = events.len() - self.max_history_size;
            events.drain(0..excess);
        }
    }

    /// Set session metadata
    pub fn set_metadata(&self, key: String, value: String) {
        let mut metadata = self.session_metadata.lock().unwrap();
        metadata.insert(key, value);
    }

    /// Get session metadata
    pub fn get_metadata(&self, key: &str) -> Option<String> {
        let metadata = self.session_metadata.lock().unwrap();
        metadata.get(key).cloned()
    }

    /// Reset all diagnostics
    pub fn reset(&self) {
        let mut events = self.events.lock().unwrap();
        events.clear();
        drop(events);

        let mut active = self.active.lock().unwrap();
        active.clear();
        drop(active);

        let mut next_id = self.next_id.lock().unwrap();
        *next_id = 0;
    }

    /// Generate a diagnostic report
    pub fn generate_report(&self) -> String {
        let stats = self.get_statistics();
        let leaks = self.detect_leaks(Duration::from_secs(300)); // 5 minute threshold
        let distribution = self.get_size_distribution();

        let mut report = String::new();
        report.push_str("=== GPU Memory Diagnostics Report ===\n\n");

        // Statistics
        report.push_str(&format!("Total Allocations: {}\n", stats.total_allocations));
        report.push_str(&format!(
            "Total Deallocations: {}\n",
            stats.total_deallocations
        ));
        report.push_str(&format!(
            "Active Allocations: {}\n",
            stats.active_allocations
        ));
        report.push_str(&format!(
            "Current Usage: {:.2} MB\n",
            stats.current_usage as f64 / 1_048_576.0
        ));
        report.push_str(&format!(
            "Peak Usage: {:.2} MB\n",
            stats.peak_usage as f64 / 1_048_576.0
        ));
        report.push_str(&format!(
            "Average Allocation Size: {:.2} KB\n",
            stats.average_allocation_size / 1024.0
        ));
        report.push_str(&format!(
            "Average Lifetime: {:.2} ms\n",
            stats.average_lifetime_us / 1000.0
        ));
        report.push_str(&format!(
            "Allocation Rate: {:.2} allocs/sec\n",
            stats.allocation_rate
        ));
        report.push_str(&format!(
            "Fragmentation Score: {:.2}\n\n",
            stats.fragmentation_score
        ));

        // Leak detection
        report.push_str(&format!("Detected Leaks: {}\n", leaks.leak_count));
        if leaks.leak_count > 0 {
            report.push_str(&format!(
                "Leaked Memory: {:.2} MB\n",
                leaks.leaked_bytes as f64 / 1_048_576.0
            ));
            report.push_str(&format!(
                "Average Leak Size: {:.2} KB\n",
                leaks.average_leak_size / 1024.0
            ));
            if let Some(age) = leaks.oldest_leak_age {
                report.push_str(&format!(
                    "Oldest Leak Age: {:.2} seconds\n",
                    age.as_secs_f64()
                ));
            }
        }
        report.push('\n');

        // Size distribution
        report.push_str("Size Distribution:\n");
        for bucket in distribution.iter() {
            if bucket.count > 0 {
                let min_kb = bucket.min_size / 1024;
                let max_kb = if bucket.max_size == usize::MAX {
                    String::from("∞")
                } else {
                    format!("{}", bucket.max_size / 1024)
                };
                report.push_str(&format!(
                    "  {} KB - {} KB: {} allocations ({:.2} MB)\n",
                    min_kb,
                    max_kb,
                    bucket.count,
                    bucket.total_bytes as f64 / 1_048_576.0
                ));
            }
        }

        report
    }
}

impl Default for MemoryDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global diagnostics instance for easy access
static GLOBAL_DIAGNOSTICS: once_cell::sync::Lazy<MemoryDiagnostics> =
    once_cell::sync::Lazy::new(|| MemoryDiagnostics::new());

/// Get the global diagnostics instance
pub fn global_diagnostics() -> &'static MemoryDiagnostics {
    &GLOBAL_DIAGNOSTICS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_tracking() {
        let diag = MemoryDiagnostics::new();

        let id1 = diag.record_allocation(1024, 1024, 16, 0, Some("test1".to_string()));
        let id2 = diag.record_allocation(2048, 2048, 16, 0, Some("test2".to_string()));

        let stats = diag.get_statistics();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.current_usage, 3072);

        diag.record_deallocation(id1);

        let stats = diag.get_statistics();
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.current_usage, 2048);

        diag.record_deallocation(id2);

        let stats = diag.get_statistics();
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.current_usage, 0);
    }

    #[test]
    fn test_leak_detection() {
        let diag = MemoryDiagnostics::new();

        let _id1 = diag.record_allocation(1024, 1024, 16, 0, Some("leak_test".to_string()));

        // Immediate check should show no leaks
        let leaks = diag.detect_leaks(Duration::from_secs(1));
        assert_eq!(leaks.leak_count, 0);

        // After waiting, should detect leak
        std::thread::sleep(Duration::from_millis(100));
        let leaks = diag.detect_leaks(Duration::from_millis(50));
        assert_eq!(leaks.leak_count, 1);
        assert_eq!(leaks.leaked_bytes, 1024);
    }

    #[test]
    fn test_size_distribution() {
        let diag = MemoryDiagnostics::new();

        diag.record_allocation(512, 512, 16, 0, None); // < 1KB
        diag.record_allocation(2048, 2048, 16, 0, None); // 1-16KB
        diag.record_allocation(100000, 100000, 16, 0, None); // 16-256KB

        let distribution = diag.get_size_distribution();

        assert_eq!(distribution[0].count, 1); // < 1KB
        assert_eq!(distribution[1].count, 1); // 1-16KB
        assert_eq!(distribution[2].count, 1); // 16-256KB
    }

    #[test]
    fn test_statistics_calculation() {
        let diag = MemoryDiagnostics::new();

        let id1 = diag.record_allocation(1000, 1024, 16, 0, None);
        let id2 = diag.record_allocation(2000, 2048, 16, 0, None);

        std::thread::sleep(Duration::from_millis(10));
        diag.record_deallocation(id1);

        let stats = diag.get_statistics();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.active_allocations, 1);
        assert!(stats.average_allocation_size > 0.0);
        assert!(stats.average_lifetime_us > 0.0);

        diag.record_deallocation(id2);
    }

    #[test]
    fn test_reset() {
        let diag = MemoryDiagnostics::new();

        diag.record_allocation(1024, 1024, 16, 0, None);
        diag.record_allocation(2048, 2048, 16, 0, None);

        let stats = diag.get_statistics();
        assert_eq!(stats.total_allocations, 2);

        diag.reset();

        let stats = diag.get_statistics();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.active_allocations, 0);
    }

    #[test]
    fn test_report_generation() {
        let diag = MemoryDiagnostics::new();

        diag.record_allocation(1024, 1024, 16, 0, Some("test".to_string()));
        diag.record_allocation(2048, 2048, 16, 0, None);

        let report = diag.generate_report();
        assert!(report.contains("GPU Memory Diagnostics Report"));
        assert!(report.contains("Total Allocations: 2"));
        assert!(report.contains("Active Allocations: 2"));
    }
}
