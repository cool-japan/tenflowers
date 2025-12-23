/// Comprehensive GPU Memory Diagnostics for TenfloweRS
///
/// This module provides advanced memory diagnostics, profiling, and leak detection
/// capabilities for GPU memory management, integrating allocation tracing with
/// memory pool diagnostics and providing actionable insights.
use super::memory_tracing::{
    AllocationInfo, GpuMemoryTracker, MemoryEvent, MemoryReport, MemoryStats, MemoryTracingConfig,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Memory fragmentation analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct FragmentationAnalysis {
    /// Total free memory
    pub total_free: usize,
    /// Largest contiguous free block
    pub largest_free_block: usize,
    /// Number of free blocks
    pub free_block_count: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_ratio: f64,
    /// Average free block size
    pub average_free_block_size: usize,
    /// Wasted memory due to fragmentation
    pub wasted_memory: usize,
}

impl FragmentationAnalysis {
    /// Compute fragmentation metrics from free block sizes
    pub fn analyze(free_blocks: &[usize]) -> Self {
        if free_blocks.is_empty() {
            return Self {
                total_free: 0,
                largest_free_block: 0,
                free_block_count: 0,
                fragmentation_ratio: 0.0,
                average_free_block_size: 0,
                wasted_memory: 0,
            };
        }

        let total_free: usize = free_blocks.iter().sum();
        let largest_free_block = *free_blocks.iter().max().unwrap_or(&0);
        let free_block_count = free_blocks.len();
        let average_free_block_size = if free_block_count > 0 {
            total_free / free_block_count
        } else {
            0
        };

        // Fragmentation ratio: measure how scattered the free memory is
        // High fragmentation = memory is divided into many small blocks
        // We use a combination of:
        // 1. How much memory is in small blocks vs the largest block
        // 2. How many blocks there are
        let fragmentation_ratio = if total_free > 0 && free_block_count > 0 {
            // Calculate what percentage of total memory is NOT in the largest block
            let scattered_ratio = 1.0 - (largest_free_block as f64 / total_free as f64);

            // Count how many "small" blocks we have (< 5% of largest)
            let small_block_count = free_blocks
                .iter()
                .filter(|&&size| size < largest_free_block / 20)
                .count() as f64;

            // Fragmentation increases with more small blocks and scattered memory
            let block_factor = (small_block_count / free_block_count as f64).max(0.0);

            // Weighted combination favoring block fragmentation
            (scattered_ratio * 0.3 + block_factor * 0.7).min(1.0)
        } else {
            0.0
        };

        // Wasted memory: memory that cannot be used efficiently due to fragmentation
        let wasted_memory = total_free.saturating_sub(largest_free_block);

        Self {
            total_free,
            largest_free_block,
            free_block_count,
            fragmentation_ratio,
            average_free_block_size,
            wasted_memory,
        }
    }

    /// Check if fragmentation is severe
    pub fn is_severe(&self) -> bool {
        self.fragmentation_ratio > 0.5
    }

    /// Get human-readable severity level
    pub fn severity_level(&self) -> &'static str {
        if self.fragmentation_ratio < 0.3 {
            "Low"
        } else if self.fragmentation_ratio < 0.5 {
            "Moderate"
        } else {
            "Severe"
        }
    }
}

/// Memory leak detection result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct LeakDetectionResult {
    /// Suspected leaks (allocations older than threshold)
    pub suspected_leaks: Vec<AllocationInfo>,
    /// Total memory potentially leaked
    pub total_leaked_bytes: usize,
    /// Number of suspected leak sites
    pub leak_count: usize,
    /// Leaks grouped by operation
    pub leaks_by_operation: HashMap<String, Vec<AllocationInfo>>,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
}

impl LeakDetectionResult {
    /// Create from list of suspected leaks
    pub fn from_allocations(allocations: Vec<AllocationInfo>) -> Self {
        let total_leaked_bytes = allocations.iter().map(|a| a.size).sum();
        let leak_count = allocations.len();

        let mut leaks_by_operation: HashMap<String, Vec<AllocationInfo>> = HashMap::new();
        for alloc in &allocations {
            leaks_by_operation
                .entry(alloc.operation.clone())
                .or_insert_with(Vec::new)
                .push(alloc.clone());
        }

        // Confidence based on age and count
        let confidence = if leak_count > 10 { 0.9 } else { 0.5 };

        Self {
            suspected_leaks: allocations,
            total_leaked_bytes,
            leak_count,
            leaks_by_operation,
            confidence,
        }
    }

    /// Check if leaks are detected
    pub fn has_leaks(&self) -> bool {
        !self.suspected_leaks.is_empty()
    }
}

/// Per-operation memory profiling result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct OperationProfile {
    /// Operation name
    pub operation: String,
    /// Total memory allocated
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Average allocation size
    pub average_size: usize,
    /// Total allocation time (if available)
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub total_time: Option<Duration>,
    /// Memory efficiency score (0.0 - 1.0)
    pub efficiency_score: f64,
}

impl OperationProfile {
    /// Create from allocation data
    pub fn new(operation: String, allocations: &[&AllocationInfo]) -> Self {
        let total_allocated = allocations.iter().map(|a| a.size).sum();
        let allocation_count = allocations.len();
        let average_size = if allocation_count > 0 {
            total_allocated / allocation_count
        } else {
            0
        };

        // Peak usage (sum of concurrent allocations)
        let peak_usage = total_allocated; // Simplified - could compute actual peak

        // Efficiency based on allocation patterns
        let efficiency_score = if allocation_count > 0 {
            // Higher score for fewer, larger allocations
            let size_variance = allocations
                .iter()
                .map(|a| (a.size as f64 - average_size as f64).abs())
                .sum::<f64>()
                / allocation_count as f64;
            let normalized_variance = (size_variance / average_size as f64).min(1.0);
            1.0 - normalized_variance
        } else {
            1.0
        };

        Self {
            operation,
            total_allocated,
            peak_usage,
            allocation_count,
            average_size,
            total_time: None,
            efficiency_score,
        }
    }
}

/// Comprehensive diagnostic report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct DiagnosticReport {
    /// Timestamp of report generation
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub timestamp: Instant,
    /// Memory statistics
    pub memory_stats: MemoryStats,
    /// Fragmentation analysis
    pub fragmentation: Option<FragmentationAnalysis>,
    /// Leak detection result
    pub leak_detection: LeakDetectionResult,
    /// Per-operation profiles
    pub operation_profiles: Vec<OperationProfile>,
    /// Top memory consumers
    pub top_consumers: Vec<(String, usize)>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl DiagnosticReport {
    /// Print a comprehensive diagnostic report
    pub fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════╗");
        println!("║   GPU Memory Diagnostic Report                      ║");
        println!("╚══════════════════════════════════════════════════════╝");

        println!("\n📊 Memory Statistics:");
        println!(
            "   Current Usage:  {:.2} MB",
            self.memory_stats.total_allocated as f64 / 1_048_576.0
        );
        println!(
            "   Peak Usage:     {:.2} MB",
            self.memory_stats.peak_usage as f64 / 1_048_576.0
        );
        println!(
            "   Active Allocs:  {}",
            self.memory_stats.active_allocations
        );
        println!(
            "   Total Allocs:   {}",
            self.memory_stats.total_allocations_lifetime
        );
        println!(
            "   Total Frees:    {}",
            self.memory_stats.total_frees_lifetime
        );

        if let Some(ref frag) = self.fragmentation {
            println!("\n🔍 Fragmentation Analysis:");
            println!(
                "   Severity:       {} ({:.1}%)",
                frag.severity_level(),
                frag.fragmentation_ratio * 100.0
            );
            println!("   Free Blocks:    {}", frag.free_block_count);
            println!(
                "   Largest Block:  {:.2} MB",
                frag.largest_free_block as f64 / 1_048_576.0
            );
            println!(
                "   Wasted Memory:  {:.2} MB",
                frag.wasted_memory as f64 / 1_048_576.0
            );
        }

        if self.leak_detection.has_leaks() {
            println!("\n⚠️  Memory Leak Detection:");
            println!("   Suspected Leaks:  {}", self.leak_detection.leak_count);
            println!(
                "   Leaked Memory:    {:.2} MB",
                self.leak_detection.total_leaked_bytes as f64 / 1_048_576.0
            );
            println!(
                "   Confidence:       {:.0}%",
                self.leak_detection.confidence * 100.0
            );

            println!("\n   Leaks by Operation:");
            for (op, leaks) in &self.leak_detection.leaks_by_operation {
                let total: usize = leaks.iter().map(|l| l.size).sum();
                println!(
                    "      {} - {:.2} MB ({} allocations)",
                    op,
                    total as f64 / 1_048_576.0,
                    leaks.len()
                );
            }
        }

        println!("\n🎯 Top Memory Consumers:");
        for (i, (op, size)) in self.top_consumers.iter().enumerate().take(5) {
            println!(
                "   {}. {} - {:.2} MB",
                i + 1,
                op,
                *size as f64 / 1_048_576.0
            );
        }

        if !self.operation_profiles.is_empty() {
            println!("\n📈 Operation Profiles:");
            for profile in self.operation_profiles.iter().take(5) {
                println!(
                    "   {} - {:.2} MB (efficiency: {:.0}%)",
                    profile.operation,
                    profile.total_allocated as f64 / 1_048_576.0,
                    profile.efficiency_score * 100.0
                );
            }
        }

        if !self.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("   {}. {}", i + 1, rec);
            }
        }

        println!("\n═══════════════════════════════════════════════════════\n");
    }

    /// Export to JSON format
    #[cfg(feature = "serialize")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// GPU Memory Diagnostics Engine
pub struct GpuMemoryDiagnostics {
    tracker: Arc<Mutex<GpuMemoryTracker>>,
    config: DiagnosticsConfig,
}

/// Diagnostics configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct DiagnosticsConfig {
    /// Age threshold for leak detection (default: 5 minutes)
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub leak_detection_threshold: Duration,
    /// Enable automatic periodic diagnostics
    pub auto_diagnostics: bool,
    /// Diagnostics interval
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub diagnostics_interval: Duration,
    /// Enable fragmentation analysis
    pub analyze_fragmentation: bool,
    /// Enable operation profiling
    pub enable_profiling: bool,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            leak_detection_threshold: Duration::from_secs(300), // 5 minutes
            auto_diagnostics: false,
            diagnostics_interval: Duration::from_secs(60),
            analyze_fragmentation: true,
            enable_profiling: true,
        }
    }
}

impl GpuMemoryDiagnostics {
    /// Create a new diagnostics engine with existing tracker
    pub fn new(tracker: Arc<Mutex<GpuMemoryTracker>>) -> Self {
        Self::with_config(tracker, DiagnosticsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(tracker: Arc<Mutex<GpuMemoryTracker>>, config: DiagnosticsConfig) -> Self {
        Self { tracker, config }
    }

    /// Run comprehensive diagnostics
    pub fn run_diagnostics(&self) -> DiagnosticReport {
        let tracker = self.tracker.lock().unwrap();

        // Get base statistics
        let memory_stats = tracker.global_stats().clone();

        // Leak detection
        let suspected_leaks = tracker
            .find_potential_leaks(self.config.leak_detection_threshold)
            .into_iter()
            .cloned()
            .collect();
        let leak_detection = LeakDetectionResult::from_allocations(suspected_leaks);

        // Operation profiling
        let operation_profiles = if self.config.enable_profiling {
            self.profile_operations(&tracker)
        } else {
            Vec::new()
        };

        // Top consumers
        let usage_by_op = tracker.usage_by_operation();
        let mut top_consumers: Vec<_> = usage_by_op.into_iter().collect();
        top_consumers.sort_by(|a, b| b.1.cmp(&a.1));
        top_consumers.truncate(10);

        // Generate recommendations
        let recommendations =
            self.generate_recommendations(&memory_stats, &leak_detection, &operation_profiles);

        // Fragmentation analysis (if enabled)
        let fragmentation = if self.config.analyze_fragmentation {
            // Simulate free blocks from active allocations for analysis
            let active = tracker.active_allocations();
            let sizes: Vec<usize> = active.values().map(|a| a.size).collect();
            if !sizes.is_empty() {
                Some(FragmentationAnalysis::analyze(&sizes))
            } else {
                None
            }
        } else {
            None
        };

        DiagnosticReport {
            timestamp: Instant::now(),
            memory_stats,
            fragmentation,
            leak_detection,
            operation_profiles,
            top_consumers,
            recommendations,
        }
    }

    /// Profile memory usage by operation
    fn profile_operations(&self, tracker: &GpuMemoryTracker) -> Vec<OperationProfile> {
        let active = tracker.active_allocations();

        // Group by operation
        let mut by_operation: HashMap<String, Vec<&AllocationInfo>> = HashMap::new();
        for alloc in active.values() {
            by_operation
                .entry(alloc.operation.clone())
                .or_insert_with(Vec::new)
                .push(alloc);
        }

        // Create profiles
        let mut profiles: Vec<_> = by_operation
            .into_iter()
            .map(|(op, allocs)| OperationProfile::new(op, &allocs))
            .collect();

        profiles.sort_by(|a, b| b.total_allocated.cmp(&a.total_allocated));
        profiles
    }

    /// Generate actionable recommendations
    fn generate_recommendations(
        &self,
        stats: &MemoryStats,
        leak_detection: &LeakDetectionResult,
        profiles: &[OperationProfile],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // High memory usage
        if stats.total_allocated > 1_073_741_824 {
            // > 1GB
            recommendations.push(
                "High memory usage detected. Consider reducing batch sizes or using gradient checkpointing.".to_string()
            );
        }

        // Memory leaks
        if leak_detection.has_leaks() && leak_detection.confidence > 0.7 {
            recommendations.push(format!(
                "Potential memory leaks detected in {} operations. Review deallocation logic.",
                leak_detection.leaks_by_operation.len()
            ));
        }

        // Inefficient operations
        for profile in profiles.iter().take(3) {
            if profile.efficiency_score < 0.5 {
                recommendations.push(format!(
                    "Operation '{}' has low memory efficiency ({:.0}%). Consider batching or pooling.",
                    profile.operation,
                    profile.efficiency_score * 100.0
                ));
            }
        }

        // Many small allocations
        if stats.active_allocations > 1000 && stats.average_allocation_size < 1024 {
            recommendations.push(
                "High number of small allocations. Consider using memory pooling or buffer reuse."
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("Memory usage looks healthy. No major issues detected.".to_string());
        }

        recommendations
    }

    /// Check for memory leaks
    pub fn check_for_leaks(&self) -> LeakDetectionResult {
        let tracker = self.tracker.lock().unwrap();
        let suspected_leaks = tracker
            .find_potential_leaks(self.config.leak_detection_threshold)
            .into_iter()
            .cloned()
            .collect();
        LeakDetectionResult::from_allocations(suspected_leaks)
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.tracker.lock().unwrap().current_usage()
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.tracker.lock().unwrap().peak_usage()
    }

    /// Reset diagnostics
    pub fn reset(&self) {
        self.tracker.lock().unwrap().reset();
    }
}

/// Global diagnostics instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_GPU_DIAGNOSTICS: GpuMemoryDiagnostics = {
        GpuMemoryDiagnostics::new(
            Arc::clone(&super::memory_tracing::GLOBAL_GPU_MEMORY_TRACKER)
        )
    };
}

/// Convenience function to run diagnostics
pub fn run_gpu_diagnostics() -> DiagnosticReport {
    GLOBAL_GPU_DIAGNOSTICS.run_diagnostics()
}

/// Convenience function to print diagnostics
pub fn print_gpu_diagnostics() {
    run_gpu_diagnostics().print();
}

/// Convenience function to check for leaks
pub fn check_gpu_memory_leaks() -> LeakDetectionResult {
    GLOBAL_GPU_DIAGNOSTICS.check_for_leaks()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragmentation_analysis() {
        let free_blocks = vec![1000, 500, 250, 125];
        let analysis = FragmentationAnalysis::analyze(&free_blocks);

        assert_eq!(analysis.total_free, 1875);
        assert_eq!(analysis.largest_free_block, 1000);
        assert_eq!(analysis.free_block_count, 4);
        assert!(analysis.fragmentation_ratio > 0.0);
        assert!(analysis.fragmentation_ratio < 1.0);
    }

    #[test]
    fn test_fragmentation_severity() {
        let low_frag = vec![1000, 900, 800];
        let analysis = FragmentationAnalysis::analyze(&low_frag);
        assert!(!analysis.is_severe());

        let high_frag = vec![1000, 10, 10, 10, 10];
        let analysis = FragmentationAnalysis::analyze(&high_frag);
        assert!(analysis.is_severe());
    }

    #[test]
    fn test_leak_detection_result() {
        let alloc1 = AllocationInfo::new(1, 1024, 0, "op1".to_string());
        let alloc2 = AllocationInfo::new(2, 2048, 0, "op2".to_string());

        let result = LeakDetectionResult::from_allocations(vec![alloc1, alloc2]);

        assert_eq!(result.leak_count, 2);
        assert_eq!(result.total_leaked_bytes, 3072);
        assert!(result.has_leaks());
    }

    #[test]
    fn test_diagnostics_engine() {
        let tracker = Arc::new(Mutex::new(GpuMemoryTracker::new()));
        let diagnostics = GpuMemoryDiagnostics::new(tracker.clone());

        // Add some allocations
        {
            let mut t = tracker.lock().unwrap();
            t.record_allocation(1024, 0, "test_op".to_string());
            t.record_allocation(2048, 0, "test_op".to_string());
        }

        let report = diagnostics.run_diagnostics();
        assert!(report.memory_stats.active_allocations > 0);
        assert!(!report.recommendations.is_empty());
    }
}
