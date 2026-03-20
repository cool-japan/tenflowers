//! Performance reporting types for eager execution

use std::time::Duration;

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_hits: usize,
    pub hit_rate: f64,
    pub avg_execution_time: Duration,
}

/// Eager execution performance report
#[derive(Debug, Clone)]
pub struct EagerPerformanceReport {
    pub total_operations: usize,
    pub operations_meeting_target: usize,
    pub success_rate: f64,
    pub avg_overhead: Duration,
    pub min_overhead: Duration,
    pub max_overhead: Duration,
    pub cache_statistics: CacheStatistics,
    pub cache_hit_rate: f64,
    pub target_overhead: Duration,
    pub recommendations: Vec<String>,
}

impl Default for EagerPerformanceReport {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_meeting_target: 0,
            success_rate: 0.0,
            avg_overhead: Duration::ZERO,
            min_overhead: Duration::ZERO,
            max_overhead: Duration::ZERO,
            cache_statistics: CacheStatistics {
                total_entries: 0,
                total_hits: 0,
                hit_rate: 0.0,
                avg_execution_time: Duration::ZERO,
            },
            cache_hit_rate: 0.0,
            target_overhead: Duration::from_millis(1),
            recommendations: Vec::new(),
        }
    }
}

impl EagerPerformanceReport {
    /// Print a formatted performance report
    pub fn print_report(&self) {
        println!("Eager Execution Performance Report");
        println!("=====================================");
        println!();
        println!("Overall Performance:");
        println!("  Total operations: {}", self.total_operations);
        println!(
            "  Operations meeting target: {}/{}",
            self.operations_meeting_target, self.total_operations
        );
        println!("  Success rate: {:.1}%", self.success_rate * 100.0);
        println!();

        println!("Overhead Analysis:");
        println!("  Target overhead: {:?}", self.target_overhead);
        println!("  Average overhead: {:?}", self.avg_overhead);
        println!("  Minimum overhead: {:?}", self.min_overhead);
        println!("  Maximum overhead: {:?}", self.max_overhead);

        let target_met = self.avg_overhead <= self.target_overhead;
        if target_met {
            println!("  Average overhead meets target!");
        } else {
            let gap = self.avg_overhead.as_nanos() - self.target_overhead.as_nanos();
            println!("  Average overhead exceeds target by {gap}ns");
        }
        println!();

        println!("Cache Performance:");
        println!("  Cache entries: {}", self.cache_statistics.total_entries);
        println!("  Cache hit rate: {:.1}%", self.cache_hit_rate * 100.0);
        println!(
            "  Average execution time: {:?}",
            self.cache_statistics.avg_execution_time
        );
        println!();

        if !self.recommendations.is_empty() {
            println!("Recommendations:");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }

        println!("=====================================");
    }
}
