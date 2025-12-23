//! Memory Pool Diagnostics Integration
//!
//! This module provides integration between memory pools and the GPU memory diagnostics
//! system, enabling comprehensive monitoring, health analysis, and automatic optimization
//! of memory pool behavior.

use super::pools::{MemoryPool, MemoryPoolStats, MemoryPressureLevel};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use crate::gpu::memory_diagnostics::{
    DiagnosticReport, FragmentationAnalysis, GpuMemoryDiagnostics, LeakDetectionResult,
};

/// Memory pool health status
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PoolHealthStatus {
    /// Pool is operating normally
    #[default]
    Healthy,
    /// Minor issues detected (mild fragmentation or pressure)
    Warning,
    /// Significant issues requiring attention
    Degraded,
    /// Critical state requiring immediate intervention
    Critical,
}

/// Memory pool health metrics
#[derive(Debug, Clone)]
pub struct PoolHealthMetrics {
    pub status: PoolHealthStatus,
    pub fragmentation_score: f32,     // 0.0-1.0, where 1.0 is worst
    pub pressure_score: f32,          // 0.0-1.0, where 1.0 is critical
    pub efficiency_score: f32,        // 0.0-1.0, where 1.0 is best
    pub allocation_success_rate: f32, // 0.0-1.0
    pub average_allocation_time_us: f32,
    pub defragmentation_needed: bool,
    pub recommendations: Vec<String>,
}

impl PoolHealthMetrics {
    /// Create health metrics from pool statistics
    pub fn from_stats(stats: &MemoryPoolStats) -> Self {
        let fragmentation_score = stats.fragmentation_ratio;
        let pressure_score = stats.memory_pressure;

        // Calculate efficiency: how well is memory being utilized?
        let efficiency_score = if stats.total_allocated + stats.total_free > 0 {
            1.0 - fragmentation_score
        } else {
            1.0
        };

        // Estimate allocation success rate (simplified)
        let allocation_success_rate = if stats.allocation_count > 0 {
            1.0 - (fragmentation_score * 0.5) // High fragmentation reduces success
        } else {
            1.0
        };

        // Determine health status
        let status = if pressure_score > 0.95 || fragmentation_score > 0.7 {
            PoolHealthStatus::Critical
        } else if pressure_score > 0.8 || fragmentation_score > 0.5 {
            PoolHealthStatus::Degraded
        } else if pressure_score > 0.6 || fragmentation_score > 0.3 {
            PoolHealthStatus::Warning
        } else {
            PoolHealthStatus::Healthy
        };

        // Check if defragmentation is needed
        let defragmentation_needed = fragmentation_score > 0.25
            || (stats.blocks_free > 10 && stats.largest_free_block < stats.total_free / 2);

        // Generate recommendations
        let mut recommendations = Vec::new();

        if pressure_score > 0.8 {
            recommendations.push(
                "High memory pressure: Consider increasing pool size or reducing allocations"
                    .to_string(),
            );
        }

        if fragmentation_score > 0.5 {
            recommendations
                .push("Severe fragmentation detected: Run defragmentation immediately".to_string());
        } else if fragmentation_score > 0.3 {
            recommendations
                .push("Moderate fragmentation: Schedule defragmentation soon".to_string());
        }

        if stats.blocks_free > 20 {
            recommendations.push(format!(
                "High block count ({} free blocks): Fragmentation likely, defragmentation recommended",
                stats.blocks_free
            ));
        }

        if efficiency_score < 0.5 {
            recommendations.push("Low memory efficiency: Review allocation patterns".to_string());
        }

        Self {
            status,
            fragmentation_score,
            pressure_score,
            efficiency_score,
            allocation_success_rate,
            average_allocation_time_us: 0.0, // Would need timing data
            defragmentation_needed,
            recommendations,
        }
    }

    /// Print health metrics in a user-friendly format
    pub fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════╗");
        println!("║   Memory Pool Health Report                         ║");
        println!("╚══════════════════════════════════════════════════════╝");

        let status_icon = match self.status {
            PoolHealthStatus::Healthy => "✅",
            PoolHealthStatus::Warning => "⚠️ ",
            PoolHealthStatus::Degraded => "🔶",
            PoolHealthStatus::Critical => "🔴",
        };

        println!("\n{} Status: {:?}", status_icon, self.status);
        println!("\nMetrics:");
        println!(
            "  • Fragmentation:     {:.1}% {}",
            self.fragmentation_score * 100.0,
            if self.fragmentation_score > 0.5 {
                "⚠️"
            } else {
                ""
            }
        );
        println!(
            "  • Memory Pressure:   {:.1}% {}",
            self.pressure_score * 100.0,
            if self.pressure_score > 0.8 {
                "⚠️"
            } else {
                ""
            }
        );
        println!(
            "  • Efficiency:        {:.1}%",
            self.efficiency_score * 100.0
        );
        println!(
            "  • Success Rate:      {:.1}%",
            self.allocation_success_rate * 100.0
        );

        if self.defragmentation_needed {
            println!("\n⚠️  Defragmentation recommended");
        }

        if !self.recommendations.is_empty() {
            println!("\nRecommendations:");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }

        println!();
    }
}

/// Configuration for automatic memory pool optimization
#[derive(Debug, Clone)]
pub struct PoolOptimizationConfig {
    /// Enable automatic defragmentation
    pub auto_defrag_enabled: bool,

    /// Fragmentation threshold to trigger defragmentation (0.0-1.0)
    pub auto_defrag_threshold: f32,

    /// Minimum interval between defragmentation runs
    pub defrag_min_interval: Duration,

    /// Enable automatic health monitoring
    pub health_monitoring_enabled: bool,

    /// Interval for health checks
    pub health_check_interval: Duration,

    /// Enable diagnostic integration
    pub diagnostics_integration: bool,

    /// Maximum memory pressure before triggering aggressive cleanup
    pub max_pressure_threshold: f32,
}

impl Default for PoolOptimizationConfig {
    fn default() -> Self {
        Self {
            auto_defrag_enabled: true,
            auto_defrag_threshold: 0.25,
            defrag_min_interval: Duration::from_secs(30),
            health_monitoring_enabled: true,
            health_check_interval: Duration::from_secs(10),
            diagnostics_integration: true,
            max_pressure_threshold: 0.90,
        }
    }
}

/// Enhanced memory pool with diagnostic integration
#[cfg(feature = "gpu")]
pub struct DiagnosticMemoryPool {
    pool: Arc<MemoryPool>,
    config: PoolOptimizationConfig,
    last_health_check: Arc<std::sync::Mutex<Instant>>,
    last_diagnostic_run: Arc<std::sync::Mutex<Instant>>,
    health_history: Arc<std::sync::Mutex<Vec<PoolHealthMetrics>>>,
}

#[cfg(feature = "gpu")]
impl DiagnosticMemoryPool {
    /// Create a new diagnostic memory pool
    pub fn new(device_id: usize, pool_size: usize) -> crate::Result<Self> {
        let pool = Arc::new(MemoryPool::new(device_id, pool_size)?);

        Ok(Self {
            pool,
            config: PoolOptimizationConfig::default(),
            last_health_check: Arc::new(std::sync::Mutex::new(Instant::now())),
            last_diagnostic_run: Arc::new(std::sync::Mutex::new(Instant::now())),
            health_history: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }

    /// Create with custom configuration
    pub fn with_config(
        device_id: usize,
        pool_size: usize,
        config: PoolOptimizationConfig,
    ) -> crate::Result<Self> {
        let pool = Arc::new(MemoryPool::new(device_id, pool_size)?);

        Ok(Self {
            pool,
            config,
            last_health_check: Arc::new(std::sync::Mutex::new(Instant::now())),
            last_diagnostic_run: Arc::new(std::sync::Mutex::new(Instant::now())),
            health_history: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }

    /// Get the underlying memory pool
    pub fn pool(&self) -> &Arc<MemoryPool> {
        &self.pool
    }

    /// Check pool health and return metrics
    pub fn check_health(&self) -> PoolHealthMetrics {
        let stats = self.pool.stats();
        let metrics = PoolHealthMetrics::from_stats(&stats);

        // Store in history
        if let Ok(mut history) = self.health_history.lock() {
            history.push(metrics.clone());
            // Keep only last 100 health checks
            if history.len() > 100 {
                history.remove(0);
            }
        }

        // Update last check time
        if let Ok(mut last_check) = self.last_health_check.lock() {
            *last_check = Instant::now();
        }

        metrics
    }

    /// Run automatic optimization based on health metrics
    pub fn auto_optimize(&self) -> OptimizationResult {
        let metrics = self.check_health();
        let mut result = OptimizationResult::default();

        // Check if defragmentation is needed
        if self.config.auto_defrag_enabled && metrics.defragmentation_needed {
            if let Ok(last_defrag) = self.last_health_check.lock() {
                if last_defrag.elapsed() >= self.config.defrag_min_interval {
                    self.pool.defragment();
                    result.defragmentation_performed = true;
                    result.actions.push("Performed defragmentation".to_string());
                }
            }
        }

        // Check memory pressure
        if metrics.pressure_score > self.config.max_pressure_threshold {
            if let Ok(freed) = self.pool.aggressive_cleanup(1024) {
                result.bytes_freed = freed;
                result
                    .actions
                    .push(format!("Aggressive cleanup freed {} bytes", freed));
            }
        }

        result.health_status = metrics.status;
        result
    }

    /// Integrate with global GPU diagnostics system
    pub fn run_integrated_diagnostics(&self) -> IntegratedDiagnosticReport {
        let pool_stats = self.pool.stats();
        let pool_health = PoolHealthMetrics::from_stats(&pool_stats);

        // Get GPU diagnostics if available
        let gpu_diagnostics = if self.config.diagnostics_integration {
            #[cfg(feature = "gpu")]
            {
                Some(crate::gpu::memory_diagnostics::GLOBAL_GPU_DIAGNOSTICS.run_diagnostics())
            }
            #[cfg(not(feature = "gpu"))]
            {
                None
            }
        } else {
            None
        };

        IntegratedDiagnosticReport {
            pool_stats,
            pool_health,
            gpu_diagnostics,
            timestamp: Instant::now(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PoolOptimizationConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PoolOptimizationConfig) {
        self.config = config;
    }

    /// Get health history
    pub fn health_history(&self) -> Vec<PoolHealthMetrics> {
        self.health_history.lock().unwrap().clone()
    }
}

/// Result of optimization operations
#[derive(Debug, Clone, Default)]
pub struct OptimizationResult {
    pub health_status: PoolHealthStatus,
    pub defragmentation_performed: bool,
    pub bytes_freed: usize,
    pub actions: Vec<String>,
}

/// Integrated diagnostic report combining pool and GPU diagnostics
#[derive(Debug, Clone)]
pub struct IntegratedDiagnosticReport {
    pub pool_stats: MemoryPoolStats,
    pub pool_health: PoolHealthMetrics,
    #[cfg(feature = "gpu")]
    pub gpu_diagnostics: Option<DiagnosticReport>,
    #[cfg(not(feature = "gpu"))]
    pub gpu_diagnostics: Option<()>,
    pub timestamp: Instant,
}

impl IntegratedDiagnosticReport {
    /// Print comprehensive diagnostic report
    pub fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════╗");
        println!("║   Integrated Memory Diagnostic Report               ║");
        println!("╚══════════════════════════════════════════════════════╝");

        println!("\n📊 Memory Pool Statistics:");
        println!(
            "  • Total Allocated:    {} bytes",
            self.pool_stats.total_allocated
        );
        println!(
            "  • Total Free:         {} bytes",
            self.pool_stats.total_free
        );
        println!(
            "  • Blocks Allocated:   {}",
            self.pool_stats.blocks_allocated
        );
        println!("  • Blocks Free:        {}", self.pool_stats.blocks_free);
        println!(
            "  • Peak Allocated:     {} bytes",
            self.pool_stats.peak_allocated
        );
        println!(
            "  • Allocations:        {}",
            self.pool_stats.allocation_count
        );
        println!(
            "  • Deallocations:      {}",
            self.pool_stats.deallocation_count
        );
        println!(
            "  • Defragmentations:   {}",
            self.pool_stats.defragmentation_count
        );

        self.pool_health.print();

        #[cfg(feature = "gpu")]
        if let Some(ref gpu_diag) = self.gpu_diagnostics {
            println!("\n🖥️  GPU Memory Diagnostics:");
            gpu_diag.print();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_health_metrics_creation() {
        let stats = MemoryPoolStats {
            total_allocated: 1024 * 1024, // 1 MB
            total_free: 1024 * 1024,      // 1 MB
            blocks_allocated: 10,
            blocks_free: 5,
            fragmentation_ratio: 0.35, // Above 0.3 threshold for Warning
            peak_allocated: 1024 * 1024,
            allocation_count: 100,
            deallocation_count: 90,
            defragmentation_count: 2,
            largest_free_block: 512 * 1024,
            average_block_size: 100.0 * 1024.0,
            memory_pressure: 0.65, // Above 0.6 threshold for Warning
        };

        let metrics = PoolHealthMetrics::from_stats(&stats);

        assert_eq!(metrics.status, PoolHealthStatus::Warning);
        assert!((metrics.fragmentation_score - 0.35).abs() < 0.01);
        assert!((metrics.pressure_score - 0.65).abs() < 0.01);
        assert!(metrics.efficiency_score > 0.6);
    }

    #[test]
    fn test_health_status_determination() {
        // Test healthy status
        let healthy_stats = MemoryPoolStats {
            total_allocated: 100,
            total_free: 900,
            blocks_allocated: 1,
            blocks_free: 1,
            fragmentation_ratio: 0.1,
            peak_allocated: 150,
            allocation_count: 10,
            deallocation_count: 9,
            defragmentation_count: 0,
            largest_free_block: 900,
            average_block_size: 100.0,
            memory_pressure: 0.1,
        };
        let metrics = PoolHealthMetrics::from_stats(&healthy_stats);
        assert_eq!(metrics.status, PoolHealthStatus::Healthy);

        // Test critical status
        let critical_stats = MemoryPoolStats {
            total_allocated: 960,
            total_free: 40,
            blocks_allocated: 20,
            blocks_free: 50,
            fragmentation_ratio: 0.8,
            peak_allocated: 960,
            allocation_count: 1000,
            deallocation_count: 980,
            defragmentation_count: 10,
            largest_free_block: 10,
            average_block_size: 20.0,
            memory_pressure: 0.96,
        };
        let metrics = PoolHealthMetrics::from_stats(&critical_stats);
        assert_eq!(metrics.status, PoolHealthStatus::Critical);
    }

    #[test]
    fn test_optimization_config_default() {
        let config = PoolOptimizationConfig::default();

        assert!(config.auto_defrag_enabled);
        assert_eq!(config.auto_defrag_threshold, 0.25);
        assert!(config.health_monitoring_enabled);
        assert!(config.diagnostics_integration);
    }
}
