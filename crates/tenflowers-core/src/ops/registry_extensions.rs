/// Registry Extensions for GPU Optimization and Advanced Dispatch
///
/// This module provides GPU-specific optimizations, kernel warming,
/// and intelligent fallback strategies for the operation registry.
use super::registry::{Kernel, OpRegistry, OpVersion, OP_REGISTRY};
use crate::{DType, Device, Result, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Device capability information
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device type
    pub device: Device,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability (for GPUs)
    pub compute_capability: Option<(u32, u32)>,
    /// Maximum workgroup size
    pub max_workgroup_size: Option<usize>,
    /// Supports half precision
    pub supports_fp16: bool,
    /// Supports bfloat16
    pub supports_bf16: bool,
    /// Supports tensor cores
    pub supports_tensor_cores: bool,
}

impl DeviceCapabilities {
    /// Get capabilities for a device
    pub fn for_device(device: Device) -> Self {
        match device {
            Device::Cpu => Self {
                device,
                available_memory: 0, // Query system memory
                compute_capability: None,
                max_workgroup_size: None,
                supports_fp16: true,
                supports_bf16: true,
                supports_tensor_cores: false,
            },
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => Self {
                device,
                available_memory: 0,              // Query GPU memory
                compute_capability: Some((8, 0)), // Default to modern GPU
                max_workgroup_size: Some(1024),
                supports_fp16: true,
                supports_bf16: true,
                supports_tensor_cores: true,
            },
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => Self {
                device,
                available_memory: 0,
                compute_capability: Some((9, 0)),
                max_workgroup_size: Some(1024),
                supports_fp16: true,
                supports_bf16: true,
                supports_tensor_cores: false,
            },
        }
    }

    /// Check if this device can handle the given dtype
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        match dtype {
            DType::Float16 => self.supports_fp16,
            DType::BFloat16 => self.supports_bf16,
            _ => true,
        }
    }
}

/// Kernel selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSelectionStrategy {
    /// Always use fastest available kernel
    Performance,
    /// Prefer memory-efficient kernels
    MemoryEfficient,
    /// Balance performance and memory
    Balanced,
    /// Use most compatible kernels (prefer CPU fallback)
    Compatible,
}

/// Kernel execution statistics
#[derive(Debug, Clone, Default)]
pub struct KernelStats {
    /// Number of executions
    pub execution_count: u64,
    /// Total execution time
    pub total_time: Duration,
    /// Average execution time
    pub avg_time: Duration,
    /// Last execution time
    pub last_execution: Option<Instant>,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
}

impl KernelStats {
    /// Record a successful execution
    pub fn record_success(&mut self, duration: Duration) {
        self.execution_count += 1;
        self.success_count += 1;
        self.total_time += duration;
        self.avg_time = self.total_time / self.execution_count as u32;
        self.last_execution = Some(Instant::now());
    }

    /// Record a failed execution
    pub fn record_failure(&mut self) {
        self.execution_count += 1;
        self.failure_count += 1;
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.execution_count as f64
        }
    }
}

/// Enhanced registry with GPU optimizations
pub struct EnhancedRegistry {
    /// Base registry
    base: &'static OpRegistry,
    /// Device capabilities cache
    device_capabilities: Mutex<HashMap<Device, DeviceCapabilities>>,
    /// Kernel execution statistics
    kernel_stats: Mutex<HashMap<String, KernelStats>>,
    /// Kernel selection strategy
    strategy: Mutex<KernelSelectionStrategy>,
    /// Pre-warmed kernels
    warmed_kernels: Mutex<HashMap<String, Arc<dyn Kernel>>>,
}

impl EnhancedRegistry {
    /// Create a new enhanced registry
    pub fn new() -> Self {
        Self {
            base: &OP_REGISTRY,
            device_capabilities: Mutex::new(HashMap::new()),
            kernel_stats: Mutex::new(HashMap::new()),
            strategy: Mutex::new(KernelSelectionStrategy::Balanced),
            warmed_kernels: Mutex::new(HashMap::new()),
        }
    }

    /// Get device capabilities
    pub fn get_device_capabilities(&self, device: Device) -> DeviceCapabilities {
        let mut caps = self.device_capabilities.lock().unwrap();
        caps.entry(device)
            .or_insert_with(|| DeviceCapabilities::for_device(device))
            .clone()
    }

    /// Set kernel selection strategy
    pub fn set_strategy(&self, strategy: KernelSelectionStrategy) {
        *self.strategy.lock().unwrap() = strategy;
    }

    /// Get kernel with intelligent device selection and fallback
    pub fn get_kernel_smart(
        &self,
        op_name: &str,
        preferred_device: Device,
        dtype: DType,
    ) -> Result<Arc<dyn Kernel>> {
        // Try preferred device first
        if let Some(kernel) = self.base.get_kernel(op_name, preferred_device, dtype) {
            return Ok(kernel);
        }

        // Check if preferred device supports this dtype
        let caps = self.get_device_capabilities(preferred_device);
        if !caps.supports_dtype(dtype) {
            return Err(TensorError::unsupported_device(
                op_name,
                &format!("{:?}", preferred_device),
                true,
            ));
        }

        // Try fallback to CPU
        if preferred_device != Device::Cpu {
            if let Some(kernel) = self.base.get_kernel(op_name, Device::Cpu, dtype) {
                // Log fallback
                return Ok(kernel);
            }
        }

        Err(TensorError::not_implemented_simple(format!(
            "No kernel available for operation '{}' on {:?} with {:?}",
            op_name, preferred_device, dtype
        )))
    }

    /// Warm up frequently used kernels
    pub fn warm_kernels(&self, ops: &[(String, Device, DType)]) {
        let mut warmed = self.warmed_kernels.lock().unwrap();

        for (op_name, device, dtype) in ops {
            let cache_key = format!("{}_{}_{:?}_{:?}", op_name, "warmed", device, dtype);
            if let Some(kernel) = self.base.get_kernel(op_name, *device, *dtype) {
                warmed.insert(cache_key, kernel);
            }
        }
    }

    /// Get warmed kernel (ultra-fast path)
    pub fn get_warmed_kernel(
        &self,
        op_name: &str,
        device: Device,
        dtype: DType,
    ) -> Option<Arc<dyn Kernel>> {
        let cache_key = format!("{}_{}_{:?}_{:?}", op_name, "warmed", device, dtype);
        let warmed = self.warmed_kernels.lock().unwrap();
        warmed.get(&cache_key).cloned()
    }

    /// Record kernel execution
    pub fn record_execution(
        &self,
        op_name: &str,
        device: Device,
        dtype: DType,
        duration: Duration,
        success: bool,
    ) {
        let key = format!("{}_{:?}_{:?}", op_name, device, dtype);
        let mut stats = self.kernel_stats.lock().unwrap();
        let entry = stats.entry(key).or_insert_with(KernelStats::default);

        if success {
            entry.record_success(duration);
        } else {
            entry.record_failure();
        }
    }

    /// Get kernel statistics
    pub fn get_kernel_stats(&self, op_name: &str, device: Device, dtype: DType) -> KernelStats {
        let key = format!("{}_{:?}_{:?}", op_name, device, dtype);
        let stats = self.kernel_stats.lock().unwrap();
        stats.get(&key).cloned().unwrap_or_default()
    }

    /// Get all kernel statistics
    pub fn get_all_stats(&self) -> HashMap<String, KernelStats> {
        self.kernel_stats.lock().unwrap().clone()
    }

    /// Find optimal device for an operation
    pub fn find_optimal_device(&self, op_name: &str, dtype: DType, data_size: usize) -> Device {
        let strategy = *self.strategy.lock().unwrap();

        match strategy {
            KernelSelectionStrategy::Performance => {
                // Prefer GPU for large operations
                #[cfg(feature = "gpu")]
                if data_size > 10_000
                    && self
                        .base
                        .get_kernel(op_name, Device::Gpu(0), dtype)
                        .is_some()
                {
                    return Device::Gpu(0);
                }
                Device::Cpu
            }
            KernelSelectionStrategy::MemoryEfficient => {
                // Always prefer CPU to save GPU memory
                Device::Cpu
            }
            KernelSelectionStrategy::Balanced => {
                // Use GPU for very large operations
                #[cfg(feature = "gpu")]
                if data_size > 100_000
                    && self
                        .base
                        .get_kernel(op_name, Device::Gpu(0), dtype)
                        .is_some()
                {
                    return Device::Gpu(0);
                }
                Device::Cpu
            }
            KernelSelectionStrategy::Compatible => {
                // Always use CPU for maximum compatibility
                Device::Cpu
            }
        }
    }

    /// Suggest optimizations based on statistics
    pub fn suggest_optimizations(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        let stats = self.kernel_stats.lock().unwrap();

        for (key, stat) in stats.iter() {
            // Suggest warming for frequently used kernels
            if stat.execution_count > 100 && !key.contains("warmed") {
                suggestions.push(format!(
                    "Consider warming kernel '{}' (executed {} times)",
                    key, stat.execution_count
                ));
            }

            // Suggest fallback for failing kernels
            if stat.failure_count > 10 && stat.success_rate() < 0.5 {
                suggestions.push(format!(
                    "Kernel '{}' has high failure rate ({:.1}%), consider using CPU fallback",
                    key,
                    (1.0 - stat.success_rate()) * 100.0
                ));
            }

            // Suggest GPU for slow CPU operations
            if key.contains("Cpu") && stat.avg_time > Duration::from_millis(100) {
                suggestions.push(format!(
                    "Kernel '{}' is slow on CPU (avg {:.2}ms), consider GPU acceleration",
                    key,
                    stat.avg_time.as_secs_f64() * 1000.0
                ));
            }
        }

        suggestions
    }

    /// Clear all statistics
    pub fn reset_statistics(&self) {
        self.kernel_stats.lock().unwrap().clear();
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let stats = self.kernel_stats.lock().unwrap();

        let total_executions: u64 = stats.values().map(|s| s.execution_count).sum();
        let total_successes: u64 = stats.values().map(|s| s.success_count).sum();
        let total_failures: u64 = stats.values().map(|s| s.failure_count).sum();

        let mut slowest_kernels: Vec<_> =
            stats.iter().map(|(k, s)| (k.clone(), s.avg_time)).collect();
        slowest_kernels.sort_by(|a, b| b.1.cmp(&a.1));
        slowest_kernels.truncate(10);

        let mut most_used: Vec<_> = stats
            .iter()
            .map(|(k, s)| (k.clone(), s.execution_count))
            .collect();
        most_used.sort_by(|a, b| b.1.cmp(&a.1));
        most_used.truncate(10);

        PerformanceReport {
            total_executions,
            total_successes,
            total_failures,
            overall_success_rate: if total_executions > 0 {
                total_successes as f64 / total_executions as f64
            } else {
                0.0
            },
            slowest_kernels,
            most_used_kernels: most_used,
            optimization_suggestions: self.suggest_optimizations(),
        }
    }
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total kernel executions
    pub total_executions: u64,
    /// Total successful executions
    pub total_successes: u64,
    /// Total failed executions
    pub total_failures: u64,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Slowest kernels (name, avg time)
    pub slowest_kernels: Vec<(String, Duration)>,
    /// Most used kernels (name, count)
    pub most_used_kernels: Vec<(String, u64)>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
}

impl PerformanceReport {
    /// Print a formatted report
    pub fn print(&self) {
        println!("=== Registry Performance Report ===");
        println!("\nOverall Statistics:");
        println!("  Total Executions:  {}", self.total_executions);
        println!("  Successes:         {}", self.total_successes);
        println!("  Failures:          {}", self.total_failures);
        println!(
            "  Success Rate:      {:.2}%",
            self.overall_success_rate * 100.0
        );

        println!("\nTop 10 Slowest Kernels:");
        for (i, (name, time)) in self.slowest_kernels.iter().enumerate() {
            println!(
                "  {}: {} ({:.2}ms avg)",
                i + 1,
                name,
                time.as_secs_f64() * 1000.0
            );
        }

        println!("\nTop 10 Most Used Kernels:");
        for (i, (name, count)) in self.most_used_kernels.iter().enumerate() {
            println!("  {}: {} ({} executions)", i + 1, name, count);
        }

        if !self.optimization_suggestions.is_empty() {
            println!("\n💡 Optimization Suggestions:");
            for suggestion in &self.optimization_suggestions {
                println!("  • {}", suggestion);
            }
        }

        println!("\n===================================");
    }
}

impl Default for EnhancedRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global enhanced registry instance
lazy_static::lazy_static! {
    pub static ref ENHANCED_REGISTRY: EnhancedRegistry = EnhancedRegistry::new();
}

/// Convenience function to get kernel with smart selection
pub fn get_kernel_smart(
    op_name: &str,
    preferred_device: Device,
    dtype: DType,
) -> Result<Arc<dyn Kernel>> {
    ENHANCED_REGISTRY.get_kernel_smart(op_name, preferred_device, dtype)
}

/// Convenience function to warm kernels
pub fn warm_kernels(ops: &[(String, Device, DType)]) {
    ENHANCED_REGISTRY.warm_kernels(ops);
}

/// Convenience function to get performance report
pub fn generate_performance_report() -> PerformanceReport {
    ENHANCED_REGISTRY.generate_performance_report()
}

/// Convenience function to print performance report
pub fn print_performance_report() {
    generate_performance_report().print();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_capabilities() {
        let cpu_caps = DeviceCapabilities::for_device(Device::Cpu);
        assert!(!cpu_caps.supports_tensor_cores);
        assert!(cpu_caps.supports_fp16);
    }

    #[test]
    fn test_kernel_stats() {
        let mut stats = KernelStats::default();
        stats.record_success(Duration::from_millis(10));
        stats.record_success(Duration::from_millis(20));
        stats.record_failure();

        assert_eq!(stats.execution_count, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
        assert!((stats.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_enhanced_registry() {
        let registry = EnhancedRegistry::new();
        registry.set_strategy(KernelSelectionStrategy::Performance);

        let optimal = registry.find_optimal_device("matmul", DType::Float32, 1_000_000);
        #[cfg(feature = "gpu")]
        assert!(matches!(optimal, Device::Cpu | Device::Gpu(_)));
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(optimal, Device::Cpu));
    }
}
