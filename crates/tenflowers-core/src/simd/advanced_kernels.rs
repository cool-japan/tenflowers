//! Advanced High-Performance Kernel Registry
//!
//! This module provides a comprehensive registry of specialized kernels optimized
//! for different hardware platforms, data sizes, and operation types.

use crate::{Result, TensorError};
use scirs2_core::profiling::Profiler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Advanced kernel registry for automatic kernel selection
#[allow(dead_code)]
pub struct AdvancedKernelRegistry {
    /// Registered kernels by operation type
    kernels: Arc<Mutex<HashMap<String, Vec<SpecializedKernel>>>>,
    /// Performance profiler
    profiler: Arc<Profiler>,
    /// Kernel selection strategy
    selection_strategy: KernelOptimizationStrategy,
    /// Runtime performance cache
    performance_cache: Arc<Mutex<HashMap<String, KernelPerformanceData>>>,
}

/// Specialized high-performance kernel
#[derive(Debug, Clone)]
pub struct SpecializedKernel {
    /// Kernel identifier
    pub id: String,
    /// Kernel name
    pub name: String,
    /// Target operation
    pub operation: String,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// Optimal data characteristics
    pub optimal_data_profile: DataProfile,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Kernel implementation
    pub implementation: KernelImplementation,
    /// Validation function
    pub validator: Option<ValidationFunction>,
}

/// Hardware requirements for kernel execution
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    /// Required CPU features
    pub required_cpu_features: Vec<String>,
    /// Minimum cache sizes
    pub min_cache_sizes: CacheSizeRequirements,
    /// Memory bandwidth requirements
    pub min_memory_bandwidth: f64,
    /// SIMD register requirements
    pub min_simd_registers: usize,
    /// Architecture preference
    pub preferred_architecture: Vec<String>,
}

/// Cache size requirements
#[derive(Debug, Clone)]
pub struct CacheSizeRequirements {
    pub min_l1_size: usize,
    pub min_l2_size: usize,
    pub min_l3_size: usize,
}

/// Data characteristics for optimal kernel performance
#[derive(Debug, Clone)]
pub struct DataProfile {
    /// Optimal data size range
    pub size_range: (usize, usize),
    /// Optimal data alignment
    pub alignment_requirement: usize,
    /// Data access pattern
    pub access_pattern: AccessPattern,
    /// Memory layout preference
    pub layout_preference: MemoryLayout,
    /// Sparsity tolerance
    pub sparsity_tolerance: f64,
}

/// Memory access pattern types
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    Sequential,
    Strided,
    Random,
    BlockedSequential,
    CacheOblivious,
}

/// Memory layout preferences
#[derive(Debug, Clone, Copy)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked,
    Tiled,
    Interleaved,
}

/// Kernel performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Expected throughput (ops/sec)
    pub expected_throughput: f64,
    /// Expected latency (seconds)
    pub expected_latency: f64,
    /// Memory efficiency (0-1)
    pub memory_efficiency: f64,
    /// Cache efficiency (0-1)
    pub cache_efficiency: f64,
    /// Energy efficiency (ops/joule)
    pub energy_efficiency: f64,
    /// Scalability factor
    pub scalability_factor: f64,
}

/// Kernel implementation variants
#[derive(Debug, Clone)]
pub enum KernelImplementation {
    /// Native Rust implementation
    Native(NativeKernelFn),
    /// Assembly-optimized implementation
    Assembly(AssemblyKernelFn),
    /// SIMD-vectorized implementation
    Vectorized(VectorizedKernelFn),
    /// GPU-accelerated implementation
    Gpu(GpuKernelFn),
    /// Hybrid CPU-GPU implementation
    Hybrid(HybridKernelFn),
}

/// Function type for native Rust kernels
pub type NativeKernelFn = fn(&[f32], &[f32], &mut [f32], &KernelParams) -> Result<()>;

/// Function type for assembly-optimized kernels
pub type AssemblyKernelFn =
    unsafe fn(*const f32, *const f32, *mut f32, &KernelParams) -> Result<()>;

/// Function type for vectorized kernels
pub type VectorizedKernelFn = fn(&[f32], &[f32], &mut [f32], &KernelParams) -> Result<()>;

/// Function type for GPU kernels
pub type GpuKernelFn = fn(&[f32], &[f32], &mut [f32], &KernelParams) -> Result<()>;

/// Function type for hybrid kernels
pub type HybridKernelFn = fn(&[f32], &[f32], &mut [f32], &KernelParams) -> Result<()>;

/// Kernel validation function
pub type ValidationFunction = fn(&[f32], &[f32], &[f32], &KernelParams) -> bool;

/// Kernel execution parameters
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Matrix/tensor dimensions
    pub dimensions: Vec<usize>,
    /// Stride information
    pub strides: Vec<usize>,
    /// Data type information
    pub data_type: String,
    /// Operation-specific parameters
    pub operation_params: HashMap<String, f64>,
    /// Performance hints
    pub performance_hints: Vec<String>,
}

/// Kernel optimization strategy
#[derive(Debug, Clone)]
pub enum KernelOptimizationStrategy {
    /// Maximize throughput
    MaxThroughput,
    /// Minimize latency
    MinLatency,
    /// Optimize for energy efficiency
    EnergyEfficient,
    /// Balance performance and energy
    Balanced,
    /// Adaptive based on workload
    Adaptive,
}

/// Runtime performance data for kernel selection
#[derive(Debug, Clone)]
pub struct KernelPerformanceData {
    /// Measured throughput
    pub measured_throughput: f64,
    /// Measured latency
    pub measured_latency: f64,
    /// Success rate
    pub success_rate: f64,
    /// Number of executions
    pub execution_count: u64,
    /// Last update timestamp
    pub last_updated: std::time::Instant,
}

impl AdvancedKernelRegistry {
    /// Create new advanced kernel registry
    pub fn new(strategy: KernelOptimizationStrategy) -> Self {
        let kernels = Arc::new(Mutex::new(HashMap::new()));
        let profiler = Arc::new(Profiler::new());
        let performance_cache = Arc::new(Mutex::new(HashMap::new()));

        let mut registry = Self {
            kernels,
            profiler,
            selection_strategy: strategy,
            performance_cache,
        };

        // Register default high-performance kernels
        registry
            .register_default_kernels()
            .expect("Failed to register default kernels");

        registry
    }

    /// Register a new specialized kernel
    pub fn register_kernel(&self, kernel: SpecializedKernel) -> Result<()> {
        let mut kernels = self.kernels.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock kernel registry".to_string())
        })?;

        let operation_kernels = kernels
            .entry(kernel.operation.clone())
            .or_insert_with(Vec::new);
        operation_kernels.push(kernel);

        // Sort kernels by expected performance
        operation_kernels.sort_by(|a, b| {
            b.performance_profile
                .expected_throughput
                .partial_cmp(&a.performance_profile.expected_throughput)
                .unwrap()
        });

        Ok(())
    }

    /// Select optimal kernel for given operation and data characteristics
    pub fn select_optimal_kernel(
        &self,
        operation: &str,
        data_size: usize,
        data_profile: &DataProfile,
    ) -> Result<SpecializedKernel> {
        let kernels = self.kernels.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock kernel registry".to_string())
        })?;

        let operation_kernels = kernels.get(operation).ok_or_else(|| {
            TensorError::compute_error_simple(format!(
                "No kernels registered for operation: {}",
                operation
            ))
        })?;

        // Score each kernel based on suitability
        let mut scored_kernels: Vec<(f64, &SpecializedKernel)> = operation_kernels
            .iter()
            .map(|kernel| (self.score_kernel(kernel, data_size, data_profile), kernel))
            .collect();

        // Sort by score (highest first)
        scored_kernels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        if let Some((score, kernel)) = scored_kernels.first() {
            if *score > 0.0 {
                return Ok((*kernel).clone());
            }
        }

        Err(TensorError::compute_error_simple(
            "No suitable kernel found".to_string(),
        ))
    }

    /// Score kernel suitability for given data characteristics
    fn score_kernel(
        &self,
        kernel: &SpecializedKernel,
        data_size: usize,
        data_profile: &DataProfile,
    ) -> f64 {
        let mut score = 0.0;

        // Size compatibility score
        if data_size >= kernel.optimal_data_profile.size_range.0
            && data_size <= kernel.optimal_data_profile.size_range.1
        {
            score += 0.3;
        }

        // Access pattern compatibility score
        if std::mem::discriminant(&kernel.optimal_data_profile.access_pattern)
            == std::mem::discriminant(&data_profile.access_pattern)
        {
            score += 0.2;
        }

        // Memory layout compatibility score
        if std::mem::discriminant(&kernel.optimal_data_profile.layout_preference)
            == std::mem::discriminant(&data_profile.layout_preference)
        {
            score += 0.2;
        }

        // Performance score based on strategy
        match self.selection_strategy {
            KernelOptimizationStrategy::MaxThroughput => {
                score += kernel.performance_profile.expected_throughput / 1e12 * 0.3;
            }
            KernelOptimizationStrategy::MinLatency => {
                score += (1.0 / kernel.performance_profile.expected_latency.max(1e-9)) / 1e9 * 0.3;
            }
            KernelOptimizationStrategy::EnergyEfficient => {
                score += kernel.performance_profile.energy_efficiency / 1e12 * 0.3;
            }
            KernelOptimizationStrategy::Balanced => {
                score += (kernel.performance_profile.expected_throughput / 1e12
                    + kernel.performance_profile.energy_efficiency / 1e12)
                    * 0.15;
            }
            KernelOptimizationStrategy::Adaptive => {
                // Use historical performance data if available
                score += self.get_adaptive_score(kernel) * 0.3;
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Get adaptive score based on historical performance
    fn get_adaptive_score(&self, kernel: &SpecializedKernel) -> f64 {
        if let Ok(cache) = self.performance_cache.lock() {
            if let Some(perf_data) = cache.get(&kernel.id) {
                return perf_data.measured_throughput / 1e12 * perf_data.success_rate;
            }
        }

        // Fallback to expected performance
        kernel.performance_profile.expected_throughput / 1e12
    }

    /// Execute kernel with performance monitoring
    pub fn execute_kernel(
        &self,
        kernel: &SpecializedKernel,
        input_a: &[f32],
        input_b: &[f32],
        output: &mut [f32],
        params: &KernelParams,
    ) -> Result<KernelExecutionResult> {
        let start_time = std::time::Instant::now();

        // Execute kernel based on implementation type
        let result = match &kernel.implementation {
            KernelImplementation::Native(kernel_fn) => kernel_fn(input_a, input_b, output, params),
            KernelImplementation::Vectorized(kernel_fn) => {
                kernel_fn(input_a, input_b, output, params)
            }
            _ => {
                // Fallback to native implementation for unsupported types
                Err(TensorError::compute_error_simple(
                    "Unsupported kernel implementation".to_string(),
                ))
            }
        };

        let execution_time = start_time.elapsed();

        // Update performance cache
        self.update_performance_cache(&kernel.id, &result, execution_time);

        // Validate result if validator is provided
        if let Some(validator) = &kernel.validator {
            let is_valid = validator(input_a, input_b, output, params);
            if !is_valid {
                return Err(TensorError::compute_error_simple(
                    "Kernel validation failed".to_string(),
                ));
            }
        }

        Ok(KernelExecutionResult {
            success: result.is_ok(),
            execution_time,
            throughput: self.calculate_throughput(params, execution_time),
            energy_estimate: self.estimate_energy_consumption(kernel, execution_time),
            cache_efficiency: self.estimate_cache_efficiency(kernel, params),
        })
    }

    /// Update performance cache with execution results
    fn update_performance_cache(
        &self,
        kernel_id: &str,
        result: &Result<()>,
        execution_time: std::time::Duration,
    ) {
        if let Ok(mut cache) = self.performance_cache.lock() {
            let entry = cache
                .entry(kernel_id.to_string())
                .or_insert(KernelPerformanceData {
                    measured_throughput: 0.0,
                    measured_latency: 0.0,
                    success_rate: 0.0,
                    execution_count: 0,
                    last_updated: std::time::Instant::now(),
                });

            entry.execution_count += 1;
            entry.measured_latency = execution_time.as_secs_f64();

            if result.is_ok() {
                entry.success_rate = (entry.success_rate * (entry.execution_count - 1) as f64
                    + 1.0)
                    / entry.execution_count as f64;
            } else {
                entry.success_rate = (entry.success_rate * (entry.execution_count - 1) as f64)
                    / entry.execution_count as f64;
            }

            entry.last_updated = std::time::Instant::now();
        }
    }

    /// Calculate throughput based on operation parameters
    fn calculate_throughput(
        &self,
        params: &KernelParams,
        execution_time: std::time::Duration,
    ) -> f64 {
        let total_ops = params.dimensions.iter().product::<usize>() as f64;
        total_ops / execution_time.as_secs_f64()
    }

    /// Estimate energy consumption for kernel execution
    fn estimate_energy_consumption(
        &self,
        kernel: &SpecializedKernel,
        execution_time: std::time::Duration,
    ) -> f64 {
        // Simple energy estimation based on performance profile
        let base_power = 50.0; // Watts
        let efficiency_multiplier = kernel.performance_profile.energy_efficiency / 1e12;
        base_power * execution_time.as_secs_f64() / efficiency_multiplier
    }

    /// Estimate cache efficiency for kernel execution
    fn estimate_cache_efficiency(&self, kernel: &SpecializedKernel, _params: &KernelParams) -> f64 {
        kernel.performance_profile.cache_efficiency
    }

    /// Register default high-performance kernels
    fn register_default_kernels(&mut self) -> Result<()> {
        // High-performance matrix multiplication kernel
        self.register_kernel(SpecializedKernel {
            id: "matmul_high_perf".to_string(),
            name: "High-Performance Matrix Multiplication".to_string(),
            operation: "matmul".to_string(),
            hardware_requirements: HardwareRequirements {
                required_cpu_features: vec!["avx2".to_string()],
                min_cache_sizes: CacheSizeRequirements {
                    min_l1_size: 32768,
                    min_l2_size: 262144,
                    min_l3_size: 8388608,
                },
                min_memory_bandwidth: 50e9,
                min_simd_registers: 16,
                preferred_architecture: vec!["x86_64".to_string()],
            },
            optimal_data_profile: DataProfile {
                size_range: (1024, usize::MAX),
                alignment_requirement: 64,
                access_pattern: AccessPattern::BlockedSequential,
                layout_preference: MemoryLayout::RowMajor,
                sparsity_tolerance: 0.1,
            },
            performance_profile: PerformanceProfile {
                expected_throughput: 2e12,
                expected_latency: 1e-6,
                memory_efficiency: 0.9,
                cache_efficiency: 0.85,
                energy_efficiency: 1e12,
                scalability_factor: 0.95,
            },
            implementation: KernelImplementation::Vectorized(high_perf_matmul),
            validator: Some(validate_matmul_result),
        })?;

        // Cache-friendly element-wise operations kernel
        self.register_kernel(SpecializedKernel {
            id: "elementwise_cache_friendly".to_string(),
            name: "Cache-Friendly Element-wise Operations".to_string(),
            operation: "elementwise".to_string(),
            hardware_requirements: HardwareRequirements {
                required_cpu_features: vec![],
                min_cache_sizes: CacheSizeRequirements {
                    min_l1_size: 16384,
                    min_l2_size: 131072,
                    min_l3_size: 4194304,
                },
                min_memory_bandwidth: 25e9,
                min_simd_registers: 8,
                preferred_architecture: vec!["x86_64".to_string(), "aarch64".to_string()],
            },
            optimal_data_profile: DataProfile {
                size_range: (64, usize::MAX),
                alignment_requirement: 32,
                access_pattern: AccessPattern::Sequential,
                layout_preference: MemoryLayout::RowMajor,
                sparsity_tolerance: 0.5,
            },
            performance_profile: PerformanceProfile {
                expected_throughput: 4e12,
                expected_latency: 5e-7,
                memory_efficiency: 0.95,
                cache_efficiency: 0.9,
                energy_efficiency: 2e12,
                scalability_factor: 0.98,
            },
            implementation: KernelImplementation::Vectorized(cache_friendly_elementwise),
            validator: Some(validate_elementwise_result),
        })?;

        Ok(())
    }

    /// Get comprehensive kernel registry statistics
    pub fn get_registry_statistics(&self) -> Result<KernelRegistryStatistics> {
        let kernels = self.kernels.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock kernel registry".to_string())
        })?;

        let cache = self.performance_cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock performance cache".to_string())
        })?;

        let total_kernels: usize = kernels.values().map(|v| v.len()).sum();
        let total_operations = kernels.len();
        let cached_performance_data = cache.len();

        Ok(KernelRegistryStatistics {
            total_kernels,
            total_operations,
            cached_performance_data,
            selection_strategy: self.selection_strategy.clone(),
            average_kernel_throughput: self.calculate_average_throughput(&kernels),
            cache_hit_rate: self.calculate_cache_hit_rate(&cache),
        })
    }

    fn calculate_average_throughput(
        &self,
        kernels: &HashMap<String, Vec<SpecializedKernel>>,
    ) -> f64 {
        let mut total_throughput = 0.0;
        let mut kernel_count = 0;

        for kernel_list in kernels.values() {
            for kernel in kernel_list {
                total_throughput += kernel.performance_profile.expected_throughput;
                kernel_count += 1;
            }
        }

        if kernel_count > 0 {
            total_throughput / kernel_count as f64
        } else {
            0.0
        }
    }

    fn calculate_cache_hit_rate(&self, cache: &HashMap<String, KernelPerformanceData>) -> f64 {
        let total_executions: u64 = cache.values().map(|data| data.execution_count).sum();
        let successful_executions: f64 = cache
            .values()
            .map(|data| data.execution_count as f64 * data.success_rate)
            .sum();

        if total_executions > 0 {
            successful_executions / total_executions as f64
        } else {
            0.0
        }
    }
}

/// Kernel execution result
#[derive(Debug, Clone)]
pub struct KernelExecutionResult {
    pub success: bool,
    pub execution_time: std::time::Duration,
    pub throughput: f64,
    pub energy_estimate: f64,
    pub cache_efficiency: f64,
}

/// Kernel registry statistics
#[derive(Debug, Clone)]
pub struct KernelRegistryStatistics {
    pub total_kernels: usize,
    pub total_operations: usize,
    pub cached_performance_data: usize,
    pub selection_strategy: KernelOptimizationStrategy,
    pub average_kernel_throughput: f64,
    pub cache_hit_rate: f64,
}

// High-performance kernel implementations

/// High-performance matrix multiplication kernel
fn high_perf_matmul(a: &[f32], b: &[f32], c: &mut [f32], params: &KernelParams) -> Result<()> {
    let (m, n, k) = if params.dimensions.len() >= 3 {
        (
            params.dimensions[0],
            params.dimensions[1],
            params.dimensions[2],
        )
    } else {
        return Err(TensorError::compute_error_simple(
            "Invalid dimensions for matmul".to_string(),
        ));
    };

    // Simple blocked implementation
    const BLOCK_SIZE: usize = 64;

    for i in (0..m).step_by(BLOCK_SIZE) {
        for j in (0..n).step_by(BLOCK_SIZE) {
            for l in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (i + BLOCK_SIZE).min(m);
                let j_end = (j + BLOCK_SIZE).min(n);
                let l_end = (l + BLOCK_SIZE).min(k);

                for ii in i..i_end {
                    for jj in j..j_end {
                        let mut sum = 0.0;
                        for ll in l..l_end {
                            sum += a[ii * k + ll] * b[ll * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Cache-friendly element-wise operations kernel
fn cache_friendly_elementwise(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    params: &KernelParams,
) -> Result<()> {
    let operation = params.operation_params.get("operation").unwrap_or(&0.0) as &f64;

    match *operation as i32 {
        0 => {
            // Add
            for i in 0..a.len() {
                c[i] = a[i] + b[i];
            }
        }
        1 => {
            // Multiply
            for i in 0..a.len() {
                c[i] = a[i] * b[i];
            }
        }
        _ => {
            return Err(TensorError::compute_error_simple(
                "Unsupported element-wise operation".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate matrix multiplication result
fn validate_matmul_result(a: &[f32], b: &[f32], c: &[f32], _params: &KernelParams) -> bool {
    // Simple validation: check that result is not all zeros (for non-zero inputs)
    let has_nonzero_input = a.iter().any(|&x| x != 0.0) && b.iter().any(|&x| x != 0.0);
    let has_nonzero_output = c.iter().any(|&x| x != 0.0);

    !has_nonzero_input || has_nonzero_output
}

/// Validate element-wise operation result
fn validate_elementwise_result(a: &[f32], b: &[f32], c: &[f32], _params: &KernelParams) -> bool {
    // Simple validation: check dimensions match
    a.len() == b.len() && b.len() == c.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_registry_creation() {
        let registry = AdvancedKernelRegistry::new(KernelOptimizationStrategy::MaxThroughput);
        let stats = registry.get_registry_statistics().unwrap();

        assert!(stats.total_kernels > 0);
        assert!(stats.total_operations > 0);
    }

    #[test]
    fn test_kernel_selection() {
        let registry = AdvancedKernelRegistry::new(KernelOptimizationStrategy::MaxThroughput);

        let data_profile = DataProfile {
            size_range: (1024, usize::MAX),
            alignment_requirement: 64,
            access_pattern: AccessPattern::Sequential,
            layout_preference: MemoryLayout::RowMajor,
            sparsity_tolerance: 0.1,
        };

        let kernel = registry.select_optimal_kernel("matmul", 2048, &data_profile);
        assert!(kernel.is_ok());
    }

    #[test]
    fn test_kernel_execution() {
        let registry = AdvancedKernelRegistry::new(KernelOptimizationStrategy::MaxThroughput);

        let data_profile = DataProfile {
            size_range: (64, usize::MAX),
            alignment_requirement: 32,
            access_pattern: AccessPattern::Sequential,
            layout_preference: MemoryLayout::RowMajor,
            sparsity_tolerance: 0.5,
        };

        let kernel = registry
            .select_optimal_kernel("matmul", 512, &data_profile)
            .unwrap();

        let a = vec![1.0; 64];
        let b = vec![2.0; 64];
        let mut c = vec![0.0; 64];

        let params = KernelParams {
            dimensions: vec![8, 8, 8],
            strides: vec![8, 8, 8],
            data_type: "f32".to_string(),
            operation_params: HashMap::new(),
            performance_hints: vec![],
        };

        let result = registry.execute_kernel(&kernel, &a, &b, &mut c, &params);
        assert!(result.is_ok());

        let execution_result = result.unwrap();
        assert!(execution_result.success);
        assert!(execution_result.throughput > 0.0);
    }

    #[test]
    fn test_performance_cache_update() {
        let registry = AdvancedKernelRegistry::new(KernelOptimizationStrategy::Adaptive);

        // Execute a kernel multiple times to populate cache
        let data_profile = DataProfile {
            size_range: (64, usize::MAX),
            alignment_requirement: 32,
            access_pattern: AccessPattern::Sequential,
            layout_preference: MemoryLayout::RowMajor,
            sparsity_tolerance: 0.5,
        };

        let kernel = registry
            .select_optimal_kernel("elementwise", 256, &data_profile)
            .unwrap();

        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let mut c = vec![0.0; 16];

        let mut params = KernelParams {
            dimensions: vec![16],
            strides: vec![1],
            data_type: "f32".to_string(),
            operation_params: HashMap::new(),
            performance_hints: vec![],
        };
        params.operation_params.insert("operation".to_string(), 0.0); // Add operation

        for _ in 0..5 {
            let _ = registry.execute_kernel(&kernel, &a, &b, &mut c, &params);
        }

        let stats = registry.get_registry_statistics().unwrap();
        assert!(stats.cached_performance_data > 0);
    }
}
