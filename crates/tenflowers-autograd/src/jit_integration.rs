//! JIT Compiler Integration with Autograd System
//!
//! This module integrates the JIT compiler with the existing gradient computation system,
//! providing seamless runtime optimization of gradient kernels.

use crate::{
    jit_compiler::{
        CompiledKernel, DeviceFeatures, JitCompiler, KernelSignature, OptimizationLevel,
    },
    GradientTape, Result,
};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use tenflowers_core::Tensor;

/// Global JIT compiler instance
static JIT_COMPILER: OnceLock<Arc<RwLock<JitCompiler>>> = OnceLock::new();

/// Configuration for JIT compilation
#[derive(Debug, Clone)]
pub struct JitConfig {
    pub enabled: bool,
    pub optimization_level: OptimizationLevel,
    pub cache_size_limit: usize, // Maximum number of cached kernels
    pub auto_tune: bool,         // Automatically tune kernels based on actual performance
    pub debug_output: bool,      // Generate debug information
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_level: OptimizationLevel::Balanced,
            cache_size_limit: 1000,
            auto_tune: false,
            debug_output: false,
        }
    }
}

/// JIT-enhanced gradient computation context
#[derive(Debug)]
pub struct JitGradientContext {
    config: JitConfig,
    performance_tracker: Arc<RwLock<HashMap<String, KernelPerformanceStats>>>,
}

/// Performance statistics for kernel execution
#[derive(Debug, Clone)]
struct KernelPerformanceStats {
    execution_count: u64,
    total_execution_time_us: f64,
    average_execution_time_us: f64,
    last_execution_time_us: f64,
    compile_time_ms: f64,
}

impl JitGradientContext {
    /// Create a new JIT gradient context
    pub fn new(config: JitConfig) -> Self {
        Self {
            config,
            performance_tracker: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the global JIT compiler
    pub fn initialize_global_compiler(
        device_features: DeviceFeatures,
        config: &JitConfig,
    ) -> Result<()> {
        let compiler = JitCompiler::new(device_features, config.optimization_level);

        JIT_COMPILER
            .set(Arc::new(RwLock::new(compiler)))
            .map_err(|_| {
                tenflowers_core::TensorError::unsupported_operation_simple(
                    "JIT compiler already initialized".to_string(),
                )
            })?;

        Ok(())
    }

    /// Get reference to global JIT compiler
    fn get_compiler() -> Result<Arc<RwLock<JitCompiler>>> {
        JIT_COMPILER
            .get()
            .ok_or_else(|| {
                tenflowers_core::TensorError::unsupported_operation_simple(
                    "JIT compiler not initialized. Call initialize_global_compiler first."
                        .to_string(),
                )
            })
            .cloned()
    }

    /// Compile gradient kernel for given operation and tensor shapes
    pub async fn compile_gradient_kernel<T>(
        &self,
        operation: &str,
        inputs: &[&Tensor<T>],
        output_shape: &[usize],
    ) -> Result<CompiledKernel>
    where
        T: Clone + std::fmt::Debug + 'static + Default + scirs2_core::num_traits::Zero,
    {
        if !self.config.enabled {
            return Err(tenflowers_core::TensorError::unsupported_operation_simple(
                "JIT compilation is disabled".to_string(),
            ));
        }

        let signature = self.create_kernel_signature(operation, inputs, output_shape)?;

        let compiler = Self::get_compiler()?;
        let compiled_kernel = {
            let compiler_guard = compiler.read().map_err(|_| {
                tenflowers_core::TensorError::invalid_operation_simple(
                    "jit compiler lock poisoned".to_string(),
                )
            })?;
            compiler_guard.compile_gradient_kernel(signature)?
        };

        if self.config.debug_output {
            println!(
                "JIT: Compiled kernel for {} in {:.2}ms",
                operation, compiled_kernel.compile_time_ms
            );
            println!(
                "JIT: Estimated performance: {:.2}μs, {:.2e} FLOPS",
                compiled_kernel
                    .estimated_performance
                    .estimated_execution_time_us,
                compiled_kernel.estimated_performance.estimated_flops
            );
        }

        Ok(compiled_kernel)
    }

    /// Create kernel signature from operation and tensors
    fn create_kernel_signature<T>(
        &self,
        operation: &str,
        inputs: &[&Tensor<T>],
        output_shape: &[usize],
    ) -> Result<KernelSignature>
    where
        T: Clone + std::fmt::Debug + 'static + Default + scirs2_core::num_traits::Zero,
    {
        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|tensor| tensor.shape().dims().to_vec())
            .collect();

        let dtype = std::any::type_name::<T>().to_string();

        // Get device features (simplified for now)
        let device_features = self.detect_device_features(inputs)?;

        Ok(KernelSignature {
            operation: operation.to_string(),
            input_shapes,
            output_shape: output_shape.to_vec(),
            dtype,
            device_features,
        })
    }

    /// Detect device features from input tensors
    fn detect_device_features<T>(&self, _inputs: &[&Tensor<T>]) -> Result<DeviceFeatures>
    where
        T: Clone + std::fmt::Debug + 'static + Default + scirs2_core::num_traits::Zero,
    {
        // For now, return default features
        // In a real implementation, this would query the actual device capabilities
        Ok(DeviceFeatures::default())
    }

    /// Execute JIT-compiled gradient kernel
    pub async fn execute_jit_gradient<T>(
        &self,
        operation: &str,
        inputs: &[&Tensor<T>],
        grad_output: &Tensor<T>,
        compiled_kernel: &CompiledKernel,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone + std::fmt::Debug + 'static + Default + scirs2_core::num_traits::Zero,
    {
        let start_time = std::time::Instant::now();

        // Compiled-kernel execution is not yet wired up; delegate to the fallback path.
        // The fallback surfaces an honest error (no tape context) rather than fabricating
        // zero gradients, so this propagates that error to the caller.
        let gradients = self
            .fallback_gradient_computation(operation, inputs, grad_output)
            .await?;

        let execution_time = start_time.elapsed().as_micros() as f64;

        // Update performance statistics
        self.update_performance_stats(operation, execution_time, compiled_kernel.compile_time_ms);

        if self.config.debug_output {
            println!("JIT: Executed {operation} in {execution_time:.2}μs");
        }

        Ok(gradients)
    }

    /// Fallback gradient computation when no compiled JIT kernel is available.
    ///
    /// This entry point receives only the operation name, the forward input tensors,
    /// and the upstream gradient — it does **not** receive the [`GradientTape`] nor the
    /// recorded computation graph, so it cannot reconstruct the real backward pass on its
    /// own. Returning zero tensors here would silently fabricate gradients (corrupting any
    /// downstream optimizer), so instead we surface an honest, actionable error directing
    /// the caller to the tape-based gradient path.
    ///
    /// To compute gradients without a JIT kernel, call [`GradientTape::gradient`] directly,
    /// which performs reverse-mode differentiation over the recorded graph.
    async fn fallback_gradient_computation<T>(
        &self,
        operation: &str,
        _inputs: &[&Tensor<T>],
        _grad_output: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone + std::fmt::Debug + 'static + Default + scirs2_core::num_traits::Zero,
    {
        Err(tenflowers_core::TensorError::not_implemented_simple(
            format!(
                "JIT fallback for operation '{operation}' has no tape context and cannot \
             compute real gradients; call GradientTape::gradient directly to differentiate \
             over the recorded graph"
            ),
        ))
    }

    /// Update performance statistics for a kernel
    fn update_performance_stats(
        &self,
        operation: &str,
        execution_time_us: f64,
        compile_time_ms: f64,
    ) {
        let mut tracker = self
            .performance_tracker
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let stats = tracker
            .entry(operation.to_string())
            .or_insert(KernelPerformanceStats {
                execution_count: 0,
                total_execution_time_us: 0.0,
                average_execution_time_us: 0.0,
                last_execution_time_us: 0.0,
                compile_time_ms,
            });

        stats.execution_count += 1;
        stats.total_execution_time_us += execution_time_us;
        stats.average_execution_time_us =
            stats.total_execution_time_us / stats.execution_count as f64;
        stats.last_execution_time_us = execution_time_us;
    }

    /// Get performance report for all executed kernels
    pub fn get_performance_report(&self) -> String {
        let tracker = self
            .performance_tracker
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut report = String::new();

        report.push_str("# JIT Kernel Performance Report\n\n");
        report.push_str(
            "| Operation | Executions | Avg Time (μs) | Last Time (μs) | Compile Time (ms) |\n",
        );
        report.push_str(
            "|-----------|------------|---------------|----------------|-------------------|\n",
        );

        for (operation, stats) in tracker.iter() {
            report.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} |\n",
                operation,
                stats.execution_count,
                stats.average_execution_time_us,
                stats.last_execution_time_us,
                stats.compile_time_ms
            ));
        }

        report
    }

    /// Auto-tune kernels based on performance feedback
    pub async fn auto_tune_kernels(&self) -> Result<()> {
        if !self.config.auto_tune {
            return Ok(());
        }

        let tracker = self.performance_tracker.read().map_err(|_| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "jit compiler lock poisoned".to_string(),
            )
        })?;
        let compiler = Self::get_compiler()?;

        for (operation, stats) in tracker.iter() {
            // If a kernel is significantly slower than estimated, recompile with different optimizations
            let _compiler_guard = compiler.read().map_err(|_| {
                tenflowers_core::TensorError::invalid_operation_simple(
                    "jit compiler lock poisoned".to_string(),
                )
            })?;
            // This would implement the auto-tuning logic
            if self.config.debug_output {
                println!(
                    "JIT: Auto-tuning kernel {} (avg: {:.2}μs)",
                    operation, stats.average_execution_time_us
                );
            }
        }

        Ok(())
    }

    /// Clear performance statistics
    pub fn clear_performance_stats(&self) {
        let mut tracker = self
            .performance_tracker
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        tracker.clear();
    }
}

/// Extension trait for GradientTape to support JIT compilation
pub trait JitGradientTapeExt {
    /// Enable JIT compilation for this tape
    fn enable_jit(&mut self, config: JitConfig) -> Result<()>;

    /// Disable JIT compilation for this tape
    fn disable_jit(&mut self);

    /// Get JIT performance report
    fn jit_performance_report(&self) -> Option<String>;
}

impl JitGradientTapeExt for GradientTape {
    fn enable_jit(&mut self, _config: JitConfig) -> Result<()> {
        // The GradientTape does not yet carry a JIT compilation hook. Returning `Ok(())`
        // would silently claim JIT was enabled while changing nothing, so we return an
        // honest error instead. Use `JitGradientContext` directly to compile and execute
        // gradient kernels until tape-level integration lands.
        Err(tenflowers_core::TensorError::not_implemented_simple(
            "GradientTape does not yet support per-tape JIT enablement; \
             use JitGradientContext directly to compile/execute gradient kernels"
                .to_string(),
        ))
    }

    fn disable_jit(&mut self) {
        // No per-tape JIT state exists yet (see `enable_jit`), so there is nothing to
        // disable. This is intentionally a no-op rather than a fabricated state change.
    }

    fn jit_performance_report(&self) -> Option<String> {
        // The GradientTape does not track JIT performance; that data lives on
        // `JitGradientContext::get_performance_report`. Report `None` honestly rather
        // than synthesizing an empty-but-plausible report.
        None
    }
}

/// Utility functions for JIT compilation integration
pub mod utils {
    use super::*;

    /// Initialize JIT compilation with auto-detected device features
    pub async fn initialize_jit() -> Result<()> {
        let device_features = auto_detect_device_features().await?;
        let config = JitConfig::default();

        JitGradientContext::initialize_global_compiler(device_features, &config)
    }

    /// Auto-detect device features.
    ///
    /// No live device-capability query is wired in yet. Rather than fabricating
    /// optimistic "modern GPU" specifications (which would skew the JIT cost model toward
    /// non-existent hardware), this returns the conservative [`DeviceFeatures::default`]
    /// baseline. When a real device probe is added, replace this with the actual query.
    async fn auto_detect_device_features() -> Result<DeviceFeatures> {
        Ok(DeviceFeatures::default())
    }

    /// Create a JIT-enabled gradient context with reasonable defaults
    pub fn create_jit_context() -> JitGradientContext {
        JitGradientContext::new(JitConfig::default())
    }

    /// Create a debug JIT context with verbose output
    pub fn create_debug_jit_context() -> JitGradientContext {
        JitGradientContext::new(JitConfig {
            debug_output: true,
            optimization_level: OptimizationLevel::Debug,
            ..Default::default()
        })
    }

    /// Benchmark JIT vs non-JIT gradient computation.
    ///
    /// Returns `(jit_time_us, regular_time_us)` averaged over `iterations`. Both legs go
    /// through the gradient-execution path, which currently has no compiled kernel and no
    /// tape context, so this propagates the honest "not implemented" error from
    /// [`JitGradientContext::execute_jit_gradient`] rather than reporting fabricated
    /// timings for a no-op. It will yield real numbers once the execution path is wired up.
    pub async fn benchmark_jit_performance<T>(
        operation: &str,
        inputs: &[&Tensor<T>],
        grad_output: &Tensor<T>,
        iterations: usize,
    ) -> Result<(f64, f64)>
    // (jit_time_us, regular_time_us)
    where
        T: Clone + std::fmt::Debug + 'static + Default + scirs2_core::num_traits::Zero,
    {
        let jit_context = create_jit_context();

        // Benchmark JIT compilation + execution
        let jit_start = std::time::Instant::now();
        for _ in 0..iterations {
            let output_shape = grad_output.shape().dims();
            let compiled_kernel = jit_context
                .compile_gradient_kernel(operation, inputs, output_shape)
                .await?;
            let _gradients = jit_context
                .execute_jit_gradient(operation, inputs, grad_output, &compiled_kernel)
                .await?;
        }
        let jit_time = jit_start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark regular computation
        let regular_start = std::time::Instant::now();
        for _ in 0..iterations {
            let _gradients = jit_context
                .fallback_gradient_computation(operation, inputs, grad_output)
                .await?;
        }
        let regular_time = regular_start.elapsed().as_micros() as f64 / iterations as f64;

        Ok((jit_time, regular_time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_jit_context_creation() {
        let config = JitConfig::default();
        let context = JitGradientContext::new(config);

        let report = context.get_performance_report();
        assert!(report.contains("JIT Kernel Performance Report"));
    }

    #[test]
    fn test_jit_initialization() {
        // Test synchronous JIT configuration setup
        let config = JitConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_jit_config_default() {
        let config = JitConfig::default();
        assert!(config.enabled);
        assert_eq!(config.cache_size_limit, 1000);
        assert!(!config.debug_output);
    }

    #[test]
    fn test_device_feature_detection() {
        let device_features = DeviceFeatures::default();
        assert!(device_features.max_workgroup_size > 0);
        assert!(device_features.compute_units > 0);
        assert!(device_features.memory_bandwidth_gb_s > 0.0);
    }

    #[test]
    fn test_performance_stats_update() {
        let context = utils::create_jit_context();
        context.update_performance_stats("test_op", 100.0, 50.0);

        let report = context.get_performance_report();
        assert!(report.contains("test_op"));
        assert!(report.contains("100.00"));
    }
}
