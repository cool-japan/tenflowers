/// Unified Dispatch System for TenfloweRS
///
/// This module provides a unified dispatch mechanism that automatically selects
/// and executes the best available kernel based on device, dtype, and backend availability.
/// It handles conditional compilation, fallback strategies, and performance optimization.
use super::registry::{Kernel, OpRegistry, OP_REGISTRY};
use super::registry_extensions::{EnhancedRegistry, KernelSelectionStrategy};
use crate::{DType, Device, Result, Tensor, TensorError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Backend types supported by TenfloweRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Backend {
    /// CPU execution
    Cpu,
    /// WebGPU (cross-platform GPU)
    #[cfg(feature = "gpu")]
    WebGpu,
    /// NVIDIA CUDA
    #[cfg(feature = "cuda")]
    Cuda,
    /// AMD ROCm
    #[cfg(feature = "rocm")]
    Rocm,
    /// Apple Metal
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
    /// OpenCL
    #[cfg(feature = "opencl")]
    OpenCl,
}

impl Backend {
    /// Get all available backends (based on compiled features)
    pub fn available() -> Vec<Backend> {
        let mut backends = vec![Backend::Cpu];

        #[cfg(feature = "gpu")]
        backends.push(Backend::WebGpu);

        #[cfg(feature = "cuda")]
        backends.push(Backend::Cuda);

        #[cfg(feature = "rocm")]
        backends.push(Backend::Rocm);

        #[cfg(all(feature = "metal", target_os = "macos"))]
        backends.push(Backend::Metal);

        #[cfg(feature = "opencl")]
        backends.push(Backend::OpenCl);

        backends
    }

    /// Check if this backend is available (compiled in)
    pub fn is_available(&self) -> bool {
        Self::available().contains(self)
    }

    /// Get backend priority (higher is better)
    pub fn priority(&self) -> u8 {
        match self {
            Backend::Cpu => 0,
            #[cfg(feature = "gpu")]
            Backend::WebGpu => 10,
            #[cfg(feature = "cuda")]
            Backend::Cuda => 20,
            #[cfg(feature = "rocm")]
            Backend::Rocm => 20,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => 15,
            #[cfg(feature = "opencl")]
            Backend::OpenCl => 5,
        }
    }

    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            #[cfg(feature = "gpu")]
            Backend::WebGpu => "WebGPU",
            #[cfg(feature = "cuda")]
            Backend::Cuda => "CUDA",
            #[cfg(feature = "rocm")]
            Backend::Rocm => "ROCm",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => "Metal",
            #[cfg(feature = "opencl")]
            Backend::OpenCl => "OpenCL",
        }
    }

    /// Convert Device to Backend
    pub fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Backend::Cpu,
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => Backend::WebGpu,
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => Backend::Rocm,
        }
    }
}

/// Dispatch context for kernel execution
#[derive(Debug, Clone)]
pub struct DispatchContext {
    /// Preferred backend
    pub preferred_backend: Backend,
    /// Fallback backends (in priority order)
    pub fallback_backends: Vec<Backend>,
    /// Data type
    pub dtype: DType,
    /// Enable automatic fallback
    pub auto_fallback: bool,
    /// Enable performance profiling
    pub profile: bool,
}

impl DispatchContext {
    /// Create a new dispatch context with defaults
    pub fn new(device: &Device, dtype: DType) -> Self {
        let preferred_backend = Backend::from_device(device);
        let mut fallback_backends: Vec<_> = Backend::available()
            .into_iter()
            .filter(|b| *b != preferred_backend)
            .collect();

        // Sort by priority (highest first)
        fallback_backends.sort_by_key(|b| std::cmp::Reverse(b.priority()));

        // Always add CPU as final fallback
        if !fallback_backends.contains(&Backend::Cpu) {
            fallback_backends.push(Backend::Cpu);
        }

        Self {
            preferred_backend,
            fallback_backends,
            dtype,
            auto_fallback: true,
            profile: false,
        }
    }

    /// Create context with no fallback
    pub fn strict(device: &Device, dtype: DType) -> Self {
        let mut ctx = Self::new(device, dtype);
        ctx.auto_fallback = false;
        ctx.fallback_backends.clear();
        ctx
    }

    /// Enable performance profiling
    pub fn with_profiling(mut self) -> Self {
        self.profile = true;
        self
    }
}

/// Unified dispatcher for operations
pub struct UnifiedDispatcher {
    /// Enhanced registry
    registry: EnhancedRegistry,
    /// Execution statistics
    stats: std::sync::Mutex<HashMap<String, DispatchStats>>,
}

/// Dispatch execution statistics
#[derive(Debug, Clone, Default)]
pub struct DispatchStats {
    /// Total dispatches
    total_dispatches: u64,
    /// Successful primary backend dispatches
    primary_successes: u64,
    /// Fallback dispatches
    fallback_dispatches: u64,
    /// Failed dispatches
    failures: u64,
}

impl UnifiedDispatcher {
    /// Create a new unified dispatcher
    pub fn new() -> Self {
        Self {
            registry: EnhancedRegistry::new(),
            stats: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Get the global dispatcher instance
    pub fn global() -> &'static Self {
        use once_cell::sync::Lazy;
        static DISPATCHER: Lazy<UnifiedDispatcher> = Lazy::new(UnifiedDispatcher::new);
        &DISPATCHER
    }

    /// Dispatch an operation to the appropriate kernel
    pub fn dispatch(
        &self,
        op_name: &str,
        inputs: &[&dyn Any],
        attrs: &HashMap<String, super::registry::AttrValue>,
        context: &DispatchContext,
    ) -> Result<Vec<Box<dyn Any>>> {
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            let op_stats = stats.entry(op_name.to_string()).or_default();
            op_stats.total_dispatches += 1;
        }

        // Try preferred backend first
        let device = self.backend_to_device(&context.preferred_backend);
        if let Some(kernel) = OP_REGISTRY.get_kernel(op_name, device, context.dtype) {
            match kernel.compute(inputs, attrs) {
                Ok(result) => {
                    self.record_success(op_name, true);
                    return Ok(result);
                }
                Err(e) if !context.auto_fallback => {
                    self.record_failure(op_name);
                    return Err(e);
                }
                Err(_) => {
                    // Continue to fallback
                }
            }
        }

        // Try fallback backends
        if context.auto_fallback {
            for backend in &context.fallback_backends {
                if !backend.is_available() {
                    continue;
                }

                let device = self.backend_to_device(backend);
                if let Some(kernel) = OP_REGISTRY.get_kernel(op_name, device, context.dtype) {
                    match kernel.compute(inputs, attrs) {
                        Ok(result) => {
                            self.record_success(op_name, false);
                            return Ok(result);
                        }
                        Err(_) => continue,
                    }
                }
            }
        }

        // All backends failed
        self.record_failure(op_name);
        Err(TensorError::not_implemented_simple(format!(
            "No available kernel for operation '{}' with dtype {:?}",
            op_name, context.dtype
        )))
    }

    /// Helper: Convert backend to device
    fn backend_to_device(&self, backend: &Backend) -> Device {
        match backend {
            Backend::Cpu => Device::Cpu,
            #[cfg(feature = "gpu")]
            Backend::WebGpu => Device::Gpu(0),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Device::Gpu(0), // Map to GPU for now
            #[cfg(feature = "rocm")]
            Backend::Rocm => Device::Gpu(0), // Map to GPU for now
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => Device::Gpu(0), // Map to GPU for now
            #[cfg(feature = "opencl")]
            Backend::OpenCl => Device::Gpu(0), // Map to GPU for now
        }
    }

    /// Record a successful dispatch
    fn record_success(&self, op_name: &str, primary: bool) {
        let mut stats = self.stats.lock().unwrap();
        let op_stats = stats.entry(op_name.to_string()).or_default();
        if primary {
            op_stats.primary_successes += 1;
        } else {
            op_stats.fallback_dispatches += 1;
        }
    }

    /// Record a failed dispatch
    fn record_failure(&self, op_name: &str) {
        let mut stats = self.stats.lock().unwrap();
        let op_stats = stats.entry(op_name.to_string()).or_default();
        op_stats.failures += 1;
    }

    /// Get dispatch statistics
    pub fn get_stats(&self, op_name: &str) -> Option<DispatchStats> {
        let stats = self.stats.lock().unwrap();
        stats.get(op_name).cloned()
    }

    /// Print dispatch statistics report
    pub fn print_stats(&self) {
        let stats = self.stats.lock().unwrap();
        println!("=== Unified Dispatch Statistics ===");
        for (op_name, op_stats) in stats.iter() {
            println!("\nOperation: {}", op_name);
            println!("  Total Dispatches:  {}", op_stats.total_dispatches);
            println!("  Primary Successes: {}", op_stats.primary_successes);
            println!("  Fallback Uses:     {}", op_stats.fallback_dispatches);
            println!("  Failures:          {}", op_stats.failures);
            if op_stats.total_dispatches > 0 {
                let success_rate = (op_stats.primary_successes + op_stats.fallback_dispatches)
                    as f64
                    / op_stats.total_dispatches as f64
                    * 100.0;
                println!("  Success Rate:      {:.2}%", success_rate);
            }
        }
        println!("===================================");
    }

    /// Set kernel selection strategy
    pub fn set_strategy(&self, strategy: KernelSelectionStrategy) {
        self.registry.set_strategy(strategy);
    }
}

impl Default for UnifiedDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to dispatch an operation
pub fn dispatch_op(
    op_name: &str,
    inputs: &[&dyn Any],
    attrs: &HashMap<String, super::registry::AttrValue>,
    device: &Device,
    dtype: DType,
) -> Result<Vec<Box<dyn Any>>> {
    let context = DispatchContext::new(device, dtype);
    UnifiedDispatcher::global().dispatch(op_name, inputs, attrs, &context)
}

/// Macro to simplify kernel registration with feature gates
#[macro_export]
macro_rules! register_kernel_with_backend {
    ($op_name:expr, $backend:ident, $dtype:expr, $kernel:expr) => {
        #[cfg(feature = stringify!($backend))]
        {
            let device = match stringify!($backend) {
                "cpu" => $crate::Device::Cpu,
                "gpu" => $crate::Device::Gpu(0),
                "cuda" => $crate::Device::Gpu(0),
                "rocm" => $crate::Device::Gpu(0),
                "metal" => $crate::Device::Gpu(0),
                _ => $crate::Device::Cpu,
            };

            $crate::ops::registry::OP_REGISTRY
                .register_kernel($op_name, device, $dtype, std::sync::Arc::new($kernel))
                .ok();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_availability() {
        let backends = Backend::available();
        assert!(!backends.is_empty());
        assert!(backends.contains(&Backend::Cpu));
    }

    #[test]
    fn test_backend_priority() {
        assert_eq!(Backend::Cpu.priority(), 0);
        #[cfg(feature = "gpu")]
        assert!(Backend::WebGpu.priority() > Backend::Cpu.priority());
    }

    #[test]
    fn test_dispatch_context_creation() {
        let device = Device::Cpu;
        let dtype = DType::Float32;
        let context = DispatchContext::new(&device, dtype);

        assert_eq!(context.preferred_backend, Backend::Cpu);
        assert!(context.auto_fallback);
        assert!(!context.fallback_backends.is_empty());
    }

    #[test]
    fn test_strict_context() {
        let device = Device::Cpu;
        let dtype = DType::Float32;
        let context = DispatchContext::strict(&device, dtype);

        assert_eq!(context.preferred_backend, Backend::Cpu);
        assert!(!context.auto_fallback);
        assert!(context.fallback_backends.is_empty());
    }

    #[test]
    fn test_dispatcher_creation() {
        let dispatcher = UnifiedDispatcher::new();
        let stats = dispatcher.get_stats("nonexistent_op");
        assert!(stats.is_none());
    }

    #[test]
    fn test_backend_names() {
        assert_eq!(Backend::Cpu.name(), "CPU");
        #[cfg(feature = "gpu")]
        assert_eq!(Backend::WebGpu.name(), "WebGPU");
    }
}
