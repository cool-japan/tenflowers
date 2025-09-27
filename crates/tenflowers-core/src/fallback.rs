//! Automatic fallback mechanisms for operation recovery
//!
//! This module provides utilities for automatic fallback when operations fail,
//! particularly for GPU-to-CPU fallback scenarios.

#[cfg(feature = "gpu")]
use crate::Device;
use crate::{Result, Tensor, TensorError};
use num_traits;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to enable/disable automatic fallback
static AUTO_FALLBACK_ENABLED: AtomicBool = AtomicBool::new(true);

/// Configuration for fallback behavior
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Enable automatic GPU-to-CPU fallback
    pub gpu_to_cpu: bool,
    /// Enable automatic precision reduction
    pub reduce_precision: bool,
    /// Enable memory cleanup and retry
    pub memory_cleanup: bool,
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Log fallback attempts
    pub log_fallbacks: bool,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            gpu_to_cpu: true,
            reduce_precision: false,
            memory_cleanup: true,
            max_retries: 3,
            log_fallbacks: true,
        }
    }
}

/// Global fallback configuration
#[allow(static_mut_refs)]
static mut GLOBAL_FALLBACK_CONFIG: Option<FallbackConfig> = None;
static FALLBACK_CONFIG_INIT: std::sync::Once = std::sync::Once::new();

/// Get the global fallback configuration
#[allow(static_mut_refs)]
pub fn get_fallback_config() -> FallbackConfig {
    unsafe {
        FALLBACK_CONFIG_INIT.call_once(|| {
            GLOBAL_FALLBACK_CONFIG = Some(FallbackConfig::default());
        });
        GLOBAL_FALLBACK_CONFIG.as_ref().unwrap().clone()
    }
}

/// Set the global fallback configuration
#[allow(static_mut_refs)]
pub fn set_fallback_config(config: FallbackConfig) {
    unsafe {
        GLOBAL_FALLBACK_CONFIG = Some(config);
    }
}

/// Enable or disable automatic fallback globally
pub fn set_auto_fallback_enabled(enabled: bool) {
    AUTO_FALLBACK_ENABLED.store(enabled, Ordering::SeqCst);
}

/// Check if automatic fallback is enabled
pub fn is_auto_fallback_enabled() -> bool {
    AUTO_FALLBACK_ENABLED.load(Ordering::SeqCst)
}

/// Trait for operations that support fallback
pub trait FallbackOperation<T> {
    /// Execute the operation with automatic fallback
    fn with_fallback(self) -> Result<T>;

    /// Execute the operation on CPU as fallback
    fn fallback_to_cpu(self) -> Result<T>;
}

/// Execute a binary operation with automatic fallback
pub fn execute_binary_op_with_fallback<T, F>(
    operation_name: &str,
    tensor_a: &Tensor<T>,
    tensor_b: &Tensor<T>,
    gpu_op: F,
    #[allow(unused_variables)] cpu_op: F,
) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::Zero + num_traits::One + Send + Sync + 'static + bytemuck::Pod,
    F: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
{
    let config = get_fallback_config();

    if !is_auto_fallback_enabled() {
        return gpu_op(tensor_a, tensor_b);
    }

    // Try the primary operation first
    match gpu_op(tensor_a, tensor_b) {
        Ok(result) => Ok(result),
        Err(error) => {
            if config.log_fallbacks {
                eprintln!("Operation '{operation_name}' failed: {error}. Attempting fallback...");
            }

            // Check if this error supports fallback
            if error.supports_fallback() && config.gpu_to_cpu {
                // Try to move tensors to CPU and retry
                match (tensor_a.device(), tensor_b.device()) {
                    #[cfg(feature = "gpu")]
                    (Device::Gpu(_), _) | (_, Device::Gpu(_)) => {
                        if config.log_fallbacks {
                            eprintln!(
                                "Falling back to CPU execution for operation '{}'",
                                operation_name
                            );
                        }

                        // Move tensors to CPU
                        let cpu_a = tensor_a.to_device(Device::Cpu)?;
                        let cpu_b = tensor_b.to_device(Device::Cpu)?;

                        // Execute on CPU
                        match cpu_op(&cpu_a, &cpu_b) {
                            Ok(result) => {
                                if config.log_fallbacks {
                                    eprintln!(
                                        "CPU fallback successful for operation '{}'",
                                        operation_name
                                    );
                                }
                                Ok(result)
                            }
                            Err(cpu_error) => {
                                if config.log_fallbacks {
                                    eprintln!(
                                        "CPU fallback also failed for operation '{}': {}",
                                        operation_name, cpu_error
                                    );
                                }
                                Err(cpu_error)
                            }
                        }
                    }
                    _ => {
                        // Already on CPU or other device, can't fallback further
                        Err(error)
                    }
                }
            } else {
                Err(error)
            }
        }
    }
}

/// Execute a unary operation with automatic fallback
pub fn execute_unary_op_with_fallback<T, F>(
    operation_name: &str,
    tensor: &Tensor<T>,
    gpu_op: F,
    #[allow(unused_variables)] cpu_op: F,
) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::Zero + num_traits::One + Send + Sync + 'static + bytemuck::Pod,
    F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
{
    let config = get_fallback_config();

    if !is_auto_fallback_enabled() {
        return gpu_op(tensor);
    }

    // Try the primary operation first
    match gpu_op(tensor) {
        Ok(result) => Ok(result),
        Err(error) => {
            if config.log_fallbacks {
                eprintln!("Operation '{operation_name}' failed: {error}. Attempting fallback...");
            }

            // Check if this error supports fallback
            if error.supports_fallback() && config.gpu_to_cpu {
                // Try to move tensor to CPU and retry
                #[cfg(feature = "gpu")]
                return if let Device::Gpu(_) = tensor.device() {
                    if config.log_fallbacks {
                        eprintln!(
                            "Falling back to CPU execution for operation '{}'",
                            operation_name
                        );
                    }

                    // Move tensor to CPU
                    let cpu_tensor = tensor.to_device(Device::Cpu)?;

                    // Execute on CPU
                    match cpu_op(&cpu_tensor) {
                        Ok(result) => {
                            if config.log_fallbacks {
                                eprintln!(
                                    "CPU fallback successful for operation '{}'",
                                    operation_name
                                );
                            }
                            Ok(result)
                        }
                        Err(cpu_error) => {
                            if config.log_fallbacks {
                                eprintln!(
                                    "CPU fallback also failed for operation '{}': {}",
                                    operation_name, cpu_error
                                );
                            }
                            Err(cpu_error)
                        }
                    }
                } else {
                    // Already on CPU or other device, can't fallback further
                    Err(error)
                };

                #[cfg(not(feature = "gpu"))]
                return Err(error);
            } else {
                Err(error)
            }
        }
    }
}

/// Memory cleanup utility for fallback scenarios
pub fn cleanup_memory_and_retry<T, F>(operation: F, max_retries: usize) -> Result<T>
where
    F: Fn() -> Result<T>,
{
    let mut attempt = 0;

    loop {
        match operation() {
            Ok(result) => return Ok(result),
            Err(error) => {
                attempt += 1;

                if attempt >= max_retries {
                    return Err(error);
                }

                // Check if this is a memory-related error
                match &error {
                    TensorError::AllocationError { .. } | TensorError::ResourceExhausted { .. } => {
                        eprintln!("Memory error detected, attempting cleanup (attempt {attempt}/{max_retries})");

                        // Trigger garbage collection if available
                        #[cfg(feature = "gpu")]
                        {
                            // Clear GPU memory pools
                            crate::memory::global_monitor().clear();
                        }

                        // Force garbage collection
                        std::hint::black_box(Vec::<u8>::new());

                        // Short delay before retry
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    _ => {
                        // Not a memory error, don't retry
                        return Err(error);
                    }
                }
            }
        }
    }
}

/// Wrapper for automatic fallback of results
pub struct FallbackWrapper<T> {
    result: Result<T>,
    operation_name: String,
}

impl<T> FallbackWrapper<T> {
    pub fn new(result: Result<T>, operation_name: &str) -> Self {
        Self {
            result,
            operation_name: operation_name.to_string(),
        }
    }

    pub fn with_cpu_fallback<F>(self, cpu_fallback: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        match self.result {
            Ok(result) => Ok(result),
            Err(error) => {
                if error.supports_fallback() && is_auto_fallback_enabled() {
                    let config = get_fallback_config();
                    if config.log_fallbacks {
                        eprintln!(
                            "Attempting CPU fallback for operation '{}'",
                            self.operation_name
                        );
                    }
                    cpu_fallback()
                } else {
                    Err(error)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device, Tensor};

    #[test]
    fn test_fallback_config() {
        let config = FallbackConfig::default();
        assert!(config.gpu_to_cpu);
        assert!(config.memory_cleanup);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_auto_fallback_flag() {
        assert!(is_auto_fallback_enabled()); // Default is true

        set_auto_fallback_enabled(false);
        assert!(!is_auto_fallback_enabled());

        set_auto_fallback_enabled(true);
        assert!(is_auto_fallback_enabled());
    }

    #[test]
    fn test_fallback_wrapper() {
        let success_result: Result<i32> = Ok(42);
        let wrapper = FallbackWrapper::new(success_result, "test_op");

        let result = wrapper.with_cpu_fallback(|| Ok(100));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_error_supports_fallback() {
        let gpu_error = TensorError::unsupported_device("test", "gpu:0", true);
        assert!(gpu_error.supports_fallback());

        let shape_error = TensorError::shape_mismatch("test", "[2, 2]", "[3, 3]");
        assert!(!shape_error.supports_fallback());
    }
}
