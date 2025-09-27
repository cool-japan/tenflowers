//! Metal Device Management
//!
//! This module provides the core Metal device wrapper and management functionality
//! for GPU operations on Apple devices.

use super::types::{DispatchConfig, MemoryAccessPattern};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{Result, TensorError};
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal;
use std::collections::HashMap;

/// Metal device wrapper for managing Metal context
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
pub struct MetalDevice {
    /// Metal device handle
    device: metal::Device,
    /// Command queue for submitting GPU work
    command_queue: metal::CommandQueue,
    /// Library containing compiled shaders
    library: metal::Library,
    /// Cache of compiled compute pipelines
    pipeline_cache: HashMap<String, metal::ComputePipelineState>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalDevice {
    /// Create a new Metal device instance
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or_else(|| {
            TensorError::device_error_simple("No Metal device available".to_string())
        })?;

        let command_queue = device.new_command_queue();

        // Compile embedded shaders
        let library_source = include_str!("shaders/metal_kernels.metal");
        let library = device
            .new_library_with_source(library_source, &metal::CompileOptions::new())
            .map_err(|e| {
                TensorError::device_error_simple(format!("Failed to compile Metal shaders: {}", e))
            })?;

        Ok(MetalDevice {
            device,
            command_queue,
            library,
            pipeline_cache: HashMap::new(),
        })
    }

    /// Get the underlying Metal device
    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &metal::CommandQueue {
        &self.command_queue
    }

    /// Get the shader library
    pub fn library(&self) -> &metal::Library {
        &self.library
    }

    /// Get or create a compute pipeline for the specified kernel
    pub fn get_or_create_pipeline(
        &mut self,
        kernel_name: &str,
    ) -> Result<&metal::ComputePipelineState> {
        if !self.pipeline_cache.contains_key(kernel_name) {
            let function = self.library.get_function(kernel_name, None).or_else(|_| {
                Err(TensorError::device_error_simple(format!(
                    "Kernel '{}' not found",
                    kernel_name
                )))
            })?;

            let pipeline_state = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    TensorError::device_error_simple(format!(
                        "Failed to create pipeline for '{}': {}",
                        kernel_name, e
                    ))
                })?;

            self.pipeline_cache
                .insert(kernel_name.to_string(), pipeline_state);
        }

        Ok(self.pipeline_cache.get(kernel_name).unwrap())
    }

    /// Calculate optimal dispatch configuration for given tensor shapes
    pub fn calculate_optimal_dispatch_config(&self, shapes: &[usize]) -> Result<DispatchConfig> {
        let total_elements: usize = shapes.iter().product();

        // Optimize for Apple Silicon characteristics
        let max_threads_per_group = 1024; // Typical for Apple Silicon
        let preferred_warp_size = 32; // SIMD width for Apple Silicon

        let threads_per_group = if total_elements < max_threads_per_group {
            // Round up to next multiple of warp size
            ((total_elements + preferred_warp_size - 1) / preferred_warp_size) * preferred_warp_size
        } else {
            max_threads_per_group
        };

        let thread_groups = (total_elements + threads_per_group - 1) / threads_per_group;

        Ok(DispatchConfig {
            thread_groups: metal::MTLSize::new(thread_groups as u64, 1, 1),
            threads_per_group: metal::MTLSize::new(threads_per_group as u64, 1, 1),
            memory_access: MemoryAccessPattern::Sequential,
        })
    }

    /// Optimize memory access patterns for given tensor shapes
    pub fn optimize_memory_access_pattern(
        &self,
        shapes: &[&[usize]],
    ) -> Result<MemoryAccessPattern> {
        if shapes.is_empty() {
            return Ok(MemoryAccessPattern::Sequential);
        }

        // Analyze shapes to determine optimal access pattern
        let total_size: usize = shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();

        if total_size < 1024 * 1024 {
            // < 1MB
            Ok(MemoryAccessPattern::Sequential)
        } else if shapes.len() == 2 && shapes[0].len() == 2 && shapes[1].len() == 2 {
            // Matrix operations - use tiling
            let tile_size = if total_size > 16 * 1024 * 1024 {
                // > 16MB
                (32, 32)
            } else {
                (16, 16)
            };
            Ok(MemoryAccessPattern::Tiled { tile_size })
        } else {
            // Large tensors - use blocking
            let block_size = if total_size > 64 * 1024 * 1024 {
                // > 64MB
                8192
            } else {
                4096
            };
            Ok(MemoryAccessPattern::Blocked { block_size })
        }
    }

    /// Check device capabilities and features
    pub fn get_device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_metal_3: self.device.supports_family(metal::MTLGPUFamily::Apple8),
            max_threads_per_threadgroup: 1024, // Typical for Apple Silicon
            supports_simdgroup_matrix: true,   // Available on Apple Silicon
            memory_bandwidth_gbps: self.estimate_memory_bandwidth(),
            compute_units: self.get_compute_unit_count(),
        }
    }

    fn estimate_memory_bandwidth(&self) -> f64 {
        // Rough estimates based on Apple Silicon generations
        // M1: ~68 GB/s, M1 Pro: ~200 GB/s, M1 Max: ~400 GB/s, M2: ~100 GB/s
        400.0 // Conservative estimate
    }

    fn get_compute_unit_count(&self) -> u32 {
        // This is an approximation - actual count varies by chip
        // M1: 8 cores, M1 Pro: 14-16 cores, M1 Max: 24-32 cores
        16 // Conservative estimate
    }
}

/// Metal device capabilities information
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub supports_metal_3: bool,
    pub max_threads_per_threadgroup: u32,
    pub supports_simdgroup_matrix: bool,
    pub memory_bandwidth_gbps: f64,
    pub compute_units: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_device_creation() {
        // Only run if Metal is available
        if let Ok(device) = MetalDevice::new() {
            let caps = device.get_device_capabilities();
            assert!(caps.max_threads_per_threadgroup > 0);
            assert!(caps.compute_units > 0);
            assert!(caps.memory_bandwidth_gbps > 0.0);
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_dispatch_config_calculation() {
        if let Ok(device) = MetalDevice::new() {
            let shapes = &[1024, 1024];
            let config = device.calculate_optimal_dispatch_config(shapes);
            assert!(config.is_ok());

            let config = config.unwrap();
            assert!(config.thread_groups.width > 0);
            assert!(config.threads_per_group.width > 0);
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_memory_access_optimization() {
        if let Ok(device) = MetalDevice::new() {
            // Test different tensor shapes
            let small_shapes = vec![&[64, 64][..]];
            let pattern = device.optimize_memory_access_pattern(&small_shapes);
            assert!(pattern.is_ok());

            let large_matrix_shapes = vec![&[1024, 1024][..], &[1024, 1024][..]];
            let pattern = device.optimize_memory_access_pattern(&large_matrix_shapes);
            assert!(pattern.is_ok());

            if let Ok(MemoryAccessPattern::Tiled { tile_size }) = pattern {
                assert!(tile_size.0 > 0 && tile_size.1 > 0);
            }
        }
    }
}
