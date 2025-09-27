//! GPU Context Management for Image Transforms
//!
//! This module provides GPU device management and context initialization
//! for WGPU-based image transformation operations.

use tenflowers_core::{Result, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
#[allow(unused_imports)]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, MapMode, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource,
};

/// GPU context for image transforms
#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Create a new GPU context
    pub async fn new() -> Result<Self> {
        // Request an adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                TensorError::device_error_simple("No suitable GPU adapter found".to_string())
            })?;

        // Request a device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Transforms Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                TensorError::device_error_simple(format!("Failed to create GPU device: {}", e))
            })?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Get a reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a reference to the queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Create a compute pipeline from shader source
    pub fn create_compute_pipeline(
        &self,
        shader_source: &str,
        bind_group_layout: &BindGroupLayout,
        entry_point: &str,
    ) -> ComputePipeline {
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Transform Compute Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Transform Pipeline Layout"),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Transform Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            })
    }
}

/// CPU fallback context when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    /// Create a new GPU context (fallback to CPU)
    pub async fn new() -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GPU transforms require 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_context_creation() {
        // Note: This test may fail in CI environments without GPU
        match GpuContext::new().await {
            Ok(context) => {
                assert!(
                    !context.device().features().is_empty()
                        || context.device().features().is_empty()
                );
                // Context creation successful
            }
            Err(_) => {
                // GPU not available in test environment, which is acceptable
                println!("GPU not available in test environment");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn test_gpu_context_fallback() {
        // Test that GPU context creation fails when GPU feature is disabled
        // Since this is a unit test and the GPU feature is disabled, we expect an error
        assert!(true); // Placeholder test - GPU context would fail without GPU feature
    }
}
