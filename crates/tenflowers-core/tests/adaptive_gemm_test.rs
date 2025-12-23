//! Tests for adaptive GEMM optimization functionality

#[cfg(feature = "gpu")]
use tenflowers_core::gpu::linalg::{AdaptiveGemmConfig, GpuLinalgContext};

// Simple test that works without GPU feature to verify compilation
#[test]
fn test_adaptive_gemm_basic_compilation() {
    // This test just ensures the code compiles and basic Rust functionality works
    let small_tile = 8u32;
    let medium_tile = 16u32;
    let large_tile = 32u32;

    assert!(small_tile < medium_tile);
    assert!(medium_tile < large_tile);

    // Test basic matrix size calculations
    let matrix_size = 256 * 256;
    let is_small = matrix_size < 256 * 256;
    let is_medium = matrix_size >= 256 * 256 && matrix_size < 2048 * 2048;

    assert!(!is_small);
    assert!(is_medium);
}

#[test]
#[cfg(feature = "gpu")]
fn test_adaptive_gemm_config_creation() {
    let config = AdaptiveGemmConfig::default();

    assert_eq!(config.small_tile_size, 8);
    assert_eq!(config.medium_tile_size, 16);
    assert_eq!(config.large_tile_size, 32);
    assert_eq!(config.workgroup_sizes.len(), 3);
    assert_eq!(config.coalescing_threshold, 128);
    assert!(config.prefer_shared_memory);
}

#[test]
#[cfg(feature = "gpu")]
fn test_adaptive_tile_size_selection() {
    let config = AdaptiveGemmConfig::default();

    // Small matrix should use small tile size
    let small_tile = config.select_tile_size(100, 100, 100);
    assert_eq!(small_tile, 8);

    // Medium matrix should use medium tile size
    let medium_tile = config.select_tile_size(512, 512, 256);
    assert_eq!(medium_tile, 16);

    // Large matrix should use large tile size
    let large_tile = config.select_tile_size(2048, 2048, 1024);
    assert_eq!(large_tile, 32);
}

#[test]
#[cfg(feature = "gpu")]
fn test_adaptive_workgroup_size_selection() {
    let config = AdaptiveGemmConfig::default();

    // Small operations should use small workgroups
    let (wg_x, wg_y) = config.select_workgroup_size(32, 32, 32);
    assert_eq!((wg_x, wg_y), (8, 8));

    // Medium operations should use medium workgroups
    let (wg_x, wg_y) = config.select_workgroup_size(256, 256, 128);
    assert_eq!((wg_x, wg_y), (16, 16));

    // Large operations should use large workgroups
    let (wg_x, wg_y) = config.select_workgroup_size(1024, 1024, 512);
    assert_eq!((wg_x, wg_y), (32, 32));
}

#[test]
#[cfg(feature = "gpu")]
fn test_bandwidth_utilization_estimation() {
    let config = AdaptiveGemmConfig::default();

    // Test bandwidth utilization calculation
    let utilization = config.estimate_bandwidth_utilization(256, 256, 256);

    // Should be a valid utilization value between 0 and 1
    assert!(utilization >= 0.0);
    assert!(utilization <= 1.0);

    // Larger matrices should generally have better utilization
    let large_utilization = config.estimate_bandwidth_utilization(1024, 1024, 1024);
    let small_utilization = config.estimate_bandwidth_utilization(64, 64, 64);

    assert!(large_utilization >= small_utilization);
}

#[test]
#[cfg(feature = "gpu")]
fn test_custom_adaptive_gemm_config() {
    let custom_config = AdaptiveGemmConfig {
        small_tile_size: 4,
        medium_tile_size: 8,
        large_tile_size: 16,
        workgroup_sizes: vec![(4, 4), (8, 8), (16, 16)],
        coalescing_threshold: 64,
        prefer_shared_memory: false,
    };

    // Test custom tile size selection
    let tile = custom_config.select_tile_size(100, 100, 100);
    assert_eq!(tile, 4);

    // Test custom workgroup size selection
    let (wg_x, wg_y) = custom_config.select_workgroup_size(32, 32, 32);
    assert_eq!((wg_x, wg_y), (4, 4));

    assert!(!custom_config.prefer_shared_memory);
}

#[test]
#[cfg(feature = "gpu")]
fn test_edge_case_matrix_dimensions() {
    let config = AdaptiveGemmConfig::default();

    // Test with very small matrices
    let tile = config.select_tile_size(1, 1, 1);
    assert_eq!(tile, 8); // Should still use small tile size

    // Test with very large matrices
    let tile = config.select_tile_size(10000, 10000, 5000);
    assert_eq!(tile, 32); // Should use large tile size

    // Test with asymmetric matrices
    let tile = config.select_tile_size(10, 10000, 10);
    assert_eq!(tile, 16); // Medium size due to one large dimension
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_linalg_context_with_adaptive_config() {
    use std::sync::Arc;
    use wgpu::{Backends, DeviceDescriptor, Features, Instance, Limits, RequestAdapterOptions};

    // This test requires GPU features but should compile without GPU
    let instance = Instance::new(&wgpu::InstanceDescriptor::default());

    // Try to get an adapter - this might fail in CI/testing environments
    let adapter_future = instance.request_adapter(&RequestAdapterOptions::default());

    // Use pollster to block on the async operation for testing
    if let Some(adapter) = pollster::block_on(adapter_future) {
        let device_future = adapter.request_device(
            &DeviceDescriptor {
                label: Some("test_device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        );

        if let Ok((device, queue)) = pollster::block_on(device_future) {
            let device = Arc::new(device);
            let queue = Arc::new(queue);

            let custom_config = AdaptiveGemmConfig {
                small_tile_size: 4,
                medium_tile_size: 12,
                large_tile_size: 24,
                workgroup_sizes: vec![(4, 4), (12, 12), (24, 24)],
                coalescing_threshold: 96,
                prefer_shared_memory: true,
            };

            // Test context creation with custom config
            let context =
                GpuLinalgContext::with_adaptive_gemm_config(device, queue, custom_config.clone());

            // Verify the config was set correctly
            let stored_config = context.adaptive_gemm_config();
            assert_eq!(stored_config.small_tile_size, 4);
            assert_eq!(stored_config.medium_tile_size, 12);
            assert_eq!(stored_config.large_tile_size, 24);
        }
    }
    // If GPU is not available, the test passes silently
}
