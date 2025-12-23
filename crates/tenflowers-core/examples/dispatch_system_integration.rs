//! Comprehensive Dispatch System Integration Example
//!
//! This example demonstrates the full power of TenfloweRS core infrastructure:
//! - Unified dispatch registry for cross-backend execution
//! - Automatic kernel selection based on device and capability
//! - Shape error taxonomy for clear error messages
//! - Performance monitoring and profiling
//! - Elementwise fusion optimization
//! - GPU memory diagnostics
//!
//! Run with: cargo run --example dispatch_system_integration --features gpu

use scirs2_core::ndarray::array;
use tenflowers_core::{
    dispatch_registry::{BackendType, DispatchRegistry, KernelImplementation, OperationDescriptor},
    shape_error_taxonomy::{validate_elementwise_shapes, ShapeErrorBuilder, ShapeErrorCategory},
    DType, Device, Result, Shape, Tensor, TensorError,
};

/// Custom kernel: element-wise squared operation (x^2)
fn squared_cpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + Clone
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable
        + 'static,
{
    println!("  [CPU] Executing squared operation on CPU");

    // Use tensor multiplication
    input.mul(input)
}

/// SIMD-optimized squared operation (placeholder - would use actual SIMD)
#[cfg(feature = "simd")]
fn squared_simd<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + Clone
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable
        + 'static,
{
    println!("  [SIMD] Executing squared operation with SIMD optimization");
    // In a real implementation, this would use SIMD instructions
    input.mul(input)
}

/// Custom binary kernel: weighted addition (a * w1 + b * w2)
fn weighted_add_cpu(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Result<Tensor<f32>> {
    println!("  [CPU] Executing weighted addition on CPU");

    // Validate shapes using the shape error taxonomy
    validate_elementwise_shapes("weighted_add", lhs.shape(), rhs.shape())?;

    // For this example, use equal weights (0.5 * a + 0.5 * b)
    // Simply average the two tensors
    let sum = lhs.add(rhs)?;
    let two = Tensor::from_array(array![2.0f32].into_dyn());
    sum.div(&two)
}

/// Demonstrate automatic backend selection
fn demo_backend_selection() -> Result<()> {
    println!("\n=== Demo 1: Automatic Backend Selection ===\n");

    // Create a custom registry for demonstration
    let registry: DispatchRegistry<f32> = DispatchRegistry::new();

    // Register the "squared" operation
    let desc = OperationDescriptor::new("squared", "unary")
        .with_dtypes(vec![DType::Float32, DType::Float64])
        .with_broadcast();

    registry.register_operation(desc)?;

    // Register CPU kernel (always available)
    registry.register_kernel(
        "squared",
        KernelImplementation::unary(BackendType::Cpu, squared_cpu),
    )?;

    // Register SIMD kernel (if available)
    #[cfg(feature = "simd")]
    {
        registry.register_kernel(
            "squared",
            KernelImplementation::unary(BackendType::SimdCpu, squared_simd),
        )?;
    }

    // Create test tensor
    let data = array![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_array(data.into_dyn());

    println!("Input tensor: {:?}", tensor.data());

    // Dispatch will automatically select best available backend
    let result = registry.dispatch_unary("squared", &tensor)?;

    println!("Result tensor: {:?}", result.data());
    println!(
        "\nAvailable backends for 'squared': {:?}",
        registry.available_backends("squared")
    );

    Ok(())
}

/// Demonstrate shape validation with helpful error messages
fn demo_shape_validation() -> Result<()> {
    println!("\n=== Demo 2: Shape Validation with Taxonomy ===\n");

    let a = Tensor::<f32>::zeros(&[3, 4]);
    let b = Tensor::<f32>::zeros(&[3, 5]); // Incompatible shape!

    println!("Tensor A shape: {:?}", a.shape());
    println!("Tensor B shape: {:?}", b.shape());

    // Attempt element-wise operation with incompatible shapes
    match validate_elementwise_shapes("add", a.shape(), b.shape()) {
        Ok(_) => println!("Shapes are compatible"),
        Err(e) => {
            println!("\n[Expected Error] Shape validation failed:");
            println!("{}", e);

            // Build a detailed shape error with suggestions
            let detailed_error =
                ShapeErrorBuilder::new("custom_op", ShapeErrorCategory::ElementwiseMismatch)
                    .expected("Shapes must match exactly or be broadcastable")
                    .got(&format!(
                        "Got shapes {:?} and {:?}",
                        a.shape().dims(),
                        b.shape().dims()
                    ))
                    .detail("Dimension 1 differs: 4 vs 5")
                    .suggestion("Ensure both tensors have shape [3, 4] or adjust the second tensor")
                    .build();

            println!("\nDetailed error message:");
            println!("{}", detailed_error);
        }
    }

    Ok(())
}

/// Demonstrate multi-backend execution with performance comparison
fn demo_performance_comparison() -> Result<()> {
    println!("\n=== Demo 3: Performance Comparison Across Backends ===\n");

    let registry: DispatchRegistry<f32> = DispatchRegistry::new();

    // Register operation
    let desc = OperationDescriptor::new("weighted_add", "binary")
        .with_dtypes(vec![DType::Float32])
        .with_broadcast();

    registry.register_operation(desc)?;
    registry.register_kernel(
        "weighted_add",
        KernelImplementation::binary(BackendType::Cpu, weighted_add_cpu),
    )?;

    // Create test tensors
    let size = 1000;
    let a = Tensor::ones(&[size]);
    let b = Tensor::ones(&[size]);

    println!("Running weighted_add on tensors of size {}", size);

    // Execute and time
    let start = std::time::Instant::now();
    let result = registry.dispatch_binary("weighted_add", &a, &b)?;
    let duration = start.elapsed();

    println!("Execution time: {:?}", duration);
    println!("Result sample: {:?}", &result.data()[0..5]);

    Ok(())
}

/// Demonstrate shape inference integration
fn demo_shape_inference() -> Result<()> {
    println!("\n=== Demo 4: Shape Inference Integration ===\n");

    use tenflowers_core::ops::infer_binary_elementwise_validated;

    let shape_a = Shape::from_slice(&[2, 3, 4]);
    let shape_b = Shape::from_slice(&[4]);

    println!("Shape A: {:?}", shape_a.dims());
    println!("Shape B: {:?}", shape_b.dims());

    match infer_binary_elementwise_validated(&shape_a, &shape_b) {
        Ok(result_shape) => {
            println!("Inferred output shape: {:?}", result_shape.dims());
            println!("✓ Broadcasting is valid!");
        }
        Err(e) => println!("Broadcasting failed: {}", e),
    }

    Ok(())
}

/// Demonstrate GPU memory diagnostics (when GPU feature is enabled)
#[cfg(feature = "gpu")]
fn demo_gpu_diagnostics() -> Result<()> {
    use std::time::Duration;
    use tenflowers_core::gpu::memory_diagnostics::{print_gpu_diagnostics, DiagnosticsConfig};

    println!("\n=== Demo 5: GPU Memory Diagnostics ===\n");

    // Configure diagnostics
    let config = DiagnosticsConfig {
        leak_detection_threshold: Duration::from_secs(300),
        auto_diagnostics: true,
        diagnostics_interval: Duration::from_secs(60),
        analyze_fragmentation: true,
        enable_profiling: true,
    };

    println!(
        "GPU diagnostics config: auto={}, profiling={}",
        config.auto_diagnostics, config.enable_profiling
    );

    // Create some GPU tensors to generate allocation events
    let gpu_device = Device::Gpu(0);
    let tensor1 = Tensor::<f32>::zeros(&[100, 100]);
    let tensor2 = Tensor::<f32>::ones(&[200, 200]);

    // Transfer to GPU
    let tensor1_gpu = tensor1.to_device(gpu_device.clone())?;
    let tensor2_gpu = tensor2.to_device(gpu_device)?;

    println!(
        "Created GPU tensors: {} and {}",
        tensor1_gpu.shape(),
        tensor2_gpu.shape()
    );

    // Print diagnostics
    print_gpu_diagnostics();

    Ok(())
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  TenfloweRS Dispatch System Integration Example          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    // Run all demonstrations
    demo_backend_selection()?;
    demo_shape_validation()?;
    demo_performance_comparison()?;
    demo_shape_inference()?;

    #[cfg(feature = "gpu")]
    demo_gpu_diagnostics()?;

    #[cfg(not(feature = "gpu"))]
    println!("\n[Info] GPU diagnostics demo skipped (enable 'gpu' feature to see it)");

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  All demonstrations completed successfully!               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
