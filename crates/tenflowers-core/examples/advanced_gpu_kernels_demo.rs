//! Advanced GPU Kernels Demonstration
//!
//! This example showcases the cutting-edge GPU optimization features in TenfloweRS,
//! including Tensor Core utilization, SIMD-group operations, and wavefront primitives.
//!
//! To run this example:
//! ```bash
//! # For CUDA with Tensor Cores (requires NVIDIA RTX/Tesla/A100)
//! cargo run --features gpu,cuda --example advanced_gpu_kernels_demo
//!
//! # For Metal on Apple Silicon
//! cargo run --features gpu,metal --example advanced_gpu_kernels_demo
//!
//! # For ROCm on AMD GPUs
//! cargo run --features gpu,rocm --example advanced_gpu_kernels_demo
//! ```

use tenflowers_core::ops::random::random_normal_f32_device;
use tenflowers_core::{DType, Device, Result, Tensor};

#[cfg(feature = "gpu")]
use tenflowers_core::gpu::advanced_kernel_manager::{
    AdvancedKernelManager, ComputeIntensity, KernelStrategy, MemoryPattern, OptimizationHints,
    PrecisionRequirements,
};

use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 TenfloweRS Advanced GPU Kernels Demo");
    println!("========================================");

    // Check if GPU features are available
    #[cfg(not(feature = "gpu"))]
    {
        println!("❌ GPU features not enabled. Please compile with --features gpu");
        println!("   For CUDA: --features gpu,cuda");
        println!("   For Metal: --features gpu,metal");
        println!("   For ROCm: --features gpu,rocm");
        return Ok(());
    }

    #[cfg(feature = "gpu")]
    {
        // Initialize GPU device
        let device = Device::try_gpu(0).unwrap_or_else(|_| {
            println!("⚠️  GPU device not available, using CPU");
            Device::Cpu
        });

        match &device {
            Device::Gpu(_) => {
                println!("✅ GPU device initialized successfully");
                run_advanced_gpu_demo(&device)?;
            }
            _ => {
                println!("⚠️  Running on CPU - advanced GPU features not available");
                run_cpu_fallback_demo()?;
            }
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_advanced_gpu_demo(device: &Device) -> Result<()> {
    println!("\n🎯 Initializing Advanced Kernel Manager");

    // Create advanced kernel manager with automatic hardware detection
    let kernel_manager = AdvancedKernelManager::new(device)?;
    println!("✅ Kernel manager initialized with optimal strategy");

    // Demo 1: High-Performance Matrix Multiplication
    println!("\n🔢 Demo 1: Advanced Matrix Multiplication");
    demo_advanced_matmul(&kernel_manager, device)?;

    // Demo 2: Precision Optimization Showcase
    println!("\n🎯 Demo 2: Precision Optimization");
    demo_precision_optimization(&kernel_manager, device)?;

    // Demo 3: Memory Pattern Optimization
    println!("\n💾 Demo 3: Memory Access Pattern Optimization");
    demo_memory_optimization(&kernel_manager, device)?;

    // Demo 4: Performance Profiling
    println!("\n📊 Demo 4: Performance Profiling");
    demo_performance_profiling(&kernel_manager)?;

    println!("\n✅ All advanced GPU demos completed successfully!");
    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_advanced_matmul(kernel_manager: &AdvancedKernelManager, device: &Device) -> Result<()> {
    // Test different matrix sizes to showcase scaling performance
    let test_sizes = vec![
        (512, 512, 512),    // Medium workload
        (1024, 1024, 1024), // Large workload
        (2048, 2048, 2048), // Very large workload
    ];

    for (m, n, k) in test_sizes {
        println!("  📐 Testing {}x{}x{} matrix multiplication", m, n, k);

        // Create test matrices
        let a = random_normal_f32_device(&[m, k], 0.0, 1.0, None, device)?;
        let b = random_normal_f32_device(&[k, n], 0.0, 1.0, None, device)?;

        // Set optimization hints for compute-bound workload
        let hints = OptimizationHints {
            compute_intensity: ComputeIntensity::ComputeBound,
            tensor_shapes: vec![vec![m, k], vec![k, n]],
            is_repetitive: true,
            memory_pattern: MemoryPattern::Coalesced,
            precision_requirements: PrecisionRequirements::StandardPrecision,
        };

        // Benchmark standard vs advanced implementation
        let start = Instant::now();
        let result_standard = a.matmul(&b)?;
        let time_standard = start.elapsed();

        let start = Instant::now();
        let result_advanced = kernel_manager.optimized_matmul(&a, &b, hints)?;
        let time_advanced = start.elapsed();

        // Verify numerical correctness
        let diff = tenflowers_core::ops::sub(&result_standard, &result_advanced)?;
        let abs_diff = diff.abs()?;
        let max_diff = abs_diff.max(None, false)?;
        println!(
            "    ⏱️  Standard: {:.2}ms | Advanced: {:.2}ms",
            time_standard.as_secs_f32() * 1000.0,
            time_advanced.as_secs_f32() * 1000.0
        );
        println!(
            "    🎯 Speedup: {:.2}x | Max difference: {:.2e}",
            time_standard.as_secs_f32() / time_advanced.as_secs_f32(),
            max_diff.to_scalar()?
        );

        // Performance verification
        if time_advanced <= time_standard {
            println!("    ✅ Advanced implementation is faster!");
        } else {
            println!("    ⚠️  Advanced implementation may need tuning for this size");
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_precision_optimization(
    kernel_manager: &AdvancedKernelManager,
    device: &Device,
) -> Result<()> {
    let size = 1024;

    // Test different precision modes
    let precisions = vec![
        (
            DType::Float32,
            PrecisionRequirements::StandardPrecision,
            "FP32",
        ),
        (
            DType::Float16,
            PrecisionRequirements::ReducedPrecision,
            "FP16",
        ),
    ];

    for (dtype, precision_req, name) in precisions {
        if !device.supports_dtype(dtype) {
            println!("  ⚠️  {} not supported on this device", name);
            continue;
        }

        println!("  🎯 Testing {} precision", name);

        let a = random_normal_f32_device(&[size, size], 0.0, 1.0, None, device)?;
        let b = random_normal_f32_device(&[size, size], 0.0, 1.0, None, device)?;

        let hints = OptimizationHints {
            compute_intensity: ComputeIntensity::ComputeBound,
            tensor_shapes: vec![vec![size, size], vec![size, size]],
            is_repetitive: true,
            memory_pattern: MemoryPattern::Coalesced,
            precision_requirements: precision_req,
        };

        let start = Instant::now();
        let result = kernel_manager.optimized_matmul(&a, &b, hints)?;
        let elapsed = start.elapsed();

        let gflops =
            (2.0 * size as f64 * size as f64 * size as f64) / (elapsed.as_secs_f64() * 1e9);

        println!(
            "    ⏱️  Time: {:.2}ms | Performance: {:.1} GFLOPS",
            elapsed.as_secs_f32() * 1000.0,
            gflops
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_memory_optimization(kernel_manager: &AdvancedKernelManager, device: &Device) -> Result<()> {
    let size = 2048;

    // Test different memory access patterns
    let patterns = vec![
        (MemoryPattern::Coalesced, "Coalesced"),
        (MemoryPattern::Sequential, "Sequential"),
        (MemoryPattern::Strided { stride: 2 }, "Strided"),
    ];

    for (pattern, name) in patterns {
        println!("  🧠 Testing {} memory pattern", name);

        let a = random_normal_f32_device(&[size, size], 0.0, 1.0, None, device)?;
        let b = random_normal_f32_device(&[size, size], 0.0, 1.0, None, device)?;

        let hints = OptimizationHints {
            compute_intensity: ComputeIntensity::Balanced,
            tensor_shapes: vec![vec![size, size], vec![size, size]],
            is_repetitive: true,
            memory_pattern: pattern,
            precision_requirements: PrecisionRequirements::StandardPrecision,
        };

        let start = Instant::now();
        let result = kernel_manager.optimized_matmul(&a, &b, hints)?;
        let elapsed = start.elapsed();

        // Estimate memory bandwidth utilization
        let bytes_transferred = (size * size * 3 * 4) as f64; // 3 matrices * 4 bytes per float
        let bandwidth_gbps = bytes_transferred / (elapsed.as_secs_f64() * 1e9);

        println!(
            "    ⏱️  Time: {:.2}ms | Bandwidth: {:.1} GB/s",
            elapsed.as_secs_f32() * 1000.0,
            bandwidth_gbps
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_performance_profiling(kernel_manager: &AdvancedKernelManager) -> Result<()> {
    println!("  📈 Retrieving performance statistics...");

    let stats = kernel_manager.get_performance_stats()?;

    if stats.is_empty() {
        println!("    ℹ️  No performance data available yet");
        return Ok(());
    }

    println!("    📊 Performance Statistics:");
    for (kernel_id, perf_data) in stats.iter() {
        println!("      🔧 {}", kernel_id);
        println!(
            "         ⏱️  Avg execution: {:.2}ms",
            perf_data.avg_execution_time / 1000.0
        );
        println!(
            "         💾 Memory utilization: {:.1}%",
            perf_data.memory_bandwidth_util
        );
        println!(
            "         🖥️  Compute utilization: {:.1}%",
            perf_data.compute_utilization
        );
        println!(
            "         🚀 Speedup factor: {:.2}x",
            perf_data.speedup_factor
        );
        println!("         📈 Executions: {}", perf_data.execution_count);
    }

    Ok(())
}

fn run_cpu_fallback_demo() -> Result<()> {
    println!("\n💻 Running CPU fallback demonstration");

    let device = Device::Cpu;
    let size = 512; // Smaller size for CPU demo

    println!("  🔢 Creating {}x{} test matrices", size, size);
    let a = random_normal_f32_device(&[size, size], 0.0, 1.0, None, &device)?;
    let b = random_normal_f32_device(&[size, size], 0.0, 1.0, None, &device)?;

    println!("  ⚡ Computing matrix multiplication...");
    let start = Instant::now();
    let result = a.matmul(&b)?;
    let elapsed = start.elapsed();

    let gflops = (2.0 * size as f64 * size as f64 * size as f64) / (elapsed.as_secs_f64() * 1e9);

    println!("  ✅ Result shape: {:?}", result.shape());
    println!(
        "  ⏱️  Execution time: {:.2}ms",
        elapsed.as_secs_f32() * 1000.0
    );
    println!("  📊 Performance: {:.1} GFLOPS", gflops);
    println!("  ℹ️  Enable GPU features for advanced optimizations!");

    Ok(())
}

// Utility trait extension for device capability checking
trait DeviceExt {
    fn supports_dtype(&self, dtype: DType) -> bool;
}

impl DeviceExt for Device {
    fn supports_dtype(&self, dtype: DType) -> bool {
        match self {
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => {
                // Most modern GPUs support these types
                matches!(dtype, DType::Float32 | DType::Float16 | DType::Int32)
            }
            _ => {
                // CPU supports all standard types
                matches!(
                    dtype,
                    DType::Float32 | DType::Float64 | DType::Int32 | DType::Int64
                )
            }
        }
    }
}
