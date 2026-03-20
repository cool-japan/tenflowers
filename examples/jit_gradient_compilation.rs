/// JIT Gradient Compilation Example
/// 
/// This example demonstrates how to use the JIT compilation system for gradient kernels.
/// It shows kernel caching, performance optimization, and runtime compilation.

use tenflowers_autograd::{
    GradientTape, TrackedTensor,
    jit_utils, JitConfig, JitGradientContext, OptimizationLevel, DeviceFeatures,
};
use tenflowers_core::{Tensor, Device, Result};
use std::time::Instant;

fn main() -> Result<()> {
    pollster::block_on(async_main())
}

async fn async_main() -> Result<()> {
    println!("🔥 TenfloweRS JIT Gradient Compilation Example");
    println!("=============================================");

    // Initialize JIT compilation system
    println!("🚀 Initializing JIT compilation system...");
    jit_utils::initialize_jit().await?;
    
    // Run JIT compilation examples
    basic_jit_example().await?;
    performance_comparison_example().await?;
    optimization_levels_example().await?;
    custom_kernel_example().await?;
    
    println!("\n🎉 JIT compilation example completed!");
    Ok(())
}

/// Basic JIT compilation example
async fn basic_jit_example() -> Result<()> {
    println!("\n1. Basic JIT Compilation");
    println!("-----------------------");
    
    let jit_context = jit_utils::create_jit_context();
    
    // Create test tensors
    let a: Tensor<f32> = Tensor::randn(&[1000, 1000])?;
    let b: Tensor<f32> = Tensor::randn(&[1000, 1000])?;
    let grad_output: Tensor<f32> = Tensor::ones(&[1000, 1000])?;
    
    println!("Tensor shapes: A={:?}, B={:?}", a.shape().dims(), b.shape().dims());
    
    // Compile and execute gradient kernel for addition
    let inputs = vec![&a, &b];
    let compiled_kernel = jit_context.compile_gradient_kernel(
        "add_backward", 
        &inputs, 
        grad_output.shape().dims()
    ).await?;
    
    println!("✅ Compiled kernel in {:.2}ms", compiled_kernel.compile_time_ms);
    println!("   Estimated performance: {:.2}μs", 
        compiled_kernel.estimated_performance.estimated_execution_time_us);
    println!("   Memory bound ratio: {:.1}%", 
        compiled_kernel.estimated_performance.memory_bound_ratio * 100.0);
    
    // Execute the compiled kernel
    let gradients = jit_context.execute_jit_gradient(
        "add_backward",
        &inputs,
        &grad_output,
        &compiled_kernel
    ).await?;
    
    println!("✅ Executed gradient computation, got {} gradients", gradients.len());
    
    Ok(())
}

/// Performance comparison between JIT and regular computation
async fn performance_comparison_example() -> Result<()> {
    println!("\n2. Performance Comparison");
    println!("------------------------");
    
    let shapes = vec![
        vec![100, 100],
        vec![500, 500], 
        vec![1000, 1000],
        vec![2000, 2000],
    ];
    
    println!("| Shape | JIT Time (μs) | Regular Time (μs) | Speedup |");
    println!("|-------|---------------|-------------------|---------|");
    
    for shape in shapes {
        let a: Tensor<f32> = Tensor::randn(&shape)?;
        let b: Tensor<f32> = Tensor::randn(&shape)?;
        let grad_output: Tensor<f32> = Tensor::ones(&shape)?;
        
        let inputs = vec![&a, &b];
        
        // Benchmark JIT vs regular computation
        let (jit_time, regular_time) = jit_utils::benchmark_jit_performance(
            "mul_backward",
            &inputs,
            &grad_output,
            10 // iterations
        ).await?;
        
        let speedup = regular_time / jit_time;
        let shape_str = format!("{}x{}", shape[0], shape[1]);
        
        println!("| {} | {:.1} | {:.1} | {:.2}x |", 
            shape_str, jit_time, regular_time, speedup);
    }
    
    Ok(())
}

/// Demonstrate different optimization levels
async fn optimization_levels_example() -> Result<()> {
    println!("\n3. Optimization Levels");
    println!("---------------------");
    
    let optimization_levels = vec![
        (OptimizationLevel::Debug, "Debug"),
        (OptimizationLevel::Balanced, "Balanced"),
        (OptimizationLevel::Aggressive, "Aggressive"),
    ];
    
    let a: Tensor<f32> = Tensor::randn(&[1024, 1024])?;
    let b: Tensor<f32> = Tensor::randn(&[1024, 1024])?;
    let grad_output: Tensor<f32> = Tensor::ones(&[1024, 1024])?;
    let inputs = vec![&a, &b];
    
    println!("| Optimization | Compile Time (ms) | Est. Exec Time (μs) | Workgroup Size |");
    println!("|--------------|-------------------|-------------------|----------------|");
    
    for (opt_level, name) in optimization_levels {
        let mut config = JitConfig::default();
        config.optimization_level = opt_level;
        
        let jit_context = JitGradientContext::new(config);
        
        let compiled_kernel = jit_context.compile_gradient_kernel(
            "sigmoid_backward",
            &inputs,
            grad_output.shape().dims()
        ).await?;
        
        println!("| {} | {:.2} | {:.2} | {}x{}x{} |",
            name,
            compiled_kernel.compile_time_ms,
            compiled_kernel.estimated_performance.estimated_execution_time_us,
            compiled_kernel.workgroup_size.0,
            compiled_kernel.workgroup_size.1,
            compiled_kernel.workgroup_size.2
        );
    }
    
    Ok(())
}

/// Custom kernel template example
async fn custom_kernel_example() -> Result<()> {
    println!("\n4. Custom Kernel Templates");
    println!("--------------------------");
    
    // This would register a custom gradient kernel template
    // For now, we'll just demonstrate the concept
    
    let mut config = JitConfig::default();
    config.debug_output = true;
    
    let jit_context = JitGradientContext::new(config);
    
    // Simulate different custom operations
    let custom_ops = vec![
        "custom_activation_backward",
        "fused_conv_relu_backward",
        "custom_attention_backward",
    ];
    
    println!("Custom operations that could be JIT compiled:");
    for op in custom_ops {
        println!("  ✨ {}", op);
    }
    
    println!("\nIn a full implementation, these would be registered as:");
    println!("  - Custom WGSL template generation");
    println!("  - Shape-specific optimizations");
    println!("  - Fused operation kernels");
    println!("  - Device-specific tuning");
    
    Ok(())
}

/// Advanced JIT features demonstration
async fn advanced_jit_features() -> Result<()> {
    println!("\n5. Advanced JIT Features");
    println!("-----------------------");
    
    let mut config = JitConfig::default();
    config.auto_tune = true;
    config.debug_output = true;
    config.cache_size_limit = 500;
    
    let jit_context = JitGradientContext::new(config);
    
    // Simulate multiple operations to build up cache
    let operations = vec![
        "add_backward",
        "mul_backward", 
        "relu_backward",
        "sigmoid_backward",
        "tanh_backward",
    ];
    
    let a: Tensor<f32> = Tensor::randn(&[512, 512])?;
    let b: Tensor<f32> = Tensor::randn(&[512, 512])?;
    let grad_output: Tensor<f32> = Tensor::ones(&[512, 512])?;
    let inputs = vec![&a, &b];
    
    println!("Compiling and executing multiple operations...");
    
    for operation in operations {
        let start = Instant::now();
        
        let compiled_kernel = jit_context.compile_gradient_kernel(
            operation,
            &inputs,
            grad_output.shape().dims()
        ).await?;
        
        let _gradients = jit_context.execute_jit_gradient(
            operation,
            &inputs,
            &grad_output,
            &compiled_kernel
        ).await?;
        
        let total_time = start.elapsed().as_micros();
        println!("  {} completed in {}μs", operation, total_time);
    }
    
    // Show performance report
    println!("\nPerformance Report:");
    println!("{}", jit_context.get_performance_report());
    
    // Auto-tune kernels
    println!("Running auto-tuning...");
    jit_context.auto_tune_kernels().await?;
    
    Ok(())
}

/// Demonstration of kernel caching benefits
async fn kernel_caching_example() -> Result<()> {
    println!("\n6. Kernel Caching Benefits");
    println!("-------------------------");
    
    let jit_context = jit_utils::create_debug_jit_context();
    
    let a: Tensor<f32> = Tensor::randn(&[256, 256])?;
    let b: Tensor<f32> = Tensor::randn(&[256, 256])?;
    let grad_output: Tensor<f32> = Tensor::ones(&[256, 256])?;
    let inputs = vec![&a, &b];
    
    println!("First compilation (cache miss):");
    let start = Instant::now();
    let compiled_kernel = jit_context.compile_gradient_kernel(
        "add_backward",
        &inputs,
        grad_output.shape().dims()
    ).await?;
    let first_compile_time = start.elapsed();
    println!("  Compile time: {:?}", first_compile_time);
    
    println!("\nSecond compilation (cache hit):");
    let start = Instant::now();
    let _compiled_kernel2 = jit_context.compile_gradient_kernel(
        "add_backward",
        &inputs,
        grad_output.shape().dims()
    ).await?;
    let second_compile_time = start.elapsed();
    println!("  Compile time: {:?}", second_compile_time);
    
    let speedup = first_compile_time.as_nanos() as f64 / second_compile_time.as_nanos() as f64;
    println!("\nCache speedup: {:.1}x faster", speedup);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jit_example() -> Result<()> {
        // Run a basic test version
        jit_utils::initialize_jit().await?;
        basic_jit_example().await?;
        Ok(())
    }
}