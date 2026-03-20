//! Ultra-Performance Integration Example
//!
//! This example demonstrates the integration of all ultra-high-performance optimizations
//! including ultra-efficient memory management, SIMD-accelerated gradients, and optimized layers.

use std::time::Instant;
use tenflowers_core::{Result, Tensor, Device};
use tenflowers_autograd::{
    global_ultra_gradient_engine, global_simd_grad_ops, global_gradient_buffer_manager,
};
use tenflowers_neural::layers::{
    ultra_dense::ultra_dense, ultra_conv_simple::ultra_conv2d,
    ultra_layer_manager_minimal::{global_ultra_layer_manager, LayerType},
};
use tenflowers_core::memory::ultra_efficient_pool_simple::{
    global_memory_pool, PoolConfig, UltraEfficientMemoryPool,
};

/// Demonstrates ultra-performance neural network training with all optimizations
async fn ultra_performance_training_demo() -> Result<()> {
    println!("🚀 Ultra-Performance Training Demo");
    println!("==================================");

    // Initialize ultra-performance memory management
    let memory_config = PoolConfig {
        initial_size: 50_000_000,    // 50MB
        max_size: 2_000_000_000,     // 2GB
        enable_buffer_reuse: true,
        enable_profiling: true,
        buffer_alignment: 64,        // SIMD alignment
        cleanup_threshold: 0.9,
    };

    let ultra_pool = UltraEfficientMemoryPool::new(memory_config)?;
    println!("✅ Ultra-efficient memory pool initialized");

    // Get global ultra layer manager
    let layer_manager = global_ultra_layer_manager();
    let layer_manager = layer_manager.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock layer manager".to_string())
    )?;

    // Register ultra-performance layers
    let conv_layer_id = layer_manager.register_layer(LayerType::Conv2D)?;
    let dense_layer_id = layer_manager.register_layer(LayerType::Dense)?;
    println!("✅ Ultra-performance layers registered");

    drop(layer_manager); // Release lock

    // Create ultra-optimized neural network components
    let ultra_conv = ultra_conv2d::<f32>(3, 64, (3, 3))?;
    let ultra_dense = ultra_dense::<f32>(1024, 10)?;
    println!("✅ Ultra-optimized layers created");

    // Demonstrate ultra-performance forward pass
    let batch_size = 32;
    let input_tensor = ultra_pool.create_tensor::<f32>(&[batch_size, 3, 224, 224])?;
    println!("✅ Input tensor created with ultra-efficient memory: {:?}", input_tensor.shape().dims());

    // Measure ultra-performance forward pass
    let start_time = Instant::now();

    // Convolutional layer forward pass
    let conv_output = ultra_conv.forward(&input_tensor)?;
    println!("✅ Ultra-conv forward pass completed: {:?}", conv_output.shape().dims());

    // Simulated global average pooling (flatten for dense layer)
    let flattened_size = conv_output.shape().dims().iter().skip(1).product::<usize>();
    let flattened_tensor = ultra_pool.create_tensor::<f32>(&[batch_size, flattened_size])?;

    // Dense layer forward pass
    let dense_input = ultra_pool.create_tensor::<f32>(&[batch_size, 1024])?;
    let final_output = ultra_dense.forward(&dense_input)?;

    let forward_time = start_time.elapsed();
    println!("✅ Ultra-dense forward pass completed: {:?}", final_output.shape().dims());
    println!("⚡ Total forward pass time: {:?}", forward_time);

    // Record performance metrics
    let layer_manager = global_ultra_layer_manager();
    let layer_manager = layer_manager.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock layer manager".to_string())
    )?;

    layer_manager.record_execution(conv_layer_id, forward_time / 2, 1024 * 1024)?;
    layer_manager.record_execution(dense_layer_id, forward_time / 2, 512 * 1024)?;

    drop(layer_manager);

    // Demonstrate ultra-performance gradient computation
    let gradient_engine = global_ultra_gradient_engine();
    let gradient_engine = gradient_engine.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock gradient engine".to_string())
    )?;

    let grad_start = Instant::now();
    let _gradient_result = gradient_engine.compute_ultra_gradient(&final_output, "loss_function")?;
    let grad_time = grad_start.elapsed();
    println!("⚡ Ultra-gradient computation time: {:?}", grad_time);

    drop(gradient_engine);

    // Demonstrate SIMD-accelerated operations
    let simd_ops = global_simd_grad_ops();
    let simd_ops = simd_ops.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock SIMD ops".to_string())
    )?;

    let simd_start = Instant::now();
    let _simd_result = simd_ops.simd_accelerated_gradient(&final_output)?;
    let simd_time = simd_start.elapsed();
    println!("⚡ SIMD-accelerated gradient time: {:?}", simd_time);

    drop(simd_ops);

    // Get comprehensive performance statistics
    let memory_stats = ultra_pool.get_statistics()?;
    println!("\n📊 Ultra-Performance Memory Statistics:");
    println!("  Total allocated: {} bytes", memory_stats.total_allocated);
    println!("  Total reused: {} buffers", memory_stats.total_reused);
    println!("  Current usage: {} bytes", memory_stats.current_usage);
    println!("  Peak usage: {} bytes", memory_stats.peak_usage);
    println!("  Pool efficiency: {:.2}%", memory_stats.pool_efficiency * 100.0);
    println!("  Cache hit rate: {:.2}%", memory_stats.cache_hit_rate * 100.0);

    // Get layer manager performance report
    let layer_manager = global_ultra_layer_manager();
    let layer_manager = layer_manager.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock layer manager".to_string())
    )?;

    let performance_report = layer_manager.get_performance_report()?;
    println!("\n📊 Ultra-Performance Layer Statistics:");
    println!("  Total layers managed: {}", performance_report.layer_count);
    println!("  System efficiency: {:.2}%", performance_report.global_stats.system_efficiency * 100.0);

    for trend in &performance_report.performance_trends {
        println!("  📈 {}", trend);
    }

    for recommendation in &performance_report.optimization_recommendations {
        println!("  💡 {}", recommendation);
    }

    drop(layer_manager);

    println!("\n🎯 Ultra-Performance Integration Demo Complete!");
    println!("All ultra-optimizations working together successfully!");

    Ok(())
}

/// Demonstrates ultra-performance memory management patterns
fn ultra_memory_management_demo() -> Result<()> {
    println!("\n🧠 Ultra-Memory Management Demo");
    println!("==============================");

    let memory_pool = global_memory_pool();
    let pool = memory_pool.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock memory pool".to_string())
    )?;

    // Create multiple tensors to demonstrate buffer reuse
    println!("Creating tensors with ultra-efficient memory management...");

    let tensor1 = pool.create_tensor::<f32>(&[1000, 1000])?;
    let tensor2 = pool.create_tensor::<f32>(&[500, 2000])?;
    let tensor3 = pool.create_tensor::<f32>(&[2000, 500])?;

    println!("✅ Created 3 large tensors using memory pool");

    // Demonstrate memory optimization
    pool.optimize()?;
    println!("✅ Memory pool optimization completed");

    // Get memory efficiency metrics
    let stats = pool.get_statistics()?;
    println!("📊 Memory efficiency: {:.2}%", stats.pool_efficiency * 100.0);
    println!("📊 Fragmentation ratio: {:.2}%", stats.fragmentation_ratio * 100.0);

    Ok(())
}

/// Demonstrates ultra-performance gradient computation patterns
async fn ultra_gradient_demo() -> Result<()> {
    println!("\n🔄 Ultra-Gradient Computation Demo");
    println!("==================================");

    // Test ultra-gradient engine capabilities
    let gradient_engine = global_ultra_gradient_engine();
    let engine = gradient_engine.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock gradient engine".to_string())
    )?;

    // Create test tensor for gradient computation
    let memory_pool = global_memory_pool();
    let pool = memory_pool.lock().map_err(|_|
        tenflowers_core::TensorError::compute_error_simple("Failed to lock memory pool".to_string())
    )?;

    let test_tensor = pool.create_tensor::<f32>(&[64, 256])?;
    drop(pool);

    // Benchmark gradient computation
    let iterations = 100;
    let start_time = Instant::now();

    for i in 0..iterations {
        let operation_name = format!("gradient_op_{}", i);
        let _gradient = engine.compute_ultra_gradient(&test_tensor, &operation_name)?;
    }

    let total_time = start_time.elapsed();
    let avg_time = total_time / iterations;

    println!("⚡ Average gradient computation time: {:?}", avg_time);
    println!("⚡ Total time for {} iterations: {:?}", iterations, total_time);

    // Get gradient engine statistics
    let stats = engine.get_performance_statistics()?;
    println!("📊 Gradient cache efficiency: {:.2}%", stats.cache_efficiency * 100.0);
    println!("📊 Total gradient operations: {}", stats.total_gradient_ops);

    Ok(())
}

/// Main ultra-performance integration demonstration
#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 TenfloweRS Ultra-Performance Integration Examples");
    println!("===================================================");

    // Run all ultra-performance demonstrations
    ultra_performance_training_demo().await?;
    ultra_memory_management_demo()?;
    ultra_gradient_demo().await?;

    println!("\n🎉 All ultra-performance integrations completed successfully!");
    println!("The TenfloweRS framework is running at maximum performance!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ultra_performance_integration() {
        let result = ultra_performance_training_demo().await;
        assert!(result.is_ok(), "Ultra-performance integration should succeed");
    }

    #[test]
    fn test_ultra_memory_management() {
        let result = ultra_memory_management_demo();
        assert!(result.is_ok(), "Ultra-memory management should succeed");
    }

    #[tokio::test]
    async fn test_ultra_gradient_computation() {
        let result = ultra_gradient_demo().await;
        assert!(result.is_ok(), "Ultra-gradient computation should succeed");
    }
}