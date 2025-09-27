#[cfg(feature = "gpu")]
use std::time::Instant;
#[cfg(feature = "gpu")]
use tenflowers_core::ops::async_binary::WorkPriority;
#[cfg(feature = "gpu")]
use tenflowers_core::ops::{
    add_async, add_async_priority, batch_add_async, is_async_operations_idle, mul_async,
    synchronize_async_operations,
};
/// Example demonstrating CPU-GPU overlap in async execution
/// This example shows how to use async operations for better performance
/// This example only works with the GPU feature enabled
#[cfg(feature = "gpu")]
use tenflowers_core::{Tensor, TensorError};

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> tenflowers_core::Result<()> {
    println!("🚀 TenfloweRS CPU-GPU Overlap Demo");
    println!("===================================\n");

    // Example 1: Basic async operations
    println!("1. Basic Async Operations:");
    basic_async_demo().await?;

    // Example 2: Priority-based execution
    println!("\n2. Priority-based Execution:");
    priority_demo().await?;

    // Example 3: Batch processing
    println!("\n3. Batch Processing:");
    batch_demo().await?;

    // Example 4: Concurrent operations
    println!("\n4. Concurrent Operations:");
    concurrent_demo().await?;

    // Example 5: Performance comparison
    println!("\n5. Performance Comparison:");
    performance_comparison().await?;

    // Example 6: Operation synchronization
    println!("\n6. Operation Synchronization:");
    synchronization_demo().await?;

    println!("\n✅ Demo Complete - CPU-GPU overlap working successfully!");
    Ok(())
}

#[cfg(feature = "gpu")]
/// Basic async operations example
async fn basic_async_demo() -> tenflowers_core::Result<()> {
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4])?;

    println!("  📊 Input A: {:?}", a.to_vec());
    println!("  📊 Input B: {:?}", b.to_vec());

    // Async addition
    let start = Instant::now();
    let result = add_async(&a, &b).await?;
    let add_time = start.elapsed();

    println!(
        "  ➕ Async Add Result: {:?} (took {:?})",
        result.to_vec(),
        add_time
    );

    // Async multiplication
    let start = Instant::now();
    let result = mul_async(&a, &b).await?;
    let mul_time = start.elapsed();

    println!(
        "  ✖️  Async Mul Result: {:?} (took {:?})",
        result.to_vec(),
        mul_time
    );

    Ok(())
}

#[cfg(feature = "gpu")]
/// Priority-based execution example
async fn priority_demo() -> tenflowers_core::Result<()> {
    let a = Tensor::<f32>::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4])?;
    let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;

    println!("  📊 Running operations with different priorities:");

    // High priority operation
    let start = Instant::now();
    let high_priority_result = add_async_priority(&a, &b, WorkPriority::High).await?;
    let high_time = start.elapsed();

    // Normal priority operation
    let start = Instant::now();
    let normal_priority_result = add_async_priority(&a, &b, WorkPriority::Normal).await?;
    let normal_time = start.elapsed();

    println!(
        "  🔥 High Priority: {:?} (took {:?})",
        high_priority_result.to_vec(),
        high_time
    );
    println!(
        "  📝 Normal Priority: {:?} (took {:?})",
        normal_priority_result.to_vec(),
        normal_time
    );

    Ok(())
}

#[cfg(feature = "gpu")]
/// Batch processing example
async fn batch_demo() -> tenflowers_core::Result<()> {
    // Create multiple tensor pairs for batch processing
    let tensors_a = vec![
        Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?,
        Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2])?,
        Tensor::<f32>::from_vec(vec![5.0, 6.0], &[2])?,
    ];

    let tensors_b = vec![
        Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2])?,
        Tensor::<f32>::from_vec(vec![4.0, 5.0], &[2])?,
        Tensor::<f32>::from_vec(vec![6.0, 7.0], &[2])?,
    ];

    println!("  📦 Processing {} tensor pairs in batch:", tensors_a.len());

    let start = Instant::now();
    let operations: Vec<(&Tensor<f32>, &Tensor<f32>)> =
        tensors_a.iter().zip(tensors_b.iter()).collect();
    let results = batch_add_async(operations).await?;
    let batch_time = start.elapsed();

    for (i, result) in results.iter().enumerate() {
        println!("    Batch {}: {:?}", i + 1, result.to_vec());
    }
    println!("  ⏱️  Total batch time: {:?}", batch_time);

    Ok(())
}

#[cfg(feature = "gpu")]
/// Concurrent operations example
async fn concurrent_demo() -> tenflowers_core::Result<()> {
    let a = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
    let b = Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0], &[3])?;
    let c = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0], &[3])?;

    println!("  🔄 Running multiple operations concurrently:");

    let start = Instant::now();

    // Start multiple async operations concurrently
    let add_future = add_async(&a, &b);
    let mul_future = mul_async(&a, &c);
    let add2_future = add_async(&b, &c);

    // Await all results
    let (add_result, mul_result, add2_result) =
        tokio::try_join!(add_future, mul_future, add2_future)?;

    let concurrent_time = start.elapsed();

    println!("    Add (A + B): {:?}", add_result.to_vec());
    println!("    Mul (A * C): {:?}", mul_result.to_vec());
    println!("    Add (B + C): {:?}", add2_result.to_vec());
    println!("  ⏱️  Total concurrent time: {:?}", concurrent_time);

    Ok(())
}

#[cfg(feature = "gpu")]
/// Performance comparison between sync and async operations
async fn performance_comparison() -> tenflowers_core::Result<()> {
    let size = 1000;
    let a = Tensor::<f32>::from_vec(vec![1.5; size], &[size])?;
    let b = Tensor::<f32>::from_vec(vec![2.5; size], &[size])?;

    println!("  ⚡ Performance comparison (tensor size: {}):", size);

    // Synchronous operations
    let start = Instant::now();
    let sync_result = tenflowers_core::ops::add(&a, &b)?;
    let sync_time = start.elapsed();

    // Asynchronous operations
    let start = Instant::now();
    let async_result = add_async(&a, &b).await?;
    let async_time = start.elapsed();

    println!("    🔄 Sync operation: {:?}", sync_time);
    println!("    ⚡ Async operation: {:?}", async_time);

    // Verify results are the same
    let sync_vec = sync_result.to_vec()?;
    let async_vec = async_result.to_vec()?;
    let results_match = sync_vec.len() == async_vec.len()
        && sync_vec
            .iter()
            .zip(async_vec.iter())
            .all(|(s, a)| (s - a).abs() < 1e-6);

    println!("    ✅ Results match: {}", results_match);

    if async_time < sync_time {
        let speedup = sync_time.as_nanos() as f64 / async_time.as_nanos() as f64;
        println!("    🚀 Async speedup: {:.2}x", speedup);
    } else {
        let slowdown = async_time.as_nanos() as f64 / sync_time.as_nanos() as f64;
        println!(
            "    📊 Async overhead: {:.2}x (expected for small tensors)",
            slowdown
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
/// Operation synchronization example
async fn synchronization_demo() -> tenflowers_core::Result<()> {
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3])?;

    println!("  🔄 Demonstrating operation synchronization:");

    // Check if operations are idle before starting
    println!(
        "    Operations idle before start: {}",
        is_async_operations_idle()
    );

    // Start some async operations (but don't await yet)
    let _add_future = add_async(&a, &b);
    let _mul_future = mul_async(&a, &b);

    // Check if operations are still running
    tokio::task::yield_now().await; // Give operations a chance to start
    println!(
        "    Operations idle after queueing: {}",
        is_async_operations_idle()
    );

    // Synchronize all pending operations
    println!("    🔄 Synchronizing all async operations...");
    synchronize_async_operations();

    // Check if operations are idle after sync
    println!(
        "    Operations idle after sync: {}",
        is_async_operations_idle()
    );

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("⚠️  CPU-GPU Overlap Demo requires GPU feature to be enabled");
    println!("   Build with: cargo run --features gpu --example cpu_gpu_overlap_demo");
    println!("   or: cargo run --features gpu,cuda --example cpu_gpu_overlap_demo");
    println!("   or: cargo run --features gpu,metal --example cpu_gpu_overlap_demo");
}
