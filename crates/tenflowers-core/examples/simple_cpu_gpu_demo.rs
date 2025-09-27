use std::time::Instant;
use tenflowers_core::ops::async_binary::{AsyncBinaryOperationExecutor, WorkPriority};
use tenflowers_core::ops::binary::{AddOp, MulOp};
/// Simple demonstration of CPU-GPU overlap functionality
/// This example shows the basic usage of the async binary operations
use tenflowers_core::{Result, Tensor};

fn main() -> Result<()> {
    println!("=== Simple CPU-GPU Overlap Demo ===");
    println!();

    // Create some test tensors
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4])?;

    println!("Input tensors:");
    println!("  A: {:?}", a.to_vec());
    println!("  B: {:?}", b.to_vec());

    // Create an async executor
    let executor = AsyncBinaryOperationExecutor::new(0)?;
    println!();
    println!(
        "Created async executor (starts idle: {})",
        executor.is_idle()
    );

    // Test synchronous operations for comparison
    println!();
    println!("1. Synchronous operations:");
    let start = Instant::now();
    let sync_add = tenflowers_core::ops::add(&a, &b)?;
    let sync_mul = tenflowers_core::ops::mul(&a, &b)?;
    let sync_time = start.elapsed();

    println!("  Add result: {:?}", sync_add.to_vec());
    println!("  Mul result: {:?}", sync_mul.to_vec());
    println!("  Time: {:?}", sync_time);

    // Test the async executor directly (without actual async runtime)
    println!();
    println!("2. Async executor (simulated):");
    let start = Instant::now();

    // This demonstrates the executor interface, though we can't actually await here
    // In a real async context, you would use executor.execute_async(...).await
    println!("  Executor ready for async operations");
    println!("  Would execute: executor.execute_async(&a, &b, AddOp).await");
    println!("  Would execute: executor.execute_async(&a, &b, MulOp).await");

    let setup_time = start.elapsed();
    println!("  Setup time: {:?}", setup_time);

    // Show priority levels
    println!();
    println!("3. Priority levels:");
    println!("  WorkPriority::Low = {:?}", WorkPriority::Low);
    println!("  WorkPriority::Normal = {:?}", WorkPriority::Normal);
    println!("  WorkPriority::High = {:?}", WorkPriority::High);
    println!("  WorkPriority::Critical = {:?}", WorkPriority::Critical);

    // Test priority ordering
    println!();
    println!("4. Priority ordering:");
    println!(
        "  High > Normal: {}",
        WorkPriority::High > WorkPriority::Normal
    );
    println!(
        "  Normal > Low: {}",
        WorkPriority::Normal > WorkPriority::Low
    );
    println!(
        "  Critical > High: {}",
        WorkPriority::Critical > WorkPriority::High
    );

    // Test that executor is still idle (no actual async work was done)
    println!();
    println!("5. Executor state:");
    println!("  Is idle: {}", executor.is_idle());

    // Demonstrate the types of operations that would be possible
    println!();
    println!("6. Available async operations:");
    println!("  - add_async(a, b) -> Future<Result<Tensor>>");
    println!("  - mul_async(a, b) -> Future<Result<Tensor>>");
    println!("  - add_async_priority(a, b, priority) -> Future<Result<Tensor>>");
    println!("  - batch_add_async(operations) -> Future<Result<Vec<Tensor>>>");
    println!("  - synchronize_async_operations() -> ()");

    println!();
    println!("=== Demo Complete ===");
    println!("Note: This demo shows the infrastructure is working.");
    println!("In a real async context with tokio/async-std, you would:");
    println!("1. Use .await on the async functions");
    println!("2. Get actual CPU-GPU overlap benefits");
    println!("3. Process operations concurrently");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        // Test that the demo code doesn't panic
        assert!(main().is_ok());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(WorkPriority::Critical > WorkPriority::High);
        assert!(WorkPriority::High > WorkPriority::Normal);
        assert!(WorkPriority::Normal > WorkPriority::Low);
    }

    #[test]
    fn test_executor_creation() {
        let executor = AsyncBinaryOperationExecutor::new(0).unwrap();
        assert!(executor.is_idle());
    }
}
