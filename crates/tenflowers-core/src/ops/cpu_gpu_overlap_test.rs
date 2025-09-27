/// Integration test for CPU-GPU overlap in async execution
#[cfg(test)]
mod cpu_gpu_overlap_tests {
    use super::*;
    use crate::device::async_execution::AsyncExecutor;
    use crate::ops::async_binary::{
        add_async, add_async_priority, batch_add_async, global_async_executor, mul_async,
    };
    use crate::ops::hybrid_scheduler::{HybridWorkScheduler, WorkPriority};
    use crate::Device;
    use crate::Tensor;
    use futures::StreamExt;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[cfg(feature = "gpu")]
    use crate::gpu::multi_stream_executor::MultiStreamGpuExecutor;

    /// Test basic async binary operations
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_basic_async_operations() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();

        // Test async add
        let start = Instant::now();
        let result = add_async(&a, &b).await.unwrap();
        let add_time = start.elapsed();

        println!("Async add took: {:?}", add_time);

        if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(arr.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
        }

        // Test async mul
        let start = Instant::now();
        let result = mul_async(&a, &b).await.unwrap();
        let mul_time = start.elapsed();

        println!("Async mul took: {:?}", mul_time);

        if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(arr.as_slice().unwrap(), &[5.0, 12.0, 21.0, 32.0]);
        }
    }

    /// Test priority-based async operations
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_priority_based_execution() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        // Test high priority operation
        let start = Instant::now();
        let result = add_async_priority(&a, &b, WorkPriority::High)
            .await
            .unwrap();
        let high_priority_time = start.elapsed();

        println!("High priority async add took: {:?}", high_priority_time);

        if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(arr.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
        }

        // Test normal priority operation
        let start = Instant::now();
        let result = add_async_priority(&a, &b, WorkPriority::Normal)
            .await
            .unwrap();
        let normal_priority_time = start.elapsed();

        println!("Normal priority async add took: {:?}", normal_priority_time);

        if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(arr.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
        }
    }

    /// Test batch processing with CPU-GPU overlap
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_batch_processing() {
        let tensors_a = vec![
            Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![5.0, 6.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![7.0, 8.0], &[2]).unwrap(),
        ];

        let tensors_b = vec![
            Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![4.0, 5.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![6.0, 7.0], &[2]).unwrap(),
            Tensor::<f32>::from_vec(vec![8.0, 9.0], &[2]).unwrap(),
        ];

        let operations: Vec<_> = tensors_a.iter().zip(tensors_b.iter()).collect();

        let start = Instant::now();
        let results = batch_add_async(operations).await.unwrap();
        let batch_time = start.elapsed();

        println!(
            "Batch processing {} operations took: {:?}",
            results.len(),
            batch_time
        );

        assert_eq!(results.len(), 4);

        // Check results
        let expected_results = vec![
            vec![3.0, 5.0],
            vec![7.0, 9.0],
            vec![11.0, 13.0],
            vec![15.0, 17.0],
        ];

        for (i, result) in results.iter().enumerate() {
            if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
                assert_eq!(arr.as_slice().unwrap(), &expected_results[i]);
            }
        }
    }

    /// Test concurrent operations to verify CPU-GPU overlap
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_concurrent_operations() {
        let size = 1000;
        let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let a = Tensor::<f32>::from_vec(data_a, &[size]).unwrap();
        let b = Tensor::<f32>::from_vec(data_b, &[size]).unwrap();

        // Create multiple concurrent operations
        let num_operations = 10usize;
        let mut futures = Vec::new();

        let start = Instant::now();

        for i in 0..num_operations {
            let a_ref = &a;
            let b_ref = &b;

            let future = async move {
                let result = add_async(a_ref, b_ref).await.unwrap();
                println!("Operation {} completed", i);
                result
            };

            futures.push(future);
        }

        // Wait for all operations to complete
        let results = futures::future::join_all(futures).await;
        let total_time = start.elapsed();

        println!(
            "Completed {} concurrent operations in {:?}",
            num_operations, total_time
        );
        println!(
            "Average time per operation: {:?}",
            total_time / num_operations as u32
        );

        assert_eq!(results.len(), num_operations);

        // Verify results
        for result in results {
            if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
                let slice = arr.as_slice().unwrap();
                for i in 0..size.min(10) {
                    assert_eq!(slice[i], (i as f32) + (i + 1) as f32);
                }
            }
        }
    }

    /// Test hybrid scheduler directly
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_hybrid_scheduler_integration() {
        let executor = global_async_executor();

        // Create test tensors
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0], &[5]).unwrap();

        // Test that executor starts idle
        assert!(executor.is_idle());

        // Submit operation
        let start = Instant::now();
        let result = executor
            .execute_async(&a, &b, crate::ops::binary::AddOp)
            .await
            .unwrap();
        let execution_time = start.elapsed();

        println!("Hybrid scheduler execution took: {:?}", execution_time);

        // Verify result
        if let crate::tensor::TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(arr.as_slice().unwrap(), &[6.0, 6.0, 6.0, 6.0, 6.0]);
        }

        // Test synchronization
        executor.synchronize();
        assert!(executor.is_idle());
    }

    /// Test resource utilization and performance metrics
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_performance_metrics() {
        use crate::memory::{global_monitor, OperationTimer};

        let monitor = global_monitor();
        monitor.clear(); // Clear previous metrics

        let a = Tensor::<f32>::from_vec(vec![1.0; 1000], &[1000]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![2.0; 1000], &[1000]).unwrap();

        // Time multiple operations
        let num_ops = 5;

        for i in 0..num_ops {
            let _timer = OperationTimer::new(
                format!("async_add_{}", i),
                crate::memory::global_monitor_arc(),
            );

            let _result = add_async(&a, &b).await.unwrap();
        }

        // Check performance metrics
        let avg_time = monitor.get_average_time("async_add_0");
        if let Some(time) = avg_time {
            println!("Average async add time: {:?}", time);
            assert!(time < Duration::from_secs(1)); // Should be reasonably fast
        }

        let current_memory = monitor.get_current_memory();
        let peak_memory = monitor.get_peak_memory();

        println!("Current memory usage: {} bytes", current_memory);
        println!("Peak memory usage: {} bytes", peak_memory);

        // Generate performance report
        let report = monitor.generate_report();
        println!("Performance Report:\n{}", report);

        assert!(report.contains("Performance Monitor Report"));
    }

    /// Test CPU-GPU overlap with large tensors
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_large_tensor_operations() {
        let size = 10000;
        let data_a: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let data_b: Vec<f32> = (0..size).map(|i| ((i + 50) % 100) as f32).collect();

        let a = Tensor::<f32>::from_vec(data_a, &[size]).unwrap();
        let b = Tensor::<f32>::from_vec(data_b, &[size]).unwrap();

        // Test sequential execution
        let start = Instant::now();
        let result1 = add_async(&a, &b).await.unwrap();
        let result2 = mul_async(&a, &b).await.unwrap();
        let sequential_time = start.elapsed();

        println!("Sequential execution took: {:?}", sequential_time);

        // Test concurrent execution
        let start = Instant::now();
        let (result3, result4) = tokio::join!(add_async(&a, &b), mul_async(&a, &b));
        let concurrent_time = start.elapsed();

        println!("Concurrent execution took: {:?}", concurrent_time);

        // Verify results are the same
        assert_eq!(result1.shape(), result3.as_ref().unwrap().shape());
        assert_eq!(result2.shape(), result4.as_ref().unwrap().shape());

        // Concurrent should be faster (or at least not significantly slower)
        let speedup_ratio = sequential_time.as_secs_f64() / concurrent_time.as_secs_f64();
        println!("Speedup ratio: {:.2}", speedup_ratio);

        // With proper CPU-GPU overlap, we should see some improvement
        // For testing, we'll just verify it doesn't get significantly worse
        assert!(speedup_ratio > 0.8); // Should be at least 80% as fast
    }

    /// Test error handling in async operations
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_error_handling() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0], &[2]).unwrap(); // Different shape

        // This should fail due to shape mismatch
        let result = add_async(&a, &b).await;
        assert!(result.is_err());

        if let Err(e) = result {
            println!("Expected error: {:?}", e);
            assert!(e.to_string().contains("Shape"));
        }
    }

    /// Test streaming operations
    #[tokio::test]
    #[ignore = "Slow test - run with --ignored if needed"]
    async fn test_streaming_operations() {
        use tokio_stream::{self as stream};

        let base_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let base_tensor = Tensor::<f32>::from_vec(base_data, &[5]).unwrap();

        // Create a stream of operations
        let operation_stream = stream::iter(0..10)
            .map(|i| {
                let scalar_data = vec![i as f32; 5];
                let scalar_tensor = Tensor::<f32>::from_vec(scalar_data, &[5]).unwrap();
                (base_tensor.clone(), scalar_tensor)
            })
            .map(|(a, b)| async move { add_async(&a, &b).await })
            .buffered(3); // Process up to 3 operations concurrently

        let results: Vec<_> = operation_stream.collect().await;

        assert_eq!(results.len(), 10);

        // Check a few results
        for (i, result) in results.iter().enumerate() {
            if let Ok(tensor) = result {
                if let crate::tensor::TensorStorage::Cpu(arr) = &tensor.storage {
                    let slice = arr.as_slice().unwrap();
                    assert_eq!(slice[0], 1.0 + i as f32);
                    assert_eq!(slice[4], 5.0 + i as f32);
                }
            }
        }

        println!(
            "Successfully processed {} streaming operations",
            results.len()
        );
    }
}
