//! CPU Performance Optimization Demo
//!
//! This example demonstrates the performance improvements achieved through
//! optimized CPU binary operations compared to the baseline implementation.

use tenflowers_core::ops::performance_benchmark::run_performance_benchmark;
use tenflowers_core::{ops::binary, ops::optimized_binary, Result, Shape, Tensor};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS CPU Performance Optimization Demo");
    println!("============================================\n");

    // Run comprehensive benchmark
    let _results = run_performance_benchmark()?;

    // Demonstrate specific optimization features
    demonstrate_vectorization()?;
    demonstrate_scalar_broadcast()?;
    demonstrate_parallel_processing()?;

    Ok(())
}

fn demonstrate_vectorization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Vectorization Demonstration");
    println!("------------------------------");

    // Create medium-sized arrays to show vectorization benefits
    let size = 50000;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
    let b_data: Vec<f32> = (0..size).map(|i| i as f32 + 2.0).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    println!("Array size: {} elements", size);

    // Time original implementation
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _result = binary::add(&a, &b)?;
    }
    let original_time = start.elapsed();

    // Time optimized implementation
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _result = optimized_binary::optimized_add(&a, &b)?;
    }
    let optimized_time = start.elapsed();

    let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!("Original time:  {:?}", original_time);
    println!("Optimized time: {:?}", optimized_time);
    println!("Speedup:        {:.2}x", speedup);

    // Verify correctness
    let orig_result = binary::add(&a, &b)?;
    let opt_result = optimized_binary::optimized_add(&a, &b)?;

    let orig_data = orig_result.to_vec()?;
    let opt_data = opt_result.to_vec()?;

    let is_correct = orig_data
        .iter()
        .zip(opt_data.iter())
        .all(|(o, p)| (o - p).abs() < 1e-6);

    println!(
        "Correctness:    {}",
        if is_correct {
            "✅ Verified"
        } else {
            "❌ Failed"
        }
    );

    Ok(())
}

fn demonstrate_scalar_broadcast() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n📡 Scalar Broadcasting Optimization");
    println!("-----------------------------------");

    // Create scalar and large array for broadcasting
    let scalar = Tensor::from_vec(vec![3.14f32], &[1])?;
    let array_size = 100000;
    let array_data: Vec<f32> = (0..array_size).map(|i| i as f32).collect();
    let array = Tensor::from_vec(array_data, &[array_size])?;

    println!("Scalar: [3.14], Array size: {} elements", array_size);

    // Time original implementation
    let start = std::time::Instant::now();
    for _ in 0..50 {
        let _result = binary::mul(&scalar, &array)?;
    }
    let original_time = start.elapsed();

    // Time optimized implementation (should detect and optimize scalar broadcast)
    let start = std::time::Instant::now();
    for _ in 0..50 {
        let _result = optimized_binary::optimized_mul(&scalar, &array)?;
    }
    let optimized_time = start.elapsed();

    let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!("Original time:  {:?}", original_time);
    println!("Optimized time: {:?}", optimized_time);
    println!("Speedup:        {:.2}x", speedup);

    // Verify a few results
    let orig_result = binary::mul(&scalar, &array)?;
    let opt_result = optimized_binary::optimized_mul(&scalar, &array)?;

    let orig_data = orig_result.to_vec()?;
    let opt_data = opt_result.to_vec()?;

    let sample_correct = orig_data[..10]
        .iter()
        .zip(opt_data[..10].iter())
        .all(|(o, p)| (o - p).abs() < 1e-6);

    println!(
        "Sample check:   {}",
        if sample_correct {
            "✅ Verified"
        } else {
            "❌ Failed"
        }
    );
    println!("First few results: {:?}", &opt_data[..5]);

    Ok(())
}

fn demonstrate_parallel_processing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Parallel Processing Demonstration");
    println!("------------------------------------");

    // Create very large arrays to trigger parallel processing
    let size = 1_000_000;
    println!("Creating large arrays with {} elements each...", size);

    let a_data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    println!(
        "Array size: {} elements (should trigger parallel processing)",
        size
    );

    // Test multiple operations to see parallel benefits
    let operations: Vec<(
        &str,
        Box<dyn Fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>>>,
        Box<dyn Fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>>>,
    )> = vec![
        (
            "Addition",
            Box::new(|a: &Tensor<f32>, b: &Tensor<f32>| binary::add(a, b)),
            Box::new(|a: &Tensor<f32>, b: &Tensor<f32>| optimized_binary::optimized_add(a, b)),
        ),
        (
            "Multiplication",
            Box::new(|a: &Tensor<f32>, b: &Tensor<f32>| binary::mul(a, b)),
            Box::new(|a: &Tensor<f32>, b: &Tensor<f32>| optimized_binary::optimized_mul(a, b)),
        ),
        (
            "Subtraction",
            Box::new(|a: &Tensor<f32>, b: &Tensor<f32>| binary::sub(a, b)),
            Box::new(|a: &Tensor<f32>, b: &Tensor<f32>| optimized_binary::optimized_sub(a, b)),
        ),
    ];

    for (name, orig_fn, opt_fn) in operations {
        println!("\nTesting {}:", name);

        // Time original
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _result = orig_fn(&a, &b)?;
        }
        let original_time = start.elapsed();

        // Time optimized
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _result = opt_fn(&a, &b)?;
        }
        let optimized_time = start.elapsed();

        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        let throughput_opt = (size as f64 * 10.0) / optimized_time.as_secs_f64();

        println!("  Original:   {:?}", original_time);
        println!("  Optimized:  {:?}", optimized_time);
        println!("  Speedup:    {:.2}x", speedup);
        println!("  Throughput: {:.1e} elements/sec", throughput_opt);
    }

    Ok(())
}
