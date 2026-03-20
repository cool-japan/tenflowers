use tenflowers_core::{
    Device, Tensor, DType,
    ops::benchmark::{BenchmarkSuite, BenchmarkConfig, benchmark_binary_op, benchmark_unary_op, benchmark_matmul_sizes},
};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 TenfloweRS Performance Benchmark Suite");
    println!("=========================================");
    
    // Create benchmark configurations for different scenarios
    let quick_config = BenchmarkConfig {
        warmup_iterations: 3,
        measurement_iterations: 10,
        measure_memory: true,
        calculate_flops: true,
        min_execution_time: Duration::from_micros(1),
        max_execution_time: Duration::from_secs(10),
    };
    
    let comprehensive_config = BenchmarkConfig {
        warmup_iterations: 10,
        measurement_iterations: 50,
        measure_memory: true,
        calculate_flops: true,
        min_execution_time: Duration::from_micros(1),
        max_execution_time: Duration::from_secs(30),
    };
    
    println!("🚀 Running Quick Benchmarks...\n");
    
    // Test devices (include GPU if available)
    let devices = vec![Device::CPU];
    #[cfg(feature = "gpu")]
    let devices = {
        let mut devices = vec![Device::CPU];
        if let Ok(gpu_device) = Device::best_gpu() {
            devices.push(gpu_device);
        }
        devices
    };
    
    println!("📊 Testing devices: {:?}\n", devices);
    
    // 1. Benchmark Binary Operations
    println!("1. Binary Operations Benchmark");
    println!("------------------------------");
    
    let binary_ops = ["Add", "Mul", "Sub", "Div"];
    let test_shapes = vec![
        (vec![1000], vec![1000]),           // Small vectors
        (vec![100, 100], vec![100, 100]),   // Small matrices
        (vec![1000, 1000], vec![1000, 1000]), // Large matrices
    ];
    
    for op_name in &binary_ops {
        println!("Testing {op_name}:");
        
        for (shape_a, shape_b) in &test_shapes {
            println!("  Shape: {:?} × {:?}", shape_a, shape_b);
            
            match benchmark_binary_op::<f32>(op_name, shape_a, shape_b, &devices, &[DType::F32]) {
                Ok(results) => {
                    for result in results {
                        let duration_us = result.duration.as_micros();
                        let throughput = result.throughput
                            .map(|t| format!("{:.2e} elem/s", t))
                            .unwrap_or_else(|| "N/A".to_string());
                        
                        println!("    {:?}: {}μs, {}", result.device, duration_us, throughput);
                    }
                }
                Err(e) => {
                    println!("    Error: {}", e);
                }
            }
        }
        println!();
    }
    
    // 2. Benchmark Unary Operations
    println!("2. Unary Operations Benchmark");
    println!("------------------------------");
    
    let unary_ops = ["ReLU", "Sigmoid", "Tanh"];
    let unary_shapes = vec![
        vec![10000],           // Large vector
        vec![100, 100],        // Square matrix
        vec![32, 32, 32],      // 3D tensor
    ];
    
    for op_name in &unary_ops {
        println!("Testing {op_name}:");
        
        for shape in &unary_shapes {
            println!("  Shape: {:?}", shape);
            
            match benchmark_unary_op::<f32>(op_name, shape, &devices, &[DType::F32]) {
                Ok(results) => {
                    for result in results {
                        let duration_us = result.duration.as_micros();
                        let throughput = result.throughput
                            .map(|t| format!("{:.2e} elem/s", t))
                            .unwrap_or_else(|| "N/A".to_string());
                        
                        println!("    {:?}: {}μs, {}", result.device, duration_us, throughput);
                    }
                }
                Err(e) => {
                    println!("    Error: {}", e);
                }
            }
        }
        println!();
    }
    
    // 3. Matrix Multiplication Benchmark
    println!("3. Matrix Multiplication Benchmark");
    println!("-----------------------------------");
    
    let matmul_sizes = vec![
        (64, 64, 64),       // Small
        (128, 128, 128),    // Medium
        (256, 256, 256),    // Large
        (512, 512, 512),    // Very large
        (1024, 1024, 1024), // Extra large
    ];
    
    match benchmark_matmul_sizes::<f32>(&matmul_sizes, &devices, &[DType::F32]) {
        Ok(results) => {
            println!("Matrix Multiplication Results:");
            println!("| Size (M×K×N) | Device | Duration (μs) | FLOPS | Throughput |");
            println!("|--------------|--------|---------------|-------|------------|");
            
            // Group results by size
            let mut size_groups: std::collections::HashMap<String, Vec<_>> = std::collections::HashMap::new();
            
            for result in results {
                if let Some(ref shapes) = result.input_shapes.get(0) {
                    let key = format!("{}×{}×{}", 
                        shapes.dims().get(0).unwrap_or(&0),
                        shapes.dims().get(1).unwrap_or(&0),
                        result.input_shapes.get(1)
                            .and_then(|s| s.dims().get(1))
                            .unwrap_or(&0)
                    );
                    size_groups.entry(key).or_default().push(result);
                }
            }
            
            for (size, size_results) in size_groups {
                for result in size_results {
                    let duration_us = result.duration.as_micros();
                    let flops = result.flops
                        .map(|f| format!("{:.2e}", f / result.duration.as_secs_f64()))
                        .unwrap_or_else(|| "N/A".to_string());
                    let throughput = result.throughput
                        .map(|t| format!("{:.2e}", t))
                        .unwrap_or_else(|| "N/A".to_string());
                    
                    println!("| {} | {:?} | {} | {} | {} |", 
                        size, result.device, duration_us, flops, throughput);
                }
            }
        }
        Err(e) => {
            println!("MatMul benchmark error: {}", e);
        }
    }
    
    println!("\n4. Comprehensive Benchmarking");
    println!("==============================");
    
    // Create a comprehensive benchmark suite
    let suite = BenchmarkSuite::new(comprehensive_config);
    
    // Test a variety of operations
    test_tensor_operations(&suite)?;
    
    // Generate and display report
    let report = suite.generate_report();
    println!("{}", report);
    
    // Export to JSON if serialize feature is available
    #[cfg(feature = "serialize")]
    {
        match suite.export_json() {
            Ok(json) => {
                std::fs::write("benchmark_results.json", json)?;
                println!("📁 Benchmark results saved to benchmark_results.json");
            }
            Err(e) => {
                println!("Failed to export JSON: {}", e);
            }
        }
    }
    
    println!("\n🎉 Benchmark completed!");
    println!("\n💡 Performance Tips:");
    println!("   - GPU operations are typically faster for large tensors");
    println!("   - Consider data layout (contiguous vs strided) for performance");
    println!("   - Batch operations when possible to amortize overhead");
    println!("   - Use appropriate data types (f32 vs f64) based on precision needs");
    
    Ok(())
}

fn test_tensor_operations(suite: &BenchmarkSuite) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing various tensor operations...");
    
    // Create test tensors
    let tensor_small: Tensor<f32> = Tensor::ones(&[100, 100])?;
    let tensor_large: Tensor<f32> = Tensor::ones(&[1000, 1000])?;
    let vector: Tensor<f32> = Tensor::ones(&[10000])?;
    
    let attrs = std::collections::HashMap::new();
    
    // Test basic arithmetic
    let ops_to_test = vec![
        ("Add", vec![&tensor_small, &tensor_small]),
        ("Mul", vec![&tensor_small, &tensor_small]),
        ("MatMul", vec![&tensor_small, &tensor_small]),
    ];
    
    for (op_name, inputs) in ops_to_test {
        println!("  Testing {op_name}...");
        match suite.benchmark_operation(op_name, &inputs, &attrs) {
            Ok(result) => {
                println!("    ✅ {op_name}: {:.2}μs", result.duration.as_micros());
            }
            Err(e) => {
                println!("    ❌ {op_name}: {}", e);
            }
        }
    }
    
    Ok(())
}

/// Compare TenfloweRS performance with theoretical peaks
fn analyze_performance_characteristics() {
    println!("\n5. Performance Analysis");
    println!("=======================");
    
    println!("Performance Characteristics:");
    println!("- Memory bandwidth bound operations: element-wise ops (Add, Mul, etc.)");
    println!("- Compute bound operations: MatMul, Conv2D");
    println!("- CPU: Good for small tensors, sequential operations");
    println!("- GPU: Good for large tensors, parallel operations");
    
    println!("\nOptimization Opportunities:");
    println!("- Kernel fusion for element-wise operations");
    println!("- Better memory layout for cache efficiency");
    println!("- SIMD vectorization for CPU operations");
    println!("- Async execution for overlapping computation");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_configurations() {
        let config = BenchmarkConfig::default();
        assert!(config.warmup_iterations > 0);
        assert!(config.measurement_iterations > 0);
    }
    
    #[test]
    fn test_quick_benchmark() -> Result<(), Box<dyn std::error::Error>> {
        // Test a simple operation
        let devices = vec![Device::CPU];
        let results = benchmark_binary_op::<f32>("Add", &[10, 10], &[10, 10], &devices, &[DType::F32])?;
        assert!(!results.is_empty());
        Ok(())
    }
}