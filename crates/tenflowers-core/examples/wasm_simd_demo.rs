//! WebAssembly SIMD demonstration
//!
//! This example demonstrates how to use TenfloweRS WebAssembly SIMD optimizations
//! for high-performance tensor operations in browser and edge environments.
//!
//! To build for WebAssembly with SIMD:
//! ```bash
//! # For SIMD-enabled build
//! rustup target add wasm32-unknown-unknown
//! RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --features wasm
//!
//! # For regular WASM build (fallback to scalar)
//! cargo build --target wasm32-unknown-unknown --features wasm
//! ```

use tenflowers_core::{wasm_utils, WasmContext};

fn main() {
    println!("TenfloweRS WebAssembly SIMD Demonstration");
    println!("==========================================");

    // Check if we're running in WASM environment
    if wasm_utils::is_wasm() {
        println!("✓ Running in WebAssembly environment");
        #[cfg(target_arch = "wasm32")]
        run_wasm_demo();
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!("Error: WASM environment detected but not compiled for WASM target");
            return;
        }
    } else {
        println!("ℹ Running in native environment (WASM simulation)");
        run_native_demo();
    }
}

#[cfg(target_arch = "wasm32")]
fn run_wasm_demo() {
    use tenflowers_core::wasm::{WasmFeatures, WasmTensorOps};

    // Detect WASM features
    let features = WasmFeatures::detect();
    println!("\nWASM Feature Detection:");
    println!("  SIMD: {}", features.has_simd());
    println!("  Threads: {}", features.has_threads());
    println!("  Bulk Memory: {}", features.bulk_memory);
    println!("  Reference Types: {}", features.reference_types);

    // Create WASM context
    let mut ctx = WasmContext::new();
    let ops = ctx.ops();

    // Demonstrate SIMD operations
    println!("\nSIMD Operation Benchmarks:");

    // Test different array sizes
    let sizes = vec![16, 64, 256, 1024];

    for size in sizes {
        println!("\n  Array size: {}", size);

        // Create test data
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.2).collect();
        let mut result = vec![0.0; size];

        // Time SIMD addition
        let start_time = ops.now();
        ops.add_simd(&a, &b, &mut result).unwrap();
        let add_time = ops.now() - start_time;

        // Time SIMD multiplication
        let start_time = ops.now();
        ops.mul_simd(&a, &b, &mut result).unwrap();
        let mul_time = ops.now() - start_time;

        // Time ReLU
        let start_time = ops.now();
        ops.relu_simd(&a, &mut result).unwrap();
        let relu_time = ops.now() - start_time;

        println!("    Addition: {:.3}ms", add_time);
        println!("    Multiplication: {:.3}ms", mul_time);
        println!("    ReLU: {:.3}ms", relu_time);

        // Verify correctness (spot check)
        if size >= 4 {
            let expected_add = a[3] + b[3];
            ops.add_simd(&a, &b, &mut result).unwrap();
            let actual_add = result[3];
            println!(
                "    Verification: expected {:.2}, got {:.2} ✓",
                expected_add, actual_add
            );
        }
    }

    // Demonstrate matrix multiplication
    println!("\nMatrix Multiplication Demo:");
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
    let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
    let mut result = vec![0.0; 4];

    let start_time = ops.now();
    ops.matmul_wasm(&a, &b, &mut result, 2, 2, 2).unwrap();
    let matmul_time = ops.now() - start_time;

    println!("  2x2 matrix multiplication: {:.3}ms", matmul_time);
    println!("  Result: {:?}", result);

    // Performance recommendations
    println!("\n🔧 Performance Tips:");
    println!("  • Compile with SIMD128 feature for optimal performance");
    println!("  • Use array sizes that are multiples of 4 for best SIMD utilization");
    println!("  • Consider chunking large operations to stay within WASM memory limits");
    println!("  • Memory limit: {} bytes", ctx.available_memory());
}

#[cfg(not(target_arch = "wasm32"))]
fn run_native_demo() {
    println!("\nWASM SIMD simulation (native environment):");
    println!("  This demonstrates the API without actual WASM execution");

    // Create context (will be a no-op struct for non-WASM)
    let _ctx = WasmContext::new();

    println!(
        "  Optimal chunk size: {} elements",
        wasm_utils::optimal_chunk_size()
    );
    println!(
        "  Recommended memory limit: {} bytes",
        wasm_utils::recommended_memory_limit()
    );

    println!("\n💡 To run with actual WASM SIMD:");
    println!("   1. Build for wasm32-unknown-unknown target");
    println!("   2. Enable simd128 target feature");
    println!("   3. Run in a WebAssembly environment");
}

// Additional helper functions for benchmarking
#[cfg(target_arch = "wasm32")]
fn benchmark_operation<F>(name: &str, iterations: usize, mut operation: F)
where
    F: FnMut(),
{
    use tenflowers_core::wasm::WasmTensorOps;

    let ops = WasmTensorOps::new();
    let start_time = ops.now();

    for _ in 0..iterations {
        operation();
    }

    let total_time = ops.now() - start_time;
    let avg_time = total_time / iterations as f64;

    println!("{}: {:.3}ms total, {:.6}ms avg", name, total_time, avg_time);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_utils() {
        // These functions should work in any environment
        let chunk_size = wasm_utils::optimal_chunk_size();
        assert!(chunk_size > 0);

        let memory_limit = wasm_utils::recommended_memory_limit();
        assert!(memory_limit > 0);

        // is_wasm() should return true only in WASM environment
        let is_wasm = wasm_utils::is_wasm();
        #[cfg(target_arch = "wasm32")]
        assert!(is_wasm);
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!is_wasm);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_operations() {
        use tenflowers_core::wasm::WasmTensorOps;

        let ops = WasmTensorOps::new();

        // Test basic operations
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        // Test addition
        ops.add_simd(&a, &b, &mut result).unwrap();
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);

        // Test multiplication
        ops.mul_simd(&a, &b, &mut result).unwrap();
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);

        // Test ReLU
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut result = vec![0.0; 5];
        ops.relu_simd(&input, &mut result).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}
