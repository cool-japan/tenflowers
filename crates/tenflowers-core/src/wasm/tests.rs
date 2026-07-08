//! Tests for the WASM platform support module.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::super::*;

    #[test]
    fn test_utils() {
        let chunk_size = utils::optimal_chunk_size();
        assert!(chunk_size > 0);

        let memory_limit = utils::recommended_memory_limit();
        assert!(memory_limit > 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_non_wasm_context() {
        let _ctx = WasmContext::new();
        assert!(!utils::is_wasm());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_context() {
        let ctx = WasmContext::new();
        assert!(ctx.available_memory() > 0);
        assert!(utils::is_wasm());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_allocator() {
        let allocator = WasmAllocator::new(1024 * 1024); // 1MB limit

        let ptr = allocator
            .allocate(1024)
            .expect("test: allocate should succeed");
        assert!(!ptr.is_null());
        assert_eq!(allocator.total_allocated(), 1024);

        unsafe {
            allocator
                .deallocate(ptr)
                .expect("test: deallocate should succeed");
        }
        assert_eq!(allocator.total_allocated(), 0);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_ops_registry() {
        let registry = WasmOpRegistry::new();

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = registry
            .execute("add", &a, &b)
            .expect("test: execute should succeed");
        assert_eq!(result, vec![5.0, 7.0, 9.0]);

        let result = registry
            .execute("mul", &a, &b)
            .expect("test: execute should succeed");
        assert_eq!(result, vec![4.0, 10.0, 18.0]);

        let result = registry
            .execute("sub", &a, &b)
            .expect("test: execute should succeed");
        assert_eq!(result, vec![-3.0, -3.0, -3.0]);

        // "div" should match plain elementwise a[i] / b[i].
        let result = registry
            .execute("div", &a, &b)
            .expect("test: execute should succeed");
        let expected_div: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
        assert_eq!(result, expected_div);

        // Division by zero should follow standard IEEE-754 semantics (as
        // documented on `WasmTensorOps::div_simd`), not be treated as an error.
        let result = registry
            .execute("div", &[1.0f32], &[0.0f32])
            .expect("test: execute should succeed");
        assert_eq!(result, vec![f32::INFINITY]);

        // "relu" is unary; the second argument is ignored, so reuse `a`. The
        // registry dispatch must match calling `relu_simd` directly, and (since
        // every element of `a` is already >= 0) also matches `a` unchanged.
        let mut expected_relu = vec![0.0; a.len()];
        WasmTensorOps::new()
            .relu_simd(&a, &mut expected_relu)
            .expect("test: relu_simd should succeed");
        let result = registry
            .execute("relu", &a, &a)
            .expect("test: execute should succeed");
        assert_eq!(result, expected_relu);
        assert_eq!(result, a);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_tensor_ops() {
        let ops = WasmTensorOps::new();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        ops.add_simd(&a, &b, &mut result)
            .expect("test: add_simd should succeed");
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_matmul() {
        let ops = WasmTensorOps::new();

        // 2x2 * 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2], [3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6], [7,8]]
        let mut result = vec![0.0; 4];

        ops.matmul_wasm(&a, &b, &mut result, 2, 2, 2)
            .expect("test: matmul_wasm should succeed");
        // Expected: [[19,22], [43,50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_simd_operations() {
        let ops = WasmTensorOps::new();

        // Test SIMD operations with larger arrays (to test SIMD path)
        let size = 16; // Multiple of 4 for SIMD
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        // Test multiplication
        let mut result = vec![0.0; size];
        ops.mul_simd(&a, &b, &mut result)
            .expect("test: mul_simd should succeed");
        for i in 0..size {
            assert_eq!(result[i], (i as f32) * ((i + 1) as f32));
        }

        // Test subtraction
        let mut result = vec![0.0; size];
        ops.sub_simd(&a, &b, &mut result)
            .expect("test: sub_simd should succeed");
        for i in 0..size {
            assert_eq!(result[i], (i as f32) - ((i + 1) as f32));
        }

        // Test ReLU
        let input: Vec<f32> = (-8..8).map(|i| i as f32).collect();
        let mut result = vec![0.0; input.len()];
        ops.relu_simd(&input, &mut result)
            .expect("test: relu_simd should succeed");

        for (i, &val) in input.iter().enumerate() {
            assert_eq!(result[i], val.max(0.0));
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_simd_edge_cases() {
        let ops = WasmTensorOps::new();

        // Test with non-SIMD-aligned sizes
        let sizes = vec![1, 3, 5, 7, 15, 17];

        for size in sizes {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i + 10) as f32).collect();
            let mut result = vec![0.0; size];

            // Test that operations work correctly with non-aligned sizes
            ops.add_simd(&a, &b, &mut result)
                .expect("test: add_simd should succeed");
            for i in 0..size {
                assert_eq!(result[i], (i as f32) + ((i + 10) as f32));
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_feature_detection() {
        let features = WasmFeatures::detect();

        // These should always be available in modern WASM
        assert!(features.bulk_memory);
        assert!(features.reference_types);

        // SIMD detection should work (may be true or false depending on compile flags)
        let _simd_available = features.has_simd();

        // Threads typically require SharedArrayBuffer which may not be available
        let _threads_available = features.has_threads();
    }
}
