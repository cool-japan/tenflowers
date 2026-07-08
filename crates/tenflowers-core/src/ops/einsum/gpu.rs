//! GPU-Accelerated Operations for Einstein Summation
//!
//! This module contains GPU-optimized implementations for einsum operations
//! using compute shaders and GPU kernels when the gpu feature is enabled.

#[allow(unused_imports)]
use crate::TensorError;
use crate::{Result, Tensor};
use scirs2_core::numeric::{One, Zero};

// GPU einsum implementations
#[cfg(feature = "gpu")]
pub fn gpu_einsum_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::ops::execute_einsum_matmul;
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            // Ensure matrices are 2D for standard matmul
            if a.shape().dims().len() != 2 || b.shape().dims().len() != 2 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum matrix multiplication requires 2D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            // Check dimensions are compatible for matmul
            if a_shape[1] != b_shape[0] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_matmul_gpu".to_string(),
                    expected: format!("({}, K) and (K, {})", a_shape[0], b_shape[1]),
                    got: format!(
                        "({}, {}) and ({}, {})",
                        a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                    ),
                    context: None,
                });
            }

            let output_shape = crate::Shape::new(vec![a_shape[0], b_shape[1]]);

            // Execute GPU einsum matrix multiplication
            let result_buffer = execute_einsum_matmul(
                gpu_a,
                gpu_b,
                &a_shape,
                &b_shape,
                output_shape.dims().iter().product::<usize>(),
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            crate::ops::matmul(a, b)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_batched_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // Ensure matrices are 3D for batched matmul (batch, height, width)
            if a.shape().dims().len() != 3 || b.shape().dims().len() != 3 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum batched matrix multiplication requires 3D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            // Check dimensions are compatible for batched matmul
            if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_batch_matmul_gpu".to_string(),
                    expected: "(B, M, K) and (B, K, N)".to_string(),
                    got: format!(
                        "({}, {}, {}) and ({}, {}, {})",
                        a_shape[0], a_shape[1], a_shape[2], b_shape[0], b_shape[1], b_shape[2]
                    ),
                    context: None,
                });
            }

            // The GPU batched-matmul kernel (`execute_einsum_batched_matmul`) is
            // not correctly implemented yet: a real per-batch GEMM would be
            // required, but the previous shortcut silently produced wrong
            // results, so the kernel now returns an honest error instead. Read
            // both operands back to the host (a real device->host transfer) and
            // delegate to the CPU einsum implementation, which is known-correct
            // for "bij,bjk->bik".
            let cpu_a = a.to_cpu()?;
            let cpu_b = b.to_cpu()?;
            crate::ops::einsum::einsum("bij,bjk->bik", &[&cpu_a, &cpu_b])
        }
        _ => {
            // Fall back to CPU implementation
            crate::ops::matmul(a, b)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_transpose<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::tensor::TensorStorage;

    match &tensor.storage {
        TensorStorage::Gpu(_) => {
            // Ensure tensor is 2D for transpose
            if tensor.shape().dims().len() != 2 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum transpose requires 2D tensor".to_string(),
                ));
            }

            // The GPU transpose kernel (`execute_einsum_transpose`) is not
            // correctly implemented yet: it previously ignored the real input
            // shape and produced silently-wrong data, so it now returns an
            // honest error instead. Read the operand back to the host (a real
            // device->host transfer) and delegate to the CPU einsum
            // implementation, which is known-correct for "ij->ji".
            let cpu_tensor = tensor.to_cpu()?;
            crate::ops::einsum::einsum("ij->ji", &[&cpu_tensor])
        }
        _ => {
            // Fall back to CPU implementation
            tensor.transpose()
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_diagonal<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Ensure tensor is 2D for diagonal extraction
    if tensor.shape().dims().len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "GPU einsum diagonal requires 2D tensor".to_string(),
        ));
    }

    // The GPU diagonal-extraction kernel (`execute_einsum_diagonal`) is not
    // correctly implemented yet: a correct gather of the strided diagonal
    // elements would be required, but the previous shortcut silently returned
    // the wrong elements, so the kernel now returns an honest error instead.
    // Read the operand back to the host (a real device->host transfer, or a
    // no-op clone if it is already CPU-resident) and delegate to the CPU
    // einsum implementation, which is known-correct for "ii->i".
    let cpu_tensor = tensor.to_cpu()?;
    crate::ops::einsum::einsum("ii->i", &[&cpu_tensor])
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_outer_product<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Ensure tensors are 1D for outer product
    if a.shape().dims().len() != 1 || b.shape().dims().len() != 1 {
        return Err(TensorError::invalid_shape_simple(
            "GPU einsum outer product requires 1D tensors".to_string(),
        ));
    }

    // The GPU outer-product kernel (`execute_einsum_outer_product`) is not
    // correctly implemented yet: a real broadcasted outer product would be
    // required, but the previous shortcut silently computed an element-wise
    // product instead, so the kernel now returns an honest error instead.
    // Read both operands back to the host (a real device->host transfer, or a
    // no-op clone for operands already CPU-resident) and delegate to the CPU
    // einsum implementation, which is known-correct for "i,j->ij".
    let cpu_a = a.to_cpu()?;
    let cpu_b = b.to_cpu()?;
    crate::ops::einsum::einsum("i,j->ij", &[&cpu_a, &cpu_b])
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_vector_dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::ops::execute_einsum_vector_dot;
    use crate::tensor::TensorStorage;

    match (&a.storage, &b.storage) {
        (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) => {
            // Ensure tensors are 1D for vector dot product
            if a.shape().dims().len() != 1 || b.shape().dims().len() != 1 {
                return Err(TensorError::invalid_shape_simple(
                    "GPU einsum vector dot product requires 1D tensors".to_string(),
                ));
            }

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            // Check dimensions are compatible for dot product
            if a_shape[0] != b_shape[0] {
                return Err(TensorError::ShapeMismatch {
                    operation: "einsum_dot_gpu".to_string(),
                    expected: format!("({},) and ({},)", a_shape[0], a_shape[0]),
                    got: format!("({},) and ({},)", a_shape[0], b_shape[0]),
                    context: None,
                });
            }

            let output_shape = crate::Shape::new(vec![]); // Scalar result

            // Execute GPU einsum vector dot product
            let result_buffer = execute_einsum_vector_dot(
                gpu_a,
                gpu_b,
                &a_shape,
                &b_shape,
                output_shape.dims().iter().product::<usize>(),
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, output_shape))
        }
        _ => {
            // Fall back to CPU implementation
            crate::ops::dot(a, b)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_einsum_trace<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Ensure tensor is 2D for trace calculation
    if tensor.shape().dims().len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "GPU einsum trace requires 2D tensor".to_string(),
        ));
    }

    let input_shape = tensor.shape().dims();

    // Check if it's a square matrix
    if input_shape[0] != input_shape[1] {
        return Err(TensorError::invalid_shape_simple(
            "GPU einsum trace requires square matrix".to_string(),
        ));
    }

    // The GPU trace kernel (`execute_einsum_trace`) is not correctly
    // implemented yet: a correct trace sums only the diagonal elements, but
    // the previous shortcut silently summed the entire buffer instead, so the
    // kernel now returns an honest error instead. Read the operand back to the
    // host (a real device->host transfer, or a no-op clone if it is already
    // CPU-resident) and delegate to the CPU einsum implementation, which is
    // known-correct for "ii->".
    let cpu_tensor = tensor.to_cpu()?;
    crate::ops::einsum::einsum("ii->", &[&cpu_tensor])
}

// Fallback implementations when GPU is not available
#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_matmul<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: this is the non-gpu fallback. Previously it called
    // `unreachable!()`, which would PANIC any caller in a build without the `gpu`
    // feature. Return a recoverable error so the call site can fall back / surface
    // the missing capability instead of aborting the process.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_batched_matmul<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: non-gpu fallback. Return a recoverable error instead of the
    // previous process-aborting `unreachable!()`.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_transpose<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: non-gpu fallback. Return a recoverable error instead of the
    // previous process-aborting `unreachable!()`.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_diagonal<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: non-gpu fallback. Return a recoverable error instead of the
    // previous process-aborting `unreachable!()`.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_outer_product<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: non-gpu fallback. Return a recoverable error instead of the
    // previous process-aborting `unreachable!()`.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_vector_dot<T>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: non-gpu fallback. Return a recoverable error instead of the
    // previous process-aborting `unreachable!()`.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_einsum_trace<T>(_tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // HONEST ERROR: non-gpu fallback. Return a recoverable error instead of the
    // previous process-aborting `unreachable!()`.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum requires the gpu feature".to_string(),
    ))
}

// Tests for the non-gpu fallback path. These verify that each `gpu_einsum_*`
// helper returns an HONEST, recoverable `UnsupportedOperation` error instead of
// the old `unreachable!()` panic (which would abort any caller built without the
// `gpu` feature). They run only in a non-gpu build, where these fallback
// definitions are the active ones.
#[cfg(all(test, not(feature = "gpu")))]
mod tests {
    use super::*;

    /// Assert the error is the honest "requires the gpu feature" `UnsupportedOperation`.
    fn assert_requires_gpu_feature(err: TensorError) {
        match err {
            TensorError::UnsupportedOperation { reason, .. } => {
                assert!(
                    reason.contains("gpu feature"),
                    "expected an honest 'gpu feature' error, got: {reason}"
                );
            }
            other => panic!("expected UnsupportedOperation, got: {other:?}"),
        }
    }

    #[test]
    fn matmul_fallback_returns_honest_error_not_panic() {
        let a = Tensor::<f32>::zeros(&[2, 2]);
        let b = Tensor::<f32>::zeros(&[2, 2]);
        let err = gpu_einsum_matmul(&a, &b).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }

    #[test]
    fn batched_matmul_fallback_returns_honest_error_not_panic() {
        let a = Tensor::<f32>::zeros(&[2, 2, 2]);
        let b = Tensor::<f32>::zeros(&[2, 2, 2]);
        let err = gpu_einsum_batched_matmul(&a, &b).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }

    #[test]
    fn transpose_fallback_returns_honest_error_not_panic() {
        let t = Tensor::<f32>::zeros(&[2, 3]);
        let err = gpu_einsum_transpose(&t).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }

    #[test]
    fn diagonal_fallback_returns_honest_error_not_panic() {
        let t = Tensor::<f32>::zeros(&[3, 3]);
        let err = gpu_einsum_diagonal(&t).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }

    #[test]
    fn outer_product_fallback_returns_honest_error_not_panic() {
        let a = Tensor::<f32>::zeros(&[3]);
        let b = Tensor::<f32>::zeros(&[4]);
        let err = gpu_einsum_outer_product(&a, &b).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }

    #[test]
    fn vector_dot_fallback_returns_honest_error_not_panic() {
        let a = Tensor::<f32>::zeros(&[5]);
        let b = Tensor::<f32>::zeros(&[5]);
        let err = gpu_einsum_vector_dot(&a, &b).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }

    #[test]
    fn trace_fallback_returns_honest_error_not_panic() {
        let t = Tensor::<f32>::zeros(&[4, 4]);
        let err = gpu_einsum_trace(&t).expect_err("must not fabricate a result");
        assert_requires_gpu_feature(err);
    }
}

// End-to-end tests for the GPU-resident code paths of the 5 patterns whose
// underlying kernels (`execute_einsum_batched_matmul`, `execute_einsum_transpose`,
// `execute_einsum_diagonal`, `execute_einsum_outer_product`, `execute_einsum_trace`
// in `gpu/ops/einsum_ops.rs`) are not correctly implemented. These build real
// GPU-resident tensors and call the actual `gpu_einsum_*` wrappers, verifying the
// device->host readback + CPU-delegate fallback produces numerically correct
// results (not just "doesn't panic").
//
// A GPU adapter is not guaranteed to be present in every environment that builds
// with `--features gpu` (e.g. a headless CI runner). `Tensor::to(Device::Gpu(0))`
// surfaces adapter/device creation failures as an honest `Err` (see
// `GpuContext::new`) rather than panicking, so each test attempts the transfer and
// skips its assertions - without failing the suite - if no adapter is available.
// This mirrors the convention used by `ops::random::tests::test_gpu_random_normal_f32`.
#[cfg(all(test, feature = "gpu"))]
mod gpu_delegate_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_batched_matmul_matches_cpu_reference() {
        // batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]]    = [[19,22],[43,50]]
        // batch 1: [[1,0],[0,1]] @ [[9,10],[11,12]] = [[9,10],[11,12]]
        let a_cpu =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2])
                .expect("test: from_vec should succeed");
        let b_cpu =
            Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 2, 2])
                .expect("test: from_vec should succeed");

        let (a_gpu, b_gpu) = match (a_cpu.to(Device::Gpu(0)), b_cpu.to(Device::Gpu(0))) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let result = gpu_einsum_batched_matmul(&a_gpu, &b_gpu)
            .expect("test: gpu_einsum_batched_matmul should succeed with a real adapter");
        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn gpu_transpose_matches_cpu_reference() {
        let a_cpu = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("test: from_vec should succeed");

        let a_gpu = match a_cpu.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        let result = gpu_einsum_transpose(&a_gpu)
            .expect("test: gpu_einsum_transpose should succeed with a real adapter");
        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn gpu_diagonal_matches_cpu_reference() {
        let a_cpu =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .expect("test: from_vec should succeed");

        let a_gpu = match a_cpu.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        let result = gpu_einsum_diagonal(&a_gpu)
            .expect("test: gpu_einsum_diagonal should succeed with a real adapter");
        assert_eq!(result.shape().dims(), &[3]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn gpu_outer_product_matches_cpu_reference() {
        let a_cpu = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let b_cpu =
            Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).expect("test: from_vec should succeed");

        let (a_gpu, b_gpu) = match (a_cpu.to(Device::Gpu(0)), b_cpu.to(Device::Gpu(0))) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let result = gpu_einsum_outer_product(&a_gpu, &b_gpu)
            .expect("test: gpu_einsum_outer_product should succeed with a real adapter");
        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn gpu_trace_matches_cpu_reference() {
        let a_cpu =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .expect("test: from_vec should succeed");

        let a_gpu = match a_cpu.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        let result = gpu_einsum_trace(&a_gpu)
            .expect("test: gpu_einsum_trace should succeed with a real adapter");
        let data = result.to_vec().expect("test: to_vec should succeed");
        // Trace = 1 + 5 + 9 = 15; must NOT equal the sum of all 9 entries (45).
        assert_eq!(data, vec![15.0]);
    }
}
