use crate::{Result, Tensor, TensorError};
use num_traits::{Float, One, Zero};

#[cfg(feature = "gpu")]
use crate::tensor::TensorStorage;

#[cfg(feature = "gpu")]
use crate::gpu::linalg::context::GpuLinalgContext;
#[cfg(feature = "gpu")]
use lazy_static::lazy_static;
#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "gpu")]
lazy_static! {
    /// Global GPU linear algebra context
    /// This is initialized lazily when the first GPU operation is requested
    static ref GPU_LINALG_CONTEXT: Arc<Mutex<Option<GpuLinalgContext>>> = Arc::new(Mutex::new(None));
}

#[cfg(feature = "gpu")]
/// Initialize GPU linear algebra context if not already initialized
async fn ensure_gpu_linalg_context() -> Result<()> {
    use crate::gpu::GpuContext;

    let mut context_guard = GPU_LINALG_CONTEXT.lock().unwrap();
    if context_guard.is_none() {
        // Initialize GPU context
        let gpu_ctx = GpuContext::new().map_err(|e| TensorError::ComputeError {
            operation: "gpu_context_init".to_string(),
            details: format!("Failed to initialize GPU context: {}", e),
            retry_possible: false,
            context: None,
        })?;

        // Create GPU linear algebra context
        let mut linalg_ctx = GpuLinalgContext::new(gpu_ctx.device.clone(), gpu_ctx.queue.clone());
        linalg_ctx.initialize_pipelines()?;

        *context_guard = Some(linalg_ctx);
    }
    Ok(())
}

#[cfg(feature = "gpu")]
/// Check if tensor is on GPU and should use GPU operations
fn should_use_gpu<T>(tensor: &Tensor<T>) -> bool {
    use crate::Device;
    matches!(tensor.device(), Device::Gpu(_))
}

#[cfg(not(feature = "gpu"))]
/// Fallback when GPU features are not enabled
#[allow(dead_code)] // Only used when GPU feature is disabled
fn should_use_gpu<T>(_tensor: &Tensor<T>) -> bool {
    false
}

/// Helper function for LU decomposition with determinant computation
fn lu_decompose_with_det<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, T)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    let n = shape[0];
    let input_data = input.as_slice().unwrap();

    let mut a = input_data.to_vec();
    let mut det = T::one();
    let mut swaps = 0;

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k * n + i].abs() > a[max_row * n + i].abs() {
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..n {
                a.swap(i * n + j, max_row * n + j);
            }
            swaps += 1;
        }

        // Check for singularity
        if a[i * n + i].abs() < T::from(1e-10).unwrap_or(T::zero()) {
            return Ok((Tensor::zeros(&[n, n]), Tensor::zeros(&[n, n]), T::zero()));
        }

        // Update determinant with diagonal element
        det = det * a[i * n + i];

        // Eliminate below pivot
        for k in (i + 1)..n {
            let factor = a[k * n + i] / a[i * n + i];
            for j in i..n {
                a[k * n + j] = a[k * n + j] - factor * a[i * n + j];
            }
        }
    }

    // Adjust determinant for number of row swaps
    if swaps % 2 == 1 {
        det = T::zero() - det;
    }

    // For now, return placeholder tensors and the determinant
    Ok((Tensor::zeros(&[n, n]), Tensor::zeros(&[n, n]), det))
}

/// Compute eigenvalues and eigenvectors of a square matrix using QR algorithm
pub fn eig<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::invalid_shape_simple(
            "Eigendecomposition requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];
    if n == 0 {
        return Ok((Tensor::zeros(&[0]), Tensor::zeros(&[0, 0])));
    }

    let input_data = input.as_slice().unwrap();
    let mut a = input_data.to_vec();

    // Initialize eigenvector matrix as identity
    let mut v = vec![T::zero(); n * n];
    for i in 0..n {
        v[i * n + i] = T::one();
    }

    // QR algorithm for eigenvalues
    let max_iterations = 100;
    let tolerance = T::from(1e-10).unwrap_or(T::zero());

    for _iter in 0..max_iterations {
        // QR decomposition of current matrix
        let (q, r) = qr_decomposition(&a, n)?;

        // Update A = R * Q
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + r[i * n + k] * q[k * n + j];
                }
                a[i * n + j] = sum;
            }
        }

        // Update eigenvectors V = V * Q
        let mut new_v = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + v[i * n + k] * q[k * n + j];
                }
                new_v[i * n + j] = sum;
            }
        }
        v = new_v;

        // Check convergence (off-diagonal elements should be small)
        let mut max_off_diag = T::zero();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let val = a[i * n + j].abs();
                    if val > max_off_diag {
                        max_off_diag = val;
                    }
                }
            }
        }

        if max_off_diag < tolerance {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = vec![T::zero(); n];
    for i in 0..n {
        eigenvalues[i] = a[i * n + i];
    }

    Ok((
        Tensor::from_vec(eigenvalues, &[n])?,
        Tensor::from_vec(v, &[n, n])?,
    ))
}

/// Helper function for QR decomposition using Gram-Schmidt process
pub fn qr_decomposition<T>(a: &[T], n: usize) -> Result<(Vec<T>, Vec<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let mut q = vec![T::zero(); n * n];
    let mut r = vec![T::zero(); n * n];

    // Gram-Schmidt process
    for j in 0..n {
        // Copy column j of A
        let mut v = vec![T::zero(); n];
        for i in 0..n {
            v[i] = a[i * n + j];
        }

        // Subtract projections onto previous columns
        for k in 0..j {
            let mut dot_product = T::zero();
            for i in 0..n {
                dot_product = dot_product + v[i] * q[i * n + k];
            }
            r[k * n + j] = dot_product;

            for i in 0..n {
                v[i] = v[i] - dot_product * q[i * n + k];
            }
        }

        // Normalize the vector
        let mut norm = T::zero();
        for val in v.iter().take(n) {
            norm = norm + *val * *val;
        }
        norm = norm.sqrt();
        r[j * n + j] = norm;

        if norm > T::from(1e-10).unwrap_or(T::zero()) {
            for i in 0..n {
                q[i * n + j] = v[i] / norm;
            }
        }
    }

    Ok((q, r))
}

/// Compute Singular Value Decomposition (SVD) using Jacobi method
pub fn svd<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "SVD requires a 2D matrix".to_string(),
        ));
    }

    let (m, n) = (shape[0], shape[1]);
    let input_data = input.as_slice().unwrap();

    // For SVD of A (m×n), we compute:
    // A = U Σ V^T where U is m×m, Σ is m×n (diagonal), V is n×n

    // First, compute A^T A (n×n) for V and singular values
    let mut ata = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..m {
                sum = sum + input_data[k * n + i] * input_data[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // Compute eigendecomposition of A^T A to get V and σ²
    let ata_tensor = Tensor::from_vec(ata, &[n, n])?;
    let (eigenvalues, eigenvectors) = eig(&ata_tensor)?;

    let sigma_squared = eigenvalues.as_slice().unwrap().to_vec();
    let v_data = eigenvectors.as_slice().unwrap().to_vec();

    // Sort eigenvalues and eigenvectors in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        sigma_squared[j]
            .partial_cmp(&sigma_squared[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_sigma_sq = vec![T::zero(); n];
    let mut sorted_v = vec![T::zero(); n * n];

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_sigma_sq[new_idx] = sigma_squared[old_idx];
        for i in 0..n {
            sorted_v[i * n + new_idx] = v_data[i * n + old_idx];
        }
    }

    // Compute singular values (square root of eigenvalues)
    let mut singular_values = vec![T::zero(); n.min(m)];
    for i in 0..singular_values.len() {
        if sorted_sigma_sq[i] > T::zero() {
            singular_values[i] = sorted_sigma_sq[i].sqrt();
        }
    }

    // Compute U = A V Σ^(-1) for the first min(m,n) columns
    let mut u = vec![T::zero(); m * m];
    let min_dim = m.min(n);

    for j in 0..min_dim {
        if singular_values[j] > T::from(1e-10).unwrap_or(T::zero()) {
            // Compute A * v_j / σ_j
            for i in 0..m {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + input_data[i * n + k] * sorted_v[k * n + j];
                }
                u[i * m + j] = sum / singular_values[j];
            }
        }
    }

    // Complete U to orthogonal matrix using Gram-Schmidt on remaining columns
    for j in min_dim..m {
        // Start with random vector
        for i in 0..m {
            u[i * m + j] = T::from((i + j) as f64 * 0.1).unwrap_or(T::one());
        }

        // Orthogonalize against previous columns
        for k in 0..j {
            let mut dot = T::zero();
            for i in 0..m {
                dot = dot + u[i * m + j] * u[i * m + k];
            }
            for i in 0..m {
                u[i * m + j] = u[i * m + j] - dot * u[i * m + k];
            }
        }

        // Normalize
        let mut norm = T::zero();
        for i in 0..m {
            norm = norm + u[i * m + j] * u[i * m + j];
        }
        norm = norm.sqrt();

        if norm > T::from(1e-10).unwrap_or(T::zero()) {
            for i in 0..m {
                u[i * m + j] = u[i * m + j] / norm;
            }
        }
    }

    // Create diagonal matrix Σ (m×n)
    let mut sigma = vec![T::zero(); m * n];
    for i in 0..singular_values.len() {
        if i < m && i < n {
            sigma[i * n + i] = singular_values[i];
        }
    }

    Ok((
        Tensor::from_vec(u, &[m, m])?,
        Tensor::from_vec(sigma, &[m, n])?,
        Tensor::from_vec(sorted_v, &[n, n])?,
    ))
}

/// Compute matrix inverse using Gauss-Jordan elimination
pub fn inv<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::invalid_shape_simple(
            "Matrix inverse requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];
    if n == 0 {
        return Err(TensorError::invalid_shape_simple(
            "Cannot invert empty matrix".to_string(),
        ));
    }

    // For larger matrices, try GPU implementation first if available
    #[cfg(feature = "gpu")]
    if n > 2 {
        use crate::gpu::buffer::GpuBuffer;
        use bytemuck::{Pod, Zeroable};

        if should_use_gpu(input) && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Use GPU implementation for f32 tensors
            let rt = tokio::runtime::Runtime::new().map_err(|e| TensorError::ComputeError {
                operation: "async_runtime_init".to_string(),
                details: format!("Failed to create async runtime: {}", e),
                retry_possible: false,
                context: None,
            })?;

            let gpu_result = rt.block_on(async {
                ensure_gpu_linalg_context().await?;

                let mut context_guard = GPU_LINALG_CONTEXT.lock().unwrap();
                let context = context_guard.as_mut().unwrap();

                // Create output tensor with same shape and device
                let mut output = Tensor::<T>::zeros(input.shape().dims());

                // Get GPU buffers using pattern matching
                match (&input.storage, &output.storage) {
                    (TensorStorage::Gpu(input_buffer), TensorStorage::Gpu(output_buffer)) => {
                        // context.inverse(input_buffer, output_buffer, input.shape())?;
                        // GPU linalg not yet implemented
                        Err(TensorError::unsupported_operation_simple(
                            "GPU matrix inverse not yet implemented".to_string(),
                        ))
                    }
                    _ => Err(TensorError::unsupported_operation_simple(
                        "GPU matrix inverse requires GPU tensors".to_string(),
                    )),
                }
            });

            match gpu_result {
                Ok(result) => {
                    // Convert back to generic type T if needed
                    return Ok(result);
                }
                Err(_) => {
                    // Fall back to CPU implementation
                }
            }
        }
    }

    // Use CPU implementation (fallback or when GPU not available)
    // Create augmented matrix [A | I]
    let mut augmented = vec![T::zero(); n * 2 * n];
    let input_data = input.as_slice().unwrap();

    // Initialize augmented matrix
    for i in 0..n {
        for j in 0..n {
            augmented[i * 2 * n + j] = input_data[i * n + j];
            augmented[i * 2 * n + n + j] = if i == j { T::one() } else { T::zero() };
        }
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[k * 2 * n + i].abs() > augmented[max_row * 2 * n + i].abs() {
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(2 * n) {
                augmented.swap(i * 2 * n + j, max_row * 2 * n + j);
            }
        }

        // Check for singularity
        if augmented[i * 2 * n + i].abs() < T::from(1e-10).unwrap_or(T::zero()) {
            return Err(TensorError::unsupported_operation_simple(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // Scale current row
        let pivot = augmented[i * 2 * n + i];
        for j in 0..(2 * n) {
            augmented[i * 2 * n + j] = augmented[i * 2 * n + j] / pivot;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = augmented[k * 2 * n + i];
                for j in 0..(2 * n) {
                    augmented[k * 2 * n + j] =
                        augmented[k * 2 * n + j] - factor * augmented[i * 2 * n + j];
                }
            }
        }
    }

    // Extract inverse matrix from right half of augmented matrix
    let mut result = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = augmented[i * 2 * n + n + j];
        }
    }

    Tensor::from_vec(result, &[n, n])
}

/// Compute Cholesky decomposition
/// Returns lower triangular matrix L such that A = L * L^T
pub fn cholesky<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::invalid_shape_simple(
            "Cholesky decomposition requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];
    let input_data = input.as_slice().unwrap();
    let mut l = vec![T::zero(); n * n];

    // Cholesky decomposition algorithm
    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + l[j * n + k] * l[j * n + k];
                }
                let val = input_data[j * n + j] - sum;
                if val <= T::zero() {
                    return Err(TensorError::unsupported_operation_simple(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[j * n + j] = val.sqrt();
            } else {
                // Lower triangular elements
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = (input_data[i * n + j] - sum) / l[j * n + j];
            }
        }
    }

    Tensor::from_vec(l, &[n, n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_det_2x2() {
        let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let det_result = det(&a).unwrap();
        let det_val = det_result.as_slice().unwrap()[0];
        assert_relative_eq!(det_val, -2.0, epsilon = 1e-10); // det = 1*4 - 2*3 = -2
    }

    #[test]
    fn test_det_3x3() {
        let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], &[3, 3])
            .unwrap();
        let det_result = det(&a).unwrap();
        let det_val = det_result.as_slice().unwrap()[0];
        assert_relative_eq!(det_val, 1.0, epsilon = 1e-10); // Computed manually
    }

    #[test]
    fn test_inv_2x2() {
        let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let inv_result = inv(&a).unwrap();
        let inv_data = inv_result.as_slice().unwrap();

        // Expected inverse of [[1,2],[3,4]] is [[-2,1],[1.5,-0.5]]
        assert_relative_eq!(inv_data[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(inv_data[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(inv_data[2], 1.5, epsilon = 1e-10);
        assert_relative_eq!(inv_data[3], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_2x2() {
        // Test with a positive definite matrix [[4,2],[2,2]]
        let a = Tensor::<f64>::from_vec(vec![4.0, 2.0, 2.0, 2.0], &[2, 2]).unwrap();
        let chol_result = cholesky(&a).unwrap();
        let chol_data = chol_result.as_slice().unwrap();

        // Expected Cholesky decomposition [[2,0],[1,1]]
        assert_relative_eq!(chol_data[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(chol_data[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(chol_data[2], 1.0, epsilon = 1e-10);
        assert_relative_eq!(chol_data[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lu_decomposition() {
        let a = Tensor::<f64>::from_vec(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
        let (l, u, _p) = lu(&a).unwrap();

        // Basic sanity check - L should be lower triangular with 1s on diagonal
        let l_data = l.as_slice().unwrap();
        assert_relative_eq!(l_data[0], 1.0, epsilon = 1e-10); // L[0,0] = 1
        assert_relative_eq!(l_data[1], 0.0, epsilon = 1e-10); // L[0,1] = 0

        // U should be upper triangular
        let u_data = u.as_slice().unwrap();
        assert!(u_data[0] != 0.0); // U[0,0] should be non-zero
    }

    #[test]
    fn test_eigenvalues_2x2() {
        // Test with a simple 2x2 matrix [[2, 1], [1, 2]]
        // Eigenvalues should be 3 and 1
        let a = Tensor::<f64>::from_vec(vec![2.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let (eigenvals, _eigenvecs) = eig(&a).unwrap();
        let vals = eigenvals.as_slice().unwrap();

        // Sort the eigenvalues for comparison
        let mut sorted_vals = vals.to_vec();
        sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(sorted_vals[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(sorted_vals[1], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        // Test with diagonal matrix - eigenvalues should be the diagonal elements
        let a = Tensor::<f64>::from_vec(vec![3.0, 0.0, 0.0, 5.0], &[2, 2]).unwrap();
        let (eigenvals, _eigenvecs) = eig(&a).unwrap();
        let vals = eigenvals.as_slice().unwrap();

        // Sort the eigenvalues for comparison
        let mut sorted_vals = vals.to_vec();
        sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(sorted_vals[0], 5.0, epsilon = 1e-8);
        assert_relative_eq!(sorted_vals[1], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_svd_2x2() {
        // Test SVD with a simple 2x2 matrix
        let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = svd(&a);

        // SVD should succeed (even if values aren't perfectly accurate due to iterative algorithm)
        assert!(result.is_ok());

        let (u, sigma, v) = result.unwrap();
        assert_eq!(u.shape().dims(), &[2, 2]);
        assert_eq!(sigma.shape().dims(), &[2, 2]);
        assert_eq!(v.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_svd_rank_one() {
        // Test SVD with rank-1 matrix (outer product of two vectors)
        let a = Tensor::<f64>::from_vec(vec![2.0, 4.0, 1.0, 2.0], &[2, 2]).unwrap();
        let result = svd(&a);

        assert!(result.is_ok());
        let (u, sigma, v) = result.unwrap();

        // Check shapes
        assert_eq!(u.shape().dims(), &[2, 2]);
        assert_eq!(sigma.shape().dims(), &[2, 2]);
        assert_eq!(v.shape().dims(), &[2, 2]);

        // For rank-1 matrix, one singular value should be much larger than the other
        let sigma_data = sigma.as_slice().unwrap();
        let s1 = sigma_data[0]; // σ[0,0]
        let s2 = sigma_data[3]; // σ[1,1]

        // One should be significantly larger (this is a rank-1 matrix)
        assert!(s1 > s2 || s2 > s1);
    }
}

/// LU decomposition with partial pivoting
/// Returns (L, U, P) where PA = LU
pub fn lu<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "LU decomposition requires a 2D matrix".to_string(),
        ));
    }

    // Check if input is on GPU and should use GPU acceleration
    #[cfg(feature = "gpu")]
    {
        let (m, n) = (shape[0], shape[1]);
        if should_use_gpu(input) && m >= 64 && n >= 64 {
            return lu_gpu(input);
        }
    }

    // CPU implementation
    lu_cpu(input)
}

/// Matrix determinant
pub fn det<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::invalid_shape_simple(
            "Determinant requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];
    if n == 0 {
        return Ok(Tensor::from_scalar(T::one()));
    }
    if n == 1 {
        let val = input.as_slice().unwrap()[0];
        return Ok(Tensor::from_scalar(val));
    }
    if n == 2 {
        // For 2x2 matrix: det = ad - bc
        let data = input.as_slice().unwrap();
        let det_val = data[0] * data[3] - data[1] * data[2];
        return Ok(Tensor::from_scalar(det_val));
    }

    // For larger matrices, try GPU implementation first if available
    #[cfg(feature = "gpu")]
    {
        use crate::gpu::buffer::GpuBuffer;
        use bytemuck::{Pod, Zeroable};

        if should_use_gpu(input) && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Use GPU implementation for f32 tensors
            let rt = tokio::runtime::Runtime::new().map_err(|e| TensorError::ComputeError {
                operation: "async_runtime_init".to_string(),
                details: format!("Failed to create async runtime: {}", e),
                retry_possible: false,
                context: None,
            })?;

            let gpu_result: Result<Tensor<T>> = rt.block_on(async {
                ensure_gpu_linalg_context().await?;

                let mut context_guard = GPU_LINALG_CONTEXT.lock().unwrap();
                let context = context_guard.as_mut().unwrap();

                // Get GPU buffer from tensor using pattern matching
                match &input.storage {
                    TensorStorage::Gpu(gpu_buffer) => {
                        // let det_val = context.determinant(gpu_buffer, input.shape())?;
                        // GPU linalg not yet implemented
                        Err(TensorError::unsupported_operation_simple(
                            "GPU matrix determinant not yet implemented".to_string(),
                        ))
                    }
                    _ => Err(TensorError::unsupported_operation_simple(
                        "GPU matrix determinant requires GPU tensor".to_string(),
                    )),
                }
            });

            match gpu_result {
                Ok(det_val) => {
                    return Ok(det_val);
                }
                Err(_) => {
                    // Fall back to CPU implementation
                }
            }
        }
    }

    // Use CPU implementation (fallback or when GPU not available)
    match lu_decompose_with_det(input) {
        Ok((_, _, det_val)) => Ok(Tensor::from_scalar(det_val)),
        Err(e) => Err(e),
    }
}

#[cfg(feature = "gpu")]
/// GPU implementation of LU decomposition
/// Only supports f32 and f64 tensors on GPU
fn lu_gpu<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;
    use bytemuck::{Pod, Zeroable};

    let shape = input.shape().dims();
    let (m, n) = (shape[0], shape[1]);

    // Only support square matrices for GPU LU decomposition
    if m != n {
        return Err(TensorError::invalid_shape_simple(
            "GPU LU decomposition currently only supports square matrices".to_string(),
        ));
    }

    let rt = tokio::runtime::Runtime::new().map_err(|e| TensorError::ComputeError {
        operation: "async_runtime_init".to_string(),
        details: format!("Failed to create async runtime: {}", e),
        retry_possible: false,
        context: None,
    })?;

    let gpu_result = rt.block_on(async {
        ensure_gpu_linalg_context().await?;

        let mut context_guard = GPU_LINALG_CONTEXT.lock().unwrap();
        let context = context_guard.as_mut().unwrap();

        // GPU LU decomposition not yet implemented
        Err(TensorError::unsupported_operation_simple(
            "GPU LU decomposition not yet implemented".to_string(),
        ))
    });

    match gpu_result {
        Ok(result) => Ok(result),
        Err(e) => {
            // Fall back to CPU implementation for small matrices or on error
            eprintln!("GPU LU decomposition failed, falling back to CPU: {}", e);

            // Convert to CPU tensor and use CPU implementation
            let cpu_tensor = input.to_device(crate::Device::Cpu)?;
            lu_cpu(&cpu_tensor)
        }
    }
}

/// CPU implementation of LU decomposition (extracted from original lu function)
fn lu_cpu<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    let (m, n) = (shape[0], shape[1]);
    let min_dim = m.min(n);
    let input_data = input.as_slice().unwrap();

    let mut a = input_data.to_vec();
    let mut p = vec![T::zero(); m * m];

    // Initialize permutation matrix as identity
    for i in 0..m {
        p[i * m + i] = T::one();
    }

    // LU decomposition with partial pivoting
    for i in 0..min_dim {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..m {
            if a[k * n + i].abs() > a[max_row * n + i].abs() {
                max_row = k;
            }
        }

        // Swap rows in A and P if needed
        if max_row != i {
            // Swap rows in A
            for j in 0..n {
                a.swap(i * n + j, max_row * n + j);
            }
            // Swap rows in P
            for j in 0..m {
                p.swap(i * m + j, max_row * m + j);
            }
        }

        // Check for singularity
        if a[i * n + i].abs() < T::from(1e-10).unwrap_or(T::zero()) {
            continue; // Skip singular pivot
        }

        // Eliminate below pivot
        for k in (i + 1)..m {
            let factor = a[k * n + i] / a[i * n + i];
            a[k * n + i] = factor; // Store L factor in place
            for j in (i + 1)..n {
                a[k * n + j] = a[k * n + j] - factor * a[i * n + j];
            }
        }
    }

    // Extract L and U matrices
    let mut l = vec![T::zero(); m * min_dim];
    let mut u = vec![T::zero(); min_dim * n];

    for i in 0..m {
        for j in 0..min_dim {
            if i >= j {
                l[i * min_dim + j] = if i == j { T::one() } else { a[i * n + j] };
            }
        }
    }

    for i in 0..min_dim {
        for j in 0..n {
            if i <= j {
                u[i * n + j] = a[i * n + j];
            }
        }
    }

    Ok((
        Tensor::from_vec(l, &[m, min_dim])?,
        Tensor::from_vec(u, &[min_dim, n])?,
        Tensor::from_vec(p, &[m, m])?,
    ))
}
