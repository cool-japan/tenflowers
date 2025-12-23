//! Second-Order Gradient Methods
//!
//! This module provides utilities for computing and working with second-order gradients,
//! including Hessian approximations, Hessian-vector products, and related operations
//! useful for advanced optimization algorithms like Newton's method and natural gradient.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

use crate::GradientTape;

/// Finite-difference approximation for Hessian-vector product
///
/// Computes the product of the Hessian matrix with a vector v using finite differences,
/// without explicitly forming the Hessian. This is much more memory-efficient than
/// computing the full Hessian.
///
/// Uses the formula: Hv ≈ (∇f(x + εv) - ∇f(x - εv)) / (2ε)
///
/// # Arguments
///
/// * `gradient_fn` - Function that computes gradients at a point
/// * `x` - Point at which to compute Hessian-vector product
/// * `v` - Vector to multiply with Hessian
/// * `epsilon` - Finite difference step size (default: 1e-5)
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::second_order::hessian_vector_product_fd;
/// use tenflowers_core::Tensor;
///
/// let x = Tensor::from_vec(vec![1.0f32, 2.0], vec![2])?;
/// let v = Tensor::from_vec(vec![1.0f32, 0.0], vec![2])?;
///
/// let hv = hessian_vector_product_fd(
///     |x_val| {
///         // Gradient of f(x) = x1^2 + x2^2 is [2*x1, 2*x2]
///         Ok(vec![x_val.mul(&Tensor::from_scalar(2.0f32))?])
///     },
///     &x,
///     &v,
///     Some(1e-5),
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn hessian_vector_product_fd<F>(
    gradient_fn: F,
    x: &Tensor<f32>,
    v: &Tensor<f32>,
    epsilon: Option<f32>,
) -> Result<Tensor<f32>>
where
    F: Fn(&Tensor<f32>) -> Result<Vec<Tensor<f32>>>,
{
    let eps = epsilon.unwrap_or(1e-5);

    // Compute x + εv
    let eps_v = v.mul(&Tensor::from_scalar(eps))?;
    let x_plus = x.add(&eps_v)?;

    // Compute x - εv
    let x_minus = x.sub(&eps_v)?;

    // Compute ∇f(x + εv)
    let grad_plus = gradient_fn(&x_plus)?;
    if grad_plus.is_empty() {
        return Err(TensorError::invalid_operation_simple(
            "Gradient function returned empty vector".to_string(),
        ));
    }

    // Compute ∇f(x - εv)
    let grad_minus = gradient_fn(&x_minus)?;
    if grad_minus.is_empty() {
        return Err(TensorError::invalid_operation_simple(
            "Gradient function returned empty vector".to_string(),
        ));
    }

    // Hv ≈ (∇f(x + εv) - ∇f(x - εv)) / (2ε)
    let diff = grad_plus[0].sub(&grad_minus[0])?;
    let two_eps = Tensor::from_scalar(2.0 * eps);
    let hv = diff.div(&two_eps)?;

    Ok(hv)
}

/// Diagonal Hessian approximation
///
/// Computes an approximation of the diagonal of the Hessian matrix.
/// The diagonal Hessian is useful for diagonal scaling in optimization
/// and for uncertainty estimation.
///
/// Uses finite differences: H_ii ≈ (∇f(x + εe_i) - ∇f(x - εe_i)) / (2ε)
/// where e_i is the i-th unit vector.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::second_order::diagonal_hessian_fd;
/// use tenflowers_core::Tensor;
///
/// let x = Tensor::from_vec(vec![1.0f32, 2.0], vec![2])?;
///
/// let diag_h = diagonal_hessian_fd(
///     |x_val| {
///         // Gradient computation
///         Ok(vec![x_val.mul(&Tensor::from_scalar(2.0f32))?])
///     },
///     &x,
///     Some(1e-5),
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn diagonal_hessian_fd<F>(
    gradient_fn: F,
    x: &Tensor<f32>,
    epsilon: Option<f32>,
) -> Result<Tensor<f32>>
where
    F: Fn(&Tensor<f32>) -> Result<Vec<Tensor<f32>>>,
{
    let eps = epsilon.unwrap_or(1e-5);
    let shape = x.shape().dims();

    if shape.len() != 1 {
        return Err(TensorError::invalid_shape_simple(
            "Diagonal Hessian only supported for 1D tensors".to_string(),
        ));
    }

    let n = shape[0];
    let x_data = x.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access tensor data".to_string())
    })?;

    let mut diag_values = Vec::with_capacity(n);

    // Compute each diagonal element
    for i in 0..n {
        // Create unit vector e_i
        let mut e_i = vec![0.0f32; n];
        e_i[i] = 1.0;
        let unit_vec = Tensor::from_data(e_i, shape)?;

        // Compute x + εe_i and x - εe_i
        let eps_e = unit_vec.mul(&Tensor::from_scalar(eps))?;
        let x_plus = x.add(&eps_e)?;
        let x_minus = x.sub(&eps_e)?;

        // Compute gradients
        let grad_plus = gradient_fn(&x_plus)?;
        let grad_minus = gradient_fn(&x_minus)?;

        if grad_plus.is_empty() || grad_minus.is_empty() {
            return Err(TensorError::invalid_operation_simple(
                "Gradient function returned empty vector".to_string(),
            ));
        }

        // Extract i-th component of gradient difference
        let grad_plus_data = grad_plus[0].as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
        })?;

        let grad_minus_data = grad_minus[0].as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
        })?;

        // H_ii ≈ (∇f_i(x + εe_i) - ∇f_i(x - εe_i)) / (2ε)
        let h_ii = (grad_plus_data[i] - grad_minus_data[i]) / (2.0 * eps);
        diag_values.push(h_ii);
    }

    Tensor::from_data(diag_values, shape)
}

/// Gauss-Newton Hessian approximation
///
/// For least-squares problems, the Gauss-Newton approximation to the Hessian is:
/// H ≈ J^T J
///
/// where J is the Jacobian of the residuals. This is positive semi-definite
/// and works well for problems near the optimum.
///
/// # Arguments
///
/// * `jacobian` - Jacobian matrix (m × n) where m is number of residuals, n is number of parameters
///
/// # Returns
///
/// Approximate Hessian matrix (n × n)
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::second_order::gauss_newton_hessian;
/// use tenflowers_core::Tensor;
///
/// // Jacobian for m=10 residuals, n=5 parameters
/// let jacobian = Tensor::randn(&[10, 5])?;
/// let hessian_approx = gauss_newton_hessian(&jacobian)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gauss_newton_hessian(jacobian: &Tensor<f32>) -> Result<Tensor<f32>> {
    let shape = jacobian.shape().dims();

    if shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "Jacobian must be a 2D matrix".to_string(),
        ));
    }

    // H = J^T J
    let jacobian_t = jacobian.transpose()?;
    let hessian = jacobian_t.matmul(jacobian)?;

    Ok(hessian)
}

/// BFGS (Broyden-Fletcher-Goldfarb-Shanno) update
///
/// Updates a Hessian approximation using the BFGS quasi-Newton formula.
/// This is one of the most popular methods for approximating the inverse Hessian.
///
/// Update formula: B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)
///
/// where:
/// - s_k = x_{k+1} - x_k (step taken)
/// - y_k = ∇f(x_{k+1}) - ∇f(x_k) (gradient difference)
/// - B_k is the current inverse Hessian approximation
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::second_order::bfgs_update;
/// use tenflowers_core::Tensor;
///
/// let b_k = Tensor::eye(5)?;  // Initial inverse Hessian (identity)
/// let s_k = Tensor::from_vec(vec![0.1f32; 5], vec![5])?;  // Step
/// let y_k = Tensor::from_vec(vec![0.2f32; 5], vec![5])?;  // Gradient change
///
/// let b_k_plus_1 = bfgs_update(&b_k, &s_k, &y_k)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn bfgs_update(b_k: &Tensor<f32>, s_k: &Tensor<f32>, y_k: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Check dimensions
    let b_shape = b_k.shape().dims();
    let s_shape = s_k.shape().dims();
    let y_shape = y_k.shape().dims();

    if b_shape.len() != 2 || b_shape[0] != b_shape[1] {
        return Err(TensorError::invalid_shape_simple(
            "B_k must be square matrix".to_string(),
        ));
    }

    if s_shape != y_shape {
        return Err(TensorError::invalid_shape_simple(
            "s_k and y_k must have same shape".to_string(),
        ));
    }

    let n = b_shape[0];

    // Compute y_k^T s_k (scalar)
    let y_t_s = compute_dot_product(y_k, s_k)?;

    // Check for positive definiteness condition
    if y_t_s <= 0.0 {
        // If condition is violated, skip update (return B_k unchanged)
        return Ok(b_k.clone());
    }

    // Compute B_k s_k
    let s_k_2d = s_k.reshape(&[n, 1])?;
    let b_s = b_k.matmul(&s_k_2d)?;

    // Compute s_k^T B_k
    let s_t_2d = s_k.reshape(&[1, n])?;
    let s_t_b = s_t_2d.matmul(b_k)?;

    // Compute s_k^T B_k s_k (scalar)
    let s_t_b_s_tensor = s_t_b.matmul(&s_k_2d)?;
    let s_t_b_s = s_t_b_s_tensor.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access scalar value".to_string())
    })?[0];

    // Compute outer products
    // B_k s_k s_k^T B_k
    let b_s_s_t_b = b_s.matmul(&s_t_b)?;

    // y_k y_k^T
    let y_k_2d = y_k.reshape(&[n, 1])?;
    let y_t_2d = y_k.reshape(&[1, n])?;
    let y_y_t = y_k_2d.matmul(&y_t_2d)?;

    // BFGS update
    // B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)
    let term1 = b_s_s_t_b.div(&Tensor::from_scalar(s_t_b_s))?;
    let term2 = y_y_t.div(&Tensor::from_scalar(y_t_s))?;

    let b_k_plus_1 = b_k.sub(&term1)?.add(&term2)?;

    Ok(b_k_plus_1)
}

/// L-BFGS storage for limited-memory BFGS
///
/// Stores the last `m` vector pairs (s_k, y_k) for computing Hessian-vector
/// products efficiently without storing the full Hessian.
pub struct LBFGSStorage {
    /// Maximum number of vector pairs to store
    m: usize,
    /// Step vectors: s_k = x_{k+1} - x_k
    s_list: Vec<Tensor<f32>>,
    /// Gradient difference vectors: y_k = ∇f(x_{k+1}) - ∇f(x_k)
    y_list: Vec<Tensor<f32>>,
    /// Scalar products: ρ_k = 1 / (y_k^T s_k)
    rho_list: Vec<f32>,
}

impl LBFGSStorage {
    /// Create new L-BFGS storage
    ///
    /// # Arguments
    ///
    /// * `m` - Maximum number of vector pairs to store (typically 5-20)
    pub fn new(m: usize) -> Self {
        Self {
            m,
            s_list: Vec::new(),
            y_list: Vec::new(),
            rho_list: Vec::new(),
        }
    }

    /// Add a new vector pair to storage
    ///
    /// # Arguments
    ///
    /// * `s_k` - Step vector
    /// * `y_k` - Gradient difference vector
    pub fn push(&mut self, s_k: Tensor<f32>, y_k: Tensor<f32>) -> Result<()> {
        // Compute ρ_k = 1 / (y_k^T s_k)
        let y_t_s = compute_dot_product(&y_k, &s_k)?;

        if y_t_s.abs() < 1e-10 {
            // Skip this update if y_k^T s_k is too small
            return Ok(());
        }

        let rho_k = 1.0 / y_t_s;

        // If at capacity, remove oldest
        if self.s_list.len() >= self.m {
            self.s_list.remove(0);
            self.y_list.remove(0);
            self.rho_list.remove(0);
        }

        self.s_list.push(s_k);
        self.y_list.push(y_k);
        self.rho_list.push(rho_k);

        Ok(())
    }

    /// Compute Hessian-vector product using L-BFGS two-loop recursion
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient vector (or any vector to multiply with approximate Hessian)
    /// * `gamma` - Scaling factor (typically γ = (y_k^T s_k) / (y_k^T y_k))
    pub fn multiply(&self, grad: &Tensor<f32>, gamma: f32) -> Result<Tensor<f32>> {
        if self.s_list.is_empty() {
            // No history, return scaled gradient
            return grad.mul(&Tensor::from_scalar(gamma));
        }

        let k = self.s_list.len();
        let mut q = grad.clone();
        let mut alpha = vec![0.0f32; k];

        // First loop (backward)
        for i in (0..k).rev() {
            let alpha_i = self.rho_list[i] * compute_dot_product(&self.s_list[i], &q)?;
            alpha[i] = alpha_i;

            let y_alpha = self.y_list[i].mul(&Tensor::from_scalar(alpha_i))?;
            q = q.sub(&y_alpha)?;
        }

        // Scale
        let mut r = q.mul(&Tensor::from_scalar(gamma))?;

        // Second loop (forward)
        for (i, &alpha_i) in alpha.iter().enumerate() {
            let beta = self.rho_list[i] * compute_dot_product(&self.y_list[i], &r)?;
            let diff = alpha_i - beta;

            let s_diff = self.s_list[i].mul(&Tensor::from_scalar(diff))?;
            r = r.add(&s_diff)?;
        }

        Ok(r)
    }

    /// Get the number of stored vector pairs
    pub fn len(&self) -> usize {
        self.s_list.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.s_list.is_empty()
    }

    /// Clear all stored vectors
    pub fn clear(&mut self) {
        self.s_list.clear();
        self.y_list.clear();
        self.rho_list.clear();
    }
}

/// Compute dot product between two vectors
fn compute_dot_product(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<f32> {
    let a_data = a.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access tensor data".to_string())
    })?;

    let b_data = b.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access tensor data".to_string())
    })?;

    if a_data.len() != b_data.len() {
        return Err(TensorError::invalid_shape_simple(
            "Vectors must have same length for dot product".to_string(),
        ));
    }

    let dot: f32 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();

    Ok(dot)
}

/// Fisher Information Matrix utilities
///
/// The Fisher Information Matrix (FIM) is important for natural gradient descent
/// and uncertainty quantification in neural networks.
pub mod fisher {
    use super::*;

    /// Empirical Fisher Information Matrix approximation
    ///
    /// Computes F ≈ E[∇log p(y|x) ∇log p(y|x)^T] using samples.
    ///
    /// This is approximated as the outer product of gradients averaged over samples.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Collection of gradient vectors from different samples
    ///
    /// # Returns
    ///
    /// Approximate Fisher Information Matrix
    pub fn empirical_fisher(gradients: &[Tensor<f32>]) -> Result<Tensor<f32>> {
        if gradients.is_empty() {
            return Err(TensorError::invalid_operation_simple(
                "Need at least one gradient sample".to_string(),
            ));
        }

        let shape = gradients[0].shape().dims();
        if shape.len() != 1 {
            return Err(TensorError::invalid_shape_simple(
                "Gradients must be 1D vectors".to_string(),
            ));
        }

        let n = shape[0];

        // Initialize Fisher matrix
        let mut fisher_data = vec![0.0f32; n * n];

        // Sum outer products: F ≈ (1/N) Σ ∇log p_i ∇log p_i^T
        for grad in gradients {
            let grad_data = grad.as_slice().ok_or_else(|| {
                TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
            })?;

            for i in 0..n {
                for j in 0..n {
                    fisher_data[i * n + j] += grad_data[i] * grad_data[j];
                }
            }
        }

        // Average
        let num_samples = gradients.len() as f32;
        for val in &mut fisher_data {
            *val /= num_samples;
        }

        Tensor::from_data(fisher_data, &[n, n])
    }

    /// Diagonal Fisher Information Matrix (Diagonal approximation)
    ///
    /// Computes only the diagonal of the Fisher matrix, which is much cheaper
    /// and often sufficient for adaptive learning rate methods.
    pub fn diagonal_fisher(gradients: &[Tensor<f32>]) -> Result<Tensor<f32>> {
        if gradients.is_empty() {
            return Err(TensorError::invalid_operation_simple(
                "Need at least one gradient sample".to_string(),
            ));
        }

        let shape = gradients[0].shape().dims();
        if shape.len() != 1 {
            return Err(TensorError::invalid_shape_simple(
                "Gradients must be 1D vectors".to_string(),
            ));
        }

        let n = shape[0];
        let mut diag = vec![0.0f32; n];

        // Sum squared gradients: F_ii ≈ (1/N) Σ (∇log p_i)^2
        for grad in gradients {
            let grad_data = grad.as_slice().ok_or_else(|| {
                TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
            })?;

            for i in 0..n {
                diag[i] += grad_data[i] * grad_data[i];
            }
        }

        // Average
        let num_samples = gradients.len() as f32;
        for val in &mut diag {
            *val /= num_samples;
        }

        Tensor::from_data(diag, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hessian_vector_product() {
        // Test on f(x) = x^2, where Hessian is constant = 2
        let x = Tensor::from_data(vec![1.0f32], &[1]).unwrap();
        let v = Tensor::from_data(vec![1.0f32], &[1]).unwrap();

        let hv = hessian_vector_product_fd(
            |x_val| {
                // Gradient of x^2 is 2x
                let grad = x_val.mul(&Tensor::from_scalar(2.0f32)).unwrap();
                Ok(vec![grad])
            },
            &x,
            &v,
            Some(1e-5),
        )
        .unwrap();

        let hv_data = hv.as_slice().unwrap();
        // Hv should be approximately 2 * 1 = 2
        // Use a more relaxed tolerance for finite difference approximations
        assert!(
            (hv_data[0] - 2.0).abs() < 0.1,
            "Expected ~2.0, got {}",
            hv_data[0]
        );
    }

    #[test]
    fn test_lbfgs_storage() {
        let mut storage = LBFGSStorage::new(3);

        // Add some vector pairs
        let s1 = Tensor::from_data(vec![0.1f32, 0.2], &[2]).unwrap();
        let y1 = Tensor::from_data(vec![0.15f32, 0.25], &[2]).unwrap();

        storage.push(s1, y1).unwrap();

        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());

        // Test multiply
        let grad = Tensor::from_data(vec![1.0f32, 1.0], &[2]).unwrap();
        let result = storage.multiply(&grad, 1.0).unwrap();

        assert_eq!(result.shape().dims(), vec![2]);
    }

    #[test]
    fn test_diagonal_fisher() {
        let grads = vec![
            Tensor::from_data(vec![1.0f32, 2.0], &[2]).unwrap(),
            Tensor::from_data(vec![2.0f32, 3.0], &[2]).unwrap(),
        ];

        let fisher_diag = fisher::diagonal_fisher(&grads).unwrap();
        let fisher_data = fisher_diag.as_slice().unwrap();

        // F[0,0] = (1^2 + 2^2) / 2 = 2.5
        // F[1,1] = (2^2 + 3^2) / 2 = 6.5
        assert!((fisher_data[0] - 2.5).abs() < 1e-5);
        assert!((fisher_data[1] - 6.5).abs() < 1e-5);
    }

    #[test]
    fn test_gauss_newton_hessian() {
        // Test with a simple 3x2 Jacobian
        let jac_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let jacobian = Tensor::from_data(jac_data, &[3, 2]).unwrap();

        let hessian = gauss_newton_hessian(&jacobian).unwrap();

        assert_eq!(hessian.shape().dims(), vec![2, 2]);
    }
}
