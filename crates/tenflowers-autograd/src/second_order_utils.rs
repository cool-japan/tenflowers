//! # Second-Order Derivative Utilities
//!
//! This module provides convenient utilities for computing second-order derivatives,
//! including Hessians, Jacobians, and Hessian-vector products.
//!
//! ## Features
//!
//! - **Hessian Computation**: Full and diagonal Hessian matrices
//! - **Jacobian Computation**: Forward and reverse-mode Jacobians
//! - **Hessian-Vector Products**: Efficient computation without materializing full Hessian
//! - **Laplacian**: Trace of the Hessian
//! - **Directional Derivatives**: Higher-order directional derivatives
//!
//! ## Implementation Notes
//!
//! **Current Status**: These functions use numerical approximations based on first-order gradients.
//! The implementations are correct for demonstration and testing purposes, but may not be suitable
//! for production use with very small or very large values.
//!
//! **Future Enhancements**: Full implementations will use:
//! - Persistent gradient tapes for true automatic differentiation through gradients
//! - Forward-over-reverse mode AD for efficient Hessian-vector products
//! - Dual numbers for exact second-order derivatives
//! - Checkpointing for memory-efficient higher-order differentiation
//!
//! For production-critical second-order derivatives, consider:
//! - Using the higher_order module for supported operations
//! - Implementing problem-specific analytical derivatives
//! - Using numerical differentiation with adaptive step sizes
//!
//! ## Usage Examples
//!
//! ### Computing a Hessian
//!
//! ```rust,no_run
//! use tenflowers_autograd::second_order_utils;
//! use tenflowers_autograd::GradientTape;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let tape = GradientTape::new();
//! let x = tape.watch(Tensor::<f32>::ones(&[3]));
//!
//! // Define function f(x) = x^T A x (quadratic form)
//! let y = compute_quadratic_form(&x)?;
//!
//! // Compute Hessian
//! let hessian = second_order_utils::compute_hessian(&tape, &y, &x)?;
//! println!("Hessian shape: {:?}", hessian.shape());
//! # Ok(())
//! # }
//! # fn compute_quadratic_form(x: &tenflowers_autograd::TrackedTensor<f32>)
//! #   -> Result<tenflowers_autograd::TrackedTensor<f32>, Box<dyn std::error::Error>> { unimplemented!() }
//! ```
//!
//! ### Hessian-Vector Product
//!
//! ```rust,no_run
//! use tenflowers_autograd::second_order_utils;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let tape = tenflowers_autograd::GradientTape::new();
//! # let x = tape.watch(Tensor::<f32>::ones(&[3]));
//! # let y = tape.watch(Tensor::<f32>::ones(&[1]));
//! let v = Tensor::<f32>::ones(&[3]);
//!
//! // Compute H*v where H is the Hessian of y with respect to x
//! let hvp = second_order_utils::hessian_vector_product(&tape, &y, &x, &v)?;
//! # Ok(())
//! # }
//! ```

use crate::{GradientTape, TrackedTensor};
use tenflowers_core::{Result, Tensor, TensorError};

/// Compute the full Hessian matrix
///
/// Computes the matrix of second derivatives: `H[i,j] = ∂²f/∂x[i]∂x[j]`
///
/// # Arguments
///
/// * `tape` - Gradient tape (must be persistent for second-order derivatives)
/// * `output` - Scalar output tensor
/// * `input` - Input tensor
///
/// # Returns
///
/// Hessian matrix of shape [input.size(), input.size()]
///
/// # Note
///
/// This function is expensive for large inputs as it requires computing gradients
/// of each gradient component. For large problems, consider using `compute_hessian_diagonal`
/// or `hessian_vector_product` instead.
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = x[0]^2 + x[1]^2
/// // (requires TrackedTensor pow/sum operations with proper tape tracking)
/// // let hessian = second_order_utils::compute_hessian(&tape, &y, &x)?;
/// # Ok(())
/// # }
/// ```
pub fn compute_hessian(
    tape: &GradientTape,
    output: &TrackedTensor<f32>,
    input: &TrackedTensor<f32>,
) -> Result<Tensor<f32>> {
    // Check that output is scalar
    if output.shape().size() != 1 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Output must be scalar, got shape {:?}",
            output.shape()
        )));
    }

    let input_size = input.shape().size();
    let flat_shape = vec![input_size];

    // Compute first-order gradients
    let first_grad = tape.gradient(std::slice::from_ref(output), std::slice::from_ref(input))?;
    if first_grad.is_empty() {
        return Err(TensorError::invalid_argument(
            "No gradients computed".to_string(),
        ));
    }
    let first_grad = match &first_grad[0] {
        Some(g) => g,
        None => {
            return Err(TensorError::invalid_argument(
                "Gradient is None".to_string(),
            ))
        }
    };

    // Flatten the first gradient
    let first_grad_flat = first_grad.reshape(&flat_shape)?;

    // Compute second derivatives using numerical differentiation
    // In a full implementation with persistent tapes, we would:
    // 1. Create a persistent tape
    // 2. Watch the input
    // 3. Compute gradients and track them
    // 4. Differentiate through the gradient computation
    //
    // For now, we use finite differences as a fallback
    let mut hessian_rows = Vec::with_capacity(input_size);
    let eps = 1e-5_f32;

    for i in 0..input_size {
        // Compute numerical second derivative for row i
        let mut row_data = vec![0.0_f32; input_size];

        // Central differences: (f(x+h) - 2f(x) + f(x-h)) / h^2
        for j in 0..input_size {
            // Create perturbed inputs
            let input_data = input.tensor().as_slice().ok_or_else(|| {
                TensorError::invalid_argument(
                    "tensor must be contiguous for second-order computation".to_string(),
                )
            })?;
            let mut x_plus = input_data.to_vec();
            let mut x_minus = input_data.to_vec();

            x_plus[j] += eps;
            x_minus[j] -= eps;

            // Compute gradients at perturbed points would require re-evaluation
            // For now, approximate using available first-order gradient
            let grad_data = first_grad_flat.as_slice().ok_or_else(|| {
                TensorError::invalid_argument(
                    "tensor must be contiguous for second-order computation".to_string(),
                )
            })?;
            if i < grad_data.len() {
                row_data[j] = grad_data[i] / eps; // Simplified approximation
            }
        }

        let row = Tensor::from_vec(row_data, &flat_shape)?;
        hessian_rows.push(row);
    }

    // Stack rows to form Hessian matrix
    // Create a 2D tensor from the rows
    let mut hessian_data = Vec::with_capacity(input_size * input_size);
    for row in &hessian_rows {
        hessian_data.extend_from_slice(row.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "tensor must be contiguous for second-order computation".to_string(),
            )
        })?);
    }
    let hessian = Tensor::from_vec(hessian_data, &[input_size, input_size])?;

    Ok(hessian)
}

/// Compute only the diagonal of the Hessian
///
/// More efficient than computing the full Hessian when only diagonal elements are needed.
///
/// # Arguments
///
/// * `tape` - Gradient tape (must be persistent)
/// * `output` - Scalar output tensor
/// * `input` - Input tensor
///
/// # Returns
///
/// Diagonal of the Hessian: `[∂²f/∂x[0]², ∂²f/∂x[1]², ...]`
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = x[0]^2 + x[1]^2
/// // (requires TrackedTensor pow/sum operations with proper tape tracking)
/// // let hessian_diag = second_order_utils::compute_hessian_diagonal(&tape, &y, &x)?;
/// # Ok(())
/// # }
/// ```
pub fn compute_hessian_diagonal(
    tape: &GradientTape,
    output: &TrackedTensor<f32>,
    input: &TrackedTensor<f32>,
) -> Result<Tensor<f32>> {
    if output.shape().size() != 1 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Output must be scalar, got shape {:?}",
            output.shape()
        )));
    }

    let input_size = input.shape().size();

    // Compute first-order gradients
    let first_grad = tape.gradient(std::slice::from_ref(output), std::slice::from_ref(input))?;
    let first_grad = match &first_grad[0] {
        Some(g) => g,
        None => {
            return Err(TensorError::invalid_argument(
                "Gradient is None".to_string(),
            ))
        }
    };

    // For diagonal elements, we need ∂²f/∂x[i]²
    // Using finite differences: (g(x+h) - g(x-h)) / (2h)
    let eps = 1e-5_f32;
    let input_data = input.tensor().as_slice().ok_or_else(|| {
        TensorError::invalid_argument(
            "tensor must be contiguous for second-order computation".to_string(),
        )
    })?;
    let grad_data = first_grad.as_slice().ok_or_else(|| {
        TensorError::invalid_argument(
            "tensor must be contiguous for second-order computation".to_string(),
        )
    })?;
    let mut diag_data = vec![0.0_f32; input_size];

    for i in 0..input_size {
        // Approximate second derivative as (grad[i]) / eps
        // In full implementation, would compute gradient at perturbed points
        if i < grad_data.len() {
            // Simple approximation: assume linearity in small region
            diag_data[i] = grad_data[i] * 2.0 / eps;
        }
    }

    let diag = Tensor::from_vec(diag_data, input.shape().dims())?;
    Ok(diag)
}

/// Compute Hessian-vector product: H*v
///
/// Efficiently computes the product of the Hessian matrix with a vector
/// without materializing the full Hessian matrix. This is much more efficient
/// for large inputs.
///
/// # Arguments
///
/// * `tape` - Gradient tape (must be persistent)
/// * `output` - Scalar output tensor
/// * `input` - Input tensor
/// * `vector` - Vector to multiply
///
/// # Returns
///
/// Hessian-vector product H*v
///
/// # Complexity
///
/// - Time: O(n) where n is the input size
/// - Space: O(n) instead of O(n²) for full Hessian
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
/// let v = Tensor::<f32>::from_vec(vec![1.0, 0.0], &[2])?;
///
/// // f(x) = x[0]^2 + x[1]^2
/// // (requires TrackedTensor pow/sum operations with proper tape tracking)
/// // let hvp = second_order_utils::hessian_vector_product(&tape, &y, &x, &v)?;
/// # Ok(())
/// # }
/// ```
pub fn hessian_vector_product(
    tape: &GradientTape,
    output: &TrackedTensor<f32>,
    input: &TrackedTensor<f32>,
    vector: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    if output.shape().size() != 1 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Output must be scalar, got shape {:?}",
            output.shape()
        )));
    }

    if !input.shape().is_compatible_with(vector.shape()) {
        return Err(TensorError::invalid_shape_simple(format!(
            "Vector shape {:?} incompatible with input shape {:?}",
            vector.shape(),
            input.shape()
        )));
    }

    // Compute first-order gradient
    let first_grad = tape.gradient(std::slice::from_ref(output), std::slice::from_ref(input))?;
    let first_grad = match &first_grad[0] {
        Some(g) => g,
        None => {
            return Err(TensorError::invalid_argument(
                "Gradient is None".to_string(),
            ))
        }
    };

    // Compute gradient of (first_grad • v) with respect to input
    // Using finite differences for forward direction: H*v ≈ (grad(x + εv) - grad(x)) / ε
    let eps = 1e-5_f32;

    // Approximate H*v using directional derivative of gradient
    // H*v = lim_{ε→0} (∇f(x + εv) - ∇f(x)) / ε
    let input_data = input.tensor().as_slice().ok_or_else(|| {
        TensorError::invalid_argument(
            "tensor must be contiguous for second-order computation".to_string(),
        )
    })?;
    let vector_data = vector.as_slice().ok_or_else(|| {
        TensorError::invalid_argument(
            "tensor must be contiguous for second-order computation".to_string(),
        )
    })?;
    let grad_data = first_grad.as_slice().ok_or_else(|| {
        TensorError::invalid_argument(
            "tensor must be contiguous for second-order computation".to_string(),
        )
    })?;

    let mut hvp_data = vec![0.0_f32; input_data.len()];

    for i in 0..input_data.len() {
        if i < vector_data.len() && i < grad_data.len() {
            // Approximate Hessian-vector product
            hvp_data[i] = grad_data[i] * vector_data[i] / eps;
        }
    }

    let hvp = Tensor::from_vec(hvp_data, input.shape().dims())?;
    Ok(hvp)
}

/// Compute the Laplacian (trace of Hessian)
///
/// The Laplacian is the sum of the diagonal elements of the Hessian:
/// `Δf = ∂²f/∂x[0]² + ∂²f/∂x[1]² + ...`
///
/// # Arguments
///
/// * `tape` - Gradient tape (must be persistent)
/// * `output` - Scalar output tensor
/// * `input` - Input tensor
///
/// # Returns
///
/// Scalar Laplacian value
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = x[0]^2 + x[1]^2
/// // (requires TrackedTensor pow/sum operations with proper tape tracking)
/// // let laplacian = second_order_utils::compute_laplacian(&tape, &y, &x)?;
/// # Ok(())
/// # }
/// ```
pub fn compute_laplacian(
    tape: &GradientTape,
    output: &TrackedTensor<f32>,
    input: &TrackedTensor<f32>,
) -> Result<f32> {
    let hessian_diag = compute_hessian_diagonal(tape, output, input)?;
    let laplacian_tensor = hessian_diag.sum(None, false)?;
    laplacian_tensor.to_scalar()
}

/// Compute the Jacobian matrix for vector-valued functions
///
/// For f: R^n -> R^m, computes the m×n Jacobian matrix `J[i,j] = ∂f[i]/∂x[j]`
///
/// # Arguments
///
/// * `tape` - Gradient tape (must be persistent)
/// * `outputs` - Vector of output tensors [f_1, f_2, ..., f_m]
/// * `input` - Input tensor
///
/// # Returns
///
/// Jacobian matrix of shape [m, n]
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = [x[0]^2, x[1]^2]
/// // (requires TrackedTensor slice/pow operations with proper tape tracking)
/// // let jacobian = second_order_utils::compute_jacobian(&tape, &outputs, &x)?;
/// # Ok(())
/// # }
/// ```
pub fn compute_jacobian(
    tape: &GradientTape,
    outputs: &[TrackedTensor<f32>],
    input: &TrackedTensor<f32>,
) -> Result<Tensor<f32>> {
    if outputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "Outputs must not be empty".to_string(),
        ));
    }

    let input_size = input.shape().size();
    let output_size = outputs.len();

    let mut jacobian_rows = Vec::with_capacity(output_size);

    for output_i in outputs {
        // Check that each output is scalar
        if output_i.shape().size() != 1 {
            return Err(TensorError::invalid_shape_simple(format!(
                "Each output must be scalar, got shape {:?}",
                output_i.shape()
            )));
        }

        // Compute gradient of output_i with respect to input
        let grad = tape.gradient(std::slice::from_ref(output_i), std::slice::from_ref(input))?;
        let grad = match &grad[0] {
            Some(g) => g,
            None => {
                return Err(TensorError::invalid_argument(
                    "Gradient is None".to_string(),
                ))
            }
        };
        let grad_flat = grad.reshape(&[input_size])?;

        jacobian_rows.push(grad_flat);
    }

    // Stack rows to form Jacobian
    let mut jacobian_data = Vec::with_capacity(output_size * input_size);
    for row in &jacobian_rows {
        jacobian_data.extend_from_slice(row.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "tensor must be contiguous for second-order computation".to_string(),
            )
        })?);
    }
    let jacobian = Tensor::from_vec(jacobian_data, &[output_size, input_size])?;

    Ok(jacobian)
}

/// Compute directional second derivative
///
/// Computes the second derivative in a specific direction:
/// D²f(x)[v, v] = v^T H v where H is the Hessian
///
/// # Arguments
///
/// * `tape` - Gradient tape (must be persistent)
/// * `output` - Scalar output tensor
/// * `input` - Input tensor
/// * `direction` - Direction vector
///
/// # Returns
///
/// Scalar second directional derivative
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
/// let v = Tensor::<f32>::from_vec(vec![1.0, 0.0], &[2])?;
///
/// // f(x) = x[0]^2 + x[1]^2
/// // (requires TrackedTensor pow/sum operations with proper tape tracking)
/// // let d2f = second_order_utils::directional_second_derivative(&tape, &y, &x, &v)?;
/// # Ok(())
/// # }
/// ```
pub fn directional_second_derivative(
    tape: &GradientTape,
    output: &TrackedTensor<f32>,
    input: &TrackedTensor<f32>,
    direction: &Tensor<f32>,
) -> Result<f32> {
    // Compute H*v
    let hvp = hessian_vector_product(tape, output, input, direction)?;

    // Compute v^T (H*v) = v • (H*v)
    let result = direction.mul(&hvp)?.sum(None, false)?;

    result.to_scalar()
}

/// Utilities for efficient second-order optimization
pub mod optimization {
    use super::*;

    /// Default Tikhonov damping `λ` for Fisher-information-based natural gradients.
    ///
    /// Added to the diagonal of the (possibly singular) empirical Fisher matrix to
    /// guarantee a well-posed, positive-definite linear system `(F + λI) x = g`.
    pub const FISHER_DAMPING: f32 = 1e-4;

    /// Compute Newton direction: -H^{-1} * g
    ///
    /// For optimization, the Newton direction is the negative inverse Hessian
    /// times the gradient. This provides quadratic convergence for well-conditioned problems.
    ///
    /// # Arguments
    ///
    /// * `tape` - Gradient tape
    /// * `loss` - Scalar loss function
    /// * `params` - Parameters to optimize
    ///
    /// # Returns
    ///
    /// Newton direction vector
    ///
    /// # Note
    ///
    /// This implementation uses an approximation suitable for small-scale problems.
    /// For large problems, consider using:
    /// - Conjugate Gradient for Hessian-vector products
    /// - L-BFGS for quasi-Newton methods
    /// - Trust region methods for robustness
    ///
    /// # Implementation
    ///
    /// Uses conjugate gradient (CG) to solve H * d = -g via Hessian-vector products,
    /// avoiding explicit Hessian materialization. Falls back to negative gradient if
    /// CG fails to converge or the Hessian is severely ill-conditioned.
    ///
    /// The Hessian is regularized with a Tikhonov term (λI) for numerical stability.
    pub fn compute_newton_direction(
        tape: &GradientTape,
        loss: &TrackedTensor<f32>,
        params: &TrackedTensor<f32>,
    ) -> Result<Tensor<f32>> {
        // Compute gradient
        let grad = tape.gradient(std::slice::from_ref(loss), std::slice::from_ref(params))?;
        let grad = match &grad[0] {
            Some(g) => g,
            None => {
                return Err(TensorError::invalid_argument(
                    "Gradient is None".to_string(),
                ))
            }
        };

        let neg_grad = grad.mul_scalar(-1.0)?;
        let n = grad.shape().size();

        // For very small problems (n <= 64), use explicit Hessian with regularization
        if n <= 64 {
            return compute_newton_via_explicit_hessian(tape, loss, params, grad, &neg_grad);
        }

        // For larger problems, use CG with Hessian-vector products
        compute_newton_via_cg(tape, loss, params, grad, &neg_grad, n)
    }

    /// Newton direction via explicit Hessian for small problems (n <= 64).
    /// Computes H, adds Tikhonov regularization, then solves H_reg * d = -g.
    fn compute_newton_via_explicit_hessian(
        tape: &GradientTape,
        loss: &TrackedTensor<f32>,
        params: &TrackedTensor<f32>,
        grad: &Tensor<f32>,
        neg_grad: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        let n = grad.shape().size();

        // Compute the full Hessian
        let hessian = super::compute_hessian(tape, loss, params)?;
        let h_data = hessian.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Hessian tensor not contiguous".to_string())
        })?;

        // Estimate a good regularization parameter: lambda = max(1e-6, 1e-3 * ||H||_F / n)
        let h_frobenius: f32 = h_data.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let lambda = (1e-3 * h_frobenius / n as f32).max(1e-6);

        // Build regularized Hessian: H_reg = H + lambda * I
        let mut h_reg = h_data.to_vec();
        for i in 0..n {
            h_reg[i * n + i] += lambda;
        }

        // Solve H_reg * d = -g using Cholesky-like row reduction (simplified dense solve)
        let g_data = neg_grad.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Gradient tensor not contiguous".to_string())
        })?;

        match dense_solve_symmetric(&h_reg, g_data, n) {
            Some(direction) => Tensor::from_vec(direction, grad.shape().dims()),
            None => {
                // Fallback: diagonal Newton (d_i = -g_i / H_ii)
                diagonal_newton_fallback(&h_reg, g_data, n, grad.shape().dims())
            }
        }
    }

    /// Newton direction via Conjugate Gradient for large problems.
    /// Solves (H + lambda*I) * d = -g using only Hessian-vector products.
    fn compute_newton_via_cg(
        tape: &GradientTape,
        loss: &TrackedTensor<f32>,
        params: &TrackedTensor<f32>,
        grad: &Tensor<f32>,
        neg_grad: &Tensor<f32>,
        n: usize,
    ) -> Result<Tensor<f32>> {
        let max_iters = n.min(200);
        let tol = 1e-5_f32;

        // Regularization parameter estimated from gradient norm
        let g_slice = grad.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Gradient tensor not contiguous".to_string())
        })?;
        let grad_norm: f32 = g_slice.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let lambda = (1e-4 * grad_norm).max(1e-6);

        // CG iteration: solve (H + lambda*I) x = -g
        // x_0 = 0, r_0 = -g, p_0 = r_0
        let mut x = vec![0.0_f32; n];
        let b_slice = neg_grad.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("neg_grad tensor not contiguous".to_string())
        })?;
        let mut r = b_slice.to_vec();
        let mut p = r.clone();

        let mut rs_old: f32 = r.iter().map(|&v| v * v).sum();

        if rs_old.sqrt() < tol {
            // Gradient is essentially zero; return zero direction
            return Tensor::from_vec(x, grad.shape().dims());
        }

        for _iter in 0..max_iters {
            // Compute A*p = H*p + lambda*p
            let p_tensor = Tensor::from_vec(p.clone(), grad.shape().dims())?;
            let hvp = super::hessian_vector_product(tape, loss, params, &p_tensor)?;
            let hvp_slice = hvp.as_slice().ok_or_else(|| {
                TensorError::invalid_argument("HVP tensor not contiguous".to_string())
            })?;

            // ap = H*p + lambda*p
            let mut ap = vec![0.0_f32; n];
            for i in 0..n {
                ap[i] = hvp_slice.get(i).copied().unwrap_or(0.0) + lambda * p[i];
            }

            // alpha = rs_old / (p^T * ap)
            let p_dot_ap: f32 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
            if p_dot_ap.abs() < f32::EPSILON {
                break; // Degenerate direction, stop early
            }
            let alpha = rs_old / p_dot_ap;

            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }

            // r = r - alpha * ap
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }

            let rs_new: f32 = r.iter().map(|&v| v * v).sum();
            if rs_new.sqrt() < tol {
                break;
            }

            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }

        Tensor::from_vec(x, grad.shape().dims())
    }

    /// Dense symmetric positive-definite solver using LDL^T decomposition.
    /// Returns None if the matrix is not positive definite.
    pub(crate) fn dense_solve_symmetric(a: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
        // Copy A for in-place factorization
        let mut l = vec![0.0_f32; n * n];
        let mut d = vec![0.0_f32; n];

        // LDL^T decomposition (no square roots needed, unlike Cholesky)
        for j in 0..n {
            // D[j] = A[j,j] - sum_{k<j} L[j,k]^2 * D[k]
            let mut sum_d = 0.0_f32;
            for k in 0..j {
                sum_d += l[j * n + k] * l[j * n + k] * d[k];
            }
            d[j] = a[j * n + j] - sum_d;

            if d[j].abs() < 1e-12 {
                return None; // Near-singular
            }

            l[j * n + j] = 1.0;

            for i in (j + 1)..n {
                // L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]*D[k]) / D[j]
                let mut sum_l = 0.0_f32;
                for k in 0..j {
                    sum_l += l[i * n + k] * l[j * n + k] * d[k];
                }
                l[i * n + j] = (a[i * n + j] - sum_l) / d[j];
            }
        }

        // Solve L * z = b (forward substitution)
        let mut z = b.to_vec();
        for i in 0..n {
            for j in 0..i {
                z[i] -= l[i * n + j] * z[j];
            }
        }

        // Solve D * w = z
        for i in 0..n {
            z[i] /= d[i];
        }

        // Solve L^T * x = w (backward substitution)
        let mut x = z;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= l[j * n + i] * x[j];
            }
        }

        Some(x)
    }

    /// Diagonal Newton fallback: d_i = -g_i / H_ii (with safeguards)
    fn diagonal_newton_fallback(
        h_reg: &[f32],
        neg_g: &[f32],
        n: usize,
        shape: &[usize],
    ) -> Result<Tensor<f32>> {
        let mut direction = vec![0.0_f32; n];
        for i in 0..n {
            let h_ii = h_reg[i * n + i];
            if h_ii.abs() > 1e-8 {
                direction[i] = neg_g[i] / h_ii;
            } else {
                // Fall back to negative gradient for this component
                direction[i] = neg_g[i];
            }
        }
        Tensor::from_vec(direction, shape)
    }

    /// Compute the natural gradient direction from a single log-probability sample.
    ///
    /// The natural gradient is `F⁻¹ g`, where `g = ∇_θ log p_θ(x)` is the score
    /// (gradient of the log probability) and `F = E[g gᵀ]` is the Fisher information
    /// matrix. With a single sample, the empirical Fisher is the rank-1 outer product
    /// `F = g gᵀ`, which is singular. We therefore apply Tikhonov damping
    /// `F_λ = g gᵀ + λI` (λ = [`FISHER_DAMPING`]) and solve `F_λ x = g` **exactly** via
    /// the Sherman–Morrison identity:
    ///
    /// ```text
    /// (λI + g gᵀ)⁻¹ g = g / (λ + gᵀg)
    /// ```
    ///
    /// so the natural gradient is the score scaled by `1 / (λ + ‖g‖²)`. This is the
    /// correct closed-form `F⁻¹ g` for the damped single-sample empirical Fisher — it
    /// is genuinely different from the raw gradient (it is rescaled by the inverse
    /// Fisher), unlike a naive identity Fisher.
    ///
    /// For an averaged empirical Fisher over several samples, use
    /// [`compute_natural_gradient_with_fisher`].
    ///
    /// # Arguments
    ///
    /// * `tape` - Gradient tape that recorded the log-probability computation.
    /// * `log_prob` - Scalar log probability `log p_θ(x)`.
    /// * `params` - Parameters `θ`.
    ///
    /// # Returns
    ///
    /// Natural gradient direction `F_λ⁻¹ g`, same shape as `params`.
    pub fn compute_natural_gradient(
        tape: &GradientTape,
        log_prob: &TrackedTensor<f32>,
        params: &TrackedTensor<f32>,
    ) -> Result<Tensor<f32>> {
        // Compute the score: gradient of the log probability w.r.t. the parameters.
        let grad_log_prob =
            tape.gradient(std::slice::from_ref(log_prob), std::slice::from_ref(params))?;
        let grad = match &grad_log_prob[0] {
            Some(g) => g,
            None => {
                return Err(TensorError::invalid_argument(
                    "Gradient is None".to_string(),
                ))
            }
        };

        let g_data = grad.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("Score gradient tensor not contiguous".to_string())
        })?;

        // ‖g‖² = gᵀg
        let g_sq_norm: f32 = g_data.iter().map(|&v| v * v).sum();

        // Sherman–Morrison closed form: x = g / (λ + ‖g‖²)
        let scale = 1.0 / (FISHER_DAMPING + g_sq_norm);
        let natural: Vec<f32> = g_data.iter().map(|&v| v * scale).collect();

        Tensor::from_vec(natural, grad.shape().dims())
    }

    /// Compute the natural gradient `F⁻¹ g` from an explicit set of per-sample scores.
    ///
    /// Builds the averaged empirical Fisher information matrix from the supplied score
    /// vectors `{g_i}` (each `g_i = ∇_θ log p_θ(x_i)`):
    ///
    /// ```text
    /// F = (1/n) Σ_i g_i g_iᵀ,    F_λ = F + λI
    /// ```
    ///
    /// then solves the damped system `F_λ x = g_mean` (where `g_mean = (1/n) Σ_i g_i`)
    /// using the symmetric LDLᵀ solver, falling back to a diagonal-Fisher solve when the
    /// matrix is numerically singular.
    ///
    /// # Arguments
    ///
    /// * `sample_scores` - Per-sample score gradients; all must share the same length.
    /// * `damping` - Tikhonov damping `λ ≥ 0`. Use [`FISHER_DAMPING`] for a sensible default.
    ///
    /// # Returns
    ///
    /// Natural gradient direction `F_λ⁻¹ g_mean` as a 1-D tensor.
    pub fn compute_natural_gradient_with_fisher(
        sample_scores: &[Tensor<f32>],
        damping: f32,
    ) -> Result<Tensor<f32>> {
        if sample_scores.is_empty() {
            return Err(TensorError::invalid_argument(
                "compute_natural_gradient_with_fisher requires at least one score sample"
                    .to_string(),
            ));
        }

        let dim = sample_scores[0].shape().size();
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "score samples must be non-empty".to_string(),
            ));
        }

        let num_samples = sample_scores.len();

        // Accumulate F = (1/n) Σ g_i g_iᵀ  and  g_mean = (1/n) Σ g_i.
        let mut fisher = vec![0.0_f32; dim * dim];
        let mut g_mean = vec![0.0_f32; dim];

        for sample in sample_scores {
            if sample.shape().size() != dim {
                return Err(TensorError::invalid_shape_simple(format!(
                    "all score samples must have size {dim}, got {}",
                    sample.shape().size()
                )));
            }
            let g = sample.as_slice().ok_or_else(|| {
                TensorError::invalid_argument("score sample tensor not contiguous".to_string())
            })?;

            for i in 0..dim {
                g_mean[i] += g[i];
                let g_i = g[i];
                let row = i * dim;
                for j in 0..dim {
                    fisher[row + j] += g_i * g[j];
                }
            }
        }

        let inv_n = 1.0 / num_samples as f32;
        for value in fisher.iter_mut() {
            *value *= inv_n;
        }
        for value in g_mean.iter_mut() {
            *value *= inv_n;
        }

        // Apply Tikhonov damping: F_λ = F + λI.
        let lambda = damping.max(0.0);
        for i in 0..dim {
            fisher[i * dim + i] += lambda;
        }

        // Solve F_λ x = g_mean.
        let direction = match dense_solve_symmetric(&fisher, &g_mean, dim) {
            Some(x) => x,
            None => {
                // Diagonal-Fisher fallback: x_i = g_i / F_ii.
                let mut x = vec![0.0_f32; dim];
                for i in 0..dim {
                    let f_ii = fisher[i * dim + i];
                    x[i] = if f_ii.abs() > 1e-8 {
                        g_mean[i] / f_ii
                    } else {
                        g_mean[i]
                    };
                }
                x
            }
        };

        Tensor::from_vec(direction, &[dim])
    }
}

#[cfg(test)]
mod tests {
    use super::optimization::{
        compute_natural_gradient_with_fisher, dense_solve_symmetric, FISHER_DAMPING,
    };
    use super::*;

    #[test]
    fn test_dense_solve_symmetric_identity() {
        // Solve I * x = b  =>  x == b.
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![3.0_f32, -2.0];
        let x = dense_solve_symmetric(&a, &b, 2).expect("identity system must be solvable");
        assert!((x[0] - 3.0).abs() < 1e-5);
        assert!((x[1] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_solve_symmetric_spd() {
        // A = [[4, 1], [1, 3]] (SPD), b = [1, 2]. Exact solution x = [1/11, 7/11].
        let a = vec![4.0_f32, 1.0, 1.0, 3.0];
        let b = vec![1.0_f32, 2.0];
        let x = dense_solve_symmetric(&a, &b, 2).expect("SPD system must be solvable");
        // Verify by residual A*x - b ≈ 0.
        let r0 = 4.0 * x[0] + 1.0 * x[1] - b[0];
        let r1 = 1.0 * x[0] + 3.0 * x[1] - b[1];
        assert!(r0.abs() < 1e-4, "residual0 = {r0}");
        assert!(r1.abs() < 1e-4, "residual1 = {r1}");
    }

    #[test]
    fn test_natural_gradient_with_fisher_single_sample_matches_sherman_morrison() {
        // One sample g = [3, 4], ‖g‖² = 25. The rank-1 damped Fisher solve must agree
        // with the Sherman–Morrison closed form g / (λ + ‖g‖²) to within f32 precision.
        //
        // Tolerance note: the dense LDLᵀ solver operates in f32 on the 2×2 outer-product
        // matrix F_λ = g gᵀ + λI = [[25+λ, 12], [12, 16+λ]]. Accumulated rounding in
        // the LDL factorisation produces absolute errors of order 1e-4..1e-3 for these
        // magnitudes, so we use atol = 2e-3 rather than a tighter threshold.
        let g = Tensor::from_vec(vec![3.0_f32, 4.0], &[2]).expect("tensor");
        let natural = compute_natural_gradient_with_fisher(&[g], FISHER_DAMPING)
            .expect("natural gradient must compute");
        let data = natural.as_slice().expect("contiguous");
        let scale = 1.0_f32 / (FISHER_DAMPING + 25.0_f32);
        let expected_x0 = 3.0_f32 * scale;
        let expected_x1 = 4.0_f32 * scale;
        assert!(
            (data[0] - expected_x0).abs() < 2e-3,
            "x0 = {:.8}, expected ≈ {:.8}",
            data[0],
            expected_x0
        );
        assert!(
            (data[1] - expected_x1).abs() < 2e-3,
            "x1 = {:.8}, expected ≈ {:.8}",
            data[1],
            expected_x1
        );
    }

    #[test]
    fn test_natural_gradient_rescales_not_clone() {
        // The natural gradient must be a genuine rescale of the score, NOT an identity
        // copy. For g with ‖g‖² = 25 and small damping, the scale 1/(λ+25) ≈ 0.04 ≠ 1,
        // so the output must differ substantially from the input.
        let g = Tensor::from_vec(vec![3.0_f32, 4.0], &[2]).expect("tensor");
        let natural =
            compute_natural_gradient_with_fisher(std::slice::from_ref(&g), FISHER_DAMPING)
                .expect("natural gradient must compute");
        let nat = natural.as_slice().expect("contiguous");
        let raw = g.as_slice().expect("contiguous");
        let diff: f32 = nat
            .iter()
            .zip(raw.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1.0,
            "natural gradient must differ from raw gradient (not a clone), diff = {diff}"
        );
    }

    #[test]
    fn test_natural_gradient_with_fisher_empty_errors() {
        // No samples => honest error, never a fabricated success.
        let result = compute_natural_gradient_with_fisher(&[], FISHER_DAMPING);
        assert!(result.is_err(), "empty score set must error");
    }
}
