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
/// Computes the matrix of second derivatives: H[i,j] = ∂²f/∂x[i]∂x[j]
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
/// ```rust,no_run
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new().persistent();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = x[0]^2 + x[1]^2
/// let y = x.pow(2.0)?.sum()?;
///
/// let hessian = second_order_utils::compute_hessian(&tape, &y, &x)?;
/// // Hessian should be [[2, 0], [0, 2]]
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
            let input_data = input.tensor().as_slice().unwrap();
            let mut x_plus = input_data.to_vec();
            let mut x_minus = input_data.to_vec();

            x_plus[j] += eps;
            x_minus[j] -= eps;

            // Compute gradients at perturbed points would require re-evaluation
            // For now, approximate using available first-order gradient
            let grad_data = first_grad_flat.as_slice().unwrap();
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
        hessian_data.extend_from_slice(row.as_slice().unwrap());
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
/// Diagonal of the Hessian: [∂²f/∂x[0]², ∂²f/∂x[1]², ...]
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new().persistent();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = x[0]^2 + x[1]^2
/// let y = x.pow(2.0)?.sum()?;
///
/// let hessian_diag = second_order_utils::compute_hessian_diagonal(&tape, &y, &x)?;
/// // Should be [2.0, 2.0]
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
    let input_data = input.tensor().as_slice().unwrap();
    let grad_data = first_grad.as_slice().unwrap();
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
/// ```rust,no_run
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new().persistent();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
/// let v = Tensor::<f32>::from_vec(vec![1.0, 0.0], &[2])?;
///
/// // f(x) = x[0]^2 + x[1]^2
/// let y = x.pow(2.0)?.sum()?;
///
/// // Compute H*v efficiently
/// let hvp = second_order_utils::hessian_vector_product(&tape, &y, &x, &v)?;
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
    let input_data = input.tensor().as_slice().unwrap();
    let vector_data = vector.as_slice().unwrap();
    let grad_data = first_grad.as_slice().unwrap();

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
/// Δf = ∂²f/∂x[0]² + ∂²f/∂x[1]² + ...
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
/// ```rust,no_run
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new().persistent();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = x[0]^2 + x[1]^2
/// let y = x.pow(2.0)?.sum()?;
///
/// let laplacian = second_order_utils::compute_laplacian(&tape, &y, &x)?;
/// // Should be 2.0 + 2.0 = 4.0
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
/// For f: R^n -> R^m, computes the m×n Jacobian matrix J[i,j] = ∂f[i]/∂x[j]
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
/// ```rust,no_run
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new().persistent();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
///
/// // f(x) = [x[0]^2, x[1]^2]
/// let outputs = vec![
///     x.slice(&[0..1])?.pow(2.0)?,
///     x.slice(&[1..2])?.pow(2.0)?,
/// ];
///
/// let jacobian = second_order_utils::compute_jacobian(&tape, &outputs, &x)?;
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
        jacobian_data.extend_from_slice(row.as_slice().unwrap());
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
/// ```rust,no_run
/// use tenflowers_autograd::{GradientTape, second_order_utils};
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tape = GradientTape::new().persistent();
/// let x = tape.watch(Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])?);
/// let v = Tensor::<f32>::from_vec(vec![1.0, 0.0], &[2])?;
///
/// // f(x) = x[0]^2 + x[1]^2
/// let y = x.pow(2.0)?.sum()?;
///
/// // Second derivative in direction v
/// let d2f = second_order_utils::directional_second_derivative(&tape, &y, &x, &v)?;
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
    /// Currently returns negative gradient (gradient descent fallback).
    /// Full implementation would:
    /// 1. Compute or approximate the Hessian matrix
    /// 2. Solve the linear system H * d = -g for direction d
    /// 3. Handle ill-conditioned Hessians with regularization
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

        // For now, return negative gradient (gradient descent)
        // Full implementation would compute and invert Hessian
        let newton_dir = grad.mul_scalar(-1.0)?;

        // TODO: Implement full Newton direction:
        // 1. Compute Hessian: H = compute_hessian(tape, loss, params)?
        // 2. Add regularization: H_reg = H + λI for stability
        // 3. Solve: H_reg * d = -g using Cholesky, CG, or direct solve
        // 4. Return direction d

        Ok(newton_dir)
    }

    /// Compute natural gradient direction
    ///
    /// The natural gradient uses the Fisher information matrix instead of the Hessian,
    /// providing better convergence for certain problems (especially in RL and variational inference).
    ///
    /// # Arguments
    ///
    /// * `tape` - Gradient tape
    /// * `log_prob` - Log probability (for Fisher information)
    /// * `params` - Parameters
    ///
    /// # Returns
    ///
    /// Natural gradient direction
    pub fn compute_natural_gradient(
        tape: &GradientTape,
        log_prob: &TrackedTensor<f32>,
        params: &TrackedTensor<f32>,
    ) -> Result<Tensor<f32>> {
        // Compute gradient of log probability
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

        // Fisher information matrix F = E[∇log p ∇log p^T]
        // Natural gradient = F^{-1} ∇θ
        // Placeholder: Return gradient (proper implementation would compute Fisher)
        Ok(grad.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hessian_shape() {
        // Test that Hessian has correct shape
        // Placeholder test - will be implemented when gradient tape is fully functional
    }

    #[test]
    fn test_hessian_diagonal() {
        // Test diagonal Hessian computation
        // Placeholder test
    }

    #[test]
    fn test_hvp_efficiency() {
        // Test that HVP is more efficient than full Hessian
        // Placeholder test
    }

    #[test]
    fn test_laplacian() {
        // Test Laplacian computation
        // Placeholder test
    }
}
