use scirs2_core::numeric::Float;
use std::f64::consts::PI;
use tenflowers_core::{Result, Tensor, TensorError};

/// Special mathematical functions with gradient support
///
/// This module implements various special functions commonly used in machine learning
/// and scientific computing, along with their corresponding gradients for automatic
/// differentiation.
///
/// Gamma function backward pass
/// For y = Γ(x), grad_x = grad_y * Γ(x) * ψ(x)
/// where ψ(x) is the digamma function (derivative of log Γ(x))
pub fn gamma_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Compute Γ(x) using the gamma function
    let gamma_values = gamma_function(input)?;

    // Compute ψ(x) using the digamma function
    let digamma_values = digamma_function(input)?;

    // grad_x = grad_output * Γ(x) * ψ(x)
    let gamma_digamma = gamma_values.mul(&digamma_values)?;
    grad_output.mul(&gamma_digamma)
}

/// Log-gamma function backward pass
/// For y = log Γ(x), grad_x = grad_y * ψ(x)
/// where ψ(x) is the digamma function
pub fn lgamma_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Compute ψ(x) using the digamma function
    let digamma_values = digamma_function(input)?;

    // grad_x = grad_output * ψ(x)
    grad_output.mul(&digamma_values)
}

/// Digamma function backward pass
/// For y = ψ(x), grad_x = grad_y * ψ₁(x)
/// where ψ₁(x) is the trigamma function (derivative of ψ(x))
pub fn digamma_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Compute ψ₁(x) using the trigamma function
    let trigamma_values = trigamma_function(input)?;

    // grad_x = grad_output * ψ₁(x)
    grad_output.mul(&trigamma_values)
}

/// Error function backward pass
/// For y = erf(x), grad_x = grad_y * (2/√π) * exp(-x²)
pub fn erf_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Compute -x²
    let x_squared = input.mul(input)?;
    let neg_x_squared = x_squared.mul(&Tensor::from_scalar(-T::one()))?;

    // Compute exp(-x²)
    let exp_neg_x_squared = exp_function(&neg_x_squared)?;

    // Compute (2/√π) * exp(-x²)
    let sqrt_pi = T::from(PI).unwrap().sqrt();
    let two_over_sqrt_pi = T::from(2.0).unwrap() / sqrt_pi;
    let derivative = exp_neg_x_squared.mul(&Tensor::from_scalar(two_over_sqrt_pi))?;

    // grad_x = grad_output * (2/√π) * exp(-x²)
    grad_output.mul(&derivative)
}

/// Complementary error function backward pass
/// For y = erfc(x), grad_x = grad_y * (-2/√π) * exp(-x²)
pub fn erfc_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // erfc(x) = 1 - erf(x), so its derivative is -erf'(x)
    let erf_grad = erf_backward(grad_output, input)?;
    erf_grad.mul(&Tensor::from_scalar(-T::one()))
}

/// Bessel function of the first kind J₀(x) backward pass
/// For y = J₀(x), grad_x = grad_y * (-J₁(x))
pub fn bessel_j0_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Compute J₁(x)
    let j1_values = bessel_j1_function(input)?;

    // grad_x = grad_output * (-J₁(x))
    let neg_j1 = j1_values.mul(&Tensor::from_scalar(-T::one()))?;
    grad_output.mul(&neg_j1)
}

/// Bessel function of the first kind J₁(x) backward pass
/// For y = J₁(x), grad_x = grad_y * (J₀(x) - J₁(x)/x)
pub fn bessel_j1_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Compute J₀(x)
    let j0_values = bessel_j0_function(input)?;

    // Compute J₁(x)
    let j1_values = bessel_j1_function(input)?;

    // Compute J₁(x)/x, handling x=0 case
    let j1_over_x = safe_divide(&j1_values, input)?;

    // grad_x = grad_output * (J₀(x) - J₁(x)/x)
    let derivative = j0_values.sub(&j1_over_x)?;
    grad_output.mul(&derivative)
}

/// Beta function backward pass
/// For y = B(a,b) = Γ(a)Γ(b)/Γ(a+b), compute gradients w.r.t. both arguments
pub fn beta_backward<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    // ∂B/∂a = B(a,b) * (ψ(a) - ψ(a+b))
    // ∂B/∂b = B(a,b) * (ψ(b) - ψ(a+b))

    let beta_value = beta_function(a, b)?;
    let a_plus_b = a.add(b)?;

    let digamma_a = digamma_function(a)?;
    let digamma_b = digamma_function(b)?;
    let digamma_a_plus_b = digamma_function(&a_plus_b)?;

    // ∂B/∂a = B(a,b) * (ψ(a) - ψ(a+b))
    let grad_a_factor = digamma_a.sub(&digamma_a_plus_b)?;
    let grad_a_local = beta_value.mul(&grad_a_factor)?;
    let grad_a = grad_output.mul(&grad_a_local)?;

    // ∂B/∂b = B(a,b) * (ψ(b) - ψ(a+b))
    let grad_b_factor = digamma_b.sub(&digamma_a_plus_b)?;
    let grad_b_local = beta_value.mul(&grad_b_factor)?;
    let grad_b = grad_output.mul(&grad_b_local)?;

    Ok((grad_a, grad_b))
}

// Helper functions for computing the special functions themselves
// These would typically call into specialized numerical libraries or use approximations

/// Compute gamma function Γ(x) using Lanczos approximation
fn gamma_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // For now, use a simple approximation or placeholder
    // In a real implementation, this would use a high-quality gamma function
    // such as the Lanczos approximation

    // Simple approximation for demonstration
    // This should be replaced with a proper implementation
    let data = input
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| {
            if x <= T::zero() {
                T::infinity() // Gamma is undefined for non-positive integers
            } else {
                // Very crude approximation - should use proper Lanczos or Stirling
                gamma_lanczos_approx(x)
            }
        })
        .collect::<Vec<_>>();

    Tensor::from_vec(data, input.shape().dims())
}

/// Compute log-gamma function log Γ(x)
#[allow(dead_code)]
fn lgamma_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let gamma_values = gamma_function(input)?;
    // Compute log of gamma values
    let data = gamma_values
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| x.ln())
        .collect::<Vec<_>>();
    Tensor::from_vec(data, input.shape().dims())
}

/// Compute digamma function ψ(x) = d/dx log Γ(x)
fn digamma_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Digamma function approximation
    let data = input
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| digamma_approx(x))
        .collect::<Vec<_>>();

    Tensor::from_vec(data, input.shape().dims())
}

/// Compute trigamma function ψ₁(x) = d/dx ψ(x)
fn trigamma_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Trigamma function approximation
    let data = input
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| trigamma_approx(x))
        .collect::<Vec<_>>();

    Tensor::from_vec(data, input.shape().dims())
}

/// Compute exponential function
fn exp_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let data = input
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| x.exp())
        .collect::<Vec<_>>();
    Tensor::from_vec(data, input.shape().dims())
}

/// Compute Bessel function J₀(x)
fn bessel_j0_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let data = input
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| bessel_j0_approx(x))
        .collect::<Vec<_>>();

    Tensor::from_vec(data, input.shape().dims())
}

/// Compute Bessel function J₁(x)
fn bessel_j1_function<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let data = input
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| bessel_j1_approx(x))
        .collect::<Vec<_>>();

    Tensor::from_vec(data, input.shape().dims())
}

/// Compute Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
fn beta_function<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let gamma_a = gamma_function(a)?;
    let gamma_b = gamma_function(b)?;
    let a_plus_b = a.add(b)?;
    let gamma_a_plus_b = gamma_function(&a_plus_b)?;

    let numerator = gamma_a.mul(&gamma_b)?;
    numerator.div(&gamma_a_plus_b)
}

/// Safe division that handles x/0 cases
fn safe_divide<T>(numerator: &Tensor<T>, denominator: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let eps = T::from(1e-15).unwrap();
    let safe_denom_data = denominator
        .as_slice()
        .ok_or_else(|| TensorError::compute_error_simple("Cannot access tensor data".to_string()))?
        .iter()
        .map(|&x| {
            if x.abs() < eps {
                if x >= T::zero() {
                    eps
                } else {
                    -eps
                }
            } else {
                x
            }
        })
        .collect::<Vec<_>>();

    let safe_denominator = Tensor::from_vec(safe_denom_data, denominator.shape().dims())?;
    numerator.div(&safe_denominator)
}

// Numerical approximations for special functions
// These are simplified implementations - production code should use optimized libraries

fn gamma_lanczos_approx<T: Float>(x: T) -> T {
    // Simplified Lanczos approximation
    // This is a very basic implementation - use a proper library in production
    if x < T::one() {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        let pi = T::from(PI).unwrap();
        let pi_z = pi * x;
        let sin_pi_z = pi_z.sin();
        if sin_pi_z.abs() < T::from(1e-15).unwrap() {
            T::infinity()
        } else {
            pi / (sin_pi_z * gamma_lanczos_approx(T::one() - x))
        }
    } else {
        // Stirling's approximation for large x
        let two_pi = T::from(2.0 * PI).unwrap();
        let sqrt_two_pi = two_pi.sqrt();
        let x_minus_1 = x - T::one();
        sqrt_two_pi * x_minus_1.powf(x_minus_1) * (-x_minus_1).exp() / x_minus_1.sqrt()
    }
}

fn digamma_approx<T: Float>(x: T) -> T {
    // Simple approximation of digamma function
    // For large x: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴)
    if x > T::from(5.0).unwrap() {
        let ln_x = x.ln();
        let inv_x = T::one() / x;
        let inv_x2 = inv_x * inv_x;
        let inv_x4 = inv_x2 * inv_x2;

        ln_x - T::from(0.5).unwrap() * inv_x - T::from(1.0 / 12.0).unwrap() * inv_x2
            + T::from(1.0 / 120.0).unwrap() * inv_x4
    } else {
        // Use recurrence relation for small x
        digamma_approx(x + T::one()) - T::one() / x
    }
}

fn trigamma_approx<T: Float>(x: T) -> T {
    // Simple approximation of trigamma function
    // For large x: ψ₁(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵)
    if x > T::from(5.0).unwrap() {
        let inv_x = T::one() / x;
        let inv_x2 = inv_x * inv_x;
        let inv_x3 = inv_x2 * inv_x;
        let inv_x5 = inv_x3 * inv_x2;

        inv_x + T::from(0.5).unwrap() * inv_x2 + T::from(1.0 / 6.0).unwrap() * inv_x3
            - T::from(1.0 / 30.0).unwrap() * inv_x5
    } else {
        // Use recurrence relation for small x
        trigamma_approx(x + T::one()) + T::one() / (x * x)
    }
}

fn bessel_j0_approx<T: Float>(x: T) -> T {
    // Simple approximation for Bessel J₀(x)
    // For small x: J₀(x) ≈ 1 - x²/4 + x⁴/64
    // For large x: J₀(x) ≈ √(2/(πx)) * cos(x - π/4)
    let abs_x = x.abs();
    if abs_x < T::from(3.0).unwrap() {
        let x2 = x * x;
        let x4 = x2 * x2;
        T::one() - x2 / T::from(4.0).unwrap() + x4 / T::from(64.0).unwrap()
    } else {
        let pi = T::from(PI).unwrap();
        let sqrt_factor = (T::from(2.0).unwrap() / (pi * abs_x)).sqrt();
        let phase = abs_x - pi / T::from(4.0).unwrap();
        sqrt_factor * phase.cos()
    }
}

fn bessel_j1_approx<T: Float>(x: T) -> T {
    // Simple approximation for Bessel J₁(x)
    // For small x: J₁(x) ≈ x/2 - x³/16 + x⁵/384
    // For large x: J₁(x) ≈ √(2/(πx)) * sin(x - 3π/4)
    let abs_x = x.abs();
    if abs_x < T::from(3.0).unwrap() {
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        x / T::from(2.0).unwrap() - x3 / T::from(16.0).unwrap() + x5 / T::from(384.0).unwrap()
    } else {
        let pi = T::from(PI).unwrap();
        let sqrt_factor = (T::from(2.0).unwrap() / (pi * abs_x)).sqrt();
        let phase = abs_x - T::from(3.0).unwrap() * pi / T::from(4.0).unwrap();
        let result = sqrt_factor * phase.sin();
        if x < T::zero() {
            -result
        } else {
            result
        }
    }
}
