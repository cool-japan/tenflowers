//! # Forward-Mode Automatic Differentiation via Scalar Dual Numbers
//!
//! This module provides `ForwardGrad`, a scalar dual-number type for forward-mode
//! automatic differentiation. Dual numbers extend the real numbers with an
//! infinitesimal component `ε` satisfying `ε² = 0`, enabling exact first-order
//! derivatives to be propagated alongside function values in a single forward pass.
//!
//! ## Dual Number Arithmetic
//!
//! A dual number has the form `a + bε` where:
//! - `a` is the **primal value** (the function value)
//! - `b` is the **tangent** (the directional derivative)
//!
//! Arithmetic follows naturally from `ε² = 0`:
//! - `(a + bε) + (c + dε) = (a+c) + (b+d)ε`
//! - `(a + bε) * (c + dε) = ac + (ad + bc)ε`  (product rule)
//! - `(a + bε) / (c + dε) = a/c + ((bc - ad)/c²)ε`  (quotient rule)
//!
//! Transcendental functions follow from the chain rule: for `f(a + bε) = f(a) + f'(a)·bε`
//!
//! ## Usage
//!
//! ```rust
//! use tenflowers_autograd::forward_grad::ForwardGrad;
//!
//! // Compute df/dx for f(x) = x^2 * sin(x) at x = 2.0
//! // Set value=2.0, tangent=1.0 to compute df/dx
//! let x = ForwardGrad::new(2.0_f32, 1.0);
//! let result = (x * x) * x.sin();
//! println!("f(2) = {}", result.value);
//! println!("f'(2) = {}", result.tangent);
//! ```
//!
//! ## Comparison with Reverse-Mode
//!
//! Forward-mode AD is most efficient for functions `f: R^n -> R^m` where `n << m`
//! (few inputs, many outputs — e.g., Jacobian-vector products). Reverse-mode is
//! more efficient for `n >> m` (many inputs, few outputs — e.g., neural network training).
//! For scalar functions of a single variable, both modes have equal cost.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A scalar dual number for forward-mode automatic differentiation.
///
/// `ForwardGrad` holds a primal `value` and a `tangent` (directional derivative).
/// All arithmetic and transcendental operations propagate the tangent via the chain rule.
///
/// # Invariants
///
/// - `value` holds the result of evaluating the expression at the input point.
/// - `tangent` holds `df/dx * dx/dt` where `t` is the scalar "seed" direction.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::forward_grad::ForwardGrad;
///
/// // f(x) = x³, f'(x) = 3x²
/// // At x = 2.0: f(2) = 8, f'(2) = 12
/// let x = ForwardGrad::new(2.0_f32, 1.0);
/// let result = x * x * x;
/// assert!((result.value - 8.0).abs() < 1e-6);
/// assert!((result.tangent - 12.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardGrad {
    /// The primal value: `f(x)`.
    pub value: f32,
    /// The tangent: `df/dx * seed`, where `seed` is the input direction.
    pub tangent: f32,
}

impl ForwardGrad {
    /// Create a new dual number with the given value and tangent.
    ///
    /// To differentiate `f` with respect to a single variable `x`, construct
    /// `ForwardGrad::new(x_value, 1.0)` for that variable and use `0.0` tangent
    /// for all constants.
    ///
    /// # Arguments
    ///
    /// * `value` - The primal value at the evaluation point.
    /// * `tangent` - The seed tangent (typically `1.0` for the variable being differentiated).
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// let x = ForwardGrad::new(3.0_f32, 1.0); // differentiate with respect to x
    /// let c = ForwardGrad::new(5.0_f32, 0.0); // constant, no derivative
    /// let f = x * c; // f(x) = 5x
    /// assert!((f.value - 15.0).abs() < 1e-6); // f(3) = 15
    /// assert!((f.tangent - 5.0).abs() < 1e-6); // f'(3) = 5
    /// ```
    #[inline]
    pub fn new(value: f32, tangent: f32) -> Self {
        Self { value, tangent }
    }

    /// Create a dual number representing a constant (zero tangent).
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// let c = ForwardGrad::constant(3.14_f32);
    /// assert_eq!(c.tangent, 0.0);
    /// ```
    #[inline]
    pub fn constant(value: f32) -> Self {
        Self { value, tangent: 0.0 }
    }

    /// Create a dual number representing an input variable (unit tangent).
    ///
    /// Use this when differentiating `f(x)` with respect to `x`. For multi-variable
    /// functions, use [`new`](Self::new) with explicit tangent seeds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// let x = ForwardGrad::variable(2.0_f32);
    /// assert_eq!(x.value, 2.0);
    /// assert_eq!(x.tangent, 1.0);
    /// ```
    #[inline]
    pub fn variable(value: f32) -> Self {
        Self { value, tangent: 1.0 }
    }

    /// Compute sine with its tangent propagation.
    ///
    /// Chain rule: `d/dx sin(f(x)) = cos(f(x)) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx sin(x) = cos(x); at x = 0: derivative = 1.0
    /// let x = ForwardGrad::variable(0.0_f32);
    /// let s = x.sin();
    /// assert!((s.value - 0.0_f32.sin()).abs() < 1e-6);
    /// assert!((s.tangent - 1.0).abs() < 1e-6); // cos(0) = 1
    /// ```
    #[inline]
    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            tangent: self.value.cos() * self.tangent,
        }
    }

    /// Compute cosine with its tangent propagation.
    ///
    /// Chain rule: `d/dx cos(f(x)) = -sin(f(x)) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx cos(x) = -sin(x); at x = 0: derivative = 0.0
    /// let x = ForwardGrad::variable(0.0_f32);
    /// let c = x.cos();
    /// assert!((c.value - 1.0).abs() < 1e-6); // cos(0) = 1
    /// assert!((c.tangent - 0.0).abs() < 1e-6); // -sin(0) = 0
    /// ```
    #[inline]
    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            tangent: -self.value.sin() * self.tangent,
        }
    }

    /// Compute tangent with its tangent propagation.
    ///
    /// Chain rule: `d/dx tan(f(x)) = (1 / cos²(f(x))) * f'(x) = sec²(f(x)) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx tan(x) = sec²(x); at x = 0: derivative = 1.0
    /// let x = ForwardGrad::variable(0.0_f32);
    /// let t = x.tan();
    /// assert!((t.value - 0.0_f32.tan()).abs() < 1e-6);
    /// assert!((t.tangent - 1.0).abs() < 1e-6); // sec²(0) = 1
    /// ```
    #[inline]
    pub fn tan(self) -> Self {
        let cos_val = self.value.cos();
        Self {
            value: self.value.tan(),
            tangent: self.tangent / (cos_val * cos_val),
        }
    }

    /// Compute the natural exponential with tangent propagation.
    ///
    /// Chain rule: `d/dx exp(f(x)) = exp(f(x)) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx exp(x) = exp(x); at x = 0: derivative = 1.0
    /// let x = ForwardGrad::variable(0.0_f32);
    /// let e = x.exp();
    /// assert!((e.value - 1.0).abs() < 1e-6); // exp(0) = 1
    /// assert!((e.tangent - 1.0).abs() < 1e-6); // exp(0) * 1 = 1
    /// ```
    #[inline]
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Self {
            value: exp_val,
            tangent: exp_val * self.tangent,
        }
    }

    /// Compute the natural logarithm with tangent propagation.
    ///
    /// Chain rule: `d/dx ln(f(x)) = f'(x) / f(x)`
    ///
    /// # Panics
    ///
    /// The tangent is set to `NaN` when `value <= 0` because the derivative of `ln`
    /// is undefined at non-positive reals. The value will be `f32::NEG_INFINITY` or
    /// `f32::NAN` as per IEEE 754 arithmetic.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx ln(x) = 1/x; at x = 1: derivative = 1.0
    /// let x = ForwardGrad::variable(1.0_f32);
    /// let l = x.ln();
    /// assert!((l.value - 0.0).abs() < 1e-6); // ln(1) = 0
    /// assert!((l.tangent - 1.0).abs() < 1e-6); // 1/1 = 1
    /// ```
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            tangent: self.tangent / self.value,
        }
    }

    /// Compute the square root with tangent propagation.
    ///
    /// Chain rule: `d/dx sqrt(f(x)) = f'(x) / (2 * sqrt(f(x)))`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx sqrt(x) = 1/(2*sqrt(x)); at x = 4: derivative = 0.25
    /// let x = ForwardGrad::variable(4.0_f32);
    /// let s = x.sqrt();
    /// assert!((s.value - 2.0).abs() < 1e-6); // sqrt(4) = 2
    /// assert!((s.tangent - 0.25).abs() < 1e-6); // 1/(2*2) = 0.25
    /// ```
    #[inline]
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.value.sqrt();
        Self {
            value: sqrt_val,
            tangent: self.tangent / (2.0 * sqrt_val),
        }
    }

    /// Compute `self` raised to a scalar power with tangent propagation.
    ///
    /// Chain rule: `d/dx f(x)^n = n * f(x)^(n-1) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx x^3 = 3x²; at x = 2: derivative = 12
    /// let x = ForwardGrad::variable(2.0_f32);
    /// let p = x.powi(3);
    /// assert!((p.value - 8.0).abs() < 1e-6);
    /// assert!((p.tangent - 12.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self {
            value: self.value.powi(n),
            tangent: (n as f32) * self.value.powi(n - 1) * self.tangent,
        }
    }

    /// Compute `self` raised to a floating-point power with tangent propagation.
    ///
    /// Chain rule: `d/dx f(x)^p = p * f(x)^(p-1) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx x^0.5 = 0.5 * x^(-0.5); at x = 4: derivative = 0.25
    /// let x = ForwardGrad::variable(4.0_f32);
    /// let p = x.powf(0.5);
    /// assert!((p.value - 2.0).abs() < 1e-6);
    /// assert!((p.tangent - 0.25).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn powf(self, p: f32) -> Self {
        Self {
            value: self.value.powf(p),
            tangent: p * self.value.powf(p - 1.0) * self.tangent,
        }
    }

    /// Compute hyperbolic sine with tangent propagation.
    ///
    /// Chain rule: `d/dx sinh(f(x)) = cosh(f(x)) * f'(x)`
    #[inline]
    pub fn sinh(self) -> Self {
        Self {
            value: self.value.sinh(),
            tangent: self.value.cosh() * self.tangent,
        }
    }

    /// Compute hyperbolic cosine with tangent propagation.
    ///
    /// Chain rule: `d/dx cosh(f(x)) = sinh(f(x)) * f'(x)`
    #[inline]
    pub fn cosh(self) -> Self {
        Self {
            value: self.value.cosh(),
            tangent: self.value.sinh() * self.tangent,
        }
    }

    /// Compute hyperbolic tangent with tangent propagation.
    ///
    /// Chain rule: `d/dx tanh(f(x)) = (1 - tanh²(f(x))) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx tanh(x) = 1 - tanh²(x); at x = 0: derivative = 1.0
    /// let x = ForwardGrad::variable(0.0_f32);
    /// let t = x.tanh();
    /// assert!((t.value - 0.0).abs() < 1e-6);
    /// assert!((t.tangent - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn tanh(self) -> Self {
        let tanh_val = self.value.tanh();
        Self {
            value: tanh_val,
            tangent: (1.0 - tanh_val * tanh_val) * self.tangent,
        }
    }

    /// Compute the sigmoid function `1 / (1 + exp(-x))` with tangent propagation.
    ///
    /// Chain rule: `d/dx sigmoid(f(x)) = sigmoid(f(x)) * (1 - sigmoid(f(x))) * f'(x)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)); at x = 0: derivative = 0.25
    /// let x = ForwardGrad::variable(0.0_f32);
    /// let s = x.sigmoid();
    /// assert!((s.value - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
    /// assert!((s.tangent - 0.25).abs() < 1e-6); // 0.5 * 0.5 = 0.25
    /// ```
    #[inline]
    pub fn sigmoid(self) -> Self {
        let sig_val = 1.0 / (1.0 + (-self.value).exp());
        Self {
            value: sig_val,
            tangent: sig_val * (1.0 - sig_val) * self.tangent,
        }
    }

    /// Compute the ReLU activation `max(0, x)` with tangent propagation.
    ///
    /// The derivative is 1 for `x > 0`, 0 for `x < 0`, and conventionally 0 at `x = 0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tenflowers_autograd::forward_grad::ForwardGrad;
    ///
    /// let x_pos = ForwardGrad::variable(2.0_f32);
    /// let r_pos = x_pos.relu();
    /// assert!((r_pos.value - 2.0).abs() < 1e-6);
    /// assert!((r_pos.tangent - 1.0).abs() < 1e-6);
    ///
    /// let x_neg = ForwardGrad::variable(-1.0_f32);
    /// let r_neg = x_neg.relu();
    /// assert!((r_neg.value - 0.0).abs() < 1e-6);
    /// assert!((r_neg.tangent - 0.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn relu(self) -> Self {
        if self.value > 0.0 {
            self
        } else {
            Self { value: 0.0, tangent: 0.0 }
        }
    }

    /// Absolute value of the dual number.
    ///
    /// The derivative is `1` for positive, `-1` for negative, and `0` at zero (subgradient).
    #[inline]
    pub fn abs(self) -> Self {
        if self.value >= 0.0 {
            self
        } else {
            Self {
                value: -self.value,
                tangent: -self.tangent,
            }
        }
    }

    /// Compute `exp(self) - 1` with tangent propagation.
    ///
    /// More numerically stable than `exp(x) - 1` for small `x`.
    /// Chain rule: `d/dx expm1(f(x)) = exp(f(x)) * f'(x)`
    #[inline]
    pub fn exp_m1(self) -> Self {
        Self {
            value: self.value.exp_m1(),
            tangent: self.value.exp() * self.tangent,
        }
    }

    /// Compute `ln(1 + self)` with tangent propagation.
    ///
    /// More numerically stable than `ln(1 + x)` for small `x`.
    /// Chain rule: `d/dx ln1p(f(x)) = f'(x) / (1 + f(x))`
    #[inline]
    pub fn ln_1p(self) -> Self {
        Self {
            value: self.value.ln_1p(),
            tangent: self.tangent / (1.0 + self.value),
        }
    }

    /// Return the primal value.
    #[inline]
    pub fn value(self) -> f32 {
        self.value
    }

    /// Return the tangent (derivative).
    #[inline]
    pub fn tangent(self) -> f32 {
        self.tangent
    }

    /// Return the square of the dual number: equivalent to `self * self` but faster.
    ///
    /// `d/dx f(x)^2 = 2 * f(x) * f'(x)`
    #[inline]
    pub fn sq(self) -> Self {
        Self {
            value: self.value * self.value,
            tangent: 2.0 * self.value * self.tangent,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Standard operator trait implementations
// ───────────────────────────────────────────────────────────────────────────

impl Add for ForwardGrad {
    type Output = Self;

    /// Addition with tangent propagation: `(a + bε) + (c + dε) = (a+c) + (b+d)ε`
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl Sub for ForwardGrad {
    type Output = Self;

    /// Subtraction with tangent propagation: `(a + bε) - (c + dε) = (a-c) + (b-d)ε`
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl Mul for ForwardGrad {
    type Output = Self;

    /// Multiplication via the product rule: `(a + bε)(c + dε) = ac + (ad + bc)ε`
    ///
    /// The `ε²` term vanishes since `ε² = 0`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}

impl Div for ForwardGrad {
    type Output = Self;

    /// Division via the quotient rule:
    /// `(a + bε) / (c + dε) = a/c + ((bc - ad)/c²)ε`
    ///
    /// # Note
    ///
    /// Division by zero follows IEEE 754 — the `value` becomes `±Inf` or `NaN`
    /// and the `tangent` likewise follows IEEE 754 rules.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            tangent: (self.tangent * rhs.value - self.value * rhs.tangent) / denom,
        }
    }
}

impl Neg for ForwardGrad {
    type Output = Self;

    /// Negation: `-(a + bε) = -a + (-b)ε`
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}

// Scalar operations: ForwardGrad op f32 and f32 op ForwardGrad

impl Add<f32> for ForwardGrad {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f32) -> Self {
        Self { value: self.value + rhs, tangent: self.tangent }
    }
}

impl Add<ForwardGrad> for f32 {
    type Output = ForwardGrad;

    #[inline]
    fn add(self, rhs: ForwardGrad) -> ForwardGrad {
        ForwardGrad { value: self + rhs.value, tangent: rhs.tangent }
    }
}

impl Sub<f32> for ForwardGrad {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f32) -> Self {
        Self { value: self.value - rhs, tangent: self.tangent }
    }
}

impl Sub<ForwardGrad> for f32 {
    type Output = ForwardGrad;

    #[inline]
    fn sub(self, rhs: ForwardGrad) -> ForwardGrad {
        ForwardGrad { value: self - rhs.value, tangent: -rhs.tangent }
    }
}

impl Mul<f32> for ForwardGrad {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self { value: self.value * rhs, tangent: self.tangent * rhs }
    }
}

impl Mul<ForwardGrad> for f32 {
    type Output = ForwardGrad;

    #[inline]
    fn mul(self, rhs: ForwardGrad) -> ForwardGrad {
        ForwardGrad { value: self * rhs.value, tangent: self * rhs.tangent }
    }
}

impl Div<f32> for ForwardGrad {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self {
        Self { value: self.value / rhs, tangent: self.tangent / rhs }
    }
}

impl Div<ForwardGrad> for f32 {
    type Output = ForwardGrad;

    #[inline]
    fn div(self, rhs: ForwardGrad) -> ForwardGrad {
        // d/dx (c / f(x)) = -c * f'(x) / f(x)^2
        let denom = rhs.value * rhs.value;
        ForwardGrad {
            value: self / rhs.value,
            tangent: (-self * rhs.tangent) / denom,
        }
    }
}

impl fmt::Display for ForwardGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ForwardGrad {{ value: {}, tangent: {} }}", self.value, self.tangent)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Jacobian-Vector Product helper for multi-variable scalar functions
// ───────────────────────────────────────────────────────────────────────────

/// Compute the directional derivative of a scalar function in a given direction.
///
/// Given a function `f: R^n -> R` and a direction `v ∈ R^n`, this computes
/// the directional derivative `∇f(x) · v` using a single forward pass.
///
/// # Arguments
///
/// * `values` - Input point `x ∈ R^n`.
/// * `direction` - Direction vector `v ∈ R^n` (need not be a unit vector).
/// * `f` - Function accepting `&[ForwardGrad]` and returning `ForwardGrad`.
///
/// # Returns
///
/// A tuple `(f_value, directional_derivative)`.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::forward_grad::{ForwardGrad, directional_derivative};
///
/// // f(x, y) = x^2 + y^2; ∇f = (2x, 2y)
/// // At (1, 2) in direction (1, 0): directional derivative = 2*1 = 2
/// let (val, dval) = directional_derivative(
///     &[1.0, 2.0],
///     &[1.0, 0.0],
///     |inputs| inputs[0] * inputs[0] + inputs[1] * inputs[1],
/// );
/// assert!((val - 5.0).abs() < 1e-6);
/// assert!((dval - 2.0).abs() < 1e-6);
/// ```
pub fn directional_derivative<F>(values: &[f32], direction: &[f32], f: F) -> (f32, f32)
where
    F: FnOnce(&[ForwardGrad]) -> ForwardGrad,
{
    let dual_inputs: Vec<ForwardGrad> = values
        .iter()
        .zip(direction.iter())
        .map(|(&v, &d)| ForwardGrad::new(v, d))
        .collect();

    let result = f(&dual_inputs);
    (result.value, result.tangent)
}

/// Compute the full gradient of a scalar function at a given point.
///
/// This uses `n` forward passes (one per input dimension) to compute
/// the full gradient vector `∇f(x) ∈ R^n`.
///
/// For functions with many inputs and a scalar output, reverse-mode AD
/// is more efficient (one pass). Use this for small `n` or when reverse-mode
/// is unavailable.
///
/// # Arguments
///
/// * `values` - Input point `x ∈ R^n`.
/// * `f` - Function accepting `&[ForwardGrad]` and returning `ForwardGrad`.
///
/// # Returns
///
/// A tuple `(f_value, gradient_vector)` where `gradient_vector[i] = ∂f/∂x_i`.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::forward_grad::{ForwardGrad, full_gradient};
///
/// // f(x, y) = x^2 + 3*y; gradient = (2x, 3)
/// // At (2, 1): gradient = (4, 3)
/// let (val, grad) = full_gradient(&[2.0, 1.0], |inputs| {
///     inputs[0] * inputs[0] + ForwardGrad::constant(3.0) * inputs[1]
/// });
/// assert!((val - 7.0).abs() < 1e-6);
/// assert!((grad[0] - 4.0).abs() < 1e-6);
/// assert!((grad[1] - 3.0).abs() < 1e-6);
/// ```
pub fn full_gradient<F>(values: &[f32], f: F) -> (f32, Vec<f32>)
where
    F: Fn(&[ForwardGrad]) -> ForwardGrad,
{
    let n = values.len();
    let mut gradient = vec![0.0_f32; n];

    // Evaluate once for the primal value
    let primal_inputs: Vec<ForwardGrad> = values.iter().map(|&v| ForwardGrad::constant(v)).collect();
    let primal_result = f(&primal_inputs);
    let f_value = primal_result.value;

    // One forward pass per input dimension
    for i in 0..n {
        let mut direction = vec![0.0_f32; n];
        direction[i] = 1.0;

        let dual_inputs: Vec<ForwardGrad> = values
            .iter()
            .zip(direction.iter())
            .map(|(&v, &d)| ForwardGrad::new(v, d))
            .collect();

        let result = f(&dual_inputs);
        gradient[i] = result.tangent;
    }

    (f_value, gradient)
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    // Helper: numerical derivative via central finite differences
    fn numerical_deriv<F: Fn(f32) -> f32>(f: F, x: f32) -> f32 {
        let h = 1e-4_f32;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    // ── Basic arithmetic ────────────────────────────────────────────────

    #[test]
    fn test_add_tangent_propagation() {
        let x = ForwardGrad::new(3.0, 1.0);
        let y = ForwardGrad::constant(4.0);
        let result = x + y;
        assert!((result.value - 7.0).abs() < EPS);
        assert!((result.tangent - 1.0).abs() < EPS); // dx only
    }

    #[test]
    fn test_sub_tangent_propagation() {
        let x = ForwardGrad::new(5.0, 1.0);
        let y = ForwardGrad::new(2.0, 0.5);
        let result = x - y;
        assert!((result.value - 3.0).abs() < EPS);
        assert!((result.tangent - 0.5).abs() < EPS); // 1.0 - 0.5
    }

    #[test]
    fn test_mul_product_rule() {
        // f(x) = x * x = x², f'(x) = 2x, f'(3) = 6
        let x = ForwardGrad::variable(3.0);
        let result = x * x;
        assert!((result.value - 9.0).abs() < EPS);
        assert!((result.tangent - 6.0).abs() < EPS);
    }

    #[test]
    fn test_div_quotient_rule() {
        // f(x) = 1/x, f'(x) = -1/x², f'(2) = -0.25
        let x = ForwardGrad::variable(2.0);
        let one = ForwardGrad::constant(1.0);
        let result = one / x;
        assert!((result.value - 0.5).abs() < EPS);
        assert!((result.tangent - (-0.25)).abs() < EPS);
    }

    #[test]
    fn test_neg() {
        let x = ForwardGrad::new(2.0, 3.0);
        let result = -x;
        assert!((result.value - (-2.0)).abs() < EPS);
        assert!((result.tangent - (-3.0)).abs() < EPS);
    }

    // ── Transcendental functions ─────────────────────────────────────────

    #[test]
    fn test_sin_chain_rule() {
        // d/dx sin(x²) = 2x * cos(x²), at x = 1: 2 * cos(1)
        let x = ForwardGrad::variable(1.0_f32);
        let x_sq = x * x;
        let result = x_sq.sin();

        let expected_tangent = numerical_deriv(|v| (v * v).sin(), 1.0);
        assert!(
            (result.value - 1.0_f32.sin()).abs() < EPS,
            "primal: got {}, expected {}",
            result.value,
            1.0_f32.sin()
        );
        assert!(
            (result.tangent - expected_tangent).abs() < 1e-3,
            "tangent: got {}, expected {}",
            result.tangent,
            expected_tangent
        );
    }

    #[test]
    fn test_cos_chain_rule() {
        // d/dx cos(x) = -sin(x), at x = π/4
        let pi_over_4 = std::f32::consts::PI / 4.0;
        let x = ForwardGrad::variable(pi_over_4);
        let result = x.cos();

        let expected_value = pi_over_4.cos();
        let expected_tangent = numerical_deriv(|v| v.cos(), pi_over_4);

        assert!((result.value - expected_value).abs() < EPS);
        assert!((result.tangent - expected_tangent).abs() < 1e-3);
    }

    #[test]
    fn test_exp_chain_rule() {
        // d/dx exp(3x) = 3 * exp(3x), at x = 0: tangent = 3
        let x = ForwardGrad::variable(0.0_f32);
        let three_x = x * ForwardGrad::constant(3.0);
        let result = three_x.exp();

        assert!((result.value - 1.0).abs() < EPS); // exp(0) = 1
        assert!((result.tangent - 3.0).abs() < EPS); // 3 * exp(0) * 1 = 3
    }

    #[test]
    fn test_ln_chain_rule() {
        // d/dx ln(x^2) = 2/x, at x = 2: 1.0
        let x = ForwardGrad::variable(2.0_f32);
        let x_sq = x * x;
        let result = x_sq.ln();

        let expected_value = 4.0_f32.ln();
        let expected_tangent = numerical_deriv(|v| (v * v).ln(), 2.0);

        assert!((result.value - expected_value).abs() < EPS);
        assert!((result.tangent - expected_tangent).abs() < 1e-3);
    }

    #[test]
    fn test_sqrt_chain_rule() {
        // d/dx sqrt(x) = 1/(2*sqrt(x)), at x = 9: 1/6
        let x = ForwardGrad::variable(9.0_f32);
        let result = x.sqrt();

        assert!((result.value - 3.0).abs() < EPS);
        assert!((result.tangent - (1.0 / 6.0)).abs() < EPS);
    }

    #[test]
    fn test_powi_chain_rule() {
        // d/dx x^4 = 4x^3, at x = 2: 32
        let x = ForwardGrad::variable(2.0_f32);
        let result = x.powi(4);

        assert!((result.value - 16.0).abs() < EPS);
        assert!((result.tangent - 32.0).abs() < EPS);
    }

    // ── Compound expressions ─────────────────────────────────────────────

    #[test]
    fn test_compound_polynomial() {
        // f(x) = x^3 - 2x^2 + 3x - 1, f'(x) = 3x^2 - 4x + 3
        // At x = 2: f(2) = 8 - 8 + 6 - 1 = 5, f'(2) = 12 - 8 + 3 = 7
        let x = ForwardGrad::variable(2.0_f32);
        let one = ForwardGrad::constant(1.0);
        let two = ForwardGrad::constant(2.0);
        let three = ForwardGrad::constant(3.0);

        let x3 = x * x * x;
        let x2 = x * x;
        let f = x3 - two * x2 + three * x - one;

        assert!((f.value - 5.0).abs() < EPS, "f(2) = {}", f.value);
        assert!((f.tangent - 7.0).abs() < EPS, "f'(2) = {}", f.tangent);
    }

    #[test]
    fn test_chain_rule_nested() {
        // f(x) = exp(sin(x)), at x = 0
        // f(0) = exp(sin(0)) = exp(0) = 1
        // f'(x) = exp(sin(x)) * cos(x), f'(0) = exp(0) * cos(0) = 1
        let x = ForwardGrad::variable(0.0_f32);
        let result = x.sin().exp();

        assert!((result.value - 1.0).abs() < EPS);
        assert!((result.tangent - 1.0).abs() < EPS);
    }

    #[test]
    fn test_forward_matches_numerical_for_sigmoid() {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let x_val = 1.0_f32;
        let x = ForwardGrad::variable(x_val);
        let result = x.sigmoid();

        let expected_tangent = numerical_deriv(|v| 1.0 / (1.0 + (-v).exp()), x_val);
        assert!((result.tangent - expected_tangent).abs() < 1e-3);
    }

    #[test]
    fn test_relu_positive() {
        let x = ForwardGrad::variable(3.0_f32);
        let result = x.relu();
        assert!((result.value - 3.0).abs() < EPS);
        assert!((result.tangent - 1.0).abs() < EPS);
    }

    #[test]
    fn test_relu_negative() {
        let x = ForwardGrad::variable(-2.0_f32);
        let result = x.relu();
        assert!((result.value - 0.0).abs() < EPS);
        assert!((result.tangent - 0.0).abs() < EPS);
    }

    // ── Scalar operations ─────────────────────────────────────────────────

    #[test]
    fn test_scalar_mul() {
        // f(x) = 5 * x, f'(x) = 5
        let x = ForwardGrad::variable(3.0_f32);
        let result = x * 5.0;
        assert!((result.value - 15.0).abs() < EPS);
        assert!((result.tangent - 5.0).abs() < EPS);
    }

    #[test]
    fn test_scalar_div() {
        // f(x) = x / 4, f'(x) = 0.25
        let x = ForwardGrad::variable(8.0_f32);
        let result = x / 4.0;
        assert!((result.value - 2.0).abs() < EPS);
        assert!((result.tangent - 0.25).abs() < EPS);
    }

    // ── Multi-variable helpers ─────────────────────────────────────────────

    #[test]
    fn test_directional_derivative_function() {
        // f(x, y) = x^2 + y^2; direction = (1, 0) -> df = 2x
        let (val, dval) = directional_derivative(
            &[2.0, 3.0],
            &[1.0, 0.0],
            |inputs| inputs[0] * inputs[0] + inputs[1] * inputs[1],
        );
        assert!((val - 13.0).abs() < EPS);
        assert!((dval - 4.0).abs() < EPS); // 2 * 2 = 4
    }

    #[test]
    fn test_full_gradient_function() {
        // f(x, y) = x^2 + 3*y, gradient = (2x, 3), at (2, 1): (4, 3)
        let (val, grad) = full_gradient(&[2.0, 1.0], |inputs| {
            inputs[0] * inputs[0] + ForwardGrad::constant(3.0) * inputs[1]
        });
        assert!((val - 7.0).abs() < EPS);
        assert!((grad[0] - 4.0).abs() < EPS);
        assert!((grad[1] - 3.0).abs() < EPS);
    }

    #[test]
    fn test_tanh_derivative() {
        // d/dx tanh(x) = 1 - tanh²(x), at x = 0: 1.0
        let x = ForwardGrad::variable(0.0_f32);
        let result = x.tanh();
        assert!((result.value - 0.0).abs() < EPS);
        assert!((result.tangent - 1.0).abs() < EPS);
    }

    // ── Eq / Display ──────────────────────────────────────────────────────

    #[test]
    fn test_display() {
        let x = ForwardGrad::new(1.5, 2.5);
        let s = format!("{}", x);
        assert!(s.contains("1.5"));
        assert!(s.contains("2.5"));
    }
}
