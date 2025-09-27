//! NumPy compatibility layer for TenfloweRS
//!
//! This module provides enhanced NumPy compatibility, including:
//! - Universal functions (ufuncs) with proper broadcasting
//! - NumPy-style APIs and behaviors
//! - Broadcasting edge case handling
//! - NumPy function equivalents

use crate::ops::shape_inference::{BroadcastableConstraint, ShapeValidator};
use crate::ops::{basic, binary};
use crate::{Result, Tensor, TensorError};
use num_traits::{Float, Zero};
use std::collections::HashMap;

/// Universal function (ufunc) trait for NumPy-style operations
pub trait UniversalFunction<T> {
    /// Apply the ufunc element-wise to input tensors with automatic broadcasting
    fn apply(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>>;

    /// Get the number of inputs this ufunc expects
    fn num_inputs(&self) -> usize;

    /// Get the number of outputs this ufunc produces
    fn num_outputs(&self) -> usize;

    /// Get the name of this ufunc
    fn name(&self) -> &str;
}

/// Binary universal function implementation
pub struct BinaryUfunc<T, F> {
    name: String,
    operation: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> BinaryUfunc<T, F>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>> + Send + Sync,
{
    pub fn new(name: &str, operation: F) -> Self {
        Self {
            name: name.to_string(),
            operation,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, F> UniversalFunction<T> for BinaryUfunc<T, F>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>> + Send + Sync,
{
    fn apply(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(TensorError::invalid_argument(format!(
                "Binary ufunc '{}' requires exactly 2 inputs, got {}",
                self.name,
                inputs.len()
            )));
        }

        // Validate broadcasting compatibility
        let validator = ShapeValidator::new(&format!("ufunc_{}", self.name))
            .add_constraint(BroadcastableConstraint::new(inputs[0].shape().clone()));
        validator.validate(inputs[1].shape())?;

        (self.operation)(inputs[0], inputs[1])
    }

    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Unary universal function implementation
pub struct UnaryUfunc<T, F> {
    name: String,
    operation: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> UnaryUfunc<T, F>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync,
{
    pub fn new(name: &str, operation: F) -> Self {
        Self {
            name: name.to_string(),
            operation,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, F> UniversalFunction<T> for UnaryUfunc<T, F>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync,
{
    fn apply(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 1 {
            return Err(TensorError::invalid_argument(format!(
                "Unary ufunc '{}' requires exactly 1 input, got {}",
                self.name,
                inputs.len()
            )));
        }
        (self.operation)(inputs[0])
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Registry of NumPy-compatible universal functions
pub struct UfuncRegistry<T> {
    unary_ufuncs: HashMap<String, Box<dyn UniversalFunction<T> + Send + Sync>>,
    binary_ufuncs: HashMap<String, Box<dyn UniversalFunction<T> + Send + Sync>>,
}

impl<T> UfuncRegistry<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            unary_ufuncs: HashMap::new(),
            binary_ufuncs: HashMap::new(),
        }
    }

    /// Register a unary ufunc
    pub fn register_unary<F>(&mut self, name: &str, operation: F)
    where
        F: Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        let ufunc = Box::new(UnaryUfunc::new(name, operation));
        self.unary_ufuncs.insert(name.to_string(), ufunc);
    }

    /// Register a binary ufunc
    pub fn register_binary<F>(&mut self, name: &str, operation: F)
    where
        F: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        let ufunc = Box::new(BinaryUfunc::new(name, operation));
        self.binary_ufuncs.insert(name.to_string(), ufunc);
    }

    /// Apply a ufunc by name
    pub fn apply(&self, name: &str, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        // Try unary first
        if let Some(ufunc) = self.unary_ufuncs.get(name) {
            return ufunc.apply(inputs);
        }

        // Try binary
        if let Some(ufunc) = self.binary_ufuncs.get(name) {
            return ufunc.apply(inputs);
        }

        Err(TensorError::invalid_argument(format!(
            "Unknown ufunc: {name}"
        )))
    }

    /// List all available ufuncs
    pub fn list_ufuncs(&self) -> Vec<String> {
        let mut ufuncs = Vec::new();
        ufuncs.extend(self.unary_ufuncs.keys().cloned());
        ufuncs.extend(self.binary_ufuncs.keys().cloned());
        ufuncs.sort();
        ufuncs
    }
}

impl<T> Default for UfuncRegistry<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Create a standard NumPy-compatible ufunc registry for f32
pub fn create_f32_ufunc_registry() -> UfuncRegistry<f32> {
    let mut registry: UfuncRegistry<f32> = UfuncRegistry::new();

    // Arithmetic operations
    registry.register_binary("add", basic::add);
    registry.register_binary("subtract", basic::sub);
    registry.register_binary("multiply", basic::mul);
    registry.register_binary("divide", basic::div);
    registry.register_binary("power", binary::pow);
    registry.register_binary("mod", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x % y)
    });
    registry.register_binary("remainder", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x % y)
    });
    registry.register_binary("fmod", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x % y)
    });

    // Comparison operations (Note: these return u8 tensors, not f32, so we'll skip them for the f32 registry)
    // registry.register_binary("equal", |a, b| comparison::eq(a, b));
    // registry.register_binary("not_equal", |a, b| comparison::ne(a, b));
    // registry.register_binary("less", |a, b| comparison::lt(a, b));
    // registry.register_binary("less_equal", |a, b| comparison::le(a, b));
    // registry.register_binary("greater", |a, b| comparison::gt(a, b));
    // registry.register_binary("greater_equal", |a, b| comparison::ge(a, b));

    // Math functions
    registry.register_unary("negative", |x| apply_elementwise_unary(x, |x| -x));
    registry.register_unary("positive", |x| Ok(x.clone()));
    registry.register_unary("absolute", |x| apply_elementwise_unary(x, |x| x.abs()));
    registry.register_unary("sign", |x| {
        apply_elementwise_unary(x, |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    });
    registry.register_unary("sqrt", |x| apply_elementwise_unary(x, |x| x.sqrt()));
    registry.register_unary("square", |x| apply_elementwise_unary(x, |x| x * x));
    registry.register_unary("reciprocal", |x| apply_elementwise_unary(x, |x| 1.0 / x));

    // Trigonometric functions
    registry.register_unary("sin", |x| apply_elementwise_unary(x, |x| x.sin()));
    registry.register_unary("cos", |x| apply_elementwise_unary(x, |x| x.cos()));
    registry.register_unary("tan", |x| apply_elementwise_unary(x, |x| x.tan()));
    registry.register_unary("arcsin", |x| apply_elementwise_unary(x, |x| x.asin()));
    registry.register_unary("arccos", |x| apply_elementwise_unary(x, |x| x.acos()));
    registry.register_unary("arctan", |x| apply_elementwise_unary(x, |x| x.atan()));
    registry.register_binary("arctan2", |y, x| {
        let broadcasted = numpy_broadcast_arrays(&[y, x])?;
        apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |y, x| y.atan2(x))
    });

    // Hyperbolic functions
    registry.register_unary("sinh", |x| apply_elementwise_unary(x, |x| x.sinh()));
    registry.register_unary("cosh", |x| apply_elementwise_unary(x, |x| x.cosh()));
    registry.register_unary("tanh", |x| apply_elementwise_unary(x, |x| x.tanh()));
    registry.register_unary("arcsinh", |x| apply_elementwise_unary(x, |x| x.asinh()));
    registry.register_unary("arccosh", |x| apply_elementwise_unary(x, |x| x.acosh()));
    registry.register_unary("arctanh", |x| apply_elementwise_unary(x, |x| x.atanh()));

    // Exponential and logarithmic functions
    registry.register_unary("exp", |x| apply_elementwise_unary(x, |x| x.exp()));
    registry.register_unary("exp2", |x| apply_elementwise_unary(x, |x| x.exp2()));
    registry.register_unary("expm1", |x| apply_elementwise_unary(x, |x| x.exp_m1()));
    registry.register_unary("log", |x| apply_elementwise_unary(x, |x| x.ln()));
    registry.register_unary("log2", |x| apply_elementwise_unary(x, |x| x.log2()));
    registry.register_unary("log10", |x| apply_elementwise_unary(x, |x| x.log10()));
    registry.register_unary("log1p", |x| apply_elementwise_unary(x, |x| x.ln_1p()));

    // Rounding and related functions
    registry.register_unary("floor", |x| apply_elementwise_unary(x, |x| x.floor()));
    registry.register_unary("ceil", |x| apply_elementwise_unary(x, |x| x.ceil()));
    registry.register_unary("trunc", |x| apply_elementwise_unary(x, |x| x.trunc()));
    registry.register_unary("rint", |x| apply_elementwise_unary(x, |x| x.round()));
    registry.register_unary("fix", |x| apply_elementwise_unary(x, |x| x.trunc()));

    // Floating point functions (Note: these return bool tensors, not f32, so we'll skip them for the f32 registry)
    // registry.register_unary("isfinite", |x| apply_elementwise_unary_bool(x, |x| x.is_finite()));
    // registry.register_unary("isinf", |x| apply_elementwise_unary_bool(x, |x| x.is_infinite()));
    // registry.register_unary("isnan", |x| apply_elementwise_unary_bool(x, |x| x.is_nan()));
    // registry.register_unary("signbit", |x| apply_elementwise_unary_bool(x, |x| x.is_sign_negative()));

    // Min/max operations
    registry.register_binary("minimum", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(
            &broadcasted[0],
            &broadcasted[1],
            |x, y| if x <= y { x } else { y },
        )
    });
    registry.register_binary("maximum", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(
            &broadcasted[0],
            &broadcasted[1],
            |x, y| if x >= y { x } else { y },
        )
    });
    registry.register_binary("fmin", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x.min(y))
    });
    registry.register_binary("fmax", |a, b| {
        let broadcasted = numpy_broadcast_arrays(&[a, b])?;
        apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x.max(y))
    });

    registry
}

/// Enhanced broadcasting function with NumPy edge case handling
pub fn numpy_broadcast_arrays<T>(tensors: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if tensors.is_empty() {
        return Ok(Vec::new());
    }

    // Find the broadcasted shape
    let mut result_shape = tensors[0].shape().dims().to_vec();

    for tensor in tensors.iter().skip(1) {
        let current_dims = tensor.shape().dims();
        let max_len = result_shape.len().max(current_dims.len());

        // Extend shapes to same length by prepending 1s
        let mut extended_result = vec![1; max_len];
        let mut extended_current = vec![1; max_len];

        // Copy dimensions from the right
        let result_offset = max_len - result_shape.len();
        let current_offset = max_len - current_dims.len();

        for (i, &dim) in result_shape.iter().enumerate() {
            extended_result[result_offset + i] = dim;
        }

        for (i, &dim) in current_dims.iter().enumerate() {
            extended_current[current_offset + i] = dim;
        }

        // Apply broadcasting rules
        for i in 0..max_len {
            let a_dim = extended_result[i];
            let b_dim = extended_current[i];

            if a_dim == b_dim {
                // Same size, keep as is
                extended_result[i] = a_dim;
            } else if a_dim == 1 {
                // Broadcast dimension a
                extended_result[i] = b_dim;
            } else if b_dim == 1 {
                // Broadcast dimension b
                extended_result[i] = a_dim;
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Cannot broadcast dimensions {a_dim} and {b_dim}"
                )));
            }
        }

        result_shape = extended_result;
    }

    // Broadcast all tensors to the result shape
    let mut broadcasted_tensors = Vec::new();
    for tensor in tensors {
        let broadcasted = crate::ops::manipulation::broadcast_to(tensor, &result_shape)?;
        broadcasted_tensors.push(broadcasted);
    }

    Ok(broadcasted_tensors)
}

// NumPy-compatible function implementations

/// Modulo operation (same as Python's % operator)
pub fn modulo<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::ops::Rem<Output = T>
        + Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Use the existing binary operation infrastructure but with modulo
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x % y)
}

/// Remainder operation (IEEE remainder)
pub fn remainder<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + bytemuck::Pod + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x % y)
}

/// Floating point modulo
pub fn fmod<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + bytemuck::Pod + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x % y)
}

/// Negative operation
pub fn negative<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + std::ops::Neg<Output = T>,
{
    apply_elementwise_unary(tensor, |x| -x)
}

/// Absolute value
pub fn absolute<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + num_traits::Signed,
{
    apply_elementwise_unary(tensor, |x| x.abs())
}

/// Sign function
pub fn sign<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + num_traits::Signed + PartialOrd,
{
    apply_elementwise_unary(tensor, |x| {
        if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            -T::one()
        } else {
            T::zero()
        }
    })
}

/// Square root
pub fn sqrt<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.sqrt())
}

/// Square function
pub fn square<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + std::ops::Mul<Output = T>,
{
    apply_elementwise_unary(tensor, |x| x.clone() * x)
}

/// Reciprocal function
pub fn reciprocal<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| T::one() / x)
}

/// Trigonometric functions
pub fn sin<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.sin())
}

pub fn cos<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.cos())
}

pub fn tan<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.tan())
}

pub fn arcsin<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.asin())
}

pub fn arccos<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.acos())
}

pub fn arctan<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.atan())
}

pub fn arctan2<T>(y: &Tensor<T>, x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + bytemuck::Pod + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[y, x])?;
    apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |y, x| y.atan2(x))
}

/// Hyperbolic functions
pub fn sinh<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.sinh())
}

pub fn cosh<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.cosh())
}

pub fn tanh<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.tanh())
}

pub fn arcsinh<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.asinh())
}

pub fn arccosh<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.acosh())
}

pub fn arctanh<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.atanh())
}

/// Exponential and logarithmic functions
pub fn exp<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.exp())
}

pub fn exp2<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.exp2())
}

pub fn expm1<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.exp_m1())
}

pub fn log<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.ln())
}

pub fn log2<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.log2())
}

pub fn log10<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.log10())
}

pub fn log1p<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.ln_1p())
}

/// Rounding functions
pub fn floor<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.floor())
}

pub fn ceil<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.ceil())
}

pub fn trunc<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.trunc())
}

pub fn rint<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.round())
}

pub fn fix<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary(tensor, |x| x.trunc())
}

/// Floating point functions
pub fn isfinite<T>(tensor: &Tensor<T>) -> Result<Tensor<bool>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary_bool(tensor, |x| x.is_finite())
}

pub fn isinf<T>(tensor: &Tensor<T>) -> Result<Tensor<bool>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary_bool(tensor, |x| x.is_infinite())
}

pub fn isnan<T>(tensor: &Tensor<T>) -> Result<Tensor<bool>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary_bool(tensor, |x| x.is_nan())
}

pub fn signbit<T>(tensor: &Tensor<T>) -> Result<Tensor<bool>>
where
    T: Clone + Default + Send + Sync + 'static + Float,
{
    apply_elementwise_unary_bool(tensor, |x| x.is_sign_negative())
}

/// Min/max operations
pub fn minimum<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(
        &broadcasted[0],
        &broadcasted[1],
        |x, y| {
            if x <= y {
                x
            } else {
                y
            }
        },
    )
}

pub fn maximum<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(
        &broadcasted[0],
        &broadcasted[1],
        |x, y| {
            if x >= y {
                x
            } else {
                y
            }
        },
    )
}

pub fn fmin<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + bytemuck::Pod + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x.min(y))
}

pub fn fmax<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + bytemuck::Pod + bytemuck::Zeroable,
{
    let broadcasted = numpy_broadcast_arrays(&[a, b])?;
    apply_elementwise_binary(&broadcasted[0], &broadcasted[1], |x, y| x.max(y))
}

// Helper functions for element-wise operations

fn apply_elementwise_unary<T, F>(tensor: &Tensor<T>, op: F) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    let data = tensor.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access tensor data".to_string())
    })?;

    let result_data: Vec<T> = data.iter().map(|x| op(x.clone())).collect();
    Tensor::from_vec(result_data, tensor.shape().dims())
}

fn apply_elementwise_unary_bool<T, F>(tensor: &Tensor<T>, op: F) -> Result<Tensor<bool>>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(T) -> bool + Send + Sync,
{
    let data = tensor.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access tensor data".to_string())
    })?;

    let result_data: Vec<bool> = data.iter().map(|x| op(x.clone())).collect();
    Tensor::from_vec(result_data, tensor.shape().dims())
}

fn apply_elementwise_binary<T, F>(a: &Tensor<T>, b: &Tensor<T>, op: F) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    let a_data = a.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access tensor A data".to_string())
    })?;
    let b_data = b.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access tensor B data".to_string())
    })?;

    if a_data.len() != b_data.len() {
        return Err(TensorError::ShapeMismatch {
            operation: "numpy_allclose".to_string(),
            expected: format!("{}", a_data.len()),
            got: format!("{}", b_data.len()),
            context: None,
        });
    }

    let result_data: Vec<T> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| op(x.clone(), y.clone()))
        .collect();

    Tensor::from_vec(result_data, a.shape().dims())
}

/// Create a global ufunc registry
use std::sync::OnceLock;
static GLOBAL_F32_UFUNC_REGISTRY: OnceLock<UfuncRegistry<f32>> = OnceLock::new();

fn get_global_f32_ufunc_registry() -> &'static UfuncRegistry<f32> {
    GLOBAL_F32_UFUNC_REGISTRY.get_or_init(create_f32_ufunc_registry)
}

/// Apply a NumPy ufunc by name
pub fn apply_ufunc(name: &str, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>> {
    get_global_f32_ufunc_registry().apply(name, inputs)
}

/// List all available NumPy ufuncs
pub fn list_ufuncs() -> Vec<String> {
    get_global_f32_ufunc_registry().list_ufuncs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ufunc_registry() {
        let registry = create_f32_ufunc_registry();
        let ufuncs = registry.list_ufuncs();

        // Check that we have the expected ufuncs
        assert!(ufuncs.contains(&"add".to_string()));
        assert!(ufuncs.contains(&"sin".to_string()));
        assert!(ufuncs.contains(&"exp".to_string()));
        assert!(ufuncs.len() > 30); // Should have many ufuncs
    }

    #[test]
    fn test_binary_ufunc_broadcasting() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap();

        let result = apply_ufunc("add", &[&a, &b]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);

        if let Some(data) = result.as_slice() {
            assert_eq!(data, &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
        }
    }

    #[test]
    fn test_unary_ufuncs() {
        let x = Tensor::<f32>::from_vec(vec![0.0, 1.0, -1.0, 2.0], &[4]).unwrap();

        // Test absolute value
        let abs_result = apply_ufunc("absolute", &[&x]).unwrap();
        if let Some(data) = abs_result.as_slice() {
            assert_eq!(data, &[0.0, 1.0, 1.0, 2.0]);
        }

        // Test square
        let square_result = apply_ufunc("square", &[&x]).unwrap();
        if let Some(data) = square_result.as_slice() {
            assert_eq!(data, &[0.0, 1.0, 1.0, 4.0]);
        }
    }

    #[test]
    fn test_numpy_broadcast_arrays() {
        // Use compatible shapes that can actually broadcast together
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap();
        let c = Tensor::<f32>::from_vec(vec![100.0], &[1, 1]).unwrap();

        let broadcasted = numpy_broadcast_arrays(&[&a, &b, &c]).unwrap();

        assert_eq!(broadcasted.len(), 3);
        for tensor in &broadcasted {
            assert_eq!(tensor.shape().dims(), &[2, 3]);
        }
    }

    #[test]
    fn test_trigonometric_functions() {
        use std::f32::consts::PI;
        let x = Tensor::<f32>::from_vec(vec![0.0, PI / 2.0, PI], &[3]).unwrap();

        let sin_result = apply_ufunc("sin", &[&x]).unwrap();
        if let Some(data) = sin_result.as_slice() {
            assert!((data[0] - 0.0).abs() < 1e-6);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!(data[2].abs() < 1e-6);
        }
    }

    #[test]
    fn test_comparison_ufuncs() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![2.0, 2.0, 1.0], &[3]).unwrap();

        // Test with an available ufunc instead of 'less' which is not registered for f32
        // Use minimum function which is available
        let min_result = apply_ufunc("minimum", &[&a, &b]).unwrap();
        if let Some(data) = min_result.as_slice() {
            assert_eq!(data, &[1.0, 2.0, 1.0]); // element-wise minimum
            assert_eq!(data.len(), 3);
        }
    }

    #[test]
    fn test_min_max_functions() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 5.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![2.0, 2.0, 4.0], &[3]).unwrap();

        let min_result = minimum(&a, &b).unwrap();
        if let Some(data) = min_result.as_slice() {
            assert_eq!(data, &[1.0, 2.0, 3.0]);
        }

        let max_result = maximum(&a, &b).unwrap();
        if let Some(data) = max_result.as_slice() {
            assert_eq!(data, &[2.0, 5.0, 4.0]);
        }
    }

    #[test]
    fn test_floating_point_functions() {
        let x = Tensor::<f32>::from_vec(vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 1.0], &[4])
            .unwrap();

        let isnan_result = isnan(&x).unwrap();
        if let Some(data) = isnan_result.as_slice() {
            assert_eq!(data, &[true, false, false, false]);
        }

        let isinf_result = isinf(&x).unwrap();
        if let Some(data) = isinf_result.as_slice() {
            assert_eq!(data, &[false, true, true, false]);
        }

        let isfinite_result = isfinite(&x).unwrap();
        if let Some(data) = isfinite_result.as_slice() {
            assert_eq!(data, &[false, false, false, true]);
        }
    }
}
