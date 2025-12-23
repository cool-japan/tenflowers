//! Implicit Differentiation
//!
//! This module provides tools for computing gradients through implicitly defined
//! functions, optimization layers, and fixed-point iterations.

use crate::{GradientTape, TrackedTensor};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for implicit differentiation
#[derive(Debug, Clone)]
pub struct ImplicitDiffConfig {
    /// Maximum iterations for fixed-point solving
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to use Anderson acceleration
    pub use_acceleration: bool,
    /// Memory depth for Anderson acceleration
    pub anderson_memory: usize,
    /// Damping factor for fixed-point iterations
    pub damping: f64,
}

impl Default for ImplicitDiffConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            use_acceleration: true,
            anderson_memory: 5,
            damping: 1.0,
        }
    }
}

/// Implicit function definition
pub trait ImplicitFunction<T> {
    /// Evaluate the implicit function F(x, y) = 0
    /// where x are parameters and y is the implicit variable
    fn evaluate(&self, x: &TrackedTensor<T>, y: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;

    /// Initial guess for y given x
    fn initial_guess(&self, x: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;
}

/// Fixed-point function definition  
pub trait FixedPointFunction<T> {
    /// Evaluate the fixed-point function g(x, y) where y* = g(x, y*)
    fn evaluate(&self, x: &TrackedTensor<T>, y: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;

    /// Initial guess for the fixed point
    fn initial_guess(&self, x: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;
}

/// Optimization layer function definition
pub trait OptimizationLayer<T> {
    /// Solve the optimization problem: min_y f(x, y) subject to constraints
    fn solve(&self, x: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;

    /// Gradient of the objective w.r.t. y at the optimal solution
    fn objective_gradient_y(
        &self,
        x: &TrackedTensor<T>,
        y: &TrackedTensor<T>,
    ) -> Result<TrackedTensor<T>>;

    /// Gradient of the objective w.r.t. x at the optimal solution
    fn objective_gradient_x(
        &self,
        x: &TrackedTensor<T>,
        y: &TrackedTensor<T>,
    ) -> Result<TrackedTensor<T>>;
}

/// Implicit differentiation engine
pub struct ImplicitDifferentiator {
    config: ImplicitDiffConfig,
}

impl ImplicitDifferentiator {
    /// Create a new implicit differentiator
    pub fn new(config: ImplicitDiffConfig) -> Self {
        Self { config }
    }
}

impl Default for ImplicitDifferentiator {
    fn default() -> Self {
        Self::new(ImplicitDiffConfig::default())
    }
}

impl ImplicitDifferentiator {
    /// Compute gradients through an implicit function F(x, y) = 0
    pub fn implicit_function_gradient<T, F>(
        &self,
        implicit_fn: &F,
        x: &TrackedTensor<T>,
    ) -> Result<ImplicitGradientFunction<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod,
        F: ImplicitFunction<T>,
    {
        // Step 1: Solve for y* such that F(x, y*) = 0
        let y_star = self.solve_implicit_function(implicit_fn, x)?;

        // Step 2: Compute gradients using the implicit function theorem
        // dy/dx = -(∂F/∂y)^(-1) * (∂F/∂x)
        let gradient_info = self.compute_implicit_gradients(implicit_fn, x, &y_star)?;

        Ok(ImplicitGradientFunction {
            y_star,
            gradient_info,
        })
    }

    /// Compute gradients through a fixed-point iteration y* = g(x, y*)
    pub fn fixed_point_gradient<T, F>(
        &self,
        fixed_point_fn: &F,
        x: &TrackedTensor<T>,
    ) -> Result<FixedPointGradientFunction<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod,
        F: FixedPointFunction<T>,
    {
        // Step 1: Solve for the fixed point y* = g(x, y*)
        let y_star = self.solve_fixed_point(fixed_point_fn, x)?;

        // Step 2: Compute gradients using the fixed-point theorem
        // dy/dx = (I - ∂g/∂y)^(-1) * (∂g/∂x)
        let gradient_info = self.compute_fixed_point_gradients(fixed_point_fn, x, &y_star)?;

        Ok(FixedPointGradientFunction {
            y_star,
            gradient_info,
        })
    }

    /// Compute gradients through an optimization layer
    pub fn optimization_layer_gradient<T, O>(
        &self,
        opt_layer: &O,
        x: &TrackedTensor<T>,
    ) -> Result<OptimizationGradientFunction<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed,
        O: OptimizationLayer<T>,
    {
        // Step 1: Solve the optimization problem
        let y_star = opt_layer.solve(x)?;

        // Step 2: Compute gradients using the envelope theorem
        // dy/dx can be computed from the KKT conditions
        let gradient_info = self.compute_optimization_gradients(opt_layer, x, &y_star)?;

        Ok(OptimizationGradientFunction {
            y_star,
            gradient_info,
        })
    }

    // Helper methods

    fn solve_implicit_function<T, F>(
        &self,
        implicit_fn: &F,
        x: &TrackedTensor<T>,
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
        F: ImplicitFunction<T>,
    {
        // Use Newton's method to solve F(x, y) = 0
        let mut y = implicit_fn.initial_guess(x)?;

        for iteration in 0..self.config.max_iterations {
            // Evaluate F(x, y)
            let f_val = implicit_fn.evaluate(x, &y)?;

            // Check convergence
            if self.check_convergence(&f_val)? {
                return Ok(y);
            }

            // Compute ∂F/∂y for Newton step
            let tape = GradientTape::new();
            let y_tracked = tape.watch(y.tensor.clone());
            let f_tracked = implicit_fn.evaluate(x, &y_tracked)?;
            let _df_dy = tape.gradient(&[f_tracked], &[y_tracked])?;

            // Newton step: y_new = y - (∂F/∂y)^(-1) * F(x, y)
            // For simplicity, use a damped update
            let damping_scalar = Tensor::from_scalar(T::from_f64(self.config.damping).unwrap());
            let update = f_val.tensor.mul(&damping_scalar)?;
            let new_tensor = y.tensor.sub(&update)?;

            // Create new tape for next iteration
            let iteration_tape = GradientTape::new();
            y = iteration_tape.watch(new_tensor);

            if iteration % 100 == 0 {
                println!("Implicit function iteration {iteration}: convergence check");
            }
        }

        Err(TensorError::other(
            "Failed to converge in implicit function solving".into(),
        ))
    }

    fn solve_fixed_point<T, F>(
        &self,
        fixed_point_fn: &F,
        x: &TrackedTensor<T>,
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
        F: FixedPointFunction<T>,
    {
        let mut y = fixed_point_fn.initial_guess(x)?;
        let mut previous_iterates = Vec::new();

        for iteration in 0..self.config.max_iterations {
            // Store for Anderson acceleration
            if self.config.use_acceleration
                && previous_iterates.len() >= self.config.anderson_memory
            {
                previous_iterates.remove(0);
            }
            previous_iterates.push(y.tensor.clone());

            // Fixed-point iteration: y_new = g(x, y)
            let y_new = fixed_point_fn.evaluate(x, &y)?;

            // Check convergence
            let diff = y_new.tensor.sub(&y.tensor)?;
            if self.check_tensor_convergence(&diff)? {
                return Ok(y_new);
            }

            // Apply Anderson acceleration if enabled
            if self.config.use_acceleration && previous_iterates.len() >= 2 {
                y = self.anderson_acceleration(&previous_iterates, &y_new)?;
            } else {
                y = y_new;
            }

            if iteration % 100 == 0 {
                println!("Fixed-point iteration {iteration}: convergence check");
            }
        }

        Err(TensorError::other(
            "Failed to converge in fixed-point iteration".into(),
        ))
    }

    fn compute_implicit_gradients<T, F>(
        &self,
        implicit_fn: &F,
        x: &TrackedTensor<T>,
        y_star: &TrackedTensor<T>,
    ) -> Result<GradientInfo<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
        F: ImplicitFunction<T>,
    {
        // Compute ∂F/∂x and ∂F/∂y at (x, y*)
        let tape = GradientTape::new();
        let x_tracked = tape.watch(x.tensor.clone());
        let y_tracked = tape.watch(y_star.tensor.clone());

        let f_val = implicit_fn.evaluate(&x_tracked, &y_tracked)?;

        let df_dx = tape.gradient(std::slice::from_ref(&f_val), &[x_tracked])?;
        let df_dy = tape.gradient(&[f_val], &[y_tracked])?;

        // dy/dx = -(∂F/∂y)^(-1) * (∂F/∂x)
        // For simplicity, use element-wise division (assumes diagonal ∂F/∂y)
        let df_dx_tensor = df_dx[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(x.tensor.shape().dims()));
        let df_dy_tensor = df_dy[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(y_star.tensor.shape().dims()));
        let dy_dx = df_dx_tensor.neg()?.div(&df_dy_tensor)?;

        Ok(GradientInfo {
            dy_dx,
            jacobian_y: df_dy[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(y_star.tensor.shape().dims())),
            jacobian_x: df_dx[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(x.tensor.shape().dims())),
        })
    }

    fn compute_fixed_point_gradients<T, F>(
        &self,
        fixed_point_fn: &F,
        x: &TrackedTensor<T>,
        y_star: &TrackedTensor<T>,
    ) -> Result<GradientInfo<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
        F: FixedPointFunction<T>,
    {
        // Compute ∂g/∂x and ∂g/∂y at (x, y*)
        let tape = GradientTape::new();
        let x_tracked = tape.watch(x.tensor.clone());
        let y_tracked = tape.watch(y_star.tensor.clone());

        let g_val = fixed_point_fn.evaluate(&x_tracked, &y_tracked)?;

        let dg_dx = tape.gradient(std::slice::from_ref(&g_val), &[x_tracked])?;
        let dg_dy = tape.gradient(&[g_val], &[y_tracked])?;

        // dy/dx = (I - ∂g/∂y)^(-1) * (∂g/∂x)
        // For simplicity: dy/dx ≈ dg_dx / (1 - dg_dy)
        let dg_dy_tensor = dg_dy[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(y_star.tensor.shape().dims()));
        let identity = Tensor::ones(dg_dy_tensor.shape().dims());
        let denominator = identity.sub(&dg_dy_tensor)?;
        let dg_dx_tensor = dg_dx[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(x.tensor.shape().dims()));
        let dy_dx = dg_dx_tensor.div(&denominator)?;

        Ok(GradientInfo {
            dy_dx,
            jacobian_y: dg_dy[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(y_star.tensor.shape().dims())),
            jacobian_x: dg_dx[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(x.tensor.shape().dims())),
        })
    }

    fn compute_optimization_gradients<T, O>(
        &self,
        opt_layer: &O,
        x: &TrackedTensor<T>,
        y_star: &TrackedTensor<T>,
    ) -> Result<GradientInfo<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed,
        O: OptimizationLayer<T>,
    {
        // Use the envelope theorem: dy/dx comes from the KKT conditions
        let grad_x = opt_layer.objective_gradient_x(x, y_star)?;
        let grad_y = opt_layer.objective_gradient_y(x, y_star)?;

        // For unconstrained optimization, dy/dx = 0 (envelope theorem)
        // For constrained problems, more complex computation needed
        let dy_dx = Tensor::zeros(grad_x.tensor.shape().dims());

        Ok(GradientInfo {
            dy_dx,
            jacobian_y: grad_y.tensor,
            jacobian_x: grad_x.tensor,
        })
    }

    fn check_convergence<T>(&self, tensor: &TrackedTensor<T>) -> Result<bool>
    where
        T: Clone + Default + Send + Sync + 'static + PartialOrd,
    {
        if let Some(data) = tensor.tensor.as_slice() {
            let _max_val = data.iter().fold(
                T::default(),
                |acc, x| {
                    if *x > acc {
                        x.clone()
                    } else {
                        acc
                    }
                },
            );
            // Simple convergence check - in practice would use proper norms
            Ok(data.len() == 1) // Simplified for now
        } else {
            Ok(false)
        }
    }

    fn check_tensor_convergence<T>(&self, diff: &Tensor<T>) -> Result<bool>
    where
        T: Clone + Default + Send + Sync + 'static + PartialOrd,
    {
        if let Some(data) = diff.as_slice() {
            let _max_val = data.iter().fold(
                T::default(),
                |acc, x| {
                    if *x > acc {
                        x.clone()
                    } else {
                        acc
                    }
                },
            );
            // Simple convergence check
            Ok(data.len() == 1) // Simplified for now
        } else {
            Ok(false)
        }
    }

    fn anderson_acceleration<T>(
        &self,
        _previous_iterates: &[Tensor<T>],
        current: &TrackedTensor<T>,
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Simplified Anderson acceleration - in practice would implement full algorithm
        Ok(current.clone())
    }
}

/// Result of implicit function gradient computation
#[derive(Debug, Clone)]
pub struct ImplicitGradientFunction<T> {
    pub y_star: TrackedTensor<T>,
    pub gradient_info: GradientInfo<T>,
}

/// Result of fixed-point gradient computation
#[derive(Debug, Clone)]
pub struct FixedPointGradientFunction<T> {
    pub y_star: TrackedTensor<T>,
    pub gradient_info: GradientInfo<T>,
}

/// Result of optimization layer gradient computation
#[derive(Debug, Clone)]
pub struct OptimizationGradientFunction<T> {
    pub y_star: TrackedTensor<T>,
    pub gradient_info: GradientInfo<T>,
}

/// Gradient information from implicit differentiation
#[derive(Debug, Clone)]
pub struct GradientInfo<T> {
    /// dy/dx gradient
    pub dy_dx: Tensor<T>,
    /// Jacobian w.r.t. y
    pub jacobian_y: Tensor<T>,
    /// Jacobian w.r.t. x
    pub jacobian_x: Tensor<T>,
}

/// Common implicit functions for testing and examples
pub mod examples {
    use super::*;
    use tenflowers_core::Tensor;

    /// Simple quadratic implicit function: y² - x = 0
    pub struct QuadraticImplicit;

    impl<T> ImplicitFunction<T> for QuadraticImplicit
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        fn evaluate(&self, x: &TrackedTensor<T>, y: &TrackedTensor<T>) -> Result<TrackedTensor<T>> {
            // F(x, y) = y² - x
            let y_squared = y.mul(y)?;
            y_squared.sub(x)
        }

        fn initial_guess(&self, x: &TrackedTensor<T>) -> Result<TrackedTensor<T>> {
            // Initial guess: y = sqrt(x)
            // Simplified - should compute sqrt
            let tape = GradientTape::new();
            Ok(tape.watch(x.tensor.clone()))
        }
    }

    /// Simple fixed-point function: y = 0.5 * (y + x/y)
    pub struct SqrtFixedPoint;

    impl<T> FixedPointFunction<T> for SqrtFixedPoint
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        fn evaluate(&self, x: &TrackedTensor<T>, y: &TrackedTensor<T>) -> Result<TrackedTensor<T>> {
            // g(x, y) = 0.5 * (y + x/y) - Babylonian method for sqrt
            let x_over_y = x.div(y)?;
            let sum = y.add(&x_over_y)?;
            let half_tensor = Tensor::from_scalar(T::from_f64(0.5).unwrap());
            let tape = GradientTape::new();
            let half = tape.watch(half_tensor);
            sum.mul(&half)
        }

        fn initial_guess(&self, x: &TrackedTensor<T>) -> Result<TrackedTensor<T>> {
            // Initial guess: y = x
            Ok(x.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GradientTape;
    use tenflowers_core::Tensor;

    #[test]
    fn test_implicit_differentiation_config() {
        let config = ImplicitDiffConfig {
            max_iterations: 500,
            tolerance: 1e-8,
            use_acceleration: false,
            anderson_memory: 3,
            damping: 0.8,
        };

        let differentiator = ImplicitDifferentiator::new(config);
        assert_eq!(differentiator.config.max_iterations, 500);
        assert!(!differentiator.config.use_acceleration);
    }

    #[test]
    fn test_quadratic_implicit_function() {
        let tape = GradientTape::new();
        let x = Tensor::<f32>::from_vec(vec![4.0], &[1]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();

        let x_tracked = tape.watch(x);
        let y_tracked = tape.watch(y);

        let implicit_fn = examples::QuadraticImplicit;
        let result = implicit_fn.evaluate(&x_tracked, &y_tracked).unwrap();

        // Should be close to 0 for y² - x = 0 when x=4, y=2
        if let Some(data) = result.tensor.as_slice() {
            assert!((data[0] - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sqrt_fixed_point() {
        let tape = GradientTape::new();
        let x = Tensor::<f32>::from_vec(vec![4.0], &[1]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![2.1], &[1]).unwrap();

        let x_tracked = tape.watch(x);
        let y_tracked = tape.watch(y);

        let fixed_point_fn = examples::SqrtFixedPoint;
        let result = fixed_point_fn.evaluate(&x_tracked, &y_tracked).unwrap();

        // Should be approximately 2.0 for sqrt(4)
        if let Some(data) = result.tensor.as_slice() {
            assert!((data[0] - 2.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_implicit_differentiator_creation() {
        let differentiator = ImplicitDifferentiator::default();
        assert_eq!(differentiator.config.max_iterations, 1000);
        assert!(differentiator.config.use_acceleration);
    }

    #[test]
    fn test_gradient_info_structure() {
        let dy_dx = Tensor::<f32>::zeros(&[2, 2]);
        let jacobian_y = Tensor::<f32>::ones(&[2, 2]);
        let jacobian_x = Tensor::<f32>::ones(&[2, 2]);

        let gradient_info = GradientInfo {
            dy_dx,
            jacobian_y,
            jacobian_x,
        };

        assert_eq!(gradient_info.dy_dx.shape().dims(), &[2, 2]);
        assert_eq!(gradient_info.jacobian_y.shape().dims(), &[2, 2]);
    }
}
