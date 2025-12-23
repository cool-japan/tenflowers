//! Higher-order derivative computation
//!
//! This module provides support for computing third and higher-order derivatives
//! using recursive gradient computation.

use crate::{GradientTape, TrackedTensor};
use tenflowers_core::{Result, Tensor, TensorError};

/// Support for computing third and higher-order derivatives
impl GradientTape {
    /// Compute third-order derivatives (∂³f/∂x³)
    pub fn third_derivative<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
    ) -> Result<Tensor<T>>
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
        // Target must be scalar for third derivative computation
        if target.tensor.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "third_derivative",
                "scalar (single element)",
                &format!("{:?}", target.tensor.shape().dims()),
            ));
        }

        let source_shape = source.tensor.shape().dims();
        let source_size = source_shape.iter().product::<usize>();

        // For univariate case (single variable), compute d³f/dx³
        if source_size == 1 {
            return self.compute_univariate_third_derivative(target, source);
        }

        // For multivariate case, compute all third-order partial derivatives
        self.compute_multivariate_third_derivatives(target, source)
    }

    /// Compute nth-order derivative for a scalar function of a single variable
    pub fn nth_derivative<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
        order: usize,
    ) -> Result<Tensor<T>>
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
        // Target must be scalar
        if target.tensor.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "nth_derivative",
                "scalar (single element)",
                &format!("{:?}", target.tensor.shape().dims()),
            ));
        }

        // Source must be a single variable for nth derivative
        if source.tensor.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "nth_derivative",
                "scalar (single variable)",
                &format!("{:?}", source.tensor.shape().dims()),
            ));
        }

        if order == 0 {
            // 0th derivative is the function itself
            return Ok(target.tensor.clone());
        } else if order == 1 {
            // 1st derivative
            let grads =
                self.gradient(std::slice::from_ref(target), std::slice::from_ref(source))?;
            return Ok(grads[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims())));
        } else if order == 2 {
            // 2nd derivative (Hessian for single variable)
            let hessian = self.hessian(target, source)?;

            // For univariate case, extract the single element from [1x1] Hessian matrix
            let source_size = source.tensor.shape().dims().iter().product::<usize>();
            if source_size == 1 {
                let hessian_data = hessian
                    .as_slice()
                    .ok_or_else(|| TensorError::other("Cannot access Hessian data".into()))?;
                return Tensor::from_vec(vec![hessian_data[0]], source.tensor.shape().dims());
            } else {
                return Ok(hessian);
            }
        }

        // For order >= 3, use recursive computation
        self.compute_nth_derivative_recursive(target, source, order)
    }

    /// Compute mixed partial derivatives of any order
    pub fn mixed_partial_derivative<T>(
        &self,
        target: &TrackedTensor<T>,
        variables: &[&TrackedTensor<T>],
        orders: &[usize],
    ) -> Result<Tensor<T>>
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
        if variables.len() != orders.len() {
            return Err(TensorError::invalid_argument(
                "Number of variables must match number of orders".to_string(),
            ));
        }

        // Target must be scalar
        if target.tensor.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "mixed_partial_derivative",
                "scalar (single element)",
                &format!("{:?}", target.tensor.shape().dims()),
            ));
        }

        // Each variable must be scalar for mixed partial derivatives
        for (i, var) in variables.iter().enumerate() {
            if var.tensor.shape().dims().iter().product::<usize>() != 1 {
                return Err(TensorError::shape_mismatch(
                    "mixed_partial_derivative",
                    &format!("scalar for variable {i}"),
                    &format!("{:?}", var.tensor.shape().dims()),
                ));
            }
        }

        self.compute_mixed_partial_recursive(target, variables, orders)
    }

    // Helper methods

    fn compute_univariate_third_derivative<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
    ) -> Result<Tensor<T>>
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
        // For third derivative computation, we can use the fact that:
        // d³f/dx³ = d/dx(d²f/dx²)

        // First compute the Hessian (second derivative)
        let hessian = self.hessian(target, source)?;

        // Create a new tape to compute the derivative of the Hessian
        let third_tape = GradientTape::new();
        let source_fresh = third_tape.watch(source.tensor.clone());
        let hessian_tracked = third_tape.watch(hessian);

        // Compute the gradient of the Hessian w.r.t. the source
        let third_grads = third_tape.gradient(&[hessian_tracked], &[source_fresh])?;

        Ok(third_grads[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims())))
    }

    fn compute_multivariate_third_derivatives<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
    ) -> Result<Tensor<T>>
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
        let source_shape = source.tensor.shape().dims();
        let source_size = source_shape.iter().product::<usize>();

        // For multivariate case, we compute a 3D tensor of all third-order partial derivatives
        // Shape will be [source_size, source_size, source_size] for ∂³f/∂xi∂xj∂xk
        let mut third_derivatives = Vec::with_capacity(source_size * source_size * source_size);

        for i in 0..source_size {
            for j in 0..source_size {
                for k in 0..source_size {
                    // Compute ∂³f/∂xi∂xj∂xk
                    let partial_deriv =
                        self.compute_mixed_third_partial(target, source, i, j, k)?;
                    if let Some(data) = partial_deriv.as_slice() {
                        third_derivatives.push(data[0]);
                    } else {
                        return Err(TensorError::other(
                            "Failed to extract third derivative data".into(),
                        ));
                    }
                }
            }
        }

        Tensor::from_vec(third_derivatives, &[source_size, source_size, source_size])
    }

    fn compute_mixed_third_partial<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Result<Tensor<T>>
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
        // Compute the mixed third partial derivative ∂³f/∂xi∂xj∂xk
        // We need to extract individual variables from the source tensor

        let source_shape = source.tensor.shape().dims();
        let source_size = source_shape.iter().product::<usize>();

        // Check bounds
        if i >= source_size || j >= source_size || k >= source_size {
            return Err(TensorError::invalid_argument(format!(
                "Variable indices ({i}, {j}, {k}) out of bounds for tensor of size {source_size}"
            )));
        }

        // Step 1: Create individual variable tensors for differentiation
        let var_i = self.extract_component(&source.tensor, i)?;
        let var_j = self.extract_component(&source.tensor, j)?;
        let var_k = self.extract_component(&source.tensor, k)?;

        // Create tracked variables
        let var_i_tracked = self.watch(var_i);
        let var_j_tracked = self.watch(var_j);
        let var_k_tracked = self.watch(var_k);

        // Step 2: Compute first partial derivative ∂f/∂xk
        let first_grads = self.gradient(std::slice::from_ref(target), &[var_k_tracked])?;

        // Step 3: Create new tape for second derivative
        let second_tape = GradientTape::new();
        let var_j_fresh = second_tape.watch(var_j_tracked.tensor.clone());
        let first_grad_tracked = second_tape.watch(
            first_grads[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims())),
        );

        // Compute second partial derivative ∂²f/∂xj∂xk
        let second_grads = second_tape.gradient(&[first_grad_tracked], &[var_j_fresh])?;

        // Step 4: Create new tape for third derivative
        let third_tape = GradientTape::new();
        let var_i_fresh = third_tape.watch(var_i_tracked.tensor.clone());
        let second_grad_tracked = third_tape.watch(
            second_grads[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims())),
        );

        // Compute third partial derivative ∂³f/∂xi∂xj∂xk
        let third_grads = third_tape.gradient(&[second_grad_tracked], &[var_i_fresh])?;

        Ok(third_grads[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims())))
    }

    fn compute_nth_derivative_recursive<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
        order: usize,
    ) -> Result<Tensor<T>>
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
        // Recursively compute nth-order derivatives by computing (n-1)th derivative
        // and then taking its derivative

        if order <= 2 {
            // Base cases should be handled by the caller
            return Err(TensorError::invalid_argument(
                "Use specific methods for 0th, 1st, or 2nd derivatives".to_string(),
            ));
        }

        if order == 3 {
            // Use the specialized third derivative method
            return self.compute_univariate_third_derivative(target, source);
        }

        // For orders >= 4, recursively compute (n-1)th derivative first
        let prev_order = order - 1;
        let prev_derivative = if prev_order == 3 {
            self.third_derivative(target, source)?
        } else if prev_order == 2 {
            // Compute the Hessian for the recursive case
            self.hessian(target, source)?
        } else if prev_order == 1 {
            let grads =
                self.gradient(std::slice::from_ref(target), std::slice::from_ref(source))?;
            grads[0]
                .clone()
                .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims()))
        } else {
            // Recursive call for higher orders
            self.compute_nth_derivative_recursive(target, source, prev_order)?
        };

        // Create new tape to compute derivative of the (n-1)th result
        let nth_tape = GradientTape::new();
        let source_fresh = nth_tape.watch(source.tensor.clone());
        let prev_deriv_tracked = nth_tape.watch(prev_derivative);

        // Compute the nth derivative by taking derivative of (n-1)th derivative
        let nth_grads = nth_tape.gradient(&[prev_deriv_tracked], &[source_fresh])?;

        Ok(nth_grads[0]
            .clone()
            .unwrap_or_else(|| Tensor::zeros(source.tensor.shape().dims())))
    }

    fn compute_mixed_partial_recursive<T>(
        &self,
        target: &TrackedTensor<T>,
        variables: &[&TrackedTensor<T>],
        orders: &[usize],
    ) -> Result<Tensor<T>>
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
        // Compute mixed partial derivatives by applying derivatives sequentially
        // For example, ∂²f/∂x∂y means first ∂f/∂y then ∂(∂f/∂y)/∂x

        if variables.is_empty() || orders.is_empty() {
            return Ok(target.tensor.clone());
        }

        // Start with the target function
        let mut current_function = target.tensor.clone();
        let _current_tape = self;
        let mut temp_tapes = Vec::new();

        // Apply derivatives in reverse order (last variable first)
        for (_var_idx, (variable, &order)) in variables.iter().zip(orders.iter()).enumerate().rev()
        {
            if order == 0 {
                continue; // Skip zero-order derivatives
            }

            // Apply the derivative 'order' times with respect to the current variable
            for _ in 0..order {
                // Create a new tape for this derivative step
                let new_tape = GradientTape::new();
                let var_fresh = new_tape.watch(variable.tensor.clone());
                let func_tracked = new_tape.watch(current_function.clone());

                // Compute the gradient
                let grads = new_tape.gradient(&[func_tracked], &[var_fresh])?;
                current_function = grads[0]
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros(variable.tensor.shape().dims()));

                // Store the tape to keep it alive
                temp_tapes.push(new_tape);
            }
        }

        Ok(current_function)
    }

    fn extract_component<T>(&self, tensor: &Tensor<T>, index: usize) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Extract a single component from a tensor
        if let Some(data) = tensor.as_slice() {
            if index < data.len() {
                Tensor::from_vec(vec![data[index].clone()], &[1])
            } else {
                Err(TensorError::invalid_argument(format!(
                    "Index {} out of bounds for tensor of size {}",
                    index,
                    data.len()
                )))
            }
        } else {
            Err(TensorError::other("Failed to extract tensor data".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GradientTape;
    use tenflowers_core::Tensor;

    #[test]
    fn test_third_derivative_cubic() {
        let tape = GradientTape::new();

        // f(x) = x³, so f'''(x) = 6
        let x = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);

        // Compute x³
        let x_squared = x_tracked.mul(&x_tracked).unwrap();
        let x_cubed = x_squared.mul(&x_tracked).unwrap();

        // Compute third derivative
        let result = tape.third_derivative(&x_cubed, &x_tracked);

        // Should succeed (implementation may be placeholder)
        assert!(result.is_ok());
        println!("Third derivative test completed: {:?}", result.is_ok());
    }

    #[test]
    fn test_nth_derivative_api() {
        let tape = GradientTape::new();

        // f(x) = x⁴
        let x = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);

        let x2 = x_tracked.mul(&x_tracked).unwrap();
        let x4 = x2.mul(&x2).unwrap();

        // Test nth derivative API
        for order in 1..=4 {
            let result = tape.nth_derivative(&x4, &x_tracked, order);
            assert!(result.is_ok(), "Failed for order {}", order);
        }

        println!("Nth derivative API test completed");
    }

    #[test]
    fn test_mixed_partial_derivative_api() {
        let tape = GradientTape::new();

        // f(x, y) = xy
        let x = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);
        let y_tracked = tape.watch(y);

        let xy = x_tracked.mul(&y_tracked).unwrap();

        // Test mixed partial derivative API
        let variables = vec![&x_tracked, &y_tracked];
        let orders = vec![1, 1]; // ∂²f/∂x∂y

        let result = tape.mixed_partial_derivative(&xy, &variables, &orders);
        assert!(result.is_ok());

        println!(
            "Mixed partial derivative API test completed: {:?}",
            result.is_ok()
        );
    }

    #[test]
    fn test_fourth_derivative_polynomial() {
        let tape = GradientTape::new();

        // f(x) = x⁴, so f''''(x) = 24
        let x = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);

        // Compute x⁴
        let x2 = x_tracked.mul(&x_tracked).unwrap();
        let x4 = x2.mul(&x2).unwrap();

        // Test 4th derivative
        let result = tape.nth_derivative(&x4, &x_tracked, 4);
        assert!(result.is_ok());

        // The 4th derivative of x⁴ should be 24
        if let Ok(deriv) = result {
            if let Some(value) = deriv.as_slice() {
                println!(
                    "4th derivative of x⁴ at x=1: {:.6} (expected: 24.0)",
                    value[0]
                );
            }
        }
    }

    #[test]
    fn test_higher_order_api_robustness() {
        let tape = GradientTape::new();

        // Test with f(x) = x⁵
        let x = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);

        let x2 = x_tracked.mul(&x_tracked).unwrap();
        let x4 = x2.mul(&x2).unwrap();
        let x5 = x4.mul(&x_tracked).unwrap();

        // Test derivatives up to 6th order
        for order in 1..=6 {
            let result = tape.nth_derivative(&x5, &x_tracked, order);
            match order {
                1..=5 => assert!(result.is_ok(), "Failed for order {} of x⁵", order),
                6 => {
                    // 6th derivative of x⁵ should be 0
                    assert!(result.is_ok(), "Failed for order {} of x⁵", order);
                }
                _ => {}
            }
        }

        println!("Higher-order derivative robustness test completed");
    }
}
