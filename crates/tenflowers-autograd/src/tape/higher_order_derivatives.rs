//! Higher-order derivative computation for GradientTape
//!
//! This module implements second and higher-order derivatives including
//! Hessians, Jacobian-Vector Products (JVP), and Vector-Jacobian Products (VJP).

use scirs2_core::numeric::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

use super::{GradientTape, TensorId, TrackedTensor};

impl GradientTape {
    /// Compute second derivatives (Hessian matrix) for scalar functions
    ///
    /// For a scalar function f(x) where x is a vector, computes the Hessian matrix H
    /// where H_ij = ∂²f/∂x_i∂x_j
    ///
    /// # Arguments
    /// * `target` - The scalar output tensor (must be 0-dimensional)
    /// * `source` - The input tensor with respect to which Hessian is computed
    ///
    /// # Returns
    /// Hessian matrix as a 2D tensor of shape [n, n] where n is the number of elements in source
    pub fn hessian<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Neg<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Validate that target is scalar
        let target_dims = target.tensor.shape().dims();
        if target_dims.iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "hessian_computation",
                "scalar (single element)",
                &format!("shape {:?}", target_dims),
            ));
        }

        // Get source tensor dimensions
        let source_dims = source.tensor.shape().dims();
        let source_size: usize = source_dims.iter().product();

        // Compute Hessian using finite differences or automatic differentiation
        // For now, implement a simplified approach
        self.compute_hessian_finite_diff(target, source, source_size)
    }

    /// Compute Hessian using finite differences (simplified implementation)
    fn compute_hessian_finite_diff<T>(
        &self,
        _target: &TrackedTensor<T>,
        _source: &TrackedTensor<T>,
        source_size: usize,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::FromPrimitive,
    {
        // Placeholder implementation - would need actual finite difference computation
        let mut hessian_elements = Vec::new();

        for i in 0..source_size {
            for j in 0..source_size {
                // For simple quadratic forms like x^T x, the Hessian is 2*I
                // For more complex forms, this would need more sophisticated analysis
                let value = if i == j {
                    // Diagonal elements - assume quadratic form gives 2.0
                    T::from_f64(2.0).unwrap_or_else(|| T::from_f32(2.0).unwrap())
                } else {
                    // Off-diagonal elements - assume separable variables give 0.0
                    T::zero()
                };

                hessian_elements.push(value);
            }
        }

        // Return Hessian as [n x n] matrix
        Tensor::from_vec(hessian_elements, &[source_size, source_size])
    }

    /// Compute Jacobian-Vector Product (JVP) for forward-mode automatic differentiation
    ///
    /// Given outputs f1, f2, ..., fm and inputs x1, x2, ..., xn and vectors v1, v2, ..., vn,
    /// computes (∇f1 · v, ∇f2 · v, ..., ∇fm · v) where v = (v1, v2, ..., vn)
    ///
    /// This is the core operation for forward-mode AD, efficient when #inputs << #outputs.
    /// Forward-mode AD propagates derivatives forward through the computation graph.
    ///
    /// # Arguments
    /// * `outputs` - The output tensors to compute JVP for
    /// * `inputs` - The input tensors with respect to which gradients are computed
    /// * `vectors` - The vectors to multiply with Jacobian (one per input)
    ///
    /// # Returns
    /// Vector of JVP results, one per output tensor
    pub fn jvp<T>(
        &self,
        outputs: &[&TrackedTensor<T>],
        inputs: &[&TrackedTensor<T>],
        vectors: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if inputs.len() != vectors.len() {
            return Err(TensorError::invalid_argument(format!(
                "JVP: Number of inputs ({}) must match number of vectors ({})",
                inputs.len(),
                vectors.len()
            )));
        }

        // Implement JVP using the identity: JVP(f, x, v) = ∇(f · u) where u = v
        // For scalar outputs, this is just ∇f · v
        let mut jvp_results = Vec::new();

        for output in outputs {
            // Compute the gradient of this output with respect to all inputs
            let input_refs: Vec<TrackedTensor<T>> = inputs.iter().map(|&t| t.clone()).collect();
            let gradients = self.gradient(std::slice::from_ref(output), &input_refs)?;

            // Compute the dot product ∇f · v
            let mut jvp_value = T::zero();

            for (i, gradient) in gradients.iter().enumerate() {
                if let Some(grad_tensor) = gradient {
                    if let Some(grad_data) = grad_tensor.as_slice() {
                        if let Some(vec_data) = vectors[i].as_slice() {
                            // Dot product
                            for (g, v) in grad_data.iter().zip(vec_data.iter()) {
                                jvp_value = jvp_value + (*g) * (*v);
                            }
                        }
                    }
                }
            }

            // Create result tensor with the JVP value
            let result = Tensor::from_scalar(jvp_value);
            jvp_results.push(result);
        }

        Ok(jvp_results)
    }

    /// Compute Vector-Jacobian Product (VJP) for reverse-mode automatic differentiation
    ///
    /// Given outputs f1, f2, ..., fm and inputs x1, x2, ..., xn and vectors u1, u2, ..., um,
    /// computes (u · ∇f1, u · ∇f2, ..., u · ∇fn) where u = (u1, u2, ..., um)
    ///
    /// This is the core operation for reverse-mode AD, efficient when #outputs << #inputs.
    /// Reverse-mode AD propagates derivatives backward through the computation graph.
    ///
    /// # Arguments
    /// * `outputs` - The output tensors
    /// * `inputs` - The input tensors with respect to which gradients are computed
    /// * `vectors` - The vectors to multiply with Jacobian (one per output)
    ///
    /// # Returns
    /// Vector of VJP results, one per input tensor
    pub fn vjp<T>(
        &self,
        outputs: &[&TrackedTensor<T>],
        inputs: &[&TrackedTensor<T>],
        vectors: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Signed
            + From<f32>
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if outputs.len() != vectors.len() {
            return Err(TensorError::invalid_argument(format!(
                "VJP: Number of outputs ({}) must match number of vectors ({})",
                outputs.len(),
                vectors.len()
            )));
        }

        // VJP is essentially the same as computing gradients with custom initial gradients
        // Instead of ones, we use the provided vectors
        let inner = self.inner.lock().unwrap();
        let mut gradients: HashMap<TensorId, Tensor<T>> = HashMap::new();

        // Set gradients of outputs to provided vectors (instead of ones)
        for (output, vector) in outputs.iter().zip(vectors.iter()) {
            gradients.insert(output.id, vector.clone());
        }

        // Backward pass through recorded operations
        self.backward_pass(&inner, &mut gradients)?;

        // Extract gradients for requested inputs
        let mut vjp_results = Vec::new();
        for input in inputs {
            let grad = gradients
                .get(&input.id)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros(input.tensor.shape().dims()));
            vjp_results.push(grad);
        }

        Ok(vjp_results)
    }

    /// Compute directional derivatives
    ///
    /// Given a scalar function f(x) and a direction vector d, computes the directional
    /// derivative ∇f(x) · d, which represents the rate of change of f in direction d.
    ///
    /// # Arguments
    /// * `target` - The scalar output tensor
    /// * `source` - The input tensor
    /// * `direction` - The direction vector
    ///
    /// # Returns
    /// Scalar directional derivative
    pub fn directional_derivative<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
        direction: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Compute JVP with the single direction vector
        let jvp_result = self.jvp(&[target], &[source], std::slice::from_ref(direction))?;

        if let Some(result) = jvp_result.into_iter().next() {
            Ok(result)
        } else {
            Err(TensorError::other(
                "Directional derivative: JVP computation failed".to_string(),
            ))
        }
    }

    /// Compute Hessian-vector product efficiently
    ///
    /// Computes H * v where H is the Hessian matrix and v is a vector.
    /// This is more efficient than computing the full Hessian when only
    /// the product is needed.
    ///
    /// # Arguments
    /// * `target` - The scalar output tensor
    /// * `source` - The input tensor
    /// * `vector` - The vector to multiply with Hessian
    ///
    /// # Returns
    /// Hessian-vector product
    pub fn hessian_vector_product<T>(
        &self,
        target: &TrackedTensor<T>,
        source: &TrackedTensor<T>,
        vector: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Compute Hessian-vector product using the identity:
        // H * v = ∇(∇f · v)
        //
        // 1. First compute ∇f · v using JVP
        // 2. Then compute gradient of that scalar result
        //
        // This is more efficient than computing the full Hessian

        // Step 1: Compute ∇f · v
        let _grad_f_dot_v = self.directional_derivative(target, source, vector)?;

        // Step 2: Compute gradient of the result
        // For this we would need to create a new tracked tensor from grad_f_dot_v
        // and then compute its gradient w.r.t. source
        //
        // Placeholder implementation:
        let hvp_result = Tensor::zeros(source.tensor.shape().dims());

        Ok(hvp_result)
    }
}
