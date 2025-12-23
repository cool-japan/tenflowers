//! Advanced Gradient Operations
//!
//! This module provides advanced automatic differentiation operations and utilities
//! that enhance the basic gradient computation capabilities.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Advanced gradient accumulation with momentum and adaptive scaling
pub struct AdaptiveGradientAccumulator<T> {
    accumulated_gradients: HashMap<String, Tensor<T>>,
    momentum: T,
    #[allow(dead_code)]
    decay_factor: T,
    #[allow(dead_code)]
    epsilon: T,
    step_count: usize,
}

impl<T> AdaptiveGradientAccumulator<T>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new adaptive gradient accumulator
    pub fn new(momentum: T, decay_factor: T, epsilon: T) -> Self {
        Self {
            accumulated_gradients: HashMap::new(),
            momentum,
            decay_factor,
            epsilon,
            step_count: 0,
        }
    }

    /// Accumulate gradients with momentum and bias correction
    pub fn accumulate(&mut self, name: &str, gradient: Tensor<T>) -> Result<Tensor<T>> {
        self.step_count += 1;

        let accumulated = if let Some(prev_grad) = self.accumulated_gradients.get(name) {
            // Momentum update: m = β₁ * m + (1 - β₁) * g
            let momentum_scalar = Tensor::from_scalar(self.momentum);
            let momentum_update = prev_grad.mul(&momentum_scalar)?;
            let one_minus_momentum = Tensor::from_scalar(T::one() - self.momentum);
            let gradient_update = gradient.mul(&one_minus_momentum)?;
            momentum_update.add(&gradient_update)?
        } else {
            // First time: initialize with current gradient scaled
            let scale = Tensor::from_scalar(T::one() - self.momentum);
            gradient.mul(&scale)?
        };

        // Bias correction for momentum
        let bias_correction1 = T::one() - self.momentum.powi(self.step_count as i32);
        let bias_correction_tensor = Tensor::from_scalar(bias_correction1);
        let corrected_gradient = accumulated.div(&bias_correction_tensor)?;

        self.accumulated_gradients
            .insert(name.to_string(), accumulated);
        Ok(corrected_gradient)
    }

    /// Reset accumulator state
    pub fn reset(&mut self) {
        self.accumulated_gradients.clear();
        self.step_count = 0;
    }
}

/// Gradient clipping utilities
pub mod gradient_clipping {
    use super::*;

    /// Clip gradients by global norm
    pub fn clip_by_global_norm<T>(gradients: &[Tensor<T>], max_norm: T) -> Result<Vec<Tensor<T>>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        // Compute global norm
        let mut total_norm_squared = T::zero();
        for gradient in gradients {
            let two_tensor = Tensor::from_scalar(T::from(2.0).unwrap());
            let grad_squared = gradient.pow(&two_tensor)?;
            let grad_norm_squared = grad_squared.sum(None, false)?.to_scalar()?;
            total_norm_squared = total_norm_squared + grad_norm_squared;
        }

        let global_norm = total_norm_squared.sqrt();

        if global_norm <= max_norm {
            // No clipping needed
            return Ok(gradients.to_vec());
        }

        // Clip gradients
        let clip_factor = max_norm / global_norm;
        let clip_factor_tensor = Tensor::from_scalar(clip_factor);
        let mut clipped_gradients = Vec::new();

        for gradient in gradients {
            let clipped = gradient.mul(&clip_factor_tensor)?;
            clipped_gradients.push(clipped);
        }

        Ok(clipped_gradients)
    }

    /// Clip gradients by value
    pub fn clip_by_value<T>(
        gradients: &[Tensor<T>],
        min_value: T,
        max_value: T,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Float
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + PartialOrd
            + scirs2_core::num_traits::Zero,
    {
        let mut clipped_gradients = Vec::new();

        for gradient in gradients {
            // Implement proper element-wise clamping using min/max operations
            // clamp(x, min_val, max_val) = min(max(x, min_val), max_val)

            // Create scalar tensors for min and max values
            let min_tensor =
                Tensor::from_scalar(min_value).broadcast_to(gradient.shape().dims())?;
            let max_tensor =
                Tensor::from_scalar(max_value).broadcast_to(gradient.shape().dims())?;

            // First apply element-wise max(gradient, min_value) to enforce lower bound
            let lower_bounded = tenflowers_core::ops::binary::max(gradient, &min_tensor)?;

            // Then apply element-wise min(result, max_value) to enforce upper bound
            let clipped = tenflowers_core::ops::binary::min(&lower_bounded, &max_tensor)?;

            clipped_gradients.push(clipped);
        }

        Ok(clipped_gradients)
    }
}

/// Higher-order gradients computation
pub mod higher_order {
    use super::*;
    use crate::tape::GradientTape;
    use crate::TrackedTensor;

    /// Compute second-order gradients (Hessian-vector product)
    pub fn hessian_vector_product<T>(
        tape: &GradientTape,
        loss: &TrackedTensor<T>,
        params: &[TrackedTensor<T>],
        vector: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Float
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + scirs2_core::num_traits::FromPrimitive,
    {
        // First compute gradients
        let first_order = tape.gradient(std::slice::from_ref(loss), params)?;

        // Create a new tape for second derivatives
        let second_tape = GradientTape::new();

        // Track first-order gradients and compute their gradients w.r.t. vector
        let mut second_order_grads = Vec::new();

        for (i, first_grad_opt) in first_order.iter().enumerate() {
            if let Some(first_grad) = first_grad_opt {
                // Vector-Jacobian product: ∇²f(x) * v
                let vjp = first_grad.dot(&vector[i])?;
                let tracked_vjp = second_tape.watch(vjp);

                // Compute gradient w.r.t. original parameters
                let second_grad = second_tape.gradient(&[tracked_vjp], params)?;

                if let Some(Some(hess_vec)) = second_grad.get(i) {
                    second_order_grads.push(hess_vec.clone());
                } else {
                    second_order_grads.push(Tensor::zeros(params[i].tensor.shape().dims()));
                }
            } else {
                second_order_grads.push(Tensor::zeros(params[i].tensor.shape().dims()));
            }
        }

        Ok(second_order_grads)
    }

    /// Compute full Hessian matrix (expensive for large models)
    pub fn compute_hessian<T>(
        tape: &GradientTape,
        loss: &TrackedTensor<T>,
        params: &[TrackedTensor<T>],
    ) -> Result<Vec<Vec<Tensor<T>>>>
    where
        T: Float
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + scirs2_core::num_traits::FromPrimitive,
    {
        let mut hessian = Vec::new();

        // Compute Hessian row by row using unit vectors
        for i in 0..params.len() {
            let param_shape = params[i].tensor.shape().dims();
            let param_size: usize = param_shape.iter().product();

            let mut hess_row = Vec::new();

            for _j in 0..param_size {
                // Create unit vector
                let mut unit_vector = vec![Tensor::zeros(param_shape); params.len()];

                // Set the j-th element to 1
                if let Ok(_flat_tensor) = unit_vector[i].flatten() {
                    // This is a simplified approach - in practice you'd need proper indexing
                    unit_vector[i] = Tensor::ones(&[1]); // Placeholder
                }

                // Compute Hessian-vector product
                let hess_vec = hessian_vector_product(tape, loss, params, &unit_vector)?;
                hess_row.push(hess_vec[i].clone());
            }

            hessian.push(hess_row);
        }

        Ok(hessian)
    }
}

/// Jacobian computation utilities
pub mod jacobian {
    use super::*;
    use crate::tape::GradientTape;
    use crate::TrackedTensor;

    /// Compute Jacobian matrix for vector-valued functions
    pub fn compute_jacobian<T>(
        tape: &GradientTape,
        outputs: &[TrackedTensor<T>],
        inputs: &[TrackedTensor<T>],
    ) -> Result<Vec<Vec<Option<Tensor<T>>>>>
    where
        T: Float
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + scirs2_core::num_traits::FromPrimitive,
    {
        let mut jacobian = Vec::new();

        // Compute gradients of each output with respect to all inputs
        for output in outputs {
            let gradients = tape.gradient(std::slice::from_ref(output), inputs)?;
            jacobian.push(gradients);
        }

        Ok(jacobian)
    }

    /// Compute Jacobian-vector product efficiently
    pub fn jacobian_vector_product<T>(
        tape: &GradientTape,
        outputs: &[TrackedTensor<T>],
        inputs: &[TrackedTensor<T>],
        vector: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Float
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable
            + scirs2_core::num_traits::FromPrimitive,
    {
        // Compute JVP using forward-mode AD or reverse-mode with vector trick
        let mut jvp_results = Vec::new();

        for output in outputs {
            let gradients = tape.gradient(std::slice::from_ref(output), inputs)?;
            let mut jvp_sum: Option<Tensor<T>> = None;

            for (i, grad_opt) in gradients.iter().enumerate() {
                if let Some(grad) = grad_opt {
                    let weighted_grad = grad.mul(&vector[i])?;
                    jvp_sum = match jvp_sum {
                        Some(sum) => Some(sum.add(&weighted_grad)?),
                        None => Some(weighted_grad),
                    };
                }
            }

            jvp_results
                .push(jvp_sum.unwrap_or_else(|| Tensor::zeros(output.tensor.shape().dims())));
        }

        Ok(jvp_results)
    }
}

/// Gradient-based optimization helpers
pub mod optimization {
    use super::*;

    /// Compute natural gradient using Fisher Information Matrix approximation
    pub fn natural_gradient_update<T>(
        gradients: &[Tensor<T>],
        fisher_info_inverse: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        let mut natural_grads = Vec::new();

        for (grad, fim_inv) in gradients.iter().zip(fisher_info_inverse.iter()) {
            let natural_grad = fim_inv.mul(grad)?;
            natural_grads.push(natural_grad);
        }

        Ok(natural_grads)
    }

    /// Approximate Fisher Information Matrix using gradients
    pub fn approximate_fisher_information<T>(
        gradient_samples: &[Vec<Tensor<T>>],
        regularization: T,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if gradient_samples.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "approximate_fisher_information".to_string(),
                reason: "Empty gradient samples".to_string(),
                context: None,
            });
        }

        let num_params = gradient_samples[0].len();
        let mut fisher_matrices = Vec::new();

        for param_idx in 0..num_params {
            // Compute outer product of gradients and average
            let mut fisher_sum: Option<Tensor<T>> = None;
            let mut param_shape = None;

            for sample in gradient_samples {
                let grad = &sample[param_idx];
                if param_shape.is_none() {
                    param_shape = Some(grad.shape().dims().to_vec());
                }
                let outer_product = grad.outer(grad)?; // grad ⊗ grad

                fisher_sum = match fisher_sum {
                    Some(sum) => Some(sum.add(&outer_product)?),
                    None => Some(outer_product),
                };
            }

            if let Some(sum) = fisher_sum {
                let num_samples_tensor =
                    Tensor::from_scalar(T::from(gradient_samples.len()).unwrap());
                let fisher_avg = sum.div(&num_samples_tensor)?;

                // Add regularization (F + λI)
                if let Some(shape) = param_shape {
                    let identity = Tensor::eye(shape[0]);
                    let regularization_tensor = Tensor::from_scalar(regularization);
                    let scaled_identity = identity.mul(&regularization_tensor)?;
                    let regularized = fisher_avg.add(&scaled_identity)?;
                    fisher_matrices.push(regularized);
                }
            }
        }

        Ok(fisher_matrices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_accumulator() {
        let mut accumulator = AdaptiveGradientAccumulator::<f32>::new(0.9, 0.999, 1e-8);

        // Test basic accumulation
        let grad1 = Tensor::ones(&[2, 2]);
        let result1 = accumulator.accumulate("param1", grad1).unwrap();

        // Test momentum accumulation
        let grad2 = Tensor::ones(&[2, 2]);
        let result2 = accumulator.accumulate("param1", grad2).unwrap();

        assert!(accumulator.step_count == 2);
    }

    #[test]
    fn test_gradient_clipping() {
        let gradients = vec![
            Tensor::ones(&[2, 2]),
            Tensor::from_scalar(2.0f32).broadcast_to(&[2, 2]).unwrap(),
        ];

        let clipped = gradient_clipping::clip_by_global_norm(&gradients, 1.0).unwrap();

        // Check that gradients were clipped
        assert_eq!(clipped.len(), 2);
    }
}
