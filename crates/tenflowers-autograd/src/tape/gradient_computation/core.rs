//! Core gradient computation logic
//!
//! This module contains the main algorithms for computing gradients through
//! the recorded computation graph using reverse-mode automatic differentiation.

use crate::grad_ops;
use scirs2_core::numeric::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

use super::super::helpers::get_tensor_value;
use super::super::structures::GradientTapeInner;
use super::super::{GradientTape, Operation, TensorId, TrackedTensor};

impl GradientTape {
    /// Compute gradients with respect to target tensors
    pub fn gradient<T>(
        &self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<Vec<Option<Tensor<T>>>>
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
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let inner = self.inner.lock().map_err(|_| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "gradient tape lock poisoned".to_string(),
            )
        })?;

        // Initialize gradients map
        let mut gradients: HashMap<TensorId, Tensor<T>> = HashMap::new();

        // Set gradients of targets to ones (initial gradient)
        for target in targets {
            let ones = Tensor::ones(target.tensor.shape().dims());
            gradients.insert(target.id, ones);
        }

        // Backward pass through recorded operations
        self.backward_pass(&inner, &mut gradients)?;

        // Extract gradients for requested sources
        self.extract_source_gradients(&gradients, sources)
    }

    /// Perform backward pass through the computation graph
    pub(crate) fn backward_pass<T>(
        &self,
        inner: &GradientTapeInner,
        gradients: &mut HashMap<TensorId, Tensor<T>>,
    ) -> Result<()>
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
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Simple backward pass through recorded operations
        // Note: This is a simplified implementation
        // A full implementation would require topological sorting
        for node in inner.nodes.iter().rev() {
            if let Some(grad_output) = gradients.get(&node.id).cloned() {
                self.process_operation_backward(inner, &node.operation, &grad_output, gradients)?;
            }
        }

        Ok(())
    }

    /// Process backward pass for a specific operation
    pub(super) fn process_operation_backward<T>(
        &self,
        inner: &GradientTapeInner,
        operation: &Operation,
        grad_output: &Tensor<T>,
        gradients: &mut HashMap<TensorId, Tensor<T>>,
    ) -> Result<()>
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
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        match operation {
            // Basic arithmetic operations - delegated to basic_ops module
            Operation::Add { lhs, rhs } => super::basic_ops::process_add_backward(
                self,
                inner,
                grad_output,
                *lhs,
                *rhs,
                gradients,
            ),
            Operation::Mul { lhs, rhs } => super::basic_ops::process_mul_backward(
                self,
                inner,
                grad_output,
                *lhs,
                *rhs,
                gradients,
            ),
            Operation::Sub { lhs, rhs } => super::basic_ops::process_sub_backward(
                self,
                inner,
                grad_output,
                *lhs,
                *rhs,
                gradients,
            ),
            Operation::Div { lhs, rhs } => super::basic_ops::process_div_backward(
                self,
                inner,
                grad_output,
                *lhs,
                *rhs,
                gradients,
            ),
            Operation::Pow { lhs, rhs } => super::basic_ops::process_pow_backward(
                self,
                inner,
                grad_output,
                *lhs,
                *rhs,
                gradients,
            ),
            Operation::MatMul { lhs, rhs } => super::basic_ops::process_matmul_backward(
                self,
                inner,
                grad_output,
                *lhs,
                *rhs,
                gradients,
            ),

            // Tensor manipulation operations - delegated to tensor_ops module
            Operation::Transpose { input, axes: _ } => {
                super::tensor_ops::process_transpose_backward(
                    self,
                    inner,
                    grad_output,
                    *input,
                    gradients,
                )
            }
            Operation::Reshape {
                input,
                original_shape,
                new_shape: _,
            } => super::tensor_ops::process_reshape_backward(
                self,
                inner,
                grad_output,
                *input,
                original_shape,
                gradients,
            ),
            Operation::Squeeze {
                input,
                axes: _,
                original_shape,
            } => {
                // grad_x = unsqueeze(grad_y) back to the pre-squeeze shape.
                let grad_input = grad_ops::squeeze_backward(grad_output, original_shape)?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }
            Operation::Unsqueeze { input, axes } => {
                // grad_x = squeeze(grad_y) along the axes that were inserted.
                let grad_input = grad_ops::unsqueeze_backward(grad_output, axes)?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }
            Operation::Slice {
                input,
                slice_specs,
                input_shape,
            } => {
                // grad_x = zeros(input_shape) with grad_y scattered back to
                // the original per-dimension (start, step) positions.
                let grad_input = grad_ops::slice_backward(grad_output, input_shape, slice_specs)?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }
            Operation::Concat {
                inputs,
                axis,
                input_shapes,
            } => {
                // grad_y is split along `axis` into one gradient per input,
                // in the same order as `inputs`/`input_shapes` (concat_backward
                // iterates `input_shapes` in that same order). Every input
                // must receive its own accumulated gradient, not just the
                // first.
                let per_input_grads = grad_ops::concat_backward(grad_output, input_shapes, *axis)?;
                for (id, grad) in inputs.iter().zip(per_input_grads) {
                    super::super::utils::accumulate_gradient(gradients, *id, grad)?;
                }
                Ok(())
            }
            Operation::Stack { inputs, axis } => {
                // grad_y is unstacked along `axis` into one gradient per
                // input, in the same order as `inputs`.
                let per_input_grads =
                    grad_ops::stack_backward(grad_output, inputs.len(), *axis)?;
                for (id, grad) in inputs.iter().zip(per_input_grads) {
                    super::super::utils::accumulate_gradient(gradients, *id, grad)?;
                }
                Ok(())
            }
            Operation::Split { input, .. } => {
                // Each split output is recorded as its own tape node whose
                // Operation is actually `Operation::Slice` under the hood
                // (see `TrackedTensor::split`'s doc comment for the design
                // rationale) — a split output IS just a slice of the input
                // along an axis, so backward reuses the real, now-correct
                // `Slice` scatter-back instead of duplicating the logic here.
                // This dispatch arm is therefore unreachable in practice:
                // no tape node is ever recorded with `Operation::Split`
                // itself. It is kept only so the `Operation` enum's variant
                // remains meaningful/self-documenting; if it is ever hit,
                // surface a clear error rather than silently dropping the
                // gradient.
                let _ = input;
                Err(tenflowers_core::TensorError::not_implemented_simple(
                    "Split backward: TrackedTensor::split records each output as its own \
                     Operation::Slice node, so a raw Operation::Split node should never appear \
                     on the tape; if you see this error, something constructed one directly"
                        .to_string(),
                ))
            }
            Operation::Gather {
                input,
                indices,
                axis,
            } => {
                let input_tensor = get_tensor_value::<T>(inner, *input).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "Gather backward: input tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                let grad_input = grad_ops::gather_backward(
                    grad_output,
                    input_tensor.shape().dims(),
                    indices,
                    *axis,
                )?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }
            Operation::Sum {
                input,
                axes: _,
                keepdims: _,
            } => {
                super::tensor_ops::process_sum_backward(self, inner, grad_output, *input, gradients)
            }
            Operation::Mean {
                input,
                axes: _,
                keepdims: _,
            } => super::tensor_ops::process_mean_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),

            // Activation functions - delegated to activation_ops module
            Operation::Relu { input } => super::activation_ops::process_relu_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Sigmoid { input } => super::activation_ops::process_sigmoid_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Tanh { input } => super::activation_ops::process_tanh_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Softmax { input, axis } => {
                super::activation_ops::process_softmax_backward(
                    self,
                    inner,
                    grad_output,
                    *input,
                    *axis,
                    gradients,
                )
            }
            Operation::Gelu { input } => super::activation_ops::process_gelu_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Swish { input } => super::activation_ops::process_swish_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Mish { input } => super::activation_ops::process_mish_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::LeakyRelu { input, alpha } => {
                super::activation_ops::process_leaky_relu_backward(
                    self,
                    inner,
                    grad_output,
                    *input,
                    *alpha,
                    gradients,
                )
            }
            Operation::Elu { input, alpha } => super::activation_ops::process_elu_backward(
                self,
                inner,
                grad_output,
                *input,
                *alpha,
                gradients,
            ),
            Operation::Prelu { input, alpha } => super::activation_ops::process_prelu_backward(
                self,
                inner,
                grad_output,
                *input,
                *alpha,
                gradients,
            ),
            Operation::Log { input } => super::activation_ops::process_log_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Abs { input } => super::activation_ops::process_abs_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::Clamp { input, min, max } => super::activation_ops::process_clamp_backward(
                self,
                inner,
                grad_output,
                *input,
                *min,
                *max,
                gradients,
            ),
            Operation::Relu6 { input } => super::activation_ops::process_relu6_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::HardSwish { input } => super::activation_ops::process_hard_swish_backward(
                self,
                inner,
                grad_output,
                *input,
                gradients,
            ),
            Operation::LogSoftmax { input, axis } => {
                super::activation_ops::process_log_softmax_backward(
                    self,
                    inner,
                    grad_output,
                    *input,
                    *axis,
                    gradients,
                )
            }

            // Neural network operations - delegated to neural_ops module
            Operation::Conv1D {
                input,
                weight,
                bias,
                stride,
                padding,
            } => super::neural_ops::process_conv1d_backward(
                self,
                inner,
                grad_output,
                *input,
                *weight,
                *bias,
                *stride,
                padding,
                gradients,
            ),
            Operation::Conv2D {
                input,
                weight,
                bias,
                stride,
                padding,
            } => super::neural_ops::process_conv2d_backward(
                self,
                inner,
                grad_output,
                *input,
                *weight,
                *bias,
                *stride,
                padding,
                gradients,
            ),
            Operation::Conv3D {
                input,
                weight,
                bias,
                stride,
                padding,
            } => super::neural_ops::process_conv3d_backward(
                self,
                inner,
                grad_output,
                *input,
                *weight,
                *bias,
                *stride,
                padding,
                gradients,
            ),
            Operation::BatchNorm {
                input,
                gamma,
                beta,
                running_mean,
                running_var,
                epsilon,
                training,
            } => super::neural_ops::process_batchnorm_backward(
                self,
                inner,
                grad_output,
                *input,
                *gamma,
                *beta,
                *running_mean,
                *running_var,
                *epsilon,
                *training,
                gradients,
            ),
            Operation::LayerNorm {
                input,
                gamma,
                beta,
                normalized_shape,
                epsilon,
            } => super::neural_ops::process_layernorm_backward(
                self,
                inner,
                grad_output,
                *input,
                *gamma,
                *beta,
                normalized_shape.clone(),
                *epsilon,
                gradients,
            ),
            Operation::GroupNorm {
                input,
                gamma,
                beta,
                num_groups,
                epsilon,
            } => {
                let input_tensor = get_tensor_value::<T>(inner, *input).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "GroupNorm backward: input tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                let gamma_tensor = get_tensor_value::<T>(inner, *gamma).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "GroupNorm backward: gamma tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                let beta_tensor = get_tensor_value::<T>(inner, *beta).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "GroupNorm backward: beta tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;

                let epsilon_t = T::from(*epsilon).unwrap_or_else(|| T::default());

                let (grad_input, grad_gamma, grad_beta) =
                    crate::ops::normalization_ops::group_norm_backward(
                        grad_output,
                        &input_tensor,
                        &gamma_tensor,
                        &beta_tensor,
                        *num_groups,
                        epsilon_t,
                    )?;

                super::super::utils::accumulate_gradient(gradients, *input, grad_input)?;
                super::super::utils::accumulate_gradient(gradients, *gamma, grad_gamma)?;
                super::super::utils::accumulate_gradient(gradients, *beta, grad_beta)
            }
            Operation::InstanceNorm {
                input,
                gamma,
                beta,
                epsilon,
            } => {
                let input_tensor = get_tensor_value::<T>(inner, *input).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "InstanceNorm backward: input tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                let gamma_tensor = get_tensor_value::<T>(inner, *gamma).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "InstanceNorm backward: gamma tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                let beta_tensor = get_tensor_value::<T>(inner, *beta).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "InstanceNorm backward: beta tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;

                let epsilon_t = T::from(*epsilon).unwrap_or_else(|| T::default());

                let (grad_input, grad_gamma, grad_beta) =
                    crate::ops::normalization_ops::instance_norm_backward(
                        grad_output,
                        &input_tensor,
                        &gamma_tensor,
                        &beta_tensor,
                        epsilon_t,
                    )?;

                super::super::utils::accumulate_gradient(gradients, *input, grad_input)?;
                super::super::utils::accumulate_gradient(gradients, *gamma, grad_gamma)?;
                super::super::utils::accumulate_gradient(gradients, *beta, grad_beta)
            }

            // Pooling operations - dispatch to the real pooling backward kernels
            Operation::MaxPool2D {
                input,
                kernel_size,
                stride,
                padding,
            } => {
                let input_tensor = get_tensor_value::<T>(inner, *input).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "MaxPool2D backward: input tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                // The forward pooling op does not record a dilation factor, so the
                // backward pass uses the default unit dilation that matches it.
                let grad_input = crate::ops::convolution_ops::max_pool2d_backward(
                    grad_output,
                    &input_tensor,
                    *kernel_size,
                    *stride,
                    padding,
                    (1, 1),
                )?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }
            Operation::AvgPool2D {
                input,
                kernel_size,
                stride,
                padding,
            } => {
                let input_tensor = get_tensor_value::<T>(inner, *input).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "AvgPool2D backward: input tensor value not recorded on the tape"
                            .to_string(),
                    )
                })?;
                let grad_input = crate::ops::convolution_ops::avg_pool2d_backward(
                    grad_output,
                    &input_tensor,
                    *kernel_size,
                    *stride,
                    padding,
                )?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }

            // Special operations
            Operation::Constant => {
                // Constants don't contribute to gradients
                Ok(())
            }
            Operation::Identity { input } => {
                // Identity operation passes gradient through unchanged
                super::super::utils::accumulate_gradient(gradients, *input, grad_output.clone())
            }
            Operation::Neg { input } => {
                // Negation operation changes sign of gradient
                let neg_grad = tenflowers_core::ops::neg(grad_output)?;
                super::super::utils::accumulate_gradient(gradients, *input, neg_grad)
            }
            Operation::StopGradient { .. } => {
                // Stop gradient operations don't propagate gradients
                Ok(())
            }

            // Einsum operation - simplified backward pass
            Operation::Einsum {
                inputs, equation, ..
            } => {
                // Handle different einsum patterns
                if equation == "ij->ji" && inputs.len() == 1 {
                    // Transpose operation: gradient needs to be transposed back
                    let transposed_grad = tenflowers_core::ops::transpose(grad_output)?;
                    super::super::utils::accumulate_gradient(
                        gradients,
                        inputs[0],
                        transposed_grad,
                    )?;
                } else {
                    // For other operations (element-wise, matrix multiply), pass gradient through
                    for &input_id in inputs {
                        super::super::utils::accumulate_gradient(
                            gradients,
                            input_id,
                            grad_output.clone(),
                        )?;
                    }
                }
                Ok(())
            }

            // Linear algebra operations
            Operation::Pinv { input } => {
                // Get the input tensor for pseudoinverse backward computation. The
                // input value must have been recorded on the tape; otherwise the
                // gradient cannot be computed and we surface a hard error rather
                // than silently dropping it.
                let input_tensor = get_tensor_value::<T>(inner, *input).ok_or_else(|| {
                    tenflowers_core::TensorError::invalid_operation_simple(
                        "Pinv backward: input tensor value not recorded on the tape".to_string(),
                    )
                })?;
                let grad_input = grad_ops::pinv_backward(grad_output, &input_tensor)?;
                super::super::utils::accumulate_gradient(gradients, *input, grad_input)
            }

            // Any operation without an explicit backward arm above does NOT have a
            // gradient implementation. Returning Ok(()) here would silently discard
            // the gradient and make training appear to work while parameters never
            // update, so we surface a hard error instead.
            _ => Err(tenflowers_core::TensorError::not_implemented_simple(
                format!("backward pass not implemented for operation: {operation:?}"),
            )),
        }
    }

    /// Extract gradients for requested source tensors
    pub(crate) fn extract_source_gradients<T>(
        &self,
        gradients: &HashMap<TensorId, Tensor<T>>,
        sources: &[TrackedTensor<T>],
    ) -> Result<Vec<Option<Tensor<T>>>>
    where
        T: Clone,
    {
        let mut result = Vec::with_capacity(sources.len());

        for source in sources {
            let gradient = gradients.get(&source.id).cloned();
            result.push(gradient);
        }

        Ok(result)
    }
}
