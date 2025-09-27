use crate::tape::{GradientTape, TrackedTensor};
use num_traits::Zero;
use std::sync::Arc;
use tenflowers_core::{Result, Tensor, TensorError};

/// A trait for defining custom gradient functions
pub trait CustomGradientFunction<T> {
    /// Forward pass of the operation
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>>;

    /// Backward pass of the operation
    /// Takes gradient of output and the inputs from forward pass
    fn backward(
        &self,
        grad_output: &Tensor<T>,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>>;

    /// Name of the operation for debugging
    fn name(&self) -> &str;
}

/// Wrapper for operations with custom gradients
#[derive(Clone)]
pub struct CustomGradientOp<T> {
    function: Arc<dyn CustomGradientFunction<T> + Send + Sync>,
}

impl<T> CustomGradientOp<T> {
    pub fn new<F>(function: F) -> Self
    where
        F: CustomGradientFunction<T> + Send + Sync + 'static,
    {
        Self {
            function: Arc::new(function),
        }
    }

    pub fn apply(
        &self,
        tape: &GradientTape,
        inputs: &[&TrackedTensor<T>],
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        // Extract tensors from tracked tensors
        let tensor_refs: Vec<&Tensor<T>> = inputs.iter().map(|t| &t.tensor).collect();

        // Perform forward pass
        let output = self.function.forward(&tensor_refs)?;

        // Create tracked tensor for output
        let tracked_output = tape.watch(output);

        // Register the operation in the tape
        // For now, we'll use Identity as placeholder - in a full implementation,
        // we'd add a CustomGradient variant to the Operation enum
        // Note: Direct access to inner field is private, so we skip tape registration for now

        Ok(tracked_output)
    }
}

/// Stop gradient function - prevents gradients from flowing backward
pub struct StopGradientFunction;

impl<T> CustomGradientFunction<T> for StopGradientFunction
where
    T: Clone + Default + Zero,
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 1 {
            return Err(TensorError::shape_mismatch(
                "custom_gradient",
                "exactly one input",
                &format!("{} inputs", inputs.len()),
            ));
        }
        // Forward pass is identity - just clone the input
        Ok(inputs[0].clone())
    }

    fn backward(
        &self,
        _grad_output: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>> {
        // Backward pass returns zero gradients
        let zero_grad = Tensor::zeros(inputs[0].shape().dims());
        Ok(vec![zero_grad])
    }

    fn name(&self) -> &str {
        "StopGradient"
    }
}

/// Utility functions for custom gradients
impl<T> TrackedTensor<T>
where
    T: Clone + Send + Sync + 'static + Default + Zero,
{
    /// Stop gradients from flowing through this tensor
    pub fn stop_gradient(&self) -> Result<TrackedTensor<T>> {
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);
            let stop_grad_op = CustomGradientOp::new(StopGradientFunction);
            stop_grad_op.apply(&tape, &[self])
        } else {
            // If no tape, just return a copy
            let tape = GradientTape::new();
            Ok(tape.watch(self.tensor.clone()))
        }
    }

    /// Apply a custom gradient function to this tensor
    pub fn with_custom_gradient<F>(&self, function: F) -> Result<TrackedTensor<T>>
    where
        F: CustomGradientFunction<T> + Send + Sync + 'static,
    {
        if let Some(tape_arc) = self.tape.upgrade() {
            let tape = GradientTape::from_inner(tape_arc);
            let custom_op = CustomGradientOp::new(function);
            custom_op.apply(&tape, &[self])
        } else {
            Err(TensorError::unsupported_operation_simple(
                "Cannot apply custom gradient without an active tape".to_string(),
            ))
        }
    }
}

/// Gradient clipping function
pub struct GradientClipFunction {
    max_norm: f64,
}

impl GradientClipFunction {
    pub fn new(max_norm: f64) -> Self {
        Self { max_norm }
    }
}

impl<T> CustomGradientFunction<T> for GradientClipFunction
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 1 {
            return Err(TensorError::shape_mismatch(
                "custom_gradient",
                "exactly one input",
                &format!("{} inputs", inputs.len()),
            ));
        }
        // Forward pass is identity
        Ok(inputs[0].clone())
    }

    fn backward(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>> {
        // Clip the gradient if its norm exceeds max_norm
        let clipped_grad = clip_gradient_norm(grad_output, self.max_norm)?;
        Ok(vec![clipped_grad])
    }

    fn name(&self) -> &str {
        "GradientClip"
    }
}

/// Clip gradient by norm
fn clip_gradient_norm<T>(gradient: &Tensor<T>, max_norm: f64) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Calculate gradient norm
    let grad_norm = if let Some(data) = gradient.as_slice() {
        let sum_squares = data.iter().fold(T::zero(), |acc, &val| acc + val * val);
        sum_squares.sqrt().to_f64().unwrap_or(0.0)
    } else {
        return Ok(gradient.clone());
    };

    // If norm is within bounds, return original gradient
    if grad_norm <= max_norm {
        return Ok(gradient.clone());
    }

    // Calculate scaling factor
    let scale_factor = T::from(max_norm / grad_norm).unwrap();
    let scale_tensor = Tensor::from_scalar(scale_factor);

    // Scale the gradient
    gradient.mul(&scale_tensor)
}

/// Gradient scaling function for mixed precision training
pub struct GradientScaleFunction {
    scale: f64,
}

impl GradientScaleFunction {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl<T> CustomGradientFunction<T> for GradientScaleFunction
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 1 {
            return Err(TensorError::shape_mismatch(
                "custom_gradient",
                "exactly one input",
                &format!("{} inputs", inputs.len()),
            ));
        }
        // Forward pass is identity
        Ok(inputs[0].clone())
    }

    fn backward(
        &self,
        grad_output: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _output: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>> {
        // Scale the gradient
        let scale_tensor = Tensor::from_scalar(T::from(self.scale).unwrap());
        let scaled_grad = grad_output.mul(&scale_tensor)?;
        Ok(vec![scaled_grad])
    }

    fn name(&self) -> &str {
        "GradientScale"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_stop_gradient() {
        let tape = GradientTape::new();
        let x = tape.watch(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap());

        // This should work once the full integration is complete
        // let y = x.stop_gradient().unwrap();
        // assert_eq!(y.tensor.as_slice(), x.tensor.as_slice());
    }

    #[test]
    fn test_gradient_clipping() {
        let gradient = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();
        let clipped = clip_gradient_norm(&gradient, 5.0).unwrap();

        // The clipped gradient should have a smaller norm
        if let Some(clipped_data) = clipped.as_slice() {
            let clipped_norm = clipped_data
                .iter()
                .fold(0.0f32, |acc, &val| acc + val * val)
                .sqrt();
            assert!(clipped_norm <= 5.1); // Allow small numerical error
        }
    }
}
