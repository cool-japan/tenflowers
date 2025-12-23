//! Autograd wrapper for neural network layers

use super::traits::NeuralLayer;
// use crate::gradient_accumulation::GradientAccumulator; // Unused for now
use crate::tape::{GradientTape, TrackedTensor};
use scirs2_core::numeric::{Float, One, Zero};
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor};

/// Wrapper for neural network layers that provides autograd integration
#[derive(Debug, Clone)]
pub struct AutogradLayer<T, L> {
    /// The underlying neural network layer
    layer: L,
    /// Gradient tape for automatic differentiation
    tape: Arc<Mutex<GradientTape>>,
    /// Tracked parameters for gradient computation
    tracked_parameters: Vec<TrackedTensor<T>>,
    /// Parameter names for debugging and serialization
    parameter_names: Vec<String>,
    /// Training mode flag
    training: bool,
}

impl<T, L> AutogradLayer<T, L>
where
    T: Float
        + Clone
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
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
    L: NeuralLayer<T> + Clone,
{
    /// Create a new autograd layer wrapping a neural network layer
    pub fn new(layer: L, tape: Arc<Mutex<GradientTape>>) -> Result<Self> {
        // Extract parameters from the layer and wrap them with autograd tracking
        let parameters = layer.parameters();
        let mut tracked_parameters = Vec::new();
        let mut parameter_names = Vec::new();

        for (i, param) in parameters.iter().enumerate() {
            println!(
                "Debug: Creating AutogradLayer - param {} shape: {:?}",
                i,
                param.shape()
            );
            let tracked = {
                let tape_ref = tape.lock().unwrap();
                tape_ref.watch((*param).clone())
            };
            println!(
                "Debug: Tracked param {} shape: {:?}",
                i,
                tracked.tensor.shape()
            );
            tracked_parameters.push(tracked);
            parameter_names.push(format!("param_{i}"));
        }

        Ok(Self {
            layer,
            tape,
            tracked_parameters,
            parameter_names,
            training: false,
        })
    }

    /// Forward pass through the layer with automatic gradient tracking
    pub fn forward(&self, input: &TrackedTensor<T>) -> Result<TrackedTensor<T>> {
        // For dense layers, perform matrix multiplication directly
        if self.tracked_parameters.len() >= 2 {
            // Assume first parameter is weight, second is bias
            let weight = &self.tracked_parameters[0];
            let bias = &self.tracked_parameters[1];

            println!(
                "Debug: AutogradLayer forward - input shape: {:?}",
                input.tensor.shape()
            );
            println!(
                "Debug: AutogradLayer forward - weight shape: {:?}",
                weight.tensor.shape()
            );
            println!(
                "Debug: AutogradLayer forward - bias shape: {:?}",
                bias.tensor.shape()
            );

            // Handle 1D inputs by reshaping to [1, features] for matrix multiplication
            let input_2d = if input.tensor.ndim() == 1 {
                let reshaped_tensor = input.tensor.reshape(&[1, input.tensor.shape().dims()[0]])?;
                println!(
                    "Debug: AutogradLayer forward - reshaped input shape: {:?}",
                    reshaped_tensor.shape()
                );
                let tape_ref = self.tape.lock().unwrap();
                tape_ref.watch(reshaped_tensor)
            } else {
                input.clone()
            };

            // Perform matrix multiplication: output = input @ weight + bias
            let matmul_result = input_2d.matmul(weight)?;
            println!(
                "Debug: AutogradLayer forward - matmul result shape: {:?}",
                matmul_result.tensor.shape()
            );
            let output = matmul_result.add(bias)?;
            println!(
                "Debug: AutogradLayer forward - after bias add shape: {:?}",
                output.tensor.shape()
            );

            // If input was 1D, squeeze the output back to 1D
            if input.tensor.ndim() == 1 {
                // Check if we can actually squeeze axis 0
                if output.tensor.shape().dims()[0] == 1 {
                    let squeezed_tensor = output.tensor.squeeze(Some(&[0]))?;
                    let tape_ref = self.tape.lock().unwrap();
                    Ok(tape_ref.watch(squeezed_tensor))
                } else {
                    // Cannot squeeze, return as-is
                    println!(
                        "Warning: Cannot squeeze axis 0 of size {}, returning unsqueezed",
                        output.tensor.shape().dims()[0]
                    );
                    Ok(output)
                }
            } else {
                Ok(output)
            }
        } else {
            // Fallback for other layer types
            let mut layer_copy = self.layer.clone();
            layer_copy.set_training(self.training);

            let result = layer_copy.forward(&input.tensor)?;
            let tape_ref = self.tape.lock().unwrap();
            Ok(tape_ref.watch(result))
        }
    }

    /// Get tracked parameters
    pub fn parameters(&self) -> &[TrackedTensor<T>] {
        &self.tracked_parameters
    }

    /// Get mutable tracked parameters
    pub fn parameters_mut(&mut self) -> &mut [TrackedTensor<T>] {
        &mut self.tracked_parameters
    }

    /// Get parameter names
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.layer.set_training(training);
    }

    /// Check if layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Apply gradients to parameters with learning rate
    pub fn apply_gradients(&mut self, gradients: &[Tensor<T>], learning_rate: T) -> Result<()> {
        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.tracked_parameters.len() {
                // Create updated parameter tensor
                let current_param = &self.tracked_parameters[i].tensor;
                let scaled_gradient = gradient.mul_scalar(learning_rate)?;
                let updated_param = current_param.sub(&scaled_gradient)?;

                // Update tracked parameter
                let tape_ref = self.tape.lock().unwrap();
                self.tracked_parameters[i] = tape_ref.watch(updated_param);
            }
        }
        Ok(())
    }

    /// Get the underlying layer
    pub fn layer(&self) -> &L {
        &self.layer
    }

    /// Get mutable reference to the underlying layer
    pub fn layer_mut(&mut self) -> &mut L {
        &mut self.layer
    }

    /// Clone the layer with a new gradient tape
    pub fn clone_with_tape(&self, new_tape: Arc<Mutex<GradientTape>>) -> Result<Self> {
        Self::new(self.layer.clone(), new_tape)
    }
}
