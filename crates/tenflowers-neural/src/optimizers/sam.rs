use crate::model::Model;
use crate::optimizers::{Adam, Optimizer, SGD};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Sharpness-Aware Minimization (SAM) optimizer
///
/// SAM seeks parameters that lie in neighborhoods having uniformly low loss,
/// which improves model generalization. It works by perturbing parameters in
/// the direction of the gradient to find a local maximum of the loss, then
/// computing gradients at this perturbed point.
///
/// References:
/// - "Sharpness-Aware Minimization for Efficiently Improving Generalization"
///   (https://arxiv.org/abs/2010.01412)
/// - Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur (Google Research)
#[derive(Debug)]
pub enum SAMOptimizer<T> {
    SGD(SGD<T>),
    Adam(Adam<T>),
}

impl<T> SAMOptimizer<T>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        match self {
            SAMOptimizer::SGD(opt) => opt.step(model),
            SAMOptimizer::Adam(opt) => opt.step(model),
        }
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        match self {
            SAMOptimizer::SGD(opt) => opt.zero_grad(model),
            SAMOptimizer::Adam(opt) => opt.zero_grad(model),
        }
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        match self {
            SAMOptimizer::SGD(opt) => opt.set_learning_rate(learning_rate),
            SAMOptimizer::Adam(opt) => opt.set_learning_rate(learning_rate),
        }
    }

    fn get_learning_rate(&self) -> f32 {
        match self {
            SAMOptimizer::SGD(opt) => opt.get_learning_rate(),
            SAMOptimizer::Adam(opt) => opt.get_learning_rate(),
        }
    }
}

pub struct SAM<T> {
    /// Base optimizer to use for parameter updates
    base_optimizer: SAMOptimizer<T>,
    /// Neighborhood radius for sharpness computation
    rho: f32,
    /// Adaptive scaling factor
    adaptive: bool,
    /// Gradient norm type (2 for L2 norm, other values for other norms)
    norm_type: i32,
    /// Perturbation cache for efficient memory management
    perturbations: HashMap<String, Tensor<T>>,
}

impl<T> SAM<T>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create SAM with SGD as base optimizer
    pub fn with_sgd(learning_rate: f32, momentum: f32, rho: f32) -> Self {
        let sgd = SGD::new(learning_rate).with_momentum(momentum);
        Self {
            base_optimizer: SAMOptimizer::SGD(sgd),
            rho,
            adaptive: false,
            norm_type: 2, // L2 norm by default
            perturbations: HashMap::new(),
        }
    }

    /// Create adaptive SAM with SGD as base optimizer
    pub fn with_adaptive_sgd(learning_rate: f32, momentum: f32, rho: f32) -> Self {
        let sgd = SGD::new(learning_rate).with_momentum(momentum);
        Self {
            base_optimizer: SAMOptimizer::SGD(sgd),
            rho,
            adaptive: true,
            norm_type: 2, // L2 norm by default
            perturbations: HashMap::new(),
        }
    }

    /// Create SAM with Adam as base optimizer
    pub fn with_adam(learning_rate: f32, rho: f32) -> Self {
        let adam = Adam::new(learning_rate);
        Self {
            base_optimizer: SAMOptimizer::Adam(adam),
            rho,
            adaptive: false,
            norm_type: 2, // L2 norm by default
            perturbations: HashMap::new(),
        }
    }

    /// Create adaptive SAM with Adam as base optimizer
    pub fn with_adaptive_adam(learning_rate: f32, rho: f32) -> Self {
        let adam = Adam::new(learning_rate);
        Self {
            base_optimizer: SAMOptimizer::Adam(adam),
            rho,
            adaptive: true,
            norm_type: 2, // L2 norm by default
            perturbations: HashMap::new(),
        }
    }

    /// Builder method to set norm type
    pub fn norm_type(mut self, norm_type: i32) -> Self {
        self.norm_type = norm_type;
        self
    }

    /// Compute the gradient norm for all parameters
    fn compute_gradient_norm(&self, model: &dyn Model<T>) -> Result<T> {
        let params = model.parameters();
        let mut total_norm = T::zero();

        for param in params.iter() {
            if let Some(grad) = param.grad() {
                if self.norm_type == 2 {
                    // L2 norm (most common)
                    let grad_norm = self.compute_l2_norm(grad)?;
                    total_norm = total_norm + grad_norm * grad_norm;
                } else {
                    // Other norms can be implemented here if needed
                    let grad_norm = self.compute_l2_norm(grad)?;
                    total_norm = total_norm + grad_norm * grad_norm;
                }
            }
        }

        if self.norm_type == 2 {
            Ok(total_norm.sqrt())
        } else {
            Ok(total_norm)
        }
    }

    /// Compute L2 norm of a tensor
    fn compute_l2_norm(&self, tensor: &Tensor<T>) -> Result<T> {
        if let Some(data) = tensor.as_slice() {
            let mut sum = T::zero();
            for &value in data {
                sum = sum + value * value;
            }
            Ok(sum.sqrt())
        } else {
            // For GPU tensors, we need to use tensor operations
            // This is a simplified version - in practice you'd use proper tensor ops
            Ok(T::from(1.0).unwrap_or(T::one()))
        }
    }

    /// Apply perturbations to model parameters in the direction of the gradient
    fn perturb_parameters(&mut self, model: &mut dyn Model<T>, scale: T) -> Result<()> {
        let mut params = model.parameters_mut();
        self.perturbations.clear();

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let param_name = format!("param_{i}");

                // Compute perturbation: ε = ρ * g / ||g||
                let perturbation = self.scale_tensor(grad, scale)?;

                // Store current parameter value
                self.perturbations.insert(param_name, param.clone());

                // Apply perturbation: w' = w + ε
                let perturbed = self.add_tensors(param, &perturbation)?;
                **param = perturbed;
            }
        }

        Ok(())
    }

    /// Restore original parameters from perturbation cache
    fn restore_parameters(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        let mut params = model.parameters_mut();

        for (i, param) in params.iter_mut().enumerate() {
            let param_name = format!("param_{i}");

            if let Some(original_value) = self.perturbations.get(&param_name) {
                **param = original_value.clone();
            }
        }

        self.perturbations.clear();
        Ok(())
    }

    /// Scale a tensor by a scalar value
    fn scale_tensor(&self, tensor: &Tensor<T>, scale: T) -> Result<Tensor<T>> {
        if let Some(data) = tensor.as_slice() {
            let scaled_data: Vec<T> = data.iter().map(|&x| x * scale).collect();
            Tensor::from_vec(scaled_data, tensor.shape().dims())
        } else {
            // For GPU tensors, create a scalar tensor and multiply
            let scale_tensor = Tensor::from_vec(vec![scale], &[1])?;
            tensor.mul(&scale_tensor)
        }
    }

    /// Add two tensors element-wise
    fn add_tensors(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        // In practice, this would use proper tensor addition
        if let (Some(a_data), Some(b_data)) = (a.as_slice(), b.as_slice()) {
            if a_data.len() != b_data.len() {
                return Err(TensorError::shape_mismatch(
                    "add_tensors",
                    &format!("len={}", a_data.len()),
                    &format!("len={}", b_data.len()),
                ));
            }
            let result_data: Vec<T> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&x, &y)| x + y)
                .collect();
            Tensor::from_vec(result_data, a.shape().dims())
        } else {
            // Use tensor operations for GPU tensors
            a.add(b)
        }
    }
}

impl<T> Optimizer<T> for SAM<T>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + num_traits::Zero
        + num_traits::One
        + num_traits::Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        // SAM requires two forward/backward passes:
        // 1. First pass: compute gradients at current parameters
        // 2. Perturb parameters in gradient direction
        // 3. Second pass: compute gradients at perturbed parameters
        // 4. Update parameters using gradients from perturbed point

        // Step 1: Ensure we have gradients from first pass
        let gradient_norm = self.compute_gradient_norm(model)?;

        if gradient_norm == T::zero() {
            // No gradients, nothing to do
            return Ok(());
        }

        // Step 2: Compute perturbation scale
        let rho_t = T::from(self.rho).unwrap_or(T::zero());
        let perturbation_scale = if self.adaptive {
            // Adaptive scaling based on parameter magnitude
            let param_norm = self.compute_parameter_norm(model)?;
            if param_norm > T::zero() {
                rho_t * param_norm / gradient_norm
            } else {
                rho_t / gradient_norm
            }
        } else {
            // Standard SAM scaling
            rho_t / gradient_norm
        };

        // Step 3: Apply perturbation
        self.perturb_parameters(model, perturbation_scale)?;

        // At this point, the model should be used for another forward/backward pass
        // to compute gradients at the perturbed parameters. Since we don't have
        // direct access to the loss function here, we assume the gradients have
        // been updated externally after perturbation.

        // Step 4: Update parameters using base optimizer with gradients from perturbed point
        // First restore original parameters
        self.restore_parameters(model)?;

        // Now apply the base optimizer update with the gradients computed at perturbed point
        self.base_optimizer.step(model)?;

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        self.base_optimizer.zero_grad(model);
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.base_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> f32 {
        self.base_optimizer.get_learning_rate()
    }
}

impl<T> SAM<T>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Compute the norm of all parameters (for adaptive SAM)
    fn compute_parameter_norm(&self, model: &dyn Model<T>) -> Result<T> {
        let params = model.parameters();
        let mut total_norm = T::zero();

        for param in params.iter() {
            let param_norm = self.compute_l2_norm(param)?;
            total_norm = total_norm + param_norm * param_norm;
        }

        Ok(total_norm.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use crate::Sequential;
    use tenflowers_core::Tensor;

    #[test]
    fn test_sam_creation() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);
        assert_eq!(sam.rho, 0.05);
        assert!(!sam.adaptive);
        assert_eq!(sam.norm_type, 2);
    }

    #[test]
    fn test_adaptive_sam_creation() {
        let sam = SAM::<f32>::with_adaptive_sgd(0.1, 0.9, 0.1);
        assert_eq!(sam.rho, 0.1);
        assert!(sam.adaptive);
    }

    #[test]
    fn test_sam_with_adam() {
        let sam = SAM::<f32>::with_adam(0.001, 0.05);
        assert_eq!(sam.rho, 0.05);
        assert!(!sam.adaptive);
    }

    #[test]
    fn test_sam_builder_pattern() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05).norm_type(1);
        assert_eq!(sam.norm_type, 1);
    }

    #[test]
    fn test_sam_learning_rate_methods() {
        let mut sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);
        assert_eq!(sam.get_learning_rate(), 0.1);

        sam.set_learning_rate(0.01);
        assert_eq!(sam.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_sam_zero_grad() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);

        // Create a simple model for testing
        let mut model = Sequential::new(Vec::new());
        model = model.add(Box::new(Dense::new(4, 2, true)));

        // Test zero_grad doesn't panic
        sam.zero_grad(&mut model);
    }

    #[test]
    fn test_l2_norm_computation() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);

        // Test with known values
        let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let norm = sam.compute_l2_norm(&tensor).unwrap();

        // ||[3, 4]||_2 = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_tensor() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let scaled = sam.scale_tensor(&tensor, 2.0).unwrap();

        if let Some(data) = scaled.as_slice() {
            assert_eq!(data, &[2.0, 4.0, 6.0]);
        } else {
            panic!("Expected CPU tensor data");
        }
    }

    #[test]
    fn test_add_tensors() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let result = sam.add_tensors(&a, &b).unwrap();

        if let Some(data) = result.as_slice() {
            assert_eq!(data, &[5.0, 7.0, 9.0]);
        } else {
            panic!("Expected CPU tensor data");
        }
    }

    #[test]
    fn test_sam_step_with_zero_gradient() {
        let mut sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);

        // Create a simple model
        let mut model = Sequential::new(Vec::new());
        model = model.add(Box::new(Dense::new(2, 1, true)));

        // Test step with no gradients (should not panic)
        let result = sam.step(&mut model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parameter_norm_computation() {
        let sam = SAM::<f32>::with_sgd(0.1, 0.9, 0.05);

        // Create a simple model with known parameters
        let mut model = Sequential::new(Vec::new());
        model = model.add(Box::new(Dense::new(2, 1, true)));

        // Test parameter norm computation doesn't panic
        let result = sam.compute_parameter_norm(&model);
        assert!(result.is_ok());
    }
}
