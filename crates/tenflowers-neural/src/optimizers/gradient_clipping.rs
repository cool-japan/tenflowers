use crate::model::Model;
use scirs2_core::num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Gradient clipping by value - clips each gradient element to [-clip_value, clip_value]
pub fn clip_gradients_by_value<T, M>(model: &mut M, clip_value: T) -> Result<()>
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
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable
        + scirs2_core::num_traits::Signed,
    M: Model<T>,
{
    let clip_tensor = Tensor::from_scalar(clip_value);
    let neg_clip_tensor = Tensor::from_scalar(-clip_value);

    for param in model.parameters_mut() {
        if let Some(grad) = param.grad() {
            // Implement clamp using where_op: clamp(x, min, max) = max(min, min(x, max))
            // First: min(grad, clip_value)
            let upper_clipped =
                tenflowers_core::ops::where_op(&grad.le(&clip_tensor)?, grad, &clip_tensor)?;

            // Second: max(upper_clipped, -clip_value)
            let clipped_grad = tenflowers_core::ops::where_op(
                &upper_clipped.ge(&neg_clip_tensor)?,
                &upper_clipped,
                &neg_clip_tensor,
            )?;

            param.set_grad(Some(clipped_grad));
        }
    }

    Ok(())
}

/// Gradient clipping by norm - scales gradients to have maximum norm of max_norm
pub fn clip_gradients_by_norm<T, M>(model: &mut M, max_norm: T) -> Result<()>
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
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::FromPrimitive
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    M: Model<T>,
{
    // Calculate total gradient norm
    let mut total_norm_squared = T::zero();
    let mut gradients = Vec::new();

    for param in model.parameters() {
        if let Some(grad) = param.grad() {
            gradients.push(grad.clone());
            // Sum of squared elements for this gradient
            let grad_squared = grad.mul(grad)?;
            let norm_squared = tenflowers_core::ops::sum(&grad_squared, None, false)?;
            if let Some(norm_data) = norm_squared.as_slice() {
                total_norm_squared = total_norm_squared + T::from(norm_data[0]).unwrap();
            }
        }
    }

    let total_norm = total_norm_squared.sqrt();

    // Only clip if norm exceeds max_norm
    if total_norm > max_norm {
        let scale_factor = max_norm / total_norm;
        let scale_tensor = Tensor::from_scalar(scale_factor);

        let mut grad_idx = 0;
        for param in model.parameters_mut() {
            if param.grad().is_some() {
                let scaled_grad = gradients[grad_idx].mul(&scale_tensor)?;
                param.set_grad(Some(scaled_grad));
                grad_idx += 1;
            }
        }
    }

    Ok(())
}

/// Gradient clipping by global norm - alternative implementation using L2 norm
pub fn clip_gradients_by_global_norm<T, M>(model: &mut M, max_norm: T) -> Result<T>
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
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::FromPrimitive
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    M: Model<T>,
{
    // Calculate global norm of all gradients
    let mut total_norm_squared = T::zero();
    let mut gradients = Vec::new();

    for param in model.parameters() {
        if let Some(grad) = param.grad() {
            gradients.push(grad.clone());
            // Calculate L2 norm squared for this gradient
            let grad_squared = grad.mul(grad)?;
            let norm_squared = tenflowers_core::ops::sum(&grad_squared, None, false)?;
            if let Some(norm_data) = norm_squared.as_slice() {
                total_norm_squared = total_norm_squared + T::from(norm_data[0]).unwrap();
            }
        }
    }

    let global_norm = total_norm_squared.sqrt();

    // Clip gradients if global norm exceeds max_norm
    if global_norm > max_norm {
        let clip_coeff = max_norm / global_norm;
        let clip_tensor = Tensor::from_scalar(clip_coeff);

        let mut grad_idx = 0;
        for param in model.parameters_mut() {
            if param.grad().is_some() {
                let clipped_grad = gradients[grad_idx].mul(&clip_tensor)?;
                param.set_grad(Some(clipped_grad));
                grad_idx += 1;
            }
        }
    }

    Ok(global_norm)
}

/// Adaptive gradient clipping - clips gradients based on parameter norms
pub fn clip_gradients_adaptive<T, M>(model: &mut M, clip_factor: T) -> Result<()>
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
        + std::cmp::PartialOrd
        + scirs2_core::num_traits::FromPrimitive
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    M: Model<T>,
{
    for param in model.parameters_mut() {
        if let Some(grad) = param.grad() {
            // Calculate parameter norm
            let param_squared = param.mul(param)?;
            let param_norm_squared = tenflowers_core::ops::sum(&param_squared, None, false)?;
            let param_norm = if let Some(norm_data) = param_norm_squared.as_slice() {
                T::from(norm_data[0]).unwrap().sqrt()
            } else {
                T::one()
            };

            // Calculate gradient norm
            let grad_squared = grad.mul(grad)?;
            let grad_norm_squared = tenflowers_core::ops::sum(&grad_squared, None, false)?;
            let grad_norm = if let Some(norm_data) = grad_norm_squared.as_slice() {
                T::from(norm_data[0]).unwrap().sqrt()
            } else {
                T::one()
            };

            // Adaptive clipping threshold
            let threshold = clip_factor * param_norm;

            // Clip if gradient norm exceeds threshold
            if grad_norm > threshold {
                let scale_factor = threshold / grad_norm;
                let scale_tensor = Tensor::from_scalar(scale_factor);
                let clipped_grad = grad.mul(&scale_tensor)?;
                param.set_grad(Some(clipped_grad));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    // Simple test model that implements the Model trait
    struct TestModel {
        params: Vec<Tensor<f32>>,
    }

    impl TestModel {
        fn new() -> Self {
            let mut param1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
            let mut param2 = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

            // Set gradients for testing
            param1.set_grad(Some(
                Tensor::<f32>::from_vec(vec![2.0, -3.0], &[2]).unwrap(),
            ));
            param2.set_grad(Some(Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap()));

            TestModel {
                params: vec![param1, param2],
            }
        }
    }

    impl crate::model::Model<f32> for TestModel {
        fn forward(&self, _input: &Tensor<f32>) -> tenflowers_core::Result<Tensor<f32>> {
            // Not needed for testing
            Ok(Tensor::from_scalar(0.0))
        }

        fn parameters(&self) -> Vec<&Tensor<f32>> {
            self.params.iter().collect()
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
            self.params.iter_mut().collect()
        }

        fn zero_grad(&mut self) {
            for param in &mut self.params {
                param.set_grad(None);
            }
        }

        fn set_training(&mut self, _training: bool) {
            // Not needed for testing
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_clip_gradients_by_value() {
        let mut model = TestModel::new();

        // Clip gradients by value
        clip_gradients_by_value(&mut model, 1.5).unwrap();

        // Check that gradients are clipped
        for param in model.parameters() {
            if let Some(grad) = param.grad() {
                if let Some(grad_data) = grad.as_slice() {
                    for &val in grad_data {
                        assert!(val >= -1.5);
                        assert!(val <= 1.5);
                    }
                }
            }
        }
    }

    #[test]
    fn test_clip_gradients_by_norm() {
        let mut model = TestModel::new();

        // Clip gradients by norm (original norm should be sqrt(2^2 + 3^2 + 3^2 + 4^2) = sqrt(38) ≈ 6.16)
        clip_gradients_by_norm(&mut model, 3.0).unwrap();

        // Check that gradients are scaled down
        let expected_scale = 3.0 / (38.0_f32.sqrt());
        for param in model.parameters() {
            if let Some(grad) = param.grad() {
                if let Some(grad_data) = grad.as_slice() {
                    // Each gradient should be scaled by the same factor
                    assert!(grad_data[0].abs() < 3.0); // Should be scaled down
                    assert!(grad_data[1].abs() < 3.0); // Should be scaled down
                }
            }
        }
    }

    #[test]
    fn test_clip_gradients_by_global_norm() {
        let mut model = TestModel::new();

        // Clip gradients by global norm
        let global_norm = clip_gradients_by_global_norm(&mut model, 3.0).unwrap();

        // Check that global norm is calculated correctly (should be sqrt(38) ≈ 6.16)
        assert!((global_norm - 38.0_f32.sqrt()).abs() < 1e-5);

        // Check that gradients are scaled down
        for param in model.parameters() {
            if let Some(grad) = param.grad() {
                if let Some(grad_data) = grad.as_slice() {
                    // Each gradient should be scaled by the same factor
                    assert!(grad_data[0].abs() < 3.0); // Should be scaled down
                    assert!(grad_data[1].abs() < 3.0); // Should be scaled down
                }
            }
        }
    }
}
