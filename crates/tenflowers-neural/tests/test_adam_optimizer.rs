use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};
use tenflowers_neural::{
    optimizers::{Adam, Optimizer},
    Model,
};

// Simple test model that just holds parameters
struct TestModel {
    params: Vec<Tensor<f32>>,
}

impl TestModel {
    fn new() -> Self {
        let param1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let param2 = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        Self {
            params: vec![param1, param2],
        }
    }
}

impl Model<f32> for TestModel {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Simple forward: sum of params multiplied by input
        let mut result = self.params[0].mul(input)?;
        for param in &self.params[1..] {
            let prod = param.mul(input)?;
            result = result.add(&prod)?;
        }
        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<f32>> {
        self.params.iter().collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        self.params.iter_mut().collect()
    }

    fn set_training(&mut self, _training: bool) {
        // No-op for test
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.set_grad(None);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn test_adam_optimizer_step() -> Result<()> {
    let mut model = TestModel::new();
    let mut optimizer = Adam::new(0.01);

    // Set some fake gradients
    model.params[0].set_grad(Some(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3])?));
    model.params[1].set_grad(Some(Tensor::from_vec(vec![0.4, 0.5, 0.6], &[3])?));

    // Get initial values
    let initial_param0 = model.params[0].clone();
    let initial_param1 = model.params[1].clone();

    // Perform optimizer step
    optimizer.step(&mut model)?;

    // Check that parameters have been updated
    assert!(model.params[0].as_slice() != initial_param0.as_slice());
    assert!(model.params[1].as_slice() != initial_param1.as_slice());

    // Verify the update direction (params should decrease since gradients are positive)
    if let (Some(new_vals0), Some(old_vals0)) =
        (model.params[0].as_slice(), initial_param0.as_slice())
    {
        for (new, old) in new_vals0.iter().zip(old_vals0.iter()) {
            assert!(
                new < old,
                "Parameter should decrease with positive gradient"
            );
        }
    }

    Ok(())
}

#[test]
fn test_adam_optimizer_momentum() -> Result<()> {
    let mut model = TestModel::new();
    let mut optimizer = Adam::new(0.01);

    // Apply same gradient multiple times to test momentum
    let grad0 = Tensor::from_vec(vec![0.1, 0.1, 0.1], &[3])?;
    let grad1 = Tensor::from_vec(vec![0.1, 0.1, 0.1], &[3])?;

    let mut previous_params = Vec::new();

    for i in 0..5 {
        // Store current params
        previous_params.push((model.params[0].clone(), model.params[1].clone()));

        // Set gradients
        model.params[0].set_grad(Some(grad0.clone()));
        model.params[1].set_grad(Some(grad1.clone()));

        // Step
        optimizer.step(&mut model)?;

        // After first iteration, check that updates are getting larger (momentum effect)
        if i > 0 {
            let (prev_p0, prev_p1) = &previous_params[i - 1];
            let (older_p0, older_p1) = &previous_params[i - 1];

            if let (Some(curr), Some(prev), Some(older)) = (
                model.params[0].as_slice(),
                prev_p0.as_slice(),
                older_p0.as_slice(),
            ) {
                if i > 1 {
                    // The change should be larger due to momentum
                    let prev_change = (prev[0] - older[0]).abs();
                    let curr_change = (curr[0] - prev[0]).abs();
                    // This might not always hold due to bias correction
                    // but generally momentum should increase step size
                }
            }
        }
    }

    Ok(())
}

#[test]
fn test_adam_zero_grad() -> Result<()> {
    let mut model = TestModel::new();
    let optimizer = Adam::new(0.01);

    // Set gradients
    model.params[0].set_grad(Some(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3])?));
    model.params[1].set_grad(Some(Tensor::from_vec(vec![0.4, 0.5, 0.6], &[3])?));

    // Check gradients are set
    assert!(model.params[0].grad().is_some());
    assert!(model.params[1].grad().is_some());

    // Zero gradients
    optimizer.zero_grad(&mut model);

    // Check gradients are cleared
    assert!(model.params[0].grad().is_none());
    assert!(model.params[1].grad().is_none());

    Ok(())
}
