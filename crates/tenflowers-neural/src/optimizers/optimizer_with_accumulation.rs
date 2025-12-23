use crate::model::Model;
use crate::optimizers::Optimizer;
use std::marker::PhantomData;
use tenflowers_autograd::GradientAccumulator;
use tenflowers_core::{Result, Tensor};

/// Optimizer wrapper that supports gradient accumulation
///
/// This wrapper allows any optimizer to work with gradient accumulation,
/// enabling large batch training by accumulating gradients over multiple
/// micro-batches before applying the optimizer step.
pub struct OptimizerWithAccumulation<T, O> {
    inner_optimizer: O,
    _accumulator: GradientAccumulator,
    accumulation_steps: usize,
    current_step: usize,
    _phantom: PhantomData<T>,
}

impl<T, O> OptimizerWithAccumulation<T, O>
where
    O: Optimizer<T>,
{
    /// Create a new optimizer with gradient accumulation
    ///
    /// # Arguments
    /// * `optimizer` - The base optimizer to wrap
    /// * `accumulation_steps` - Number of micro-batches to accumulate before optimizer step
    pub fn new(optimizer: O, accumulation_steps: usize) -> Self {
        Self {
            inner_optimizer: optimizer,
            _accumulator: GradientAccumulator::new(true), // Average gradients by default
            accumulation_steps,
            current_step: 0,
            _phantom: PhantomData,
        }
    }

    /// Accumulate gradients without updating parameters
    /// Call this method for each micro-batch
    pub fn accumulate_gradients(&mut self, model: &mut dyn Model<T>) -> Result<()>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Store current gradients in accumulator
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                // We need to simulate accumulation here since we don't have TrackedTensor
                // For now, we'll use the parameter itself as a key and manually accumulate
                // This is a simplified implementation - in a real scenario, we'd integrate
                // more deeply with the autograd system

                // For now, just scale the gradient by 1/accumulation_steps
                let scale = T::from_usize(self.accumulation_steps).unwrap();
                let scaled_grad = grad.div(&Tensor::from_scalar(scale))?;
                param.set_grad(Some(scaled_grad));
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Check if it's time to apply the optimizer step
    pub fn should_step(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Apply accumulated gradients and reset accumulation
    pub fn apply_accumulated_gradients(&mut self, model: &mut dyn Model<T>) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::Float
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive,
    {
        if self.should_step() {
            // Apply the accumulated gradients using the inner optimizer
            self.inner_optimizer.step(model)?;

            // Reset accumulation
            self.current_step = 0;
        }

        Ok(())
    }

    /// Convenience method that combines accumulate_gradients and apply_accumulated_gradients
    pub fn step_with_accumulation(&mut self, model: &mut dyn Model<T>) -> Result<bool>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + PartialOrd
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        self.accumulate_gradients(model)?;

        if self.should_step() {
            self.apply_accumulated_gradients(model)?;
            Ok(true) // Parameters were updated
        } else {
            Ok(false) // Parameters were not updated
        }
    }

    /// Set the number of accumulation steps
    pub fn set_accumulation_steps(&mut self, steps: usize) {
        self.accumulation_steps = steps;
        self.current_step = 0; // Reset accumulation
    }

    /// Get the number of accumulation steps
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Get the current accumulation progress
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get a reference to the inner optimizer
    pub fn inner_optimizer(&self) -> &O {
        &self.inner_optimizer
    }

    /// Get a mutable reference to the inner optimizer
    pub fn inner_optimizer_mut(&mut self) -> &mut O {
        &mut self.inner_optimizer
    }
}

impl<T, O> Optimizer<T> for OptimizerWithAccumulation<T, O>
where
    O: Optimizer<T>,
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Float
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive,
{
    /// Standard step method - applies accumulated gradients if ready
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        // For standard step, we assume gradients are already accumulated
        // and we should apply them if we have enough steps
        self.current_step += 1;

        if self.should_step() {
            self.inner_optimizer.step(model)?;
            self.current_step = 0;
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        self.inner_optimizer.zero_grad(model);
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.inner_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> f32 {
        self.inner_optimizer.get_learning_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use crate::model::sequential::Sequential;
    use crate::optimizers::sgd::SGD;

    #[test]
    fn test_optimizer_with_accumulation_creation() {
        let sgd = SGD::<f32>::new(0.01);
        let optimizer = OptimizerWithAccumulation::new(sgd, 4);

        assert_eq!(optimizer.accumulation_steps(), 4);
        assert_eq!(optimizer.current_step(), 0);
        assert!(!optimizer.should_step());
        assert_eq!(optimizer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_accumulation_steps() {
        let sgd = SGD::<f32>::new(0.01);
        let mut optimizer = OptimizerWithAccumulation::new(sgd, 3);

        // Test step counting
        assert!(!optimizer.should_step());

        optimizer.current_step = 1;
        assert!(!optimizer.should_step());

        optimizer.current_step = 2;
        assert!(!optimizer.should_step());

        optimizer.current_step = 3;
        assert!(optimizer.should_step());

        // Test setting accumulation steps
        optimizer.set_accumulation_steps(5);
        assert_eq!(optimizer.accumulation_steps(), 5);
        assert_eq!(optimizer.current_step(), 0);
    }

    #[test]
    fn test_inner_optimizer_access() {
        let sgd = SGD::<f32>::new(0.01);
        let mut optimizer = OptimizerWithAccumulation::new(sgd, 2);

        // Test immutable access
        assert_eq!(optimizer.inner_optimizer().get_learning_rate(), 0.01);

        // Test mutable access
        optimizer.inner_optimizer_mut().set_learning_rate(0.1);
        assert_eq!(optimizer.get_learning_rate(), 0.1);
    }
}
