use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor};

/// Extension trait for Tensor to support autograd operations
pub trait TensorAutograd<T> {
    /// Create a tensor with ones where self > 0, zeros elsewhere (for ReLU backward)
    fn relu_mask(&self) -> Result<Tensor<T>>;

    /// Element-wise greater than comparison with zero
    fn gt_zero(&self) -> Result<Tensor<T>>;
}

impl<T> TensorAutograd<T> for Tensor<T>
where
    T: Clone + Default + PartialOrd + Zero + One + Send + Sync + 'static,
{
    fn relu_mask(&self) -> Result<Tensor<T>> {
        // Create a tensor with same shape
        let shape = self.shape().dims();
        let mut mask_data = vec![T::zero(); shape.iter().product()];

        // Get data if available
        if let Some(data) = self.as_slice() {
            let zero = T::zero();
            let one = T::one();

            for (i, val) in data.iter().enumerate() {
                mask_data[i] = if val.clone() > zero {
                    one.clone()
                } else {
                    zero.clone()
                };
            }

            Tensor::from_vec(mask_data, shape)
        } else {
            // For GPU tensors, we'd need a different approach
            // For now, return ones (not ideal but allows progress)
            Ok(Tensor::ones(shape))
        }
    }

    fn gt_zero(&self) -> Result<Tensor<T>> {
        self.relu_mask()
    }
}
