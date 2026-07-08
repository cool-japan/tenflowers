use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

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
            // The element-wise ReLU mask is defined as `1 where x > 0, else 0`.
            // Computing it requires reading the input element values. For tensors
            // whose data is not directly accessible on the host (e.g. GPU-resident
            // storage), the values cannot be inspected here, and returning all-ones
            // would be a silent fabrication that breaks ReLU backward. Honest error
            // instead: a correct mask is not obtainable on this path.
            Err(TensorError::unsupported_operation_simple(
                "relu_mask requires host-accessible tensor data; the mask cannot be \
                 computed for tensors without an accessible CPU slice (move to CPU first)"
                    .to_string(),
            ))
        }
    }

    fn gt_zero(&self) -> Result<Tensor<T>> {
        self.relu_mask()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_mask_mixed_sign() {
        // Mixed-sign input: negatives, zero, and positives.
        // The ReLU mask must be 1 where x > 0 and 0 elsewhere (including x == 0).
        let input = Tensor::from_vec(vec![-2.0f32, -0.5, 0.0, 0.5, 3.0], &[5])
            .expect("test: tensor construction should succeed");

        let mask = input
            .relu_mask()
            .expect("test: relu_mask should succeed on CPU data");

        let mask_data = mask
            .as_slice()
            .expect("test: mask should be host-accessible");

        // Strictly-positive entries -> 1.0; zero and negatives -> 0.0.
        assert_eq!(mask_data, &[0.0f32, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_relu_mask_all_negative_is_not_all_ones() {
        // Guards against the previous fabrication that returned all-ones:
        // an all-negative input must yield an all-zero mask.
        let input = Tensor::from_vec(vec![-1.0f32, -2.0, -3.0, -4.0], &[4])
            .expect("test: tensor construction should succeed");

        let mask = input
            .relu_mask()
            .expect("test: relu_mask should succeed on CPU data");
        let mask_data = mask
            .as_slice()
            .expect("test: mask should be host-accessible");

        assert_eq!(mask_data, &[0.0f32, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_mask_all_positive() {
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 0.25], &[3])
            .expect("test: tensor construction should succeed");

        let mask = input
            .relu_mask()
            .expect("test: relu_mask should succeed on CPU data");
        let mask_data = mask
            .as_slice()
            .expect("test: mask should be host-accessible");

        assert_eq!(mask_data, &[1.0f32, 1.0, 1.0]);
    }
}
