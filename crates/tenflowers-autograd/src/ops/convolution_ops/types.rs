use num_traits::{One, Zero};
use tenflowers_core::{Result, Tensor};

/// Type alias for Conv2D backward result to reduce complexity
pub type Conv2dBackwardResult<T> = Result<(Tensor<T>, Tensor<T>, Option<Tensor<T>>)>;

/// Type alias for Conv3D backward result to reduce complexity
pub type Conv3dBackwardResult<T> = Result<(Tensor<T>, Tensor<T>, Option<Tensor<T>>)>;

/// Type alias for ConvTranspose2D backward result to reduce complexity
pub type ConvTranspose2dBackwardResult<T> = Result<(Tensor<T>, Tensor<T>, Option<Tensor<T>>)>;

/// Helper function to get a specific element from a 4D tensor
pub(crate) fn get_tensor_element_4d<T>(
    tensor: &Tensor<T>,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Option<T>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let shape = tensor.shape().dims();
    if b >= shape[0] || c >= shape[1] || h >= shape[2] || w >= shape[3] {
        return None;
    }

    // Flatten index calculation for 4D tensor [batch, channel, height, width]
    let flat_index =
        b * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w;

    // Use tensor's data access method (this is a simplified approach)
    // In a real implementation, you'd use the tensor's proper indexing
    if let Ok(vec_data) = tensor.to_vec() {
        vec_data.get(flat_index).cloned()
    } else {
        None
    }
}
