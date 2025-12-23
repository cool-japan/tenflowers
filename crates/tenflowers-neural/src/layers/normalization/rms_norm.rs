//! RMS Normalization implementation
//!
//! RMS Normalization is a simpler and more efficient alternative to LayerNorm,
//! used in modern language models like LLaMA for improved performance and stability.
//!
//! RMSNorm normalizes by the root mean square without centering (no mean subtraction).
//! Formula: RMSNorm(x) = (x / RMS(x)) * scale, where RMS(x) = sqrt(mean(x^2) + epsilon)
//!
//! Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// RMS Normalization - A simpler and more efficient alternative to LayerNorm
/// Used in modern language models like LLaMA for improved performance and stability
///
/// RMSNorm normalizes by the root mean square without centering (no mean subtraction)
/// Formula: RMSNorm(x) = (x / RMS(x)) * scale, where RMS(x) = sqrt(mean(x^2) + epsilon)
///
/// Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
#[derive(Debug, Clone)]
pub struct RMSNorm<T> {
    /// Shape of the normalized features
    normalized_shape: Vec<usize>,
    /// Learnable scale parameter (equivalent to gamma in LayerNorm)
    scale: Tensor<T>,
    /// Small constant for numerical stability
    epsilon: f32,
}

impl<T> RMSNorm<T>
where
    T: Clone + Default + Zero + One + bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new RMSNorm layer
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the input features to normalize over
    ///   Typically the last dimension(s) of the input tensor
    ///
    /// # Example
    /// ```rust,ignore
    /// // For transformer models, typically normalize over the last dimension (hidden_size)
    /// let rms_norm = RMSNorm::new(&[768]); // hidden_size = 768
    ///
    /// // For multi-dimensional normalization
    /// let rms_norm = RMSNorm::new(&[512, 128]); // normalize over last 2 dimensions
    /// ```
    pub fn new(normalized_shape: &[usize]) -> Self {
        Self {
            normalized_shape: normalized_shape.to_vec(),
            // Initialize scale to ones (no scaling initially)
            scale: Tensor::ones(normalized_shape),
            epsilon: 1e-6, // Default epsilon for numerical stability
        }
    }

    /// Set the epsilon value for numerical stability
    ///
    /// # Arguments
    /// * `epsilon` - Small constant added to the variance for numerical stability
    ///   Default is 1e-6, which works well for most applications
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Get the normalized shape
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Get the epsilon value
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl<T> Layer<T> for RMSNorm<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let shape = input.shape();
        let ndim = shape.rank();

        // RMSNorm normalizes over the last N dimensions
        let norm_dims = self.normalized_shape.len();
        if ndim < norm_dims {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Input rank {ndim} is less than normalized dimensions {norm_dims}"
            )));
        }

        // Check that the last dimensions match normalized_shape
        let input_dims = &shape.dims()[ndim - norm_dims..];
        if input_dims != self.normalized_shape.as_slice() {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Expected last {} dimensions to be {:?}, got {:?}",
                norm_dims, self.normalized_shape, input_dims
            )));
        }

        // Compute RMS over the normalized dimensions
        let axes: Vec<i32> = (ndim - norm_dims..ndim).map(|i| i as i32).collect();

        // Step 1: Compute x^2
        let squared = input.mul(input)?;

        // Step 2: Compute mean(x^2) over normalized dimensions
        let mean_squared = squared.mean(Some(&axes), true)?;

        // Step 3: Add epsilon for numerical stability: mean(x^2) + epsilon
        let eps = T::from(self.epsilon).unwrap();
        let eps_tensor = Tensor::from_scalar(eps);
        let variance_with_eps = mean_squared.add(&eps_tensor)?;

        // Step 4: Compute RMS = sqrt(mean(x^2) + epsilon)
        let rms = variance_with_eps.sqrt()?;

        // Step 5: Normalize: x / RMS
        let normalized = input.div(&rms)?;

        // Step 6: Apply learnable scale parameter
        let output = normalized.mul(&self.scale)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.scale] // Only scale parameter, no bias unlike LayerNorm
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.scale]
    }

    fn set_training(&mut self, _training: bool) {
        // RMSNorm behavior doesn't change between training and eval
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> crate::layers::LayerType {
        crate::layers::LayerType::RMSNorm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_creation() {
        let rms_norm = RMSNorm::<f32>::new(&[768]);
        assert_eq!(rms_norm.normalized_shape, vec![768]);
        assert_eq!(rms_norm.epsilon, 1e-6);
    }

    #[test]
    fn test_rms_norm_with_epsilon() {
        let rms_norm = RMSNorm::<f32>::new(&[512]).with_epsilon(1e-8);
        assert_eq!(rms_norm.epsilon, 1e-8);
    }

    #[test]
    fn test_rms_norm_parameters() {
        let rms_norm = RMSNorm::<f32>::new(&[256]);
        let params = rms_norm.parameters();
        assert_eq!(params.len(), 1); // Only scale parameter, no bias
    }

    #[test]
    fn test_rms_norm_multidimensional() {
        let rms_norm = RMSNorm::<f32>::new(&[64, 128]);
        assert_eq!(rms_norm.normalized_shape, vec![64, 128]);
    }

    #[test]
    fn test_rms_norm_getters() {
        let rms_norm = RMSNorm::<f32>::new(&[1024]).with_epsilon(2e-6);
        assert_eq!(rms_norm.normalized_shape(), &[1024]);
        assert_eq!(rms_norm.epsilon(), 2e-6);
    }

    #[test]
    fn test_rms_norm_training_mode() {
        let mut rms_norm = RMSNorm::<f32>::new(&[512]);

        // RMSNorm doesn't change behavior based on training mode
        rms_norm.set_training(true);
        rms_norm.set_training(false);

        // Should succeed without error
        assert_eq!(rms_norm.normalized_shape(), &[512]);
    }

    #[test]
    fn test_rms_norm_layer_type() {
        let rms_norm = RMSNorm::<f32>::new(&[128]);
        assert_eq!(rms_norm.layer_type(), crate::layers::LayerType::RMSNorm);
    }
}
