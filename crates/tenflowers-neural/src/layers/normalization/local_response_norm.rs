//! Local Response Normalization implementation
//!
//! Local Response Normalization (LRN) normalizes across nearby channels or spatial locations.
//! This was used in older CNN architectures like AlexNet for normalization before BatchNorm
//! became popular. Formula: LRN(x) = x / (k + alpha * sum(x_i^2))^beta
//!
//! In CrossChannel mode (default), the sum is over a local neighborhood of channels.
//! In WithinChannel mode, the sum is over a local spatial neighborhood within each channel.

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Local Response Normalization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalResponseNormMode {
    /// Cross-channel normalization (default): normalizes across adjacent channels
    CrossChannel,
    /// Within-channel normalization: normalizes within each channel across spatial dimensions
    WithinChannel,
}

/// Local Response Normalization (LRN) - normalizes across nearby channels or spatial locations
/// Used in older CNN architectures like AlexNet for normalization
/// Formula: LRN(x) = x / (k + alpha * sum(x_i^2))^beta
///
/// In CrossChannel mode (default), the sum is over a local neighborhood of channels.
/// In WithinChannel mode, the sum is over a local spatial neighborhood within each channel.
#[derive(Clone)]
pub struct LocalResponseNorm<T> {
    size: usize,                 // Number of channels/spatial locations to normalize over
    alpha: T,                    // Scaling parameter
    beta: T,                     // Exponent parameter
    k: T,                        // Bias parameter
    mode: LocalResponseNormMode, // Normalization mode
}

impl<T> LocalResponseNorm<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new LocalResponseNorm layer
    ///
    /// # Arguments
    /// * `size` - Number of channels to normalize over (default: 5)
    /// * `alpha` - Scaling parameter (default: 0.0001)
    /// * `beta` - Exponent parameter (default: 0.75)
    /// * `k` - Bias parameter (default: 1.0)
    pub fn new(size: usize, alpha: T, beta: T, k: T) -> Self {
        Self {
            size,
            alpha,
            beta,
            k,
            mode: LocalResponseNormMode::CrossChannel, // Default to cross-channel mode
        }
    }

    /// Create a new LocalResponseNorm layer with specified mode
    ///
    /// # Arguments
    /// * `size` - Number of channels/spatial locations to normalize over
    /// * `alpha` - Scaling parameter
    /// * `beta` - Exponent parameter
    /// * `k` - Bias parameter
    /// * `mode` - Normalization mode (CrossChannel or WithinChannel)
    pub fn new_with_mode(
        size: usize,
        alpha: T,
        beta: T,
        k: T,
        mode: LocalResponseNormMode,
    ) -> Self {
        Self {
            size,
            alpha,
            beta,
            k,
            mode,
        }
    }

    /// Create a new LocalResponseNorm layer with default parameters
    pub fn new_default() -> Self {
        Self::new(
            5,
            T::from(0.0001).unwrap(),
            T::from(0.75).unwrap(),
            T::one(),
        )
    }

    /// Create a new within-channel LocalResponseNorm layer with default parameters
    pub fn new_within_channel() -> Self {
        Self::new_with_mode(
            5,
            T::from(0.0001).unwrap(),
            T::from(0.75).unwrap(),
            T::one(),
            LocalResponseNormMode::WithinChannel,
        )
    }

    /// Set the size parameter
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Set the alpha parameter
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the beta parameter
    pub fn with_beta(mut self, beta: T) -> Self {
        self.beta = beta;
        self
    }

    /// Set the k parameter
    pub fn with_k(mut self, k: T) -> Self {
        self.k = k;
        self
    }

    /// Set the normalization mode
    pub fn with_mode(mut self, mode: LocalResponseNormMode) -> Self {
        self.mode = mode;
        self
    }

    /// Get the size parameter
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// Get the beta parameter
    pub fn beta(&self) -> T {
        self.beta
    }

    /// Get the k parameter
    pub fn k(&self) -> T {
        self.k
    }

    /// Get the normalization mode
    pub fn mode(&self) -> LocalResponseNormMode {
        self.mode
    }
}

impl<T> Layer<T> for LocalResponseNorm<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
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

        // Expect 4D input (NCHW format)
        if ndim != 4 {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "LocalResponseNorm expects 4D input (NCHW), got {ndim}D"
            )));
        }

        let batch_size = shape.dims()[0];
        let channels = shape.dims()[1];
        let height = shape.dims()[2];
        let width = shape.dims()[3];

        // Compute squared input
        let input_squared = input.mul(input)?;

        let mut normalized_tensor = Vec::new();
        let radius = self.size / 2;

        match self.mode {
            LocalResponseNormMode::CrossChannel => {
                // Cross-channel normalization (original behavior)
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..height {
                            for w in 0..width {
                                // Sum over neighboring channels
                                let mut sum_squared = T::zero();
                                let start_ch = c.saturating_sub(radius);
                                let end_ch = std::cmp::min(c + radius + 1, channels);

                                for ch in start_ch..end_ch {
                                    let idx = [b, ch, h, w];
                                    if let Some(val) = input_squared.get(&idx) {
                                        sum_squared = sum_squared + val;
                                    }
                                }

                                // Compute normalization factor: (k + alpha * sum)^beta
                                let norm_factor =
                                    (self.k + self.alpha * sum_squared).powf(self.beta);

                                // Get original value and normalize
                                let original_val = input.get(&[b, c, h, w]).unwrap_or(T::zero());
                                let normalized_val = original_val / norm_factor;

                                normalized_tensor.push(normalized_val);
                            }
                        }
                    }
                }
            }
            LocalResponseNormMode::WithinChannel => {
                // Within-channel normalization across spatial dimensions
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..height {
                            for w in 0..width {
                                // Sum over neighboring spatial locations within the same channel
                                let mut sum_squared = T::zero();

                                // Define spatial neighborhood
                                let start_h = h.saturating_sub(radius);
                                let end_h = std::cmp::min(h + radius + 1, height);
                                let start_w = w.saturating_sub(radius);
                                let end_w = std::cmp::min(w + radius + 1, width);

                                for spatial_h in start_h..end_h {
                                    for spatial_w in start_w..end_w {
                                        let idx = [b, c, spatial_h, spatial_w];
                                        if let Some(val) = input_squared.get(&idx) {
                                            sum_squared = sum_squared + val;
                                        }
                                    }
                                }

                                // Compute normalization factor: (k + alpha * sum)^beta
                                let norm_factor =
                                    (self.k + self.alpha * sum_squared).powf(self.beta);

                                // Get original value and normalize
                                let original_val = input.get(&[b, c, h, w]).unwrap_or(T::zero());
                                let normalized_val = original_val / norm_factor;

                                normalized_tensor.push(normalized_val);
                            }
                        }
                    }
                }
            }
        }

        // Create output tensor
        Tensor::from_vec(normalized_tensor, shape.dims())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![] // LRN has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![] // LRN has no learnable parameters
    }

    fn set_training(&mut self, _training: bool) {
        // LRN behavior doesn't change between training and eval
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_response_norm_creation() {
        let lrn = LocalResponseNorm::<f32>::new(5, 0.0001, 0.75, 1.0);
        assert_eq!(lrn.size(), 5);
        assert_eq!(lrn.alpha(), 0.0001);
        assert_eq!(lrn.beta(), 0.75);
        assert_eq!(lrn.k(), 1.0);
        assert_eq!(lrn.mode(), LocalResponseNormMode::CrossChannel);
    }

    #[test]
    fn test_local_response_norm_with_mode() {
        let lrn = LocalResponseNorm::<f32>::new_with_mode(
            7,
            0.0002,
            0.5,
            2.0,
            LocalResponseNormMode::WithinChannel,
        );
        assert_eq!(lrn.size(), 7);
        assert_eq!(lrn.mode(), LocalResponseNormMode::WithinChannel);
    }

    #[test]
    fn test_local_response_norm_default() {
        let lrn = LocalResponseNorm::<f32>::new_default();
        assert_eq!(lrn.size(), 5);
        assert_eq!(lrn.alpha(), 0.0001);
        assert_eq!(lrn.beta(), 0.75);
        assert_eq!(lrn.k(), 1.0);
        assert_eq!(lrn.mode(), LocalResponseNormMode::CrossChannel);
    }

    #[test]
    fn test_local_response_norm_within_channel() {
        let lrn = LocalResponseNorm::<f32>::new_within_channel();
        assert_eq!(lrn.mode(), LocalResponseNormMode::WithinChannel);
    }

    #[test]
    fn test_local_response_norm_builder_pattern() {
        let lrn = LocalResponseNorm::<f32>::new_default()
            .with_size(7)
            .with_alpha(0.0002)
            .with_beta(0.5)
            .with_k(2.0)
            .with_mode(LocalResponseNormMode::WithinChannel);

        assert_eq!(lrn.size(), 7);
        assert_eq!(lrn.alpha(), 0.0002);
        assert_eq!(lrn.beta(), 0.5);
        assert_eq!(lrn.k(), 2.0);
        assert_eq!(lrn.mode(), LocalResponseNormMode::WithinChannel);
    }

    #[test]
    fn test_local_response_norm_no_parameters() {
        let lrn = LocalResponseNorm::<f32>::new_default();
        let params = lrn.parameters();
        assert_eq!(params.len(), 0); // LRN has no learnable parameters
    }

    #[test]
    fn test_local_response_norm_mode_enum() {
        assert_eq!(
            LocalResponseNormMode::CrossChannel,
            LocalResponseNormMode::CrossChannel
        );
        assert_ne!(
            LocalResponseNormMode::CrossChannel,
            LocalResponseNormMode::WithinChannel
        );
    }

    #[test]
    fn test_local_response_norm_training_mode() {
        let mut lrn = LocalResponseNorm::<f32>::new_default();

        // LRN doesn't change behavior based on training mode
        lrn.set_training(true);
        lrn.set_training(false);

        // Should succeed without error
        assert_eq!(lrn.size(), 5);
    }

    #[test]
    fn test_local_response_norm_getters() {
        let alpha_val = 0.0003_f32;
        let beta_val = 0.8_f32;
        let k_val = 1.5_f32;

        let lrn = LocalResponseNorm::<f32>::new(3, alpha_val, beta_val, k_val);

        assert_eq!(lrn.size(), 3);
        assert_eq!(lrn.alpha(), alpha_val);
        assert_eq!(lrn.beta(), beta_val);
        assert_eq!(lrn.k(), k_val);
        assert_eq!(lrn.mode(), LocalResponseNormMode::CrossChannel);
    }
}
