//! Spectral Normalization implementation
//!
//! Spectral Normalization normalizes weight matrices by their spectral norm (largest singular value).
//! This is used primarily for stabilizing GAN training by constraining the Lipschitz constant
//! of the network, which helps prevent mode collapse and improves training stability.

use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Spectral Normalization - normalizes weight matrices by their spectral norm
/// Used primarily for stabilizing GAN training by constraining Lipschitz constant
pub struct SpectralNorm<T> {
    weight: Tensor<T>,
    u: Tensor<T>, // Left singular vector estimate
    v: Tensor<T>, // Right singular vector estimate
    num_power_iterations: usize,
    eps: T,
}

impl<T> SpectralNorm<T>
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
    /// Create a new SpectralNorm layer for a weight matrix
    /// weight_shape: shape of the weight matrix [out_features, in_features] for linear layers
    pub fn new(weight_shape: &[usize]) -> Result<Self> {
        if weight_shape.len() != 2 {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "SpectralNorm currently only supports 2D weight matrices, got {}D tensor",
                weight_shape.len()
            )));
        }

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        // Initialize weight matrix (using ones for simplicity - in practice would be initialized externally)
        let weight = Tensor::ones(weight_shape);

        // Initialize u and v vectors (normalized ones)
        let u = Tensor::ones(&[out_features]);
        let v = Tensor::ones(&[in_features]);

        Ok(Self {
            weight,
            u,
            v,
            num_power_iterations: 1,
            eps: T::from(1e-12).unwrap(),
        })
    }

    pub fn with_power_iterations(mut self, num_power_iterations: usize) -> Self {
        self.num_power_iterations = num_power_iterations;
        self
    }

    pub fn with_eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }

    /// Compute spectral norm using power iteration method
    fn compute_spectral_norm(&mut self) -> Result<T> {
        let mut u = self.u.clone();
        let mut v = self.v.clone();

        // Power iteration to find dominant singular value
        for _ in 0..self.num_power_iterations {
            // v = W^T * u / ||W^T * u||
            let wt_u = self
                .weight
                .transpose()?
                .matmul(&u.reshape(&[u.shape().dims()[0], 1])?)?;
            let wt_u_flat = wt_u.reshape(&[wt_u.shape().size()])?;
            // Compute L2 norm manually: sqrt(sum(x^2))
            let wt_u_squared = wt_u_flat.mul(&wt_u_flat)?;
            let sum_squares = tenflowers_core::ops::sum(&wt_u_squared, None, false)?;
            let v_norm = sum_squares.sqrt()?;

            let eps_tensor = Tensor::from_scalar(self.eps);
            let v_norm_safe = v_norm.add(&eps_tensor)?;
            v = wt_u_flat.div(&v_norm_safe)?;

            // u = W * v / ||W * v||
            let w_v = self.weight.matmul(&v.reshape(&[v.shape().dims()[0], 1])?)?;
            let w_v_flat = w_v.reshape(&[w_v.shape().size()])?;
            // Compute L2 norm manually: sqrt(sum(x^2))
            let w_v_squared = w_v_flat.mul(&w_v_flat)?;
            let sum_squares = tenflowers_core::ops::sum(&w_v_squared, None, false)?;
            let u_norm = sum_squares.sqrt()?;

            let u_norm_safe = u_norm.add(&eps_tensor)?;
            u = w_v_flat.div(&u_norm_safe)?;
        }

        // Update stored u and v
        self.u = u.clone();
        self.v = v.clone();

        // Compute spectral norm: sigma = u^T * W * v
        let w_v = self.weight.matmul(&v.reshape(&[v.shape().dims()[0], 1])?)?;
        let w_v_flat = w_v.reshape(&[w_v.shape().size()])?;
        let u_reshaped = u.reshape(&[1, u.shape().dims()[0]])?;
        let sigma = u_reshaped.matmul(&w_v_flat.reshape(&[w_v_flat.shape().dims()[0], 1])?)?;

        // Extract scalar value
        if let Some(sigma_data) = sigma.as_slice() {
            Ok(sigma_data[0])
        } else {
            Err(tenflowers_core::TensorError::invalid_argument(
                "Could not extract spectral norm scalar".to_string(),
            ))
        }
    }

    /// Get the normalized weight matrix
    pub fn get_normalized_weight(&mut self) -> Result<Tensor<T>> {
        let sigma = self.compute_spectral_norm()?;
        let sigma_tensor = Tensor::from_scalar(sigma.max(self.eps));

        // Return W / sigma
        self.weight.div(&sigma_tensor)
    }

    /// Update the weight matrix (used during training)
    pub fn update_weight(&mut self, new_weight: Tensor<T>) -> Result<()> {
        if new_weight.shape() != self.weight.shape() {
            return Err(tenflowers_core::TensorError::shape_mismatch(
                "SpectralNorm::update_weight",
                &self.weight.shape().to_string(),
                &new_weight.shape().to_string(),
            ));
        }
        self.weight = new_weight;
        Ok(())
    }

    /// Get the current weight matrix
    pub fn weight(&self) -> &Tensor<T> {
        &self.weight
    }

    /// Get the current left singular vector estimate
    pub fn u(&self) -> &Tensor<T> {
        &self.u
    }

    /// Get the current right singular vector estimate
    pub fn v(&self) -> &Tensor<T> {
        &self.v
    }

    /// Get the number of power iterations
    pub fn num_power_iterations(&self) -> usize {
        self.num_power_iterations
    }

    /// Get the epsilon value
    pub fn eps(&self) -> T {
        self.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_norm_creation() {
        let spectral_norm = SpectralNorm::<f32>::new(&[128, 64]).unwrap();
        assert_eq!(spectral_norm.weight().shape().dims(), &[128, 64]);
        assert_eq!(spectral_norm.u().shape().dims(), &[128]);
        assert_eq!(spectral_norm.v().shape().dims(), &[64]);
        assert_eq!(spectral_norm.num_power_iterations(), 1);
    }

    #[test]
    fn test_spectral_norm_invalid_shape() {
        // Should fail for non-2D tensors
        let result = SpectralNorm::<f32>::new(&[128, 64, 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_norm_with_power_iterations() {
        let spectral_norm = SpectralNorm::<f32>::new(&[64, 32])
            .unwrap()
            .with_power_iterations(5);
        assert_eq!(spectral_norm.num_power_iterations(), 5);
    }

    #[test]
    fn test_spectral_norm_with_eps() {
        let eps_val = 1e-10_f32;
        let spectral_norm = SpectralNorm::<f32>::new(&[32, 16])
            .unwrap()
            .with_eps(eps_val);
        assert_eq!(spectral_norm.eps(), eps_val);
    }

    #[test]
    fn test_spectral_norm_update_weight() {
        let mut spectral_norm = SpectralNorm::<f32>::new(&[4, 3]).unwrap();
        let new_weight = Tensor::zeros(&[4, 3]);

        let result = spectral_norm.update_weight(new_weight);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectral_norm_update_weight_wrong_shape() {
        let mut spectral_norm = SpectralNorm::<f32>::new(&[4, 3]).unwrap();
        let wrong_weight = Tensor::zeros(&[5, 3]); // Wrong shape

        let result = spectral_norm.update_weight(wrong_weight);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_norm_getters() {
        let spectral_norm = SpectralNorm::<f32>::new(&[8, 6]).unwrap();

        // Test getter methods
        assert_eq!(spectral_norm.weight().shape().dims(), &[8, 6]);
        assert_eq!(spectral_norm.u().shape().dims(), &[8]);
        assert_eq!(spectral_norm.v().shape().dims(), &[6]);
        assert_eq!(spectral_norm.num_power_iterations(), 1);
        assert!(spectral_norm.eps() > 0.0);
    }
}
