//! Utility functions for activation layers

use scirs2_core::num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Create a random tensor with normal distribution for weight initialization
/// Optimized with chunked processing for better memory performance
pub(crate) fn create_random_tensor<T>(shape: &[usize], std_dev: T) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let total_elements = shape.iter().product::<usize>();
    let std_dev_f64 = std_dev.to_f64().unwrap_or(0.01);

    // Generate random data in chunks for better cache performance
    const CHUNK_SIZE: usize = 8192; // Optimize for L2 cache
    let mut random_data = Vec::with_capacity(total_elements);

    // Simple pseudo-random generation optimized for performance
    let mut seed = 12345u64;

    for chunk_start in (0..total_elements).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(total_elements);
        let chunk_size = chunk_end - chunk_start;

        let chunk_samples: Vec<T> = (0..chunk_size)
            .map(|_| {
                // Simple LCG for fast random generation
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f64) / (u64::MAX as f64);

                // Box-Muller transform for normal distribution
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u2_norm = (seed as f64) / (u64::MAX as f64);

                let sample =
                    (-2.0 * normalized.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2_norm).cos();
                let scaled_sample = sample * std_dev_f64;
                T::from_f64(scaled_sample).unwrap_or(T::zero())
            })
            .collect();

        random_data.extend(chunk_samples);
    }

    Tensor::from_vec(random_data, shape)
}
