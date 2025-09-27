//! FFT utility functions
//!
//! This module provides utility functions for FFT operations including
//! twiddle factor generation, bit reversal tables, and helper functions.

use num_traits::{Float, FromPrimitive};

/// Generate twiddle factors for FFT computation
pub fn generate_twiddle_factors<T>(n: usize) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    let mut twiddle_factors = Vec::with_capacity(n);
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    for k in 0..n {
        let angle = two_pi * T::from(k).unwrap() / T::from(n).unwrap();
        twiddle_factors.push(angle.cos());
        twiddle_factors.push(-angle.sin());
    }

    twiddle_factors
}

/// Generate bit reversal table for FFT reordering
#[cfg(feature = "gpu")]
pub fn generate_bit_reversal_table(n: usize) -> Vec<u32> {
    let mut table = Vec::with_capacity(n);
    let log2_n = n.trailing_zeros();

    for i in 0..n {
        let mut bit_reversed = 0u32;
        let mut temp = i as u32;

        for _ in 0..log2_n {
            bit_reversed = (bit_reversed << 1) | (temp & 1);
            temp >>= 1;
        }

        table.push(bit_reversed);
    }

    table
}

// TODO: Move remaining utility functions from original fft.rs (lines 2914-4388, ~1474 lines)
// This includes:
// - Additional helper functions for GPU FFT
// - Performance optimization utilities
// - Memory management helpers
// - Test functions and benchmarking utilities

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_generate_twiddle_factors() {
        let factors: Vec<f32> = generate_twiddle_factors(4);

        // For N=4, twiddle factors should be cos and sin values for angles 0, π/2, π, 3π/2
        // Returns 2*N values (interleaved real and imaginary parts)
        assert_eq!(factors.len(), 8);

        // k=0: cos(0), -sin(0) = 1.0, 0.0
        assert_abs_diff_eq!(factors[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(factors[1], 0.0, epsilon = 1e-6);

        // k=1: cos(π/2), -sin(π/2) = 0.0, -1.0
        assert_abs_diff_eq!(factors[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(factors[3], -1.0, epsilon = 1e-6);

        // k=2: cos(π), -sin(π) = -1.0, 0.0
        assert_abs_diff_eq!(factors[4], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(factors[5], 0.0, epsilon = 1e-6);

        // k=3: cos(3π/2), -sin(3π/2) = 0.0, 1.0
        assert_abs_diff_eq!(factors[6], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(factors[7], 1.0, epsilon = 1e-6);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_generate_bit_reversal_table() {
        let table = generate_bit_reversal_table(8);

        assert_eq!(table.len(), 8);
        // For N=8, bit reversal should map:
        // 0 (000) -> 0 (000)
        // 1 (001) -> 4 (100)
        // 2 (010) -> 2 (010)
        // 3 (011) -> 6 (110)
        // etc.
        assert_eq!(table[0], 0);
        assert_eq!(table[1], 4);
        assert_eq!(table[2], 2);
        assert_eq!(table[3], 6);
    }

    // TODO: Move complete test suite from original fft.rs
    // This includes comprehensive FFT correctness tests for 1D, 2D, and 3D transforms
}
