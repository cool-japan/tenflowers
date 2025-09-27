//! SIMD Optimized Binary Operations
//!
//! This module provides SIMD-accelerated implementations for binary operations
//! using platform-specific intrinsics (AVX2/AVX512 on x86_64, NEON on ARM64).

use crate::{Result, TensorError};

/// Specialized SIMD implementations for f32 operations
pub mod simd_f32_ops {
    use super::*;

    /// Ultra-fast f32 addition with AVX2/NEON support
    pub fn simd_add_f32(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "SIMD slice length mismatch".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return simd_add_f32_avx2(a, b, output);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return simd_add_f32_neon(a, b, output);
        }

        // Fallback to vectorized implementation
        #[allow(unreachable_code)]
        for i in 0..a.len() {
            output[i] = a[i] + b[i];
        }
        Ok(())
    }

    /// Ultra-fast f32 multiplication with AVX2/NEON support
    pub fn simd_mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "SIMD slice length mismatch".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return simd_mul_f32_avx2(a, b, output);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return simd_mul_f32_neon(a, b, output);
        }

        // Fallback to vectorized implementation
        #[allow(unreachable_code)]
        for i in 0..a.len() {
            output[i] = a[i] * b[i];
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_add_f32_avx2(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        // AVX2 implementation with 8 floats per vector
        use std::arch::x86_64::*;

        let len = a.len();
        let simd_end = len & !7; // Process 8 elements at a time

        unsafe {
            for i in (0..simd_end).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), vr);
            }
        }

        // Handle remaining elements
        for i in simd_end..len {
            output[i] = a[i] + b[i];
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_mul_f32_avx2(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        // AVX2 implementation with 8 floats per vector
        use std::arch::x86_64::*;

        let len = a.len();
        let simd_end = len & !7; // Process 8 elements at a time

        unsafe {
            for i in (0..simd_end).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), vr);
            }
        }

        // Handle remaining elements
        for i in simd_end..len {
            output[i] = a[i] * b[i];
        }
        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_add_f32_neon(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        // NEON implementation with 4 floats per vector
        use std::arch::aarch64::*;

        let len = a.len();
        let simd_end = len & !3; // Process 4 elements at a time

        unsafe {
            for i in (0..simd_end).step_by(4) {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vr = vaddq_f32(va, vb);
                vst1q_f32(output.as_mut_ptr().add(i), vr);
            }
        }

        // Handle remaining elements
        for i in simd_end..len {
            output[i] = a[i] + b[i];
        }
        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_mul_f32_neon(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        // NEON implementation with 4 floats per vector
        use std::arch::aarch64::*;

        let len = a.len();
        let simd_end = len & !3; // Process 4 elements at a time

        unsafe {
            for i in (0..simd_end).step_by(4) {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vr = vmulq_f32(va, vb);
                vst1q_f32(output.as_mut_ptr().add(i), vr);
            }
        }

        // Handle remaining elements
        for i in simd_end..len {
            output[i] = a[i] * b[i];
        }
        Ok(())
    }
}
