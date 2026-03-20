//! Fused Binary Operations
//!
//! This module provides fused operations that combine multiple operations
//! for maximum performance by reducing memory bandwidth and improving cache efficiency.

use super::core::{BinaryOp, OpComplexity};
use crate::{Result, TensorError};

/// Fused operations for maximum performance
pub mod fused_ops {
    use super::*;

    /// Fused Multiply-Add operation: (a * b) + c
    #[derive(Clone)]
    pub struct FusedMulAddOp {
        pub c: f32, // Constant to add
    }

    impl BinaryOp<f32> for FusedMulAddOp {
        #[inline]
        fn apply(&self, a: f32, b: f32) -> f32 {
            (a * b) + self.c
        }

        fn name(&self) -> &str {
            "FusedMulAdd"
        }

        fn apply_slice(&self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
            if a.len() != b.len() || a.len() != output.len() {
                return Err(TensorError::invalid_argument(
                    "Fused MulAdd slice length mismatch".to_string(),
                ));
            }

            // Use FMA instructions when available
            for i in 0..a.len() {
                output[i] = a[i].mul_add(b[i], self.c);
            }
            Ok(())
        }

        fn supports_simd(&self) -> bool {
            true
        }
        fn supports_gpu(&self) -> bool {
            true
        }
        fn complexity(&self) -> OpComplexity {
            OpComplexity::Simple
        }
        fn is_associative(&self) -> bool {
            false
        }
        fn is_commutative(&self) -> bool {
            true
        } // For the multiply part
    }

    /// Fused Add-Multiply operation: (a + b) * c
    #[derive(Clone)]
    pub struct FusedAddMulOp {
        pub c: f32, // Constant to multiply
    }

    impl BinaryOp<f32> for FusedAddMulOp {
        #[inline]
        fn apply(&self, a: f32, b: f32) -> f32 {
            (a + b) * self.c
        }

        fn name(&self) -> &str {
            "FusedAddMul"
        }

        fn apply_slice(&self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
            if a.len() != b.len() || a.len() != output.len() {
                return Err(TensorError::invalid_argument(
                    "Fused AddMul slice length mismatch".to_string(),
                ));
            }

            // Optimized implementation
            for i in 0..a.len() {
                output[i] = (a[i] + b[i]) * self.c;
            }
            Ok(())
        }

        fn supports_simd(&self) -> bool {
            true
        }
        fn supports_gpu(&self) -> bool {
            true
        }
        fn complexity(&self) -> OpComplexity {
            OpComplexity::Simple
        }
        fn is_associative(&self) -> bool {
            false
        }
        fn is_commutative(&self) -> bool {
            true
        } // For the addition part
    }
}

/// Ultra-performance memory prefetching for large tensors
#[allow(dead_code)]
pub fn prefetch_memory<T>(data: &[T], stride: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe {
            for i in (0..data.len()).step_by(stride) {
                let ptr = data.as_ptr().add(i) as *const i8;
                _mm_prefetch(ptr, _MM_HINT_T0); // Prefetch to L1 cache
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            for i in (0..data.len()).step_by(stride) {
                let ptr = data.as_ptr().add(i);
                // AARCH64 prefetch - use inline assembly
                core::arch::asm!("prfm pldl1strm, [{0}]", in(reg) ptr);
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No prefetching for other architectures
        let _ = data;
        let _ = stride;
    }
}
