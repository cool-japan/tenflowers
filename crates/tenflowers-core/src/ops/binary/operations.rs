//! Concrete Binary Operation Implementations
//!
//! This module provides concrete implementations of all binary operations with
//! ultra-performance optimizations including SIMD acceleration and parallel processing.

use super::core::{get_binary_op_registry, BinaryOp, OpComplexity};
use super::simd::simd_f32_ops;
use crate::{Result, TensorError};
use rayon::prelude::*;
use scirs2_core::numeric::Zero;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

/// Ultra-performance addition operation with SIMD and parallel support
#[derive(Clone)]
pub struct AddOp;

impl<T: StdAdd<Output = T> + Clone + Send + Sync + 'static> BinaryOp<T> for AddOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        a + b
    }

    #[inline]
    fn name(&self) -> &str {
        "Add"
    }

    fn apply_slice(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "Slice length mismatch for Add operation".to_string(),
            ));
        }

        // Special case for f32 - use SIMD optimization
        if std::any::type_name::<T>() == "f32" {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let output_f32 = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
            };
            return simd_f32_ops::simd_add_f32(a_f32, b_f32, output_f32).map_err(|_| {
                TensorError::invalid_argument("SIMD Add operation failed".to_string())
            });
        }

        // Use parallel processing for large arrays
        if a.len() >= 10000 {
            get_binary_op_registry().record_parallel_usage();
            output
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(out, (a_val, b_val))| {
                    *out = a_val.clone() + b_val.clone();
                });
        } else {
            // Sequential for small arrays
            for i in 0..a.len() {
                output[i] = a[i].clone() + b[i].clone();
            }
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
        true
    }
    fn is_commutative(&self) -> bool {
        true
    }
}

/// Ultra-performance subtraction operation with SIMD and parallel support
#[derive(Clone)]
pub struct SubOp;

impl<T: StdSub<Output = T> + Clone + Send + Sync + 'static> BinaryOp<T> for SubOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        a - b
    }

    #[inline]
    fn name(&self) -> &str {
        "Sub"
    }

    fn apply_slice(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "Slice length mismatch for Sub operation".to_string(),
            ));
        }

        // Use parallel processing for large arrays
        if a.len() >= 10000 {
            get_binary_op_registry().record_parallel_usage();
            output
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(out, (a_val, b_val))| {
                    *out = a_val.clone() - b_val.clone();
                });
        } else {
            // Sequential for small arrays
            for i in 0..a.len() {
                output[i] = a[i].clone() - b[i].clone();
            }
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
    } // Subtraction is not associative
    fn is_commutative(&self) -> bool {
        false
    } // Subtraction is not commutative
}

/// Ultra-performance multiplication operation with SIMD and parallel support
#[derive(Clone)]
pub struct MulOp;

impl<T: StdMul<Output = T> + Clone + Send + Sync + 'static> BinaryOp<T> for MulOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        a * b
    }
    #[inline]
    fn name(&self) -> &str {
        "Mul"
    }

    fn apply_slice(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "Slice length mismatch for Mul operation".to_string(),
            ));
        }

        // Special case for f32 - use SIMD optimization
        if std::any::type_name::<T>() == "f32" {
            let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
            let output_f32 = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
            };
            return simd_f32_ops::simd_mul_f32(a_f32, b_f32, output_f32).map_err(|_| {
                TensorError::invalid_argument("SIMD Mul operation failed".to_string())
            });
        }

        // Use parallel processing for large arrays
        if a.len() >= 8000 {
            get_binary_op_registry().record_parallel_usage();
            output
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(out, (a_val, b_val))| {
                    *out = a_val.clone() * b_val.clone();
                });
        } else {
            // Sequential for small arrays
            for i in 0..a.len() {
                output[i] = a[i].clone() * b[i].clone();
            }
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
        true
    }
    fn is_commutative(&self) -> bool {
        true
    }
}

/// Ultra-performance division operation with SIMD and parallel support
#[derive(Clone)]
pub struct DivOp;

impl<T: StdDiv<Output = T> + Clone + Send + Sync + 'static> BinaryOp<T> for DivOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        a / b
    }

    #[inline]
    fn name(&self) -> &str {
        "Div"
    }

    fn apply_slice(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "Slice length mismatch for Div operation".to_string(),
            ));
        }

        // Use parallel processing for large arrays (division is more expensive)
        if a.len() >= 5000 {
            get_binary_op_registry().record_parallel_usage();
            output
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(out, (a_val, b_val))| {
                    *out = a_val.clone() / b_val.clone();
                });
        } else {
            // Sequential for small arrays
            for i in 0..a.len() {
                output[i] = a[i].clone() / b[i].clone();
            }
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
        OpComplexity::Moderate
    } // Division is more expensive
    fn is_associative(&self) -> bool {
        false
    } // Division is not associative
    fn is_commutative(&self) -> bool {
        false
    } // Division is not commutative
}

/// Power operation
#[derive(Clone)]
pub struct PowOp;
impl<T: scirs2_core::num_traits::Float> BinaryOp<T> for PowOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        a.powf(b)
    }
    #[inline]
    fn name(&self) -> &str {
        "Pow"
    }

    fn complexity(&self) -> OpComplexity {
        OpComplexity::Complex
    }
    fn supports_gpu(&self) -> bool {
        true
    }
}

/// Element-wise minimum operation
#[derive(Clone)]
pub struct MinOp;
impl<T: PartialOrd + Clone> BinaryOp<T> for MinOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        if a <= b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn name(&self) -> &str {
        "Min"
    }

    fn complexity(&self) -> OpComplexity {
        OpComplexity::Moderate
    }
    fn supports_simd(&self) -> bool {
        true
    }
    fn supports_gpu(&self) -> bool {
        true
    }
    fn is_associative(&self) -> bool {
        true
    }
    fn is_commutative(&self) -> bool {
        true
    }
}

/// Element-wise maximum operation
#[derive(Clone)]
pub struct MaxOp;
impl<T: PartialOrd + Clone> BinaryOp<T> for MaxOp {
    #[inline]
    fn apply(&self, a: T, b: T) -> T {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn name(&self) -> &str {
        "Max"
    }

    fn complexity(&self) -> OpComplexity {
        OpComplexity::Moderate
    }
    fn supports_simd(&self) -> bool {
        true
    }
    fn supports_gpu(&self) -> bool {
        true
    }
    fn is_associative(&self) -> bool {
        true
    }
    fn is_commutative(&self) -> bool {
        true
    }
}

/// PReLU operation: PReLU(x, alpha) = x if x > 0, else alpha * x
#[derive(Clone)]
pub struct PReLUOp;
impl<T> BinaryOp<T> for PReLUOp
where
    T: scirs2_core::num_traits::Float + PartialOrd + Zero + StdMul<Output = T>,
{
    #[inline]
    fn apply(&self, x: T, alpha: T) -> T {
        if x > T::zero() {
            x
        } else {
            alpha * x
        }
    }
    #[inline]
    fn name(&self) -> &str {
        "PReLU"
    }

    fn complexity(&self) -> OpComplexity {
        OpComplexity::Moderate
    }
    fn supports_gpu(&self) -> bool {
        true
    }
}
