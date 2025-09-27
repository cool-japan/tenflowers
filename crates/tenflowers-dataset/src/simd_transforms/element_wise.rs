//! SIMD-accelerated element-wise operations
//!
//! This module provides vectorized implementations of element-wise operations
//! using SIMD instructions for enhanced performance.

#![allow(unsafe_code)]

use crate::Transform;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Types of SIMD operations supported
#[derive(Debug, Clone, Copy)]
pub enum SimdOperation {
    Add,
    Multiply,
    Subtract,
    Divide,
}

/// SIMD-accelerated element-wise operations
pub struct SimdElementWise<T> {
    operation: SimdOperation,
    value: T,
    _phantom: PhantomData<T>,
}

impl<T> SimdElementWise<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new SIMD element-wise transform
    pub fn new(operation: SimdOperation, value: T) -> Self {
        Self {
            operation,
            value,
            _phantom: PhantomData,
        }
    }

    /// SIMD-accelerated element-wise operation for f32 data
    #[cfg(target_arch = "x86_64")]
    unsafe fn apply_f32_simd(&self, data: &mut [f32], value: f32) {
        if data.len() < 8 {
            self.apply_scalar_f32(data, value);
            return;
        }

        let value_vec = _mm256_set1_ps(value);
        let chunks = data.len() / 8;
        let remainder = data.len() % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));

            let result = match self.operation {
                SimdOperation::Add => _mm256_add_ps(values, value_vec),
                SimdOperation::Multiply => _mm256_mul_ps(values, value_vec),
                SimdOperation::Subtract => _mm256_sub_ps(values, value_vec),
                SimdOperation::Divide => _mm256_div_ps(values, value_vec),
            };

            _mm256_storeu_ps(data.as_mut_ptr().add(offset), result);
        }

        if remainder > 0 {
            let start = chunks * 8;
            self.apply_scalar_f32(&mut data[start..], value);
        }
    }

    /// Scalar fallback for element-wise operations
    fn apply_scalar(&self, data: &mut [T], value: T)
    where
        T: num_traits::Float,
    {
        for element in data.iter_mut() {
            *element = match self.operation {
                SimdOperation::Add => *element + value,
                SimdOperation::Multiply => *element * value,
                SimdOperation::Subtract => *element - value,
                SimdOperation::Divide => *element / value,
            };
        }
    }

    /// Scalar fallback for f32 element-wise operations
    #[allow(dead_code)]
    fn apply_scalar_f32(&self, data: &mut [f32], value: f32) {
        for element in data.iter_mut() {
            *element = match self.operation {
                SimdOperation::Add => *element + value,
                SimdOperation::Multiply => *element * value,
                SimdOperation::Subtract => *element - value,
                SimdOperation::Divide => *element / value,
            };
        }
    }
}

impl<T> Transform<T> for SimdElementWise<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if let Some(data) = features.as_slice() {
            let mut mutable_data = data.to_vec();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && std::mem::size_of::<T>() == 4 {
                    let value_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&self.value) };
                    let data_f32 = unsafe {
                        std::slice::from_raw_parts_mut(
                            mutable_data.as_mut_ptr() as *mut f32,
                            mutable_data.len(),
                        )
                    };

                    unsafe {
                        self.apply_f32_simd(data_f32, value_f32);
                    }
                } else {
                    self.apply_scalar(&mut mutable_data, self.value);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.apply_scalar(&mut mutable_data, self.value);
            }

            let new_features = Tensor::from_vec(mutable_data, features.shape().dims())?;
            Ok((new_features, labels))
        } else {
            Err(TensorError::invalid_argument(
                "Cannot access tensor data for element-wise operation".to_string(),
            ))
        }
    }
}
