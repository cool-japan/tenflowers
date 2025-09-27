//! Tensor Module - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by operation type:
//!
//! ## Module Organization
//!
//! - **core**: Core tensor structure, properties, and basic utilities
//! - **creation**: Tensor constructors and creation methods
//! - **device**: Device management and transfer operations
//! - **ops**: Mathematical operations, activations, and shape manipulations
//! - **comparison**: Comparison and logical operations
//! - **indexing**: Direct indexing capabilities using Rust traits
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

// Import the modularized tensor functionality
pub mod comparison;
pub mod core;
pub mod creation;
pub mod device;
pub mod indexing;
pub mod ops;

// Re-export all tensor types and functionality for backward compatibility

// Core tensor structure and storage
pub use core::{Tensor, TensorStorage};

// Note: The specific method implementations are included as part of the impl blocks
// in their respective modules, so they are automatically available when using Tensor<T>

// Tests are preserved from the original tensor.rs file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_constructors() {
        let zeros = Tensor::<f32>::zeros(&[2, 3]);
        assert_eq!(zeros.shape().dims(), &[2, 3]);
        assert_eq!(zeros.size(), 6);

        let ones = Tensor::<f32>::ones(&[2, 2]);
        assert_eq!(ones.shape().dims(), &[2, 2]);
        if let Some(data) = ones.as_slice() {
            assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
        }

        let full = Tensor::<f32>::full(&[3], 5.0);
        if let Some(data) = full.as_slice() {
            assert_eq!(data, &[5.0, 5.0, 5.0]);
        }
    }

    #[test]
    fn test_eye_tensor() {
        let eye = Tensor::<f32>::eye(3);
        assert_eq!(eye.shape().dims(), &[3, 3]);
        if let Some(data) = eye.as_slice() {
            assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn test_arange() {
        let arange = Tensor::<f32>::arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(arange.shape().dims(), &[5]);
        if let Some(data) = arange.as_slice() {
            assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 4.0]);
        }

        let arange_step = Tensor::<f32>::arange(0.0, 3.0, 0.5).unwrap();
        assert_eq!(arange_step.shape().dims(), &[6]);
        if let Some(data) = arange_step.as_slice() {
            assert_eq!(data, &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        }
    }

    #[test]
    fn test_linspace() {
        let linspace = Tensor::<f32>::linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(linspace.shape().dims(), &[5]);
        if let Some(data) = linspace.as_slice() {
            assert_eq!(data, &[0.0, 0.25, 0.5, 0.75, 1.0]);
        }

        let single_step = Tensor::<f32>::linspace(5.0, 10.0, 1).unwrap();
        assert_eq!(single_step.shape().dims(), &[1]);
        if let Some(data) = single_step.as_slice() {
            assert_eq!(data, &[5.0]);
        }
    }

    #[test]
    fn test_tensor_properties() {
        let tensor = Tensor::<f32>::zeros(&[2, 3, 4]);
        assert_eq!(tensor.size(), 24);
        assert_eq!(tensor.numel(), 24);
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.ndim(), 3);
        assert!(!tensor.is_scalar());
        assert!(!tensor.is_vector());
        assert!(!tensor.is_matrix());

        let scalar = Tensor::<f32>::from_scalar(5.0);
        assert!(scalar.is_scalar());
        assert_eq!(scalar.size(), 1);

        let vector = Tensor::<f32>::ones(&[5]);
        assert!(vector.is_vector());

        let matrix = Tensor::<f32>::ones(&[3, 4]);
        assert!(matrix.is_matrix());
    }

    #[test]
    fn test_tensor_math_operations() {
        let tensor = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let abs_tensor = tensor.abs().unwrap();
        if let Some(data) = abs_tensor.as_slice() {
            assert_eq!(data, &[2.0, 1.0, 0.0, 1.0, 2.0]);
        }

        let neg_tensor = tensor.neg().unwrap();
        if let Some(data) = neg_tensor.as_slice() {
            assert_eq!(data, &[2.0, 1.0, 0.0, -1.0, -2.0]);
        }

        let exp_tensor = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2])
            .unwrap()
            .exp()
            .unwrap();
        if let Some(data) = exp_tensor.as_slice() {
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - std::f32::consts::E).abs() < 1e-6);
        }
    }

    #[test]
    fn test_trig_functions() {
        use std::f32::consts::PI;

        let tensor = Tensor::<f32>::from_vec(vec![0.0, PI / 2.0, PI], &[3]).unwrap();

        let sin_tensor = tensor.sin().unwrap();
        if let Some(data) = sin_tensor.as_slice() {
            assert!((data[0] - 0.0).abs() < 1e-6);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!(data[2].abs() < 1e-6);
        }

        let cos_tensor = tensor.cos().unwrap();
        if let Some(data) = cos_tensor.as_slice() {
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!(data[1].abs() < 1e-6);
            assert!((data[2] + 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_error_conditions() {
        assert!(Tensor::<f32>::arange(0.0, 1.0, 0.0).is_err());
        assert!(Tensor::<f32>::linspace(0.0, 1.0, 0).is_err());
    }

    #[test]
    fn test_comparison_operations() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[4]).unwrap();

        let eq_result = a.eq(&b).unwrap();
        if let Some(data) = eq_result.as_slice() {
            assert_eq!(data, &[false, true, false, false]);
        }

        let ne_result = a.ne(&b).unwrap();
        if let Some(data) = ne_result.as_slice() {
            assert_eq!(data, &[true, false, true, true]);
        }

        let gt_result = a.gt(&b).unwrap();
        if let Some(data) = gt_result.as_slice() {
            assert_eq!(data, &[false, false, true, true]);
        }

        let ge_result = a.ge(&b).unwrap();
        if let Some(data) = ge_result.as_slice() {
            assert_eq!(data, &[false, true, true, true]);
        }

        let lt_result = a.lt(&b).unwrap();
        if let Some(data) = lt_result.as_slice() {
            assert_eq!(data, &[true, false, false, false]);
        }

        let le_result = a.le(&b).unwrap();
        if let Some(data) = le_result.as_slice() {
            assert_eq!(data, &[true, true, false, false]);
        }
    }

    #[test]
    fn test_comparison_broadcasting() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![2.0, 2.0], &[1, 2]).unwrap();

        let gt_result = a.gt(&b).unwrap();
        assert_eq!(gt_result.shape().dims(), &[3, 2]);

        if let Some(data) = gt_result.as_slice() {
            assert_eq!(data, &[false, false, false, false, true, true]);
        }
    }

    #[test]
    fn test_logical_operations() {
        let a = Tensor::<bool>::from_vec(vec![true, false, true, false], &[4]).unwrap();
        let b = Tensor::<bool>::from_vec(vec![true, true, false, false], &[4]).unwrap();

        let and_result = a.logical_and(&b).unwrap();
        if let Some(data) = and_result.as_slice() {
            assert_eq!(data, &[true, false, false, false]);
        }

        let or_result = a.logical_or(&b).unwrap();
        if let Some(data) = or_result.as_slice() {
            assert_eq!(data, &[true, true, true, false]);
        }

        let not_result = a.logical_not().unwrap();
        if let Some(data) = not_result.as_slice() {
            assert_eq!(data, &[false, true, false, true]);
        }

        let xor_result = a.logical_xor(&b).unwrap();
        if let Some(data) = xor_result.as_slice() {
            assert_eq!(data, &[false, true, true, false]);
        }
    }

    #[test]
    fn test_logical_broadcasting() {
        let a = Tensor::<bool>::from_vec(vec![true, false], &[2, 1]).unwrap();
        let b = Tensor::<bool>::from_vec(vec![true, false], &[1, 2]).unwrap();

        let and_result = a.logical_and(&b).unwrap();
        assert_eq!(and_result.shape().dims(), &[2, 2]);

        if let Some(data) = and_result.as_slice() {
            assert_eq!(data, &[true, false, false, false]);
        }
    }

    #[test]
    fn test_all_any_operations() {
        let tensor =
            Tensor::<bool>::from_vec(vec![true, false, true, true, false, true], &[2, 3]).unwrap();

        let all_result = tensor.all(None, false).unwrap();
        if let Some(data) = all_result.as_slice() {
            assert_eq!(data, &[false]);
        }

        let any_result = tensor.any(None, false).unwrap();
        if let Some(data) = any_result.as_slice() {
            assert_eq!(data, &[true]);
        }

        let all_axis0 = tensor.all(Some(&[0]), false).unwrap();
        assert_eq!(all_axis0.shape().dims(), &[3]);
        if let Some(data) = all_axis0.as_slice() {
            assert_eq!(data, &[true, false, true]);
        }

        let any_axis1 = tensor.any(Some(&[1]), false).unwrap();
        assert_eq!(any_axis1.shape().dims(), &[2]);
        if let Some(data) = any_axis1.as_slice() {
            assert_eq!(data, &[true, true]);
        }
    }

    #[test]
    fn test_clamp() {
        let tensor = Tensor::<f32>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[6]).unwrap();
        let clamped = tensor.clamp(-1.0, 2.0).unwrap();

        if let Some(data) = clamped.as_slice() {
            assert_eq!(data, &[-1.0, -1.0, 0.0, 1.0, 2.0, 2.0]);
        }
    }

    #[test]
    fn test_allclose() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![1.001, 2.001, 3.001], &[3]).unwrap();
        let c = Tensor::<f32>::from_vec(vec![1.1, 2.1, 3.1], &[3]).unwrap();

        // Should be close with appropriate tolerances
        assert!(a.allclose(&b, 1e-2, 1e-2).unwrap());

        // Should not be close with tight tolerances
        assert!(!a.allclose(&c, 1e-3, 1e-3).unwrap());

        // Different shapes should return false
        let d = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert!(!a.allclose(&d, 1e-2, 1e-2).unwrap());
    }

    #[test]
    fn test_fill_() {
        let mut tensor = Tensor::<f32>::zeros(&[2, 3]);
        tensor.fill_(5.0).unwrap();

        if let Some(data) = tensor.as_slice() {
            assert_eq!(data, &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        }
    }

    #[test]
    fn test_index_trait() {
        // Test 1D indexing
        let tensor_1d = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        assert_eq!(tensor_1d[0], 1.0);
        assert_eq!(tensor_1d[2], 3.0);

        // Test 2D indexing
        let tensor_2d = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor_2d[&[0usize, 0usize][..]], 1.0);
        assert_eq!(tensor_2d[&[0usize, 1usize][..]], 2.0);
        assert_eq!(tensor_2d[&[1usize, 0usize][..]], 3.0);
        assert_eq!(tensor_2d[&[1usize, 1usize][..]], 4.0);
    }

    #[test]
    fn test_tensor_utilities() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Test utility methods
        assert!(!tensor.is_empty());
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_scalar());
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.memory_usage(), 16); // 4 elements * 4 bytes each for f32

        // Test scalar tensor
        let scalar = Tensor::<f32>::from_scalar(42.0);
        assert!(scalar.is_scalar());
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(scalar.numel(), 1);

        // Test empty tensor
        let empty = Tensor::<f32>::zeros(&[0]);
        assert!(empty.is_empty());
        assert_eq!(empty.numel(), 0);
    }

    #[test]
    fn test_shape_operations() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = Tensor::<f32>::from_vec(vec![9.0, 10.0, 11.0], &[3]).unwrap();

        // Test same_shape
        assert!(a.same_shape(&b));
        assert!(!a.same_shape(&c));

        // Test broadcasting compatibility
        let d = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let e = Tensor::<f32>::from_vec(vec![3.0, 4.0, 5.0], &[1, 3]).unwrap();
        assert!(d.is_broadcastable_with(&e)); // [2,1] can broadcast with [1,3] -> [2,3]

        let f = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let g = Tensor::<f32>::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        assert!(!f.is_broadcastable_with(&g)); // [2] cannot broadcast with [3]
    }

    #[test]
    fn test_tensor_summary() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let summary = tensor.summary();

        assert!(summary.contains("Tensor<f32>"));
        assert!(summary.contains("shape=[2, 2]"));
        assert!(summary.contains("device=Cpu"));
        assert!(summary.contains("numel=4"));
        assert!(summary.contains("memory=16B"));
        assert!(summary.contains("requires_grad=false"));
    }
}
