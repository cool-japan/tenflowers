//! Moment-based statistical operations: percentile, range, skewness, kurtosis, moment.

use super::distribution::quantile;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, ToPrimitive};

/// Helper macro to convert numeric constants without unwrap (no unwrap policy)
macro_rules! float_const {
    ($val:expr, $t:ty) => {
        <$t as scirs2_core::num_traits::NumCast>::from($val)
            .expect("float constant conversion should never fail for standard float types")
    };
}

/// Compute percentile
///
/// Computes the percentile of the input tensor values.
///
/// # Arguments
/// * `x` - Input tensor
/// * `percentiles` - Percentile values (0-100)
/// * `axis` - Optional axis along which to compute percentiles. If None, computes over flattened array.
///
/// # Returns
/// A tensor containing the percentile values
pub fn percentile<T>(x: &Tensor<T>, percentiles: &[T], axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + PartialOrd + bytemuck::Pod + bytemuck::Zeroable,
{
    // Convert percentiles to quantiles (percentile / 100)
    let quantiles: Vec<T> = percentiles
        .iter()
        .map(|&p| p / float_const!(100.0, T))
        .collect();

    quantile(x, &quantiles, axis)
}

/// Compute the range (max - min) of tensor values
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Optional axis along which to compute range. If None, computes over flattened array.
///
/// # Returns
/// A tensor containing the range values
pub fn range<T>(x: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + PartialOrd + bytemuck::Pod + bytemuck::Zeroable,
{
    use crate::ops::reduction::{max, min};

    let axes_slice = axis.map(|a| vec![a]);
    let min_val = min(x, axes_slice.as_deref(), false)?;
    let max_val = max(x, axes_slice.as_deref(), false)?;

    // Subtract min from max
    use crate::ops::binary::sub;
    sub(&max_val, &min_val)
}

/// Compute skewness (third moment)
///
/// Computes the skewness of the input tensor values.
/// Skewness measures the asymmetry of the distribution.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Optional axis along which to compute skewness. If None, computes over flattened array.
/// * `bias` - If true, use biased estimator (N). If false, use unbiased estimator (N-1).
///
/// # Returns
/// A tensor containing the skewness values
pub fn skewness<T>(x: &Tensor<T>, axis: Option<i32>, bias: bool) -> Result<Tensor<T>>
where
    T: Float
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + ToPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if axis.is_some() {
                // For now, implement flattened version
                return skewness(&x.flatten()?, None, bias);
            }

            // Flatten the array
            let flat_data: Vec<T> = arr.iter().cloned().collect();
            let n = flat_data.len();

            if n < 3 {
                return Err(TensorError::invalid_operation_simple(
                    "Skewness requires at least 3 data points".to_string(),
                ));
            }

            // Compute mean
            let mean =
                flat_data.iter().cloned().fold(T::zero(), |acc, x| acc + x) / float_const!(n, T);

            // Compute second and third central moments
            let mut m2 = T::zero();
            let mut m3 = T::zero();

            for &value in &flat_data {
                let diff = value - mean;
                let diff_sq = diff * diff;
                let diff_cube = diff_sq * diff;

                m2 = m2 + diff_sq;
                m3 = m3 + diff_cube;
            }

            let divisor = if bias {
                float_const!(n, T)
            } else {
                float_const!(n - 1, T)
            };
            m2 = m2 / divisor;
            m3 = m3 / divisor;

            // Compute skewness: m3 / (m2^(3/2))
            let std_dev = m2.sqrt();
            if std_dev == T::zero() {
                return Ok(Tensor::from_scalar(T::zero()));
            }

            let skew = m3 / (std_dev * std_dev * std_dev);
            Ok(Tensor::from_scalar(skew))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            // For GPU implementation, fall back to CPU for now
            let cpu_tensor = x.to_cpu()?;
            skewness(&cpu_tensor, axis, bias)
        }
    }
}

/// Compute kurtosis (fourth moment)
///
/// Computes the kurtosis of the input tensor values.
/// Kurtosis measures the "tailedness" of the distribution.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Optional axis along which to compute kurtosis. If None, computes over flattened array.
/// * `bias` - If true, use biased estimator (N). If false, use unbiased estimator (N-1).
/// * `fisher` - If true, return Fisher's definition (normal = 0). If false, return Pearson's definition (normal = 3).
///
/// # Returns
/// A tensor containing the kurtosis values
pub fn kurtosis<T>(x: &Tensor<T>, axis: Option<i32>, bias: bool, fisher: bool) -> Result<Tensor<T>>
where
    T: Float
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + ToPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if axis.is_some() {
                // For now, implement flattened version
                return kurtosis(&x.flatten()?, None, bias, fisher);
            }

            // Flatten the array
            let flat_data: Vec<T> = arr.iter().cloned().collect();
            let n = flat_data.len();

            if n < 4 {
                return Err(TensorError::invalid_operation_simple(
                    "Kurtosis requires at least 4 data points".to_string(),
                ));
            }

            // Compute mean
            let mean =
                flat_data.iter().cloned().fold(T::zero(), |acc, x| acc + x) / float_const!(n, T);

            // Compute central moments
            let mut m2 = T::zero();
            let mut m4 = T::zero();

            for &value in &flat_data {
                let diff = value - mean;
                let diff_sq = diff * diff;
                let diff_fourth = diff_sq * diff_sq;

                m2 = m2 + diff_sq;
                m4 = m4 + diff_fourth;
            }

            let divisor = if bias {
                float_const!(n, T)
            } else {
                float_const!(n - 1, T)
            };
            m2 = m2 / divisor;
            m4 = m4 / divisor;

            // Compute kurtosis: m4 / (m2^2)
            if m2 == T::zero() {
                return Ok(Tensor::from_scalar(T::zero()));
            }

            let kurt = m4 / (m2 * m2);

            // Apply Fisher correction if requested (subtract 3 for normal distribution = 0)
            let result = if fisher {
                kurt - float_const!(3.0, T)
            } else {
                kurt
            };

            Ok(Tensor::from_scalar(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            // For GPU implementation, fall back to CPU for now
            let cpu_tensor = x.to_cpu()?;
            kurtosis(&cpu_tensor, axis, bias, fisher)
        }
    }
}

/// Compute central moment of specified order
///
/// Computes the n-th central moment of the input tensor values.
/// Central moment is defined as E[(X - μ)^n] where μ is the mean.
///
/// # Arguments
/// * `x` - Input tensor
/// * `order` - Order of the moment (1, 2, 3, 4, etc.)
/// * `axis` - Optional axis along which to compute moment. If None, computes over flattened array.
/// * `bias` - If true, use biased estimator (N). If false, use unbiased estimator (N-1).
///
/// # Returns
/// A tensor containing the moment values
pub fn moment<T>(x: &Tensor<T>, order: usize, axis: Option<i32>, bias: bool) -> Result<Tensor<T>>
where
    T: Float
        + Default
        + Send
        + Sync
        + 'static
        + PartialOrd
        + ToPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if axis.is_some() {
                // For now, implement flattened version
                return moment(&x.flatten()?, order, None, bias);
            }

            // Flatten the array
            let flat_data: Vec<T> = arr.iter().cloned().collect();
            let n = flat_data.len();

            if n == 0 {
                return Err(TensorError::invalid_operation_simple(
                    "Cannot compute moment of empty tensor".to_string(),
                ));
            }

            // Compute mean
            let mean =
                flat_data.iter().cloned().fold(T::zero(), |acc, x| acc + x) / float_const!(n, T);

            // Compute central moment
            let mut moment_sum = T::zero();

            for &value in &flat_data {
                let diff = value - mean;
                let mut powered_diff = T::one();

                // Compute diff^order
                for _ in 0..order {
                    powered_diff = powered_diff * diff;
                }

                moment_sum = moment_sum + powered_diff;
            }

            let divisor = if bias {
                float_const!(n, T)
            } else {
                float_const!(n - 1, T)
            };
            let moment_val = moment_sum / divisor;

            Ok(Tensor::from_scalar(moment_val))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            // For GPU implementation, fall back to CPU for now
            let cpu_tensor = x.to_cpu()?;
            moment(&cpu_tensor, order, axis, bias)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::distribution::{correlation, covariance, median, quantile};
    use super::super::histogram::histogram;
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_histogram_basic() {
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test tensor creation should succeed");
        let (counts, edges) =
            histogram(&x, 5, Some((0.0, 6.0))).expect("test: operation should succeed");

        assert_eq!(counts.shape().dims(), &[5]);
        assert_eq!(edges.shape().dims(), &[6]);

        let counts_vals = counts.as_slice().expect("tensor should be contiguous");
        let edges_vals = edges.as_slice().expect("tensor should be contiguous");

        // Each bin should have 1 count
        assert_eq!(counts_vals, &[1, 1, 1, 1, 1]);

        // Check bin edges
        assert_relative_eq!(edges_vals[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(edges_vals[5], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_basic() {
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test tensor creation should succeed");
        let q = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let result = quantile(&x, &q, None).expect("test: quantile should succeed");

        assert_eq!(result.shape().dims(), &[5]);

        let vals = result.as_slice().expect("tensor should be contiguous");
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-10); // min
        assert_relative_eq!(vals[2], 3.0, epsilon = 1e-10); // median
        assert_relative_eq!(vals[4], 5.0, epsilon = 1e-10); // max
    }

    #[test]
    fn test_median() {
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test tensor creation should succeed");
        let result = median(&x, None).expect("test: median should succeed");

        assert_eq!(result.shape().dims(), &[1]);
        assert_relative_eq!(
            result.as_slice().expect("tensor should be contiguous")[0],
            3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_covariance_basic() {
        // Simple 2x2 data matrix
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
            .expect("test tensor creation should succeed");

        let result = covariance(&x, false).expect("test: covariance should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);

        let vals = result.as_slice().expect("tensor should be contiguous");
        // For this simple case, covariance should be positive
        assert!(vals[0] > 0.0); // var(x1)
        assert!(vals[3] > 0.0); // var(x2)
        assert_relative_eq!(vals[1], vals[2], epsilon = 1e-10); // symmetry
    }

    #[test]
    fn test_correlation_basic() {
        // Perfect correlation case
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2])
            .expect("test tensor creation should succeed");

        let result = correlation(&x).expect("test: correlation should succeed");

        assert_eq!(result.shape().dims(), &[2, 2]);

        let vals = result.as_slice().expect("tensor should be contiguous");
        // Diagonal should be 1.0
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(vals[3], 1.0, epsilon = 1e-10);
        // Off-diagonal should be perfect correlation
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_percentile() {
        let x = Tensor::<f64>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
        )
        .expect("test: operation should succeed");
        let percentiles = vec![0.0, 25.0, 50.0, 75.0, 100.0];

        let result = percentile(&x, &percentiles, None).expect("test: percentile should succeed");

        assert_eq!(result.shape().dims(), &[5]);
        let vals = result.as_slice().expect("tensor should be contiguous");

        // Check approximate values
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-6); // 0th percentile (min)
        assert_relative_eq!(vals[2], 5.5, epsilon = 1e-6); // 50th percentile (median)
        assert_relative_eq!(vals[4], 10.0, epsilon = 1e-6); // 100th percentile (max)
    }

    #[test]
    fn test_range() {
        let x = Tensor::<f32>::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0], &[5])
            .expect("test tensor creation should succeed");
        let result = range(&x, None).expect("test: range should succeed");

        assert_eq!(result.shape().dims(), &[] as &[usize]);
        let val = result.as_slice().expect("tensor should be contiguous")[0];
        assert_relative_eq!(val, 7.0, epsilon = 1e-6); // 8.0 - 1.0
    }

    #[test]
    fn test_skewness() {
        // Test with symmetric data (should have skewness near 0)
        let symmetric_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let x = Tensor::<f64>::from_vec(symmetric_data, &[5])
            .expect("test tensor creation should succeed");
        let result = skewness(&x, None, false).expect("test: skewness should succeed");

        let val = result.as_slice().expect("tensor should be contiguous")[0];
        assert_relative_eq!(val, 0.0, epsilon = 1e-6);

        // Test with right-skewed data
        let skewed_data = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let x_skewed = Tensor::<f64>::from_vec(skewed_data, &[5])
            .expect("test tensor creation should succeed");
        let result_skewed =
            skewness(&x_skewed, None, false).expect("test: skewness should succeed");

        let val_skewed = result_skewed
            .as_slice()
            .expect("tensor should be contiguous")[0];
        assert!(val_skewed > 0.0); // Should be positive for right-skewed data
    }

    #[test]
    fn test_kurtosis() {
        // Test with normal-like data
        let normal_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Tensor::<f64>::from_vec(normal_data, &[5])
            .expect("test tensor creation should succeed");

        // Fisher's definition (normal = 0)
        let result_fisher = kurtosis(&x, None, false, true).expect("test: kurtosis should succeed");
        let val_fisher = result_fisher
            .as_slice()
            .expect("tensor should be contiguous")[0];

        // Pearson's definition (normal = 3)
        let result_pearson =
            kurtosis(&x, None, false, false).expect("test: kurtosis should succeed");
        let val_pearson = result_pearson
            .as_slice()
            .expect("tensor should be contiguous")[0];

        // Fisher = Pearson - 3
        assert_relative_eq!(val_fisher, val_pearson - 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_moment() {
        let x = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test tensor creation should succeed");

        // First moment should be 0 (by definition of central moment)
        let m1 = moment(&x, 1, None, false).expect("test: moment should succeed");
        let val1 = m1.as_slice().expect("tensor should be contiguous")[0];
        assert_relative_eq!(val1, 0.0, epsilon = 1e-10);

        // Second moment should be variance
        let m2 = moment(&x, 2, None, false).expect("test: moment should succeed");
        let val2 = m2.as_slice().expect("tensor should be contiguous")[0];

        // Manual variance calculation: E[(X - μ)²]
        let mean = 3.0; // (1+2+3+4+5)/5
        let var_expected = ((1.0 - mean).powi(2)
            + (2.0 - mean).powi(2)
            + (3.0 - mean).powi(2)
            + (4.0 - mean).powi(2)
            + (5.0 - mean).powi(2))
            / 4.0; // N-1 for unbiased
        assert_relative_eq!(val2, var_expected, epsilon = 1e-10);
    }

    #[test]
    fn test_edge_cases() {
        // Test empty tensor error
        let empty = Tensor::<f64>::zeros(&[0]);
        assert!(moment(&empty, 2, None, false).is_err());

        // Test insufficient data for skewness
        let too_few = Tensor::<f64>::from_vec(vec![1.0, 2.0], &[2])
            .expect("test tensor creation should succeed");
        assert!(skewness(&too_few, None, false).is_err());

        // Test insufficient data for kurtosis
        let too_few_kurt = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test tensor creation should succeed");
        assert!(kurtosis(&too_few_kurt, None, false, true).is_err());
    }
}
