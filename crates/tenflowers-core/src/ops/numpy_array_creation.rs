//! NumPy-style array creation functions
//!
//! This module provides NumPy-compatible array creation functions with the same
//! APIs and behaviors as NumPy.

use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use std::ops::Range;

/// Create an array of zeros with NumPy-compatible API
pub fn zeros<T>(shape: &[usize]) -> Tensor<T>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    Tensor::zeros(shape)
}

/// Create an array of ones with NumPy-compatible API
pub fn ones<T>(shape: &[usize]) -> Tensor<T>
where
    T: Clone + Default + One + Send + Sync + 'static,
{
    Tensor::ones(shape)
}

/// Create an array filled with a specific value
pub fn full<T>(shape: &[usize], fill_value: T) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let size = shape.iter().product();
    let data = vec![fill_value; size];
    Tensor::from_vec(data, shape)
}

/// Create an array with evenly spaced values within a given interval
pub fn arange<T>(start: T, stop: T, step: T) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + PartialOrd + std::ops::Add<Output = T>,
{
    // Pre-calculate capacity to avoid reallocations
    let estimated_size = {
        let mut count = 0;
        let mut current = start.clone();
        while current < stop {
            count += 1;
            current = current + step.clone();
            // Safety break to avoid infinite loops
            if count > 1_000_000 {
                break;
            }
        }
        count.max(1)
    };

    let mut values = Vec::with_capacity(estimated_size);
    let mut current = start.clone();

    while current < stop {
        values.push(current.clone());
        current = current + step.clone();
    }

    if values.is_empty() {
        values.push(start);
    }

    let len = values.len();
    Tensor::from_vec(values, &[len])
}

/// Create an array with evenly spaced values over a specified interval
pub fn linspace<T>(start: T, stop: T, num: usize, endpoint: bool) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + FromPrimitive,
{
    if num == 0 {
        return Tensor::from_vec(Vec::new(), &[0]);
    }

    if num == 1 {
        return Tensor::from_vec(vec![start], &[1]);
    }

    let step = if endpoint {
        (stop - start) / T::from_usize(num - 1).unwrap()
    } else {
        (stop - start) / T::from_usize(num).unwrap()
    };

    let mut values = Vec::with_capacity(num);
    for i in 0..num {
        let value = start + step * T::from_usize(i).unwrap();
        values.push(value);
    }

    Tensor::from_vec(values, &[num])
}

/// Create an array with values spaced evenly on a log scale
pub fn logspace<T>(start: T, stop: T, num: usize, base: T, endpoint: bool) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + FromPrimitive,
{
    let linear = linspace(start, stop, num, endpoint)?;
    let linear_data = linear.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access linear data".to_string())
    })?;

    let log_data: Vec<T> = linear_data.iter().map(|&x| base.powf(x)).collect();

    Tensor::from_vec(log_data, &[num])
}

/// Create an array with values spaced evenly on a log scale (base 10)
pub fn geomspace<T>(start: T, stop: T, num: usize, endpoint: bool) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + Float + FromPrimitive,
{
    if start <= T::zero() || stop <= T::zero() {
        return Err(TensorError::invalid_argument(
            "geomspace requires positive start and stop values".to_string(),
        ));
    }

    let log_start = start.ln();
    let log_stop = stop.ln();

    let log_linear = linspace(log_start, log_stop, num, endpoint)?;
    let log_data = log_linear.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access log linear data".to_string())
    })?;

    let geom_data: Vec<T> = log_data.iter().map(|&x| x.exp()).collect();

    Tensor::from_vec(geom_data, &[num])
}

/// Create a 2D identity matrix
pub fn eye<T>(n: usize, m: Option<usize>, k: i32) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    let m = m.unwrap_or(n);
    let mut data = vec![T::zero(); n * m];

    // Fill diagonal with ones
    for i in 0..n {
        let j = (i as i32 + k) as usize;
        if j < m {
            data[i * m + j] = T::one();
        }
    }

    Tensor::from_vec(data, &[n, m])
}

/// Create a 2D identity matrix (simplified version)
pub fn identity<T>(n: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    eye(n, None, 0)
}

/// Create an array from a range (Python-style)
pub fn from_range<T>(range: Range<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + PartialOrd + std::ops::Add<Output = T> + One,
{
    // Pre-calculate capacity to avoid reallocations
    let estimated_size = {
        let mut count = 0;
        let mut current = range.start.clone();
        let step = T::one();
        while current < range.end {
            count += 1;
            current = current + step.clone();
            // Safety break to avoid infinite loops
            if count > 1_000_000 {
                break;
            }
        }
        count.max(1)
    };

    let mut values = Vec::with_capacity(estimated_size);
    let mut current = range.start;
    let step = T::one();

    while current < range.end {
        values.push(current.clone());
        current = current + step.clone();
    }

    let len = values.len();
    Tensor::from_vec(values, &[len])
}

/// Create a diagonal matrix from a 1D array
pub fn diag<T>(v: &Tensor<T>, k: i32) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    if v.shape().rank() != 1 {
        return Err(TensorError::invalid_argument(
            "diag requires a 1D input tensor".to_string(),
        ));
    }

    let n = v.shape().dims()[0];
    let size = n + k.unsigned_abs() as usize;
    let mut data = vec![T::zero(); size * size];

    let v_data = v.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access input data".to_string())
    })?;

    for (i, value) in v_data.iter().enumerate() {
        let row = if k >= 0 { i } else { i + (-k) as usize };
        let col = if k >= 0 { i + k as usize } else { i };

        if row < size && col < size {
            data[row * size + col] = value.clone();
        }
    }

    Tensor::from_vec(data, &[size, size])
}

/// Extract diagonal from a 2D array
pub fn diagonal<T>(matrix: &Tensor<T>, offset: i32) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if matrix.shape().rank() != 2 {
        return Err(TensorError::invalid_argument(
            "diagonal requires a 2D input tensor".to_string(),
        ));
    }

    let dims = matrix.shape().dims();
    let rows = dims[0];
    let cols = dims[1];

    let matrix_data = matrix.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access matrix data".to_string())
    })?;

    let mut diag_data = Vec::new();

    if offset >= 0 {
        let offset = offset as usize;
        for i in 0..rows {
            let j = i + offset;
            if j < cols {
                diag_data.push(matrix_data[i * cols + j].clone());
            }
        }
    } else {
        let offset = (-offset) as usize;
        for i in offset..rows {
            let j = i - offset;
            if j < cols {
                diag_data.push(matrix_data[i * cols + j].clone());
            }
        }
    }

    let len = diag_data.len();
    Tensor::from_vec(diag_data, &[len])
}

/// Create a triangular matrix (upper or lower)
pub fn tri<T>(n: usize, m: Option<usize>, k: i32, lower: bool) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    let m = m.unwrap_or(n);
    let mut data = vec![T::zero(); n * m];

    for i in 0..n {
        for j in 0..m {
            let condition = if lower {
                j as i32 <= i as i32 + k
            } else {
                j as i32 >= i as i32 + k
            };

            if condition {
                data[i * m + j] = T::one();
            }
        }
    }

    Tensor::from_vec(data, &[n, m])
}

/// Create an upper triangular matrix
pub fn triu<T>(n: usize, m: Option<usize>, k: i32) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    tri(n, m, k, false)
}

/// Create a lower triangular matrix
pub fn tril<T>(n: usize, m: Option<usize>, k: i32) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    tri(n, m, k, true)
}

/// Create meshgrid like NumPy
pub fn meshgrid<T>(x: &Tensor<T>, y: &Tensor<T>, indexing: &str) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if x.shape().rank() != 1 || y.shape().rank() != 1 {
        return Err(TensorError::invalid_argument(
            "meshgrid requires 1D input tensors".to_string(),
        ));
    }

    let x_data = x.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access x data".to_string())
    })?;
    let y_data = y.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("Cannot access y data".to_string())
    })?;

    let (x_len, y_len) = (x_data.len(), y_data.len());

    let (xx_data, yy_data, shape) = match indexing {
        "xy" => {
            // Cartesian indexing: X has shape (len(y), len(x)), Y has shape (len(y), len(x))
            let mut xx = Vec::with_capacity(y_len * x_len);
            let mut yy = Vec::with_capacity(y_len * x_len);

            for y_val in y_data {
                for x_val in x_data {
                    xx.push(x_val.clone());
                    yy.push(y_val.clone());
                }
            }

            (xx, yy, vec![y_len, x_len])
        }
        "ij" => {
            // Matrix indexing: X has shape (len(x), len(y)), Y has shape (len(x), len(y))
            let mut xx = Vec::with_capacity(x_len * y_len);
            let mut yy = Vec::with_capacity(x_len * y_len);

            for x_val in x_data {
                for y_val in y_data {
                    xx.push(x_val.clone());
                    yy.push(y_val.clone());
                }
            }

            (xx, yy, vec![x_len, y_len])
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown indexing: '{indexing}'. Use 'xy' or 'ij'"
            )));
        }
    };

    let xx_tensor = Tensor::from_vec(xx_data, &shape)?;
    let yy_tensor = Tensor::from_vec(yy_data, &shape)?;

    Ok((xx_tensor, yy_tensor))
}

/// Create array from function
pub fn fromfunction<T, F>(func: F, shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    F: Fn(&[usize]) -> T + Send + Sync,
{
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    // Generate all possible indices
    let mut indices = vec![0; shape.len()];

    for _ in 0..total_size {
        data.push(func(&indices));

        // Increment indices (like odometer)
        let mut carry = true;
        for i in (0..shape.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] < shape[i] {
                    carry = false;
                } else {
                    indices[i] = 0;
                }
            }
        }
    }

    Tensor::from_vec(data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange() {
        let result = arange(0.0f32, 5.0, 1.0).unwrap();
        if let Some(data) = result.as_slice() {
            assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_linspace() {
        let result = linspace(0.0f32, 1.0, 6, true).unwrap();
        if let Some(data) = result.as_slice() {
            assert_eq!(data.len(), 6);
            assert!((data[0] - 0.0).abs() < 1e-6);
            assert!((data[5] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_eye() {
        let result = eye::<f32>(3, None, 0).unwrap();
        assert_eq!(result.shape().dims(), &[3, 3]);

        if let Some(data) = result.as_slice() {
            let expected = &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            assert_eq!(data, expected);
        }
    }

    #[test]
    fn test_diag() {
        let v = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = diag(&v, 0).unwrap();
        assert_eq!(result.shape().dims(), &[3, 3]);

        if let Some(data) = result.as_slice() {
            let expected = &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
            assert_eq!(data, expected);
        }
    }

    #[test]
    fn test_diagonal() {
        let matrix =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .unwrap();

        let result = diagonal(&matrix, 0).unwrap();
        if let Some(data) = result.as_slice() {
            assert_eq!(data, &[1.0, 5.0, 9.0]);
        }
    }

    #[test]
    fn test_meshgrid() {
        let x = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        let (xx, yy) = meshgrid(&x, &y, "xy").unwrap();
        assert_eq!(xx.shape().dims(), &[3, 2]);
        assert_eq!(yy.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_fromfunction() {
        let result = fromfunction(|indices| (indices[0] + indices[1]) as f32, &[2, 3]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);

        if let Some(data) = result.as_slice() {
            assert_eq!(data, &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_triu_tril() {
        let upper = triu::<f32>(3, None, 0).unwrap();
        if let Some(data) = upper.as_slice() {
            let expected = &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
            assert_eq!(data, expected);
        }

        let lower = tril::<f32>(3, None, 0).unwrap();
        if let Some(data) = lower.as_slice() {
            let expected = &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
            assert_eq!(data, expected);
        }
    }
}
