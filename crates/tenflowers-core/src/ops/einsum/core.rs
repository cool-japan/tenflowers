//! Core Einstein Summation Implementation
//!
//! This module contains the main einsum function and parsing logic for
//! Einstein summation notation operations.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{One, Zero};
use std::collections::HashMap;

#[cfg(any(
    feature = "blas-openblas",
    feature = "blas-oxiblas",
    feature = "blas-mkl"
))]
use super::blas::try_blas_optimized_patterns;

use super::cache::execute_contraction_path;
use super::patterns::try_optimize_common_patterns;
use super::utils::compute_optimal_path;

/// Einstein summation convention implementation with optimizations
///
/// This function implements einsum notation for tensor operations with performance optimizations.
/// Examples:
/// - "ij,jk->ik" (matrix multiplication)
/// - "ij->ji" (transpose)
/// - "ii->i" (diagonal)
/// - "ij,ij->ij" (element-wise multiplication)
/// - "ij,ij->" (sum of element-wise multiplication)
/// - "ijk,ijk->ik" (batched operations)
/// - "bij,bjk->bik" (batched matrix multiplication)
pub fn einsum<T>(equation: &str, operands: &[&Tensor<T>]) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if operands.is_empty() {
        return Err(TensorError::invalid_argument(
            "At least one operand is required for einsum".to_string(),
        ));
    }

    // Parse the einsum equation
    let (input_subscripts, output_subscript) = parse_einsum_equation(equation)?;

    if input_subscripts.len() != operands.len() {
        return Err(TensorError::invalid_argument(format!(
            "Number of operands ({}) does not match equation ({})",
            operands.len(),
            input_subscripts.len()
        )));
    }

    // Validate device consistency and storage type
    let first_device = operands[0].device();
    for operand in operands {
        if operand.device() != first_device {
            return Err(TensorError::device_mismatch(
                "einsum",
                &first_device.to_string(),
                &operand.device().to_string(),
            ));
        }

        match &operand.storage {
            TensorStorage::Cpu(_) => {}
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // GPU einsum is now supported for common patterns
            }
        }
    }

    // Try BLAS optimizations first for CPU tensors
    #[cfg(any(
        all(feature = "blas-openblas", feature = "std"),
        all(feature = "blas-mkl", feature = "std"),
        all(feature = "blas-accelerate", feature = "std")
    ))]
    {
        let all_cpu = operands.iter().all(|op| match &op.storage {
            TensorStorage::Cpu(_) => true,
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => false,
        });

        if all_cpu {
            if let Some(blas_result) = try_blas_optimized_patterns(equation, operands) {
                return blas_result;
            }
        }
    }

    // Optimize for common patterns
    if let Some(optimized_result) = try_optimize_common_patterns(equation, operands) {
        return optimized_result;
    }

    // Handle cases by number of operands
    match operands.len() {
        1 => einsum_unary(&input_subscripts[0], &output_subscript, operands[0]),
        2 => einsum_binary(
            &input_subscripts[0],
            &input_subscripts[1],
            &output_subscript,
            operands[0],
            operands[1],
        ),
        _ => {
            // General case: use optimal contraction path
            let contraction_path = compute_optimal_path(&input_subscripts, &output_subscript)?;
            execute_contraction_path(operands, &contraction_path)
        }
    }
}

/// Parse einsum equation like "ij,jk->ik"
pub fn parse_einsum_equation(equation: &str) -> Result<(Vec<String>, String)> {
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() > 2 {
        return Err(TensorError::invalid_argument(
            "Invalid einsum equation: too many '->' separators".to_string(),
        ));
    }

    let input_part = parts[0];
    let output_part = if parts.len() == 2 { parts[1] } else { "" };

    let input_subscripts: Vec<String> = input_part
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    if input_subscripts.is_empty() {
        return Err(TensorError::invalid_argument(
            "No input subscripts found in einsum equation".to_string(),
        ));
    }

    let output_subscript = if parts.len() == 2 {
        // Explicit output subscript
        output_part.trim().to_string()
    } else {
        // Implicit output: sum over repeated indices, keep non-repeated ones
        infer_output_subscript(&input_subscripts)?
    };

    Ok((input_subscripts, output_subscript))
}

/// Infer output subscript when not explicitly provided
pub fn infer_output_subscript(input_subscripts: &[String]) -> Result<String> {
    let mut char_counts: HashMap<char, usize> = HashMap::new();

    for subscript in input_subscripts {
        for c in subscript.chars() {
            if c.is_alphabetic() {
                *char_counts.entry(c).or_insert(0) += 1;
            }
        }
    }

    // Keep only characters that appear once (not summed over)
    let mut output_chars: Vec<char> = char_counts
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&c, _)| c)
        .collect();

    output_chars.sort();
    Ok(output_chars.into_iter().collect())
}

/// Handle unary einsum operations (like transpose, diagonal, sum)
pub(super) fn einsum_unary<T>(
    input_subscript: &str,
    output_subscript: &str,
    operand: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_chars: Vec<char> = input_subscript.chars().collect();
    let output_chars: Vec<char> = output_subscript.chars().collect();

    if input_chars.len() != operand.shape().rank() {
        return Err(TensorError::invalid_argument(format!(
            "Input subscript length ({}) does not match tensor rank ({})",
            input_chars.len(),
            operand.shape().rank()
        )));
    }

    // Simple transpose case
    if input_chars.len() == output_chars.len()
        && input_chars.iter().all(|c| output_chars.contains(c))
    {
        let mut permutation = Vec::new();
        for &output_char in &output_chars {
            if let Some(pos) = input_chars.iter().position(|&c| c == output_char) {
                permutation.push(pos);
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Output character '{output_char}' not found in input"
                )));
            }
        }
        return crate::ops::manipulation::transpose_axes(operand, Some(&permutation));
    }

    // Sum reduction case
    if output_chars.is_empty() {
        return crate::ops::sum(operand, None, false);
    }

    // Partial reduction case
    let mut axes_to_reduce = Vec::new();
    for (i, &input_char) in input_chars.iter().enumerate() {
        if !output_chars.contains(&input_char) {
            axes_to_reduce.push(i as i32);
        }
    }

    if !axes_to_reduce.is_empty() {
        return crate::ops::sum(operand, Some(&axes_to_reduce), false);
    }

    // Diagonal extraction case (e.g., "ii->i")
    if input_chars.len() == 2
        && output_chars.len() == 1
        && input_chars[0] == input_chars[1]
        && input_chars[0] == output_chars[0]
    {
        return extract_diagonal(operand);
    }

    Err(TensorError::invalid_argument(format!(
        "Unsupported unary einsum: {input_subscript} -> {output_subscript}"
    )))
}

/// Handle binary einsum operations (like matrix multiplication, element-wise ops)
pub(super) fn einsum_binary<T>(
    left_subscript: &str,
    right_subscript: &str,
    output_subscript: &str,
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let left_chars: Vec<char> = left_subscript.chars().collect();
    let right_chars: Vec<char> = right_subscript.chars().collect();
    let output_chars: Vec<char> = output_subscript.chars().collect();

    if left_chars.len() != left.shape().rank() {
        return Err(TensorError::invalid_argument(format!(
            "Left subscript length ({}) does not match tensor rank ({})",
            left_chars.len(),
            left.shape().rank()
        )));
    }

    if right_chars.len() != right.shape().rank() {
        return Err(TensorError::invalid_argument(format!(
            "Right subscript length ({}) does not match tensor rank ({})",
            right_chars.len(),
            right.shape().rank()
        )));
    }

    // Matrix multiplication case: "ij,jk->ik"
    if left_chars.len() == 2
        && right_chars.len() == 2
        && output_chars.len() == 2
        && left_chars[1] == right_chars[0]
        && left_chars[0] == output_chars[0]
        && right_chars[1] == output_chars[1]
    {
        return crate::ops::matmul(left, right);
    }

    // Element-wise multiplication: "ij,ij->ij"
    if left_subscript == right_subscript && left_subscript == output_subscript {
        return left.mul(right);
    }

    // Sum of element-wise multiplication: "ij,ij->"
    if left_subscript == right_subscript && output_subscript.is_empty() {
        let elementwise = left.mul(right)?;
        return crate::ops::sum(&elementwise, None, false);
    }

    Err(TensorError::invalid_argument(format!(
        "Unsupported binary einsum: {left_subscript},{right_subscript} -> {output_subscript}"
    )))
}

/// Extract diagonal from a 2D tensor
pub fn extract_diagonal<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let shape = tensor.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_argument(
            "Diagonal extraction requires 2D tensor".to_string(),
        ));
    }

    let min_dim = shape[0].min(shape[1]);
    let mut diagonal_data = Vec::with_capacity(min_dim);

    for i in 0..min_dim {
        if let Some(val) = tensor.get(&[i, i]) {
            diagonal_data.push(val);
        } else {
            return Err(TensorError::invalid_argument(
                "Failed to extract diagonal element".to_string(),
            ));
        }
    }

    Tensor::from_vec(diagonal_data, &[min_dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests exercise `einsum` directly on CPU-resident tensors, with no
    // dependency on the `gpu` feature. They exist because the GPU einsum
    // wrappers in `gpu.rs` (`gpu_einsum_batched_matmul`, `gpu_einsum_transpose`,
    // `gpu_einsum_diagonal`, `gpu_einsum_outer_product`, `gpu_einsum_trace`) now
    // read GPU operands back to the host and delegate to this exact `einsum`
    // entry point instead of running (broken) GPU kernels directly. These tests
    // pin down the correctness of that delegate target, independent of GPU
    // availability.

    /// "bij,bjk->bik": batched matrix multiplication.
    #[test]
    fn einsum_batched_matmul_is_correct() {
        // batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]]   = [[19,22],[43,50]]
        // batch 1: [[1,0],[0,1]] @ [[9,10],[11,12]] = [[9,10],[11,12]]
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2])
            .expect("test: from_vec should succeed");
        let b =
            Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 2, 2])
                .expect("test: from_vec should succeed");

        let result = einsum("bij,bjk->bik", &[&a, &b]).expect("test: einsum should succeed");
        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0, 9.0, 10.0, 11.0, 12.0]);
    }

    /// "ij->ji": plain 2D transpose.
    #[test]
    fn einsum_transpose_is_correct() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("test: from_vec should succeed");

        let result = einsum("ij->ji", &[&a]).expect("test: einsum should succeed");
        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// "ii->i": diagonal extraction.
    #[test]
    fn einsum_diagonal_is_correct() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
            .expect("test: from_vec should succeed");

        let result = einsum("ii->i", &[&a]).expect("test: einsum should succeed");
        assert_eq!(result.shape().dims(), &[3]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![1.0, 5.0, 9.0]);
    }

    /// "i,j->ij": outer product.
    #[test]
    fn einsum_outer_product_is_correct() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let b =
            Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).expect("test: from_vec should succeed");

        let result = einsum("i,j->ij", &[&a, &b]).expect("test: einsum should succeed");
        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    /// "ii->": trace (sum of diagonal elements only, not the whole matrix).
    #[test]
    fn einsum_trace_is_correct() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
            .expect("test: from_vec should succeed");

        let result = einsum("ii->", &[&a]).expect("test: einsum should succeed");
        let data = result.to_vec().expect("test: to_vec should succeed");
        // Trace = 1 + 5 + 9 = 15; must NOT equal the sum of all 9 entries (45),
        // which is the bug the GPU kernel used to have.
        assert_eq!(data, vec![15.0]);
    }
}
